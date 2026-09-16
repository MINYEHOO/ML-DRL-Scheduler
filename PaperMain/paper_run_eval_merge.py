"""Validate and merge completed shards of one frozen live-run ID evaluation.

Confidence intervals resample complete episodes, pairing PPO with each baseline.
They describe evaluation-episode variation for one fixed trained checkpoint.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math

import numpy as np


ROOT = Path(__file__).resolve().parent
MEAN_METRICS = (
    'goodput_mbps', 'reward', 'throughput_mbps', 'completion_rate',
    'deadline_miss_rate', 'buffer_overflow_rate', 'retx_drop_rate', 'mu_depth', 'jain',
)
REQUIRED_PROTOCOL = (
    'name', 'run', 'checkpoint_kind', 'checkpoint_sha256', 'checkpoint_update',
    'training_config', 'environment_config', 'policy_config', 'source_sha256',
    'runner_sha256', 'calibration_profile', 'calibration_reference_root',
    'episode_ids', 'scheduler_keys', 'scheduler_names', 'diagnostic_only',
    'history_audit_sha256',
)
TEXT_COLUMNS = {'scheduler', 'scheduler_name', 'world'}
COUNT_COLUMNS = (
    'n_arrivals', 'n_offered', 'n_buffer_overflow', 'n_units_new',
    'n_units_first_ack', 'n_comp', 'n_miss_deadline', 'n_retx_drop',
    'n_retx_overflow_drop', 'terminal_queued_packets',
) + tuple(f'{kind}_m{depth}' for depth in range(1, 5) for kind in ('acks', 'units'))
CAVEATS = [
    'Evaluation episodes are the resampling unit; slots and packets are not independent replicates.',
    'Paired differences match episode IDs and underlying episode conditions. Scheduler-dependent '
    'shared RNG consumption can change realized traffic/CSI, so identical packet traces are not claimed.',
    'Intervals are pointwise 95% percentile bootstrap intervals, not simultaneous confidence bounds '
    'across schedulers or metrics.',
    'One frozen checkpoint from one training seed is evaluated. Intervals do not quantify training-seed '
    'variation, and a positive interval is not a universal performance guarantee.',
    'Completion and deadline-miss rates use admitted arrivals; buffer-overflow rate uses offered '
    'arrivals. Finite episodes can retain unfinished packets at the horizon.',
    'First-ACK rates pool raw ACK counts and new-unit counts. Empty depth bins are undefined, not zero.',
]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _ordered_unique(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} must be a nonempty list')
    if len(values) != len(set(values)):
        raise ValueError(f'{label} contains duplicates')
    return values


def _protocol_validate(protocol):
    if not isinstance(protocol, dict):
        raise ValueError('Missing protocol dictionary')
    missing = set(REQUIRED_PROTOCOL) - set(protocol)
    if missing:
        raise ValueError(f'Missing protocol provenance: {sorted(missing)}')
    if protocol['name'] != 'paper-live-run-id-v1':
        raise ValueError('Unsupported evaluation protocol')
    canonical(protocol)  # Reject nonfinite JSON metadata, too.
    episodes = _ordered_unique(protocol['episode_ids'], 'protocol episode_ids')
    for ep in episodes:
        _integer(ep, 'episode ID')
    if episodes != sorted(episodes):
        raise ValueError('Protocol episode_ids must be sorted')
    schedulers = _ordered_unique(protocol['scheduler_keys'], 'scheduler_keys')
    if not all(isinstance(k, str) and k for k in schedulers) or 'ppo' not in schedulers:
        raise ValueError('Scheduler keys must be nonempty strings and include ppo')
    names = protocol['scheduler_names']
    if isinstance(names, list):
        if len(names) != len(schedulers):
            raise ValueError('scheduler_names length differs from scheduler_keys')
        names = dict(zip(schedulers, names))
    if not isinstance(names, dict) or set(names) != set(schedulers):
        raise ValueError('scheduler_names must cover scheduler_keys exactly')
    if not all(isinstance(n, str) and n for n in names.values()):
        raise ValueError('Invalid scheduler display name')
    if not isinstance(protocol['diagnostic_only'], bool):
        raise ValueError('diagnostic_only must be boolean')
    return episodes, schedulers, names


def _numeric_row(raw, header, label):
    if None in raw or any(raw.get(k) is None for k in header):
        raise ValueError(f'{label}: malformed CSV row')
    row = dict(raw)
    for key in header:
        if key in TEXT_COLUMNS:
            continue
        if key == 'mean_sinr_db' and raw[key] == '' and raw.get('sinr_count') == '0':
            row[key] = None
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label}: nonnumeric {key}') from exc
        if not math.isfinite(value):
            raise ValueError(f'{label}: nonfinite {key}')
        row[key] = value
    for key in ('episode_idx',) + COUNT_COLUMNS:
        if row[key] < 0 or not row[key].is_integer():
            raise ValueError(f'{label}: invalid count {key}')
        row[key] = int(row[key])
    for depth in range(1, 5):
        if row[f'acks_m{depth}'] > row[f'units_m{depth}']:
            raise ValueError(f'{label}: ACK count exceeds units at depth {depth}')
    if sum(row[f'acks_m{d}'] for d in range(1, 5)) != row['n_units_first_ack']:
        raise ValueError(f'{label}: first-ACK depth counts do not sum to total')
    if sum(row[f'units_m{d}'] for d in range(1, 5)) != row['n_units_new']:
        raise ValueError(f'{label}: unit depth counts do not sum to total')
    if row['n_arrivals'] + row['n_buffer_overflow'] != row['n_offered']:
        raise ValueError(f'{label}: offered-arrival counts inconsistent')
    if sum(row[key] for key in ('n_comp', 'n_miss_deadline', 'n_retx_drop',
                                'n_retx_overflow_drop', 'terminal_queued_packets')) != row['n_arrivals']:
        raise ValueError(f'{label}: admitted-packet accounting inconsistent')
    for key in ('completion_rate', 'deadline_miss_rate', 'buffer_overflow_rate',
                'retx_drop_rate', 'jain'):
        if not 0 <= row[key] <= 1:
            raise ValueError(f'{label}: {key} outside [0,1]')
    return row


def _validate_row_metadata(row, protocol, label):
    if 'world' in row and row['world'] != 'ID':
        raise ValueError(f'{label}: non-ID row in ID evaluation')
    expected = {'checkpoint_update': protocol['checkpoint_update'],
                'diagnostic_only': int(protocol['diagnostic_only']),
                'train_seed': protocol['training_config']['seed']}
    env = protocol['environment_config']
    if 'debug' in env:
        expected['slots'] = env['episode_len_debug' if env['debug'] else 'episode_len_main']
    for key, value in expected.items():
        if key in row and row[key] != value:
            raise ValueError(f'{label}: {key} differs from protocol')


def load_shards(shard_dirs, allow_smoke=False):
    """Read complete inputs without creating outputs; reject all incomplete pair grids."""
    paths = [Path(p).resolve() for p in shard_dirs]
    if len(paths) < 1 or len(set(paths)) != len(paths):
        raise ValueError('Supply distinct shard directories')
    protocol, reference_header = None, None
    reference_runtime, reference_execution, reference_model = None, None, None
    seen_indices, all_rows, inputs = set(), {}, []
    shard_count = None
    for path in paths:
        manifest_bytes = (path / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get('status') != 'completed':
            raise ValueError(f'{path}: shard is not completed')
        before = manifest.get('model_state_sha256_before')
        after = manifest.get('model_state_sha256_after')
        if (manifest.get('model_state_unchanged') is not True or not isinstance(before, str)
                or len(before) != 64 or before != after):
            raise ValueError(f'{path}: model-state preservation proof missing or inconsistent')
        if manifest.get('torch_rng_unchanged') is not True:
            raise ValueError(f'{path}: Torch RNG preservation proof missing')
        runtime = manifest.get('runtime')
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError(f'{path}: runtime record missing')
        if reference_runtime is None:
            reference_runtime, reference_model = canonical(runtime), before
        elif canonical(runtime) != reference_runtime or before != reference_model:
            raise ValueError(f'{path}: runtime or loaded-model mismatch across shards')
        current = manifest.get('protocol')
        episodes, schedulers, names = _protocol_validate(current)
        if current['diagnostic_only'] and not allow_smoke:
            raise ValueError('Diagnostic shards require explicit --allow-smoke')
        if protocol is None:
            protocol = current
        elif canonical(current) != canonical(protocol):
            raise ValueError(f'{path}: protocol/provenance mismatch')
        execution = manifest.get('execution', {})
        common_execution = {k: v for k, v in execution.items()
                            if k not in ('shard_index', 'gpu', 'episode_ids')}
        if reference_execution is None:
            reference_execution = canonical(common_execution)
        elif canonical(common_execution) != reference_execution:
            raise ValueError(f'{path}: execution settings mismatch across shards')
        index = _integer(execution.get('shard_index'), 'shard_index')
        count = _integer(execution.get('shard_count'), 'shard_count', 1)
        if index >= count or index in seen_indices:
            raise ValueError(f'{path}: duplicate or invalid shard index')
        if shard_count is not None and count != shard_count:
            raise ValueError('Inconsistent shard counts')
        shard_count = count
        seen_indices.add(index)
        planned = episodes[index::count]
        if execution.get('episode_ids') != planned:
            raise ValueError(f'{path}: incorrect modulo episode split')
        metrics_bytes = (path / 'metrics.csv').read_bytes()
        digest = sha_bytes(metrics_bytes)
        if manifest.get('metrics_sha256') != digest:
            raise ValueError(f'{path}: metrics hash mismatch or missing completed metrics hash')
        raw_metrics_digest = None
        if 'raw_metrics_sha256' in manifest:
            raw_metrics_digest = sha_bytes((path / 'raw_metrics.jsonl').read_bytes())
            if manifest['raw_metrics_sha256'] != raw_metrics_digest:
                raise ValueError(f'{path}: raw metrics hash mismatch')
        reader = csv.DictReader(io.StringIO(metrics_bytes.decode('utf-8'), newline=''))
        header = reader.fieldnames
        required = {'episode_idx', 'scheduler', 'scheduler_name', *MEAN_METRICS, *COUNT_COLUMNS}
        if not header or len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError(f'{path}: missing or duplicate CSV columns')
        if 'columns' in manifest and manifest['columns'] != header:
            raise ValueError(f'{path}: recorded CSV schema mismatch')
        if reference_header is None:
            reference_header = header
        elif header != reference_header:
            raise ValueError('Different CSV schemas across shards')
        expected = {(ep, key) for ep in planned for key in schedulers}
        actual = set()
        for line, raw in enumerate(reader, 2):
            row = _numeric_row(raw, header, f'{path}/metrics.csv:{line}')
            _validate_row_metadata(row, protocol, f'{path}/metrics.csv:{line}')
            pair = (row['episode_idx'], row['scheduler'])
            if pair not in expected:
                raise ValueError(f'{path}: unplanned episode/scheduler pair {pair}')
            if pair in actual or pair in all_rows:
                raise ValueError(f'{path}: duplicate episode/scheduler pair {pair}')
            if row['scheduler_name'] != names[row['scheduler']]:
                raise ValueError(f'{path}: scheduler display name mismatch')
            actual.add(pair)
            all_rows[pair] = row
        if actual != expected:
            raise ValueError(f'{path}: missing {len(expected - actual)} planned rows')
        if 'rows_written' in manifest and manifest['rows_written'] != len(actual):
            raise ValueError(f'{path}: rows_written mismatch')
        inputs.append(dict(path=str(path), manifest_sha256=sha_bytes(manifest_bytes),
                           metrics_sha256=digest, raw_metrics_sha256=raw_metrics_digest,
                           rows=len(actual), execution=execution))
    if seen_indices != set(range(shard_count)):
        raise ValueError('Missing shards: require the entire declared shard set')
    episodes, schedulers, _ = _protocol_validate(protocol)
    rows = [all_rows[(ep, key)] for ep in episodes for key in schedulers]
    return protocol, rows, reference_header, sorted(inputs, key=lambda x: x['execution']['shard_index'])


def build_summary(protocol, rows, inputs, bootstrap_replicates=10000, bootstrap_seed=20260911):
    _integer(bootstrap_replicates, 'bootstrap_replicates', 100)
    _integer(bootstrap_seed, 'bootstrap_seed')
    episodes, schedulers, names = _protocol_validate(protocol)
    pairs = {(row['episode_idx'], row['scheduler']): row for row in rows}
    if len(pairs) != len(rows) or set(pairs) != {(ep, key) for ep in episodes for key in schedulers}:
        raise ValueError('Summary requires one row for every episode/scheduler pair')
    indices = np.random.default_rng(bootstrap_seed).integers(
        0, len(episodes), size=(bootstrap_replicates, len(episodes)))

    def estimate(values, difference=False):
        values = np.asarray(values, dtype=np.float64)
        ci = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        return {('mean_difference' if difference else 'mean'): float(values.mean()),
                'ci95': [float(x) for x in ci]}

    by_scheduler, arrays = {}, {}
    for key in schedulers:
        selected = [pairs[(ep, key)] for ep in episodes]
        arrays[key] = {metric: np.array([r[metric] for r in selected], dtype=np.float64)
                       for metric in MEAN_METRICS}
        depths = {}
        for depth in range(1, 5):
            acks = sum(r[f'acks_m{depth}'] for r in selected)
            units = sum(r[f'units_m{depth}'] for r in selected)
            depths[str(depth)] = dict(acks=acks, units=units, rate=acks / units if units else None)
        acks = sum(r['n_units_first_ack'] for r in selected)
        units = sum(r['n_units_new'] for r in selected)
        by_scheduler[key] = dict(name=names[key], episodes=len(episodes),
            metrics={metric: estimate(values) for metric, values in arrays[key].items()},
            raw_count_totals={name: sum(r[name] for r in selected) for name in COUNT_COLUMNS},
            first_ack_pooled=dict(acks=acks, units=units, rate=acks / units if units else None),
            first_ack_by_depth=depths)
    paired = {
        key: {metric: estimate(arrays['ppo'][metric] - arrays[key][metric], difference=True)
              for metric in MEAN_METRICS}
        for key in schedulers if key != 'ppo'
    }
    return dict(status='completed', protocol=protocol, input_shards=inputs,
                episodes=len(episodes), rows=len(rows),
                bootstrap=dict(method='paired-episode percentile bootstrap', confidence_level=0.95,
                               replicates=bootstrap_replicates, seed=bootstrap_seed,
                               statistical_unit='episode', interval_scope='pointwise'),
                by_scheduler=by_scheduler, paired_ppo_minus_baseline=paired, caveats=CAVEATS)


def render_markdown(summary):
    protocol = summary['protocol']
    _, schedulers, names = _protocol_validate(protocol)
    lines = [
        '# Frozen-checkpoint FTP3 ID evaluation', '',
        f"Run: `{protocol['run']}`; checkpoint: `{protocol['checkpoint_kind']}` at update "
        f"{protocol['checkpoint_update']}; SHA-256: `{protocol['checkpoint_sha256']}`.", '',
        f"{summary['episodes']} episodes; {len(schedulers)} schedulers; {summary['rows']} validated rows. "
        f"Episode IDs: {protocol['episode_ids'][0]}–{protocol['episode_ids'][-1]}. "
        f"Diagnostic only: {protocol['diagnostic_only']}.", '',
        'Values are episode means. Goodput includes its pointwise 95% bootstrap interval; '
        'all metric intervals are retained in `summary.json`.', '',
        '| Scheduler | Goodput Mbps [95% CI] | Reward | Completion % | Deadline miss % | Overflow % | MU depth | Jain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    if protocol['diagnostic_only']:
        lines[2:2] = ['**DIAGNOSTIC SMOKE RUN: these results are not paper evaluation evidence.**', '']

    def interval(value, difference=False):
        center = value['mean_difference' if difference else 'mean']
        return f"{center:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"

    for key in schedulers:
        metrics = summary['by_scheduler'][key]['metrics']
        lines.append(f"| {names[key]} | {interval(metrics['goodput_mbps'])} | "
                     f"{metrics['reward']['mean']:.2f} | {100 * metrics['completion_rate']['mean']:.3f} | "
                     f"{100 * metrics['deadline_miss_rate']['mean']:.3f} | "
                     f"{100 * metrics['buffer_overflow_rate']['mean']:.3f} | "
                     f"{metrics['mu_depth']['mean']:.3f} | {metrics['jain']['mean']:.4f} |")
    lines += ['', 'Paired PPO minus baseline differences, with pointwise 95% episode-bootstrap intervals. '
              'Positive goodput/reward differences favor PPO; negative miss/overflow differences favor PPO. '
              'Rate differences in the following table are fractions, not percentage points.', '',
              '| Baseline | Goodput Mbps difference | Reward difference | Deadline miss difference | Overflow difference |',
              '|---|---:|---:|---:|---:|']
    for key, metrics in summary['paired_ppo_minus_baseline'].items():
        cells = [interval(metrics[metric], True) for metric in
                 ('goodput_mbps', 'reward', 'deadline_miss_rate', 'buffer_overflow_rate')]
        lines.append('| ' + names[key] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Raw episode counter totals (packets). Offered includes arrivals rejected by a full queue; '
              'admitted excludes those arrivals. Terminal queued packets have not been drained.', '',
              '| Scheduler | Offered | Admitted | Completed | Deadline drops | Retx-limit drops | Retx-overflow drops | Buffer-overflow drops | Terminal queued |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    count_keys = ('n_offered', 'n_arrivals', 'n_comp', 'n_miss_deadline', 'n_retx_drop',
                  'n_retx_overflow_drop', 'n_buffer_overflow', 'terminal_queued_packets')
    for key in schedulers:
        counts = summary['by_scheduler'][key]['raw_count_totals']
        lines.append('| ' + names[key] + ' | ' + ' | '.join(str(counts[k]) for k in count_keys) + ' |')
    lines += ['', 'First-ACK rates below pool raw counts across episodes: `sum(ACKs) / sum(new units)`. '
              'Each cell gives the rate and its count denominator; an empty bin is undefined.', '',
              '| Scheduler | Depth 1 | Depth 2 | Depth 3 | Depth 4 |',
              '|---|---:|---:|---:|---:|']
    for key in schedulers:
        cells = []
        for depth in range(1, 5):
            item = summary['by_scheduler'][key]['first_ack_by_depth'][str(depth)]
            cells.append('undefined (0/0)' if item['rate'] is None else
                         f"{100 * item['rate']:.3f}% ({item['acks']}/{item['units']})")
        lines.append('| ' + names[key] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Interpretation limits:', ''] + ['- ' + text for text in summary['caveats']]
    return '\n'.join(lines) + '\n'


def merge_shards(shard_dirs, out, bootstrap_replicates=10000, bootstrap_seed=20260911, allow_smoke=False):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    protocol, rows, header, inputs = load_shards(shard_dirs, allow_smoke=allow_smoke)
    summary = build_summary(protocol, rows, inputs, bootstrap_replicates, bootstrap_seed)
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    markdown = render_markdown(summary)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'mergedmetrics.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'summary.json').open('x') as handle:
        handle.write(summary_text)
    with (out / 'summary.md').open('x') as handle:
        handle.write(markdown)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', nargs='+', required=True)
    parser.add_argument('--out', required=True, help='New merged-output directory below package results/')
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260911)
    parser.add_argument('--allow-smoke', action='store_true', help='Merge explicitly labelled diagnostic runs')
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if not out.resolve().is_relative_to((ROOT / 'results').resolve()):
        parser.error('--out must be inside package results/')
    shards = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.shards]
    try:
        summary = merge_shards(shards, out, args.bootstrap_replicates, args.bootstrap_seed, args.allow_smoke)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Merged {summary['rows']} rows from {summary['episodes']} episodes into {out}")


if __name__ == '__main__':
    main()
