"""Validate the three frozen-policy OOD workers and report paired episode statistics.

Each worker evaluates one policy and two distinct historical baselines. Baseline
rows are shared across policy comparisons only after common-world provenance is
verified. All confidence intervals use episode-level Student-t statistics.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import t as student_t

import paper_ood_eval as ev
import paper_run_eval_merge as numeric

ROOT = Path(__file__).resolve().parent
RUN_IDS = (16, 21, 22)
MEAN_METRICS = numeric.MEAN_METRICS
BASELINES = tuple(k for k in ev.SCHEDULERS if k != 'ppo')
EXTRA_OUTPUT = ['source_run', 'source_checkpoint_sha256']
COMMON_KEYS = ('name', 'mode', 'base_run', 'base_config', 'base_input_hashes',
               'current_source_sha256', 'runner_sha256', 'worlds', 'episode_ids',
               'thresholds', 'threshold_input_hashes', 'history_audit_sha256',
               'diagnostic_only', 'smoke_slots', 'candidate_thresholds')
CAVEATS = [
    'Intervals use independent evaluation episodes as the statistical unit, not slots or packets.',
    'Pointwise 95% Student-t intervals are not simultaneous bounds across worlds, schedulers or metrics.',
    'All three fixed checkpoints share one training seed; these intervals do not measure training-seed variation. Run16 continues earlier training and is not an independent training replicate.',
    'Within each world, differences pair the same episode IDs and initial conditions. Shared RNG consumption can change realized traffic/CSI after scheduler-dependent queue admissions; identical packet traces are not claimed.',
    'K scaling changes both user population and total offered load. Different K worlds do not share identical topology and are not paired against one another.',
    'D26strict changes environment deadlines to 2..6 while policy deadline normalization remains /12. Variable K uses the unchanged policy formulas with their native current-K count fractions.',
    'SUS thresholds were selected by mean reward on separate pilot episodes. Only the six declared historical baselines are compared; this does not establish superiority over every heuristic.',
    'Completion, deadline-miss and retransmission-drop rates use admitted arrivals; buffer-overflow rates use offered arrivals. Unfinished packets remain at the finite episode horizon.',
    'First-ACK rates pool raw counts across episodes. Empty depth bins are undefined and stored as null, not zero.',
]


def same(left, right, message):
    if numeric.canonical(left) != numeric.canonical(right):
        raise ValueError(message)


def sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label}: invalid SHA-256')
    return value


def hashes(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{label}: missing hash inventory')
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f'{label}: invalid hash key')
        sha(digest, label)


def run_number(run):
    if not isinstance(run, str) or re.fullmatch(r'runs/(16|21|22)_[^/]+', run) is None:
        raise ValueError('Expected a frozen source run16, run21 or run22 below runs/')
    return int(Path(run).name.split('_', 1)[0])


def _protocol(protocol, allow_smoke):
    if not isinstance(protocol, dict) or protocol.get('name') != 'paper-common-bernoulli-ood-v1' or protocol.get('mode') != 'eval':
        raise ValueError('Expected a common-world OOD evaluation protocol')
    numeric.canonical(protocol)
    for key in COMMON_KEYS:
        if key not in protocol:
            raise ValueError(f'Missing protocol provenance: {key}')
    run_id = run_number(protocol.get('run'))
    if run_number(protocol['base_run']) != 21:
        raise ValueError('Common environment must use the selected run21 Base source')
    diagnostic = protocol['diagnostic_only']
    if type(diagnostic) is not bool or (diagnostic and not allow_smoke):
        raise ValueError('Diagnostic workers require explicit --allow-smoke')
    if diagnostic:
        numeric._integer(protocol['smoke_slots'], 'smoke_slots', 2)
    elif protocol['smoke_slots'] is not None:
        raise ValueError('Full evaluation cannot shorten episodes')
    episodes = numeric._ordered_unique(protocol['episode_ids'], 'episode_ids')
    for episode in episodes:
        numeric._integer(episode, 'episode ID')
    if episodes != list(range(episodes[0], episodes[0] + len(episodes))):
        raise ValueError('Episode IDs must be a contiguous sorted range')
    worlds = numeric._ordered_unique(protocol['worlds'], 'worlds')
    if not set(worlds).issubset(ev.WORLDS):
        raise ValueError('Unknown OOD world')
    if not diagnostic and (len(episodes) != 100 or set(worlds) != set(ev.WORLDS)):
        raise ValueError('Full evaluation requires common ID plus all nine OOD worlds and 100 episodes')
    schedulers = numeric._ordered_unique(protocol['scheduler_keys'], 'scheduler_keys')
    if len(schedulers) != 3 or 'ppo' not in schedulers or not set(schedulers).issubset(ev.SCHEDULERS):
        raise ValueError('Each worker must contain its PPO and exactly two historical baselines')
    same(protocol['scheduler_names'], {key: ev.SCHEDULERS[key] for key in schedulers}, 'Scheduler names differ from the declared grid')
    for key in ('base_input_hashes', 'selected_input_hashes', 'current_source_sha256', 'runner_sha256', 'threshold_input_hashes'):
        hashes(protocol.get(key), key)
    sha(protocol.get('checkpoint_sha256'), 'checkpoint')
    sha(protocol['history_audit_sha256'], 'history audit')
    numeric._integer(protocol.get('checkpoint_update'), 'checkpoint update')
    if protocol['selected_input_hashes'].get(protocol['run'] + '/ckpt/best.pt') != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint hash is not the selected run best checkpoint')
    provenance = protocol.get('source_provenance', {})
    same(provenance.get('current_source_sha256'), protocol['current_source_sha256'], 'Current source bridge inventory mismatch')
    if provenance.get('mode') not in ('current-exact', 'reviewed-bernoulli-legacy-bridge'):
        raise ValueError('Unknown source bridge mode')
    training, base = protocol['training_config'], protocol['base_config']
    if training.get('seed') != 2024 or base.get('seed') != 2024:
        raise ValueError('This frozen three-model campaign requires training/world seed 2024')
    if training.get('traffic_model') != 'bernoulli' or base.get('traffic_model') != 'bernoulli':
        raise ValueError('This campaign requires original Bernoulli traffic')
    thresholds = protocol['thresholds']
    if (not isinstance(thresholds, dict) or thresholds.get('status') != 'validated'
            or thresholds.get('selection_rule') != ev.SELECTION_RULE or thresholds.get('selection_metric') != 'reward'
            or thresholds.get('diagnostic_only') is not diagnostic):
        raise ValueError('Missing validated, compatible pilot thresholds')
    pilot_ids = numeric._ordered_unique(thresholds.get('pilot_episode_ids'), 'pilot_episode_ids')
    for episode in pilot_ids:
        numeric._integer(episode, 'pilot episode ID')
    if set(pilot_ids) & set(episodes) or (not diagnostic and len(pilot_ids) != 8):
        raise ValueError('Pilot and test episodes overlap or pilot count is invalid')
    if protocol['candidate_thresholds'] is not None:
        raise ValueError('Evaluation must consume frozen thresholds, not sweep candidates')
    if set(protocol['configurations']) != set(worlds):
        raise ValueError('World configuration coverage differs')
    for world in worlds:
        expected_env, expected_policy = ev.make_configs(base, training, world, protocol['smoke_slots'])
        cfg = protocol['configurations'][world]
        same(cfg['environment'], expected_env, f'{world}: environment contains undeclared overrides')
        same(cfg['policy'], expected_policy, f'{world}: policy preprocessing changed beyond native K')
        threshold = thresholds.get('thresholds', {}).get(world)
        if isinstance(threshold, bool) or threshold not in ev.THRESHOLDS or cfg['baseline_threshold'] != threshold:
            raise ValueError(f'{world}: threshold differs from the fixed pilot choice')
    return run_id, worlds, episodes, schedulers


def _copies(path, protocol):
    """Verify embedded training manifests against the input inventory."""
    copies = {}
    for filename, source, inventory in (
            ('run_manifest.json', protocol['run'], protocol['selected_input_hashes']),
            ('base_manifest.json', protocol['base_run'], protocol['base_input_hashes'])):
        data = (path / filename).read_bytes()
        original = inventory.get(source + '/paper_manifest.json')
        document = json.loads(data)
        sha(original, f'{filename} original')
        if numeric.sha_bytes(data) != original:
            raise ValueError(f'{filename}: copied training manifest hash differs from input inventory')
        copies[filename] = {'sha256': numeric.sha_bytes(data), 'source_sha256': original}
        expected = protocol['training_config'] if filename == 'run_manifest.json' else protocol['base_config']
        import paper_train as pt
        same(pt.normalize_config(document['config']), expected, f'{filename}: config differs from protocol')
        if document.get('recipe') != ('narrow_mean' if source == protocol['run'] and run_number(source) == 22 else 'base'):
            raise ValueError(f'{filename}: recipe differs from selected source')
        if type(document.get('target_updates')) is not int or document['target_updates'] != 2000:
            raise ValueError(f'{filename}: source is not the completed 2000-update run')
        if filename == 'run_manifest.json':
            same(document.get('source_sha256'), protocol['source_provenance'].get('training_source_sha256'), 'Training source inventory differs from embedded manifest')
            if protocol['checkpoint_update'] >= document['target_updates']:
                raise ValueError('Selected checkpoint update exceeds training target')
    return copies


def _row(raw, protocol, label):
    if None in raw or any(raw.get(key) is None for key in ev.HEADER):
        raise ValueError(f'{label}: malformed CSV row')
    key, world = raw['scheduler'], raw['world']
    if key not in protocol['scheduler_keys'] or world not in protocol['worlds']:
        raise ValueError(f'{label}: unplanned world/scheduler')
    ppo = key == 'ppo'
    base_raw = {key: raw[key] for key in ev.metrics.HEADER}
    if not ppo:
        if raw['train_seed'] != '' or raw['checkpoint_update'] != '':
            raise ValueError(f'{label}: baseline carries policy training metadata')
        base_raw.update(train_seed='0', checkpoint_update='0')
    row = numeric._numeric_row(base_raw, ev.metrics.HEADER, label)
    cfg = protocol['configurations'][world]
    expected = {'scheduler_name': ev.SCHEDULERS[key], 'mode': 'eval',
                'run': Path(protocol['run']).name if ppo else '',
                'policy_config_sha256': ev.config_hash(cfg['policy']) if ppo else ''}
    for name, value in expected.items():
        if raw[name] != value:
            raise ValueError(f'{label}: {name} differs from protocol')
    if ppo:
        if raw['threshold'] != '' or row['train_seed'] != protocol['training_config']['seed'] or row['checkpoint_update'] != protocol['checkpoint_update']:
            raise ValueError(f'{label}: PPO checkpoint/threshold metadata differs')
        threshold = ''
    else:
        try:
            threshold = float(raw['threshold'])
        except ValueError as exc:
            raise ValueError(f'{label}: invalid baseline threshold') from exc
        if threshold != cfg['baseline_threshold']:
            raise ValueError(f'{label}: baseline threshold differs from pilot')
        row.update(train_seed='', checkpoint_update='')
    env = cfg['environment']
    try:
        env_seed = float(raw['environment_seed'])
    except ValueError as exc:
        raise ValueError(f'{label}: invalid environment seed') from exc
    slots = env['episode_len_debug' if env['debug'] else 'episode_len_main']
    if (env_seed != env['seed'] or row['slots'] != slots or row['diagnostic_only'] != int(protocol['diagnostic_only'])
            or row['episode_idx'] not in protocol['episode_ids']):
        raise ValueError(f'{label}: episode/environment metadata differs from protocol')
    if row['n_active'] != int(row['n_active']) or not 0 < row['n_active'] <= env['num_ue']:
        raise ValueError(f'{label}: invalid active population')
    if not env['n_active_min'] <= row['n_active'] <= env['n_active_max']:
        raise ValueError(f'{label}: active population outside world')
    if not env['p_arrival_min'] <= row['arrival_intensity_per_slot'] <= env['p_arrival_max']:
        raise ValueError(f'{label}: episode arrival intensity outside world')
    if not env['ue_speed_min'] - 1e-9 <= row['mean_speed_kmh'] <= env['ue_speed_max'] + 1e-9:
        raise ValueError(f'{label}: speed outside world')
    expected_rates = {'completion_rate': row['n_comp'] / max(row['n_arrivals'], 1),
                      'deadline_miss_rate': row['n_miss_deadline'] / max(row['n_arrivals'], 1),
                      'retx_drop_rate': row['n_retx_drop'] / max(row['n_arrivals'], 1),
                      'buffer_overflow_rate': row['n_buffer_overflow'] / max(row['n_offered'], 1)}
    for name, value in expected_rates.items():
        if not math.isclose(row[name], value, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{label}: {name} disagrees with raw counts')
    row.update({key: raw[key] for key in ev.EXTRA_COLUMNS})
    row.update(threshold=threshold, environment_seed=int(env_seed), source_run=protocol['run'],
               source_checkpoint_sha256=protocol['checkpoint_sha256'] if ppo else '')
    if ppo:
        number = run_number(protocol['run'])
        row.update(scheduler=f'ppo_run{number}', scheduler_name=f'PPO/run{number}')
    return row


def load_inputs(input_dirs, allow_smoke=False):
    paths = [Path(p).resolve() for p in input_dirs]
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError('Supply exactly three distinct completed worker directories')
    records, seen_runs, used_baselines, combined = [], set(), set(), {}
    common, runtime_ref, execution_ref = None, None, None
    for path in paths:
        manifest_bytes = (path / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get('status') != 'completed':
            raise ValueError(f'{path}: worker is not completed')
        protocol = manifest.get('protocol')
        run_id, worlds, episodes, schedulers = _protocol(protocol, allow_smoke)
        if run_id in seen_runs or used_baselines & (set(schedulers) - {'ppo'}):
            raise ValueError('Duplicate policy or baseline across workers')
        seen_runs.add(run_id)
        used_baselines.update(set(schedulers) - {'ppo'})
        current_common = {key: protocol[key] for key in COMMON_KEYS}
        if common is None:
            common = current_common
        else:
            same(current_common, common, 'Workers differ in common environment/pilot/history/source protocol')
        runtime = manifest.get('runtime')
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError('Missing runtime record')
        execution = manifest.get('execution', {})
        if execution.get('device') not in ('cuda', 'cpu'):
            raise ValueError('Invalid execution device')
        numeric._integer(execution.get('threads'), 'execution threads', 1)
        numeric._integer(execution.get('gpu'), 'execution GPU')
        current_execution = {key: value for key, value in execution.items() if key != 'gpu'}
        if runtime_ref is None:
            runtime_ref, execution_ref = runtime, current_execution
        else:
            same(runtime, runtime_ref, 'Workers use different software or GPU model runtimes')
            same(current_execution, execution_ref, 'Workers use incompatible execution settings')
        before, after = manifest.get('model_state_sha256_before'), manifest.get('model_state_sha256_after')
        hashes(before, 'model states')
        expected_model_keys = {ev.config_hash(protocol['configurations'][world]['policy']) for world in worlds}
        if (set(before) != expected_model_keys or len(set(before.values())) != 1 or before != after
                or manifest.get('model_state_unchanged') is not True or manifest.get('torch_rng_unchanged') is not True
                or manifest.get('inputs_unchanged') is not True):
            raise ValueError('Model/input/RNG preservation proof missing or inconsistent')
        copies = _copies(path, protocol)
        data = (path / 'metrics.csv').read_bytes()
        raw_data = (path / 'raw_metrics.jsonl').read_bytes()
        if manifest.get('metrics_sha256') != numeric.sha_bytes(data) or manifest.get('raw_metrics_sha256') != numeric.sha_bytes(raw_data):
            raise ValueError('Completed metric/raw-metric hash mismatch')
        reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
        if reader.fieldnames != ev.HEADER or manifest.get('columns') != ev.HEADER:
            raise ValueError('CSV schema differs from evaluator manifest')
        expected = {(world, ep, f'ppo_run{run_id}' if key == 'ppo' else key) for world in worlds for ep in episodes for key in schedulers}
        actual = set()
        for line, raw in enumerate(reader, 2):
            row = _row(raw, protocol, f'{path}/metrics.csv:{line}')
            key = (row['world'], row['episode_idx'], row['scheduler'])
            if key not in expected or key in actual or key in combined:
                raise ValueError('Unexpected or duplicate world/episode/scheduler row')
            actual.add(key)
            combined[key] = row
        if actual != expected or manifest.get('rows_written') != len(expected) or manifest.get('total_rows') != len(expected):
            raise ValueError('Missing rows or inconsistent completed row count')
        raw_rows = [json.loads(line) for line in raw_data.decode('utf-8').splitlines() if line]
        raw_keys = [(r['world'], r['episode_idx'], f'ppo_run{run_id}' if r['scheduler'] == 'ppo' else r['scheduler']) for r in raw_rows]
        if len(raw_keys) != len(set(raw_keys)) or set(raw_keys) != expected:
            raise ValueError('Raw metrics do not match the complete CSV episode grid')
        records.append(dict(run_id=run_id, path=str(path), manifest_sha256=numeric.sha_bytes(manifest_bytes),
                            metrics_sha256=numeric.sha_bytes(data), raw_metrics_sha256=numeric.sha_bytes(raw_data),
                            embedded_manifests=copies, protocol=protocol, model_state_sha256=next(iter(before.values())),
                            execution=execution, rows=len(actual)))
    if seen_runs != set(RUN_IDS) or used_baselines != set(BASELINES):
        raise ValueError('Campaign requires run16/run21/run22 and exactly the six historical baselines')
    schedulers = [f'ppo_run{i}' for i in RUN_IDS] + list(BASELINES)
    rows = [combined[(world, ep, key)] for world in common['worlds'] for ep in common['episode_ids'] for key in schedulers]
    # Initial episode conditions are shared within a world; realized traffic is
    # deliberately not required to match because queue admission changes RNG use.
    for world in common['worlds']:
        for ep in common['episode_ids']:
            selected = [combined[(world, ep, key)] for key in schedulers]
            for field in ('environment_seed', 'slots', 'n_active', 'arrival_intensity_per_slot', 'mean_speed_kmh'):
                if any(row[field] != selected[0][field] for row in selected[1:]):
                    raise ValueError(f'{world}/{ep}: initial episode condition {field} differs across schedulers')
    common.update(scheduler_keys=schedulers, scheduler_names={**{f'ppo_run{i}': f'PPO/run{i}' for i in RUN_IDS}, **{key: ev.SCHEDULERS[key] for key in BASELINES}})
    return common, rows, sorted(records, key=lambda item: item['run_id']), runtime_ref


def estimate(values, difference=False):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Statistics require a nonempty finite episode vector')
    center = float(values.mean())
    n = len(values)
    deviation = float(values.std(ddof=1)) if n > 1 else None
    half = float(student_t.ppf(.975, n - 1) * deviation / math.sqrt(n)) if n > 1 else None
    return {('mean_difference' if difference else 'mean'): center, 'n': n,
            'standard_deviation': deviation, 'ci95': None if half is None else [center - half, center + half]}


def build_summary(protocol, rows, inputs, runtime):
    episodes, schedulers, worlds = protocol['episode_ids'], protocol['scheduler_keys'], protocol['worlds']
    pairs = {(row['world'], row['episode_idx'], row['scheduler']): row for row in rows}
    expected = {(world, ep, key) for world in worlds for ep in episodes for key in schedulers}
    if len(pairs) != len(rows) or set(pairs) != expected:
        raise ValueError('Summary requires the complete unique comparison grid')
    by_world, mean_rows, paired_rows = {}, [], []
    ppos = [f'ppo_run{i}' for i in RUN_IDS]
    comparisons = list(itertools.product(ppos, BASELINES)) + list(itertools.combinations(ppos, 2))
    for world in worlds:
        arrays, by_scheduler = {}, {}
        for key in schedulers:
            selected = [pairs[(world, ep, key)] for ep in episodes]
            arrays[key] = {metric: np.array([row[metric] for row in selected]) for metric in MEAN_METRICS}
            metrics = {metric: estimate(value) for metric, value in arrays[key].items()}
            counts = {name: sum(row[name] for row in selected) for name in numeric.COUNT_COLUMNS}
            depth = {str(d): {'acks': counts[f'acks_m{d}'], 'units': counts[f'units_m{d}'],
                             'rate': counts[f'acks_m{d}'] / counts[f'units_m{d}'] if counts[f'units_m{d}'] else None} for d in range(1, 5)}
            by_scheduler[key] = dict(name=protocol['scheduler_names'][key], metrics=metrics, raw_count_totals=counts,
                first_ack_pooled={'acks': counts['n_units_first_ack'], 'units': counts['n_units_new'],
                                  'rate': counts['n_units_first_ack'] / counts['n_units_new'] if counts['n_units_new'] else None},
                first_ack_by_depth=depth)
            for metric, value in metrics.items():
                interval = value['ci95'] or [None, None]
                mean_rows.append(dict(world=world, scheduler=key, metric=metric, n=value['n'], mean=value['mean'],
                                      standard_deviation=value['standard_deviation'], ci95_low=interval[0], ci95_high=interval[1]))
        differences = {}
        for left, right in comparisons:
            metrics = {metric: estimate(arrays[left][metric] - arrays[right][metric], True) for metric in MEAN_METRICS}
            differences[left + '_minus_' + right] = dict(left=left, right=right, metrics=metrics)
            for metric, value in metrics.items():
                interval = value['ci95'] or [None, None]
                paired_rows.append(dict(world=world, left=left, right=right, metric=metric, n=value['n'],
                    left_mean=float(arrays[left][metric].mean()), right_mean=float(arrays[right][metric].mean()),
                    mean_difference=value['mean_difference'], standard_deviation=value['standard_deviation'],
                    ci95_low=interval[0], ci95_high=interval[1]))
        by_world[world] = dict(by_scheduler=by_scheduler, paired_differences=differences)
    summary = dict(status='completed', protocol=protocol, input_workers=inputs, runtime=runtime,
                   rows=len(rows), episodes_per_world=len(episodes), worlds=len(worlds),
                   statistics={'method': 'paired episode Student-t', 'confidence_level': .95,
                               'interval_scope': 'pointwise', 'statistical_unit': 'episode',
                               'single_episode_interval': 'undefined'},
                   by_world=by_world, caveats=CAVEATS)
    return summary, mean_rows, paired_rows


def render_markdown(summary):
    p = summary['protocol']
    lines = ['# Common-world Bernoulli OOD evaluation', '',
             f"{summary['worlds']} worlds × {summary['episodes_per_world']} episodes × 9 schedulers; {summary['rows']} validated rows.", '',
             'Three frozen best checkpoints; six historical baselines. Intervals are pointwise 95% Student-t intervals over episodes.', '']
    if p['diagnostic_only']:
        lines += ['**DIAGNOSTIC SMOKE: not paper evaluation evidence.**', '']
    for item in summary['input_workers']:
        selected = item['protocol']
        lines.append(f"- Run {item['run_id']}: `{selected['run']}`, best update {selected['checkpoint_update']}, SHA-256 `{selected['checkpoint_sha256']}`.")
    lines += ['', 'Rate columns below are percentages; raw rate differences in the CSV/JSON remain fractions.', '']
    for world, values in summary['by_world'].items():
        lines += [f'## {world}', '', '| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |',
                  '|---|---:|---:|---:|---:|---:|']
        for key in p['scheduler_keys']:
            metric = values['by_scheduler'][key]['metrics']
            gp = metric['goodput_mbps']
            ci = gp['ci95']
            interval = 'undefined' if ci is None else f'{ci[0]:.3f}, {ci[1]:.3f}'
            lines.append(f"| {p['scheduler_names'][key]} | {metric['reward']['mean']:.2f} | {gp['mean']:.3f} [{interval}] | {100*metric['deadline_miss_rate']['mean']:.3f} | {100*metric['buffer_overflow_rate']['mean']:.3f} | {metric['mu_depth']['mean']:.3f} |")
        lines += ['']
    lines += ['All 18 policy–baseline comparisons and three policy–policy comparisons per world are in `paired_statistics.csv` and `summary.json`.', '', 'Interpretation limits:', '']
    lines += ['- ' + caveat for caveat in summary['caveats']]
    return '\n'.join(lines) + '\n'


def merge(input_dirs, out, allow_smoke=False, root=ROOT):
    root, out = Path(root).resolve(), Path(out)
    out = out if out.is_absolute() else root / out
    out = out.resolve()
    results = root / 'results'
    if results.resolve() != results or out == results or not out.is_relative_to(results):
        raise ValueError('--out must be a new directory below PaperMain/results/')
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    paths = [Path(path) if Path(path).is_absolute() else root / path for path in input_dirs]
    protocol, rows, inputs, runtime = load_inputs(paths, allow_smoke)
    summary, means, paired = build_summary(protocol, rows, inputs, runtime)
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    markdown = render_markdown(summary)
    out.mkdir(parents=True, exist_ok=False)
    for name, header, values in [('metrics.csv', ev.HEADER + EXTRA_OUTPUT, rows),
                                 ('means_summary.csv', list(means[0]), means),
                                 ('paired_statistics.csv', list(paired[0]), paired)]:
        with (out / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(values)
    (out / 'summary.json').write_text(summary_text)
    (out / 'summary.md').write_text(markdown)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args(argv)
    try:
        summary = merge(args.inputs, args.out, args.allow_smoke)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Merged {summary['rows']} rows into {args.out}")


if __name__ == '__main__':
    main()
