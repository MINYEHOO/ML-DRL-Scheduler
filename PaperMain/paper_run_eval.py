#!/usr/bin/env python3
"""Read-only, sharded ID evaluation of a completed PaperMain training run.

The saved run configuration is the environment and policy configuration. Only
an explicitly diagnostic smoke run shortens the environment episode length.
Checkpoints, calibration, science sources and training records are input-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import paper_train as pt

ROOT = Path(__file__).resolve().parent
SCHEDULERS = {
    'ppo': 'PPO', 'sus_cqi': 'SUS+CQI', 'sus_cqi_m2': 'SUS+CQI@m=2',
    'sus_cqi_m3': 'SUS+CQI@m=3', 'sus_rps': 'SUS-RPS',
    'pf_greedy_sds': 'PF-Greedy-SDS', 'sumrate_greedy_sds': 'SumRate-Greedy-SDS',
}
METRIC_KEYS = [
    'reward', 'throughput_mbps', 'goodput_mbps', 'completion_rate',
    'deadline_miss_rate', 'retx_drop_rate', 'buffer_overflow_rate',
    'mu_depth', 'jain', 'first_ack_rate', 'attempts_per_acked', 'pinned_fraction',
    'mean_sinr_db', 'sinr_count', 'direction_corr', 'mean_delay', 'p95_delay', 'mean_qlen',
    'n_arrivals', 'n_offered', 'n_buffer_overflow', 'n_comp', 'n_miss_deadline',
    'n_retx_drop', 'n_retx_overflow_drop', 'terminal_queued_packets', 'n_units_new', 'n_units_first_ack',
    'n_units_acked', 'attempts_acked_sum', 'acked_bits', 'completed_bits',
    'n_active', 'mean_speed_kmh', 'arrival_intensity_per_slot',
    'realized_offered_per_active_ue_slot', 'episode_seconds', 'slot_seconds',
    'elapsed_seconds', 'reset_seconds', 'rollout_seconds',
] + [f'{prefix}_m{m}' for m in range(1, 5) for prefix in ('acks', 'units')]
HEADER = ['world', 'scheduler', 'scheduler_name', 'episode_idx', 'train_seed',
          'checkpoint_update', 'diagnostic_only', 'slots'] + METRIC_KEYS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', choices=('best', 'latest'), default='best')
    p.add_argument('--checkpoint-path', type=Path, help='optional frozen identical copy below results/')
    p.add_argument('--episode-start', type=int, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke-slots', type=int)
    p.add_argument('--history-audit', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    return p


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def below(root, value, directory, name):
    root = Path(root).resolve()
    base = root / directory
    path = (value if value.is_absolute() else root / value).resolve()
    if base.resolve() != base or path == base or not path.is_relative_to(base):
        raise ValueError(f'{name} must be below PaperMain/{directory}/')
    return path


def episode_partition(start, count, shard_index, shard_count):
    integer(start, 'episode start')
    integer(count, 'episode count', 1)
    integer(shard_count, 'shard count', 1)
    integer(shard_index, 'shard index')
    if shard_index >= shard_count or shard_count > count:
        raise ValueError('Invalid shard index/count; every shard must have an episode')
    return list(range(start, start + count))[shard_index::shard_count]


def require_same_config(left, right, description):
    # JSON canonical equality also rejects bool/int and int/float substitutions
    # in persisted metadata rather than relying on Python's True == 1.
    if pt.canonical(pt.normalize_config(left)) != pt.canonical(pt.normalize_config(right)):
        raise ValueError(description)


def validate_history_audit(document, start, count, smoke):
    if document.get('status') != 'validated':
        raise ValueError('History audit must have validated status')
    purpose = 'smoke' if smoke else 'id'
    allowed = document.get('allowed_episode_ranges')
    if not isinstance(allowed, list) or not allowed:
        raise ValueError('History audit needs allowed_episode_ranges')
    permitted = False
    for item in allowed:
        lo = integer(item.get('start'), 'audited range start')
        n = integer(item.get('episodes'), 'audited range episodes', 1)
        if item.get('purpose') == purpose and lo <= start and start + count <= lo + n:
            permitted = True
    if not permitted:
        raise ValueError('Requested episode range/purpose is not authorized by the history audit')


def plan(args, root=ROOT):
    """Validate paths, saved source/config/calibration and episode bands; no writes."""
    root = Path(root).resolve()
    selected = episode_partition(args.episode_start, args.episodes, args.shard_index, args.num_shards)
    integer(args.threads, 'threads', 1)
    integer(args.gpu, 'GPU')
    if args.smoke_slots is not None:
        integer(args.smoke_slots, 'smoke slots', 2)
    run = below(root, args.run, 'runs', '--run')
    if run.parent != root / 'runs' or not run.is_dir():
        raise ValueError('--run must be a direct run directory below PaperMain/runs/')
    out = below(root, args.out, 'results', '--out')
    if out.exists():
        raise ValueError(f'Refusing existing output directory: {out}')
    mpath, cpath = run / 'paper_manifest.json', run / 'config.json'
    manifest = json.loads(mpath.read_text())
    if type(manifest.get('version')) is not int or manifest['version'] != pt.VERSION:
        raise ValueError('Unsupported run manifest version')
    if manifest.get('source_sha256') != pt.source_hashes(root):
        raise ValueError('Run source hashes differ from current sources')
    cfg = manifest['config']
    if pt.canonical(pt.normalize_config(cfg)) != pt.canonical(cfg):
        raise ValueError('Saved run configuration is not a full normalized configuration')
    require_same_config(json.loads(cpath.read_text()), cfg, 'Run config.json differs from manifest')
    target = integer(manifest.get('target_updates'), 'target updates', 1)
    integer(cfg['seed'], 'training seed')
    if manifest.get('smoke_slots') is not None or cfg['debug']:
        raise ValueError('A diagnostic training run is not a full evaluation source')
    recipe = manifest.get('recipe')
    if recipe not in pt.RECIPES:
        raise ValueError('Unknown frozen recipe')
    if manifest.get('artifact_config_sha256') != pt.sha_file(root / 'artifacts' / recipe / 'config.json'):
        raise ValueError('Frozen recipe configuration changed')
    status_path = root / 'review' / 'logs' / f'{run.name}_launch.json'
    completion = json.loads(status_path.read_text())
    if (completion.get('status') != 'completed' or type(completion.get('returncode')) is not int
            or completion['returncode'] != 0 or completion.get('target_completed') is not True
            or integer(completion.get('checkpoint_update'), 'completion update') + 1 < target):
        raise ValueError('Source run has no successful completed-target launch record')
    full_ids = list(range(args.episode_start, args.episode_start + args.episodes))
    validation_n = integer(cfg['ppo_eval_episodes'], 'internal evaluation episodes', 1)
    if set(full_ids) & (set(range(target)) | set(range(10000, 10000 + validation_n))):
        raise ValueError('Evaluation episodes overlap source training or checkpoint-selection episodes')
    reference = pt.calibration_reference(root, manifest.get('calibration_reference_root'))
    profile = manifest.get('calibration_profile')
    if manifest.get('calibration_reference_root') is not None and profile is None:
        raise ValueError('Calibration reference requires a saved profile')
    if profile is not None:
        from calibration.profile import validate_profile_record, apply_profile, reject_calibration_episode_overlap
        validate_profile_record(profile, reference)
        if pt.canonical(apply_profile(cfg, profile['document'])) != pt.canonical(cfg):
            raise ValueError('Saved calibration beta differs from run configuration')
        offset = cfg['seed'] - profile['document']['config']['seed']
        reject_calibration_episode_overlap(profile['document'], offset + args.episode_start, args.episodes)
    checkpoint = run / 'ckpt' / f'{args.checkpoint}.pt'
    if checkpoint.resolve().parent != run / 'ckpt' or not checkpoint.is_file():
        raise ValueError('Invalid selected run checkpoint path')
    checkpoint_hash = pt.sha_file(checkpoint)
    if args.checkpoint_path is not None:
        frozen = below(root, args.checkpoint_path, 'results', '--checkpoint-path')
        if not frozen.is_file() or pt.sha_file(frozen) != checkpoint_hash:
            raise ValueError('Frozen checkpoint copy differs from selected run checkpoint')
        checkpoint = frozen
    history_hash = None
    if args.history_audit is not None:
        history = below(root, args.history_audit, 'results', '--history-audit')
        document = json.loads(history.read_text())
        validate_history_audit(document, args.episode_start, args.episodes, args.smoke_slots is not None)
        history_hash = pt.sha_file(history)
    env_cfg = copy.deepcopy(cfg)
    if args.smoke_slots is not None:
        env_cfg['debug'], env_cfg['episode_len_main'] = False, args.smoke_slots
    protocol = dict(
        name='paper-live-run-id-v1', run=run.relative_to(root).as_posix(),
        checkpoint_kind=args.checkpoint, checkpoint_sha256=checkpoint_hash,
        checkpoint_update=None, training_config=copy.deepcopy(cfg),
        environment_config=env_cfg, policy_config=copy.deepcopy(cfg),
        source_sha256=manifest['source_sha256'], runner_sha256=pt.sha_file(root / 'paper_run_eval.py'),
        run_manifest_sha256=pt.sha_file(mpath), run_config_sha256=pt.sha_file(cpath),
        completion_record_sha256=pt.sha_file(status_path),
        calibration_profile=profile, calibration_reference_root=manifest.get('calibration_reference_root'),
        episode_ids=full_ids, scheduler_keys=list(SCHEDULERS), scheduler_names=SCHEDULERS.copy(),
        diagnostic_only=args.smoke_slots is not None, history_audit_sha256=history_hash,
        caveats=[
            'ID: environment uses the complete saved training configuration; no OOD overrides.',
            'Fresh episode seeds pair topology, speeds and episode load, but shared RNG draws depend on queue admissions; scheduler traffic/CSI traces need not be identical.',
            'Finite episode boundary leaves unfinished packets; completion/miss rates use admitted arrivals and overflow rate uses all offered arrivals.',
            'Calibration is inherited unchanged; a transferred Bernoulli calibration does not guarantee FTP3 first-ACK targets.',
        ],
    )
    execution = dict(shard_index=args.shard_index, shard_count=args.num_shards,
                     gpu=str(args.gpu) if args.device == 'cuda' else None,
                     device=args.device, threads=args.threads, episode_ids=selected)
    return dict(out=out, checkpoint=checkpoint, manifest=manifest, protocol=protocol, execution=execution)


def validate_checkpoint(checkpoint, prepared):
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Checkpoint must contain a model state dictionary')
    require_same_config(checkpoint['cfg'], prepared['protocol']['training_config'],
                        'Checkpoint configuration differs from saved run')
    update = integer(checkpoint.get('update'), 'checkpoint update')
    if update >= prepared['manifest']['target_updates']:
        raise ValueError('Checkpoint update exceeds the completed training target')
    pt.validate_schedule(prepared['manifest']['schedule'], prepared['manifest']['config']['ppo_learning_rate'])
    pt.validate_checkpoint_schedule(checkpoint, prepared['manifest']['schedule'])
    prepared['protocol']['checkpoint_update'] = update


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def make_schedulers(model):
    from train_phase2 import PPOScheduler
    from baselines import (SUSCQI, SUSCQIDepth2, SUSCQIDepth3, SUSRPS, PFGreedySDS, CQIGreedySDS)
    return dict(zip(SCHEDULERS, [PPOScheduler(model), SUSCQI(), SUSCQIDepth2(),
                                SUSCQIDepth3(), SUSRPS(), PFGreedySDS(), CQIGreedySDS()]))


def validate_packet_accounting(ep, terminal_queued):
    ended = sum(ep[name] for name in ('n_comp', 'n_miss_deadline', 'n_retx_drop', 'n_retx_overflow_drop'))
    if ended + terminal_queued != ep['n_arrivals']:
        raise ValueError('Admitted packet accounting differs from completed/dropped/terminal queue counts')


def metric_row(env, cfg, key, episode, update, diagnostic, reset_seconds, rollout_seconds):
    from train_phase2 import env_episode_metrics
    ep = env.ep
    metrics = env_episode_metrics(env, cfg)
    raw = ('n_arrivals', 'n_buffer_overflow', 'n_comp', 'n_miss_deadline', 'n_retx_drop',
           'n_retx_overflow_drop', 'n_units_new', 'n_units_first_ack', 'n_units_acked',
           'attempts_acked_sum', 'acked_bits', 'completed_bits', 'sinr_count')
    metrics.update({name: ep[name] for name in raw})
    terminal_queued = sum(len(q) for q in env.traffic.queues)
    validate_packet_accounting(ep, terminal_queued)
    offered = ep['n_arrivals'] + ep['n_buffer_overflow']
    intensity = env.traffic.p_arrival_ep
    if intensity is None:
        intensity = cfg.p_arrival
    metrics.update(n_offered=offered, terminal_queued_packets=terminal_queued,
                   arrival_intensity_per_slot=intensity,
                   realized_offered_per_active_ue_slot=offered / (env.traffic.n_active * cfg.episode_len),
                   episode_seconds=cfg.episode_len * cfg.slot_duration, slot_seconds=cfg.slot_duration,
                   reset_seconds=reset_seconds, rollout_seconds=rollout_seconds,
                   elapsed_seconds=reset_seconds + rollout_seconds)
    selected = {}
    for name in METRIC_KEYS:
        value = metrics[name]
        if not math.isfinite(value):
            if name == 'mean_sinr_db' and ep['sinr_count'] == 0:
                value = ''  # mathematically undefined, not a measured zero
            else:
                raise ValueError(f'Non-finite evaluation metric {name}: {value}')
        selected[name] = value
    return dict(world='ID', scheduler=key, scheduler_name=SCHEDULERS[key], episode_idx=episode,
                train_seed=cfg.seed, checkpoint_update=update, diagnostic_only=int(diagnostic),
                slots=cfg.episode_len, **selected)


def main(argv=None):
    args = parser().parse_args(argv)
    # Configure all runtimes before importing Config/NumPy, Torch or Sionna.
    integer(args.threads, 'threads', 1)
    integer(args.gpu, 'GPU')
    pt.configure_execution(dict(device=args.device, gpu=str(args.gpu), threads=args.threads))
    prepared = plan(args)
    import csv
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(prepared['checkpoint'], map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, prepared)
    from config import Config
    from policy import ActorCritic
    model = ActorCritic(Config(**prepared['protocol']['policy_config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    if args.dry_run:
        print(json.dumps(dict(protocol=prepared['protocol'], execution=prepared['execution']), indent=2, allow_nan=False))
        return
    pt.preflight_device(torch, args.device)
    model.to(args.device)
    from env import SchedulerEnv
    cfg = Config(**prepared['protocol']['environment_config'])
    env = SchedulerEnv(cfg)
    schedulers = make_schedulers(model)
    initial_hash = model_state_hash(model)
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state_all() if args.device == 'cuda' else []
    out = prepared['out']
    out.mkdir(parents=True, exist_ok=False)
    record = dict(protocol=prepared['protocol'], execution=prepared['execution'],
                  status='running', runtime=pt.runtime_info(torch), rows_written=0,
                  model_state_sha256_before=initial_hash, columns=HEADER,
                  units={'throughput_mbps': 'Mbit/s acknowledged PHY bits',
                         'goodput_mbps': 'Mbit/s completed packet bits',
                         'mean_delay': 'slots, completed packets only', 'p95_delay': 'slots, completed packets only',
                         'arrival_intensity_per_slot': 'mean offered packets per active UE per slot',
                         'n_arrivals': 'admitted packets', 'n_offered': 'admitted + buffer-rejected packets'})
    pt.atomic_json(out / 'manifest.json', record)
    # Preserve exact consumed input metadata in addition to hashes in the manifest.
    pt.atomic_json(out / 'run_manifest.json', prepared['manifest'])
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / 'metrics.csv').open('x', newline='') as stream, (out / 'raw_metrics.jsonl').open('x') as raw_stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER)
            writer.writeheader()
            stream.flush()
            with torch.inference_mode():
                for episode in prepared['execution']['episode_ids']:
                    for key, scheduler in schedulers.items():
                        t0 = time.monotonic()
                        env.reset(episode_idx=episode)
                        t1 = time.monotonic()
                        done = False
                        while not done:
                            allocation = scheduler.schedule(env)
                            if (allocation.shape != (cfg.num_rbg, cfg.l_max)
                                    or not np.issubdtype(allocation.dtype, np.integer)
                                    or allocation.min() < 0 or allocation.max() > cfg.num_ue):
                                raise ValueError(f'Invalid action from {key}')
                            _, reward, done, _ = env.step(allocation)
                            if not np.isfinite(reward):
                                raise ValueError(f'Non-finite reward from {key}')
                        t2 = time.monotonic()
                        row = metric_row(env, cfg, key, episode, prepared['protocol']['checkpoint_update'],
                                         prepared['protocol']['diagnostic_only'], t1 - t0, t2 - t1)
                        writer.writerow(row)
                        stream.flush()
                        raw_ep = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in env.ep.items()}
                        raw_stream.write(json.dumps(dict(episode_idx=episode, scheduler=key, ep=raw_ep,
                            ue_speeds_kmh=env.ue_speeds_kmh.tolist(), n_active=env.traffic.n_active,
                            arrival_intensity_per_slot=row['arrival_intensity_per_slot']), allow_nan=False) + '\n')
                        raw_stream.flush()
                        record['rows_written'] += 1
                        record['last_completed'] = dict(episode_idx=episode, scheduler=key)
                        record['elapsed_seconds'] = time.monotonic() - started
                        pt.atomic_json(out / 'progress.json', dict(status='running', rows_written=record['rows_written'],
                            total_rows=len(prepared['execution']['episode_ids']) * len(SCHEDULERS),
                            last_completed=record['last_completed'], elapsed_seconds=record['elapsed_seconds']))
                        print(f'episode={episode} scheduler={key} rows={record["rows_written"]} reward={row["reward"]:.3f} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
        record['model_state_sha256_after'] = model_state_hash(model)
        if record['model_state_sha256_after'] != initial_hash:
            raise ValueError('Evaluation mutated model parameters or normalizer buffers')
        if not torch.equal(torch.get_rng_state(), initial_cpu_rng):
            raise ValueError('Deterministic evaluation changed Torch CPU RNG')
        if args.device == 'cuda' and any(not torch.equal(a, b) for a, b in zip(initial_cuda_rng, torch.cuda.get_rng_state_all())):
            raise ValueError('Deterministic evaluation changed Torch CUDA RNG')
        if pt.source_hashes(ROOT) != prepared['protocol']['source_sha256'] or pt.sha_file(ROOT / 'paper_run_eval.py') != prepared['protocol']['runner_sha256']:
            raise ValueError('Evaluation source files changed during execution')
        if pt.sha_file(prepared['checkpoint']) != prepared['protocol']['checkpoint_sha256']:
            raise ValueError('Evaluation input checkpoint changed during execution')
        record.update(status='completed', model_state_unchanged=True, torch_rng_unchanged=True,
                      metrics_sha256=pt.sha_file(out / 'metrics.csv'), raw_metrics_sha256=pt.sha_file(out / 'raw_metrics.jsonl'))
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        pt.atomic_json(out / 'manifest.json', record)
        pt.atomic_json(out / 'progress.json', dict(status=record['status'], rows_written=record['rows_written'],
            total_rows=len(prepared['execution']['episode_ids']) * len(SCHEDULERS),
            last_completed=record.get('last_completed'), elapsed_seconds=record['elapsed_seconds']))
    print(out, flush=True)


if __name__ == '__main__':
    main()
