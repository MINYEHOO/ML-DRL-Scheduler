#!/usr/bin/env python3
"""Frozen-checkpoint common-world Bernoulli OOD evaluation and separate SUS pilot.

The completed Base run supplies every environment. Each policy retains its own
training configuration and normalizer buffers; only the variable UE dimension
changes. Pilot rewards choose the SUS threshold before any main test is read.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time

import paper_run_eval as metrics
import paper_train as pt

ROOT = Path(__file__).resolve().parent
WORLDS = {
    'ID': {},
    'P055': {'p_arrival_min': .55, 'p_arrival_max': .55},
    'P010': {'p_arrival_min': .10, 'p_arrival_max': .10},
    'V60max': {'ue_speed_min': 5., 'ue_speed_max': 60.},
    'CSI02': {'p_csi': .2},
    'D26strict': {'deadline_min': 2, 'deadline_max': 6},
    'STORM2': {'p_arrival_min': .55, 'p_arrival_max': .55,
               'ue_speed_min': 5., 'ue_speed_max': 60., 'p_csi': .2},
    'K8': {'num_ue': 8, 'n_active_min': 8, 'n_active_max': 8},
    'K48': {'num_ue': 48, 'n_active_min': 48, 'n_active_max': 48},
    'K60': {'num_ue': 60, 'n_active_min': 60, 'n_active_max': 60},
}
SCHEDULERS = {
    'ppo': 'PPO', 'sus_cqi': 'SUS+CQI', 'su_cqi': 'SU+CQI',
    'sus_pf': 'SUS+PF', 'su_pf': 'SU+PF',
    'sus_random': 'SUS+Random', 'su_random': 'SU+Random',
}
THRESHOLDS = (.60, .65, .70, .75, .80)
SELECTION_RULE = 'maximum mean reward; ties choose smallest threshold'
EXTRA_COLUMNS = ['run', 'mode', 'threshold', 'environment_seed', 'policy_config_sha256']
HEADER = metrics.HEADER + EXTRA_COLUMNS
RUNNER_FILES = ('paper_ood_eval.py', 'paper_ood_inputs.py', 'paper_run_eval.py')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--mode', choices=('pilot', 'eval'), required=True)
    p.add_argument('--worlds', nargs='+', choices=list(WORLDS), default=list(WORLDS))
    p.add_argument('--schedulers', nargs='+', choices=list(SCHEDULERS))
    p.add_argument('--episode-start', type=int, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--history-audit', type=Path, required=True)
    p.add_argument('--thresholds', type=Path)
    p.add_argument('--smoke-slots', type=int)
    p.add_argument('--dry-run', action='store_true')
    return p


def config_hash(cfg):
    return hashlib.sha256(pt.canonical(cfg).encode()).hexdigest()


def make_configs(base_config, training_config, world, smoke_slots=None):
    if world not in WORLDS:
        raise ValueError(f'Unknown OOD world: {world}')
    if base_config.get('traffic_model') != 'bernoulli' or training_config.get('traffic_model') != 'bernoulli':
        raise ValueError('This campaign requires Bernoulli environment and policies')
    if training_config.get('deadline_max') != 12:
        raise ValueError('Expected frozen training deadline normalization /12')
    environment, policy = copy.deepcopy(base_config), copy.deepcopy(training_config)
    environment.update(WORLDS[world])
    policy['num_ue'] = environment['num_ue']
    if smoke_slots is not None:
        metrics.integer(smoke_slots, 'smoke slots', 2)
        environment.update(debug=False, episode_len_main=smoke_slots)
    return environment, policy


def validate_history(document, args, selected, base):
    purpose = 'smoke' if args.smoke_slots is not None else args.mode
    if document.get('status') != 'validated':
        raise ValueError('History audit must have validated status')
    checked = document.get('checked_run_names', [])
    if not {selected['run'].name, base['run'].name}.issubset(checked):
        raise ValueError('History audit does not include selected and common Base runs')
    if not document.get('scanned_paths'):
        raise ValueError('History audit must record scanned paths')
    allowed = document.get('allowed_episode_ranges', [])
    permitted = False
    for item in allowed:
        lo = metrics.integer(item.get('start'), 'audited range start')
        n = metrics.integer(item.get('episodes'), 'audited range episodes', 1)
        if item.get('purpose') == purpose and lo <= args.episode_start and args.episode_start + args.episodes <= lo + n:
            permitted = True
    if not permitted:
        raise ValueError('Episode range/purpose is not authorized by the history audit')
    # Independently check the selected model and Base, including calibration seeds.
    ids = set(range(args.episode_start, args.episode_start + args.episodes))
    for source in (selected, base):
        cfg, manifest = source['training_config'], source['manifest']
        forbidden = set(range(manifest['target_updates'])) | set(range(10000, 10000 + cfg['ppo_eval_episodes']))
        if ids & forbidden:
            raise ValueError('Evaluation overlaps training or checkpoint-selection episodes')
        profile = source.get('calibration_profile')
        if profile is not None:
            from calibration.profile import reject_calibration_episode_overlap
            document = profile['document']
            offset = cfg['seed'] - document['config']['seed']
            reject_calibration_episode_overlap(document, offset + args.episode_start, args.episodes)


def threshold_choices(rows, worlds, episode_ids):
    """Validate an entire pilot grid and recompute the deterministic winner."""
    grouped = {}
    expected = {(w, float(t), ep) for w in worlds for t in THRESHOLDS for ep in episode_ids}
    seen = set()
    for row in rows:
        try:
            key = (row['world'], float(row['threshold']), int(row['episode_idx']))
            reward = float(row['reward'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Malformed pilot metric row') from exc
        if key not in expected or key in seen or row.get('scheduler') != 'sus_cqi' or row.get('mode') != 'pilot':
            raise ValueError('Pilot rows are duplicated, unexpected, or use another scheduler/mode')
        if not math.isfinite(reward):
            raise ValueError('Pilot reward must be finite')
        seen.add(key)
        grouped.setdefault(key[:2], []).append(reward)
    if seen != expected:
        raise ValueError('Pilot rows do not cover every world, threshold and episode')
    return {w: max(THRESHOLDS, key=lambda t: (math.fsum(grouped[(w, t)]) / len(episode_ids), -t)) for w in worlds}


def load_thresholds(path, args, base, root=ROOT):
    path = metrics.below(root, path, 'results', '--thresholds')
    document = json.loads(path.read_text())
    if (document.get('status') != 'validated' or document.get('selection_metric') != 'reward'
            or document.get('selection_rule') != SELECTION_RULE):
        raise ValueError('Threshold document needs the fixed, validated pilot selection rule')
    diagnostic = args.smoke_slots is not None
    if type(document.get('diagnostic_only')) is not bool or document['diagnostic_only'] != diagnostic:
        raise ValueError('Diagnostic thresholds may only be consumed by diagnostic evaluation')
    pilot_ids = document.get('pilot_episode_ids')
    if not isinstance(pilot_ids, list) or len(pilot_ids) != len(set(pilot_ids)) or not pilot_ids:
        raise ValueError('Pilot episode IDs must be unique and nonempty')
    for ep in pilot_ids:
        metrics.integer(ep, 'pilot episode ID')
    if not diagnostic and len(pilot_ids) != 8:
        raise ValueError('The full SUS pilot must contain exactly eight independent episodes')
    if set(pilot_ids) & set(range(args.episode_start, args.episode_start + args.episodes)):
        raise ValueError('Pilot and evaluation episode IDs overlap')
    directories = document.get('pilot_directories', [])
    if not isinstance(directories, list) or not directories or len(directories) != len(set(directories)):
        raise ValueError('Thresholds require unique pilot source directories')
    rows, worlds, input_hashes = [], [], {str(path): pt.sha_file(path)}
    for value in directories:
        directory = metrics.below(root, Path(value), 'results', 'pilot directory')
        manifest_path, csv_path = directory / 'manifest.json', directory / 'metrics.csv'
        record = json.loads(manifest_path.read_text())
        protocol = record['protocol']
        if (record.get('status') != 'completed' or record.get('inputs_unchanged') is not True
                or record.get('model_state_unchanged') is not True
                or record.get('torch_rng_unchanged') is not True or protocol.get('mode') != 'pilot'
                or protocol.get('diagnostic_only') is not diagnostic or protocol.get('smoke_slots') != args.smoke_slots
                or protocol.get('episode_ids') != pilot_ids or protocol.get('scheduler_keys') != ['sus_cqi']
                or protocol.get('candidate_thresholds') != list(THRESHOLDS)):
            raise ValueError('Pilot source is incomplete or has incompatible diagnostic/episode settings')
        metrics.require_same_config(protocol['base_config'], base['training_config'], 'Pilot uses a different common Base configuration')
        if protocol['base_input_hashes'] != base['input_hashes']:
            raise ValueError('Pilot common Base input metadata changed')
        if protocol['current_source_sha256'] != base['source_provenance']['current_source_sha256']:
            raise ValueError('Pilot science sources differ')
        if protocol['runner_sha256'] != {name: pt.sha_file(Path(root) / name) for name in RUNNER_FILES}:
            raise ValueError('Pilot evaluator sources differ')
        if record.get('metrics_sha256') != pt.sha_file(csv_path):
            raise ValueError('Pilot metrics hash differs from completion record')
        pilot_worlds = protocol['worlds']
        if (not pilot_worlds or not set(pilot_worlds).issubset(WORLDS)
                or len(pilot_worlds) != len(set(pilot_worlds)) or set(worlds) & set(pilot_worlds)):
            raise ValueError('Pilot worlds are duplicated across shards')
        worlds.extend(pilot_worlds)
        with csv_path.open(newline='') as stream:
            shard_rows = list(csv.DictReader(stream))
        if len(shard_rows) != record.get('rows_written'):
            raise ValueError('Pilot completion row count differs from CSV')
        for row in shard_rows:
            if row['world'] not in pilot_worlds or int(row['slots']) != protocol['configurations'][row['world']]['environment']['episode_len_main']:
                raise ValueError('Pilot row world or episode length differs from manifest')
        rows.extend(shard_rows)
        input_hashes[str(manifest_path)] = pt.sha_file(manifest_path)
        input_hashes[str(csv_path)] = pt.sha_file(csv_path)
    choices = threshold_choices(rows, worlds, pilot_ids)
    if pt.canonical(document.get('thresholds')) != pt.canonical(choices) or not set(args.worlds).issubset(choices):
        raise ValueError('Frozen thresholds differ from pilot rewards or omit requested worlds')
    return document, input_hashes


def plan(args, root=ROOT):
    from paper_ood_inputs import validate_run
    root = Path(root).resolve()
    metrics.integer(args.episode_start, 'episode start')
    metrics.integer(args.episodes, 'episode count', 1)
    metrics.integer(args.threads, 'threads', 1)
    metrics.integer(args.gpu, 'GPU')
    if args.smoke_slots is not None:
        metrics.integer(args.smoke_slots, 'smoke slots', 2)
    elif args.episodes != (8 if args.mode == 'pilot' else 100):
        raise ValueError('Full pilot requires 8 episodes; full evaluation requires 100')
    schedulers = args.schedulers or (['sus_cqi'] if args.mode == 'pilot' else list(SCHEDULERS))
    if len(args.worlds) != len(set(args.worlds)) or len(schedulers) != len(set(schedulers)):
        raise ValueError('Worlds and schedulers must be unique')
    if args.mode == 'pilot' and (schedulers != ['sus_cqi'] or args.thresholds is not None):
        raise ValueError('Pilot uses SUS+CQI only and sweeps all fixed candidates')
    if args.mode == 'eval' and args.thresholds is None:
        raise ValueError('Evaluation requires thresholds frozen from a separate pilot')
    out = metrics.below(root, args.out, 'results', '--out')
    if out.exists():
        raise ValueError(f'Refusing existing output directory: {out}')
    selected = validate_run(root, args.run, checkpoint='best')
    base = validate_run(root, args.base_run, checkpoint='best')
    if base['manifest']['recipe'] != 'base' or base['training_config']['traffic_model'] != 'bernoulli':
        raise ValueError('Common environment must come from a completed Bernoulli Base recipe')
    if args.mode == 'pilot' and selected['run'] != base['run']:
        raise ValueError('Pilot must name the common Base run as --run')
    history_path = metrics.below(root, args.history_audit, 'results', '--history-audit')
    history = json.loads(history_path.read_text())
    validate_history(history, args, selected, base)
    thresholds, threshold_hashes = (None, {}) if args.mode == 'pilot' else load_thresholds(args.thresholds, args, base, root)
    configurations = {}
    for world in args.worlds:
        environment, policy = make_configs(base['training_config'], selected['training_config'], world, args.smoke_slots)
        configurations[world] = {'environment': environment, 'policy': policy,
                                 'baseline_threshold': None if thresholds is None else thresholds['thresholds'][world]}
    protocol = dict(name='paper-common-bernoulli-ood-v1', mode=args.mode,
        run=selected['run'].relative_to(root).as_posix(), base_run=base['run'].relative_to(root).as_posix(),
        training_config=selected['training_config'], base_config=base['training_config'],
        checkpoint_sha256=selected['checkpoint_sha256'], checkpoint_update=None,
        selected_input_hashes=selected['input_hashes'], base_input_hashes=base['input_hashes'],
        source_provenance=selected['source_provenance'],
        current_source_sha256=selected['source_provenance']['current_source_sha256'],
        runner_sha256={name: pt.sha_file(root / name) for name in RUNNER_FILES},
        worlds=list(args.worlds), configurations=configurations,
        scheduler_keys=schedulers, scheduler_names={key: SCHEDULERS[key] for key in schedulers},
        episode_ids=list(range(args.episode_start, args.episode_start + args.episodes)),
        candidate_thresholds=list(THRESHOLDS) if args.mode == 'pilot' else None,
        thresholds=thresholds, threshold_input_hashes=threshold_hashes,
        history_audit_sha256=pt.sha_file(history_path), diagnostic_only=args.smoke_slots is not None,
        smoke_slots=args.smoke_slots,
        caveats=[
            'Common environment is the complete saved Base configuration with only the declared OOD overrides.',
            'D26strict changes environment deadlines to 2..6; policy training normalization remains /12.',
            'Weights, return normalizers and calibration beta remain frozen. Policy config changes only num_ue, which changes native tensor dimension and native per-UE count denominators; all other feature scales remain at training values.',
            'SUS threshold is selected on separate pilot episodes and affects baseline schedulers only.',
            'K scaling also changes total offered load because per-UE arrival intensity is unchanged.',
            'Same episode seeds pair initial topology, speed and load; scheduler-dependent queue admissions can alter subsequent shared traffic/CSI RNG draws.',
            'Random schedulers use a separate generator reset per world, episode and scheduler, independent of GPU and shard order.',
            'These models share one training seed; episodes do not establish variation across independent training seeds.',
            'Finite episode boundary retains unfinished packets; admitted arrivals and offered arrivals are separate denominators.',
        ])
    return dict(out=out, selected=selected, base=base, history_path=history_path,
                protocol=protocol, execution=dict(gpu=args.gpu, device=args.device, threads=args.threads))


def make_scheduler(key, model=None):
    from baselines import SUSCQI, SUCQI, SUSPF, SUPF, SUSRandom, SURandom
    if key == 'ppo':
        from train_phase2 import PPOScheduler
        return PPOScheduler(model, deterministic=True)
    return {'sus_cqi': SUSCQI, 'su_cqi': SUCQI, 'sus_pf': SUSPF, 'su_pf': SUPF,
            'sus_random': SUSRandom, 'su_random': SURandom}[key]()


def random_seed_components(seed, world, episode, key):
    def token(value):
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')
    return [seed, episode, token(world), token(key)]


def metric_row(env, cfg, key, world, episode, update, diagnostic, reset_seconds, rollout_seconds):
    # Reuse the checked metric/accounting implementation without altering its
    # module-global scheduler registry. The placeholder only supplies its label.
    row = metrics.metric_row(env, cfg, 'sus_cqi', episode, update, diagnostic, reset_seconds, rollout_seconds)
    row.update(world=world, scheduler=key, scheduler_name=SCHEDULERS[key])
    return row


def verify_after(prepared, models, initial_hashes, cpu_rng, cuda_rng, torch, root=ROOT):
    from paper_ood_inputs import assert_inputs_unchanged
    for key, model in models.items():
        if metrics.model_state_hash(model) != initial_hashes[key]:
            raise ValueError('Evaluation mutated model parameters or normalizer buffers')
    if not torch.equal(torch.get_rng_state(), cpu_rng):
        raise ValueError('Deterministic evaluation changed Torch CPU RNG')
    if prepared['execution']['device'] == 'cuda':
        current = torch.cuda.get_rng_state_all()
        if len(current) != len(cuda_rng) or any(not torch.equal(a, b) for a, b in zip(current, cuda_rng)):
            raise ValueError('Deterministic evaluation changed Torch CUDA RNG')
    assert_inputs_unchanged(root, prepared['selected'])
    assert_inputs_unchanged(root, prepared['base'])
    for name, digest in prepared['protocol']['runner_sha256'].items():
        if pt.sha_file(Path(root) / name) != digest:
            raise ValueError('Evaluation runner sources changed during execution')
    for path, digest in prepared['protocol']['threshold_input_hashes'].items():
        if pt.sha_file(Path(path)) != digest:
            raise ValueError('Frozen threshold or pilot source changed during execution')
    if pt.sha_file(prepared['history_path']) != prepared['protocol']['history_audit_sha256']:
        raise ValueError('History audit changed during execution')


def main(argv=None):
    args = parser().parse_args(argv)
    metrics.integer(args.threads, 'threads', 1)
    metrics.integer(args.gpu, 'GPU')
    pt.configure_execution(dict(device=args.device, gpu=str(args.gpu), threads=args.threads))
    prepared = plan(args)
    import numpy as np
    import torch
    from paper_ood_inputs import validate_loaded_checkpoint, validate_runtime
    from config import Config
    from policy import ActorCritic
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(prepared['selected']['checkpoint'], map_location='cpu', weights_only=False)
    validate_loaded_checkpoint(checkpoint, prepared['selected'])
    update = prepared['selected']['checkpoint_update']
    prepared['protocol']['checkpoint_update'] = update
    runtime = pt.runtime_info(torch)
    validate_runtime(runtime, prepared['selected'])
    validate_runtime(runtime, prepared['base'])
    models, model_keys = {}, {}
    for world, config in prepared['protocol']['configurations'].items():
        key = config_hash(config['policy'])
        model_keys[world] = key
        if key not in models:
            model = ActorCritic(Config(**config['policy']))
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval().requires_grad_(False)
            models[key] = model
    if args.dry_run:
        print(json.dumps(dict(protocol=prepared['protocol'], execution=prepared['execution']), indent=2, allow_nan=False))
        return
    pt.preflight_device(torch, args.device)
    for model in models.values():
        model.to(args.device)
    from env import SchedulerEnv
    initial_hashes = {key: metrics.model_state_hash(model) for key, model in models.items()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if args.device == 'cuda' else []
    out, protocol = prepared['out'], prepared['protocol']
    expected_rows = len(args.worlds) * args.episodes * (len(THRESHOLDS) if args.mode == 'pilot' else len(protocol['scheduler_keys']))
    out.mkdir(parents=True, exist_ok=False)
    record = dict(protocol=protocol, execution=prepared['execution'], status='running', runtime=runtime,
                  rows_written=0, total_rows=expected_rows, model_state_sha256_before=initial_hashes, columns=HEADER)
    pt.atomic_json(out / 'manifest.json', record)
    shutil.copyfile(prepared['selected']['run'] / 'paper_manifest.json', out / 'run_manifest.json')
    shutil.copyfile(prepared['base']['run'] / 'paper_manifest.json', out / 'base_manifest.json')
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
                for world in args.worlds:
                    config = protocol['configurations'][world]
                    cfg = Config(**config['environment'])
                    # A fresh environment prevents episode/channel caches crossing worlds.
                    env = SchedulerEnv(cfg)
                    model = models[model_keys[world]]
                    schedulers = {key: make_scheduler(key, model) for key in protocol['scheduler_keys']}
                    combinations = [('sus_cqi', t) for t in THRESHOLDS] if args.mode == 'pilot' else [
                        (key, config['baseline_threshold'] if key != 'ppo' else None) for key in protocol['scheduler_keys']]
                    for episode in protocol['episode_ids']:
                        for key, threshold in combinations:
                            cfg.sus_ortho_threshold = config['environment']['sus_ortho_threshold'] if threshold is None else threshold
                            scheduler = schedulers[key]
                            if hasattr(scheduler, 'rng'):
                                scheduler.rng = np.random.default_rng(np.random.SeedSequence(random_seed_components(cfg.seed, world, episode, key)))
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
                            row = metric_row(env, cfg, key, world, episode, update,
                                             protocol['diagnostic_only'], t1 - t0, t2 - t1)
                            row.update(run=prepared['selected']['run'].name if key == 'ppo' else '',
                                       mode=args.mode, threshold='' if threshold is None else threshold,
                                       environment_seed=cfg.seed, policy_config_sha256=model_keys[world] if key == 'ppo' else '')
                            if key != 'ppo':
                                row.update(train_seed='', checkpoint_update='')
                            writer.writerow(row)
                            stream.flush()
                            raw_ep = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in env.ep.items()}
                            raw_stream.write(json.dumps(dict(world=world, episode_idx=episode, scheduler=key,
                                threshold=threshold, ep=raw_ep, ue_speeds_kmh=env.ue_speeds_kmh.tolist(),
                                n_active=env.traffic.n_active, arrival_intensity_per_slot=row['arrival_intensity_per_slot']), allow_nan=False) + '\n')
                            raw_stream.flush()
                            record['rows_written'] += 1
                            record['last_completed'] = dict(world=world, episode_idx=episode, scheduler=key, threshold=threshold)
                            record['elapsed_seconds'] = time.monotonic() - started
                            pt.atomic_json(out / 'progress.json', {k: record[k] for k in (
                                'status', 'rows_written', 'total_rows', 'last_completed', 'elapsed_seconds')})
                            print(f'world={world} episode={episode} scheduler={key} threshold={threshold} rows={record["rows_written"]}/{expected_rows} reward={row["reward"]:.3f} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
                    del env
        if record['rows_written'] != expected_rows:
            raise ValueError('Evaluation row count differs from the complete declared grid')
        verify_after(prepared, models, initial_hashes, cpu_rng, cuda_rng, torch)
        record.update(status='completed', model_state_unchanged=True, torch_rng_unchanged=True, inputs_unchanged=True,
                      model_state_sha256_after={key: metrics.model_state_hash(model) for key, model in models.items()},
                      metrics_sha256=pt.sha_file(out / 'metrics.csv'), raw_metrics_sha256=pt.sha_file(out / 'raw_metrics.jsonl'))
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        pt.atomic_json(out / 'manifest.json', record)
        pt.atomic_json(out / 'progress.json', {k: record.get(k) for k in (
            'status', 'rows_written', 'total_rows', 'last_completed', 'elapsed_seconds')})
    print(out, flush=True)


if __name__ == '__main__':
    main()
