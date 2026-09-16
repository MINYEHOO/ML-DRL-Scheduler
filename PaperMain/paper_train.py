#!/usr/bin/env python3
"""Portable fresh/resume training for the three frozen paper recipes.

Examples:
  python paper_train.py --recipe base --name base_seed2024 --device cuda --gpu 0
  python paper_train.py --recipe lrann --name smoke_lrann --smoke-slots 32 --num-updates 1
  python paper_train.py --recipe lrann --resume runs/smoke_lrann/ckpt/latest.pt --num-updates 2

Smoke shortens only episode length, PPO epochs/minibatches and validation count;
all 32 users, wireless/channel, queue and policy dimensions remain the recipe's.
It creates a distinct run and is never suitable as a paper performance result.
Archived artifacts are input-only. Resume accepts only a new run's latest.pt.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
VERSION = 1
SOURCES = ('paper_train.py', 'train_phase2.py', 'config.py', 'env.py',
           'channel.py', 'csi.py', 'codebook.py', 'phy.py', 'la_planner.py',
           'traffic.py', 'transmission.py', 'policy.py', 'ppo.py',
           'baselines.py', 'metrics.py', 'calibration/profile.py')
RECIPES = {
    # The historical base wrapper targeted 1500; its archived run stopped at
    # update 866. This is its launch recipe, not a claim of bitwise retraining.
    'base': {'default_updates': 1500, 'lr_final': None, 'lr_decay_updates': None},
    'lrann': {'default_updates': 866, 'lr_final': 0.0, 'lr_decay_updates': 866},
    'narrow': {'default_updates': 866, 'lr_final': None, 'lr_decay_updates': None},
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--recipe', required=True, choices=tuple(RECIPES))
    p.add_argument('--name', help='new directory name under PaperMain/runs')
    p.add_argument('--resume', type=Path, help='runs/<name>/ckpt/latest.pt only')
    p.add_argument('--seed', type=int, help='fresh-run training seed override')
    p.add_argument('--num-updates', type=int, help='total update count, not additional updates')
    p.add_argument('--lr-final', type=float,
                   help='fresh-run linear LR endpoint; requires --lr-decay-updates')
    p.add_argument('--lr-decay-updates', type=int,
                   help='fixed linear LR horizon (1..10000); requires --lr-final; restored on resume')
    p.add_argument('--lr-initial', type=float,
                   help='explicit LR for a new continuation segment; requires --lr-start-update and a linear schedule')
    p.add_argument('--lr-start-update', type=int,
                   help='global update index at which the explicit LR segment starts; restored on resume')
    p.add_argument('--replay-mode', choices=('sequential', 'batched'),
                   help='fresh-run PPO replay override; defaults to recipe and is restored on resume')
    p.add_argument('--traffic-model', choices=('bernoulli', 'ftp3'),
                   help='arrival process only: FTP3 uses Poisson counts with mean p_arrival per UE per slot; restored on resume')
    p.add_argument('--device', choices=('cpu', 'cuda'))
    p.add_argument('--gpu', help='single physical GPU identifier; requires cuda')
    p.add_argument('--threads', type=int)
    p.add_argument('--smoke-slots', type=int, help='separate execution test with shortened episodes')
    p.add_argument('--calibration-profile', type=Path,
                   help='validated CQI4 summary JSON; new runs only, restored automatically on resume')
    p.add_argument('--calibration-reference-root', type=Path,
                   help='explicit beta transfer from a frozen source snapshot under provenance/; requires a profile')
    p.add_argument('--dry-run', action='store_true', help='validate and print manifest without writes/training')
    return p


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root):
    return {name: sha_file(root / name) for name in SOURCES}


def _fail(message):
    raise ValueError(message)


def normalize_config(raw):
    from config import Config
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - known - {'git_hash', 'git_dirty_py'}
    if unknown:
        _fail(f'unknown configuration fields: {sorted(unknown)}')
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.validate_la()
    cfg.validate_traffic()
    return json.loads(canonical(dataclasses.asdict(cfg)))


def calibration_reference(root, reference):
    """Resolve an explicit source-world transfer; normal validation stays strict.

    The profile validator checks every frozen calibration source hash. Only the
    traffic implementation/configuration may differ in the destination world;
    PHY, CSI, calibration algorithm and reference recipe must remain identical.
    """
    if reference is None:
        return root
    from calibration.profile import source_hashes as science_hashes
    path = Path(reference)
    path = (path if path.is_absolute() else root / path).resolve()
    provenance = (root / 'provenance').resolve()
    if (provenance != root / 'provenance' or not path.is_relative_to(provenance)
            or path == provenance or not path.is_dir()):
        _fail('calibration reference must be a frozen directory inside PaperMain/provenance')
    current, frozen = science_hashes(root), science_hashes(path)
    if set(current) != set(frozen):
        _fail('calibration reference source inventory differs')
    changed = {name for name in current if current[name] != frozen[name]}
    if changed - {'config.py', 'traffic.py'}:
        _fail(f'calibration transfer changes non-traffic science sources: {sorted(changed)}')
    if sha_file(root / 'artifacts/base/config.json') != sha_file(path / 'artifacts/base/config.json'):
        _fail('calibration transfer reference Base configuration changed')
    return path


def validate_schedule(schedule, initial_lr):
    """Validate the persisted driver arguments without changing their horizon."""
    base_keys = {'lr_final', 'lr_decay_updates'}
    segment_keys = {'lr_initial', 'lr_start_update'}
    if not isinstance(schedule, dict) or set(schedule) not in (base_keys, base_keys | segment_keys):
        _fail('invalid saved LR schedule')
    if 'lr_initial' in schedule:
        segment_initial, start = schedule['lr_initial'], schedule['lr_start_update']
        if (isinstance(segment_initial, bool) or not isinstance(segment_initial, (float, int))
                or not math.isfinite(segment_initial) or not 0 < segment_initial <= initial_lr):
            _fail('--lr-initial must be finite, positive, and no greater than the configured initial LR')
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < 10000:
            _fail('--lr-start-update must be an integer in 0..9999')
        initial_lr = segment_initial
    final, horizon = schedule['lr_final'], schedule['lr_decay_updates']
    if final is None and horizon is None and 'lr_initial' not in schedule:
        return
    if (isinstance(final, bool) or not isinstance(final, (float, int))
            or not math.isfinite(final) or not 0 <= final <= initial_lr):
        _fail(f'--lr-final must be finite and between 0 and the initial LR ({initial_lr})')
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 10000:
        _fail('--lr-decay-updates must be an integer in 1..10000')
    if schedule.get('lr_start_update', 0) + horizon > 10000:
        _fail('LR segment endpoint must not exceed update 10000')


def validate_checkpoint_schedule(checkpoint, schedule):
    """Reject incompatible resumes before writes; legacy payloads had no schedule."""
    if int(checkpoint['update']) + 1 < schedule.get('lr_start_update', 0):
        _fail('checkpoint precedes the LR segment start')
    if 'lr_schedule' in checkpoint:
        validate_schedule(checkpoint['lr_schedule'], checkpoint['cfg']['ppo_learning_rate'])
        if checkpoint['lr_schedule'] != schedule:
            _fail('checkpoint LR schedule differs from the saved manifest')


def plan(args, root=ROOT):
    """Read-only preparation. All path/config/CLI/source checks precede writes."""
    root = root.resolve()
    if args.num_updates is not None and not 1 <= args.num_updates <= 10000:
        _fail('--num-updates must be 1..10000 (training/validation seed separation)')
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        _fail('--seed must be an integer in 0..2**32-1')
    if args.smoke_slots is not None and not 8 <= args.smoke_slots <= 256:
        _fail('--smoke-slots must be 8..256')
    if args.threads is not None and args.threads < 1:
        _fail('--threads must be positive')
    if args.gpu is not None and (',' in args.gpu or not args.gpu.strip()):
        _fail('--gpu must identify one GPU')
    if (args.lr_final is None) != (args.lr_decay_updates is None):
        _fail('--lr-final and --lr-decay-updates must be supplied together')
    requested_schedule = (None if args.lr_final is None else
                          {'lr_final': args.lr_final, 'lr_decay_updates': args.lr_decay_updates})
    if (args.lr_initial is None) != (args.lr_start_update is None):
        _fail('--lr-initial and --lr-start-update must be supplied together')
    if args.lr_initial is not None:
        if requested_schedule is None:
            _fail('--lr-initial and --lr-start-update require --lr-final and --lr-decay-updates')
        requested_schedule.update(lr_initial=args.lr_initial, lr_start_update=args.lr_start_update)
    requested_replay = None if args.replay_mode is None else args.replay_mode == 'batched'
    recipe_schedule = {k: RECIPES[args.recipe][k] for k in ('lr_final', 'lr_decay_updates')}
    recipe_file = root / 'artifacts' / args.recipe / 'config.json'
    artifact_hash = sha_file(recipe_file)
    hashes = source_hashes(root)
    if args.resume:
        if args.name:
            _fail('--name and --resume are mutually exclusive')
        checkpoint = args.resume if args.resume.is_absolute() else root / args.resume
        checkpoint = checkpoint.resolve()
        run_dir = checkpoint.parent.parent
        runs_root = (root / 'runs').resolve()
        if (runs_root != root / 'runs'
                or checkpoint.name != 'latest.pt' or checkpoint.parent.name != 'ckpt'
                or run_dir.parent != runs_root or not checkpoint.is_file()):
            _fail('resume must use runs/<name>/ckpt/latest.pt; artifacts cannot be modified')
        manifest_path = run_dir / 'paper_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('version') != VERSION or manifest.get('recipe') != args.recipe:
            _fail('resume manifest version/recipe mismatch')
        if manifest['source_sha256'] != hashes:
            _fail('training source changed; preserve this run and create a new run')
        if manifest['artifact_config_sha256'] != artifact_hash:
            _fail('frozen recipe configuration changed')
        configure_execution(manifest['execution'])
        cfg = manifest['config']
        saved_profile = manifest.get('calibration_profile')
        saved_reference = manifest.get('calibration_reference_root')
        reference_root = calibration_reference(root, saved_reference)
        if saved_reference is not None and saved_profile is None:
            _fail('calibration reference requires a saved profile')
        if args.calibration_reference_root is not None:
            supplied_reference = calibration_reference(root, args.calibration_reference_root)
            if supplied_reference != reference_root:
                _fail('calibration reference cannot change on resume')
        if saved_profile is not None:
            from calibration.profile import (validate_profile_record, load_profile_record, apply_profile)
            validate_profile_record(saved_profile, reference_root)
            if apply_profile(cfg, saved_profile['document']) != cfg:
                _fail('saved calibration beta differs from the run configuration')
        if args.calibration_profile is not None:
            if saved_profile is None:
                _fail('--calibration-profile cannot be added on resume; create a new run')
            supplied_path = args.calibration_profile
            if not supplied_path.is_absolute():
                supplied_path = root / supplied_path
            supplied_profile = load_profile_record(supplied_path, reference_root)
            if any(supplied_profile[k] != saved_profile[k]
                   for k in ('file_sha256', 'document_sha256', 'document')):
                _fail('--calibration-profile cannot change on resume; create a new run')
        if normalize_config(cfg) != cfg:
            _fail('saved configuration is incompatible with current Config')
        # The on-disk config and checkpoint must tell the same story.
        if normalize_config(json.loads((run_dir / 'config.json').read_text())) != cfg:
            _fail('run config.json differs from the manifest')
        checks = ((args.seed, cfg['seed'], '--seed'),
                  (args.traffic_model, cfg['traffic_model'], '--traffic-model'),
                  (args.smoke_slots, manifest['smoke_slots'], '--smoke-slots'),
                  (args.device, manifest['execution']['device'], '--device'),
                  (args.gpu, manifest['execution']['gpu'], '--gpu'),
                  (args.threads, manifest['execution']['threads'], '--threads'))
        for supplied, saved, flag in checks:
            if supplied is not None and supplied != saved:
                _fail(f'{flag} cannot change on resume: saved={saved!r}, requested={supplied!r}')
        schedule = manifest['schedule']
        # Old manifests have no origin and retain their strict recipe binding.
        origin = manifest.get('schedule_origin', 'recipe')
        if origin not in ('recipe', 'override'):
            _fail('unknown saved LR schedule origin')
        validate_schedule(schedule, cfg['ppo_learning_rate'])
        if origin == 'override' and schedule.get('lr_final') is None:
            _fail('saved LR override must contain a linear schedule')
        if origin == 'recipe' and schedule != recipe_schedule:
            _fail('saved schedule differs from the frozen recipe')
        if requested_schedule is not None and requested_schedule != schedule:
            _fail('LR schedule cannot change on resume; create a new run')
        replay_origin = manifest.get('replay_origin', 'recipe')
        if replay_origin not in ('recipe', 'override'):
            _fail('unknown saved replay origin')
        saved_replay = cfg['ppo_batched_replay']
        if type(saved_replay) is not bool:
            _fail('saved ppo_batched_replay must be a boolean')
        recipe_replay = normalize_config(json.loads(recipe_file.read_text()))['ppo_batched_replay']
        if type(recipe_replay) is not bool:
            _fail('frozen recipe ppo_batched_replay must be a boolean')
        if replay_origin == 'recipe' and saved_replay != recipe_replay:
            _fail('saved replay mode differs from the frozen recipe')
        if requested_replay is not None and requested_replay != saved_replay:
            _fail('replay mode cannot change on resume; create a new run')
        traffic_origin = manifest.get('traffic_origin', 'recipe')
        if traffic_origin not in ('recipe', 'override'):
            _fail('unknown saved traffic origin')
        recipe_traffic = normalize_config(json.loads(recipe_file.read_text()))['traffic_model']
        if traffic_origin == 'recipe' and cfg['traffic_model'] != recipe_traffic:
            _fail('saved traffic model differs from the frozen recipe')
        # No CLI override means restore the saved override, not the recipe.
        manifest['target_updates'] = args.num_updates or manifest['target_updates']
    else:
        if not args.name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
            _fail('fresh training needs --name using letters, digits, underscores, dots or hyphens')
        run_dir = root / 'runs' / args.name
        if run_dir.exists():
            _fail(f'run directory already exists: {run_dir}; use --resume or another --name')
        # Reject a symlinked runs directory that could redirect writes elsewhere.
        if run_dir.resolve().parent != root / 'runs':
            _fail('runs directory must remain inside this PaperMain package')
        device = args.device or ('cuda' if args.gpu is not None else 'cpu')
        if device == 'cpu' and args.gpu is not None:
            _fail('--gpu requires --device cuda')
        execution = {'device': device, 'gpu': (args.gpu or '0') if device == 'cuda' else None,
                     'threads': args.threads or 4}
        configure_execution(execution)
        cfg = normalize_config(json.loads(recipe_file.read_text()))
        if type(cfg['ppo_batched_replay']) is not bool:
            _fail('frozen recipe ppo_batched_replay must be a boolean')
        if requested_replay is not None:
            cfg['ppo_batched_replay'] = requested_replay
        if args.traffic_model is not None:
            cfg['traffic_model'] = args.traffic_model
        cfg = normalize_config(cfg)
        profile_record = None
        if args.calibration_reference_root is not None and args.calibration_profile is None:
            _fail('--calibration-reference-root requires --calibration-profile')
        reference_root = calibration_reference(root, args.calibration_reference_root)
        if args.calibration_profile is not None:
            from calibration.profile import load_profile_record, apply_profile
            profile_path = args.calibration_profile
            if not profile_path.is_absolute():
                profile_path = root / profile_path
            profile_record = load_profile_record(profile_path, reference_root)
            cfg = apply_profile(cfg, profile_record['document'])
        if args.seed is not None:
            cfg['seed'] = args.seed
        if args.smoke_slots is not None:
            cfg.update(episode_len_main=args.smoke_slots, episode_len_debug=args.smoke_slots,
                       ppo_epochs=1, ppo_minibatch_size=max(1, args.smoke_slots - 8),
                       ppo_eval_every=1, ppo_eval_episodes=1, ppo_save_every=1)
        schedule = requested_schedule if requested_schedule is not None else recipe_schedule
        validate_schedule(schedule, cfg['ppo_learning_rate'])
        if schedule.get('lr_start_update', 0) != 0:
            _fail('a nonzero LR segment start requires a continuation checkpoint in a new run')
        manifest = {
            'version': VERSION, 'recipe': args.recipe,
            'config': cfg, 'artifact_config_sha256': artifact_hash,
            'source_sha256': hashes,
            'schedule': schedule,
            'schedule_origin': 'override' if requested_schedule is not None else 'recipe',
            'replay_origin': 'override' if requested_replay is not None else 'recipe',
            'traffic_origin': 'override' if args.traffic_model is not None else 'recipe',
            'target_updates': args.num_updates or RECIPES[args.recipe]['default_updates'],
            'smoke_slots': args.smoke_slots,
            'purpose': 'execution_smoke' if args.smoke_slots else 'training',
            'execution': execution,
            'runtime': None,
            'calibration_profile': profile_record,
            'calibration_reference_root': (reference_root.relative_to(root).as_posix()
                                           if reference_root != root else None),
            'calibration_application': ('Bernoulli Base-calibrated beta transferred to FTP3 traffic; no FTP3 holdout validation'
                                        if profile_record and cfg['traffic_model'] == 'ftp3'
                                        else 'Base-calibrated beta transferred from frozen traffic implementation'
                                        if profile_record and reference_root != root
                                        else 'Base-calibrated beta transferred to NARROW training world'
                                        if profile_record and args.recipe == 'narrow'
                                        else 'Base-calibrated beta' if profile_record else 'archived beta'),
        }
        checkpoint = None
    if manifest['config']['ppo_batched_replay'] and manifest['config']['decode_order'] != 'rbg_major':
        _fail('batched replay requires decode_order=rbg_major')
    if manifest.get('calibration_profile') is not None:
        from calibration.profile import reject_calibration_episode_overlap
        document = manifest['calibration_profile']['document']
        # ChannelGenerator receives cfg.seed + episode_idx. Compare in the
        # profile's episode coordinates so a training seed override cannot
        # silently reuse a calibrated topology/channel. A negative offset is
        # valid here: it is a relative coordinate, not an episode to execute.
        seed_offset = manifest['config']['seed'] - document['config']['seed']
        reject_calibration_episode_overlap(document, seed_offset, manifest['target_updates'])
        reject_calibration_episode_overlap(
            document, seed_offset + 10000, manifest['config']['ppo_eval_episodes'])
    return run_dir, checkpoint, manifest


def configure_execution(execution):
    n = str(execution['threads'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = n
    os.environ['CUDA_VISIBLE_DEVICES'] = execution['gpu'] if execution['device'] == 'cuda' else ''
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def runtime_info(torch):
    versions = {}
    for name in ('numpy', 'torch', 'tensorflow', 'tensorflow-cpu', 'tensorflow-macos',
                 'sionna', 'tensorboard', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'packages': versions,
            'torch_cuda': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def preflight_device(torch, device):
    """Acquire the requested CUDA context before creating any run outputs.

    is_available() alone does not detect an Exclusive_Process GPU occupied by
    another process/container. A real allocation and synchronization do.
    """
    if device != 'cuda':
        return
    if not torch.cuda.is_available():
        _fail('CUDA was requested but is unavailable; no run files were written')
    try:
        probe = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        del probe
    except RuntimeError as exc:
        _fail('CUDA preflight failed: the selected GPU may be busy or unavailable; '
              f'choose an available --gpu. No run files were written. Details: {exc}')


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run_dir, checkpoint, manifest = plan(args)
        configure_execution(manifest['execution'])
        # Numeric runtimes are imported only after thread and GPU selection.
        import torch
        torch.set_num_threads(manifest['execution']['threads'])
        preflight_device(torch, manifest['execution']['device'])
        current_runtime = runtime_info(torch)
        if checkpoint:
            if manifest['runtime'] != current_runtime:
                _fail('runtime differs from the saved run; use its original environment or a new run')
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if normalize_config(ckpt['cfg']) != manifest['config']:
                _fail('checkpoint config differs from the saved manifest')
            validate_checkpoint_schedule(ckpt, manifest['schedule'])
            if int(ckpt['update']) + 1 >= manifest['target_updates']:
                _fail('checkpoint already reached --num-updates; select a larger total if intended')
            from policy import ActorCritic
            from config import Config
            ActorCritic(Config(**manifest['config'])).load_state_dict(ckpt['model'], strict=True)
            del ckpt
        else:
            manifest['runtime'] = current_runtime
        print(json.dumps({'run_dir': str(run_dir), 'resume': str(checkpoint) if checkpoint else None,
                          **manifest}, indent=2), flush=True)
        if args.dry_run:
            return
        from config import Config
        import train_phase2 as driver
        # All incompatible-input checks are complete before the first write.
        run_dir.mkdir(parents=True, exist_ok=bool(checkpoint))
        atomic_json(run_dir / 'paper_manifest.json', manifest)
        # Independent source digest avoids inheriting the parent repository's
        # git stamp when this directory is copied into or out of another repo.
        stamp = 'sha256:' + hashlib.sha256(canonical(manifest['source_sha256']).encode()).hexdigest()
        driver._git_state = lambda: (stamp, False)
        command = ['train_phase2.py', '--mode', 'queue', '--seed', str(manifest['config']['seed']),
                   '--num_updates', str(manifest['target_updates']), '--patience_evals', '100000']
        if checkpoint:
            command += ['--resume', str(checkpoint)]
        else:
            command += ['--run_root', str(run_dir.parent), '--run_name', run_dir.name]
        if manifest['schedule']['lr_final'] is not None:
            command += ['--lr_final', str(manifest['schedule']['lr_final']),
                        '--lr_decay_updates', str(manifest['schedule']['lr_decay_updates'])]
            if 'lr_initial' in manifest['schedule']:
                command += ['--lr_initial', str(manifest['schedule']['lr_initial']),
                            '--lr_start_update', str(manifest['schedule']['lr_start_update'])]
        previous = sys.argv
        try:
            sys.argv = command
            driver.main(config_override=Config(**manifest['config']))
        finally:
            sys.argv = previous
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f'paper_train: {exc}') from exc


if __name__ == '__main__':
    main()
