#!/usr/bin/env python3
"""Read-only preflight and command plan for the Run 30/34/35 ID comparison.

The default preflight requires completed runs and validates CPU checkpoint
metadata and strict model loading. --print-commands only renders a proposal;
it is not evidence that inputs, history, or checkpoints passed validation.
Each evaluation command starts a separate process for its model variant.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import sys

import paper_run_eval as ev
import paper_train as pt

ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / 'evaluation/raw_phase_30_34_35.json'


def validate_spec(spec):
    if type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Unsupported comparison specification')
    runs = spec['runs']
    expected = [(30, 'raw_invariant', 3e-4), (34, 'raw_phase_inv', 3e-4),
                (35, 'raw_phase_inv_lr6', 6e-4)]
    if [(r['id'], r['recipe'], r['learning_rate']) for r in runs] != expected:
        raise ValueError('Comparison must retain Run 30/34/35 recipe and LR assignments')
    if any(type(r['id']) is not int or type(r['learning_rate']) is not float for r in runs):
        raise ValueError('Run IDs and learning rates have invalid types')
    if len({r['run'] for r in runs}) != 3:
        raise ValueError('Comparison run paths must be distinct')
    if spec['checkpoint'] not in ('best', 'latest'):
        raise ValueError('Invalid common checkpoint policy')
    if spec['scheduler_keys'] != list(ev.SCHEDULERS):
        raise ValueError('Comparison scheduler set differs from the ID evaluator')
    ev.integer(spec['threads'], 'threads', 1)
    for key in ('episode_start', 'smoke_episode_start', 'seed'):
        ev.integer(spec[key], key)
    for key in ('episodes', 'smoke_episodes', 'num_shards', 'target_updates', 'episode_slots'):
        ev.integer(spec[key], key, 1)
    ev.integer(spec['smoke_slots'], 'smoke_slots', 2)
    ev.episode_partition(spec['episode_start'], spec['episodes'], 0, spec['num_shards'])
    full = set(range(spec['episode_start'], spec['episode_start'] + spec['episodes']))
    smoke = set(range(spec['smoke_episode_start'], spec['smoke_episode_start'] + spec['smoke_episodes']))
    if full & smoke:
        raise ValueError('Smoke episodes must be separate from full evaluation')


def evaluation_args(spec, run, shard, *, smoke=False, device='cpu', gpu=0):
    """Return argv usable by both the ordinary evaluator and its --dry-run."""
    suffix = 'smoke' if smoke else f'shard{shard}'
    argv = ['--run', run['run'], '--checkpoint', spec['checkpoint'],
            '--episode-start', str(spec['smoke_episode_start'] if smoke else spec['episode_start']),
            '--episodes', str(spec['smoke_episodes'] if smoke else spec['episodes']),
            '--num-shards', '1' if smoke else str(spec['num_shards']),
            '--shard-index', '0' if smoke else str(shard),
            '--device', device, '--gpu', str(gpu), '--threads', str(spec['threads']),
            '--history-audit', spec['history_audit'],
            '--out', f"{spec['output_root']}/run{run['id']}/{suffix}"]
    if smoke:
        argv += ['--smoke-slots', str(spec['smoke_slots'])]
    return argv


def validate_cohort(spec, prepared):
    """Check scientific comparability after each run passed ev.plan()."""
    validate_spec(spec)
    if len(prepared) != len(spec['runs']):
        raise ValueError('Missing comparison run')
    reference = prepared[0]
    reference_cfg = copy.deepcopy(reference['protocol']['training_config'])
    reference_cfg.pop('ppo_learning_rate')
    for run, item in zip(spec['runs'], prepared):
        manifest, protocol = item['manifest'], item['protocol']
        if manifest['recipe'] != run['recipe'] or protocol['run'] != run['run']:
            raise ValueError(f"Run {run['id']} identity/recipe mismatch")
        cfg = copy.deepcopy(protocol['training_config'])
        if pt.canonical(cfg.pop('ppo_learning_rate')) != pt.canonical(run['learning_rate']):
            raise ValueError(f"Run {run['id']} learning rate mismatch")
        if pt.canonical(cfg) != pt.canonical(reference_cfg):
            raise ValueError('Configurations differ beyond the predeclared learning rate')
        for key, expected in [('seed', spec['seed']), ('episode_len_main', spec['episode_slots']),
                              ('debug', False), ('traffic_model', 'bernoulli')]:
            if pt.canonical(cfg[key]) != pt.canonical(expected):
                raise ValueError(f'Unexpected common configuration field: {key}')
        if (pt.canonical(manifest['target_updates']) != pt.canonical(spec['target_updates'])
                or pt.canonical(manifest['schedule']) != pt.canonical(spec['schedule'])):
            raise ValueError('Training budget or LR schedule differs')
        if protocol['checkpoint_kind'] != spec['checkpoint'] or protocol['diagnostic_only']:
            raise ValueError('Full comparison requires one common checkpoint policy')
        for key in ('source_sha256', 'calibration_profile', 'calibration_reference_root',
                    'episode_ids', 'scheduler_keys', 'scheduler_names', 'history_audit_sha256'):
            if pt.canonical(protocol[key]) != pt.canonical(reference['protocol'][key]):
                raise ValueError(f'Comparison protocol mismatch: {key}')
        if protocol['history_audit_sha256'] is None:
            raise ValueError('Comparison requires a validated history audit')
        if protocol['episode_ids'] != list(range(spec['episode_start'], spec['episode_start'] + spec['episodes'])):
            raise ValueError('Comparison episode IDs differ from the specification')
        if protocol['scheduler_keys'] != spec['scheduler_keys']:
            raise ValueError('Comparison schedulers differ from the specification')
        ev.require_same_config(protocol['environment_config'], protocol['training_config'],
                               'Full ID evaluation must retain the saved environment')
        ev.require_same_config(protocol['policy_config'], protocol['training_config'],
                               'Evaluation must retain the saved policy configuration')


def commands(spec, device='cpu', gpu=0):
    validate_spec(spec)
    result = []
    for run in spec['runs']:
        smoke = [sys.executable, 'paper_run_eval.py', *evaluation_args(spec, run, 0, smoke=True, device=device, gpu=gpu)]
        full = [[sys.executable, 'paper_run_eval.py', *evaluation_args(spec, run, i, device=device, gpu=gpu)]
                for i in range(spec['num_shards'])]
        merge = [sys.executable, 'paper_run_eval_merge.py', '--shards',
                 *[f"{spec['output_root']}/run{run['id']}/shard{i}" for i in range(spec['num_shards'])],
                 '--out', f"{spec['output_root']}/run{run['id']}/merged",
                 '--bootstrap-replicates', '10000', '--bootstrap-seed', '20261022']
        result.append(dict(run_id=run['id'], dry_run=shlex.join(full[0] + ['--dry-run']),
                           smoke=shlex.join(smoke), evaluation=[shlex.join(c) for c in full],
                           merge=shlex.join(merge)))
    return result


def preflight(spec, root=ROOT):
    validate_spec(spec)
    pt.configure_execution(dict(device='cpu', gpu='0', threads=spec['threads']))
    # The metadata planner configures no accelerator and creates no outputs.
    prepared = []
    for run in spec['runs']:
        for shard in range(spec['num_shards']):
            item = ev.plan(ev.parser().parse_args(evaluation_args(spec, run, shard)), root)
            if shard == 0:
                prepared.append(item)
        ev.plan(ev.parser().parse_args(evaluation_args(spec, run, 0, smoke=True)), root)
    validate_cohort(spec, prepared)
    import torch
    from paper_model_loader import load_model
    torch.set_num_threads(spec['threads'])
    validated = []
    for run, item in zip(spec['runs'], prepared):
        checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
        ev.validate_checkpoint(checkpoint, item)
        model = load_model(item['manifest'], checkpoint, root=root)
        model.eval().requires_grad_(False)
        validated.append(dict(run_id=run['id'], checkpoint_sha256=item['protocol']['checkpoint_sha256'],
                              checkpoint_update=item['protocol']['checkpoint_update']))
    return dict(status='validated', scope='metadata, checkpoint metadata and strict CPU model loading; no rollout',
                episode_ids=prepared[0]['protocol']['episode_ids'], runs=validated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--print-commands', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='execution commands only')
    parser.add_argument('--gpu', type=int, default=0, help='execution commands only; select an available GPU')
    args = parser.parse_args(argv)
    ev.integer(args.gpu, 'GPU')
    spec = json.loads(args.spec.read_text())
    record = dict(status='commands_only_not_validated') if args.print_commands else preflight(spec)
    record['commands'] = commands(spec, args.device, args.gpu)
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
