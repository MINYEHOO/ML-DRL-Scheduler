#!/usr/bin/env python3
"""Prepare (--dry-run), then launch smoke/main on physical GPU 4 only."""
from datetime import datetime, timezone
from pathlib import Path
import argparse, csv, json, math, os, signal, subprocess, sys, time

ROOT = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
REFERENCE = ROOT / 'provenance/before_ftp3_sources'
NAMES = {'smoke': '17_audit_ftp3_gpu4',
         'main': '18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4'}
SCHEDULE = {'lr_final': 0.0, 'lr_decay_updates': 1000}
sys.path.insert(0, str(ROOT))
from paper_train import normalize_config, parser, plan, source_hashes


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data, exclusive=False):
    if exclusive:
        with path.open('x') as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write('\n')
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)


def free_gpu4():
    def query(fields, kind):
        return subprocess.check_output(['nvidia-smi', '-i', '4', '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader,nounits'], text=True).strip()
    assert not query('pid', 'compute-apps'), 'GPU 4 already has a compute process'
    index, used = [value.strip() for value in query('index,memory.used', 'gpu').split(',')]
    assert index == '4' and int(used) < 500, ('GPU 4 is not free', index, used)


def prepared(mode):
    total = 1 if mode == 'smoke' else 1000
    command = [sys.executable, '-u', 'paper_train.py', '--recipe', 'base', '--name', NAMES[mode],
               '--device', 'cuda', '--gpu', '4', '--threads', '4', '--seed', '2024',
               '--traffic-model', 'ftp3', '--replay-mode', 'batched', '--lr-final', '0',
               '--lr-decay-updates', '1000', '--num-updates', str(total), '--calibration-profile',
               'results/cqi4_hl_corrected_v2/summary.json', '--calibration-reference-root',
               'provenance/before_ftp3_sources']
    if mode == 'smoke':
        command += ['--smoke-slots', '32']
    run, checkpoint, manifest = plan(parser().parse_args(command[3:]), ROOT)
    assert checkpoint is None and manifest['schedule'] == SCHEDULE
    assert manifest['execution'] == {'device': 'cuda', 'gpu': '4', 'threads': 4}
    reference_hashes = source_hashes(REFERENCE)
    run16 = ROOT / 'runs/16_base_cqi4_lrrestart2000_s2024_20260910_gpu5/paper_manifest.json'
    assert reference_hashes == json.loads(run16.read_text())['source_sha256']
    if mode == 'main':
        prior = ROOT / 'runs/14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5/paper_manifest.json'
        expected = normalize_config(json.loads(prior.read_text())['config'])
        assert expected['traffic_model'] == 'bernoulli'
        expected['traffic_model'] = 'ftp3'
        assert manifest['config'] == expected, 'Main config differs beyond traffic_model'
    return dict(run_dir=str(run), resume=None, command=command,
                reference_source_sha256=reference_hashes, **manifest)


def supervise(mode, record):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state = json.loads(record.read_text())
    run = Path(state['run_dir'])
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    previous = None
    try:
        import torch
        torch.set_num_threads(1)
        phases = [(state['command'], 1 if mode == 'smoke' else 1000)]
        if mode == 'smoke':
            phases.append(([sys.executable, '-u', 'paper_train.py', '--recipe', 'base',
                            '--resume', str(run / 'ckpt/latest.pt'), '--num-updates', '2'], 2))
        for command, total in phases:
            assert source_hashes(ROOT) == state['source_sha256']
            assert source_hashes(REFERENCE) == state['reference_source_sha256']
            free_gpu4()
            started = time.monotonic()
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
            state.update(status='running', supervisor_pid=os.getpid(), training_pid=child.pid,
                         phase_target_updates=total, active_command=command)
            write_json(record, state)
            code = child.wait()
            state['returncode'] = code
            write_json(record, state)
            checkpoint = run / 'ckpt/latest.pt'
            if not checkpoint.exists():
                raise RuntimeError(f'training exited {code} without a checkpoint')
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            update = int(ck['update'])
            state['checkpoint_update'] = update
            assert ck['lr_schedule'] == SCHEDULE
            assert normalize_config(ck['cfg']) == state['config']
            assert all(torch.isfinite(value).all() for value in ck['model'].values())
            expected_lr = state['config']['ppo_learning_rate'] * (1 - update / 1000)
            assert all(math.isclose(group['lr'], expected_lr, rel_tol=0, abs_tol=1e-18)
                       for group in ck['optimizer']['param_groups'])
            if code or update != total - 1:
                raise RuntimeError(f'training exited {code} at update {update}; expected {total - 1}')
            if mode == 'smoke':
                with (run / 'csv_logs/eval_metrics.csv').open() as handle:
                    rows = [row for row in csv.DictReader(handle) if row['scheduler'] == 'PPO']
                assert [int(row['update']) for row in rows] == list(range(total))
                assert all(math.isfinite(float(row['reward'])) for row in rows)
                if previous is not None:
                    for critic in (False, True):
                        assert any(not torch.equal(value, ck['model'][key])
                                   for key, value in previous.items()
                                   if key.endswith('weight') and key.startswith('value_head.') == critic)
                previous = ck['model']
            state.setdefault('phases', []).append(dict(target_updates=total, checkpoint_update=update,
                optimizer_lr=expected_lr, seconds=time.monotonic() - started, finite_weights=True))
            write_json(record, state)
        state.update(status='completed', target_completed=True, finished_at=now(),
                     resume_verified=(mode == 'smoke'), actual_model_update_verified=(mode == 'smoke'))
        write_json(record, state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=now(), target_completed=False)
        write_json(record, state)
        raise


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('mode', choices=NAMES)
    args.add_argument('--dry-run', action='store_true')
    args.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = args.parse_args()
    name = NAMES[args.mode]
    logs = ROOT / 'review/logs'
    record, console = logs / (name + '_launch.json'), logs / (name + '.log')
    if args.supervise:
        assert not args.dry_run
        return supervise(args.mode, record)
    preflight = logs / (name + '_preflight.json')
    pre = prepared(args.mode)
    if args.dry_run:
        write_json(preflight, pre, exclusive=True)
        print(json.dumps({'preflight': str(preflight), 'config': pre['config']}, indent=2))
        return
    assert json.loads(preflight.read_text()) == pre, 'Preflight changed; do not launch'
    if args.mode == 'main':
        smoke = json.loads((logs / (NAMES['smoke'] + '_launch.json')).read_text())
        assert smoke['status'] == 'completed' and smoke['target_completed']
        assert smoke['checkpoint_update'] == 1 and smoke['resume_verified']
        assert smoke['actual_model_update_verified'] and smoke['source_sha256'] == pre['source_sha256']
        assert smoke['reference_source_sha256'] == pre['reference_source_sha256']
    assert not record.exists() and not console.exists() and not Path(pre['run_dir']).exists()
    free_gpu4()
    state = dict(pre, status='starting', name=name, console_log=str(console), created_at=now(),
                 physical_gpu=4, supervisor_pid=None, training_pid=None, target_completed=False)
    state['target_updates'] = 2 if args.mode == 'smoke' else 1000
    write_json(record, state, exclusive=True)
    try:
        with console.open('xb') as output:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), args.mode,
                                        '--supervise'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=now())
        write_json(record, state)
        raise
    print(json.dumps(dict(dispatched=True, supervisor_pid=process.pid, launch_record=str(record),
                         console_log=str(console), run_dir=pre['run_dir']), indent=2))


if __name__ == '__main__':
    main()
