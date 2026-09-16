#!/usr/bin/env python3
"""Resume the three interrupted September 14 jobs, then run their queued seeds.

The original launcher and scientific sources remain frozen. Preparation archives
all interrupted run evidence; launch applies only validated CSV prefixes when
necessary. A recovery attempt has exclusive claims and cannot be relaunched.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import fcntl

FROZEN = Path(__file__).with_name('launch_seed_replicates_20260914.py').resolve()
_module = importlib.util.spec_from_file_location('frozen_seed_launcher', FROZEN)
l = importlib.util.module_from_spec(_module)
_module.loader.exec_module(l)
ROOT = l.ROOT
EXPECTED_UPDATES = {3: 888, 4: 948, 5: 907}
BASELINES = {'SUS+CQI', 'SUS+Deadline-PF', 'SUS+PF', 'SUS+Random', 'SUS+MW', 'SUS+EDF',
             'SU+CQI', 'SU+Deadline-PF', 'SU+PF', 'SU+Random', 'SU+MW', 'SU+EDF'}
CSV_NAMES = ('env_metrics.csv', 'ppo_metrics.csv', 'eval_metrics.csv', 'per_ue_metrics.csv')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def dead_pids(pids):
    for pid in pids:
        l.require(type(pid) is int and pid > 1, 'Invalid original PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        raise ValueError(f'Original PID {pid} still exists; refusing competing recovery')


def check_locks(campaign):
    for gpu in l.GPUS:
        with (campaign / f'gpu{gpu}.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def resume_command(prepared, job):
    return [prepared['python'], '-u', job['entry'], '--recipe', job['recipe'],
            '--resume', str(Path(job['run_dir']) / 'ckpt/latest.pt'),
            '--num-updates', str(job['target_updates'])]


def csv_prefix(data, watermark, label, cfg):
    """Preserve original complete row bytes; reject gaps below the checkpoint."""
    lines = data.splitlines(keepends=True)
    l.require(lines and lines[0].endswith(b'\n'), f'{label}: missing CSV header')
    header = next(csv.reader([lines[0].decode('utf-8')]))
    l.require(header and header[0] == 'update' and len(header) == len(set(header)), f'{label}: invalid header')
    kept = [lines[0]]
    rows = []
    removed = 0
    tail_started = False
    for i, line in enumerate(lines[1:], 1):
        try:
            fields = next(csv.reader([line.decode('utf-8')], strict=True))
            update = int(fields[0])
        except (ValueError, UnicodeDecodeError, csv.Error, IndexError):
            l.require(i == len(lines) - 1, f'{label}: malformed interior row')
            # A broken final row is allowed only after every required row, which
            # the exact coverage checks below independently establish.
            removed += 1
            tail_started = True
            continue
        complete = line.endswith(b'\n') and len(fields) == len(header)
        if update > watermark:
            l.require(i == len(lines) - 1 or complete, f'{label}: malformed nonfinal tail')
            removed += 1
            tail_started = True
            continue
        l.require(complete and not tail_started, f'{label}: incomplete or reordered committed row {update}')
        rows.append(dict(zip(header, fields)))
        kept.append(line)
    evaluation = list(range(cfg['ppo_eval_every'] - 1, watermark + 1, cfg['ppo_eval_every']))
    if label in ('env_metrics.csv', 'ppo_metrics.csv'):
        l.require([int(r['update']) for r in rows] == list(range(watermark + 1)), f'{label}: committed update gap/duplicate')
    elif label == 'eval_metrics.csv':
        l.require([int(r['update']) for r in rows if r['scheduler'] == 'PPO'] == evaluation, 'Incomplete PPO evaluation history')
        baseline = [r for r in rows if r['scheduler'] != 'PPO']
        l.require(len(baseline) == len(BASELINES) and {r['scheduler'] for r in baseline} == BASELINES,
                  'Baseline history is incomplete or duplicated')
        l.require(all(int(r['update']) == cfg['ppo_eval_every'] - 1 for r in baseline), 'Unexpected baseline update')
    else:
        l.require(label == 'per_ue_metrics.csv', 'Unknown CSV')
        expected = [(u, ep, ue) for u in evaluation for ep in range(cfg['ppo_eval_episodes']) for ue in range(cfg['num_ue'])]
        actual = [(int(r['update']), int(r['ep_idx']), int(r['ue'])) for r in rows]
        l.require(actual == expected, 'Incomplete per-UE evaluation history')
    if label in ('env_metrics.csv', 'eval_metrics.csv'):
        l.require(all(math.isfinite(float(r['reward'])) for r in rows), f'{label}: nonfinite reward')
    return b''.join(kept), {'retained_rows': len(rows), 'removed_rows': removed, 'last_committed_update': watermark}


def inspect_checkpoint(prepared, job):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, prepared['root'])
    import torch
    import paper_train as pt
    torch.set_num_threads(1)
    run = Path(job['run_dir'])
    expected = {k: v for k, v in job['expected_manifest'].items() if k not in ('run_dir', 'resume')}
    l.require(l.read(run / 'paper_manifest.json') == expected, 'Interrupted manifest changed')
    l.require(pt.normalize_config(l.read(run / 'config.json')) == expected['config'], 'Interrupted configuration changed')
    latest = torch.load(run / 'ckpt/latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(run / 'ckpt/best.pt', map_location='cpu', weights_only=False)
    u = latest['update']
    l.require(type(u) is int and u == EXPECTED_UPDATES[job['gpu']] and 0 <= u < job['target_updates'] - 1,
              f'Checkpoint watermark changed: {u}')
    l.require(pt.normalize_config(latest['cfg']) == expected['config'], 'Latest configuration mismatch')
    l.require(latest['lr_schedule'] == expected['schedule'], 'Latest LR schedule mismatch')
    l.require(int(latest['model']['ret_count']) == u + 1, 'Latest return normalization count mismatch')
    l.require(all(bool(torch.isfinite(v).all()) for v in latest['model'].values()), 'Nonfinite latest model')
    lr = expected['config']['ppo_learning_rate'] * (1 - u / expected['schedule']['lr_decay_updates'])
    l.require(all(math.isclose(float(g['lr']), lr, abs_tol=1e-15, rel_tol=0) for g in latest['optimizer']['param_groups']), 'Latest optimizer LR mismatch')
    l.require(bool(latest['optimizer']['state']), 'Adam state missing')
    for state in latest['optimizer']['state'].values():
        l.require({'step', 'exp_avg', 'exp_avg_sq'} <= set(state), 'Adam state incomplete')
        l.require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite Adam state')
    cpu_rng = latest['torch_rng_state']
    cuda_rng = latest['cuda_rng_state']
    l.require(torch.is_tensor(cpu_rng) and cpu_rng.dtype == torch.uint8 and cpu_rng.ndim == 1 and cpu_rng.numel() > 0, 'CPU RNG missing/invalid')
    l.require(isinstance(cuda_rng, list) and len(cuda_rng) == 1 and all(torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() > 0 for v in cuda_rng), 'CUDA RNG missing/invalid')
    original_rng = torch.get_rng_state()
    try:
        torch.set_rng_state(cpu_rng)
    finally:
        torch.set_rng_state(original_rng)
    grid = list(range(expected['config']['ppo_eval_every'] - 1, u + 1, expected['config']['ppo_eval_every']))
    l.verify_best(best, expected, grid, pt.normalize_config, torch)
    l.require(int(best['update']) <= u and math.isclose(float(best['eval_reward']), float(latest['best_eval_reward']), abs_tol=1e-12, rel_tol=1e-12),
              'Best checkpoint is inconsistent with latest selection history')
    return {'update': u, 'resume_update': u + 1, 'best_update': int(best['update']), 'optimizer_lr': lr,
            'finite_model_optimizer': True, 'rng_verified': True}


def source_files(prepared, active):
    campaign = Path(prepared['campaign_dir'])
    paths = {campaign / 'prepared.json', campaign / 'launch.json'}
    for gpu in l.GPUS:
        paths.update((campaign / f'gpu{gpu}_status.json', campaign / f'gpu{gpu}_supervisor.log'))
    for job in prepared['jobs']:
        folder = l.state_path(prepared, job).parent
        paths.update((l.state_path(prepared, job), ROOT / 'review/logs' / (job['name'] + '_launch.json'), folder / 'expected_manifest.json'))
    for job in active:
        run = Path(job['run_dir'])
        paths.update(run / 'csv_logs' / n for n in CSV_NAMES)
        paths.update((run / 'paper_manifest.json', run / 'config.json', run / 'ckpt/latest.pt', run / 'ckpt/best.pt'))
        paths.update((l.state_path(prepared, job).parent / 'console.log', l.state_path(prepared, job).parent / 'started.json'))
        paths.update((run / 'ckpt').glob('*.tmp'))
    for path in paths:
        l.require(path.is_file() and path.resolve() == path and path.is_relative_to(ROOT), f'Archive input missing/redirected: {path}')
    return sorted(paths)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        l.require(l.sha(path) == expected, f'Recovery input changed: {path}')


def prepare(prepared_path, recovery_dir):
    prepared_path = Path(prepared_path).resolve()
    prepared = l.load_prepared(prepared_path)
    l.verify_static(prepared)
    campaign = Path(prepared['campaign_dir'])
    recovery = l.inside(ROOT, recovery_dir)
    l.require(recovery.parent == campaign and not recovery.exists(), 'Recovery must be a new direct campaign subdirectory')
    check_locks(campaign)
    active = []
    pids = set()
    for gpu in l.GPUS:
        jobs = [j for j in prepared['jobs'] if j['gpu'] == gpu]
        l.require(len(jobs) == 2, 'Expected exactly two jobs per GPU')
        first, second = jobs
        state = l.read(l.state_path(prepared, first))
        l.require(state['status'] == 'running' and state['seed'] == 1002024, 'First job is not the interrupted replicate')
        pids.update((state['supervisor_pid'], state['training_pid']))
        l.require(l.read(l.state_path(prepared, second))['status'] == 'queued', 'Second replicate is not queued')
        l.require(not Path(second['run_dir']).exists() and not (l.state_path(prepared, second).parent / 'started.json').exists(), 'Second replicate already started')
        active.append(first)
    dead_pids(sorted(pids))
    for gpu in l.GPUS:
        l.free_gpu(gpu)
    paths = source_files(prepared, active)
    original_hashes = {str(p): l.sha(p) for p in paths}
    items = []
    candidates = {}
    for job in active:
        checkpoint = inspect_checkpoint(prepared, job)
        cmd = resume_command(prepared, job)
        result = subprocess.run(cmd + ['--dry-run'], cwd=ROOT, text=True, capture_output=True)
        l.require(result.returncode == 0, f'Resume dry-run failed:\n{result.stdout}\n{result.stderr}')
        actual = json.loads(result.stdout)
        expected = dict(job['expected_manifest'], resume=str(Path(job['run_dir']) / 'ckpt/latest.pt'))
        l.require(actual == expected, 'Resume dry-run changed configuration/source/runtime/schedule')
        csv_records = {}
        for name in CSV_NAMES:
            source = Path(job['run_dir']) / 'csv_logs' / name
            data = source.read_bytes()
            prefix, report = csv_prefix(data, checkpoint['update'], name, expected['config'])
            candidate = recovery / 'csv_prefixes' / job['name'] / name
            candidates[candidate] = prefix
            csv_records[str(source)] = {'candidate': str(candidate), 'original_sha256': digest(data),
                                        'prefix_sha256': digest(prefix), 'changed': prefix != data, **report}
        stable = {str(Path(job['run_dir']) / n): original_hashes[str(Path(job['run_dir']) / n)]
                  for n in ('paper_manifest.json', 'config.json', 'ckpt/latest.pt', 'ckpt/best.pt')}
        stable.update({path: r['prefix_sha256'] for path, r in csv_records.items()})
        items.append({'name': job['name'], 'gpu': job['gpu'], 'resume_command': cmd,
                      'checkpoint': checkpoint, 'csv': csv_records, 'resume_input_sha256': stable})
    verify_hashes(original_hashes)
    l.verify_static(prepared)
    dead_pids(sorted(pids))
    recovery.mkdir()
    archives = {}
    for source in paths:
        dest = recovery / 'archive' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        l.require(l.sha(dest) == original_hashes[str(source)], 'Archive copy verification failed')
        archives[str(dest)] = original_hashes[str(source)]
    for dest, data in candidates.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_bytes(dest, data)
    plan = {'version': 1, 'created_at': l.now(), 'prepared_path': str(prepared_path),
            'prepared_sha256': l.sha(prepared_path), 'recovery_dir': str(recovery), 'items': items,
            'original_pids': sorted(pids), 'original_sha256': original_hashes, 'archive_sha256': archives,
            'recovery_launcher': str(Path(__file__).resolve()), 'recovery_launcher_sha256': l.sha(__file__),
            'frozen_launcher_sha256': l.sha(FROZEN),
            'note': 'Container restarted; resume original checkpoints and LR horizons. CSV prefixes are authoritative; TensorBoard can contain uncommitted boundary events.'}
    l.atomic_json(recovery / 'plan.json', plan)
    atomic_bytes(recovery / 'plan.sha256', (l.sha(recovery / 'plan.json') + '\n').encode())
    print(json.dumps({'plan': str(recovery / 'plan.json'), 'jobs': [{'name': i['name'], **i['checkpoint']} for i in items],
                      'csv_changes': sum(r['changed'] for i in items for r in i['csv'].values())}, indent=2))


def load_plan(path, expected_sha=None):
    path = Path(path).resolve()
    sha = l.sha(path)
    l.require(sha == (expected_sha or path.with_name('plan.sha256').read_text().strip()), 'Recovery plan changed')
    plan = l.read(path)
    l.require(plan['version'] == 1 and path == Path(plan['recovery_dir']) / 'plan.json', 'Recovery plan location/version changed')
    l.require(plan['recovery_launcher'] == str(Path(__file__).resolve()) and plan['recovery_launcher_sha256'] == l.sha(__file__), 'Recovery launcher changed')
    l.require(plan['frozen_launcher_sha256'] == l.sha(FROZEN), 'Frozen launcher changed')
    l.require(plan['prepared_sha256'] == l.sha(plan['prepared_path']), 'Original prepared record changed')
    prepared = l.load_prepared(plan['prepared_path'])
    l.verify_static(prepared)
    return plan, prepared


def apply_csv_prefixes(plan):
    # Recheck all inputs before the first live mutation, not after each rewrite.
    verify_hashes(plan['original_sha256'])
    verify_hashes(plan['archive_sha256'])
    for item in plan['items']:
        for source, record in item['csv'].items():
            l.require(l.sha(record['candidate']) == record['prefix_sha256'], 'CSV prefix candidate changed')
    for item in plan['items']:
        for source, record in item['csv'].items():
            if record['changed']:
                atomic_bytes(source, Path(record['candidate']).read_bytes())


def write_queue(plan, prepared, state):
    gpu = state['gpu']
    l.atomic_json(Path(plan['recovery_dir']) / f'gpu{gpu}_status.json', state)
    l.atomic_json(Path(prepared['campaign_dir']) / f'gpu{gpu}_status.json', state)


def supervise(plan_path, plan_sha, gpu):
    plan, prepared = load_plan(plan_path, plan_sha)
    recovery = Path(plan['recovery_dir'])
    campaign = Path(prepared['campaign_dir'])
    jobs = [j for j in prepared['jobs'] if j['gpu'] == gpu]
    l.require(len(jobs) == 2 and gpu in l.GPUS, 'Invalid recovery GPU')
    item = next(i for i in plan['items'] if i['gpu'] == gpu)
    lock = (campaign / f'gpu{gpu}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    # Claim before rewriting any status, preventing a second invocation.
    try:
        l.exclusive_json(recovery / f'gpu{gpu}_started.json', {'pid': os.getpid(), 'at': l.now(), 'plan_sha256': plan_sha})
    except BaseException:
        lock.close()
        raise
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Recovery supervisor received {signum}')
    signal.signal(signal.SIGTERM, stop)
    queue = {'gpu': gpu, 'status': 'running', 'supervisor_pid': os.getpid(), 'started_at': l.now(),
             'recovery_plan': str(plan_path), 'completed_jobs': []}
    child = None
    active = None
    write_queue(plan, prepared, queue)
    try:
        for index, job in enumerate(jobs):
            active = job
            l.verify_static(prepared)
            l.require(l.sha(__file__) == plan['recovery_launcher_sha256'], 'Recovery launcher changed')
            l.free_gpu(gpu)
            folder = l.state_path(prepared, job).parent
            state = l.read(l.state_path(prepared, job))
            if index == 0:
                l.require(state['status'] == 'recovering', 'Interrupted job no longer awaiting this recovery')
                verify_hashes(item['resume_input_sha256'])
                cmd = item['resume_command']
                mode = 'ab'
            else:
                l.require(state['status'] == 'queued' and not Path(job['run_dir']).exists(), 'Queued replicate already started/changed')
                l.exclusive_json(folder / 'started.json', {'started_at': l.now(), 'supervisor_pid': os.getpid(), 'command': job['command'], 'recovery_plan': str(plan_path)})
                cmd = job['command']
                mode = 'xb'
            state.update(status='starting', supervisor_pid=os.getpid(), active_command=cmd,
                         recovery_plan=str(plan_path), target_completed=False, returncode=None)
            l.write_status(prepared, job, state)
            with (folder / 'console.log').open(mode) as output:
                if index == 0:
                    output.write(('\n[Recovery ' + l.now() + '] Resume from update ' + str(item['checkpoint']['resume_update']) + '; original LR schedule retained.\n').encode())
                    output.flush()
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, close_fds=True)
                state.update(status='running', training_pid=child.pid)
                state['restarted_at' if index == 0 else 'started_at'] = l.now()
                l.write_status(prepared, job, state)
                queue.update(active_job=job['name'], training_pid=child.pid)
                write_queue(plan, prepared, queue)
                last_report = 0.0
                while child.poll() is None:
                    l.link_outputs(prepared, job)
                    if time.monotonic() - last_report >= 30:
                        state.update(progress=l.progress(Path(job['run_dir'])), updated_at=l.now())
                        l.write_status(prepared, job, state)
                        last_report = time.monotonic()
                    time.sleep(2)
                code = child.returncode
            l.link_outputs(prepared, job)
            state.update(status='verifying', returncode=code, updated_at=l.now())
            l.write_status(prepared, job, state)
            result = l.verify_completion(prepared, job, code)
            state.update(status='completed', target_completed=True, checkpoint_update=result['checkpoint_update'],
                         finished_at=l.now(), verification=result, progress=l.progress(Path(job['run_dir'])))
            l.write_status(prepared, job, state)
            queue['completed_jobs'].append(job['name'])
            queue.update(active_job=None, training_pid=None)
            write_queue(plan, prepared, queue)
            child = None
            active = None
        queue.update(status='completed', finished_at=l.now())
        write_queue(plan, prepared, queue)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if active:
            state = l.read(l.state_path(prepared, active))
            state.update(status='failed', target_completed=False, error=repr(exc), finished_at=l.now())
            l.write_status(prepared, active, state)
        l.mark_blocked(prepared, jobs, 'Recovery stopped: ' + repr(exc))
        queue.update(status='failed', error=repr(exc), finished_at=l.now())
        write_queue(plan, prepared, queue)
        raise
    finally:
        lock.close()


def mark_undispatched(prepared, dispatched, error):
    for gpu in l.GPUS:
        if str(gpu) in dispatched:
            continue
        for index, job in enumerate(j for j in prepared['jobs'] if j['gpu'] == gpu):
            current = l.read(l.state_path(prepared, job))
            current.update(status='failed' if index == 0 else 'blocked', target_completed=False,
                           error='Recovery supervisor was not dispatched: ' + repr(error), finished_at=l.now())
            l.write_status(prepared, job, current)


def launch(plan_path):
    plan_path = Path(plan_path).resolve()
    plan, prepared = load_plan(plan_path)
    recovery = Path(plan['recovery_dir'])
    check_locks(Path(prepared['campaign_dir']))
    dead_pids(plan['original_pids'])
    for gpu in l.GPUS:
        l.free_gpu(gpu)
    verify_hashes(plan['original_sha256'])
    verify_hashes(plan['archive_sha256'])
    record = recovery / 'launch.json'
    state = {'status': 'preparing_dispatch', 'started_at': l.now(), 'plan_sha256': l.sha(plan_path), 'supervisors': {}}
    l.exclusive_json(record, state)
    try:
        apply_csv_prefixes(plan)
        for item in plan['items']:
            job = next(j for j in prepared['jobs'] if j['name'] == item['name'])
            current = l.read(l.state_path(prepared, job))
            current.update(status='recovering', recovery_plan=str(plan_path), resume_update=item['checkpoint']['resume_update'],
                           recovery_started_at=l.now(), target_completed=False)
            l.write_status(prepared, job, current)
        for gpu in l.GPUS:
            with (recovery / f'gpu{gpu}_supervisor.log').open('xb') as output:
                proc = subprocess.Popen([prepared['python'], '-u', str(Path(__file__).resolve()), 'supervise',
                                         '--plan', str(plan_path), '--plan-sha', state['plan_sha256'], '--gpu', str(gpu)],
                                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                        start_new_session=True, close_fds=True)
            state['supervisors'][str(gpu)] = proc.pid
            l.atomic_json(record, state)
        state.update(status='dispatched', dispatched_at=l.now())
        l.atomic_json(record, state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=l.now())
        l.atomic_json(record, state)
        mark_undispatched(prepared, state['supervisors'], exc)
        raise
    print(json.dumps({'launch_record': str(record), **state}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--prepared', type=Path, required=True)
    q.add_argument('--recovery-dir', type=Path, required=True)
    sub.add_parser('launch').add_argument('--plan', type=Path, required=True)
    q = sub.add_parser('supervise')
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--plan-sha', required=True)
    q.add_argument('--gpu', type=int, choices=l.GPUS, required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.prepared, args.recovery_dir)
    elif args.action == 'launch':
        launch(args.plan)
    else:
        supervise(args.plan, args.plan_sha, args.gpu)


if __name__ == '__main__':
    main()
