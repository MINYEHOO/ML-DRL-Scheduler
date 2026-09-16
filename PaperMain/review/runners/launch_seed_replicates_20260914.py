#!/usr/bin/env python3
"""Additive, fail-stop queues for six independent PaperMain training replicates.

prepare --spec FILE performs six protected trainer dry-runs without making run
folders. launch --prepared FILE starts one detached supervisor on each allowed
GPU. Existing training code, checkpoints and reference results are read-only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
GPUS = (3, 4, 5)
SEEDS = (1002024, 2002024)
EXPECTED = {3: ('base', 'paper_train.py', 1000),
            4: ('base', 'paper_train.py', 2000),
            5: ('narrow_mean', 'paper_train_variants.py', 2000)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def exclusive_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def inside(root, value, prefix=None):
    p = Path(value)
    p = p if p.is_absolute() else root / p
    resolved = p.resolve()
    require(resolved.is_relative_to(root) and resolved != root,
            f'Path escapes PaperMain: {value}')
    require(p.absolute() == resolved, f'Symlink or noncanonical path: {value}')
    if prefix:
        require(resolved.is_relative_to(root / prefix), f'Path must be inside {prefix}: {value}')
    return resolved


def validate_spec(spec, root=ROOT):
    require(Path(spec.get('root', str(root))).resolve() == root, 'Unexpected PaperMain root')
    campaign = inside(root, spec['campaign_dir'], 'results')
    require(not campaign.exists(), f'Campaign already exists: {campaign}')
    jobs = spec['jobs']
    require(isinstance(jobs, list) and len(jobs) == 6, 'Exactly six jobs required')
    names = [j['name'] for j in jobs]
    require(len(set(names)) == 6, 'Run names must be unique')
    require(all(re.fullmatch(r'[0-9]+_[A-Za-z0-9_.-]+', n) for n in names), 'Numbered safe run names required')
    numbers = [int(n.split('_', 1)[0]) for n in names]
    require(len(set(numbers)) == 6, 'Run number prefixes must be unique')
    for j in jobs:
        gpu = j['gpu']
        require(type(gpu) is int and gpu in GPUS, f'GPU not authorized: {gpu}')
        require((j['recipe'], j['entry'], j['target_updates']) == EXPECTED[gpu], f'Wrong GPU recipe/horizon: {j}')
        require(type(j['seed']) is int and j['seed'] in SEEDS, 'Only reviewed independent seeds allowed')
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', j['reference_run']) is not None, 'Invalid reference run')
        run = inside(root, 'runs/' + j['name'], 'runs')
        require(not run.exists() and not run.is_symlink(), f'Run already exists: {run}')
        ref = inside(root, 'runs/' + j['reference_run'], 'runs')
        require(ref.is_dir(), f'Reference missing: {ref}')
        inside(root, j['calibration_profile'], 'results')
        inside(root, j['calibration_reference_root'], 'provenance')
    for gpu in GPUS:
        require([j['seed'] for j in jobs if j['gpu'] == gpu] == list(SEEDS),
                f'GPU {gpu} must queue seeds {SEEDS} in that order')
    return campaign


def command(job, python=sys.executable):
    return [python, '-u', job['entry'], '--recipe', job['recipe'], '--name', job['name'],
            '--device', 'cuda', '--gpu', str(job['gpu']), '--threads', '4',
            '--seed', str(job['seed']), '--traffic-model', 'bernoulli', '--replay-mode', 'batched',
            '--num-updates', str(job['target_updates']), '--lr-final', '0',
            '--lr-decay-updates', str(job['target_updates']),
            '--calibration-profile', job['calibration_profile'],
            '--calibration-reference-root', job['calibration_reference_root']]


def free_gpu(gpu):
    require(gpu in GPUS, 'Unauthorized GPU')
    def query(kind, fields):
        return subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader,nounits'], text=True).strip()
    require(not query('compute-apps', 'pid'), f'GPU {gpu} has another compute process; nothing killed')
    index, used = query('gpu', 'index,memory.used').split(',')
    require(int(index) == gpu and int(used) < 500, f'GPU {gpu} is not idle: {used} MiB')


def compare_manifest(actual, reference, job, root, normalize):
    expected = normalize(reference['config'])
    expected['seed'] = job['seed']
    require(actual['config'] == expected, f'{job["name"]}: config differs beyond seed')
    require(expected['traffic_model'] == 'bernoulli' and expected['ppo_batched_replay'] is True,
            'Bernoulli batched replay required')
    require(expected['ppo_learning_rate'] == 0.0003, 'Initial LR must remain 3e-4')
    require(actual['schedule'] == {'lr_final': 0.0, 'lr_decay_updates': job['target_updates']}, 'Wrong LR horizon')
    require(actual['target_updates'] == job['target_updates'], 'Wrong update target')
    require(actual['execution'] == {'device': 'cuda', 'gpu': str(job['gpu']), 'threads': 4}, 'Wrong execution settings')
    require(actual['resume'] is None and actual['run_dir'] == str(root / 'runs' / job['name']), 'Fresh run required')
    require(actual['recipe'] == job['recipe'] and actual['purpose'] == 'training' and actual['smoke_slots'] is None,
            'Wrong recipe or smoke mode')
    for key in ('source_sha256', 'artifact_config_sha256', 'calibration_profile',
                'calibration_reference_root', 'calibration_application', 'runtime', 'variant_provenance'):
        require(actual.get(key) == reference.get(key), f'{job["name"]}: reference {key} differs')
    return expected


def dependency_paths(root, jobs, manifests, pt):
    from calibration.profile import source_hashes as science_hashes
    paths = {root / name for name in pt.source_hashes(root)}
    paths.add(Path(__file__).resolve())
    paths.update(root / name for name in science_hashes(root))
    for j, m in zip(jobs, manifests):
        paths.add(root / j['entry'])
        paths.add(root / 'artifacts' / j['recipe'] / 'config.json')
        paths.add(root / 'artifacts/base/config.json')
        paths.add(inside(root, j['calibration_profile']))
        frozen = inside(root, j['calibration_reference_root'])
        paths.update(frozen / name for name in science_hashes(frozen))
        paths.add(frozen / 'artifacts/base/config.json')
        ref = root / 'runs' / j['reference_run']
        paths.update(ref / name for name in ('paper_manifest.json', 'config.json', 'ckpt/latest.pt'))
    return sorted(paths)


def verify_static(prepared):
    for name, digest in prepared['protected_sha256'].items():
        require(sha(name) == digest, f'Protected input changed: {name}')


def prepare(spec_path):
    spec = read(spec_path)
    root = ROOT.resolve()
    require(root == ROOT, 'PaperMain root must not be a symlink')
    campaign = validate_spec(spec, root)
    sys.path.insert(0, str(root))
    import paper_train as pt
    manifests = []
    jobs = []
    conventional = root / 'review/logs'
    require(conventional.resolve() == conventional, 'Conventional log directory is redirected')
    for j in spec['jobs']:
        for suffix in ('_launch.json', '.log'):
            p = conventional / (j['name'] + suffix)
            require(not p.exists() and not p.is_symlink(), f'Conventional log already exists: {p}')
    for j in spec['jobs']:
        free_gpu(j['gpu'])
        c = command(j)
        result = subprocess.run(c + ['--dry-run'], cwd=root, text=True, capture_output=True)
        require(result.returncode == 0, f'Dry run failed for {j["name"]}:\n{result.stdout}\n{result.stderr}')
        actual = json.loads(result.stdout)
        reference = read(root / 'runs' / j['reference_run'] / 'paper_manifest.json')
        compare_manifest(actual, reference, j, root, pt.normalize_config)
        require(pt.normalize_config(read(root / 'runs' / j['reference_run'] / 'config.json')) == pt.normalize_config(reference['config']),
                'Reference config.json differs from manifest')
        require(actual['source_sha256'] == pt.source_hashes(root), 'Current source differs from manifest')
        manifests.append(actual)
        jobs.append({**j, 'command': c, 'run_dir': actual['run_dir'], 'expected_manifest': actual})
    protected = {str(p): sha(p) for p in dependency_paths(root, jobs, manifests, pt)}
    prepared = {'schema_version': 1, 'created_at': now(), 'root': str(root),
                'campaign_dir': str(campaign), 'spec': spec, 'jobs': jobs,
                'protected_sha256': protected, 'launcher': str(Path(__file__).resolve()),
                'python': sys.executable, 'independent_training_replicates': True,
                'paired_recipe_seed_assignment': True,
                'stage1_note': 'GPU3 performs only the initial 1000-update segment of Run16 reproduction; low-LR continuation is not scheduled.'}
    # Last read-only check before creating this entirely new campaign directory.
    validate_spec(spec, root)
    verify_static(prepared)
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / 'jobs').mkdir()
    conventional.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        folder = campaign / 'jobs' / j['name']
        folder.mkdir()
        atomic_json(folder / 'expected_manifest.json', j['expected_manifest'])
        initial = {'name': j['name'], 'gpu': j['gpu'], 'physical_gpu': j['gpu'], 'seed': j['seed'],
                   'status': 'prepared', 'target_updates': j['target_updates'], 'run_dir': j['run_dir'],
                   'command': j['command'], 'target_completed': False, 'returncode': None, 'checkpoint_update': None}
        exclusive_json(conventional / (j['name'] + '_launch.json'), initial)
        atomic_json(folder / 'status.json', initial)
        (conventional / (j['name'] + '.log')).symlink_to(os.path.relpath(folder / 'console.log', conventional))
    atomic_json(campaign / 'prepared.json', prepared)
    print(json.dumps({'prepared': str(campaign / 'prepared.json'), 'jobs': [{k: j[k] for k in ('name', 'gpu', 'seed', 'target_updates')} for j in jobs]}, indent=2))


def load_prepared(path):
    prepared = read(path)
    require(prepared['root'] == str(ROOT) and prepared['schema_version'] == 1, 'Invalid prepared root/version')
    campaign = inside(ROOT, prepared['campaign_dir'], 'results')
    require(Path(path).resolve() == campaign / 'prepared.json', 'Prepared record must remain at its campaign path')
    require(prepared['launcher'] == str(Path(__file__).resolve()), 'Launcher path changed')
    require(len(prepared['jobs']) == 6, 'Prepared job count changed')
    for j, original in zip(prepared['jobs'], prepared['spec']['jobs']):
        require(all(j[k] == v for k, v in original.items()), 'Prepared job differs from spec')
        require(j['command'] == command(original, prepared['python']), 'Prepared command changed')
        require(j['run_dir'] == str(ROOT / 'runs' / j['name']), 'Prepared run path changed')
    return prepared


def state_path(prepared, job):
    return Path(prepared['campaign_dir']) / 'jobs' / job['name'] / 'status.json'


def write_status(prepared, job, data):
    # OOD provenance insists that this conventional path be a regular file.
    mirror = Path(prepared['root']) / 'review/logs' / (job['name'] + '_launch.json')
    require(not mirror.is_symlink(), 'Conventional launch record was redirected')
    atomic_json(mirror, data)
    atomic_json(state_path(prepared, job), data)


def link_outputs(prepared, job):
    run = Path(job['run_dir'])
    if not run.is_dir():
        return
    require(not run.is_symlink(), 'Run folder was replaced with a symlink')
    folder = state_path(prepared, job).parent
    links = {'console.log': folder / 'console.log', 'status.json': folder / 'status.json',
             'eval.csv': run / 'csv_logs/eval_metrics.csv', 'train.csv': run / 'csv_logs/env_metrics.csv',
             'ppo.csv': run / 'csv_logs/ppo_metrics.csv', 'launch.json': folder / 'status.json'}
    for name, target in links.items():
        link = run / name
        if link.is_symlink():
            require(link.resolve() == target.resolve(), f'Unexpected existing link: {link}')
        else:
            require(not link.exists(), f'Refusing to overwrite {link}')
            link.symlink_to(os.path.relpath(target, run))
    guide = run / 'RUN_FILES.md'
    if not guide.exists():
        with guide.open('x') as f:
            f.write('# Run files\n\n'
                    '- `eval.csv`: PPO and baseline evaluation metrics.\n'
                    '- `train.csv`: training episode metrics.\n'
                    '- `ppo.csv`: optimizer and policy metrics.\n'
                    '- `console.log`: live trainer output.\n'
                    '- `status.json` / `launch.json`: queue status, progress and completion verification.\n'
                    '- `paper_manifest.json`: exact training configuration and source provenance.\n'
                    '- `ckpt/latest.pt`: latest resumable checkpoint; `ckpt/best.pt`: selected validation checkpoint.\n\n'
                    + ('This is the initial 1000-update segment only; the low-LR continuation is not scheduled.\n'
                       if job['gpu'] == 3 else 'Fresh 2000-update training with linear LR decay.\n'))


def progress(run):
    result = {}
    for label, filename in [('train', 'env_metrics.csv'), ('eval', 'eval_metrics.csv')]:
        path = run / 'csv_logs' / filename
        if not path.exists():
            continue
        try:
            with path.open(newline='') as f:
                rows = csv.DictReader(f)
                last = None
                for row in rows:
                    if label != 'eval' or row.get('scheduler') == 'PPO':
                        last = row
            if last and last.get('update', '').isdigit():
                result[label + '_last_update'] = int(last['update'])
        except (OSError, csv.Error):
            pass
    return result


def verify_completion(prepared, job, returncode):
    require(returncode == 0, f'Trainer exited with {returncode}')
    root = Path(prepared['root'])
    run = Path(job['run_dir'])
    require(not run.is_symlink() and run.resolve().parent == root / 'runs', 'Invalid completed run path')
    expected = {k: v for k, v in job['expected_manifest'].items() if k not in ('resume', 'run_dir')}
    require(read(run / 'paper_manifest.json') == expected, 'Saved manifest differs from protected dry run')
    # The supervisor never owns a CUDA context or imports the simulator.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(root))
    import torch
    import paper_train as pt
    torch.set_num_threads(1)
    ck = torch.load(run / 'ckpt/latest.pt', map_location='cpu', weights_only=False)
    target = job['target_updates']
    require(int(ck['update']) == target - 1, f'Incomplete checkpoint: {ck["update"]}')
    require(pt.normalize_config(ck['cfg']) == expected['config'], 'Checkpoint configuration mismatch')
    require(pt.normalize_config(read(run / 'config.json')) == expected['config'], 'Saved configuration mismatch')
    require(ck['lr_schedule'] == expected['schedule'], 'Checkpoint LR schedule mismatch')
    require(all(bool(torch.isfinite(v).all()) for v in ck['model'].values()), 'Nonfinite model state')
    require(int(ck['model']['ret_count']) == target, 'Return normalization count differs from completed updates')
    lr = expected['config']['ppo_learning_rate'] * (1.0 - (target - 1) / target)
    require(all(math.isclose(float(g['lr']), lr, rel_tol=0, abs_tol=1e-15) for g in ck['optimizer']['param_groups']), 'Final optimizer LR mismatch')
    for state in ck['optimizer']['state'].values():
        require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite optimizer state')
    rows = list(csv.DictReader((run / 'csv_logs/env_metrics.csv').open()))
    require([int(r['update']) for r in rows] == list(range(target)), 'Training CSV update sequence incomplete')
    require(all(math.isfinite(float(r['reward'])) for r in rows), 'Nonfinite training rewards')
    eval_rows = list(csv.DictReader((run / 'csv_logs/eval_metrics.csv').open()))
    require(any(r['scheduler'] != 'PPO' for r in eval_rows), 'Fresh-run baseline evaluation missing')
    expected_eval_updates = list(range(expected['config']['ppo_eval_every'] - 1, target, expected['config']['ppo_eval_every']))
    require([int(r['update']) for r in eval_rows if r['scheduler'] == 'PPO'] == expected_eval_updates,
            'PPO evaluation update coverage is incomplete')
    require(all(math.isfinite(float(r['reward'])) for r in eval_rows), 'Nonfinite evaluation rewards')
    best = torch.load(run / 'ckpt/best.pt', map_location='cpu', weights_only=False)
    verify_best(best, expected, expected_eval_updates, pt.normalize_config, torch)
    verify_static(prepared)
    return {'checkpoint_update': target - 1, 'completed_updates': target, 'optimizer_lr': lr,
            'finite_weights': True, 'finite_optimizer': True, 'baseline_eval_verified': True,
            'checkpoint_sha256': sha(run / 'ckpt/latest.pt'), 'best_checkpoint_update': int(best['update']),
            'best_checkpoint_sha256': sha(run / 'ckpt/best.pt'), 'best_checkpoint_verified': True}


def verify_best(best, expected, eval_updates, normalize, torch):
    require(int(best['update']) in eval_updates, 'Best checkpoint is outside evaluation update grid')
    require(normalize(best['cfg']) == expected['config'], 'Best checkpoint config mismatch')
    require(best['lr_schedule'] == expected['schedule'], 'Best checkpoint LR schedule mismatch')
    require(all(bool(torch.isfinite(v).all()) for v in best['model'].values()), 'Nonfinite best checkpoint')
    require(math.isfinite(float(best['eval_reward'])), 'Nonfinite best evaluation reward')
    require(int(best['model']['ret_count']) == int(best['update']) + 1, 'Best return normalization count mismatch')


def mark_blocked(prepared, jobs, reason):
    for j in jobs:
        state = read(state_path(prepared, j))
        if state['status'] in ('prepared', 'queued'):
            state.update(status='blocked', error=reason, finished_at=now())
            write_status(prepared, j, state)


def supervise(prepared_path, gpu):
    prepared = load_prepared(prepared_path)
    require(gpu in GPUS, 'Unauthorized supervisor GPU')
    campaign = Path(prepared['campaign_dir'])
    jobs = [j for j in prepared['jobs'] if j['gpu'] == gpu]
    queue_path = campaign / f'gpu{gpu}_status.json'
    require(all(read(state_path(prepared, j))['status'] == 'queued' for j in jobs),
            'Queue is not fresh; refusing to rewrite completed or interrupted status')
    lock = (campaign / f'gpu{gpu}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    child = None
    active = None
    queue = {'gpu': gpu, 'status': 'running', 'supervisor_pid': os.getpid(), 'started_at': now(), 'completed_jobs': []}
    atomic_json(queue_path, queue)
    try:
        for index, j in enumerate(jobs):
            active = j
            verify_static(prepared)
            require(read(state_path(prepared, j))['status'] == 'queued', 'Job is not in queued state; refusing restart')
            require(not Path(j['run_dir']).exists() and not Path(j['run_dir']).is_symlink(), 'Run already exists; never overwrite or auto-resume')
            free_gpu(gpu)
            folder = state_path(prepared, j).parent
            exclusive_json(folder / 'started.json', {'started_at': now(), 'supervisor_pid': os.getpid(), 'command': j['command']})
            state = read(state_path(prepared, j))
            state.update(status='starting', started_at=now(), supervisor_pid=os.getpid(), command=j['command'])
            write_status(prepared, j, state)
            with (folder / 'console.log').open('xb') as output:
                child = subprocess.Popen(j['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, close_fds=True)
                state.update(status='running', training_pid=child.pid)
                write_status(prepared, j, state)
                queue.update(active_job=j['name'], training_pid=child.pid)
                atomic_json(queue_path, queue)
                last_report = 0.0
                while child.poll() is None:
                    link_outputs(prepared, j)
                    if time.monotonic() - last_report >= 30:
                        state.update(progress=progress(Path(j['run_dir'])), updated_at=now())
                        write_status(prepared, j, state)
                        last_report = time.monotonic()
                    time.sleep(2)
                returncode = child.returncode
            link_outputs(prepared, j)
            state.update(status='verifying', returncode=returncode, updated_at=now())
            write_status(prepared, j, state)
            result = verify_completion(prepared, j, returncode)
            state.update(status='completed', target_completed=True, checkpoint_update=result['checkpoint_update'],
                         finished_at=now(), verification=result,
                         progress=progress(Path(j['run_dir'])))
            write_status(prepared, j, state)
            queue['completed_jobs'].append(j['name'])
            queue.update(active_job=None, training_pid=None)
            atomic_json(queue_path, queue)
            child = None
            active = None
        queue.update(status='completed', finished_at=now())
        atomic_json(queue_path, queue)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if active is not None:
            state = read(state_path(prepared, active))
            state.update(status='failed', target_completed=False, error=repr(exc), finished_at=now())
            write_status(prepared, active, state)
        mark_blocked(prepared, jobs, 'Queue stopped after a preceding failure: ' + repr(exc))
        queue.update(status='failed', error=repr(exc), finished_at=now())
        atomic_json(queue_path, queue)
        raise
    finally:
        lock.close()


def launch(prepared_path):
    prepared = load_prepared(prepared_path)
    verify_static(prepared)
    campaign = Path(prepared['campaign_dir'])
    for gpu in GPUS:
        free_gpu(gpu)
    for j in prepared['jobs']:
        require(not Path(j['run_dir']).exists() and not Path(j['run_dir']).is_symlink(), 'Run exists; refusing launch')
        require(read(state_path(prepared, j))['status'] == 'prepared', 'Job already launched or changed')
    record = campaign / 'launch.json'
    state = {'status': 'dispatching', 'started_at': now(), 'supervisors': {}}
    exclusive_json(record, state)
    for j in prepared['jobs']:
        s = read(state_path(prepared, j))
        s.update(status='queued', queued_at=now())
        write_status(prepared, j, s)
    try:
        for gpu in GPUS:
            with (campaign / f'gpu{gpu}_supervisor.log').open('xb') as f:
                proc = subprocess.Popen([prepared['python'], '-u', prepared['launcher'], 'supervise',
                                         '--prepared', str(Path(prepared_path).resolve()), '--gpu', str(gpu)],
                                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                        start_new_session=True, close_fds=True)
            state['supervisors'][str(gpu)] = proc.pid
            atomic_json(record, state)
        state.update(status='dispatched', dispatched_at=now())
        atomic_json(record, state)
    except BaseException as exc:
        state.update(status='dispatch_failed', error=repr(exc), finished_at=now())
        atomic_json(record, state)
        for gpu in GPUS:
            if str(gpu) not in state['supervisors']:
                mark_blocked(prepared, [j for j in prepared['jobs'] if j['gpu'] == gpu], 'Supervisor dispatch failed')
        raise
    print(json.dumps({'launch_record': str(record), **state}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare').add_argument('--spec', type=Path, required=True)
    sub.add_parser('launch').add_argument('--prepared', type=Path, required=True)
    supervise_parser = sub.add_parser('supervise')
    supervise_parser.add_argument('--prepared', type=Path, required=True)
    supervise_parser.add_argument('--gpu', type=int, choices=GPUS, required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.spec)
    elif args.action == 'launch':
        launch(args.prepared)
    else:
        supervise(args.prepared, args.gpu)


if __name__ == '__main__':
    main()
