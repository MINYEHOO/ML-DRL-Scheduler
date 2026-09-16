#!/usr/bin/env python3
"""Reward-weight recipe runner (lambda_c 2, lambda_m 6); GPU 5.

Mirrors review/runners/run_raw_invariant.py. GPU 5 is freed by pausing GPT's
run 28 (narrow_mean seed 2002024) at a checkpoint boundary; it is resumed
natively after this run completes (see review/logs/28_pause_20260916.md).

    python review/runners/run_reward_c2m6.py smoke --preflight
    python review/runners/run_reward_c2m6.py smoke
    python review/runners/run_reward_c2m6.py train --preflight
    python review/runners/run_reward_c2m6.py train
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, os, signal, subprocess, sys, time
ROOT = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
sys.path.insert(0, str(ROOT))
import paper_train as pt
import paper_train_reward as pw
NAMES = {'smoke': '31_audit_reward_c2m6_gpu5',
         'train': '32_reward_c2m6_lr1000_s2024_20260916_gpu5'}
GPU = 5
UPDATES = {'smoke': 1, 'train': 1000}
SCHEDULE = {'lr_final': 0.0, 'lr_decay_updates': 1000}
COMPARISON = '26_base_stage1_lr1000_s2002024_20260914_gpu3'
LOGS = ROOT / 'review/logs'
EXTENSIONS = ['paper_train_reward.py', 'artifacts/reward_c2m6/config.json',
              'review/runners/run_reward_c2m6.py']


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path, data, exclusive=False):
    if exclusive:
        with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')
    else: pt.atomic_json(path, data)
def free_gpu():
    def query(kind, fields):
        return subprocess.check_output(['nvidia-smi', '-i', str(GPU), '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader,nounits'], text=True).strip()
    assert not query('compute-apps', 'pid'), f'GPU {GPU} occupied'
    index, used = query('gpu', 'index,memory.used').split(',')
    assert int(index) == GPU and int(used) < 500, (GPU, used)
def command(mode):
    c = [sys.executable, '-u', 'paper_train_reward.py', '--recipe', pw.RECIPE, '--name', NAMES[mode],
         '--device', 'cuda', '--gpu', str(GPU), '--threads', '4', '--seed', '2024',
         '--traffic-model', 'bernoulli', '--replay-mode', 'batched',
         '--num-updates', str(UPDATES[mode]), '--lr-final', '0', '--lr-decay-updates', '1000',
         '--calibration-profile', 'results/cqi4_hl_corrected_v2/summary.json',
         '--calibration-reference-root', 'provenance/before_ftp3_sources']
    if mode == 'smoke': c += ['--smoke-slots', '32']
    return c


def prepare(mode):
    c = command(mode)
    run, checkpoint, m = pw.plan(pw.parser().parse_args(c[3:]), ROOT)
    ref = json.loads((ROOT / 'runs' / COMPARISON / 'paper_manifest.json').read_text())
    assert m['source_sha256'] == pt.source_hashes(ROOT) == ref['source_sha256'], 'core sources must be untouched'
    expected = dict(ref['config'], seed=2024, **pw.NEW_WEIGHTS)
    if mode == 'smoke':
        expected.update(episode_len_main=32, episode_len_debug=32, ppo_epochs=1, ppo_minibatch_size=24,
                        ppo_eval_every=1, ppo_eval_episodes=1, ppo_save_every=1)
    assert m['config'] == expected, {k: (expected[k], m['config'].get(k)) for k in expected if expected[k] != m['config'].get(k)}
    assert checkpoint is None and m['schedule'] == SCHEDULE and m['recipe'] == pw.RECIPE
    assert m['execution'] == dict(device='cuda', gpu=str(GPU), threads=4)
    assert m['calibration_profile'] == ref['calibration_profile']
    assert m['target_updates'] == UPDATES[mode]
    assert m['purpose'] == ('execution_smoke' if mode == 'smoke' else 'training')
    return dict(command=c, run_dir=str(run), resume=None, **m,
                extension_sha256={p: sha(ROOT / p) for p in EXTENSIONS},
                comparison_run=COMPARISON,
                training_initialization='fresh model, optimizer and return normalization; seed 2024',
                same_world_as_base=True, reward_weights=pw.NEW_WEIGHTS)


def verify_static(state):
    assert pt.source_hashes(ROOT) == state['source_sha256']
    for name, digest in state['extension_sha256'].items(): assert sha(ROOT / name) == digest, name


def supervise(mode):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    record = LOGS / (NAMES[mode] + '_launch.json'); state = json.loads(record.read_text())
    run = Path(state['run_dir']); child = None
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3')
    try:
        import torch
        torch.set_num_threads(1)
        phases = [(state['command'], UPDATES[mode])]
        if mode == 'smoke':
            phases.append(([sys.executable, '-u', 'paper_train_reward.py', '--recipe', pw.RECIPE,
                            '--resume', str(run / 'ckpt/latest.pt'), '--num-updates', '2'], 2))
        previous = None
        for c, total in phases:
            verify_static(state); free_gpu(); started = time.monotonic()
            child = subprocess.Popen(c, cwd=ROOT, stdin=subprocess.DEVNULL)
            state.update(status='running', supervisor_pid=os.getpid(), training_pid=child.pid,
                         phase_target_updates=total, active_command=c)
            write(record, state)
            code = child.wait(); state['returncode'] = code
            ck = torch.load(run / 'ckpt/latest.pt', map_location='cpu', weights_only=False)
            u = int(ck['update']); state['checkpoint_update'] = u
            assert code == 0 and u == total - 1, (code, u, total)
            assert pt.normalize_config(ck['cfg']) == state['config'] and ck['lr_schedule'] == SCHEDULE
            assert ck['cfg']['lambda_c'] == 2.0 and ck['cfg']['lambda_m'] == 6.0
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
            lr = .0003 * (1 - u / 1000)
            assert all(math.isclose(g['lr'], lr, rel_tol=0, abs_tol=1e-18) for g in ck['optimizer']['param_groups'])
            assert int(ck['model']['ret_count']) == total
            if mode == 'smoke':
                rows = [r for r in csv.DictReader((run / 'csv_logs/eval_metrics.csv').open()) if r['scheduler'] == 'PPO']
                assert [int(r['update']) for r in rows] == list(range(total))
                assert all(math.isfinite(float(r['reward'])) for r in rows)
                if previous is not None:
                    for critic in (False, True):
                        assert any(not torch.equal(v, ck['model'][k]) for k, v in previous.items()
                                   if k.endswith('weight') and k.startswith('value_head.') == critic)
                previous = ck['model']
            state.setdefault('phases', []).append(dict(target_updates=total, checkpoint_update=u, optimizer_lr=lr,
                                                       seconds=time.monotonic() - started, finite_weights=True))
            write(record, state)
        verify_static(state)
        state.update(status='completed', target_completed=True, finished_at=now(),
                     resume_verified=mode == 'smoke', actual_model_update_verified=mode == 'smoke')
        write(record, state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        state.update(status='failed', target_completed=False, error=repr(exc), finished_at=now()); write(record, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=NAMES)
    p.add_argument('--preflight', action='store_true'); p.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(); mode = args.mode
    if args.supervise:
        assert not args.preflight
        return supervise(mode)
    pre = prepare(mode); path = LOGS / (NAMES[mode] + '_preflight.json')
    if args.preflight:
        free_gpu()
        result = subprocess.run(pre['command'] + ['--dry-run'], cwd=ROOT, text=True, capture_output=True)
        if result.returncode: raise RuntimeError(result.stdout + '\n' + result.stderr)
        out = result.stdout; actual = json.loads(out[out.index('{'):])
        for key in ('config', 'schedule', 'source_sha256', 'execution', 'target_updates', 'calibration_profile', 'variant_provenance'):
            assert actual[key] == pre[key], key
        assert actual['resume'] is None
        pre['runtime'] = actual['runtime']
        write(path, pre, exclusive=True)
        print(json.dumps(dict(preflight=str(path), mode=mode, gpu=GPU, fresh=True, target_updates=pre['target_updates']), indent=2)); return
    saved = json.loads(path.read_text()); pre['runtime'] = saved['runtime']; assert saved == pre, 'preflight changed'
    if mode != 'smoke':
        s = json.loads((LOGS / (NAMES['smoke'] + '_launch.json')).read_text())
        assert s['status'] == 'completed' and s['target_completed'] and s['resume_verified'] and s['actual_model_update_verified']
        assert s['source_sha256'] == pre['source_sha256'] and s['extension_sha256'] == pre['extension_sha256']
    record = LOGS / (NAMES[mode] + '_launch.json'); console = LOGS / (NAMES[mode] + '.log')
    assert not record.exists() and not console.exists() and not Path(pre['run_dir']).exists()
    free_gpu(); verify_static(pre)
    state = dict(pre, status='starting', name=NAMES[mode], console_log=str(console), created_at=now(),
                 physical_gpu=GPU, supervisor_pid=None, training_pid=None, target_completed=False)
    if mode == 'smoke': state['target_updates'] = 2
    write(record, state, exclusive=True)
    try:
        with console.open('xb') as f:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), mode, '--supervise'],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=now()); write(record, state); raise
    print(json.dumps(dict(dispatched=True, mode=mode, supervisor_pid=proc.pid, run_dir=pre['run_dir'],
                          launch_record=str(record)), indent=2))


if __name__ == '__main__': main()
