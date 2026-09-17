#!/usr/bin/env python3
"""Phase-invariant raw-input recipe runner.

Usage: run_rawpi.py <recipe> <mode>  with recipe in {pi, pi_lr6}, mode in
{smoke, train}. Same structure as run_raw_invariant.py; the audit run is
unnumbered and deleted once the real launch is dispatched.

Mirrors review/runners/run_bernoulli_weekend.py: `preflight` compares the
protected planner's dry-run against an independent plan, `smoke` runs the
32-slot execution audit (1 update, then resume to 2), `train` starts the
1000-update run. Records live in review/logs/<name>_{preflight,launch}.json and
review/logs/<name>.log so paper_run_index.py links them into runs/<name>/.

    python review/runners/run_raw_invariant.py smoke --preflight
    python review/runners/run_raw_invariant.py smoke
    python review/runners/run_raw_invariant.py train --preflight
    python review/runners/run_raw_invariant.py train
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, os, signal, subprocess, sys, time
ROOT = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
sys.path.insert(0, str(ROOT))
import paper_train as pt
import paper_train_rawpi as pr
VARIANT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('pi', 'pi_lr6') else 'pi'
sys.argv = [sys.argv[0]] + sys.argv[2:]
RECIPE = pr.RECIPE_BASE if VARIANT == 'pi' else pr.RECIPE_LR6
GPU = 0 if VARIANT == 'pi' else 1
NAMES = {'smoke': f'audit_{RECIPE}_gpu{GPU}',
         'train': ('34_raw_phase_inv_lr1000_s2024_20260917_gpu0' if VARIANT == 'pi'
                   else '35_raw_phase_inv_lr6_lr1000_s2024_20260917_gpu1')}
UPDATES = {'smoke': 1, 'train': 1000}
SCHEDULE = {'lr_final': 0.0, 'lr_decay_updates': 1000}
COMPARISON = '26_base_stage1_lr1000_s2002024_20260914_gpu3'   # same recipe shape: base, 1000u, lr->0
LOGS = ROOT / 'review/logs'
EXTENSIONS = ['paper_train_rawpi.py', f'artifacts/{RECIPE}/config.json',
              'review/runners/run_rawpi.py']


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
    c = [sys.executable, '-u', 'paper_train_rawpi.py', '--recipe', RECIPE, '--name', NAMES[mode],
         '--device', 'cuda', '--gpu', str(GPU), '--threads', '4', '--seed', '2024',
         '--traffic-model', 'bernoulli', '--replay-mode', 'batched',
         '--num-updates', str(UPDATES[mode]), '--lr-final', '0', '--lr-decay-updates', '1000',
         '--calibration-profile', 'results/cqi4_hl_corrected_v2/summary.json',
         '--calibration-reference-root', 'provenance/before_ftp3_sources']
    if mode == 'smoke': c += ['--smoke-slots', '32']
    return c


def prepare(mode):
    c = command(mode)
    run, checkpoint, m = pr.plan(pr.parser().parse_args(c[3:]), ROOT)
    ref = json.loads((ROOT / 'runs' / COMPARISON / 'paper_manifest.json').read_text())
    assert m['source_sha256'] == pt.source_hashes(ROOT) == ref['source_sha256'], 'core sources must be untouched'
    expected = dict(ref['config'], seed=2024, **pr.EXPECTED_DIFF[RECIPE])
    if mode == 'smoke':
        expected.update(episode_len_main=32, episode_len_debug=32, ppo_epochs=1, ppo_minibatch_size=24,
                        ppo_eval_every=1, ppo_eval_episodes=1, ppo_save_every=1)
    assert m['config'] == expected, {k: (expected[k], m['config'].get(k)) for k in expected if expected[k] != m['config'].get(k)}
    assert checkpoint is None and m['schedule'] == SCHEDULE and m['recipe'] == RECIPE
    assert m['execution'] == dict(device='cuda', gpu=str(GPU), threads=4)
    assert m['calibration_profile'] == ref['calibration_profile']
    assert m['target_updates'] == UPDATES[mode]
    assert m['purpose'] == ('execution_smoke' if mode == 'smoke' else 'training')
    assert m['variant_provenance']['architecture'] == pr.ARCHITECTURE
    assert m['config']['ppo_learning_rate'] == pr.EXPECTED_DIFF[RECIPE].get('ppo_learning_rate', 3e-4)
    return dict(command=c, run_dir=str(run), resume=None, **m,
                extension_sha256={p: sha(ROOT / p) for p in EXTENSIONS},
                comparison_run=COMPARISON,
                training_initialization='fresh raw-input model, optimizer and return normalization; seed 2024',
                same_world_as_base=True, network_differs_from_base=True)


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
        pr.install()
        m0 = state['config']['ppo_learning_rate']
        phases = [(state['command'], UPDATES[mode])]
        if mode == 'smoke':
            phases.append(([sys.executable, '-u', 'paper_train_rawpi.py', '--recipe', RECIPE,
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
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
            # the checkpoint must load into the RAW class and be rejected by the stock one
            model, _ = pr.load_model(json.loads((run / 'paper_manifest.json').read_text()), run / 'ckpt/latest.pt', torch)
            assert isinstance(model.encoder, torch.nn.Identity)
            lr = m0 * (1 - u / 1000)
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
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), VARIANT, mode, '--supervise'],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=now()); write(record, state); raise
    print(json.dumps(dict(dispatched=True, mode=mode, supervisor_pid=proc.pid, run_dir=pre['run_dir'],
                          launch_record=str(record)), indent=2))
    if mode == 'train':
        # user rule 2026-09-16: once the audit has served its purpose (gating the
        # real launch), drop its run folder and console log so runs/ holds only
        # paper runs; the small preflight/launch JSONs stay as evidence.
        import shutil
        smoke_dir = ROOT / 'runs' / NAMES['smoke']
        if smoke_dir.is_dir() and not smoke_dir.is_symlink():
            shutil.rmtree(smoke_dir)
        (LOGS / (NAMES['smoke'] + '.log')).unlink(missing_ok=True)
        print(json.dumps(dict(audit_removed=str(smoke_dir)), indent=2))


if __name__ == '__main__': main()
