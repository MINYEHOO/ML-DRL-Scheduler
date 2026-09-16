#!/usr/bin/env python3
"""Three-GPU, detached smoke/pilot/OOD campaign for frozen runs 16, 21 and 22."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, json, math, os, signal, subprocess, sys, time

import paper_train as pt

ROOT = Path(__file__).resolve().parent
C = ROOT/'results/bernoulli_ood_runs16_21_22_20260913'
WORLDS = ['ID','P055','P010','V60max','CSI02','D26strict','STORM2','K8','K48','K60']
RUNS = {
    16: '16_base_cqi4_lrrestart2000_s2024_20260910_gpu5',
    21: '21_base_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu3',
    22: '22_narrow_mean_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu5',
}
WORKERS = [(3,16,['ppo','sus_cqi','su_cqi']), (4,21,['ppo','sus_pf','su_pf']),
           (5,22,['ppo','sus_random','su_random'])]
PILOT_WORLDS = {3:['ID','P055','P010','K60'],4:['V60max','CSI02','D26strict'],5:['STORM2','K8','K48']}
NEW_FILES = ['paper_ood_inputs.py','paper_ood_eval.py','paper_ood_merge.py','paper_ood_campaign.py',
             'paper_run_eval.py','paper_run_eval_merge.py','paper_train_variants.py']
now = lambda: datetime.now(timezone.utc).isoformat()

def hashes():
    frozen = json.loads((C/'inputs/frozen_inputs.json').read_text())
    actual = dict(core=pt.source_hashes(ROOT),tools={p:pt.sha_file(ROOT/p) for p in NEW_FILES},
                  frozen=pt.sha_file(C/'inputs/frozen_inputs.json'),
                  history=pt.sha_file(C/'inputs/episode_history_audit.json'),runs={})
    if actual['core'] != frozen['current_source_sha256'] or actual['history'] != frozen['history_audit_sha256']:
        raise ValueError('Frozen current sources/history changed')
    for n, name in RUNS.items():
        wanted=frozen['runs'][str(n)]; run=ROOT/'runs'/name
        current=dict(checkpoint_sha256=pt.sha_file(run/'ckpt/best.pt'),manifest_sha256=pt.sha_file(run/'paper_manifest.json'),
                     config_sha256=pt.sha_file(run/'config.json'),completion_sha256=pt.sha_file(ROOT/'review/logs'/f'{name}_launch.json'))
        if any(current[k]!=wanted[k] for k in current): raise ValueError(f'Frozen run {n} input changed')
        if pt.sha_file(C/f'inputs/run{n}/best.pt')!=current['checkpoint_sha256']: raise ValueError('Checkpoint snapshot differs')
        actual['runs'][str(n)] = current
    return actual

def free(gpu):
    def query(kind,fields):
        return subprocess.check_output(['nvidia-smi','-i',str(gpu),f'--query-{kind}={fields}',
                                        '--format=csv,noheader,nounits'],text=True).strip()
    if query('compute-apps','pid') or int(query('gpu','memory.used')) >= 500:
        raise ValueError(f'GPU {gpu} is occupied')

def stage_root(mode):
    return C/'smoke' if mode=='smoke' else C

def commands(mode, phase):
    diagnostic = mode=='smoke'
    start = (138000 if phase=='pilot' else 138001) if diagnostic else (139000 if phase=='pilot' else 140000)
    episodes = 1 if diagnostic else (8 if phase=='pilot' else 100)
    output = stage_root(mode)
    result=[]
    for gpu,n,keys in WORKERS:
        selected = 21 if phase=='pilot' else n
        directory = output/f'pilot/gpu{gpu}' if phase=='pilot' else output/f'evaluation/run{n}'
        command=[sys.executable,'-u','paper_ood_eval.py','--run',f'runs/{RUNS[selected]}',
                 '--base-run',f'runs/{RUNS[21]}','--mode',phase,'--gpu',str(gpu),'--threads','4',
                 '--episode-start',str(start),'--episodes',str(episodes),'--out',str(directory),
                 '--history-audit',str(C/'inputs/episode_history_audit.json'),
                 '--worlds',*(PILOT_WORLDS[gpu] if phase=='pilot' else WORLDS),
                 '--schedulers',*(['sus_cqi'] if phase=='pilot' else keys)]
        if phase=='eval':command += ['--thresholds',str(output/'thresholds.json')]
        if diagnostic: command += ['--smoke-slots','32']
        result.append(dict(gpu=gpu,model=selected,command=command,out=str(directory)))
    return result

def completed(directory):
    directory=Path(directory);record=json.loads((directory/'manifest.json').read_text())
    if record.get('status')!='completed' or not record.get('model_state_unchanged') or not record.get('torch_rng_unchanged'):
        raise ValueError(f'Incomplete or mutated evaluation: {directory}')
    if record['rows_written']!=record['total_rows'] or record['metrics_sha256']!=pt.sha_file(directory/'metrics.csv'):
        raise ValueError(f'Evaluation row count/hash mismatch: {directory}')
    return record

def freeze_thresholds(mode):
    from paper_ood_eval import threshold_choices,SELECTION_RULE
    destination=stage_root(mode)/'thresholds.json'
    if destination.exists(): raise ValueError('Threshold selection already frozen')
    rows=[];directories=[]
    for job in commands(mode,'pilot'):
        record=completed(job['out'])
        directories.append(Path(job['out']).relative_to(ROOT).as_posix())
        with (Path(job['out'])/'metrics.csv').open() as stream:rows.extend(csv.DictReader(stream))
    ids=[138000] if mode=='smoke' else list(range(139000,139008))
    selected=threshold_choices(rows,WORLDS,ids)
    doc=dict(status='validated',selection_metric='reward',selection_rule=SELECTION_RULE,
             diagnostic_only=mode=='smoke',thresholds=selected,pilot_episode_ids=ids,
             pilot_directories=directories,created_at=now())
    pt.atomic_json(destination,doc)
    return doc

def run_phase(mode,phase,state,record):
    jobs=commands(mode,phase);handles=[];children=[]
    if hashes()!=state['hashes']: raise ValueError('Campaign inputs changed before stage')
    for gpu,_,_ in WORKERS:free(gpu)
    state.update(phase=phase,workers=[],phase_started_at=now());pt.atomic_json(record,state)
    try:
        for job in jobs:
            log=stage_root(mode)/f'{phase}_gpu{job["gpu"]}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('xb');handles.append(stream)
            child=subprocess.Popen(job['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            children.append(child)
            state['workers'].append({**job,'pid':child.pid,'log':str(log)})
            pt.atomic_json(record,state)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):
                raise RuntimeError(f'{phase} worker failed; inspect the recorded GPU log')
            time.sleep(2)
        if any(p.returncode!=0 for p in children): raise RuntimeError('Evaluation worker returned an error')
        for job in jobs:completed(job['out'])
        state.setdefault('completed_phases',[]).append(dict(phase=phase,finished_at=now(),jobs=state['workers']))
        pt.atomic_json(record,state)
    except BaseException:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        for stream in handles:stream.close()

def merge(mode):
    command=[sys.executable,'-u','paper_ood_merge.py','--inputs',
             *(str(stage_root(mode)/f'evaluation/run{n}') for _,n,_ in WORKERS),
             '--out',str(stage_root(mode)/'combined')]
    if mode=='smoke': command += ['--allow-smoke']
    subprocess.run(command,cwd=ROOT,check=True)
    return str(stage_root(mode)/'combined/summary.md')

def cross_gpu_random_check():
    job=commands('smoke','eval')[2]
    command=job['command'].copy()
    command[command.index('--gpu')+1]='3'
    out=C/'smoke/cross_gpu_run22'
    command[command.index('--out')+1]=str(out)
    pos=command.index('--schedulers')
    command[pos+1:pos+4]=['su_random','sus_random','ppo']
    free(3)
    with (C/'smoke/cross_gpu_run22.log').open('xb') as stream:
        subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
    completed(out)
    def rows(path):
        with Path(path).open() as stream:return {(r['world'],r['scheduler'],r['episode_idx']):r for r in csv.DictReader(stream)}
    left=rows(out/'metrics.csv');right=rows(Path(job['out'])/'metrics.csv')
    if set(left)!=set(right):raise ValueError('Cross-GPU diagnostic row grid differs')
    largest=0.
    for key in left:
        for metric,a in left[key].items():
            b=right[key][metric]
            if metric in ('elapsed_seconds','reset_seconds','rollout_seconds'):continue
            if a==b:continue
            try: x,y=float(a),float(b)
            except ValueError:raise ValueError(f'Cross-GPU metadata differs: {key}/{metric}')
            if not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8):raise ValueError(f'Cross-GPU metric differs: {key}/{metric}: {a} / {b}')
            largest=max(largest,abs(x-y))
    return dict(status='passed',rows=len(left),physical_gpus=[5,3],reversed_scheduler_order=True,max_abs_difference=largest)

def supervise(mode,record):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    state=json.loads(record.read_text())
    try:
        state.update(status='running',supervisor_pid=os.getpid());pt.atomic_json(record,state)
        run_phase(mode,'pilot',state,record)
        state.update(phase='threshold_selection');pt.atomic_json(record,state)
        state['thresholds']=freeze_thresholds(mode);pt.atomic_json(record,state)
        run_phase(mode,'eval',state,record)
        state.update(phase='merge');pt.atomic_json(record,state)
        state['summary']=merge(mode)
        if mode=='smoke':
            state.update(phase='cross_gpu_check');pt.atomic_json(record,state)
            state['cross_gpu_equivalence']=cross_gpu_random_check()
        if hashes()!=state['hashes']:raise ValueError('Frozen campaign inputs changed during evaluation')
        state.update(status='completed',phase='completed',finished_at=now());pt.atomic_json(record,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=now());pt.atomic_json(record,state)
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','main'])
    p.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv); record=C/f'{a.mode}_launch.json'
    if a.supervise:return supervise(a.mode,record)
    frozen_hashes=hashes()
    if record.exists(): raise ValueError('Campaign mode already launched; outputs cannot be overwritten')
    if a.mode=='main':
        smoke=json.loads((C/'smoke_launch.json').read_text())
        if (smoke.get('status')!='completed' or smoke['hashes']!=frozen_hashes
                or smoke.get('cross_gpu_equivalence',{}).get('status')!='passed'):
            raise ValueError('Main campaign requires matching successful smoke and cross-GPU checks')
    for gpu,_,_ in WORKERS:free(gpu)
    state=dict(status='starting',mode=a.mode,created_at=now(),phase='pilot',hashes=frozen_hashes,
               campaign=str(C),worlds=WORLDS,physical_gpus=[3,4,5],model_assignments=WORKERS,
               main_episodes_per_world=100,main_expected_rows=9000,workers=[],supervisor_pid=None,
               sequence=['pilot','threshold_selection','eval','merge'],
               checkpoint_selection='Pre-existing best checkpoints: run16@1989, run21@649, run22@1469; frozen before pilot/test')
    with record.open('x') as stream:json.dump(state,stream,indent=2)
    with (C/f'{a.mode}_supervisor.log').open('xb') as stream:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),a.mode,'--supervise'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(json.dumps(dict(dispatched=True,mode=a.mode,supervisor_pid=process.pid,record=str(record)),indent=2))

if __name__=='__main__': main()
