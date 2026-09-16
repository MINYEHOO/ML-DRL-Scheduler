#!/usr/bin/env python3
"""Detached two-GPU ID campaign, with disjoint smoke and main evaluation bands."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,os,signal,subprocess,sys,time
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
C=ROOT/'results/ftp3_id_run18_best_u0639_s2024_20260911';INPUT=C/'inputs'
RUN='runs/18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4'
sys.path.insert(0,str(ROOT))
from paper_train import source_hashes,atomic_json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()

def commands(mode):
    start,count=(119000,2) if mode=='smoke' else (120000,100)
    base=[sys.executable,'-u','paper_run_eval.py','--run',RUN,'--checkpoint','best',
        '--checkpoint-path',str(INPUT/'best.pt'),'--episode-start',str(start),'--episodes',str(count),
        '--num-shards','2','--threads','4','--history-audit',str(INPUT/'episode_history_audit.json')]
    if mode=='smoke':base+=['--smoke-slots','32']
    dest=C/'smoke' if mode=='smoke' else C
    return [base+['--gpu',str(gpu),'--shard-index',str(i),'--out',str(dest/f'shard{i}')] for i,gpu in enumerate([3,5])]

def check_free(gpu):
    query=lambda category,fields:subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-'+category+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    assert not query('compute-apps','pid'),f'GPU {gpu} occupied'
    assert int(query('gpu','memory.used'))<500,f'GPU {gpu} not empty'

def stable_hashes():
    return dict(core=source_hashes(ROOT),eval=sha(ROOT/'paper_run_eval.py'),merge=sha(ROOT/'paper_run_eval_merge.py'),checkpoint=sha(INPUT/'best.pt'),plan=sha(INPUT/'experiment_plan.json'),history=sha(INPUT/'episode_history_audit.json'))

def merged(mode):return (C/'smoke'/'combined') if mode=='smoke' else C/'combined'

def merge(mode):
    dest=C/'smoke' if mode=='smoke' else C
    command=[sys.executable,'-u','paper_run_eval_merge.py','--shards',str(dest/'shard0'),str(dest/'shard1'),'--out',str(merged(mode))]
    if mode=='smoke':command+=['--allow-smoke','--bootstrap-replicates','1000']
    subprocess.run(command,cwd=ROOT,check=True)

def compare_serial():
    # Independent GPU3 serial execution of the same diagnostic pair verifies
    # deterministic statistics across GPU3/GPU5 and the shard partition.
    command=commands('smoke')[0]
    command[command.index('--num-shards')+1]='1'
    command[command.index('--out')+1]=str(C/'smoke'/'serial')
    subprocess.run(command,cwd=ROOT,check=True)
    def rows(path):
        with path.open() as f:return {(int(x['episode_idx']),x['scheduler']):x for x in csv.DictReader(f)}
    serial=rows(C/'smoke'/'serial'/'metrics.csv')
    split=rows(C/'smoke'/'shard0'/'metrics.csv');split.update(rows(C/'smoke'/'shard1'/'metrics.csv'))
    assert set(serial)==set(split) and len(serial)==14
    ignored={'reset_seconds','rollout_seconds','elapsed_seconds'}
    max_abs=0.0
    for pair,left in serial.items():
        right=split[pair];assert set(left)==set(right)
        for key,value in left.items():
            if key in ignored:continue
            try:a,b=float(value),float(right[key])
            except ValueError:assert value==right[key],(pair,key,value,right[key]);continue
            assert math.isfinite(a) and math.isfinite(b)
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(pair,key,a,b)
            max_abs=max(max_abs,abs(a-b))
    # Empty-bin rates must remain null and every diagnostic row is labelled.
    summary=json.loads((merged('smoke')/'summary.json').read_text());assert summary['rows']==14 and summary['protocol']['diagnostic_only'] is True
    serial_manifest=json.loads((C/'smoke'/'serial'/'manifest.json').read_text())
    assert serial_manifest['status']=='completed' and serial_manifest['model_state_unchanged'] and serial_manifest['torch_rng_unchanged']
    return dict(status='passed',rows=14,serial_vs_two_gpu_shards=True,max_abs_difference=max_abs,rtol=1e-10,atol=1e-8)

def supervise(mode,record):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    state=json.loads(record.read_text());children=[];handles=[]
    try:
        assert stable_hashes()==state['hashes']
        for gpu in [3,5]:check_free(gpu)
        state.update(status='running',supervisor_pid=os.getpid());atomic_json(record,state)
        for i,command in enumerate(state['commands']):
            output=(C/f'{mode}_gpu{[3,5][i]}.log').open('xb');handles.append(output)
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
            children.append(child);state.setdefault('workers',[]).append(dict(shard_index=i,gpu=[3,5][i],pid=child.pid));atomic_json(record,state)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('An evaluation shard failed; see worker log')
            time.sleep(2)
        codes=[p.returncode for p in children];assert codes==[0,0],codes
        state.update(phase='merge',worker_returncodes=codes);atomic_json(record,state)
        merge(mode)
        if mode=='smoke':
            state['phase']='serial_check';atomic_json(record,state)
            state['serial_equivalence']=compare_serial()
        assert stable_hashes()==state['hashes']
        summary=json.loads((merged(mode)/'summary.json').read_text())
        assert summary['episodes']==(2 if mode=='smoke' else 100) and summary['rows']==(14 if mode=='smoke' else 700)
        state.update(status='completed',finished_at=now(),summary=str(merged(mode)/'summary.md'),total_rows=summary['rows']);atomic_json(record,state)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='failed',error=repr(exc),finished_at=now());atomic_json(record,state);raise
    finally:
        for output in handles:output.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['smoke','main']);p.add_argument('--preflight',action='store_true');p.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    record=C/f'{a.mode}_launch.json'
    if a.supervise:return supervise(a.mode,record)
    cmd=commands(a.mode);hashes=stable_hashes();plan=json.loads((INPUT/'experiment_plan.json').read_text());assert hashes['core']==plan['source_sha256'] and hashes['checkpoint']==plan['checkpoint_sha256']
    if a.preflight:
        protocols=[]
        for i,c in enumerate(cmd):
            with (C/f'{a.mode}_preflight{i}.json').open('x') as f:subprocess.run(c+['--dry-run'],cwd=ROOT,stdout=f,check=True)
            d=json.loads((C/f'{a.mode}_preflight{i}.json').read_text());protocols.append(d['protocol'])
        assert protocols[0]==protocols[1]
        with (C/f'{a.mode}_preflight_hashes.json').open('x') as f:json.dump(hashes,f,indent=2)
        print('PREFLIGHT_OK',a.mode);return
    assert json.loads((C/f'{a.mode}_preflight_hashes.json').read_text())==hashes
    if a.mode=='main':
        smoke=json.loads((C/'smoke_launch.json').read_text());assert smoke['status']=='completed' and smoke['hashes']==hashes
        assert smoke['serial_equivalence']['status']=='passed'
    for gpu in [3,5]:check_free(gpu)
    assert not record.exists()
    state=dict(status='starting',mode=a.mode,created_at=now(),commands=cmd,hashes=hashes,source_run=RUN,checkpoint_update=639,world='ID',traffic='ftp3',supervisor_pid=None,workers=[])
    with record.open('x') as f:json.dump(state,f,indent=2)
    with (C/f'{a.mode}_supervisor.log').open('xb') as f:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),a.mode,'--supervise'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(json.dumps(dict(dispatched=True,supervisor_pid=process.pid,record=str(record),campaign=str(C)),indent=2))
if __name__=='__main__':main()
