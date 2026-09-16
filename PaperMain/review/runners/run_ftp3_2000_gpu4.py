#!/usr/bin/env python3
"""Fresh FTP3 run19: 2000 updates and a 2000-update linear LR horizon on GPU4."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,os,signal,subprocess,sys
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
NAME='19_base_cqi4_ftp3_lr2000_s2024_20260911_gpu4'
PARENT='18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4'
BERNOULLI='16_base_cqi4_lrrestart2000_s2024_20260910_gpu5'
SCHEDULE={'lr_final':0.0,'lr_decay_updates':2000}
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review/runners'))
from paper_train import parser,plan,source_hashes,normalize_config,runtime_info,validate_checkpoint_schedule
from run_ftp3_gpu4 import free_gpu4,write_json,now
LOGS=ROOT/'review/logs';RUN=ROOT/'runs'/NAME
RECORD=LOGS/(NAME+'_launch.json');PREFLIGHT=LOGS/(NAME+'_preflight.json');CONSOLE=LOGS/(NAME+'.log')
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def preserved_results():
    hashes={}
    for name in (PARENT,BERNOULLI):
        for path in [ROOT/'runs'/name/'config.json',ROOT/'runs'/name/'paper_manifest.json',
                     *(ROOT/'runs'/name/'ckpt').glob('*.pt'),*(ROOT/'runs'/name/'csv_logs').glob('*.csv')]:
            assert path.is_file() and not path.is_symlink()
            hashes[str(path.relative_to(ROOT))]=SHA(path)
        state=json.loads((LOGS/(name+'_launch.json')).read_text())
        assert state['status']=='completed' and state['returncode']==0 and state['target_completed']
        assert state['checkpoint_update']==(999 if name==PARENT else 1999)
    return hashes


def prepare():
    command=[sys.executable,'-u','paper_train.py','--recipe','base','--name',NAME,
        '--device','cuda','--gpu','4','--threads','4','--seed','2024','--traffic-model','ftp3',
        '--replay-mode','batched','--num-updates','2000','--lr-final','0','--lr-decay-updates','2000',
        '--calibration-profile','results/cqi4_hl_corrected_v2/summary.json',
        '--calibration-reference-root','provenance/before_ftp3_sources']
    run,checkpoint,manifest=plan(parser().parse_args(command[3:]),ROOT)
    parent=json.loads((ROOT/'runs'/PARENT/'paper_manifest.json').read_text())
    assert run==RUN and checkpoint is None and manifest['purpose']=='training' and manifest['smoke_slots'] is None
    assert manifest['source_sha256']==source_hashes(ROOT)==parent['source_sha256']
    assert manifest['config']==parent['config']==normalize_config(parent['config'])
    for key in ('execution','calibration_profile','calibration_reference_root','calibration_application','replay_origin','traffic_origin'):
        assert manifest[key]==parent[key],key
    assert manifest['schedule']==SCHEDULE and manifest['target_updates']==2000
    assert manifest['execution']==dict(device='cuda',gpu='4',threads=4)
    assert manifest['config']['ppo_learning_rate']==.0003 and manifest['config']['traffic_model']=='ftp3'
    assert not {'--resume','--init_from','--lr-initial','--lr-start-update'}.intersection(command)
    return dict(command=command,run_dir=str(run),resume=None,comparison_run=PARENT,
                training_initialization='fresh model, optimizer and normalization state; seed 2024',
                previous_results_sha256=preserved_results(),**manifest)


def supervise():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    state=json.loads(RECORD.read_text());os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        assert source_hashes(ROOT)==state['source_sha256']
        assert preserved_results()==state['previous_results_sha256']
        free_gpu4()
        child=subprocess.Popen(state['command'],cwd=ROOT,stdin=subprocess.DEVNULL)
        state.update(status='running',supervisor_pid=os.getpid(),training_pid=child.pid);write_json(RECORD,state)
        code=child.wait();state['returncode']=code;write_json(RECORD,state)
        import torch
        torch.set_num_threads(1)
        checkpoint=RUN/'ckpt/latest.pt'
        if not checkpoint.exists():raise RuntimeError(f'exit {code} without checkpoint')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        update=int(ck['update']);state['checkpoint_update']=update
        assert normalize_config(ck['cfg'])==state['config'] and ck['lr_schedule']==SCHEDULE
        validate_checkpoint_schedule(ck,SCHEDULE)
        assert all(torch.isfinite(t).all() for t in ck['model'].values())
        assert all(math.isclose(g['lr'],.0003*(1-update/2000),rel_tol=0,abs_tol=1e-18) for g in ck['optimizer']['param_groups'])
        complete=(code==0 and update==1999)
        state.update(status='completed' if complete else ('stopped' if code==0 else 'failed'),
                     target_completed=complete,finished_at=now());write_json(RECORD,state)
    except BaseException as exc:
        state.update(status='failed',target_completed=False,error=repr(exc),finished_at=now());write_json(RECORD,state)
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--dry-run',action='store_true');cli.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    args=cli.parse_args()
    if args.supervise:
        assert not args.dry_run
        return supervise()
    pre=prepare()
    if args.dry_run:
        # Exercise the actual LR implementation, avoiding any CUDA context.
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3')
        import torch
        torch.set_num_threads(1)
        from train_phase2 import linear_learning_rate
        expected={0:.0003,500:.000225,1000:.00015,1500:.000075,1999:1.5e-7,2000:0.0}
        for u,want in expected.items():assert math.isclose(linear_learning_rate(.0003,0.,2000,u),want,rel_tol=0,abs_tol=1e-18)
        # runtime_info reads the CPU-visible GPU state; the true CUDA runtime
        # equality is checked by the separate paper_train --dry-run preflight.
        write_json(PREFLIGHT,pre,exclusive=True)
        print(json.dumps({'preflight':str(PREFLIGHT),'schedule':SCHEDULE,'lr_values':expected,'config_unchanged':True,'fresh':True},indent=2));return
    assert json.loads(PREFLIGHT.read_text())==pre,'Preflight changed'
    actual=json.loads((LOGS/(NAME+'_runtime_preflight.json')).read_text())
    parent=json.loads((ROOT/'runs'/PARENT/'paper_manifest.json').read_text())
    assert actual['runtime']==parent['runtime']
    for key in ('config','source_sha256','schedule','target_updates','execution','calibration_profile'):
        assert actual[key]==pre[key],key
    assert actual['resume'] is None
    assert not RECORD.exists() and not CONSOLE.exists() and not RUN.exists()
    free_gpu4()
    state=dict(pre,status='starting',name=NAME,created_at=now(),console_log=str(CONSOLE),physical_gpu=4,
               supervisor_pid=None,training_pid=None,target_completed=False)
    write_json(RECORD,state,exclusive=True)
    try:
        with CONSOLE.open('xb') as output:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--supervise'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=now());write_json(RECORD,state);raise
    print(json.dumps({'dispatched':True,'supervisor_pid':process.pid,'run_dir':str(RUN),'launch_record':str(RECORD)},indent=2))

if __name__=='__main__':main()
