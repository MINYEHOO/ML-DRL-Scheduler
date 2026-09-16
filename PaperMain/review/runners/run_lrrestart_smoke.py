from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, subprocess, sys, time

root=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
name='15_audit_lrrestart_gpu5';run=root/'runs'/name;logs=root/'review/logs'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
schedule=dict(lr_final=0.0,lr_decay_updates=1000,lr_initial=1e-5,lr_start_update=2)
before=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
assert before['update']==1 and 'lr_schedule' not in before
record=logs/(name+'_launch.json')
state=dict(status='starting',name=name,run_dir=str(run),purpose='execution_smoke',
           created_at=datetime.now(timezone.utc).isoformat(),physical_gpu=5,
           schedule=schedule,parent_run='13_audit_eval_fast_gpu5')
assert not record.exists()
def save():
    temp=record.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');os.replace(temp,record)
save()
with (logs/(name+'.log')).open('xb') as output:
    for phase,total in [('legacy_continuation',3),('segment_resume',4)]:
        command=[sys.executable,'-u','paper_train.py','--recipe','base',
                 '--resume',str(run/'ckpt/latest.pt'),'--num-updates',str(total)]
        started=time.perf_counter()
        process=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT)
        state.update(status='running',phase=phase,command=command,training_pid=process.pid);save()
        code=process.wait();state[phase+'_returncode']=code
        state[phase+'_seconds']=time.perf_counter()-started;save()
        if code:
            state.update(status='failed');save();raise SystemExit(code)
        ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
        assert ck['update']==total-1 and ck['lr_schedule']==schedule
        want=1e-5*(1-(ck['update']-2)/1000)
        assert all(math.isclose(g['lr'],want,rel_tol=0,abs_tol=1e-19) for g in ck['optimizer']['param_groups'])
        assert all(torch.isfinite(t).all() for t in ck['model'].values())
        state[phase+'_checkpoint_update']=ck['update']
        state[phase+'_optimizer_lr']=ck['optimizer']['param_groups'][0]['lr'];save()
actor_changed=any(not torch.equal(v,ck['model'][k]) for k,v in before['model'].items()
                  if k.endswith('weight') and not k.startswith('value_head.'))
critic_changed=any(not torch.equal(v,ck['model'][k]) for k,v in before['model'].items()
                   if k.endswith('weight') and k.startswith('value_head.'))
assert actor_changed and critic_changed
state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),
             checkpoint_update=ck['update'],optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],
             lr_segment_restored=True,actual_model_update_verified=True,
             actor_weights_changed=actor_changed,critic_weights_changed=critic_changed)
save()
print('LR_RESTART_SMOKE_PASSED',json.dumps(state),flush=True)
