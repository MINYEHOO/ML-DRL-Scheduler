from pathlib import Path
from datetime import datetime, timezone
import json, math, subprocess, sys, time

root=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
name='13_audit_eval_fast_gpu5'
run=root/'runs'/name
logs=root/'review/logs'
record=logs/(name+'_launch.json')
command=[sys.executable,'-u','paper_train.py','--recipe','base','--name',name,
         '--device','cuda','--gpu','5','--threads','4','--seed','2024',
         '--smoke-slots','32','--num-updates','1','--lr-final','0',
         '--lr-decay-updates','1000','--replay-mode','batched',
         '--calibration-profile','results/cqi4_hl_corrected_v2/summary.json']
assert not run.exists() and not record.exists()
state=dict(status='starting',name=name,run_dir=str(run),command=command,
           created_at=datetime.now(timezone.utc).isoformat(),purpose='execution_smoke',
           physical_gpu=5)
def save():
    record.write_text(json.dumps(state,indent=2)+'\n')
save()
with (logs/(name+'.log')).open('xb') as out:
    for phase,cmd in [('fresh',command),('resume',[sys.executable,'-u','paper_train.py',
            '--recipe','base','--resume',str(run/'ckpt/latest.pt'),'--num-updates','2'])]:
        start=time.perf_counter()
        child=subprocess.Popen(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT)
        state.update(status='running',phase=phase,training_pid=child.pid);save()
        code=child.wait()
        state[phase+'_returncode']=code
        state[phase+'_seconds']=time.perf_counter()-start
        save()
        if code:
            state.update(status='failed',finished_at=datetime.now(timezone.utc).isoformat());save()
            raise SystemExit(code)
import torch
torch.set_num_threads(1)
m=json.loads((run/'paper_manifest.json').read_text())
ck=torch.load(run/'ckpt/latest.pt',map_location='cpu',weights_only=False)
assert ck['update']==1 and ck['cfg']['ppo_batched_replay'] is True
assert m['replay_origin']=='override' and m['config']['ppo_batched_replay'] is True
assert m['schedule']=={'lr_final':0.0,'lr_decay_updates':1000}
assert all(math.isclose(g['lr'],.0003*(1-1/1000),rel_tol=0,abs_tol=1e-15)
           for g in ck['optimizer']['param_groups'])
assert all(torch.isfinite(t).all() for t in ck['model'].values())
state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),
             checkpoint_update=ck['update'],optimizer_lr=ck['optimizer']['param_groups'][0]['lr'],
             replay_restored=True);save()
print('SMOKE_VALIDATED',json.dumps(state),flush=True)
