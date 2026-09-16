from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys

root = Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
name = '14_base_cqi4_batched_evalfast_lr1000_s2024_20260909_gpu5'
run = root/'runs'/name
logs = root/'review/logs'
pre = json.loads((logs/(name+'_preflight.json')).read_text())
lineage = json.loads((run/'PARENT_RUN.json').read_text())
assert pre['purpose'] == 'training' and pre['smoke_slots'] is None
assert pre['resume'] == str(run/'ckpt/latest.pt')
assert pre['execution'] == {'device': 'cuda', 'gpu': '5', 'threads': 4}
assert pre['continuation'] == lineage and lineage['next_update'] == 21
assert pre['schedule'] == {'lr_final': 0.0, 'lr_decay_updates': 1000}
assert pre['target_updates'] == 1000 and pre['config']['seed'] == 2024
assert pre['config']['ppo_batched_replay'] is True and pre['config']['episode_len_main'] == 1000
assert pre['config']['la_beta_by_depth'] == [1.0162, 0.8509, 0.7732, 0.728]
for filename, h in pre['source_sha256'].items():
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == h
for filename, h in lineage['copied_checkpoint_sha256'].items():
    assert hashlib.sha256((run/'ckpt'/filename).read_bytes()).hexdigest() == h
command = [sys.executable, '-u', 'paper_train.py', '--recipe', 'base',
           '--resume', str(run/'ckpt/latest.pt'), '--num-updates', '1000']
record, console = logs/(name+'_launch.json'), logs/(name+'.log')
state = dict(status='starting', name=name, run_dir=str(run), console_log=str(console), command=command,
             created_at=datetime.now(timezone.utc).isoformat(), supervisor_pid=None, training_pid=None,
             device='cuda', physical_gpu=5, seed=2024, target_updates=1000, slots=1000,
             beta=pre['config']['la_beta_by_depth'], lr_schedule=pre['schedule'],
             initial_lr=pre['config']['ppo_learning_rate'], replay_mode='batched',
             eval_action_only=True, parent_run=lineage['parent_run'], start_update=21,
             resumed_checkpoint_update=20, target_completed=False)
assert not record.exists() and not console.exists()
with record.open('x') as f:
    json.dump(state, f, indent=2)
    f.write('\n')
supervisor_code = '''
from pathlib import Path
from datetime import datetime, timezone
import json, os, signal, subprocess, sys
signal.signal(signal.SIGHUP, signal.SIG_IGN)
record=Path(sys.argv[1]); state=json.loads(record.read_text())
def save():
    tmp=record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\\n')
    os.replace(tmp,record)
try:
    child=subprocess.Popen(state['command'],cwd=str(Path(state['run_dir']).parent.parent),stdin=subprocess.DEVNULL)
    state.update(status='running',supervisor_pid=os.getpid(),training_pid=child.pid)
    save()
    code=child.wait()
    # Read the final checkpoint on CPU only; a resumed run prints remaining updates.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(Path(state['run_dir'])/'ckpt/latest.pt',map_location='cpu',weights_only=False)
    update=int(checkpoint['update'])
    complete=(code==0 and update>=state['target_updates']-1)
    state.update(status=('completed' if complete else ('stopped' if code==0 else 'failed')),
                 returncode=code,checkpoint_update=update,target_completed=complete,
                 finished_at=datetime.now(timezone.utc).isoformat())
    save()
except BaseException as exc:
    state.update(status='supervisor_error',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
    save()
    raise
'''
with console.open('xb') as output:
    process = subprocess.Popen([sys.executable, '-u', '-c', supervisor_code, str(record)], cwd=root,
                               stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
print(json.dumps(dict(dispatched=True, supervisor_pid=process.pid, run_dir=str(run),
                      console_log=str(console), launch_record=str(record)), indent=2))
