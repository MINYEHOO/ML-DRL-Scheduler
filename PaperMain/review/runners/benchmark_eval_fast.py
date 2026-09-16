from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,copy
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']='5'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import numpy as np
import torch
torch.set_num_threads(4)
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
from train_phase2 import PPOScheduler,env_episode_metrics
from calibration.profile import load_profile

root=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
parent=root/'runs/12_base_cqi4_batched_lr1000_s2024_20260909_gpu5'
out=root/'review/logs/eval_fast_comparison';out.mkdir(exist_ok=False)
m=json.loads((parent/'paper_manifest.json').read_text());cfg=Config(**m['config'])
profile=load_profile(root/'results/cqi4_hl_corrected_v2/summary.json')
assert cfg.episode_len_main==1000 and cfg.ppo_eval_episodes==3
assert list(cfg.la_beta_by_depth)==profile['beta_rounded']
ckpath=parent/'ckpt/best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
torch.manual_seed(90007);torch.cuda.manual_seed_all(90007)
ac=ActorCritic(cfg).to('cuda');ac.load_state_dict(ck['model'],strict=True)
ac.train() # Same module mode as existing inline evaluation after ppo_update.
before={k:v.detach().clone() for k,v in ac.state_dict().items()}
env=SchedulerEnv(cfg)
for ep in range(10000,10003):env.reset(ep)
print('WARMED_CHANNEL_AND_CSI',flush=True)
class LegacyScheduler:
    def schedule(self,env):
        with torch.no_grad():
            return ac.decode(env.get_observation(),deterministic=True)['action_sequence']
arms={}
for mode,scheduler in [('legacy',LegacyScheduler()),('action_only',PPOScheduler(ac,deterministic=True))]:
    actions=[];metrics=[];episode_times=[]
    cpu_rng=torch.get_rng_state().clone();cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
    total_start=time.perf_counter()
    for episode in range(10000,10003):
        torch.cuda.synchronize();start=time.perf_counter()
        obs=env.reset(episode);done=False;episode_actions=[]
        while not done:
            allocation=scheduler.schedule(env)
            episode_actions.append(allocation.copy())
            obs,_,done,_=env.step(allocation)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        values=env_episode_metrics(env,cfg)
        actions.append(np.stack(episode_actions));metrics.append(values);episode_times.append(seconds)
        print(json.dumps({'mode':mode,'episode':episode,'seconds':seconds,'reward':values['reward'],'goodput_mbps':values['goodput_mbps']}),flush=True)
    seconds=time.perf_counter()-total_start
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,ac.state_dict()[k]) for k,v in before.items())
    trace=np.stack(actions)
    np.savez_compressed(out/(mode+'_actions.npz'),actions=trace)
    arms[mode]=dict(seconds=seconds,episode_seconds=episode_times,metrics=metrics,
                    action_sha256=hashlib.sha256(trace.tobytes()).hexdigest(),action_shape=list(trace.shape),
                    rng_unchanged=True,model_unchanged=True)
    if mode=='legacy':legacy_actions=trace.copy()
    else:assert np.array_equal(trace,legacy_actions),'Evaluation actions changed'
for a,b in zip(arms['legacy']['metrics'],arms['action_only']['metrics']):
    assert a.keys()==b.keys()
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12,equal_nan=True,err_msg=k)
result=dict(status='passed',observed_at=datetime.now(timezone.utc).isoformat(),physical_gpu=5,
            checkpoint=str(ckpath),checkpoint_update=ck['update'],checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
            episodes=[10000,10001,10002],slots_per_episode=1000,threads=4,cache='same prewarmed channel and CSI cache for both arms',
            arms=arms,exact_actions=True,metrics_match=True,metric_tolerance=1e-12,
            speedup=arms['legacy']['seconds']/arms['action_only']['seconds'],
            source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['policy.py','train_phase2.py']})
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
print('EVAL_COMPARISON_PASSED',json.dumps({'speedup':result['speedup'],'legacy_seconds':arms['legacy']['seconds'],'action_only_seconds':arms['action_only']['seconds'],'exact_actions':True,'metrics_match':True}),flush=True)
