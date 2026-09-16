#!/usr/bin/env python3
"""GPU5 execution check of fixed ranges and fresh paired policy initialization."""
import json, os, sys, hashlib
from pathlib import Path
ROOT=Path('/home/MYH/ML_DRL_Scheduler/PaperMain')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review/runners'))
from run_bernoulli_weekend import prepare, free_gpu, write, now
import paper_train as pt
base=prepare('base');narrow=prepare('narrow')
expected={'n_active_min','n_active_max','ue_speed_min','ue_speed_max','p_arrival_min','p_arrival_max'}
assert {k for k in base['config'] if base['config'][k]!=narrow['config'][k]}==expected
free_gpu(5)
pt.configure_execution(dict(device='cuda',gpu='5',threads=4))
import torch,numpy as np
from config import Config
from env import SchedulerEnv
from policy import ActorCritic
pt.preflight_device(torch,'cuda');torch.set_num_threads(4)
initial=[]
for manifest in (base,narrow):
    torch.manual_seed(2024)
    ac=ActorCritic(Config(**manifest['config']))
    assert float(ac.ret_mean)==0 and float(ac.ret_std)==1 and float(ac.ret_count)==0
    opt=torch.optim.Adam(ac.parameters(),lr=.0003)
    assert not opt.state
    h=hashlib.sha256()
    for name,t in ac.state_dict().items():h.update(name.encode());h.update(t.cpu().numpy().tobytes())
    initial.append(h.hexdigest())
assert initial[0]==initial[1]
cfg=Config(**dict(narrow['config'],episode_len_main=32,episode_len_debug=32))
env=SchedulerEnv(cfg);records=[]
for episode in (0,1,10000):
    env.reset(episode)
    assert env.traffic.n_active==24 and env.traffic.p_arrival_ep==.325
    assert env.ue_speeds_kmh.shape==(32,) and np.all(env.ue_speeds_kmh==22.5)
    records.append(dict(episode_idx=episode,n_active=env.traffic.n_active,arrival_probability=env.traffic.p_arrival_ep,speeds_kmh=env.ue_speeds_kmh.tolist()))
result=dict(status='passed',validated_at=now(),gpu=5,episode_len=32,diagnostic_only=True,
            config_differences=sorted(expected),actual_narrow_resets=records,
            initial_policy_sha256=initial[0],same_initial_model=True,empty_optimizer=True,
            return_normalizer=dict(mean=0,std=1,count=0),core_sources_unchanged=pt.source_hashes(ROOT)==base['source_sha256'])
write(ROOT/'review/logs/bernoulli_weekend_environment.json',result,exclusive=True)
print(json.dumps(result,indent=2))
