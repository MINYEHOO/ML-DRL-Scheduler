import os
os.environ["OMP_NUM_THREADS"]="6"; os.environ["MKL_NUM_THREADS"]="6"
os.environ["OPENBLAS_NUM_THREADS"]="6"; os.environ["CUDA_VISIBLE_DEVICES"]=""
import csv as csvmod, json, sys, time
import numpy as np
sys.path.insert(0,"/home/MYH/ML_DRL_Scheduler")
from config import Config
from env import SchedulerEnv
from baselines import SUSCQI
from train_phase2 import env_episode_metrics

cj=json.load(open("/home/MYH/ML_DRL_Scheduler/Run3/ScarcityK32/config.json"))
fields={k:(tuple(v) if isinstance(v,list) else v) for k,v in cj.items()
        if k in Config.__dataclass_fields__}
cfg=Config(**fields); cfg.sus_ortho_threshold=0.5
env=SchedulerEnv(cfg); sched=SUSCQI()
out=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
         "stats_sus05_ScarcityK32.csv"),"w",newline="")
w=csvmod.writer(out); w.writerow(["scheduler","seed","reward","thr","comp","drop"])
t0=time.time()
for seed in range(10000,10040):
    env.reset(seed); done=False
    while not done: _,_,done,_=env.step(sched.schedule(env))
    m=env_episode_metrics(env,cfg)
    w.writerow(["SUS-CQI@0.5",seed,m["reward"],m["throughput_mbps"],
                m["completion_rate"],m["retx_drop_rate"]]); out.flush()
    print(f"seed {seed}: {m['reward']:.1f} ({time.time()-t0:.0f}s)",flush=True)
print("done")
