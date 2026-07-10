import os
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys
import numpy as np
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config
from env import SchedulerEnv
from baselines import SUCQI
from train_phase2 import env_episode_metrics

for run in ("Uniform10_Ent002", "DeadlineScarcity"):
    cj = json.load(open(f"/home/MYH/ML_DRL_Scheduler/Run3/{run}/config.json"))
    fields = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in cj.items() if k in Config.__dataclass_fields__}
    cfg = Config(**fields)
    env = SchedulerEnv(cfg)
    sched = SUCQI()
    ms = []
    for ep_idx in range(cfg.ppo_eval_episodes):
        env.reset(ep_idx + 10000)
        done = False
        while not done:
            _, _, done, _ = env.step(sched.schedule(env))
        ms.append(env_episode_metrics(env, cfg))
    g = lambda k: np.mean([m[k] for m in ms])
    print(f"{run}: SU-CQI-greedy rew {g('reward'):8.1f}  thr {g('throughput_mbps'):6.2f}"
          f"  comp {g('completion_rate'):.3f}  drop {g('retx_drop_rate'):.3f}"
          f"  depth {g('mu_depth'):.2f}", flush=True)
