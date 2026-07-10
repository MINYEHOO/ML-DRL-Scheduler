"""Measure the SUS-CQI@0.8 bar on the three older Run3 experiments
(their eval CSVs only contain the pre-Jul-1 baseline set)."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                      # noqa: E402
from env import SchedulerEnv                   # noqa: E402
from baselines import SUSCQI, CQIGreedy        # noqa: E402
from train_phase2 import env_episode_metrics   # noqa: E402

t0 = time.time()
for run in ("ScarcityK32", "Uniform10_Ent002", "DeadlineScarcity"):
    cj = json.load(open(f"/home/MYH/ML_DRL_Scheduler/Run3/{run}/config.json"))
    fields = {}
    for k, v in cj.items():
        if k in Config.__dataclass_fields__:
            fields[k] = tuple(v) if isinstance(v, list) else v
    cfg = Config(**fields)
    env = SchedulerEnv(cfg)
    print(f"\n### {run} (K={cfg.num_ue}, deadline {cfg.deadline_min}-"
          f"{cfg.deadline_max}, speed {cfg.ue_speed_kmh})", flush=True)
    for name, sched, th in (("CQI-greedy(anchor)", CQIGreedy(), None),
                            ("SUS-CQI@0.8", SUSCQI(), 0.8)):
        cfg.sus_ortho_threshold = th if th is not None else 0.5
        ms = []
        for ep_idx in range(cfg.ppo_eval_episodes):
            env.reset(ep_idx + 10000)
            done = False
            while not done:
                _, _, done, _ = env.step(sched.schedule(env))
            ms.append(env_episode_metrics(env, cfg))
        g = lambda k: np.mean([m[k] for m in ms])   # noqa: E731
        print(f"  {name:20s} rew {g('reward'):8.1f}  thr {g('throughput_mbps'):6.2f}"
              f"  comp {g('completion_rate'):.3f}  drop {g('retx_drop_rate'):.3f}"
              f"  depth {g('mu_depth'):.2f}   ({time.time()-t0:.0f}s)", flush=True)
