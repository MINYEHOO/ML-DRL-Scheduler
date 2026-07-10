"""Exact-protocol eval: SUS-CQI at thresholds {0.5, 0.8, 0.9} + CQI-greedy
sanity anchor, on MixedLoad_L2's eval setup (seeds 10000-10002, 1000 slots).
Reproducing the logged CSV numbers for th=0.5 / CQI-greedy validates the
pipeline; th=0.8/0.9 quantify the bar PPO must beat.
"""
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

cj = json.load(open("/home/MYH/ML_DRL_Scheduler/Run3/MixedLoad_L2/config.json"))
fields = {}
for k, v in cj.items():
    if k in Config.__dataclass_fields__:
        fields[k] = tuple(v) if isinstance(v, list) else v
cfg = Config(**fields)
assert cfg.episode_len == 1000, cfg.episode_len
env = SchedulerEnv(cfg)

jobs = [("CQI-greedy(anchor)", CQIGreedy(), None),
        ("SUS-CQI@0.5(anchor)", SUSCQI(), 0.5),
        ("SUS-CQI@0.8", SUSCQI(), 0.8),
        ("SUS-CQI@0.9", SUSCQI(), 0.9)]

t0 = time.time()
print(f"eval protocol: {cfg.ppo_eval_episodes} episodes, seeds 10000+, "
      f"episode_len={cfg.episode_len}, K={cfg.num_ue}", flush=True)
results = {}
for name, sched, th in jobs:
    cfg.sus_ortho_threshold = th if th is not None else 0.5
    rews, thrs, comps, drops, depths = [], [], [], [], []
    for ep_idx in range(cfg.ppo_eval_episodes):
        env.reset(ep_idx + 10000)
        done = False
        while not done:
            alloc = sched.schedule(env)
            _, _, done, _ = env.step(alloc)
        m = env_episode_metrics(env, cfg)
        rews.append(m["reward"]); thrs.append(m["throughput_mbps"])
        comps.append(m["completion_rate"]); drops.append(m["retx_drop_rate"])
        depths.append(m["mu_depth"])
        print(f"  [{name}] ep{ep_idx}: rew {m['reward']:.0f} "
              f"thr {m['throughput_mbps']:.2f} comp {m['completion_rate']:.3f} "
              f"depth {m['mu_depth']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    results[name] = (np.mean(rews), np.mean(thrs), np.mean(comps),
                     np.mean(drops), np.mean(depths))

print("\n=== SUMMARY (mean over 3 eval episodes) ===")
print(f"{'scheduler':22s} {'reward':>8s} {'thr_mbps':>9s} {'comp':>6s} "
      f"{'drop':>6s} {'mu_depth':>8s}")
for name, (r, t, c, d, dep) in results.items():
    print(f"{name:22s} {r:8.1f} {t:9.2f} {c:6.3f} {d:6.3f} {dep:8.2f}")
print("\nCSV anchors (from Run3/MixedLoad_L2 eval at update 9): "
      "SUS-CQI 8789.6 / 80.73 / 0.816 | CQI-greedy 8517.6 / 79.12 / 0.807 "
      "| PPO@139 8349.7 / 83.05 / 0.791")
