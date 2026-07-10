"""Reviewer-defense: can a SWITCH heuristic replicate PPO's adaptation?

On MixedLoad_L2 and MixedSpeed_L2 eval protocols (seeds 10000-10002):
  - pure SU-CQI, pure SUS-CQI@0.8
  - HybridSwitch(T): per-slot, if currently-backlogged UE count <= T use
    SU-CQI behavior, else SUS-CQI@0.8  (T swept -- the strongest realizable
    observable-signal switch)
  - oracle envelope: per-seed max(pure SU, pure SUS) = a PERFECT per-episode
    switch (unrealizable upper bound for any 2-mode switch heuristic)
  - PPO best.pt (MixedLoad: own best; MixedSpeed: L2 best.pt = update 29)
Seed-major loop for channel-cache reuse.
"""
import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json                   # noqa: E402
import sys                    # noqa: E402
import time                   # noqa: E402

import numpy as np            # noqa: E402
import torch                  # noqa: E402

torch.set_num_threads(6)
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
from config import Config                              # noqa: E402
from env import SchedulerEnv                           # noqa: E402
from policy import ActorCritic                         # noqa: E402
from baselines import SUCQI, SUSCQI                    # noqa: E402
from train_phase2 import env_episode_metrics, PPOScheduler   # noqa: E402


class HybridSwitch:
    """SU-CQI when few UEs have data, SUS-CQI(0.8) when many (per-slot)."""

    def __init__(self, T):
        self.T = T
        self.name = f"Hybrid<= {T}"
        self.su = SUCQI()
        self.sus = SUSCQI()

    def schedule(self, env):
        n = int(env.get_observation()["active"].sum())
        return (self.su if n <= self.T else self.sus).schedule(env)


SEEDS = [10000, 10001, 10002]
JOBS = [("MixedLoad_L2", "Run3/MixedLoad_L2/ckpt/best.pt"),
        ("MixedSpeed_L2", "Run3/MixedSpeed_L2/ckpt/best.pt")]
T_SWEEP = (10, 14, 18, 22)

t0 = time.time()
for run, ckpt_rel in JOBS:
    cj = json.load(open(f"/home/MYH/ML_DRL_Scheduler/Run3/{run}/config.json"))
    fields = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in cj.items() if k in Config.__dataclass_fields__}
    cfg = Config(**fields)
    cfg.sus_ortho_threshold = 0.8          # all SUS uses in this probe
    env = SchedulerEnv(cfg)
    ac = ActorCritic(cfg)
    ck = torch.load(f"/home/MYH/ML_DRL_Scheduler/{ckpt_rel}",
                    map_location="cpu")
    ac.load_state_dict(ck["model"])
    ac.eval()
    scheds = ([("SU-CQI", SUCQI()), ("SUS-CQI@0.8", SUSCQI())]
              + [(h.name, h) for h in (HybridSwitch(T) for T in T_SWEEP)]
              + [("PPO-best", PPOScheduler(ac, deterministic=True))])
    per = {name: [] for name, _ in scheds}
    for seed in SEEDS:
        for name, sched in scheds:
            env.reset(seed)
            done = False
            while not done:
                _, _, done, _ = env.step(sched.schedule(env))
            m = env_episode_metrics(env, cfg)
            per[name].append(m["reward"])
            print(f"[{run}] seed {seed} {name:12s} {m['reward']:8.1f}  "
                  f"depth {m['mu_depth']:.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"\n=== {run} (PPO ckpt upd {ck['update']}, "
          f"eval {ck.get('eval_reward'):.0f}) ===")
    for name, r in per.items():
        print(f"  {name:14s} mean {np.mean(r):8.1f}  per-seed "
              f"{[f'{x:.0f}' for x in r]}")
    env_mx = [max(per["SU-CQI"][i], per["SUS-CQI@0.8"][i])
              for i in range(len(SEEDS))]
    best_hy = max(((n, np.mean(r)) for n, r in per.items()
                   if n.startswith("Hybrid")), key=lambda x: x[1])
    print(f"  {'oracle-envelope':14s} mean {np.mean(env_mx):8.1f}  "
          f"(per-episode perfect switch)")
    print(f"  best hybrid = {best_hy[0]} ({best_hy[1]:.1f});  "
          f"PPO {np.mean(per['PPO-best']):.1f}  -> PPO vs best-hybrid "
          f"{np.mean(per['PPO-best'])/best_hy[1]-1:+.2%}, vs envelope "
          f"{np.mean(per['PPO-best'])/np.mean(env_mx)-1:+.2%}")
