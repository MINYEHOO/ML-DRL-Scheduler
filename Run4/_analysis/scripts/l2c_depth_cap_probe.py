"""Is the L2c world's optimal MU depth interior (<4)?

Evaluate SUS+CQI with a hard per-RBG depth cap in {1,2,3,4} on the L2c
no-beta world (8 paired seeds, same protocol as l2c_world_baselines.py).
Motivation: freshly-trained L2c PPO saturates depth at 3.99 (== uncapped
SUS+CQI) yet the L2b donor's shallow 2.7 policy outscores both -- if a
depth-capped heuristic also beats the uncapped one, the world's optimum
is interior and the trained policy is stuck at the depth-saturated local
optimum (weak per-depth first-ACK gradient: m2 0.46 -> m4 0.41).
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")
import numpy as np
from config import Config
from env import SchedulerEnv
from baselines import SUSCQI
from train_phase2 import env_episode_metrics

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/l2c_depth_cap_probe.csv"
SEEDS = list(range(10000, 10008))

raw = json.load(open("Run4/MixedSpeed_L2c/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
env = SchedulerEnv(cfg)

def capped_sus_cqi(cap):
    class CappedSUSCQI(SUSCQI):
        name = f"SUS+CQI-cap{cap}"
        def _select(self, e, obs, r, l, cand, sel_ue, closed):
            if len(sel_ue) >= cap:
                closed[r] = True
                return None
            return super()._select(e, obs, r, l, cand, sel_ue, closed)
    return CappedSUSCQI()

KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain",
        "first_ack_rate", "attempts_per_acked"]
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["baseline", "seed"] + KEYS)
for cap in (1, 2, 3, 4):
    sch = capped_sus_cqi(cap)
    ms = []
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        em = env_episode_metrics(env, cfg)
        ms.append(em)
        w.writerow([sch.name, s] + [round(float(em[k]), 4) for k in KEYS])
    print(f"cap {cap}: rew {np.mean([m['reward'] for m in ms]):7.1f}  "
          f"depth {np.mean([m['mu_depth'] for m in ms]):.3f}  "
          f"miss {np.mean([m['deadline_miss_rate'] for m in ms]):.4f}  "
          f"1ACK {np.mean([m['first_ack_rate'] for m in ms]):.3f}  "
          f"goodput {np.mean([m['goodput_mbps'] for m in ms]):.2f}", flush=True)
print("saved ->", OUT)
