"""8-seed baseline grid for the L2c NO-BETA world (hetero, speed U(5,40)).

Same pooled-rate methodology as new_la_baselines.py; world read from the
live run's config.json so it is exactly the training world.
"""
import csv, json
import numpy as np
from config import Config
from env import SchedulerEnv
from baselines import all_baselines
from train_phase2 import env_episode_metrics

raw = json.load(open("Run4/MixedSpeed_L2c/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.la_beta == 1.0 and not cfg.la_beta_by_depth
env = SchedulerEnv(cfg)
SEEDS = list(range(10000, 10008))
KEYS = ["reward", "throughput_mbps", "goodput_mbps", "completion_rate",
        "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain",
        "first_ack_rate", "attempts_per_acked"]
w = csv.writer(open("Run4/_analysis/l2c_world_baselines.csv", "w", newline=""))
w.writerow(["baseline", "seed"] + KEYS)
for sch in all_baselines(cfg):
    ms = []
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        em = env_episode_metrics(env, cfg)
        ms.append(em)
        w.writerow([sch.name, s] + [round(float(em[k]), 4) for k in KEYS])
    print(f"{sch.name:16s} rew {np.mean([m['reward'] for m in ms]):8.1f}  "
          f"miss {np.mean([m['deadline_miss_rate'] for m in ms]):.3f}  "
          f"1ACK {np.mean([m['first_ack_rate'] for m in ms]):.3f}  "
          f"att {np.mean([m['attempts_per_acked'] for m in ms]):.3f}", flush=True)
print("saved")
