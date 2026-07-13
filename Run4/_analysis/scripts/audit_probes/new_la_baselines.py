"""New-LA world baseline evaluation: post_rzf + rbg_major + la_beta=0.6469.

Same 8 seeds as the legacy control run (paired). Reports the standard metric
set + the new HARQ instruments (first-ACK rate incl. per-depth, attempts per
ACKed unit, pinned fraction, goodput).
"""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from baselines import all_baselines
from train_phase2 import env_episode_metrics

SEEDS = list(range(10000, 10008))
cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          la_beta=0.6469,
                          p_arrival_min=0.15, p_arrival_max=0.40)
env = SchedulerEnv(cfg)
rows = {}
for sch in all_baselines(cfg):
    ms = []
    for s in SEEDS:
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        ms.append(env_episode_metrics(env, cfg))
    agg = {k: float(np.mean([m[k] for m in ms]))
           for k in ms[0] if isinstance(ms[0][k], (int, float))}
    rows[sch.name] = agg
    print(f"{sch.name:16s} rew {agg['reward']:8.1f}  "
          f"thr {agg['throughput_mbps']:6.2f}  comp {agg['completion_rate']:.3f}  "
          f"miss {agg['deadline_miss_rate']:.3f}  "
          f"rxd {agg['retx_drop_rate']:.4f}  depth {agg['mu_depth']:.2f}  "
          f"1ACK {agg['first_ack_rate']:.3f}  "
          f"att {agg['attempts_per_acked']:.2f}  "
          f"pin {agg['pinned_fraction']:.3f}  "
          f"good {agg['goodput_mbps']:.2f}", flush=True)

print("\nper-depth first-ACK (mean over seeds):")
for name, agg in rows.items():
    print(f"{name:16s} " + " ".join(
        f"m{m}:{agg[f'first_ack_m{m}']:.2f}" for m in (1, 2, 3, 4)))

import csv, os
out = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/new_la_baselines.csv"
keys = ["reward", "throughput_mbps", "completion_rate", "deadline_miss_rate",
        "retx_drop_rate", "mu_depth", "jain", "first_ack_rate",
        "attempts_per_acked", "pinned_fraction", "goodput_mbps",
        "first_ack_m1", "first_ack_m2", "first_ack_m3", "first_ack_m4"]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["baseline"] + keys)
    for name, agg in rows.items():
        w.writerow([name] + [f"{agg[k]:.4f}" for k in keys])
print(f"saved {out}")
