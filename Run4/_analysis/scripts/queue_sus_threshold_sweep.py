"""SUS+CQI threshold sweep in the QueuePostRZF beta_m world (20 seeds).

Per-world re-sweep before the Ent02 final figure (lesson from the genie
figure: 0.7 beat both the 0.5 default and L2b's 0.8 there). Seeds
10000-10019; seed-outer loop keeps the CSI cache hot.
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

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/queue_sus_threshold_sweep.csv"
SEEDS = list(range(10000, 10020))
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)

raw = json.load(open("Run4/QueuePostRZF_Ent02/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.resolved_la_mode() == "post_rzf" and cfg.la_beta_by_depth
print(f"world: queue beta_m, config threshold = {cfg.sus_ortho_threshold}", flush=True)
env = SchedulerEnv(cfg)
sch = SUSCQI()

res = {t: [] for t in THRESHOLDS}
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["threshold", "seed", "reward", "throughput_mbps", "completion_rate",
            "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain"])
for s in SEEDS:
    for t in THRESHOLDS:
        env.cfg.sus_ortho_threshold = t
        env.reset(episode_idx=s)
        done = False
        while not done:
            _, _, done, _ = env.step(sch.schedule(env))
        m = env_episode_metrics(env, cfg)
        res[t].append(m)
        w.writerow([t, s] + [round(float(m[k]), 4) for k in
                   ("reward", "throughput_mbps", "completion_rate",
                    "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain")])
    print(f"seed {s}: " + "  ".join(
        f"{t}:{res[t][-1]['reward']:.0f}" for t in THRESHOLDS), flush=True)

print("\n=== SUS+CQI by threshold (queue beta_m world, 20-seed means) ===")
for t in THRESHOLDS:
    print(f"thr {t}: reward {np.mean([m['reward'] for m in res[t]]):8.1f}  "
          f"depth {np.mean([m['mu_depth'] for m in res[t]]):.3f}  "
          f"miss {np.mean([m['deadline_miss_rate'] for m in res[t]]):.4f}", flush=True)
best = max(THRESHOLDS, key=lambda t: np.mean([m["reward"] for m in res[t]]))
print(f"\nstrongest threshold: {best}")
print("saved ->", OUT)
