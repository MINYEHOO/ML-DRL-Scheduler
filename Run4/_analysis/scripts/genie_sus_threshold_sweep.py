"""SUS threshold sweep in the GenieFineTune (perfect-CSI) world.

The old L2b figure's "SUS threshold 0.8" was an EVAL-TIME choice from a
threshold screen (sus_family_08_screen.py); every run's training config
carries the 0.5 default. Before trusting the GenieFineTune figure's
SUS bars at 0.5, screen SUS+CQI at 0.5..0.9 on the same 20 seeds --
if a stronger setting exists, the figure baselines must use it.
Seed-outer / threshold-inner loop keeps the per-seed CSI cache hot.
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

OUT = "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/genie_sus_threshold_sweep.csv"
SEEDS = list(range(10000, 10020))
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

raw = json.load(open("Run4/GenieFineTune/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = tuple(raw.get("la_beta_by_depth") or ())
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
cfg = Config(**raw)
assert cfg.pmi_mode == "genie" and cfg.p_csi == 1.0
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

print("\n=== SUS+CQI by threshold (genie world, 20-seed means) ===")
for t in THRESHOLDS:
    rr = [m["reward"] for m in res[t]]
    print(f"thr {t}: reward {np.mean(rr):8.1f}  depth "
          f"{np.mean([m['mu_depth'] for m in res[t]]):.3f}  retx "
          f"{np.mean([m['retx_drop_rate'] for m in res[t]]):.4f}  miss "
          f"{np.mean([m['deadline_miss_rate'] for m in res[t]]):.4f}", flush=True)
best = max(THRESHOLDS, key=lambda t: np.mean([m["reward"] for m in res[t]]))
print(f"\nstrongest threshold: {best}")
print("saved ->", OUT)
