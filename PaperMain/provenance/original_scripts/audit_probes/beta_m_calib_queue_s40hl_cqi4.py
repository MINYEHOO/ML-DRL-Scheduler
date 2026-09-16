"""beta_m recalibration for the S40HighLoad world + 4-bit NR CQI (2026-08-01).

Frozen recipe (beta_m_calib_holdout.py): random (RBG, m, group) samples over
empty-allocation episodes; beta_m = per-depth 10th pct of MI_actual/cap_pred;
holdout on disjoint episodes with the DEPLOYED 4-decimal values.

World = QueuePostRZF_S40HighLoad config.json with cqi_mode='nr4bit' and beta
cleared (raw-prediction ratios). Originally run from the cqi4 worktree, which
carried cqi_mode before it existed on main; that branch is merged (1d85999) and
the worktree removed, so the import below now resolves to the main tree.
For the fresh CQI4 run (user directive 2026-08-01).
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "8"
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler/Run4/_analysis/scripts/audit_probes")
sys.path.insert(0, "/home/MYH/ML_DRL_Scheduler")   # cqi4 merged into main (1d85999)
import numpy as np
from config import Config
from env import SchedulerEnv
from beta_m_calib_holdout import collect_ratios_by_episode, wilson, \
    cluster_bootstrap_ci

raw = json.load(open(
    "/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HighLoad/config.json"))
for k in ("git_hash", "git_dirty_py"):
    raw.pop(k, None)
raw["la_beta_by_depth"] = ()          # raw predictions for ratio collection
raw["la_beta"] = 1.0
raw["ue_speed_mix"] = tuple(raw.get("ue_speed_mix") or ())
raw["cqi_mode"] = "nr4bit"
cfg = Config(**raw)
env = SchedulerEnv(cfg)
print(f"world: K={cfg.num_ue} n_active U({cfg.n_active_min},{cfg.n_active_max}) "
      f"speed U({cfg.ue_speed_min},{cfg.ue_speed_max}) p U({cfg.p_arrival_min},"
      f"{cfg.p_arrival_max}) p_csi={cfg.p_csi} pmi={cfg.pmi_mode} "
      f"cqi={cfg.cqi_mode}", flush=True)

cal = collect_ratios_by_episode(env, cfg, 50000, 36, 777)
beta_raw = [float(np.percentile(np.concatenate(cal[m]), 10))
            for m in (1, 2, 3, 4)]
beta = tuple(round(b, 4) for b in beta_raw)
print("beta_raw:", " ".join(f"{b:.6f}" for b in beta_raw))
print("DEPLOYED beta_m(S40HL cqi4):", beta, flush=True)

hold = collect_ratios_by_episode(env, cfg, 70000, 12, 20250713)
for m in (1, 2, 3, 4):
    acks = [int((v >= beta[m-1]).sum()) for v in hold[m]]
    ns = [len(v) for v in hold[m]]
    k, nn = sum(acks), sum(ns)
    lo, hi = cluster_bootstrap_ci(acks, ns)
    print(f"m={m} beta={beta[m-1]:.4f} n={nn} first-ACK={k/nn:.4f} "
          f"cluster95 [{lo:.4f},{hi:.4f}]")
