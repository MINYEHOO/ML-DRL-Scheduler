"""beta_m recalibration for the L2c world (speed U(5,40); 2026-07-16).

Same scheduler-independent methodology as beta_m_calib_holdout.py (which
calibrated the queue/U(5,30) world): random (RBG, m, group) samples over
empty-allocation episodes; beta_m = per-depth 10th pct of MI_actual/cap_pred;
holdout on disjoint episodes with the DEPLOYED 4-decimal values.
World: hetero one-packet, K=32, n_active U{16..32}, speed U(5,40), p_csi 0.6.
"""
import numpy as np
from config import phase2_hetero_config
from env import SchedulerEnv
from beta_m_calib_holdout import collect_ratios_by_episode, wilson, \
    cluster_bootstrap_ci

cfg = phase2_hetero_config(num_ue=32, n_active_min=16, n_active_max=32,
                           ue_speed_min=5.0, ue_speed_max=40.0,
                           la_mode="post_rzf", decode_order="rbg_major")
env = SchedulerEnv(cfg)
print(f"world: K={cfg.num_ue} speed U({cfg.ue_speed_min},{cfg.ue_speed_max}) "
      f"p_csi={cfg.p_csi} pmi={cfg.pmi_mode}", flush=True)

cal = collect_ratios_by_episode(env, cfg, 50000, 36, 777)
beta_raw = [float(np.percentile(np.concatenate(cal[m]), 10))
            for m in (1, 2, 3, 4)]
beta = tuple(round(b, 4) for b in beta_raw)
print("beta_raw:", " ".join(f"{b:.6f}" for b in beta_raw))
print("DEPLOYED beta_m(L2c):", beta, flush=True)

hold = collect_ratios_by_episode(env, cfg, 70000, 12, 20250713)
for m in (1, 2, 3, 4):
    acks = [int((v >= beta[m-1]).sum()) for v in hold[m]]
    ns = [len(v) for v in hold[m]]
    k, nn = sum(acks), sum(ns)
    lo, hi = cluster_bootstrap_ci(acks, ns)
    print(f"m={m} beta={beta[m-1]:.4f} n={nn} first-ACK={k/nn:.4f} "
          f"cluster95 [{lo:.4f},{hi:.4f}]")
