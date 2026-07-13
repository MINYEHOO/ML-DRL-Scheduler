"""Depth-wise beta_m calibration + holdout validation (audit round 6).

Calibration: episodes 50000-50035 (36 eps), rng 777 -- scheduler-independent
random (RBG, m, UE-group) samples over episodes advanced with EMPTY
allocations; beta_m = per-depth 10th percentile of MI_actual/cap_pred.
Holdout:     episodes 70000-70011 (12 eps), rng 20250713 -- fully disjoint;
first-ACK(m) = P(ratio >= beta_m), reported with Wilson 95% CIs.

Result (2026-07-13): beta_m = (0.9815, 0.7306, 0.6466, 0.5922)
holdout first-ACK: m1 88.8% [87.7,89.8] | m2 93.5% | m3 94.2% | m4 94.8%
-- m>=2 lands ~4pp conservative of the 90% target: the deviation is an
episode-cluster effect (per-episode mixtures differ; member-level CIs
understate uncertainty), direction is conservative (MU sends slightly fewer
bits), applied identically to every scheduler. NOT retuned on holdout.
"""
import numpy as np
from config import phase4_queue_config
from env import SchedulerEnv
from la_planner import predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits

cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          p_arrival_min=0.15, p_arrival_max=0.40)
env = SchedulerEnv(cfg)
empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)


def collect(ep0, n_ep, seed):
    rng = np.random.default_rng(seed)
    ratios = {m: [] for m in (1, 2, 3, 4)}
    for ep in range(n_ep):
        env.reset(episode_idx=ep0 + ep)
        for t in range(cfg.episode_len):
            if t % 7 == 0:
                nv = env.noise_var
                for m in (1, 2, 3, 4):
                    for _ in range(2):
                        r = int(rng.integers(cfg.num_rbg))
                        grp = rng.choice(cfg.num_ue, size=m, replace=False)
                        _, sp, _ = predict_group_link_adaptation(
                            env.h_hat_slot[grp, r, :], nv, nv, cfg.p_rbg, cfg)
                        cap = (cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                               * np.log2(1 + sp))
                        mi = mi_bits(_rbg_sinr(env.h_true_slot[grp, r, :],
                                               env.h_hat_slot[grp, r, :],
                                               nv, cfg.p_rbg, nv), cfg)
                        ratios[m].extend((mi / np.maximum(cap, 1e-12)).tolist())
            env.step(empty)
    return ratios


def wilson(k, nn, z=1.96):
    p = k / nn
    d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d
    h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return c - h, c + h


if __name__ == "__main__":
    cal = collect(50000, 36, 777)
    BETA = [float(np.percentile(np.array(cal[m]), 10)) for m in (1, 2, 3, 4)]
    print("beta_m:", " ".join(f"{b:.4f}" for b in BETA))
    hold = collect(70000, 12, 20250713)
    for m in (1, 2, 3, 4):
        v = np.array(hold[m])
        k = int((v >= BETA[m - 1]).sum())
        lo, hi = wilson(k, len(v))
        print(f"m={m} beta={BETA[m-1]:.4f} n={len(v)} "
              f"first-ACK={k/len(v):.4f} CI[{lo:.4f},{hi:.4f}]")


# --- episode-cluster bootstrap CI (audit round 7) -------------------------
# Samples within an episode share its channel mixture (speeds, n_active), so
# member-level binomial CIs overstate precision. Resample the holdout
# EPISODES with replacement (B=10,000), pooled sum(ACK)/sum(units) per draw.
# Result (2026-07-13): m1 88.8% [87.5,90.0] | m2 93.5% [91.3,95.2]
# | m3 94.2% [92.2,95.8] | m4 94.8% [93.1,96.2]; per-episode 0.83..0.98.
# beta_m stays frozen -- this CI is uncertainty reporting, not retuning.
def cluster_bootstrap(ack_by_ep, n_by_ep, B=10000, seed=42):
    """ack_by_ep, n_by_ep: [n_episodes, 4] arrays -> per-depth (lo, hi)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ack_by_ep.shape[0], size=(B, ack_by_ep.shape[0]))
    out = []
    for m in range(4):
        boots = (ack_by_ep[idx, m].sum(axis=1)
                 / n_by_ep[idx, m].sum(axis=1))
        out.append(tuple(np.percentile(boots, [2.5, 97.5])))
    return out
