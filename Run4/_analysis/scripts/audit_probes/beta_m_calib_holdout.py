"""Depth-wise beta_m calibration + holdout validation (audit rounds 6-8).

Fully self-reproducing: running this file recomputes, in order,
  (1) calibration       episodes 50000-50035 (36 eps), rng 777
  (2) beta_m            per-depth 10th percentile of MI_actual/cap_pred
  (3) holdout           episodes 70000-70011 (12 eps), rng 20250713
                        (fully disjoint; beta_m is FROZEN, never retuned here)
  (4) member-level Wilson 95% CIs (reported for reference only)
  (5) episode-cluster bootstrap 95% CIs (B=10,000, seed 42) -- the honest
      uncertainty: samples within an episode share its channel mixture
      (speeds, n_active), so member-level CIs overstate precision
      (about 1.2x for m=1, 3.5-4.4x for m=2..4).

Sampling is scheduler-independent: episodes advance with EMPTY allocations,
groups are uniform random (RBG, m, UE set) draws. All sampled units are
full-cap by construction (no backlog in the sampler), so
first-ACK  <=>  MI_actual >= beta_m * cap_pred  <=>  ratio >= beta_m.

Reference results (2026-07-13): beta_m = (0.9815, 0.7306, 0.6466, 0.5922);
holdout pooled first-ACK m1 88.8% / m2 93.5% / m3 94.2% / m4 94.8% with
cluster CIs ~[87.5,90.0] / [91.3,95.2] / [92.2,95.8] / [93.1,96.2].

Run from the repo root:
  python3 Run4/_analysis/scripts/audit_probes/beta_m_calib_holdout.py
"""
import numpy as np

from config import phase4_queue_config
from env import SchedulerEnv
from la_planner import predict_group_link_adaptation
from phy import _rbg_sinr, mi_bits


def collect_ratios_by_episode(env, cfg, ep0, n_ep, seed):
    """Per-episode ratio lists: out[m] = list over episodes of 1-D arrays."""
    rng = np.random.default_rng(seed)
    empty = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    out = {m: [] for m in (1, 2, 3, 4)}
    for ep in range(n_ep):
        cur = {m: [] for m in (1, 2, 3, 4)}
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
                        cur[m].extend((mi / np.maximum(cap, 1e-12)).tolist())
            env.step(empty)
        for m in (1, 2, 3, 4):
            out[m].append(np.asarray(cur[m]))
    return out


def wilson(k, nn, z=1.96):
    p = k / nn
    d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d
    h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return c - h, c + h


def cluster_bootstrap_ci(ack_by_ep, n_by_ep, B=10000, seed=42):
    """(lo, hi) of pooled sum(ACK)/sum(units) resampling EPISODES w/ repl."""
    rng = np.random.default_rng(seed)
    n_ep = len(ack_by_ep)
    idx = rng.integers(0, n_ep, size=(B, n_ep))
    boots = (np.asarray(ack_by_ep, dtype=float)[idx].sum(axis=1)
             / np.asarray(n_by_ep, dtype=float)[idx].sum(axis=1))
    return tuple(np.percentile(boots, [2.5, 97.5]))


if __name__ == "__main__":
    cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                              p_arrival_min=0.15, p_arrival_max=0.40)
    env = SchedulerEnv(cfg)

    print("(1) calibration: episodes 50000-50035, rng 777 ...", flush=True)
    cal = collect_ratios_by_episode(env, cfg, 50000, 36, 777)
    beta_raw = [float(np.percentile(np.concatenate(cal[m]), 10))
                for m in (1, 2, 3, 4)]
    # the DEPLOYED backoff is the 4-decimal literal the trainings actually
    # use (config la_beta_by_depth); holdout must be judged against IT, not
    # the full-precision percentile (audit round 8)
    BETA_DEPLOYED = (0.9815, 0.7306, 0.6466, 0.5922)
    print("(2) beta_raw =", " ".join(f"{b:.6f}" for b in beta_raw),
          "| deployed =", BETA_DEPLOYED, flush=True)
    assert np.allclose(np.round(beta_raw, 4), BETA_DEPLOYED), \
        (beta_raw, BETA_DEPLOYED)
    beta = list(BETA_DEPLOYED)

    print("(3) holdout: episodes 70000-70011, rng 20250713 "
          "(deployed beta_m, frozen) ...", flush=True)
    hold = collect_ratios_by_episode(env, cfg, 70000, 12, 20250713)

    print(f"{'m':>2s} {'beta_m':>7s} {'n':>6s} {'firstACK':>8s} "
          f"{'Wilson 95% (ref)':>18s} {'cluster 95% CI':>18s} "
          f"{'per-ep range':>13s}")
    for m in (1, 2, 3, 4):
        acks = [int((v >= beta[m - 1]).sum()) for v in hold[m]]
        ns = [len(v) for v in hold[m]]
        k, nn = sum(acks), sum(ns)
        wlo, whi = wilson(k, nn)
        clo, chi = cluster_bootstrap_ci(acks, ns)
        rates = [a / x for a, x in zip(acks, ns)]
        print(f"{m:2d} {beta[m-1]:7.4f} {nn:6d} {k/nn:8.4f} "
              f"[{wlo:.4f}, {whi:.4f}] [{clo:.4f}, {chi:.4f}] "
              f"{min(rates):.3f}..{max(rates):.3f}")
    acks = [sum(int((hold[m][e] >= beta[m - 1]).sum()) for m in (1, 2, 3, 4))
            for e in range(12)]
    ns = [sum(len(hold[m][e]) for m in (1, 2, 3, 4)) for e in range(12)]
    clo, chi = cluster_bootstrap_ci(acks, ns)
    print(f"ALL          {sum(ns):6d} {sum(acks)/sum(ns):8.4f} "
          f"cluster [{clo:.4f}, {chi:.4f}]")
