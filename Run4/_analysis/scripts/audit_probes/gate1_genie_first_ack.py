"""Gate 1 (full): genie-world post_rzf first-ACK must be 100% at EVERY depth.

In the genie world (pmi_mode="genie", p_csi=1.0) h_hat == h_true, so the
planner's predicted post-RZF SINR equals the realized SINR and every new
full-cap unit must deliver >= B_tx in its first slot; backlog-capped units
send less than the cap and must also first-ACK. This is a *simulator
PHY-chain consistency gate* (prediction vs realization inside the same PHY
implementation), complemented by the independent closed-form checks in
independent_reference_test.py.

Coverage requirements (audit round 8): new units only; EVERY depth bin
m = 1..4 must be non-empty (asserted); full-cap and backlog-cap counted
separately. Schedulers chosen to force the depth spread: CQI-greedy (blind
MU, stacks depth 4), SUS+CQI (gated MU), SU+CQI (pure depth 1).

Run from the repo root:
  python3 Run4/_analysis/scripts/audit_probes/gate1_genie_first_ack.py
"""
import numpy as np

from config import phase2_hetero_config
from env import SchedulerEnv
from baselines import SUSCQI, CQIGreedy, SUCQI

stats = {"full_cap": [0, 0], "backlog_cap": [0, 0]}   # [first_ack, total]
depth_stats = {}

cfg = phase2_hetero_config(pmi_mode="genie", p_csi=1.0, num_ue=32,
                           n_active_min=16, n_active_max=32,
                           ue_speed_min=5.0, ue_speed_max=30.0,
                           la_mode="post_rzf", decode_order="rbg_major")
# genie keeps la_beta = 1.0 / la_beta_by_depth unset: prediction is exact.
assert cfg.la_beta == 1.0 and not cfg.la_beta_by_depth
env = SchedulerEnv(cfg)

for seed, sch in ((10000, CQIGreedy()), (10004, SUSCQI()), (10008, SUCQI())):
    env.reset(episode_idx=seed)
    for t in range(300):
        alloc = sch.schedule(env)
        env.step(alloc)
        for (r, u), b in env.last_actual_btx.items():
            # unit still alive with 1 attempt <=> first transmission NACKed
            alive = [x for x in env.txmgr.units
                     if x.rbg_id == r and x.ue_id == u and x.tx_attempts == 1]
            m = int(env.fixed_mask[r].sum()
                    + sum(1 for (rr, _u) in env.last_actual_btx if rr == r))
            cap = (cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                   * np.log2(1 + env.last_env_planner.sinr_pred[(r, u)]))
            key = "backlog_cap" if b < 0.999 * cap else "full_cap"
            first_ack = (len(alive) == 0)
            stats[key][1] += 1
            stats[key][0] += first_ack
            d = depth_stats.setdefault((key, min(m, 4)), [0, 0])
            d[1] += 1
            d[0] += first_ack

print("=== Gate 1a: scheduler-driven (realistic allocations) ===")
for key, (a, t) in stats.items():
    print(f"{key:12s}: first-ACK {a}/{t} = {a / max(t, 1) * 100:.2f}%")
print("per-depth (full-cap | backlog-cap):")
for m in (1, 2, 3, 4):
    fa, ft = depth_stats.get(("full_cap", m), (0, 0))
    ba, bt = depth_stats.get(("backlog_cap", m), (0, 0))
    print(f"  m={m}: full {fa}/{ft}  backlog {ba}/{bt}")
    assert ft + bt > 0, f"depth m={m} has ZERO new-unit samples"
assert stats["full_cap"][0] == stats["full_cap"][1], "full-cap first-ACK <100%"
assert stats["backlog_cap"][0] == stats["backlog_cap"][1], \
    "backlog-cap first-ACK <100%"

# --- Gate 1b: FORCED depth coverage (audit round 8) -----------------------
# Schedulers rarely build depth-2/3 groups (SUS saturates at 4, SU stays 1),
# so 1a leaves m=2/3 thin. Force exact-depth allocations directly (env.step
# accepts any [R, L] matrix): >=100 new units per depth, and additionally
# assert the planner's SINR prediction EQUALS the realized post-RZF SINR
# (genie: same h for prediction and transmission).
from phy import _rbg_sinr

forced = {m: [0, 0] for m in (1, 2, 3, 4)}       # [first_ack, total]
sinr_err_max = 0.0
env.reset(episode_idx=10012)
K = cfg.num_ue
t = 0
while min(v[1] for v in forced.values()) < 100:
    t += 1
    assert t < 2000, f"forced coverage did not reach 100/depth: {forced}"
    m_t = ((t - 1) % 4) + 1                      # cycle target depth 1..4
    alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
    for r in range(cfg.num_rbg):
        for j in range(m_t):
            alloc[r, j] = ((m_t * r + j + 7 * t) % K) + 1   # rotate UEs
    # capture the TRANSMISSION slot's channels BEFORE step: step() advances
    # h_true/h_hat_slot to the next slot before returning
    h_true_tx = env.h_true_slot.copy()
    h_hat_tx = env.h_hat_slot.copy()
    nv_tx = env.noise_var
    fixed_tx = [sorted(set(s)) for s in env.initial_S_r]
    env.step(alloc)
    pl = env.last_env_planner
    newk = {}
    for (r, u) in env.last_actual_btx:
        newk.setdefault(r, []).append(u)
    for r, us in newk.items():
        m = len(fixed_tx[r]) + len(us)
        if m > 4:
            m = 4
        grp = fixed_tx[r] + us                   # canonical planner order
        # realized post-RZF SINR from the same (genie) tx-slot channels
        sinr_act = _rbg_sinr(h_true_tx[grp, r, :], h_hat_tx[grp, r, :],
                             nv_tx, cfg.p_rbg, nv_tx)
        for i, u in enumerate(grp):
            if (r, u) not in pl.sinr_pred or u not in us:
                continue
            err = abs(pl.sinr_pred[(r, u)] - sinr_act[i]) \
                / max(sinr_act[i], 1e-12)
            sinr_err_max = max(sinr_err_max, err)
        for u in us:
            alive = [x for x in env.txmgr.units
                     if x.rbg_id == r and x.ue_id == u and x.tx_attempts == 1]
            forced[m][1] += 1
            forced[m][0] += (len(alive) == 0)

print("=== Gate 1b: forced exact-depth coverage ===")
for m in (1, 2, 3, 4):
    a, tt = forced[m]
    print(f"  m={m}: first-ACK {a}/{tt} = {a / max(tt, 1) * 100:.2f}%")
    assert tt >= 100, f"depth m={m} forced samples {tt} < 100"
    assert a == tt, f"depth m={m} first-ACK below 100%"
print(f"  max rel |SINR_pred - SINR_actual| = {sinr_err_max:.3e}")
assert sinr_err_max < 1e-9, "genie prediction != realization"
print("GATE 1 PASSED: 100% first-ACK, >=100 units per depth, "
      "SINR_pred == SINR_actual")
