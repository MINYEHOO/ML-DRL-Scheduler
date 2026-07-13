"""Independent analytic reference for the post_rzf planner (audit round 6).

The Gate-2/3 zero-error results share the SlotAllocationPlanner code path, so
a bug common to scheduler and env would be invisible there. This test checks
the planner's math core against CLOSED-FORM hand-derivable answers that never
touch the planner code:

(A) Orthogonal groups: h_i = g_i * e_i (scaled canonical basis rows).
    RZF: H H^H = diag(g_i^2)  ->  V columns  prop to  e_i  ->  unit-norm
    columns = e_i, per-stream power sqrt(P/m). Zero cross-interference, so
        SINR_i = g_i^2 * P / (m * sigma^2)      (exactly)
        cap_i  = beta_m * eta * N_RE * log2(1 + SINR_i)
(B) Budget accounting / epsilon-drop hand trace on a 2-UE group where UE B's
    remaining backlog is sub-epsilon: B must be ghost-dropped and A re-sized
    as an m=1 group; A's second RBG must be sized with A's exactly-debited
    remainder and swallow the sub-epsilon tail.

Run from the repo root:  python3 Run4/_analysis/scripts/audit_probes/independent_reference_test.py
"""
import numpy as np

from config import phase4_queue_config
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation

BETA_M = (0.9815, 0.7306, 0.6466, 0.5922)
cfg = phase4_queue_config(la_mode="post_rzf", decode_order="rbg_major",
                          la_beta_by_depth=BETA_M,
                          p_arrival_min=0.15, p_arrival_max=0.40)
M = cfg.num_bs_ant
P = cfg.p_rbg
NV = 1.7  # arbitrary non-unit noise variance

# ---------- (A) closed-form SINR on orthogonal groups ---------------------
rng = np.random.default_rng(3)
for m in (1, 2, 3, 4):
    g = rng.uniform(0.4, 2.5, size=m)
    h = np.zeros((m, M), dtype=complex)
    for i in range(m):
        h[i, i] = g[i]                       # h_i = g_i e_i
    _, sinr, caps = predict_group_link_adaptation(h, NV, NV, P, cfg)
    sinr_ref = g ** 2 * P / (m * NV)
    caps_ref = (BETA_M[m - 1] * cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                * np.log2(1.0 + sinr_ref))
    assert np.allclose(sinr, sinr_ref, rtol=1e-10), (m, sinr, sinr_ref)
    assert np.allclose(caps, caps_ref, rtol=1e-10), (m, caps, caps_ref)
print("(A) orthogonal closed-form SINR/cap: m=1..4 exact  OK")

# ---------- (B) budget / epsilon-drop hand trace ---------------------------
# geometry: UE0 = 2.0*e0 on both RBGs; UE1 = 1.5*e1. Orthogonal, so every
# group SINR is closed-form as above.
h_hat = np.zeros((cfg.num_ue, cfg.num_rbg, M), dtype=complex)
h_hat[0, :, 0] = 2.0
h_hat[1, :, 1] = 1.5
cap_m1_ue0 = (BETA_M[0] * cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
              * np.log2(1.0 + 4.0 * P / NV))

# UE0 backlog: cap + eps/2 above cap so RBG0 leaves a sub-epsilon tail that
# must be SWALLOWED into RBG0's unit (b = remaining), leaving exactly 0.
unc = np.zeros(cfg.num_ue)
unc[0] = cap_m1_ue0 + 0.5 * cfg.b_tx_epsilon
unc[1] = 0.5 * cfg.b_tx_epsilon            # sub-epsilon -> epsilon-dropped

pl = SlotAllocationPlanner(cfg, h_hat, NV, NV, unc)
kept, btx = pl.close_rbg(0, [], [0, 1])
# hand-derived expectations:
#  - UE1's cap(m=2) is far above its 0.5*eps backlog -> b < eps -> dropped
#  - group recomputes as m=1 {UE0}; UE0 full backlog leaves tail eps/2 < eps
#    -> swallow: b = cap + eps/2 (NOT the plain cap)
assert kept == [0] and 1 not in btx, (kept, btx)
assert abs(btx[0] - (cap_m1_ue0 + 0.5 * cfg.b_tx_epsilon)) < 1e-9, btx
assert pl.remaining(0) == 0.0 and pl.remaining(1) == unc[1]
kept2, btx2 = pl.close_rbg(1, [], [0, 1])   # both now sub-eps/zero -> empty
assert kept2 == [] and btx2 == {}
print("(B) epsilon-drop + swallow-tail + exact-debit hand trace  OK")

print("independent reference test PASSED")
