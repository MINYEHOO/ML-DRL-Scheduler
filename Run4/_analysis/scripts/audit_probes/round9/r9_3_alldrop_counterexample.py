"""Claim 3 probe A2: construct the all-drop-with-feasible-singleton case.
Key insight: with RZF alpha=noise_var, near-colinear STRONG channels saturate
at group SINR ~ 1/(m-1) (caps >> eps), so the auditor's 'strong interferers'
story cannot reach cap < eps. The reachable regime is WEAK channels: group
cap_m ~ 0.15 * singleton cap_1 (power split /m + interference + depth-beta),
so any group whose members' singleton caps are in ~[1, 6.6] bits all-drops."""
import numpy as np
from config import Config
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation

cfg = Config()
cfg.la_mode = "post_rzf"
cfg.la_beta_by_depth = (0.9815, 0.7306, 0.6466, 0.5922)   # Run4
M, nv = cfg.num_bs_ant, 1.0
g, t = 0.05, 1e-3          # weak, nearly-colinear
h_hat = np.zeros((4, 1, M), dtype=complex)
for i in range(4):
    v = np.zeros(M, dtype=complex); v[0] = 1.0; v[1 + i] = t
    h_hat[i, 0, :] = g * v

_, s4, c4 = predict_group_link_adaptation(h_hat[:, 0, :], nv, nv, cfg.p_rbg, cfg)
print("group-of-4 caps [bits]:", np.round(c4, 3), "(all < eps=1)")
for i in range(4):
    _, s1, c1 = predict_group_link_adaptation(h_hat[i:i+1, 0, :], nv, nv,
                                              cfg.p_rbg, cfg)
    print(f"  singleton u{i}: cap = {c1[0]:.3f} bits  feasible={c1[0] >= 1.0}")

remaining = np.full(4, 5000.0)
pl = SlotAllocationPlanner(cfg, h_hat, nv, nv, remaining.copy())
kept, btx = pl.close_rbg(0, [], [0, 1, 2, 3])
print("close_rbg -> kept =", kept, " btx =", btx)

# one-at-a-time removal (drop single smallest-b member, recompute)
new = [0, 1, 2, 3]
while new:
    _, s, c = predict_group_link_adaptation(h_hat[new, 0, :], nv, nv,
                                            cfg.p_rbg, cfg)
    b = np.minimum(remaining[new], c)
    if (b >= cfg.b_tx_epsilon).all():
        break
    new.pop(int(np.argmin(b)))
print("one-at-a-time would keep:", new,
      "caps:", np.round(c, 3) if new else "-")
