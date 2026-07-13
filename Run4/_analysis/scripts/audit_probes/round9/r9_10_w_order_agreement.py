"""Measure the numerical (not literal) agreement between the planner's
predicted W (canonical member order: sorted fixed + new) and the env's
transmission W (layer order: compaction order fixed + new), claim 10 Part A(a).
"""
import numpy as np
from phy import rzf_precoder
from config import Config

cfg = Config()
M = cfg.num_bs_ant
P = cfg.p_rbg
rng = np.random.default_rng(7)

max_dw = 0.0
max_dsinr = 0.0
for trial in range(500):
    m = int(rng.integers(2, 5))
    h = (rng.standard_normal((m, M)) + 1j * rng.standard_normal((m, M)))
    h *= rng.uniform(0.3, 2.5, size=(m, 1))
    alpha = rng.uniform(0.1, 2.0)
    nv = alpha
    perm = rng.permutation(m)              # env layer order vs planner order
    W_a = rzf_precoder(h, alpha, P)        # planner order
    W_b = rzf_precoder(h[perm], alpha, P)  # env order (permuted rows)
    # map env columns back to planner user order
    W_b_back = np.empty_like(W_a)
    for j, pj in enumerate(perm):
        W_b_back[:, pj] = W_b[:, j]
    dw = np.max(np.abs(W_a - W_b_back)) / np.max(np.abs(W_a))
    # per-user SINR in both orders (genie h_true = h)
    def sinr(hrows, W):
        eff = hrows @ W
        p = np.abs(eff) ** 2
        d = np.diag(p)
        return d / (p.sum(axis=1) - d + nv)
    s_a = sinr(h, W_a)
    s_b = sinr(h[perm], W_b)
    s_b_back = np.empty_like(s_a)
    for j, pj in enumerate(perm):
        s_b_back[pj] = s_b[j]
    ds = np.max(np.abs(s_a - s_b_back) / s_a)
    max_dw = max(max_dw, dw)
    max_dsinr = max(max_dsinr, ds)

print(f"500 groups m in 2..4, random permutations:")
print(f"max rel diff W columns (planner order vs env order): {max_dw:.3e}")
print(f"max rel diff per-user SINR:                          {max_dsinr:.3e}")
print(f"bit-identical W? {max_dw == 0.0}")
