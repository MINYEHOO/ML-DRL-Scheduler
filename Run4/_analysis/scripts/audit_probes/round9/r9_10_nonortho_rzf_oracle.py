"""Analytic NON-ORTHOGONAL oracle for rzf_precoder / _rbg_sinr /
predict_group_link_adaptation (claim 10 Part A(b)).

For m=2 arbitrary non-orthogonal complex channels, build
    V = H^H (H H^H + alpha I)^{-1}
entirely from SCALARS via the closed-form 2x2 complex inverse
(det + adjugate), never calling np.linalg.inv/solve on that expression.
Then unit-normalize columns, scale sqrt(P/2), and compute per-user SINR
by scalar inner products. Compare against phy._rbg_sinr and
la_planner.predict_group_link_adaptation.
"""
import numpy as np
from phy import rzf_precoder, _rbg_sinr
from la_planner import predict_group_link_adaptation
from config import Config

cfg = Config()
M = cfg.num_bs_ant
P = cfg.p_rbg
rng = np.random.default_rng(42)

def sdot(a, b):
    """scalar inner product sum_i conj(a_i) b_i via python loop (no BLAS)."""
    s = 0.0 + 0.0j
    for i in range(len(a)):
        s += np.conj(a[i]) * b[i]
    return s

max_rel_sinr_phy = 0.0
max_rel_sinr_pl = 0.0
max_rel_cap = 0.0
max_rel_w = 0.0
worst_corr = None

for trial in range(300):
    # non-orthogonal pair: h2 = mix of h1 direction and independent part
    h1 = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    h1 *= rng.uniform(0.3, 2.5)
    g = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    rho = rng.uniform(0.1, 0.95)          # forced overlap
    h2 = rho * h1 / np.abs(np.sqrt(sdot(h1, h1))) + np.sqrt(1 - rho**2) * g
    h2 *= rng.uniform(0.3, 2.5)
    alpha = rng.uniform(0.05, 3.0)
    nv = rng.uniform(0.05, 3.0)

    # ---- closed-form 2x2 inverse of A = H H^H + alpha I, from scalars ----
    a11 = sdot(h1, h1).real + alpha        # h1 h1^H (row convention: h1 . h1*)
    # careful with convention: gram = H @ H.conj().T -> gram[i,j] = sum h_i conj(h_j)
    g11 = 0.0 + 0.0j; g12 = 0.0 + 0.0j; g21 = 0.0 + 0.0j; g22 = 0.0 + 0.0j
    for i in range(M):
        g11 += h1[i] * np.conj(h1[i])
        g12 += h1[i] * np.conj(h2[i])
        g21 += h2[i] * np.conj(h1[i])
        g22 += h2[i] * np.conj(h2[i])
    a11 = g11 + alpha
    a12 = g12
    a21 = g21
    a22 = g22 + alpha
    det = a11 * a22 - a12 * a21
    i11 = a22 / det
    i12 = -a12 / det
    i21 = -a21 / det
    i22 = a11 / det

    # V = H^H A^{-1}: column j of V = conj(h1)*I[1,j] + conj(h2)*I[2,j]
    v1 = np.array([np.conj(h1[i]) * i11 + np.conj(h2[i]) * i21
                   for i in range(M)])
    v2 = np.array([np.conj(h1[i]) * i12 + np.conj(h2[i]) * i22
                   for i in range(M)])
    n1 = np.sqrt(sdot(v1, v1).real)
    n2 = np.sqrt(sdot(v2, v2).real)
    w1 = np.sqrt(P / 2.0) * v1 / n1
    w2 = np.sqrt(P / 2.0) * v2 / n2

    # oracle SINR (precoder from h_hat=h, evaluated on h_true=h)
    e11 = abs(sdot(np.conj(h1), w1)) ** 2   # |h1 @ w1|^2 ; h @ w = sum h_i w_i
    # NOTE: eff = h_true_rows @ w -> eff[i,j] = sum_t h_i[t] w_j[t] (no conj)
    def hdotw(h, w):
        s = 0.0 + 0.0j
        for i in range(M):
            s += h[i] * w[i]
        return s
    p11 = abs(hdotw(h1, w1)) ** 2
    p12 = abs(hdotw(h1, w2)) ** 2
    p21 = abs(hdotw(h2, w1)) ** 2
    p22 = abs(hdotw(h2, w2)) ** 2
    sinr_oracle = np.array([p11 / (p12 + nv), p22 / (p21 + nv)])

    H = np.stack([h1, h2])
    # repo precoder + SINR
    W_repo = rzf_precoder(H, alpha, P)
    s_repo = _rbg_sinr(H, H, alpha, P, nv)
    _, s_pl, caps_pl = predict_group_link_adaptation(H, nv, alpha, P, cfg)
    # cap oracle
    if cfg.la_beta_by_depth:
        beta = cfg.la_beta_by_depth[min(2, len(cfg.la_beta_by_depth)) - 1]
    else:
        beta = cfg.la_beta
    caps_oracle = (beta * cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
                   * np.log2(1.0 + sinr_oracle))

    W_oracle = np.stack([w1, w2], axis=1)
    rel_w = np.max(np.abs(W_repo - W_oracle)) / np.max(np.abs(W_oracle))
    rel_phy = np.max(np.abs(s_repo - sinr_oracle) / sinr_oracle)
    rel_pl = np.max(np.abs(s_pl - sinr_oracle) / sinr_oracle)
    rel_cap = np.max(np.abs(caps_pl - caps_oracle) / caps_oracle)
    max_rel_w = max(max_rel_w, rel_w)
    max_rel_sinr_phy = max(max_rel_sinr_phy, rel_phy)
    max_rel_sinr_pl = max(max_rel_sinr_pl, rel_pl)
    max_rel_cap = max(max_rel_cap, rel_cap)

corr_note = ""
print(f"trials=300 m=2 non-orthogonal (|rho| up to 0.95), alpha,nv in [0.05,3]")
print(f"max rel err W  (rzf_precoder vs scalar closed-form): {max_rel_w:.3e}")
print(f"max rel err SINR (phy._rbg_sinr vs oracle):          {max_rel_sinr_phy:.3e}")
print(f"max rel err SINR (planner predict vs oracle):        {max_rel_sinr_pl:.3e}")
print(f"max rel err btx_cap (planner caps vs oracle):        {max_rel_cap:.3e}")
ok = max(max_rel_w, max_rel_sinr_phy, max_rel_sinr_pl, max_rel_cap) < 1e-10
print("NON-ORTHOGONAL ORACLE:", "PASS (<1e-10)" if ok else "FAIL")
