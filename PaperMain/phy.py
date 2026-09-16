"""
phy.py -- PHY abstraction: RZF precoding, SINR, mutual information.

PHY abstraction (no per-RE link-level simulation). For a slot's allocation:
  1. reconstruct the BS channel estimate h_hat from fed-back PMI/CQI,
  2. compute the per-RBG RZF precoder from h_hat,
  3. evaluate per-(UE,RBG) SINR with the *true* channel,
  4. convert SINR to delivered mutual information (Shannon).

Key spec points:
  * Precoder uses fed-back CSI (PMI = direction, CQI = magnitude); SINR is
    evaluated on the true channel. Stale CSI -> precoding mismatch -> failure.
  * RZF: V = H^H (H H^H + alpha I)^-1, unit-norm columns, then per-stream
    power sqrt(P_r/m) so total RBG power = P_r.
  * Noise sigma^2 is set by calibration so the median single-user SNR equals
    the configured target SNR.
"""

from __future__ import annotations

import numpy as np

from config import Config


# ---------------------------------------------------------------------------
# noise calibration
# ---------------------------------------------------------------------------
def sigma2_from_gain(g: np.ndarray, cfg: Config) -> float:
    """sigma^2 from precomputed beamformed gains g = P_r * |h^H h_dir_hat|^2.

    Single source of the calibration rule, shared by ``calibrate_noise``
    and ``csi.precompute_episode_csi`` (which already has the gains).
    """
    if cfg.noise_mode == "simple":
        return float(10.0 ** (-cfg.target_snr_db / 10.0))
    return float(np.median(g) / (10.0 ** (cfg.target_snr_db / 10.0)))


def calibrate_noise(h_true_episode: np.ndarray, codebook,
                    cfg: Config) -> float:
    """Pick sigma^2 so the median single-user beamformed SNR == target.

    The SU beamformed signal power is computed along the codebook's
    quantized direction (so the calibration is consistent with the actual
    operating point of the chosen PMI mode):

        G_{u,r}  = P_r * |h_true^H * h_dir_hat|^2
        sigma^2  = median(G) / 10^(target_snr_db / 10).
    """
    if cfg.noise_mode == "simple":
        return float(10.0 ** (-cfg.target_snr_db / 10.0))

    direction = codebook.direction_from_channel(h_true_episode)
    bf_gain = np.abs(np.einsum("...t,...t->...",
                               np.conj(h_true_episode), direction)) ** 2
    return sigma2_from_gain(cfg.p_rbg * bf_gain, cfg)


# ---------------------------------------------------------------------------
# channel estimate reconstruction (BS side, from feedback)
# ---------------------------------------------------------------------------
def reconstruct_h_hat(direction_fb: np.ndarray, cqi_fb: np.ndarray,
                      p_rbg: float, noise_var: float) -> np.ndarray:
    """BS channel estimate from fed-back direction and CQI.

    h_hat = g * h_dir_hat, with h_dir_hat the (unit-norm) direction the BS
    reconstructed from the PMI feedback, and the magnitude g chosen so that
    the SU SNR of h_hat along that direction equals the fed-back SNR:
        P_r * |g|^2 / sigma^2 = 2^CQI - 1.
    """
    snr_fb = np.maximum(2.0 ** cqi_fb - 1.0, 0.0)          # [K, R]
    g = np.sqrt(noise_var * snr_fb / p_rbg)                # [K, R] real
    return g[..., None] * direction_fb


# ---------------------------------------------------------------------------
# RZF precoder
# ---------------------------------------------------------------------------
def rzf_precoder(h_hat_rows: np.ndarray, alpha: float,
                 p_rbg: float) -> np.ndarray:
    """Regularized zero-forcing precoder for one RBG.

    V = H^H (H H^H + alpha I)^-1 ; columns normalized to unit norm; then
    per-stream power sqrt(P_r / m) so the total RBG power is P_r.

    Parameters
    ----------
    h_hat_rows : ndarray [m, M_ant], complex
        Estimated channel rows of the m co-scheduled UEs.
    alpha : float        regularization
    p_rbg : float        per-RBG transmit power

    Returns
    -------
    W : ndarray [M_ant, m], complex -- column j = precoder for stream j.
    """
    m = h_hat_rows.shape[0]
    gram = h_hat_rows @ h_hat_rows.conj().T                # [m, m]
    gram = gram + alpha * np.eye(m, dtype=gram.dtype)
    # V = H^H (gram)^-1 = (gram^-1 H)^H  (gram is Hermitian)
    v = np.linalg.solve(gram, h_hat_rows).conj().T         # [M_ant, m]
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    v = v / np.where(norms > 0, norms, 1.0)                # unit-norm columns
    return np.sqrt(p_rbg / m) * v                          # [M_ant, m]


def _rbg_sinr(h_true_rows: np.ndarray, h_hat_rows: np.ndarray,
              alpha: float, p_rbg: float, noise_var: float) -> np.ndarray:
    """Per-stream SINR for one RBG (precoder from h_hat, eval on h_true)."""
    w = rzf_precoder(h_hat_rows, alpha, p_rbg)             # [M_ant, m]
    eff = h_true_rows @ w                                  # [m, m]
    power = np.abs(eff) ** 2                               # [m, m]
    desired = np.diag(power)                               # [m]
    interference = power.sum(axis=1) - desired             # [m]
    return desired / (interference + noise_var)


def compute_slot_sinr(allocation: np.ndarray, h_true_slot: np.ndarray,
                      h_hat_slot: np.ndarray, p_rbg: float,
                      noise_var: float, alpha: float) -> np.ndarray:
    """Per-position SINR for a whole slot allocation.

    Parameters
    ----------
    allocation : ndarray [R, L] int
        allocation[r, l] in {0..K}; 0 = no-user, k>0 = UE index k-1.
    h_true_slot : [K, R, M_ant] complex   -- true channel (evaluation)
    h_hat_slot  : [K, R, M_ant] complex   -- BS estimate (precoding)
    p_rbg, noise_var, alpha : float

    Returns
    -------
    sinr : ndarray [R, L] float -- SINR per occupied position (0 if no-user).
    """
    R, L = allocation.shape
    sinr = np.zeros((R, L), dtype=np.float64)
    for r in range(R):
        occ = np.where(allocation[r] > 0)[0]               # occupied layers
        if occ.size == 0:
            continue
        ues = allocation[r, occ] - 1                       # 0-based UE idx
        s = _rbg_sinr(h_true_slot[ues, r, :], h_hat_slot[ues, r, :],
                      alpha, p_rbg, noise_var)
        sinr[r, occ] = s
    return sinr


# ---------------------------------------------------------------------------
# mutual information / rate prediction
# ---------------------------------------------------------------------------
def mi_bits(sinr, cfg: Config):
    """Delivered mutual information [bits] for a transmission unit in a slot.

    MI = eta_data * N_RE_RBG * log2(1 + SINR).
    """
    return cfg.eta_data * cfg.n_re_rbg * np.log2(1.0 + np.asarray(sinr))


def predict_b_tx(cqi_fb_value, cfg: Config):
    """Predicted target payload [bits] from fed-back CQI.

    B_tx_pred = eta_data * N_RE_RBG * beta_rate * CQI_fb
    (still to be capped by the packet's uncommitted backlog by the caller).
    """
    return (cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
            * np.asarray(cqi_fb_value))


if __name__ == "__main__":
    from codebook import RandomUnitNormCodebook

    cfg = Config()
    cb = RandomUnitNormCodebook(cfg.codebook_size, cfg.num_bs_ant, cfg.seed)
    rng = np.random.default_rng(0)

    # --- calibration: median SU SNR should hit the target ---
    T = 50
    h_ep = (rng.standard_normal((T, cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant))
            + 1j * rng.standard_normal((T, cfg.num_ue, cfg.num_rbg,
                                        cfg.num_bs_ant)))
    sigma2 = calibrate_noise(h_ep, cb, cfg)
    pmi = cb.pmi_from_channel(h_ep)
    c = cb.get_codeword(pmi)
    g = cfg.p_rbg * np.abs(np.einsum("...m,...m->...", np.conj(h_ep), c)) ** 2
    med_snr_db = 10 * np.log10(np.median(g) / sigma2)
    assert abs(med_snr_db - cfg.target_snr_db) < 1e-6
    print(f"phy.py: calibration ok -- sigma^2={sigma2:.3e}, "
          f"median SU SNR={med_snr_db:.2f} dB (target {cfg.target_snr_db})")

    # --- RZF: total RBG power == P_r ---
    H = (rng.standard_normal((3, cfg.num_bs_ant))
         + 1j * rng.standard_normal((3, cfg.num_bs_ant)))
    W = rzf_precoder(H, alpha=sigma2, p_rbg=cfg.p_rbg)
    tot = np.sum(np.abs(W) ** 2)
    assert abs(tot - cfg.p_rbg) < 1e-9, tot
    print(f"phy.py: RZF ok -- total RBG power={tot:.4f} (P_r={cfg.p_rbg})")

    # --- orthogonal vs identical channels, interference-limited regime ---
    # (small noise so the MU interference term -- not noise -- dominates)
    e1 = np.zeros(cfg.num_bs_ant, dtype=complex); e1[0] = 1.0
    e2 = np.zeros(cfg.num_bs_ant, dtype=complex); e2[1] = 1.0
    orth = np.stack([e1, e2])
    same = np.stack([e1, e1])
    nv = 1e-3
    s_orth = _rbg_sinr(orth, orth, alpha=1e-3, p_rbg=cfg.p_rbg, noise_var=nv)
    s_same = _rbg_sinr(same, same, alpha=1e-3, p_rbg=cfg.p_rbg, noise_var=nv)
    print(f"phy.py: SINR orthogonal pair {s_orth[0]:.1f}  "
          f"vs identical pair {s_same[0]:.3f}")
    assert s_orth[0] > s_same[0] * 10, "orthogonal users should beat co-linear"
    print("phy.py smoke test passed.")
