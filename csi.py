"""
csi.py -- CSI generation and the CSI feedback buffer (mode-agnostic).

  * ``generate_true_csi`` -- given the true channel and a codebook, returns
    (raw_pmi, direction, cqi). The codebook quantizes the channel direction;
    CQI is the single-user spectral efficiency for the QUANTIZED direction:
        SNR_SU = P_r * |h_true^H * h_dir_hat|^2 / sigma^2
        CQI    = log2(1 + SNR_SU)
    This is the only fed-back quality term -- not exact 3GPP CQI, but the
    quantized-PMI-direction-conditioned SE abstraction.

  * ``CSIFeedbackBuffer`` -- per-UE Bernoulli feedback. On feedback, the buffer
    copies the current true CSI for the UE and resets Age; otherwise the
    stale values stay and Age increments.

The buffer exposes ``direction_fb`` [K, R, num_bs_ant] as the COMMON interface
downstream code consumes; ``raw_pmi_fb`` is mode-specific (int for random,
``Type2PMI`` for type2) and used only for logging / research.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from codebook import Type2PMI
from config import Config
from phy import sigma2_from_gain


# 3GPP TS 38.214 Table 5.2.2.1-3 (4-bit CQI, 256QAM): the 15 spectral-
# efficiency entries plus index 0 = "out of range" (SE 0 -> the UE reports
# the RBG as unusable; reconstruct_h_hat then yields h_hat = 0 and the
# candidate gate excludes the UE there). Used when cfg.cqi_mode == 'nr4bit'.
NR_CQI_TABLE_256QAM = np.array([
    0.0,
    0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223,
    3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063])


def quantize_cqi(cqi: np.ndarray, cqi_mode: str) -> np.ndarray:
    """Quantize the continuous SE report per cfg.cqi_mode.

    'nr4bit' floor-snaps to the NR 256QAM ladder -- the UE reports the
    highest index whose SE it can support (the BLER<=10% selection rule's
    Shannon-abstraction analog). Values below entry 1 collapse to 0.
    """
    if cqi_mode == "continuous":
        return cqi
    if cqi_mode == "nr4bit":
        idx = np.searchsorted(NR_CQI_TABLE_256QAM, cqi, side="right") - 1
        return NR_CQI_TABLE_256QAM[np.clip(idx, 0, len(NR_CQI_TABLE_256QAM) - 1)]
    raise ValueError(f"unknown cqi_mode {cqi_mode!r}")


def generate_true_csi(h_true: np.ndarray, codebook, p_rbg: float,
                      noise_var: float, cqi_mode: str = "continuous"):
    """Returns (raw_pmi, direction, cqi).

    direction : [K, R, num_bs_ant] unit-norm complex
    cqi       : [K, R] SE = log2(1 + SNR_SU), quantized per ``cqi_mode``
    """
    raw_pmi, direction = codebook.quantize(h_true)
    bf_gain = np.abs(np.einsum("...t,...t->...",
                               np.conj(h_true), direction)) ** 2
    cqi = quantize_cqi(np.log2(1.0 + p_rbg * bf_gain / noise_var), cqi_mode)
    return raw_pmi, direction, cqi


def _slice_pmi(raw_pmi_ep, t: int):
    """Slot-t view of an episode-stacked raw PMI (no copy)."""
    if isinstance(raw_pmi_ep, Type2PMI):
        return Type2PMI(atom_idx=raw_pmi_ep.atom_idx[t],
                        amp_idx=raw_pmi_ep.amp_idx[t],
                        phase_idx=raw_pmi_ep.phase_idx[t])
    return raw_pmi_ep[t]


@dataclasses.dataclass
class EpisodeCSI:
    """Whole-episode true CSI, precomputed once per episode seed.

    True CSI is a deterministic function of (h_true_episode, codebook,
    cfg) -- quantization has no RNG and sigma^2 is episode-level -- so
    computing it slot-by-slot at episode start is exactly equivalent to
    the old per-slot ``generate_true_csi`` calls, and revisiting a seed
    (eval rounds, phase-1 baseline comparisons) can reuse it instead of
    re-running OMP. Env-internal only: the actor still observes nothing
    but the feedback buffer, so no future CSI can leak.
    """
    sigma2: float
    raw_pmi: object              # Type2PMI of [T, ...] arrays, or [T, K, R]
    direction: np.ndarray        # [T, K, R, num_bs_ant]
    cqi: np.ndarray              # [T, K, R]

    def slot(self, t: int):
        """(raw_pmi, direction, cqi) views for slot t -- the exact
        return of ``generate_true_csi`` on that slot's true channel."""
        return _slice_pmi(self.raw_pmi, t), self.direction[t], self.cqi[t]


def precompute_episode_csi(h_true_episode: np.ndarray, codebook,
                           cfg: Config) -> EpisodeCSI:
    """Quantize every slot and calibrate sigma^2 in one pass.

    The loop quantizes per slot -- the same call pattern as the old
    per-slot path -- so outputs are bit-identical by construction (and
    measured faster than one big [T*K*R] batch, by cache locality).
    sigma^2 reuses the gains instead of re-quantizing the episode as
    ``calibrate_noise`` did; CQI keeps the op order of
    ``generate_true_csi``: log2(1 + (p_rbg * gain) / sigma^2).
    """
    T = h_true_episode.shape[0]
    pmis, dirs, gains = [], [], []
    for t in range(T):
        raw_pmi, direction = codebook.quantize(h_true_episode[t])
        bf_gain = np.abs(np.einsum("...t,...t->...",
                                   np.conj(h_true_episode[t]),
                                   direction)) ** 2
        pmis.append(raw_pmi)
        dirs.append(direction)
        gains.append(bf_gain)

    g = cfg.p_rbg * np.stack(gains)                  # [T, K, R]
    sigma2 = sigma2_from_gain(g, cfg)
    cqi = quantize_cqi(np.log2(1.0 + g / sigma2), cfg.cqi_mode)

    if isinstance(pmis[0], Type2PMI):
        raw_pmi_ep = Type2PMI(
            atom_idx=np.stack([p.atom_idx for p in pmis]),
            amp_idx=np.stack([p.amp_idx for p in pmis]),
            phase_idx=np.stack([p.phase_idx for p in pmis]))
    else:
        raw_pmi_ep = np.stack([np.asarray(p) for p in pmis])

    return EpisodeCSI(sigma2=sigma2, raw_pmi=raw_pmi_ep,
                      direction=np.stack(dirs), cqi=cqi)


def _clone_pmi(raw_pmi):
    if isinstance(raw_pmi, Type2PMI):
        return Type2PMI(atom_idx=raw_pmi.atom_idx.copy(),
                        amp_idx=raw_pmi.amp_idx.copy(),
                        phase_idx=raw_pmi.phase_idx.copy())
    return np.asarray(raw_pmi).copy()


def _update_pmi_mask(fb_pmi, true_pmi, mask):
    if isinstance(fb_pmi, Type2PMI):
        fb_pmi.atom_idx[mask] = true_pmi.atom_idx[mask]
        fb_pmi.amp_idx[mask] = true_pmi.amp_idx[mask]
        fb_pmi.phase_idx[mask] = true_pmi.phase_idx[mask]
    else:
        fb_pmi[mask] = true_pmi[mask]


class CSIFeedbackBuffer:
    """Fed-back direction/CQI/Age -- the only CSI the scheduler sees.

    Feedback is UE-level Bernoulli. On feedback for UE u, ALL of u's RBGs
    update at once and Age[u] = 0; otherwise the previous values are kept
    and Age[u] += 1.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        K, R = cfg.num_ue, cfg.num_rbg
        self.direction_fb = np.zeros((K, R, cfg.num_bs_ant),
                                     dtype=np.complex128)
        self.cqi_fb = np.zeros((K, R), dtype=np.float64)
        self.age = np.zeros(K, dtype=np.int64)
        self.raw_pmi_fb = None       # filled at reset; mode-specific

    def reset(self, raw_pmi_true, direction_true, cqi_true):
        """Forced initial feedback at episode start (spec refinement #10)."""
        self.direction_fb = direction_true.astype(np.complex128).copy()
        self.cqi_fb = cqi_true.astype(np.float64).copy()
        self.age[:] = 0
        self.raw_pmi_fb = _clone_pmi(raw_pmi_true)

    def step(self, raw_pmi_true, direction_true, cqi_true,
             rng: np.random.Generator) -> np.ndarray:
        """UE-level Bernoulli feedback. Returns the feedback mask [K] bool."""
        fb = rng.random(self.cfg.num_ue) < self.cfg.p_csi
        if np.any(fb):
            self.direction_fb[fb] = direction_true[fb]
            self.cqi_fb[fb] = cqi_true[fb]
            _update_pmi_mask(self.raw_pmi_fb, raw_pmi_true, fb)
        self.age[fb] = 0
        self.age[~fb] += 1
        return fb


if __name__ == "__main__":
    from codebook import make_codebook

    for mode in ("random_unit_norm", "type2_sparse_56bit"):
        cfg = Config(pmi_mode=mode)
        cb = make_codebook(cfg)
        rng = np.random.default_rng(0)
        h = (rng.standard_normal((cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant))
             + 1j * rng.standard_normal((cfg.num_ue, cfg.num_rbg,
                                         cfg.num_bs_ant)))
        raw_pmi, direction, cqi = generate_true_csi(h, cb, cfg.p_rbg, 1.0)
        assert direction.shape == (cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant)
        assert cqi.shape == (cfg.num_ue, cfg.num_rbg)
        norms = np.linalg.norm(direction, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-9), norms.max()

        buf = CSIFeedbackBuffer(cfg)
        buf.reset(raw_pmi, direction, cqi)
        assert np.all(buf.age == 0)
        for _ in range(5):
            buf.step(raw_pmi, direction, cqi, rng)
        assert buf.age.max() >= 1
        print(f"csi.py [{mode:22s}] direction unit-norm OK, "
              f"CQI range [{cqi.min():.2f},{cqi.max():.2f}], "
              f"Age range [{buf.age.min()},{buf.age.max()}]")
    print("csi.py smoke test passed.")
