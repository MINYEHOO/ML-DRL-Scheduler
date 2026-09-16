"""
codebook.py -- PMI codebooks.

Two modes are supported:

  1. ``RandomUnitNormCodebook`` ("random_unit_norm")
        M random unit-norm codewords in C^{num_bs_ant}. PMI = best codeword
        index (single integer).

  2. ``Type2SparseCodebook``  ("type2_sparse_56bit")
        Simplified Type-II-like sparse atomic feedback (NOT exact 3GPP
        Rel-15 Type II). Dictionary of (N1*O1)*(N2*O2)*N_pol unit-norm atoms
        built from oversampled 2D DFT beams per polarization. The UE picks L
        atoms with OMP + LS, quantizes amplitudes (round to amp_levels) and
        phases (QPSK), reports the indices, and the BS reconstructs the
        unit-norm direction by summing the quantized linear combination and
        re-normalizing. Payload = L*(atom_bits + amp_bits + phase_bits).

Both codebooks expose the same interface so downstream code (csi, env, phy)
is mode-agnostic:

  ``quantize(h)``                  -> (raw_pmi, direction)
  ``reconstruct(raw_pmi)``         -> direction
  ``direction_from_channel(h)``    -> direction       (convenience)

``direction`` is always unit-norm complex of shape ``[..., num_bs_ant]``.

Sionna PanelArray (dual cross-pol, 4x4 grid) port layout:
  ports 0..15 are pol0 (a.k.a. pol1 in Sionna's naming),
  ports 16..31 are pol1 (pol2 in Sionna).
  Within a polarization, port i = row*N2 + col where row varies along y and
  col along z. The Type-II dictionary atoms match this layout exactly:
  per-pol atom = block-stacked [b_{m,n}; 0] (pol0) or [0; b_{m,n}] (pol1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Random unit-norm codebook
# ---------------------------------------------------------------------------
class GenieCodebook:
    """Perfect-CSI (genie) codebook: no quantization at all.

    quantize(h) returns the EXACT channel direction h/||h|| (unit-norm),
    so downstream reconstruct_h_hat(direction, cqi) rebuilds h_true exactly
    (the CQI-derived magnitude equals ||h||). With p_csi=1.0 the BS then
    holds the true channel every slot -- the genie upper bound isolating
    "what codebook quantization + staleness cost" from everything else.
    raw_pmi is a zero placeholder (logging only; no real index exists).
    """

    name = "genie"

    def __init__(self, dim: int):
        self.dim = int(dim)

    def quantize(self, h: np.ndarray):
        norm = np.linalg.norm(h, axis=-1, keepdims=True)
        direction = h / np.maximum(norm, 1e-12)            # exact unit direction
        raw_pmi = np.zeros(h.shape[:-1], dtype=np.int64)   # placeholder
        return raw_pmi, direction.astype(np.complex128)


class RandomUnitNormCodebook:
    """M random unit-norm codewords in C^{dim}.

    PMI is a single integer codeword index in {0,...,M-1}. Used as the
    'baseline' codebook. The Type-II-like codebook is an upgrade path.
    """

    name = "random_unit_norm"

    def __init__(self, num_codewords: int, dim: int, seed: int = 0):
        self.num_codewords = int(num_codewords)
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        cw = (rng.standard_normal((self.num_codewords, self.dim))
              + 1j * rng.standard_normal((self.num_codewords, self.dim)))
        cw /= np.linalg.norm(cw, axis=1, keepdims=True)
        self.codewords = cw.astype(np.complex128)         # [M, dim], unit-norm

    # ----- legacy API (kept for clarity / external callers) -----
    def pmi_from_channel(self, h: np.ndarray) -> np.ndarray:
        corr = np.einsum("...d,md->...m", np.conj(h), self.codewords)
        return np.argmax(np.abs(corr) ** 2, axis=-1).astype(np.int64)

    def get_codeword(self, idx: np.ndarray) -> np.ndarray:
        return self.codewords[np.asarray(idx, dtype=np.int64)]

    # ----- unified interface -----
    def quantize(self, h: np.ndarray):
        """Returns (raw_pmi, direction). raw_pmi = int array, direction = codeword."""
        pmi = self.pmi_from_channel(h)
        return pmi, self.get_codeword(pmi)

    def reconstruct(self, raw_pmi) -> np.ndarray:
        return self.get_codeword(raw_pmi)

    def direction_from_channel(self, h: np.ndarray) -> np.ndarray:
        return self.get_codeword(self.pmi_from_channel(h))


# ---------------------------------------------------------------------------
# Type-II-like sparse codebook
# ---------------------------------------------------------------------------
@dataclass
class Type2PMI:
    """Sparse-atomic PMI (per (UE, RBG) feedback payload).

    Fields share the same leading shape (e.g. [K, R]) and a trailing axis of
    length L (number of atoms). Per-(UE,RBG) bits = L*(atom_bits+amp+phase).
    """
    atom_idx: np.ndarray     # [..., L] int     (which atoms)
    amp_idx: np.ndarray      # [..., L] int     amplitude index in 0..A-1
    phase_idx: np.ndarray    # [..., L] int     phase index in 0..P-1


class Type2SparseCodebook:
    """Simplified Type-II-like sparse atomic feedback model.

    NOT an exact 3GPP Rel-15 Type II implementation. Differences from the spec:
      * Polarization is folded INTO the atom dictionary -> a unit may use any
        L atoms across both polarizations (no shared-beam structure).
      * Quantization is simple uniform; spec uses non-uniform 3GPP tables.

    Dictionary construction:
      * 2D DFT beams over the (N1, N2) per-pol antenna grid, oversampled by
        (O1, O2): (N1*O1) x (N2*O2) beams, each in C^{N1*N2} unit-norm.
      * Atoms = each beam embedded into one polarization slot of the C^{N_t}
        channel space: pol0 atom = [b; 0], pol1 atom = [0; b]. Each atom is
        unit-norm in C^{N_t}.
      * Total atoms = (N1*O1)*(N2*O2) * N_pol.
    """

    name = "type2_sparse_56bit"

    def __init__(self, cfg):
        self.cfg = cfg
        self.N1 = int(cfg.bs_ant_rows)
        self.N2 = int(cfg.bs_ant_cols)
        self.O1 = int(cfg.type2_O1)
        self.O2 = int(cfg.type2_O2)
        self.n_pol = cfg.n_pol
        self.dim = self.N1 * self.N2 * self.n_pol         # 32
        self.L = int(cfg.type2_L)
        self.amp_levels = 2 ** int(cfg.type2_amp_bits)    # 8
        self.phase_levels = 2 ** int(cfg.type2_phase_bits)  # 4
        self.A = self._build_dictionary()                 # [N_atoms, dim]
        self.A_conj = np.conj(self.A)
        self.n_atoms = self.A.shape[0]

    def _build_dictionary(self) -> np.ndarray:
        """Build the unit-norm atom dictionary [n_atoms, dim] complex128."""
        N1, N2, O1, O2 = self.N1, self.N2, self.O1, self.O2
        n_beams = N1 * O1 * N2 * O2

        # 2D DFT beam grid:
        #   b_{m,n}[i*N2 + j] = (1/sqrt(N1*N2)) * exp(j2pi*m*i/(N1*O1))
        #                                       * exp(j2pi*n*j/(N2*O2))
        i = np.arange(N1)                                  # row index per pol
        j = np.arange(N2)                                  # col index per pol
        m = np.arange(N1 * O1)
        n = np.arange(N2 * O2)
        ph_row = np.exp(1j * 2 * np.pi * np.outer(m, i) / (N1 * O1))  # [N1*O1, N1]
        ph_col = np.exp(1j * 2 * np.pi * np.outer(n, j) / (N2 * O2))  # [N2*O2, N2]
        # outer product over rows x cols -> [n_beams, N1*N2]
        beams = (np.einsum("mi,nj->mnij", ph_row, ph_col)
                 .reshape(n_beams, N1 * N2)
                 / np.sqrt(N1 * N2))                       # [n_beams, N1*N2]

        # Embed into 32-dim channel space per polarization
        per_pol = N1 * N2
        n_atoms = n_beams * self.n_pol
        A = np.zeros((n_atoms, self.dim), dtype=np.complex128)
        for p in range(self.n_pol):
            A[p * n_beams:(p + 1) * n_beams,
              p * per_pol:(p + 1) * per_pol] = beams
        return A

    # ----- unified interface -----
    def quantize(self, h: np.ndarray, chunk_size: int = 4096):
        """OMP + quantize. Returns (Type2PMI, direction [..., dim]).

        Processes the flattened batch in chunks of ``chunk_size`` to bound
        memory (the [B, n_atoms] correlation tensor is the limiting factor).
        """
        h = np.ascontiguousarray(h, dtype=np.complex128)
        leading = h.shape[:-1]
        h_flat = h.reshape(-1, self.dim)                   # [B, dim]
        B = h_flat.shape[0]

        if B <= chunk_size:
            ai, mi, pi, dirs = self._quantize_batch(h_flat)
        else:
            parts = [self._quantize_batch(h_flat[s:s + chunk_size])
                     for s in range(0, B, chunk_size)]
            ai = np.concatenate([p[0] for p in parts], axis=0)
            mi = np.concatenate([p[1] for p in parts], axis=0)
            pi = np.concatenate([p[2] for p in parts], axis=0)
            dirs = np.concatenate([p[3] for p in parts], axis=0)

        pmi = Type2PMI(
            atom_idx=ai.reshape(*leading, self.L),
            amp_idx=mi.reshape(*leading, self.L),
            phase_idx=pi.reshape(*leading, self.L),
        )
        direction = dirs.reshape(*leading, self.dim)
        return pmi, direction

    def _quantize_batch(self, h_flat: np.ndarray):
        """OMP + quantize on a flat batch [B, dim]. Returns 4 flat arrays."""
        B = h_flat.shape[0]
        L = self.L
        atom_idx = np.zeros((B, L), dtype=np.int64)
        selected = np.zeros((B, self.n_atoms), dtype=bool)
        residual = h_flat.copy()
        alpha_LS = np.zeros((B, L), dtype=np.complex128)

        for m in range(L):
            corr = residual @ self.A_conj.T               # [B, n_atoms]
            mag2 = np.abs(corr) ** 2
            mag2[selected] = -np.inf
            new_idx = np.argmax(mag2, axis=-1)            # [B]
            atom_idx[:, m] = new_idx
            selected[np.arange(B), new_idx] = True

            A_S = self.A[atom_idx[:, :m + 1]].transpose(0, 2, 1)  # [B,dim,m+1]
            gram = np.einsum("bti,btj->bij", np.conj(A_S), A_S)
            rhs = np.einsum("bti,bt->bi", np.conj(A_S), h_flat)
            # Tiny Tikhonov reg on gram for numerical safety if the OMP-
            # selected set ever contains highly correlated atoms (rare in
            # practice but possible with oversampled-DFT adjacencies).
            # gram diag = 1 (unit-norm atoms), so 1e-10 is well below
            # double-precision noise on a well-conditioned solve.
            # rhs[..., None]: NumPy 2.x requires an explicit trailing K
            # axis on the batched solve; [..., 0] drops it.
            gram_reg = gram + 1e-10 * np.eye(gram.shape[-1], dtype=gram.dtype)
            alpha = np.linalg.solve(gram_reg, rhs[..., None])[..., 0]  # [B,m+1]
            residual = h_flat - np.einsum("bti,bi->bt", A_S, alpha)
            if m == L - 1:
                alpha_LS = alpha

        rho = np.abs(alpha_LS)
        rho_max = np.maximum(rho.max(axis=-1, keepdims=True), 1e-12)
        rho_norm = np.clip(rho / rho_max, 0.0, 1.0)
        amp_idx = np.rint((self.amp_levels - 1) * rho_norm).astype(np.int64)
        amp_idx = np.clip(amp_idx, 0, self.amp_levels - 1)

        phi = np.angle(alpha_LS) % (2 * np.pi)
        phase_idx = (np.rint(phi / (2 * np.pi / self.phase_levels))
                     .astype(np.int64) % self.phase_levels)

        direction = self._reconstruct_flat(atom_idx, amp_idx, phase_idx)
        return atom_idx, amp_idx, phase_idx, direction

    def reconstruct(self, raw_pmi: "Type2PMI") -> np.ndarray:
        """Reconstruct unit-norm direction from feedback indices."""
        leading = raw_pmi.atom_idx.shape[:-1]
        L = raw_pmi.atom_idx.shape[-1]
        flat = lambda x: x.reshape(-1, L)
        direction = self._reconstruct_flat(flat(raw_pmi.atom_idx),
                                           flat(raw_pmi.amp_idx),
                                           flat(raw_pmi.phase_idx))
        return direction.reshape(*leading, self.dim)

    def direction_from_channel(self, h: np.ndarray) -> np.ndarray:
        """Convenience: quantize and return only the direction."""
        _, direction = self.quantize(h)
        return direction

    # ----- helpers -----
    def _reconstruct_flat(self, atom_idx, amp_idx, phase_idx) -> np.ndarray:
        """Reconstruct from per-batch flat arrays of shape [B, L].

        Degenerate-case guard: if the quantized linear combination happens to
        nearly cancel (||unnorm|| << 1, never seen in tests but theoretically
        possible with adversarial quantization), fall back to the strongest
        atom -- OMP's first pick, which is the largest channel projection
        and is unit-norm by construction.
        """
        amp_hat = amp_idx.astype(np.float64) / (self.amp_levels - 1)
        phase_hat = phase_idx.astype(np.float64) * (2 * np.pi
                                                    / self.phase_levels)
        coef = amp_hat * np.exp(1j * phase_hat)                     # [B, L]
        atoms = self.A[atom_idx]                                    # [B, L, dim]
        unnorm = np.einsum("bl,blt->bt", coef, atoms)               # [B, dim]
        norm = np.linalg.norm(unnorm, axis=-1, keepdims=True)
        direction = unnorm / np.maximum(norm, 1e-12)

        degenerate = (norm[..., 0] < 1e-6)
        if np.any(degenerate):
            fallback = self.A[atom_idx[..., 0]]                     # [B, dim]
            direction = np.where(degenerate[:, None], fallback, direction)
        return direction


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_codebook(cfg):
    """Build the codebook selected by cfg.pmi_mode."""
    if cfg.pmi_mode == "random_unit_norm":
        return RandomUnitNormCodebook(cfg.codebook_size, cfg.num_bs_ant,
                                      cfg.seed)
    if cfg.pmi_mode == "type2_sparse_56bit":
        return Type2SparseCodebook(cfg)
    if cfg.pmi_mode == "genie":
        return GenieCodebook(cfg.num_bs_ant)
    raise ValueError(f"Unknown pmi_mode: {cfg.pmi_mode!r}")


if __name__ == "__main__":
    from config import Config

    cfg = Config(pmi_mode="type2_sparse_56bit")
    cb = Type2SparseCodebook(cfg)
    cb_r = RandomUnitNormCodebook(256, 32, seed=0)
    rng = np.random.default_rng(42)

    print("=" * 67)
    print("Thorough algorithmic validation of Type2SparseCodebook")
    print("=" * 67)

    # ===== Dictionary structure =====
    norms = np.linalg.norm(cb.A, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12), (norms.min(), norms.max())
    print(f"  [PASS]  T1  all {cb.n_atoms} atoms are unit-norm")

    n_beams = cb.A.shape[0] // cb.n_pol
    pol0 = cb.A[:n_beams]
    pol1 = cb.A[n_beams:]
    assert np.all(pol0[:, 16:] == 0), "pol0 atoms leak into pol1 slot"
    assert np.all(pol1[:, :16] == 0), "pol1 atoms leak into pol0 slot"
    assert np.all(np.abs(pol0[:, :16]).sum(axis=1) > 0)
    assert np.all(np.abs(pol1[:, 16:]).sum(axis=1) > 0)
    print(f"  [PASS]  T2  polarization block structure clean "
          f"(pol0 in [0:16], pol1 in [16:32])")

    aha_diag = np.einsum("kt,kt->k", np.conj(cb.A), cb.A).real
    assert np.allclose(aha_diag, 1.0, atol=1e-12)
    # also: inter-pol inner product is exactly 0
    cross = np.abs(pol0 @ pol1.conj().T).max()
    assert cross < 1e-12, f"pol0 x pol1 inner product max = {cross}"
    print(f"  [PASS]  T3  pol0 atoms orthogonal to pol1 atoms (inner product 0)")

    # natural DFT beams within a pol should be mutually orthogonal at the
    # natural (non-oversampled) grid points: m in {0,4,8,12} only
    # (because oversampled grid breaks orthogonality)
    natural_idx = np.array([m * (4 * 4) + n * 1
                            for m in (0, 4, 8, 12) for n in (0, 4, 8, 12)])
    gram_natural = np.abs(pol0[natural_idx] @ pol0[natural_idx].conj().T)
    off = gram_natural - np.eye(len(natural_idx))
    assert np.abs(off).max() < 1e-12, f"natural DFT beams not orthogonal: {np.abs(off).max()}"
    print(f"  [PASS]  T4  natural DFT beam sub-grid (non-oversampled) is orthonormal")

    # ===== OMP correctness =====
    for k in (0, 100, 255, 256, 400, 511):
        h = (cb.A[k] * (1.5 + 0.7j))[None, :]
        pmi, direction = cb.quantize(h)
        corr = float(np.abs(np.vdot(cb.A[k], direction[0])) ** 2)
        assert corr > 0.999, f"Atom {k}: direction_corr = {corr}"
        assert k in pmi.atom_idx[0].tolist(), \
            f"Atom {k} not picked (got {pmi.atom_idx[0].tolist()})"
    print(f"  [PASS]  T5  in-dictionary channels recover with direction_corr > 0.999")

    # Synthetic L-atom mixtures. NOTE: in an oversampled DFT dictionary,
    # adjacent atoms are ~90% correlated -> the RIP condition fails and
    # OMP is NOT guaranteed to recover the exact "true" atoms. The correct
    # quantitative check is that direction_corr is comfortably above
    # what the random codebook achieves (~0.18 on random h, T14).
    corrs = []
    for trial in range(40):
        atom_set = rng.choice(cb.n_atoms, size=cb.L, replace=False)
        weights = (rng.standard_normal(cb.L)
                   + 1j * rng.standard_normal(cb.L))
        h_syn = (weights[:, None] * cb.A[atom_set]).sum(axis=0)[None, :]
        pmi, direction = cb.quantize(h_syn)
        n_h = np.linalg.norm(h_syn)
        corrs.append(float(np.abs(np.vdot(h_syn[0], direction[0]) / n_h) ** 2))
    corrs = np.array(corrs)
    assert corrs.mean() > 0.5, \
        f"synthetic mixture: corr too low (mean {corrs.mean()})"
    print(f"  [PASS]  T6  synthetic L-atom mixtures: direction_corr "
          f"mean {corrs.mean():.4f} (range [{corrs.min():.3f},{corrs.max():.3f}])")

    h_batch = (rng.standard_normal((500, 32))
               + 1j * rng.standard_normal((500, 32)))
    pmi_batch, dir_batch = cb.quantize(h_batch)
    for i in range(500):
        u = np.unique(pmi_batch.atom_idx[i])
        assert len(u) == cb.L, f"trial {i} duplicates: {pmi_batch.atom_idx[i]}"
    print(f"  [PASS]  T7  500 random trials all produce {cb.L} distinct atoms")

    # ===== Quantization correctness =====
    assert pmi_batch.amp_idx.min() >= 0
    assert pmi_batch.amp_idx.max() <= cb.amp_levels - 1
    assert pmi_batch.phase_idx.min() >= 0
    assert pmi_batch.phase_idx.max() <= cb.phase_levels - 1
    print(f"  [PASS]  T8  amp_idx in [0,{cb.amp_levels-1}], "
          f"phase_idx in [0,{cb.phase_levels-1}]")

    max_per_row = pmi_batch.amp_idx.max(axis=-1)
    assert np.all(max_per_row == cb.amp_levels - 1), \
        f"some row's max amp_idx != {cb.amp_levels-1}: min={max_per_row.min()}"
    print(f"  [PASS]  T9  every (UE,RBG) report has at least one "
          f"amp_idx = {cb.amp_levels-1} (largest atom normalized to 1)")

    dnorms = np.linalg.norm(dir_batch, axis=-1)
    assert np.allclose(dnorms, 1.0, atol=1e-12)
    print(f"  [PASS] T10  reconstructed direction always unit-norm")

    # Controlled phase quantization
    test_phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    phi = test_phases % (2 * np.pi)
    phase_idx_ctrl = (np.rint(phi / (2 * np.pi / cb.phase_levels))
                      .astype(np.int64) % cb.phase_levels)
    assert phase_idx_ctrl.tolist() == [0, 1, 2, 3]
    print(f"  [PASS] T11  phase quantization maps "
          f"{{0, π/2, π, 3π/2}} -> [0,1,2,3] exactly")

    # ===== BS reconstruction consistency =====
    dir_bs = cb.reconstruct(pmi_batch)
    assert np.allclose(dir_batch, dir_bs, atol=1e-12)
    print(f"  [PASS] T12  BS reconstruction equals UE side bit-exact")

    # ===== CQI <-> h_hat magnitude round-trip =====
    sigma2 = 1.0
    p_rbg = 1.0
    h_test = (rng.standard_normal((20, 32))
              + 1j * rng.standard_normal((20, 32)))
    _, dir_test = cb.quantize(h_test)
    bf_gain = np.abs(np.einsum("bt,bt->b",
                               np.conj(h_test), dir_test)) ** 2
    cqi = np.log2(1.0 + p_rbg * bf_gain / sigma2)
    snr_back = 2 ** cqi - 1
    g_back = sigma2 * snr_back / p_rbg
    h_hat = np.sqrt(g_back)[:, None] * dir_test
    snr_via_h_hat = (p_rbg * np.abs(np.einsum("bt,bt->b",
                                              np.conj(h_hat), dir_test)) ** 2
                     / sigma2)
    assert np.allclose(snr_via_h_hat, snr_back, rtol=1e-9)
    print(f"  [PASS] T13  CQI <-> h_hat magnitude inversion exact "
          f"(BS recovers the SU SNR along the quantized direction)")

    # ===== Comparison: Type-II vs Random =====
    h_cmp = (rng.standard_normal((1000, 32))
             + 1j * rng.standard_normal((1000, 32)))
    h_cmp_u = h_cmp / np.linalg.norm(h_cmp, axis=-1, keepdims=True)
    _, dir_r = cb_r.quantize(h_cmp)
    _, dir_t = cb.quantize(h_cmp)
    cr = np.abs(np.einsum("bt,bt->b", np.conj(h_cmp_u), dir_r)) ** 2
    ct = np.abs(np.einsum("bt,bt->b", np.conj(h_cmp_u), dir_t)) ** 2
    assert ct.mean() > cr.mean() + 0.15
    print(f"  [PASS] T14  Type-II beats Random on direction_corr "
          f"({cr.mean():.4f} -> {ct.mean():.4f}, gap {ct.mean()-cr.mean():.3f})")

    print()
    print("=" * 67)
    print("All 14 algorithmic checks PASSED")
    print("=" * 67)
