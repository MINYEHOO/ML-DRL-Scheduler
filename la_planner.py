"""Common post-RZF link-adaptation planner (2026-07-13 redesign).

Single source of truth for B_tx sizing under ``la_mode='post_rzf'``, shared by
the env, the baselines and the PPO decoder, so that scheduler planned commits
and env actual commits match EXACTLY (audit Gate 2). Design fixed after three
review rounds with the external auditor:

- B_tx of every NEW unit on an RBG is sized from the PREDICTED post-RZF SINR
  of the FINAL group on that RBG (fixed retx members shape the precoder and
  interference but keep their historical B_tx).
- Prediction uses ONLY the fed-back channel estimates (h_hat) and the exact
  same precoder function/regularization/power as the actual transmission, so
  in the genie world (h_hat == h_true) prediction equals realization and new
  full-cap units first-ACK by construction (Gate 1).
- Groups are closed RBG-major: budgets are debited once per RBG AFTER the
  group is final -- no retroactive refunds, no scheduler/env drift.
- Epsilon-dropped members shrink the group; the close loop recomputes until
  stable (<= l_max iterations), which also removes 'ghost streams'.
"""
from __future__ import annotations

import numpy as np

from phy import rzf_precoder


def predict_group_link_adaptation(h_hat_rows: np.ndarray, noise_var: float,
                                  alpha: float, p_rbg: float, cfg):
    """Predicted post-RZF link adaptation for one FINAL RBG group.

    Parameters
    ----------
    h_hat_rows : [m, M_ant] complex -- BS channel estimates of the group
                 (retx members included; row order defines index order).

    Returns
    -------
    (W_pred [M_ant, m], sinr_pred [m], btx_cap [m])
    btx_cap = eta * N_RE * beta * log2(1 + sinr_pred)  (backlog cap is the
    caller's job). Identical math to the actual-transmission evaluation
    (phy._rbg_sinr) but on h_hat instead of h_true.
    """
    h = np.atleast_2d(np.asarray(h_hat_rows))
    w = rzf_precoder(h, alpha, p_rbg)                  # [M, m]
    eff = h @ w                                        # [m, m]
    power = np.abs(eff) ** 2
    desired = np.diag(power)
    interference = power.sum(axis=1) - desired
    sinr = desired / (interference + noise_var)
    # backoff: depth-wise beta_m (official; same first-ACK target for every
    # spatial mode) or the scalar la_beta (global-beta ablation / genie 1.0)
    if cfg.la_beta_by_depth:
        beta = cfg.la_beta_by_depth[min(h.shape[0], len(
            cfg.la_beta_by_depth)) - 1]
    else:
        beta = cfg.la_beta
    btx_cap = (beta * cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate
               * np.log2(1.0 + sinr))
    return w, sinr, btx_cap


class SlotAllocationPlanner:
    """RBG-major budget/B_tx accounting for one slot (post_rzf mode).

    Usage (identical on the scheduler side and inside the env):
        planner = SlotAllocationPlanner(cfg, h_hat_slot, noise_var, alpha,
                                        uncommitted0)
        for r in range(R):
            ... choose new users, consulting planner.remaining(u) ...
            kept, btx = planner.close_rbg(r, fixed_ues, new_ues)
    ``close_rbg`` returns the surviving new members (epsilon-dropped ones
    removed) with their exact B_tx, and debits per-UE budgets. The recorded
    plan (``planned_btx_map`` / ``planned_commit_per_ue``) is what Gate 2
    compares against the env's actually created units.
    """

    def __init__(self, cfg, h_hat_slot: np.ndarray, noise_var: float,
                 alpha: float, uncommitted0: np.ndarray):
        self.cfg = cfg
        self.h_hat_slot = h_hat_slot                    # [K, R, M]
        self.noise_var = float(noise_var)
        self.alpha = float(alpha)
        self._remaining = np.asarray(uncommitted0, dtype=np.float64).copy()
        self.planned_btx: dict[tuple[int, int], float] = {}   # (r, ue) -> B_tx
        self.sinr_pred: dict[tuple[int, int], float] = {}
        self.w_pred: dict[int, np.ndarray] = {}

    def remaining(self, ue: int) -> float:
        return float(self._remaining[ue])

    def close_rbg(self, r: int, fixed_ues, new_ues):
        """Finalize RBG r. Returns (kept_new_ues, {ue: b_tx})."""
        cfg = self.cfg
        # canonical member order: SINR is permutation-invariant only up to
        # float rounding (~1e-12), so fixed members are sorted here to make
        # scheduler-side and env-side predictions BIT-identical (Gate 2);
        # new members keep caller order, which both sides share (layer scan).
        fixed = sorted(set(int(u) for u in fixed_ues))
        new = [u for u in new_ues
               if self._remaining[u] > 0.0]             # no budget -> ghost
        while True:
            group = fixed + new
            if not group:
                self.w_pred[r] = None
                return [], {}
            h = self.h_hat_slot[group, r, :]
            w, sinr, caps = predict_group_link_adaptation(
                h, self.noise_var, self.alpha, cfg.p_rbg, cfg)
            btx, drop = {}, []
            for i, u in enumerate(group):
                if u in fixed:
                    continue                            # retx keeps old B_tx
                b = min(self._remaining[u], float(caps[i]))
                if b < cfg.b_tx_epsilon:
                    drop.append(u)
                    continue
                if self._remaining[u] - b < cfg.b_tx_epsilon:
                    b = self._remaining[u]              # swallow sub-eps tail
                btx[u] = b
            if not drop:
                self.w_pred[r] = w
                for i, u in enumerate(group):
                    self.sinr_pred[(r, u)] = float(sinr[i])
                for u, b in btx.items():
                    self._remaining[u] -= b
                    self.planned_btx[(r, u)] = b
                return [u for u in new], btx
            new = [u for u in new if u not in drop]     # shrink & recompute

    # ---- Gate-2 exports ----
    def planned_btx_map(self) -> dict:
        return dict(self.planned_btx)

    def planned_commit_per_ue(self, num_ue: int) -> np.ndarray:
        out = np.zeros(num_ue)
        for (_, u), b in self.planned_btx.items():
            out[u] += b
        return out


if __name__ == "__main__":
    # self-test: genie identity -- prediction on h == actual on h, and
    # full-cap sizing means delivered-per-slot >= B_tx (first-slot ACK).
    from config import Config
    from phy import _rbg_sinr, mi_bits
    rng = np.random.default_rng(0)
    cfg = Config()
    ok = 0
    for trial in range(200):
        m = int(rng.integers(1, 5))
        h = (rng.standard_normal((m, cfg.num_bs_ant))
             + 1j * rng.standard_normal((m, cfg.num_bs_ant)))
        h *= rng.uniform(0.3, 2.0, size=(m, 1))         # arbitrary, non-ortho
        nv = 1.0
        _, sinr_pred, caps = predict_group_link_adaptation(
            h, nv, nv, cfg.p_rbg, cfg)
        sinr_act = _rbg_sinr(h, h, nv, cfg.p_rbg, nv)   # genie: h_true == h_hat
        assert np.allclose(sinr_pred, sinr_act, rtol=1e-10), (trial, m)
        assert np.all(mi_bits(sinr_act, cfg) + 1e-9 >= caps), (trial, m)
        ok += 1
    print(f"genie identity + first-slot-deliverability: {ok}/200 groups OK")
