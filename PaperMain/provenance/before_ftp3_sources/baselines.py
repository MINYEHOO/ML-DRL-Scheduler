"""
baselines.py -- non-learning baseline schedulers.

Each baseline builds a [R, L] allocation by sequential layer-major selection
(spec sub-action order), querying the environment's helpers
(``position_candidates``, ``orthoscore_all``) -- the same helpers the Phase-2
PPO actor will use. Baselines never observe the true channel.

Schedulers (spec order):
  Random        -- uniformly random valid UE per position
  CQIGreedy     -- highest fed-back CQI
  PF            -- proportional fair: CQI / average throughput
  DeadlinePF    -- PF weighted by deadline urgency
  SUSPF         -- semi-orthogonal user selection + PF (orthogonality-aware)
"""

from __future__ import annotations

import math

import numpy as np

from config import Config
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation


class Scheduler:
    """Base class: layer-major sequential selection over the 32 positions."""

    name = "base"

    def schedule(self, env) -> np.ndarray:
        """Return a [R, L] allocation (0 = no-user, k>0 = UE index k-1).

        Selection is layer-major; a per-UE commit budget is threaded so a UE
        is not over-assigned across RBGs (mirrors env unit creation).
        """
        if env.cfg.decode_order == "rbg_major":
            return self._schedule_rbg_major(env)
        cfg: Config = env.cfg
        obs = env.get_observation()
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        occupied = obs["occupied"]
        closed = np.zeros(cfg.num_rbg, dtype=bool)   # SUS may close RBGs
        budget = obs["uncommitted"].astype(np.float64).copy()  # per-UE

        # UEs already in each RBG (preempted retransmissions)
        sel_ue = [set((occupied[r][occupied[r] > 0] - 1).tolist())
                  for r in range(cfg.num_rbg)]

        for l in range(cfg.l_max):
            for r in range(cfg.num_rbg):
                if occupied[r, l] > 0 or closed[r]:
                    continue                         # preempted / closed
                cand = env.position_candidates(r, sel_ue[r], budget)
                if not cand.any():
                    continue
                pick = self._select(env, obs, r, l, cand, sel_ue[r], closed)
                if pick is not None:
                    alloc[r, l] = pick + 1
                    sel_ue[r].add(pick)
                    budget[pick] -= env.estimate_btx(pick, r, budget[pick])
        return alloc

    def _schedule_rbg_major(self, env) -> np.ndarray:
        """RBG-major selection.

        Traversal mirrors the env's post_rzf creation scan exactly: fill an
        RBG's free layers top-down, close on the first no-pick (spec §5).

        post_rzf: budgets are threaded through the shared
        SlotAllocationPlanner and debited once per RBG with the FINAL group's
        B_tx, so the threaded budget equals the env's actual commit (Gate 2).
        The planner is kept on ``self.last_planner`` for the audit.

        legacy: same traversal/closure, but per-pick ``estimate_btx`` debits
        as in the layer-major path -- the order-effect CONTROL configuration
        (isolates traversal order from the LA change).
        """
        cfg: Config = env.cfg
        post_rzf = cfg.resolved_la_mode() == "post_rzf"
        obs = env.get_observation()
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        occupied = obs["occupied"]
        closed = np.zeros(cfg.num_rbg, dtype=bool)
        if post_rzf:
            planner = SlotAllocationPlanner(
                cfg, env.h_hat_slot, env.noise_var, env.noise_var,
                obs["uncommitted"].astype(np.float64))
            budget = planner._remaining
        else:
            planner = None
            budget = obs["uncommitted"].astype(np.float64).copy()
        self._slot_init(obs, cfg)
        # live view of the per-UE remaining budget (post_rzf: planner._remaining,
        # mutated in place at each close_rbg). Exposed so a mixin can thread it;
        # baselines that ignore it are unaffected.
        self._budget = budget
        for r in range(cfg.num_rbg):
            fixed = sorted(set((occupied[r][occupied[r] > 0] - 1).tolist()))
            sel_ue = set(fixed)
            entries = []
            for l in range(cfg.l_max):
                if occupied[r, l] > 0 or closed[r]:
                    continue
                cand = env.position_candidates(r, sel_ue, budget)
                pick = (self._select(env, obs, r, l, cand, sel_ue, closed)
                        if cand.any() else None)
                if pick is None:
                    closed[r] = True         # no-user closes the RBG (spec §5)
                    continue
                alloc[r, l] = pick + 1
                sel_ue.add(pick)
                entries.append(pick)
                if not post_rzf:
                    budget[pick] -= env.estimate_btx(pick, r, budget[pick])
            if post_rzf:
                _, btx = planner.close_rbg(r, fixed, entries)
                self._after_close(btx)
        self.last_planner = planner
        return alloc

    def _slot_init(self, obs, cfg) -> None:
        """Hook: per-slot state reset before rbg-major selection (vPF)."""

    def _after_close(self, btx: dict) -> None:
        """Hook: observe the finalized per-UE B_tx of a closed RBG (vPF)."""

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        """Return the chosen UE index, or None for no-user."""
        raise NotImplementedError

    @staticmethod
    def _argmax_masked(score: np.ndarray, cand: np.ndarray) -> int:
        masked = np.where(cand, score, -np.inf)
        return int(np.argmax(masked))


class Random(Scheduler):
    """Uniformly random valid UE per position."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        ues = np.where(cand)[0]
        return int(self.rng.choice(ues))


class CQIGreedy(Scheduler):
    """Pick the UE with the highest fed-back CQI on this RBG."""

    name = "CQI-greedy"

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        return self._argmax_masked(obs["cqi_fb"][:, r], cand)


class PF(Scheduler):
    """Proportional fair: CQI / average throughput."""

    name = "PF"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return obs["cqi_fb"][:, r] / np.maximum(obs["avg_throughput"],
                                                cfg.pf_epsilon)

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        return self._argmax_masked(self._pf_score(obs, r, env.cfg), cand)


class DeadlinePF(PF):
    """PF score weighted by deadline urgency (1 + eta_D / (deadline + 1))."""

    name = "Deadline-PF"

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        cfg = env.cfg
        urgency = 1.0 + cfg.eta_d / (obs["deadline"] + 1.0)
        score = self._pf_score(obs, r, cfg) * urgency
        return self._argmax_masked(score, cand)


class SUSPF(PF):
    """Semi-orthogonal user selection + PF.

    First stream in an RBG: best PF score. Later streams: restricted to UEs
    whose OrthoScore vs. the already-selected set clears a threshold; if none
    qualify, the RBG is closed (remaining layers stay empty).
    """

    name = "SUS+PF"

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        cfg = env.cfg
        score = self._pf_score(obs, r, cfg)
        if len(sel_ue) == 0:
            return self._argmax_masked(score, cand)        # first stream
        ortho = env.orthoscore_all(r, sel_ue)              # [K]
        ok = cand & (ortho >= cfg.sus_ortho_threshold)
        if not ok.any():
            closed[r] = True                               # stop filling RBG
            return None
        return self._argmax_masked(score, ok)


# --- SU-restricted variants (each RBG holds at most ONE UE => SU-MIMO); the
#     SU comparison anchors (full array/power to one UE, no MU interference) ---
def _su_restrict(BaseCls):
    """Wrap a baseline so each RBG is closed after its first UE (SU-MIMO)."""
    class SU(BaseCls):
        # naming rule (user, 2026-07-06): gate and metric join with '+'
        name = "SU+" + BaseCls.name
        def _select(self, env, obs, r, l, cand, sel_ue, closed):
            if len(sel_ue) > 0:
                closed[r] = True
                return None
            pick = super()._select(env, obs, r, l, cand, sel_ue, closed)
            if pick is not None:
                closed[r] = True
            return pick
    SU.__qualname__ = SU.__name__ = "SU" + BaseCls.__name__
    return SU


SUCQI = _su_restrict(CQIGreedy)
SUCQI.name = "SU+CQI"                     # symmetric with "SUS+CQI" (not "SU+CQI-greedy")
SUPF = _su_restrict(PF)                   # "SU+PF"
SUDeadlinePF = _su_restrict(DeadlinePF)   # "SU+Deadline-PF"
SURandom = _su_restrict(Random)           # "SU+Random"


class SUSCQI(SUSPF):
    """Semi-orthogonal user selection (orthogonality-aware) with a CQI-greedy
    metric instead of PF -- isolates the orthogonality contribution from the PF
    metric, i.e. the strongest orthogonality-aware, rate-maximizing MU bar."""

    name = "SUS+CQI"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return obs["cqi_fb"][:, r]


class SUSDeadlinePF(SUSPF):
    """SUS gate + deadline-urgency-weighted PF metric -- the SUS-MU
    counterpart of Deadline-PF in the 2x4 {mode x metric} baseline grid."""

    name = "SUS+Deadline-PF"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        urgency = 1.0 + cfg.eta_d / (obs["deadline"] + 1.0)
        return super()._pf_score(obs, r, cfg) * urgency


class MaxWeight(PF):
    """Blind MaxWeight: queue_bits x CQI -- the throughput-optimal reference
    metric of queueing-scheduling theory (Run4 baseline; in legacy one-packet
    mode queue_bits degenerates to the HOL backlog)."""

    name = "MW"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return (obs["queue_bits"] / cfg.b_norm) * obs["cqi_fb"][:, r]


def _edf_urgency(obs) -> np.ndarray:
    """EDF metric = min(HOL deadline, next-packet deadline) -- exactly the
    two deadlines the observation exposes (info parity with PPO). Service
    order stays FIFO; this only prioritizes WHICH UE gets drained so an
    urgent packet stuck behind the HOL is reached before it expires.
    next_deadline==0 means "no packet behind HOL" (legacy mode: always)."""
    dl, nxt = obs["deadline"], obs["next_deadline"]
    return np.where(nxt > 0, np.minimum(dl, nxt), dl)


class EDF(PF):
    """Blind earliest-deadline-first over the observable queue head."""

    name = "EDF"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return 1.0 / (_edf_urgency(obs) + 1.0)


class SUSMaxWeight(SUSPF):
    """SUS gate + MaxWeight metric (Run4 grid)."""

    name = "SUS+MW"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return (obs["queue_bits"] / cfg.b_norm) * obs["cqi_fb"][:, r]


class SUSEDF(SUSPF):
    """SUS gate + earliest-deadline metric (see _edf_urgency; Run4 grid)."""

    name = "SUS+EDF"

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return 1.0 / (_edf_urgency(obs) + 1.0)


SUMaxWeight = _su_restrict(MaxWeight)     # "SU+MW"  (Run4)
SUEDF = _su_restrict(EDF)                 # "SU+EDF" (Run4)


class SUSRandom(SUSPF):
    """SUS gate + uniformly random pick among ortho-qualifying candidates --
    the metric-free floor of the SUS-MU family."""

    name = "SUS+Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        cfg = env.cfg
        if len(sel_ue) == 0:
            return int(self.rng.choice(np.where(cand)[0]))
        ortho = env.orthoscore_all(r, sel_ue)
        ok = cand & (ortho >= cfg.sus_ortho_threshold)
        if not ok.any():
            closed[r] = True
            return None
        return int(self.rng.choice(np.where(ok)[0]))


class SUSPFVirtual(SUSPF):
    """SUS + PF with an IN-SLOT VIRTUAL PF update: after a UE is assigned, its
    throughput estimate is bumped (assume successful tx) so it is deprioritised
    for the remaining RBGs this slot -> spreads service within a slot. (Plain
    SUS+PF uses a fixed per-slot PF snapshot + a backlog-budget guard only, so a
    high-PF, data-rich UE can grab many RBGs.) The virtual increment (bits
    assigned) is a first-cut and can be tuned/scaled."""

    name = "SUS+vPF"

    def _slot_init(self, obs, cfg) -> None:
        self._vthr = np.maximum(obs["avg_throughput"].astype(np.float64),
                                cfg.pf_epsilon).copy()

    def _after_close(self, btx: dict) -> None:
        for u, b in btx.items():
            self._vthr[u] += b            # exact (finalized) in-slot PF bump

    def schedule(self, env) -> np.ndarray:
        if env.cfg.decode_order == "rbg_major":
            return self._schedule_rbg_major(env)
        cfg = env.cfg
        obs = env.get_observation()
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        occupied = obs["occupied"]
        closed = np.zeros(cfg.num_rbg, dtype=bool)
        budget = obs["uncommitted"].astype(np.float64).copy()
        # virtual per-UE throughput, mutated as UEs are served this slot
        self._vthr = np.maximum(obs["avg_throughput"].astype(np.float64),
                                cfg.pf_epsilon).copy()
        sel_ue = [set((occupied[r][occupied[r] > 0] - 1).tolist())
                  for r in range(cfg.num_rbg)]
        for l in range(cfg.l_max):
            for r in range(cfg.num_rbg):
                if occupied[r, l] > 0 or closed[r]:
                    continue
                cand = env.position_candidates(r, sel_ue[r], budget)
                if not cand.any():
                    continue
                pick = self._select(env, obs, r, l, cand, sel_ue[r], closed)
                if pick is not None:
                    alloc[r, l] = pick + 1
                    sel_ue[r].add(pick)
                    b = env.estimate_btx(pick, r, budget[pick])
                    budget[pick] -= b
                    self._vthr[pick] += b          # in-slot virtual PF update
        return alloc

    def _pf_score(self, obs, r, cfg: Config) -> np.ndarray:
        return obs["cqi_fb"][:, r] / self._vthr


# ---------------------------------------------------------------------------
# RPS-inspired deadline-feasible scheduling  (2026-08-24)
#
# RPS = Reward Per Second, Raviv & Leshem, "Scheduling for Multi-User
# Multi-Input Multi-Output Wireless Networks with Priorities and Deadlines",
# Future Internet 11(8):172, 2019. The original picks, from an EDF prefix
# window, the packet with the largest reward per expected transmission time
# among those that can still finish before their deadline.
#
# This is NOT an exact reproduction: the original assumes a single band, ZFBF,
# current CSI, non-preemptive packet service, exogenous packet priorities and
# no HARQ. Ours has 8 RBGs, RZF, quantized+aged CSI, per-slot re-selection with
# partial service, no exogenous priority (so p_u = 1) and retained IR-HARQ.
# Report as "RPS-inspired deadline-feasible scheduling adapted to RBG-based
# RZF MU-MIMO with retained HARQ".
#
#   feasibility  n_u = ceil(b_rem_u / max(b_hat_u, eps)) <= d_u
#   metric       M_u = 1{feasible} * b_hat_u / b_rem_u        (p_u = 1)
#   window       EDF prefix, W = 8 (the paper's value, not tuned)
#
# p_u is deliberately left at 1: folding the reward's urgency weight or miss
# penalty into it would make this a new custom utility rather than RPS.
#
# b_hat comes from SlotAllocationPlanner.solve_closure -- the full non-mutating
# closure (group RZF + link adaptation + epsilon-drop + budget cap), so a
# candidate is priced against the interference it inflicts, and evaluating a
# candidate never touches planner state or the budget.
# ---------------------------------------------------------------------------
class RPSBase(Scheduler):
    """Shared RPS machinery. Subclasses set the spatial policy."""

    W_RPS = 8              # EDF prefix window; the paper's value, not tuned
    name = "RPS-base"

    def _btx_of(self, planner, r, fixed, new, budget):
        """Non-mutating predicted service bits for every NEW member."""
        out = planner.solve_closure(r, fixed, new, remaining=budget)
        return out["btx"]

    @staticmethod
    def _need_slots(rem, b, eps):
        if b <= 0.0:
            return np.inf
        return math.ceil(rem / max(b, eps))

    def _edf_window(self, cand_idx, deadline):
        """EDF prefix: ascending remaining deadline, first min(W, |C|)."""
        order = sorted(cand_idx, key=lambda u: (float(deadline[u]), int(u)))
        return order[:min(self.W_RPS, len(order))]

    def _best_rps(self, env, obs, planner, r, fixed, new, cand, budget):
        """Return the admissible candidate with the largest RPS metric.

        Admissible = the candidate itself finishes in time AND every already
        selected NEW member still finishes in time in the trial group.
        Retained members are mandatory and are exempt from this rejection,
        but they are inside the RZF group so they do shape the SINR.
        """
        cfg = env.cfg
        dl = obs["deadline"].astype(np.float64)
        best = None
        for u in self._edf_window(np.flatnonzero(cand), dl):
            u = int(u)
            btx = self._btx_of(planner, r, fixed, new + [u], budget)
            b_u = float(btx.get(u, 0.0))
            if b_u <= 0.0:
                continue
            n_u = self._need_slots(float(budget[u]), b_u, cfg.b_tx_epsilon)
            if n_u > dl[u]:
                continue
            ok = True
            for v in new:                       # protect existing NEW members
                b_v = float(btx.get(v, 0.0))
                if self._need_slots(float(budget[v]), b_v,
                                    cfg.b_tx_epsilon) > dl[v]:
                    ok = False
                    break
            if not ok:
                continue
            m = b_u / max(float(budget[u]), cfg.b_tx_epsilon)
            # fixed tie-break: metric desc, deadline asc, n_req asc,
            #                  b_hat desc, ue index asc
            key = (m, -dl[u], -n_u, b_u, -u)
            if best is None or key > best[0]:
                best = (key, u)
        return None if best is None else best[1]

    def _fallback(self, env, planner, r, fixed, new, cand, budget):
        """Work-conserving fallback, PPO-aligned: an empty RBG that has a valid
        candidate may not stay empty (policy.py no-user rule), so if every RPS
        candidate is infeasible take the largest predicted service. Fixed in
        advance; never revised after seeing results."""
        best, bu = -1.0, None
        for u in np.flatnonzero(cand):
            u = int(u)
            b = float(self._btx_of(planner, r, fixed, new + [u],
                                   budget).get(u, 0.0))
            if b > best:
                best, bu = b, u
        return bu

    def _spatial_ok(self, env, obs, r, sel_ue, cand):
        return cand                                    # SU/SUS override

    def _schedule_rbg_major(self, env) -> np.ndarray:
        cfg: Config = env.cfg
        assert cfg.resolved_la_mode() == "post_rzf", "RPS requires post_rzf"
        obs = env.get_observation()
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        occupied = obs["occupied"]
        planner = SlotAllocationPlanner(
            cfg, env.h_hat_slot, env.noise_var, env.noise_var,
            obs["uncommitted"].astype(np.float64))
        budget = planner._remaining        # live; debited at each close_rbg

        for r in range(cfg.num_rbg):
            fixed = sorted(set((occupied[r][occupied[r] > 0] - 1).tolist()))
            free = int((occupied[r] == 0).sum())
            new = []
            while len(fixed) + len(new) < cfg.l_max and len(new) < free:
                if not self._may_add(fixed, new):
                    break
                cand = env.position_candidates(r, set(fixed) | set(new), budget)
                if new or fixed:
                    cand = self._spatial_ok(env, obs, r, set(fixed) | set(new),
                                            cand)
                if not cand.any():
                    break
                pick = self._best_rps(env, obs, planner, r, fixed, new, cand,
                                      budget)
                if pick is None:
                    if not fixed and not new:
                        pick = self._fallback(env, planner, r, fixed, new,
                                              cand, budget)
                        if pick is None:
                            break
                    else:
                        break                  # nonempty group -> leave empty
                new.append(int(pick))
            kept, _ = planner.close_rbg(r, fixed, new)
            free_slots = [l for l in range(cfg.l_max) if occupied[r, l] == 0]
            for l, u in zip(free_slots, kept):
                alloc[r, l] = u + 1
        self.last_planner = planner
        return alloc

    def _may_add(self, fixed, new) -> bool:
        raise NotImplementedError

    def schedule(self, env) -> np.ndarray:
        return self._schedule_rbg_major(env)


class SURPS(RPSBase):
    """SU-RPS: at most one UE per RBG, retained counting as that one."""

    name = "SU-RPS"

    def _may_add(self, fixed, new) -> bool:
        return len(fixed) + len(new) < 1


class SUSRPS(RPSBase):
    """SUS-RPS (PRIMARY traffic-aware comparator, fixed before evaluation).

    Keeps the SUS spatial-compatibility gate, ranks the eligible candidates by
    the RPS metric instead of CQI/PF.
    """

    name = "SUS-RPS"

    def _may_add(self, fixed, new) -> bool:
        return len(fixed) + len(new) < 4

    def _spatial_ok(self, env, obs, r, sel_ue, cand):
        if not sel_ue:
            return cand
        ortho = env.orthoscore_all(r, sel_ue)
        return cand & (ortho >= env.cfg.sus_ortho_threshold)


# ---------------------------------------------------------------------------
# PF-FDS + PF-Greedy SDS  (Kela-style, adapted)  2026-08-24
#
# Literature baseline: the non-learned heuristic Kela et al., "From Simulation
# to Practice: Generalizable Deep Reinforcement Learning for Cellular
# Schedulers" (arXiv:2411.08529, Appendix E.2) compare their DRL scheduler
# against. Two stages:
#
#   PF-FDS  -- the first (SU) layer of an empty RBG goes to the UE maximising
#              predicted SU rate / past average throughput.
#   PF-SDS  -- each additional spatial layer is chosen by trying EVERY
#              remaining candidate, recomputing RZF + link adaptation for the
#              WHOLE trial group, admitting only candidates that strictly raise
#              the group's predicted sum throughput, and among those taking the
#              largest PF sum. Stops when no candidate improves sum throughput
#              -> the MU rank is adaptive, not fixed at 4.
#
# This is NOT an optimum: the first UE is fixed by FDS, members are never
# removed or swapped, there is no look-ahead, and rank-1..4 combinations are
# not enumerated. It is an exhaustive candidate search PER ADDED LAYER.
# Report it as "PF-FDS + PF-Greedy SDS (Kela-style, adapted)" -- the paper
# does not publish full pseudocode, tie-breaking or the PF details.
#
# Adaptation to our retained-HARQ structure: retained members are mandatory
# and are never removed or re-placed, but they DO enter the RZF group matrix,
# so a candidate is priced against the interference it inflicts on them. No
# hidden HARQ state (target payload / accumulated MI / attempt count) is read.
#
# R_hat is the group's predicted per-member cap
# beta_|G| * eta * N_RE * log2(1 + SINR_post-RZF), i.e. exactly the quantity
# the env will size new units from; the per-UE budget cap is applied at commit
# by close_rbg, as for every other scheduler.
#
# Tie-break, fixed in advance: larger PF sum, then larger sum throughput, then
# smaller UE index. EPS_T is a float-comparison guard, not a tuned parameter.
# ---------------------------------------------------------------------------
class PFGreedySDS(Scheduler):
    """PF-FDS first layer + PF-Greedy spatial-domain scheduling."""

    name = "PF-Greedy-SDS"
    EPS_T = 1e-9          # float guard on "strictly increases sum throughput"
    # PF denominator floor: use the repository's cfg.pf_epsilon, as PF/SUS+PF
    # do. A token 1e-9 floor is NOT equivalent -- avg_throughput is exactly 0
    # for the ~14/32 UEs that have not been served yet, so a token floor lets
    # their PF weight reach 1e9 and they capture every RBG regardless of
    # channel quality.

    def _pf_denom(self, obs) -> np.ndarray:
        """PF denominator: past average throughput, floored at cfg.pf_epsilon
        exactly as PF/SUS+PF do. Overridden by the sum-rate variant."""
        return obs["avg_throughput"].astype(np.float64)

    def _rates(self, planner, r, group):
        """R_hat for every member of `group`, in the caller's order."""
        if not group:
            return np.zeros(0)
        h = planner.h_hat_slot[list(group), r, :]
        _, _, caps = predict_group_link_adaptation(
            h, planner.noise_var, planner.alpha, planner.cfg.p_rbg,
            planner.cfg)
        return np.asarray(caps, dtype=np.float64)

    def _schedule_rbg_major(self, env) -> np.ndarray:
        cfg: Config = env.cfg
        assert cfg.resolved_la_mode() == "post_rzf", \
            "PFGreedySDS requires la_mode='post_rzf'"
        obs = env.get_observation()
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        occupied = obs["occupied"]
        planner = SlotAllocationPlanner(
            cfg, env.h_hat_slot, env.noise_var, env.noise_var,
            obs["uncommitted"].astype(np.float64))
        budget = planner._remaining
        rbar = np.maximum(self._pf_denom(obs), cfg.pf_epsilon)

        for r in range(cfg.num_rbg):
            fixed = sorted(set((occupied[r][occupied[r] > 0] - 1).tolist()))
            free = int((occupied[r] == 0).sum())
            entries = []

            # ---- PF-FDS: first SU layer of an EMPTY rbg -------------------
            if not fixed and free > 0:
                cand = env.position_candidates(r, set(), budget)
                if cand.any():
                    idx = np.flatnonzero(cand)
                    su = np.array([self._rates(planner, r, [int(u)])[0]
                                   for u in idx])
                    pf = su / rbar[idx]
                    entries.append(int(idx[int(np.argmax(pf))]))

            # ---- PF-Greedy SDS: additional spatial layers -----------------
            while len(fixed) + len(entries) < len(fixed) + free:
                grp = fixed + entries
                if not grp:
                    break
                cur = float(self._rates(planner, r, grp).sum())
                cand = env.position_candidates(r, set(grp), budget)
                if not cand.any():
                    break
                best = None                      # (pf_sum, sum_rate, -ue)
                for u in np.flatnonzero(cand):
                    trial = grp + [int(u)]
                    rt = self._rates(planner, r, trial)
                    tot = float(rt.sum())
                    if tot <= cur + self.EPS_T:  # must strictly improve
                        continue
                    pf = float((rt / rbar[list(trial)]).sum())
                    key = (pf, tot, -int(u))
                    if best is None or key > best[0]:
                        best = (key, int(u))
                if best is None:
                    break                        # no candidate improves T
                entries.append(best[1])

            kept, _ = planner.close_rbg(r, fixed, entries)
            free_slots = [l for l in range(cfg.l_max) if occupied[r, l] == 0]
            for l, u in zip(free_slots, kept):
                alloc[r, l] = u + 1
        self.last_planner = planner
        return alloc

    def schedule(self, env) -> np.ndarray:
        return self._schedule_rbg_major(env)


class CQIGreedySDS(PFGreedySDS):
    """Same engine, sum-rate metric instead of PF (declared in advance).

    Our reward is deadline-driven and PF is throughput-fair; SUS+PF already
    scores 1087 against SUS+CQI's 4194 on the reserved seeds, so the PF metric
    is known to be handicapped in this world BEFORE any run. This variant
    isolates the SDS SEARCH from the PF METRIC and is reported alongside, not
    selected after the fact.
    """

    name = "SumRate-Greedy-SDS"

    def _pf_denom(self, obs):
        return np.ones(obs["avg_throughput"].shape, dtype=np.float64)


# ---------------------------------------------------------------------------
# Deadline-feasibility filtering (2026-08-24)
#
# The existing deadline-aware baselines all push resources TOWARD a UE whose
# head-of-line packet is close to expiry (Deadline-PF: score *= 1 + eta_D/(d+1);
# EDF: order by d alone). Measured on the reserved 100 seeds they are the worst
# of the grid -- SUS+DPF reward 1086 at 12.52% waste against SUS+CQI's 4194 at
# 6.09% -- because under overload "the deadline is near" mostly means "this
# packet is already lost": the bits go out, the packet still misses, and the UE
# that was displaced misses too.
#
# The untested response to the SAME observable information is the opposite one:
# do not schedule a packet that cannot finish in time. With
#     need_k = uncommitted_k   (HOL bits still owed, observed)
#     d_k    = deadline_k      (slots left, observed)
#     b_k    = predict_b_tx(cqi_fb[k, r])   (this RBG's SU rate estimate)
# the packet is infeasible when ceil(need_k / b_k) > d_k, and those UEs are
# removed from the candidate mask. No threshold, no weight, no tuning: every
# term is either an observation or an existing predictor, so the filter is
# fully specified before it is ever run.
#
# This is a CANDIDATE FILTER, not a search: the SUS gate, the metric and the
# fill order are untouched, which is what makes it a clean isolation of the
# waste mechanism (PPO 2.29% vs SUS+CQI 6.09%).
# ---------------------------------------------------------------------------
def _feasible_mask(env, obs, r, budget) -> np.ndarray:
    """[K] bool: HOL packet can still finish by its deadline at this RBG's rate."""
    cfg = env.cfg
    unc = (obs["uncommitted"].astype(np.float64) if budget is None
           else np.asarray(budget, dtype=np.float64))
    b = np.array([env.estimate_btx(u, r, unc[u]) for u in range(cfg.num_ue)])
    d = obs["deadline"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        need_slots = np.ceil(np.divide(unc, np.maximum(b, cfg.b_tx_epsilon)))
    return need_slots <= d


class FeasibleMixin:
    """Drop candidates whose HOL packet cannot make its deadline.

    If the filter empties the candidate set the RBG is left as the underlying
    scheduler would leave it with no candidates -- transmitting only doomed
    bits is strictly worse than transmitting none.
    """

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        # thread the LIVE remaining budget: using the slot-start uncommitted
        # would overstate `need` for a UE already served on an earlier RBG and
        # wrongly mark it infeasible
        ok = cand & _feasible_mask(env, obs, r, getattr(self, "_budget", None))
        if not ok.any():
            closed[r] = True
            return None
        return super()._select(env, obs, r, l, ok, sel_ue, closed)


class SUSCQIFeasible(FeasibleMixin, SUSCQI):
    name = "SUS+CQI-Feasible"


class SUSDeadlinePFFeasible(FeasibleMixin, SUSDeadlinePF):
    """The filter applied to the urgency-weighted metric: does bounding the
    waste flip the sign of urgency weighting, which is harmful on its own?"""
    name = "SUS+DPF-Feasible"


class SUCQIFeasible(FeasibleMixin, SUCQI):
    name = "SU+CQI-Feasible"


# ---------------------------------------------------------------------------
# Depth-capped SUS+CQI (2026-08-24)
#
# The measured story is that the learned policy's edge is DEPTH RESTRAINT:
# MU rank 2.71-2.78 against SUS+CQI's 3.93, waste (1 - goodput/throughput)
# 1.9-2.3% against 6.1%. That leaves one question the existing grid cannot
# answer: is the gain "use ~3 layers instead of 4" -- a single CONSTANT, no
# learning required -- or does it need the PER-SLOT adaptivity?
#
# Capping SUS+CQI at a fixed rank m separates the two. If some fixed m closes
# most of the gap, the contribution reduces to picking that integer. If no
# fixed m does, the adaptivity is the mechanism.
#
# `sel_ue` already contains the retained HARQ members (the rbg-major loop
# seeds it with them), so the cap bounds the TOTAL group size, which is what
# mu_depth measures. m is the only hyperparameter and is an integer.
# ---------------------------------------------------------------------------
class SUSCQIDepthCap(SUSCQI):
    """SUS+CQI that closes an RBG once the group reaches ``max_depth``."""

    max_depth = 4

    def _select(self, env, obs, r, l, cand, sel_ue, closed):
        if len(sel_ue) >= self.max_depth:
            closed[r] = True
            return None
        return super()._select(env, obs, r, l, cand, sel_ue, closed)


def _depth_capped(m: int):
    cls = type(f"SUSCQIDepth{m}", (SUSCQIDepthCap,),
               {"max_depth": m, "name": f"SUS+CQI@m={m}"})
    return cls


SUSCQIDepth1 = _depth_capped(1)
SUSCQIDepth2 = _depth_capped(2)
SUSCQIDepth3 = _depth_capped(3)
SUSCQIDepth4 = _depth_capped(4)


def all_baselines(cfg: Config) -> list:
    """2x4 factorial baseline grid (2026-07-02 restructure, user decision):
      {spatial mode: SUS-gated MU | SU} x {metric: CQI, Deadline-PF, PF, Random}

    Blind-MU variants (CQIGreedy/PF/DeadlinePF/Random -- packing with no
    orthogonality gate) and SUS+vPF stay defined for analysis but are dropped
    from the default comparison set: blind MU is a strawman no deployed
    scheduler uses, and vPF's in-slot spread only helps at loose thresholds
    (evidence: Run3/BASELINE_AUDIT_2026-07-02.md).

    Random variants keep offset seeds (cfg.seed + 100003/100004) disjoint from
    the episode family (7919*idx) and PPO minibatch-shuffle (31337 + update_idx).

    Queue mode (cfg.queue_size > 1, Run4) extends the grid to 2x6 with the
    queue-native metrics: MaxWeight (queue_bits x CQI) and EDF.
    """
    grid = [SUSCQI(),           # MU(SUS), CQI (rate) metric
            SUSDeadlinePF(),    # MU(SUS), deadline-weighted PF
            SUSPF(),            # MU(SUS), PF
            SUSRandom(seed=cfg.seed + 100_003),  # MU(SUS), random
            SUCQI(),            # SU, CQI
            SUDeadlinePF(),     # SU, deadline-weighted PF
            SUPF(),             # SU, PF
            SURandom(seed=cfg.seed + 100_004)]   # SU, random
    if cfg.queue_size > 1:
        grid[4:4] = [SUSMaxWeight(), SUSEDF()]   # keep MU block contiguous
        grid += [SUMaxWeight(), SUEDF()]
    return grid


if __name__ == "__main__":
    from config import debug_config
    from env import SchedulerEnv

    cfg = debug_config()
    env = SchedulerEnv(cfg)

    for sched in all_baselines(cfg):
        env.reset(episode_idx=0)
        done = False
        total_r = 0.0
        while not done:
            alloc = sched.schedule(env)
            assert alloc.shape == (cfg.num_rbg, cfg.l_max)
            assert alloc.min() >= 0 and alloc.max() <= cfg.num_ue
            _, reward, done, _ = env.step(alloc)
            total_r += reward
        ep = env.ep
        print(f"  {sched.name:12s}: reward {total_r:8.1f}  "
              f"comp {ep['n_comp']:3d}  miss {ep['n_miss_deadline']:2d}  "
              f"retx_drop {ep['n_retx_drop']:2d}  "
              f"acked {ep['acked_bits']/1e6:5.2f} Mbit")
    print("baselines.py smoke test passed.")
