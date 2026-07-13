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

import numpy as np

from config import Config
from la_planner import SlotAllocationPlanner


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
