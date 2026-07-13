"""
env.py -- Gym-style environment for the MIMO-OFDM scheduler.

This module implements *environment dynamics only*. It does NOT pick actions:
a scheduler (baselines.py, or the Phase-2 PPO actor) builds the allocation and
passes it to ``step``. The env provides observations and the helpers a
scheduler needs (``orthoscore_all``, ``position_candidates``).

One ``step`` == one scheduling slot. The action is an allocation matrix
[R, L] with allocation[r, l] in {0..K}: 0 = no-user, k>0 = UE index k-1.
Positions held by a pending (NACKed) retransmission are force-overwritten by
the env regardless of the action.

Per-slot transition order (spec):
  prepare: true channel -> CSI feedback -> Age -> traffic arrival
  step   : RZF/SINR -> MI accumulation -> backlog -> completion
           -> deadline -> miss/drop -> avg throughput -> reward
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from config import Config
from codebook import make_codebook
from channel import ChannelGenerator
from csi import CSIFeedbackBuffer, precompute_episode_csi
from traffic import TrafficModel
from transmission import TransmissionManager
from phy import reconstruct_h_hat, compute_slot_sinr, predict_b_tx
from la_planner import SlotAllocationPlanner


class SchedulerEnv:
    """Single-cell MIMO-OFDM downlink scheduling environment."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.codebook = make_codebook(cfg)
        self.channel = ChannelGenerator(cfg)
        self.csi = CSIFeedbackBuffer(cfg)
        self.traffic = TrafficModel(cfg)
        self.txmgr = TransmissionManager(cfg)

        # episode_seed -> EpisodeCSI, LRU-bounded with the same policy/cap
        # as ChannelGenerator._cache (entries pair 1:1 with channel entries;
        # the cap must exceed ppo_eval_episodes + ppo_eval_every so eval
        # entries survive the training inserts between two eval rounds).
        self._csi_cache: OrderedDict = OrderedDict()
        self._csi_cache_max = 16

        # episode state (filled by reset)
        self.rng = None
        self.slot = 0
        self.noise_var = 1.0
        self._episode_csi = None
        self.h_true_slot = None
        self.h_hat_slot = None
        cfg.validate_la()                  # fail fast on illegal LA combos
        self.last_env_planner = None       # post_rzf: shared-planner record
        self.last_actual_btx = {}          # post_rzf: (rbg, ue) -> B_tx (Gate 2)
        self.avg_throughput = np.zeros(cfg.num_ue)
        self.cum_acked_bits = np.zeros(cfg.num_ue)
        self._deadline_snap = np.zeros(cfg.num_ue)
        self.ep = {}

        # filled by compact_pending() at every slot start
        self.fixed_allocation = np.zeros((cfg.num_rbg, cfg.l_max),
                                          dtype=np.int64)
        self.fixed_mask = np.zeros((cfg.num_rbg, cfg.l_max), dtype=bool)
        self.fixed_unit_map = np.full((cfg.num_rbg, cfg.l_max), -1,
                                       dtype=np.int64)
        self.initial_S_r: list[set] = [set() for _ in range(cfg.num_rbg)]

        # overflow drops in the CURRENT slot (used in _finish_slot reward)
        self._overflow_drop_this_slot: int = 0
        self._buffer_overflow_this_slot: int = 0   # queue mode; 0 in legacy

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, episode_idx: int = 0) -> dict:
        """Start a new episode; returns the slot-0 observation."""
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed + 7919 * episode_idx)

        # new topology + whole-episode true channel
        episode_seed = cfg.seed + episode_idx
        h_true_episode = self.channel.reset(episode_seed=episode_seed)
        # per-UE speed assignment for this episode (km/h); for per-user analysis
        self.ue_speeds_kmh = np.asarray(self.channel.ue_speeds_kmh,
                                        dtype=np.float64).copy()

        # whole-episode true CSI + noise calibration, computed once per
        # seed and reused across slots / seed revisits (see csi.EpisodeCSI)
        ep_csi = self._csi_cache.get(episode_seed)
        if ep_csi is None:
            ep_csi = precompute_episode_csi(h_true_episode, self.codebook, cfg)
            self._csi_cache[episode_seed] = ep_csi
            if len(self._csi_cache) > self._csi_cache_max:
                self._csi_cache.popitem(last=False)   # evict LRU
        else:
            self._csi_cache.move_to_end(episode_seed)
        self._episode_csi = ep_csi
        self.noise_var = ep_csi.sigma2

        self.slot = 0
        self.traffic.reset()
        # Level-2 mixed-load: activate a random n_active of the K UEs this episode
        # (rest idle) so one policy sees varying contention (deterministic per seed).
        if cfg.n_active_max > 0:
            _rl = np.random.default_rng(cfg.seed + 4441 * episode_idx)
            self.traffic.n_active = int(_rl.integers(cfg.n_active_min,
                                                     cfg.n_active_max + 1))
        else:
            self.traffic.n_active = cfg.num_ue
        # Level-2 mixed-arrival: random per-episode p_arrival (dedicated RNG
        # stream like n_active above -> zero impact on all other draws;
        # decorrelates load from n_active so load must be READ from the queue
        # observations, not inferred from the active-UE count).
        if cfg.p_arrival_max > 0:
            _rp = np.random.default_rng(cfg.seed + 7919 * episode_idx)
            self.traffic.p_arrival_ep = float(_rp.uniform(cfg.p_arrival_min,
                                                          cfg.p_arrival_max))
        else:
            self.traffic.p_arrival_ep = None
        self.txmgr.reset()
        self.avg_throughput = np.zeros(cfg.num_ue)
        self.cum_acked_bits = np.zeros(cfg.num_ue)
        K = cfg.num_ue
        self.ep = dict(acked_bits=0.0, n_comp=0, n_miss_deadline=0,
                       n_retx_drop=0, n_retx_overflow_drop=0,
                       n_buffer_overflow=0,          # queue mode: full-buffer rejects
                       delays=[],                    # per-completed-packet delay [slots]
                       qlen_sum=0.0, qlen_count=0,   # queue-length sampling (queue mode)
                       n_arrivals=0, reward=0.0,
                       sinr_sum=0.0, sinr_count=0,
                       dir_corr_sum=0.0, dir_corr_count=0,
                       # --- HARQ/LA instrumentation (2026-07-13; pure
                       # counters, no reward/RNG effect; extra columns are
                       # exported only for post_rzf runs)
                       n_units_new=0, n_units_first_ack=0,
                       n_units_acked=0, attempts_acked_sum=0,
                       units_by_depth=np.zeros(4, np.int64),
                       first_ack_by_depth=np.zeros(4, np.int64),
                       pinned_pos_sum=0, sched_pos_sum=0,
                       completed_bits=0.0,
                       # --- per-UE accumulators (for per-user / per-speed analysis)
                       acked_per_ue=np.zeros(K), comp_per_ue=np.zeros(K, np.int64),
                       miss_per_ue=np.zeros(K, np.int64),
                       retx_per_ue=np.zeros(K, np.int64),
                       sched_slots_per_ue=np.zeros(K, np.int64),
                       paired_depth_sum_per_ue=np.zeros(K))

        self._prepare_slot(self.slot)
        return self.get_observation()

    def _prepare_slot(self, slot: int) -> None:
        """true channel -> CSI feedback -> Age -> retx compaction -> arrival."""
        cfg = self.cfg
        self.h_true_slot = self.channel.get_channel(slot)
        # slot-t lookup into the per-episode precompute -- identical values
        # to running generate_true_csi on h_true_slot (OMP is deterministic)
        raw_pmi_true, direction_true, cqi_true = self._episode_csi.slot(slot)

        if slot == 0:
            self.csi.reset(raw_pmi_true, direction_true, cqi_true)  # forced
        else:
            self.csi.step(raw_pmi_true, direction_true, cqi_true,
                          self.rng)                                  # Bernoulli

        # --- pending retransmission compaction (Phase 2 spec) ---
        result = self.txmgr.compact_pending(self.traffic)
        self.fixed_allocation = result["fixed_allocation"]
        self.fixed_mask = result["fixed_mask"]
        self.fixed_unit_map = result["fixed_unit_map"]
        self.initial_S_r = result["initial_S_r"]
        # track per-slot overflow drops for reward penalty
        self._overflow_drop_this_slot = len(result["overflow_drops"])
        for pid, ue in result["overflow_drops"]:
            self._remove_packet(ue, pid)
            self.ep["n_retx_overflow_drop"] += 1

        new_ues, n_ovf = self.traffic.generate_arrivals(slot, self.rng)
        self.ep["n_arrivals"] += len(new_ues)
        # queue mode: arrivals rejected by a full buffer (0 in legacy mode);
        # penalized like miss in this slot's reward (retx-overflow precedent)
        self._buffer_overflow_this_slot = n_ovf
        self.ep["n_buffer_overflow"] += n_ovf

        # BS channel estimate for precoding (from current feedback buffer)
        self.h_hat_slot = reconstruct_h_hat(
            self.csi.direction_fb, self.csi.cqi_fb,
            cfg.p_rbg, self.noise_var)

        # PMI fidelity: how well the BS's current direction belief
        # matches the current true channel direction (captures both PMI
        # quantization error AND CSI staleness)
        norm = np.linalg.norm(self.h_true_slot, axis=-1, keepdims=True)
        h_true_dir = self.h_true_slot / np.maximum(norm, 1e-12)
        corr = np.abs(np.einsum("krt,krt->kr", np.conj(h_true_dir),
                                self.csi.direction_fb)) ** 2
        self.ep["dir_corr_sum"] += float(corr.mean())
        self.ep["dir_corr_count"] += 1

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, allocation: np.ndarray):
        """Apply a [R, L] allocation for the current slot.

        Returns
        -------
        obs : dict        next-slot observation (stale if done)
        reward : float
        done : bool
        info : dict
        """
        reward, info = self._finish_slot(np.asarray(allocation, dtype=np.int64))

        self.slot += 1
        done = self.slot >= self.cfg.episode_len
        if not done:
            self._prepare_slot(self.slot)
        return self.get_observation(), reward, done, info

    def _finish_slot(self, allocation: np.ndarray):
        cfg = self.cfg

        # remaining deadline during this slot (for reward weighting)
        self._deadline_snap = np.array(
            [p.deadline if p is not None else 0
             for p in self.traffic.packets], dtype=np.float64)

        # realize the allocation: force preempted, validate, create new units
        realized = self._sanitize_and_create(allocation)

        # RZF / SINR (precoder from h_hat, evaluation on true channel)
        sinr_map = compute_slot_sinr(realized, self.h_true_slot,
                                     self.h_hat_slot, cfg.p_rbg,
                                     self.noise_var, alpha=self.noise_var)
        occ = realized > 0
        if occ.any():
            self.ep["sinr_sum"] += float(sinr_map[occ].sum())
            self.ep["sinr_count"] += int(occ.sum())

        # per-UE scheduling depth: for each RBG this UE is in, record the RBG's
        # UE-count (MU depth) so we can later tell SU (~1) from MU (>1) per user
        for r in range(cfg.num_rbg):
            row = realized[r]
            ues_r = np.unique(row[row > 0]) - 1            # 0-based UE ids
            d = int(ues_r.size)
            for u in ues_r:
                self.ep["sched_slots_per_ue"][u] += 1
                self.ep["paired_depth_sum_per_ue"][u] += d

        # MI accumulation -> ACK / NACK / drop
        outcome = self.txmgr.process_slot(sinr_map)

        # --- HARQ/LA instrumentation (counters only) ---
        depth_r = [int((np.unique(realized[r][realized[r] > 0])).size)
                   for r in range(cfg.num_rbg)]
        for un in outcome.acked:
            self.ep["n_units_acked"] += 1
            self.ep["attempts_acked_sum"] += un.tx_attempts
        for un in (outcome.acked + outcome.dropped + self.txmgr.units):
            if un.tx_attempts == 1:                    # created this slot
                self.ep["n_units_new"] += 1
                d = min(max(depth_r[un.rbg_id], 1), 4) - 1
                self.ep["units_by_depth"][d] += 1
                if un.is_acked:
                    self.ep["n_units_first_ack"] += 1
                    self.ep["first_ack_by_depth"][d] += 1
        self.ep["pinned_pos_sum"] += int(self.fixed_mask.sum())
        self.ep["sched_pos_sum"] += int(occ.sum())

        # --- retx-drop: drop the whole parent packet ---
        dropped_pids = set()
        n_retx_drop = 0
        for unit in outcome.dropped:
            if unit.packet_id in dropped_pids:
                continue
            dropped_pids.add(unit.packet_id)
            n_retx_drop += 1
            self.ep["retx_per_ue"][unit.ue_id] += 1
            self._remove_packet(unit.ue_id, unit.packet_id)

        # --- ACK: credit delivered bits, skip dropped packets ---
        acked_bits = np.zeros(cfg.num_ue)
        for unit in outcome.acked:
            if unit.packet_id in dropped_pids:
                continue
            pkt = self.traffic.packets[unit.ue_id]
            if pkt is not None and pkt.packet_id == unit.packet_id:
                pkt.ack(unit.b_tx)
                acked_bits[unit.ue_id] += unit.b_tx

        # --- completion (HOL only; next queued packet promotes on removal) ---
        n_comp = 0
        for u in range(cfg.num_ue):
            pkt = self.traffic.packets[u]
            if pkt is not None and pkt.is_complete:
                n_comp += 1
                self.ep["comp_per_ue"][u] += 1
                self.ep["completed_bits"] += float(pkt.size)
                self.ep["delays"].append(self.slot - pkt.arrival_slot + 1)
                self._remove_packet(u, pkt.packet_id)

        # --- short-term reward (before deadline decrement) ---
        weight = 1.0 + cfg.eta_d / (self._deadline_snap + 1.0)
        r_short = float(np.sum(weight * outcome.useful_per_ue) / cfg.b_norm)

        # --- deadline decrement + miss ---
        # deadlines tick for ALL queued packets (queue mode); HOL expiry needs
        # unit cleanup, and a promoted successor may itself already be expired
        # (hence the while); expired NON-HOL packets hold no units and are
        # swept inside the traffic model.
        self.traffic.decrement_deadlines()
        n_miss_deadline = 0
        for u in range(cfg.num_ue):
            while True:
                pkt = self.traffic.packets[u]
                if pkt is None or pkt.deadline > 0:
                    break
                n_miss_deadline += 1
                self.ep["miss_per_ue"][u] += 1
                self._remove_packet(u, pkt.packet_id)
        for u in self.traffic.pop_expired_queued():
            n_miss_deadline += 1
            self.ep["miss_per_ue"][u] += 1

        # --- total reward ---
        # overflow drops are also packet failures: penalize same as miss/retx-drop
        # (retx compaction overflow AND, in queue mode, full-buffer rejects)
        n_miss = (n_miss_deadline + n_retx_drop
                  + self._overflow_drop_this_slot
                  + self._buffer_overflow_this_slot)
        reward = (cfg.lambda_s * r_short
                  + cfg.lambda_c * n_comp
                  - cfg.lambda_m * n_miss)

        # --- average throughput EWMA + cumulative (ACKed bits only) ---
        a = 1.0 / cfg.t_c
        self.avg_throughput = (1.0 - a) * self.avg_throughput + a * acked_bits
        self.cum_acked_bits += acked_bits

        # --- episode metrics ---
        self.ep["acked_bits"] += float(acked_bits.sum())
        self.ep["acked_per_ue"] += acked_bits
        self.ep["n_comp"] += n_comp
        self.ep["n_miss_deadline"] += n_miss_deadline
        self.ep["n_retx_drop"] += n_retx_drop
        self.ep["reward"] += reward

        # queue-length sampling (queue mode only -- zero legacy footprint)
        if cfg.queue_size > 1:
            self.ep["qlen_sum"] += float(self.traffic.queue_lens().sum())
            self.ep["qlen_count"] += 1

        n_sched = int(occ.sum())
        slot_sinr_db = (float(10.0 * np.log10(sinr_map[occ].mean()))
                        if n_sched else float("nan"))
        info = dict(slot=self.slot, reward=reward, reward_short=r_short,
                    n_comp=n_comp, n_miss_deadline=n_miss_deadline,
                    n_retx_drop=n_retx_drop,
                    acked_bits=float(acked_bits.sum()),
                    n_scheduled=n_sched, sinr_db_mean=slot_sinr_db,
                    n_active_units=len(self.txmgr.units))
        return reward, info

    def _remove_packet(self, ue: int, packet_id: int) -> None:
        """Remove a packet and purge all its transmission units."""
        pkt = self.traffic.packets[ue]
        if pkt is not None and pkt.packet_id == packet_id:
            self.traffic.remove_packet(ue)
        self.txmgr.remove_units_of_packet(packet_id)

    # ------------------------------------------------------------------
    # allocation realization
    # ------------------------------------------------------------------
    def _sanitize_and_create(self, allocation: np.ndarray) -> np.ndarray:
        """Validate free positions, create new units.

        Pending retx positions are already in ``self.fixed_allocation`` from
        ``compact_pending()``. The actor/baseline allocation only fills the
        non-fixed positions; we validate them and create transmission units.

        Closure semantics (spec §5): when ``allocation[r, l] == 0`` (no-user)
        at a non-fixed position, the RBG is closed -- all subsequent layers
        of that RBG are ignored (auto no-user). The env enforces this
        defensively so a scheduler that does not track closure cannot inject
        UEs into "closed" layers.

        Returns the realized [R, L] allocation actually used for SINR.
        """
        cfg = self.cfg
        if cfg.resolved_la_mode() == "post_rzf":
            return self._sanitize_and_create_post_rzf(allocation)
        realized = self.fixed_allocation.copy()
        sel_ue = [s.copy() for s in self.initial_S_r]
        rbg_closed = np.zeros(cfg.num_rbg, dtype=bool)

        # m-aware link adaptation (la_mode "snr_m"; legacy CLI spelling
        # mu_aware_la=True resolves to it, default OFF = historical): size
        # B_tx with the per-stream power split the gNB itself is about to
        # create, instead of the full-power SU CQI. m_planned counts, per RBG,
        # the retx-pinned layers plus the allocation's admissible new entries
        # (same pre-checks as the creation loop below, minus the b_tx-epsilon
        # self-reference -- a later epsilon drop makes the sizing conservative).
        # Addresses the cap-limited first-NACK of depth>=2 (audit C-cluster).
        # NOTE: this de-rate reduces retx exhaustion but does NOT remove the
        # first NACK (post-RZF SINR < SNR/m for non-orthogonal groups); the
        # complete treatment is la_mode='post_rzf' (2026-07-13).
        # 2026-07-13 round-8 fix: gate on resolved_la_mode(), not the raw
        # flag -- an explicit la_mode must win over mu_aware_la both ways.
        m_planned = None
        if cfg.resolved_la_mode() == "snr_m":
            m_planned = self.fixed_mask.sum(axis=1).astype(int)   # [R]
            _seen = [set(s) for s in self.initial_S_r]
            _closed = np.zeros(cfg.num_rbg, dtype=bool)
            for l in range(cfg.l_max):
                for r in range(cfg.num_rbg):
                    if self.fixed_mask[r, l] or _closed[r]:
                        continue
                    k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                    if k == 0:
                        _closed[r] = True
                        continue
                    u = k - 1
                    if not (0 <= u < cfg.num_ue) or u in _seen[r]:
                        continue
                    pkt = self.traffic.packets[u]
                    if pkt is None or pkt.uncommitted_backlog <= 0:
                        continue
                    m_planned[r] += 1
                    _seen[r].add(u)

        # free positions, layer-major order
        for l in range(cfg.l_max):
            for r in range(cfg.num_rbg):
                if self.fixed_mask[r, l]:
                    continue                       # already filled by retx
                if rbg_closed[r]:
                    continue                       # closed by earlier no-user
                k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                if k == 0:
                    rbg_closed[r] = True           # explicit no-user closes RBG
                    continue
                u = k - 1
                if not (0 <= u < cfg.num_ue):
                    continue
                if u in sel_ue[r]:
                    continue                       # dup UE in this RBG
                pkt = self.traffic.packets[u]
                if pkt is None or pkt.uncommitted_backlog <= 0:
                    continue
                if m_planned is not None and m_planned[r] > 1:
                    # de-rate the fed-back SU CQI by the planned stream count:
                    # SE_m = log2(1 + (2^CQI - 1) / m)
                    snr_su = 2.0 ** float(self.csi.cqi_fb[u, r]) - 1.0
                    se_m = np.log2(1.0 + snr_su / float(m_planned[r]))
                    b_tx_cap = cfg.eta_data * cfg.n_re_rbg * cfg.beta_rate * se_m
                else:
                    b_tx_cap = float(predict_b_tx(self.csi.cqi_fb[u, r], cfg))
                b_tx = min(pkt.uncommitted_backlog, b_tx_cap)
                if b_tx < cfg.b_tx_epsilon:
                    continue
                if pkt.uncommitted_backlog - b_tx < cfg.b_tx_epsilon:
                    # a residual below the unit-creation floor could never be
                    # committed again -> the packet could never complete and
                    # would block the UE's HOL until deadline miss; swallow
                    # the sub-epsilon tail into this unit instead
                    b_tx = pkt.uncommitted_backlog
                self.txmgr.create_unit(pkt, r, l, b_tx)
                realized[r, l] = k
                sel_ue[r].add(u)
        return realized

    def _sanitize_and_create_post_rzf(self, allocation: np.ndarray) -> np.ndarray:
        """post_rzf unit creation: RBG-major group closure via the shared
        SlotAllocationPlanner (la_planner.py).

        New units are sized from the FINAL group's predicted post-RZF SINR
        (h_hat only -- same precoder/alpha/power as the actual transmission);
        fixed retx members shape the prediction but keep their historical
        B_tx. Budgets are debited once per RBG after closure, so the env's
        commit equals the planner's plan by construction; Gate 2 compares a
        scheduler's independently threaded plan against
        ``self.last_actual_btx``.
        """
        cfg = self.cfg
        realized = self.fixed_allocation.copy()
        uncommitted0 = np.array(
            [p.uncommitted_backlog if p is not None else 0.0
             for p in self.traffic.packets], dtype=np.float64)
        planner = SlotAllocationPlanner(cfg, self.h_hat_slot, self.noise_var,
                                        self.noise_var, uncommitted0)
        self.last_actual_btx = {}
        for r in range(cfg.num_rbg):
            fixed_ues, entries, seen, closed = [], [], set(), False
            for l in range(cfg.l_max):
                if self.fixed_mask[r, l]:
                    fu = int(self.fixed_allocation[r, l]) - 1
                    if fu >= 0 and fu not in fixed_ues:
                        fixed_ues.append(fu)
                    continue
                if closed:
                    continue
                k = int(allocation[r, l]) if allocation[r, l] > 0 else 0
                if k == 0:
                    closed = True                     # spec §5 closure
                    continue
                u = k - 1
                if not (0 <= u < cfg.num_ue) or u in seen or u in fixed_ues:
                    continue
                pkt = self.traffic.packets[u]
                if pkt is None or pkt.uncommitted_backlog <= 0:
                    continue
                seen.add(u)
                entries.append((l, u))
            _, btx = planner.close_rbg(r, fixed_ues, [u for _, u in entries])
            for l, u in entries:
                if u in btx:
                    self.txmgr.create_unit(self.traffic.packets[u], r, l,
                                           btx[u])
                    realized[r, l] = u + 1
                    self.last_actual_btx[(r, u)] = btx[u]
        self.last_env_planner = planner
        return realized

    # ------------------------------------------------------------------
    # scheduler-facing helpers
    # ------------------------------------------------------------------
    def get_observation(self) -> dict:
        """State visible to the scheduler (fed-back CSI + traffic + fairness).

        The scheduler never sees the true channel.
        """
        cfg = self.cfg
        deadline = np.zeros(cfg.num_ue)
        backlog = np.zeros(cfg.num_ue)
        uncommitted = np.zeros(cfg.num_ue)
        active = np.zeros(cfg.num_ue, dtype=bool)
        for u, p in enumerate(self.traffic.packets):
            if p is not None:
                active[u] = True
                deadline[u] = p.deadline
                backlog[u] = p.remaining_backlog
                uncommitted[u] = p.uncommitted_backlog
        return dict(
            slot=self.slot,
            noise_var=self.noise_var,   # lets a scheduler rebuild h_hat_slot
            direction_fb=self.csi.direction_fb.copy(),
            cqi_fb=self.csi.cqi_fb.copy(),
            age=self.csi.age.copy(),
            deadline=deadline,
            backlog=backlog,
            uncommitted=uncommitted,
            avg_throughput=self.avg_throughput.copy(),
            active=active,
            # --- queue state (Run4; legacy: len∈{0,1}, bits=HOL backlog, next=0)
            queue_len=self.traffic.queue_lens().astype(np.float64),
            queue_bits=self.traffic.queue_bits(),
            next_deadline=self.traffic.next_deadlines(),
            # post-compaction fixed-retx structures (Phase 2 spec):
            fixed_allocation=self.fixed_allocation.copy(),
            fixed_mask=self.fixed_mask.copy(),
            fixed_unit_map=self.fixed_unit_map.copy(),
            initial_S_r=[s.copy() for s in self.initial_S_r],
            # legacy alias for back-compat with Phase 1 baselines
            occupied=self.fixed_allocation.copy(),
        )

    def orthoscore_all(self, rbg: int, selected_ues) -> np.ndarray:
        """OrthoScore of every UE vs the selected set in an RBG.

        OrthoScore(u | S_r) = 1 - max_{v in S_r} |h_dir_u^H h_dir_v|^2 with
        the fed-back unit-norm directions. Empty S_r -> OrthoScore 1.
        """
        cfg = self.cfg
        if len(selected_ues) == 0:
            return np.ones(cfg.num_ue)
        c_all = self.csi.direction_fb[:, rbg, :]                    # [K, M_ant]
        c_sel = self.csi.direction_fb[list(selected_ues), rbg, :]   # [S, M_ant]
        corr = np.abs(c_all @ c_sel.conj().T) ** 2                  # [K, S]
        return 1.0 - corr.max(axis=1)

    def position_candidates(self, rbg: int, selected_ues,
                            uncommitted=None) -> np.ndarray:
        """Boolean [K] mask of UEs validly placeable at an RBG.

        Valid = active packet, uncommitted backlog > 0, not already in this
        RBG, and predicted B_tx >= epsilon.

        ``uncommitted`` optionally overrides the per-UE uncommitted backlog --
        a scheduler threads its own budget here so the mask stays consistent
        with what it has already assigned this slot.
        """
        cfg = self.cfg
        valid = np.zeros(cfg.num_ue, dtype=bool)
        sel = set(int(x) for x in selected_ues)
        btx_pred = predict_b_tx(self.csi.cqi_fb[:, rbg], cfg)         # [K]
        for u, p in enumerate(self.traffic.packets):
            if p is None or u in sel:
                continue
            unc = (p.uncommitted_backlog if uncommitted is None
                   else float(uncommitted[u]))
            if unc <= 0:
                continue
            if min(unc, float(btx_pred[u])) >= cfg.b_tx_epsilon:
                valid[u] = True
        return valid

    def estimate_btx(self, ue: int, rbg: int, uncommitted: float) -> float:
        """B_tx a unit at (ue, rbg) would commit, given a remaining budget.

        Same formula env.step uses; a scheduler calls this to thread its
        per-UE commit budget during sequential selection.
        """
        pred = float(predict_b_tx(self.csi.cqi_fb[ue, rbg], self.cfg))
        return min(float(uncommitted), pred)


if __name__ == "__main__":
    from config import debug_config

    cfg = debug_config()
    env = SchedulerEnv(cfg)
    obs = env.reset(episode_idx=0)
    print(f"env.py smoke test: reset ok, "
          f"obs keys = {sorted(obs.keys())}")

    rng = np.random.default_rng(0)
    total_r, steps = 0.0, 0
    done = False
    while not done:
        # trivial random scheduler (layer-major, respects constraints)
        alloc = np.zeros((cfg.num_rbg, cfg.l_max), dtype=np.int64)
        for l in range(cfg.l_max):
            for r in range(cfg.num_rbg):
                sel = set(np.where(alloc[r] > 0)[0])
                sel_ue = set((alloc[r][alloc[r] > 0] - 1).tolist())
                cand = env.position_candidates(r, sel_ue)
                ues = np.where(cand)[0]
                if ues.size and rng.random() < 0.6:
                    alloc[r, l] = int(rng.choice(ues)) + 1
        obs, reward, done, info = env.step(alloc)
        total_r += reward
        steps += 1
    print(f"  ran {steps} slots, total reward {total_r:.2f}")
    print(f"  episode: arrivals={env.ep['n_arrivals']}, "
          f"comp={env.ep['n_comp']}, miss={env.ep['n_miss_deadline']}, "
          f"retx_drop={env.ep['n_retx_drop']}, "
          f"acked={env.ep['acked_bits']/1e6:.2f} Mbit")
    if env.ep["sinr_count"]:
        mean_sinr = env.ep["sinr_sum"] / env.ep["sinr_count"]
        print(f"  mean SINR over scheduled positions: "
              f"{10*np.log10(mean_sinr):.2f} dB")
    assert np.isfinite(total_r)
    print("env.py smoke test passed.")
