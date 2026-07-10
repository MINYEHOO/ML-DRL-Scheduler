"""
transmission.py -- transmission units, MI accumulation, retransmission.

A transmission unit is uniquely identified by its global ``unit_id``. Each
unit belongs to one (packet, UE, RBG); its ``current_layer_id`` is the layer
position WITHIN its RBG for the current slot and is re-assigned at the start
of every slot by ``TransmissionManager.compact_pending`` (Phase 2 spec --
pending retx is RBG-fixed, NOT layer-fixed).

Target payload ``b_tx`` is fixed at creation. Each slot the unit transmits,
it accumulates useful mutual information from the true-channel SINR:

    useful = min(N_RE_RBG * log2(1 + SINR_true),  b_tx - i_acc)
    i_acc += useful

When ``i_acc >= b_tx`` the unit is ACKed. Otherwise it is NACKed and stays in
its RBG for retransmission. A unit gets ``1 + max_retx`` attempts; if it
still fails, the whole parent packet is dropped (all sibling units +
pending retransmissions removed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from phy import mi_bits


@dataclass
class TransmissionUnit:
    """One transmission unit, keyed by global ``unit_id``."""
    unit_id: int                # global, unique across the simulation
    packet_id: int
    ue_id: int
    rbg_id: int
    previous_layer_id: int      # layer assignment from the previous slot
    current_layer_id: int       # layer assignment for the current slot
    b_tx: float                 # target payload [bits], fixed at creation
    i_acc: float = 0.0          # accumulated useful mutual information [bits]
    tx_attempts: int = 0        # transmission attempts so far
    last_useful: float = 0.0    # useful MI increment in the most recent slot

    @property
    def is_acked(self) -> bool:
        return self.i_acc >= self.b_tx - 1e-6


@dataclass
class SlotOutcome:
    """Result of processing one slot's transmissions."""
    useful_per_ue: np.ndarray   # [K] useful MI delivered per UE this slot
    acked: list                 # list[TransmissionUnit] ACKed this slot
    dropped: list               # list[TransmissionUnit] retx-exhausted


class TransmissionManager:
    """Owns the set of active transmission units and their lifecycle."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.units: list[TransmissionUnit] = []
        self.units_by_id: dict[int, TransmissionUnit] = {}
        self._next_unit_id: int = 0

    def reset(self) -> None:
        self.units = []
        self.units_by_id = {}
        self._next_unit_id = 0

    # ------------------------------------------------------------------
    # creation
    # ------------------------------------------------------------------
    def create_unit(self, packet, rbg_id: int, layer_id: int,
                    b_tx: float) -> TransmissionUnit:
        """Create a unit at a free position and commit b_tx on the packet."""
        unit = TransmissionUnit(
            unit_id=self._next_unit_id,
            packet_id=packet.packet_id,
            ue_id=packet.ue_id,
            rbg_id=int(rbg_id),
            previous_layer_id=int(layer_id),    # at creation, no previous slot
            current_layer_id=int(layer_id),
            b_tx=float(b_tx),
        )
        self._next_unit_id += 1
        packet.commit(float(b_tx))
        self.units.append(unit)
        self.units_by_id[unit.unit_id] = unit
        return unit

    # ------------------------------------------------------------------
    # slot-start compaction (Phase 2 spec)
    # ------------------------------------------------------------------
    def compact_pending(self, traffic) -> dict:
        """Compact pending units to front layers within each RBG.

        Side effects:
          * For every surviving unit: ``previous_layer_id ← current_layer_id``
            snapshot, then ``current_layer_id`` is reassigned to its compacted
            slot.

        On overflow (len(retx_units in RBG) > L_max):
          * debug mode (cfg.debug == True): raises AssertionError.
          * safe mode: sorts by (deadline ascending, unit_id ascending), keeps
            top L_max, marks the rest for parent-packet drop. The caller
            (env) is responsible for actually removing the parent packets
            (and their sibling units in other RBGs).

        Returns
        -------
        dict with:
            fixed_allocation : [R, L] int64 -- 1-based UE id (0 if free)
            fixed_mask       : [R, L] bool  -- True at retx-fixed positions
            fixed_unit_map   : [R, L] int64 -- unit_id at fixed positions (-1 free)
            initial_S_r      : list[set[int]] -- per-RBG set of UE ids (0-based)
                                from the compacted retx units
            overflow_drops   : list[(packet_id, ue_id)] for caller to drop
        """
        cfg = self.cfg
        R, L = cfg.num_rbg, cfg.l_max

        # 1) snapshot previous_layer_id (before compaction reassigns)
        for u in self.units:
            u.previous_layer_id = u.current_layer_id

        # 2) group units by RBG
        by_rbg: list[list[TransmissionUnit]] = [[] for _ in range(R)]
        for u in self.units:
            by_rbg[u.rbg_id].append(u)

        # 3) overflow detection (per RBG)
        overflow_packet_to_ue: dict[int, int] = {}
        for r in range(R):
            units_r = by_rbg[r]
            if len(units_r) <= L:
                continue
            if cfg.debug:
                raise AssertionError(
                    f"Retx overflow in RBG {r}: "
                    f"{len(units_r)} > L_max={L}")
            # safe mode: sort by (deadline ↑, unit_id ↑), keep top L, drop rest
            def deadline_key(u: TransmissionUnit, _traffic=traffic):
                pkt = _traffic.packets[u.ue_id]
                d = (pkt.deadline
                     if pkt is not None and pkt.packet_id == u.packet_id
                     else 10**9)
                return (d, u.unit_id)
            units_r.sort(key=deadline_key)
            for u in units_r[L:]:
                overflow_packet_to_ue[u.packet_id] = u.ue_id
            by_rbg[r] = units_r[:L]

        # 4) any unit whose parent is in overflow set must also be filtered
        #    (sibling units in OTHER RBGs of an overflow-dropped packet)
        overflow_pids = set(overflow_packet_to_ue.keys())
        if overflow_pids:
            for r in range(R):
                by_rbg[r] = [u for u in by_rbg[r]
                             if u.packet_id not in overflow_pids]

        # 5) normal compaction: sort by (previous_layer_id ↑, unit_id ↑) and
        #    assign to layers 0..n-1
        fixed_alloc = np.zeros((R, L), dtype=np.int64)
        fixed_mask = np.zeros((R, L), dtype=bool)
        fixed_umap = np.full((R, L), -1, dtype=np.int64)
        sel_per_rbg: list[set] = [set() for _ in range(R)]

        for r in range(R):
            units_r = by_rbg[r]
            if not units_r:
                continue
            units_r.sort(key=lambda u: (u.previous_layer_id, u.unit_id))
            for new_layer, u in enumerate(units_r):
                u.current_layer_id = new_layer
                fixed_alloc[r, new_layer] = u.ue_id + 1
                fixed_mask[r, new_layer] = True
                fixed_umap[r, new_layer] = u.unit_id
                sel_per_rbg[r].add(u.ue_id)

        return dict(
            fixed_allocation=fixed_alloc,
            fixed_mask=fixed_mask,
            fixed_unit_map=fixed_umap,
            initial_S_r=sel_per_rbg,
            overflow_drops=list(overflow_packet_to_ue.items()),
        )

    # ------------------------------------------------------------------
    # slot processing
    # ------------------------------------------------------------------
    def process_slot(self, sinr_map: np.ndarray) -> SlotOutcome:
        """Accumulate MI for every active unit and classify ACK/NACK/drop.

        NACKed units stay in ``self.units`` (pending for next slot's compaction);
        ACKed and dropped units are removed and returned.
        """
        cfg = self.cfg
        useful_per_ue = np.zeros(cfg.num_ue, dtype=np.float64)
        acked, dropped, survivors = [], [], []
        for unit in self.units:
            sinr = sinr_map[unit.rbg_id, unit.current_layer_id]
            delta_i = float(mi_bits(sinr, cfg))
            useful = min(delta_i, unit.b_tx - unit.i_acc)
            useful = max(useful, 0.0)
            unit.i_acc += useful
            unit.last_useful = useful
            unit.tx_attempts += 1
            useful_per_ue[unit.ue_id] += useful
            if unit.is_acked:
                acked.append(unit)
                self.units_by_id.pop(unit.unit_id, None)
            elif unit.tx_attempts >= cfg.num_attempts:
                dropped.append(unit)
                self.units_by_id.pop(unit.unit_id, None)
            else:
                survivors.append(unit)        # NACK -> stays pending
        self.units = survivors
        return SlotOutcome(useful_per_ue, acked, dropped)

    def remove_units_of_packet(self, packet_id: int) -> None:
        """Purge all units of a packet (deadline-miss / retx-drop cleanup)."""
        keep = []
        for u in self.units:
            if u.packet_id == packet_id:
                self.units_by_id.pop(u.unit_id, None)
            else:
                keep.append(u)
        self.units = keep


if __name__ == "__main__":
    from traffic import Packet, TrafficModel

    cfg = Config()
    mgr = TransmissionManager(cfg)
    mgr.reset()

    # --- ACK in one slot: small b_tx, high SINR ---
    pkt = Packet(packet_id=0, ue_id=2, arrival_slot=0, size=10000, deadline=20)
    u0 = mgr.create_unit(pkt, rbg_id=0, layer_id=0, b_tx=3000.0)
    assert u0.unit_id == 0
    assert pkt.committed_bits == 3000.0
    assert mgr.units_by_id[u0.unit_id] is u0
    sinr = np.zeros((cfg.num_rbg, cfg.l_max))
    sinr[0, 0] = 100.0
    out = mgr.process_slot(sinr)
    assert len(out.acked) == 1 and len(mgr.units) == 0
    assert abs(out.useful_per_ue[2] - 3000.0) < 1e-6
    assert u0.unit_id not in mgr.units_by_id
    print(f"transmission.py: ACK ok -- useful={out.useful_per_ue[2]:.0f} bits "
          f"in {out.acked[0].tx_attempts} attempt, unit_id={u0.unit_id}")

    # --- retx/drop ---
    mgr.reset()
    pkt2 = Packet(packet_id=1, ue_id=5, arrival_slot=0, size=99999, deadline=50)
    mgr.create_unit(pkt2, rbg_id=1, layer_id=0, b_tx=99999.0)
    sinr_low = np.zeros((cfg.num_rbg, cfg.l_max))
    sinr_low[1, 0] = 0.05
    nacks = 0
    for slot in range(cfg.num_attempts):
        out = mgr.process_slot(sinr_low)
        if mgr.units:
            nacks += 1
    assert len(out.dropped) == 1
    assert out.dropped[0].tx_attempts == cfg.num_attempts
    print(f"transmission.py: retx/drop ok -- {nacks} NACKs then drop "
          f"at attempt {cfg.num_attempts}")

    # --- compaction: 2 retx in same RBG -> compacted to layers 0, 1 ---
    mgr.reset()
    traffic = TrafficModel(cfg)
    traffic.reset()
    pkt_a = Packet(packet_id=10, ue_id=3, arrival_slot=0, size=99999,
                   deadline=50)
    pkt_b = Packet(packet_id=11, ue_id=7, arrival_slot=0, size=99999,
                   deadline=50)
    traffic.packets[3] = pkt_a
    traffic.packets[7] = pkt_b
    ua = mgr.create_unit(pkt_a, rbg_id=2, layer_id=1, b_tx=99999.0)
    ub = mgr.create_unit(pkt_b, rbg_id=2, layer_id=3, b_tx=99999.0)
    # process_slot keeps them at the same layers initially (no compaction yet)
    # next slot's compaction should compact them to layer 0, 1
    sinr_low = np.zeros((cfg.num_rbg, cfg.l_max))
    sinr_low[2, 1] = 0.01
    sinr_low[2, 3] = 0.01
    mgr.process_slot(sinr_low)              # NACK both
    assert len(mgr.units) == 2

    result = mgr.compact_pending(traffic)
    assert result["fixed_mask"][2].tolist() == [True, True, False, False], \
        result["fixed_mask"][2]
    assert ua.current_layer_id == 0 and ub.current_layer_id == 1, \
        (ua.current_layer_id, ub.current_layer_id)
    assert result["fixed_unit_map"][2, 0] in (ua.unit_id, ub.unit_id)
    assert {3, 7} == result["initial_S_r"][2]
    assert result["overflow_drops"] == []
    print(f"transmission.py: compaction ok -- "
          f"unit {ua.unit_id} layer {ua.previous_layer_id}->{ua.current_layer_id}, "
          f"unit {ub.unit_id} layer {ub.previous_layer_id}->{ub.current_layer_id}")

    # --- compaction overflow safe mode: 5 retx in RBG with L_max=4 ---
    cfg_safe = Config()
    cfg_safe.debug = False
    mgr = TransmissionManager(cfg_safe)
    mgr.reset()
    traffic = TrafficModel(cfg_safe)
    traffic.reset()
    pkts = []
    for k in range(5):
        p = Packet(packet_id=20 + k, ue_id=k, arrival_slot=0,
                   size=99999, deadline=10 + k)   # deadlines 10, 11, ..., 14
        traffic.packets[k] = p
        pkts.append(p)
        mgr.create_unit(p, rbg_id=0, layer_id=k % cfg_safe.l_max,
                        b_tx=99999.0)
    # NACK all
    sinr_low = np.zeros((cfg_safe.num_rbg, cfg_safe.l_max))
    mgr.process_slot(sinr_low)
    assert len(mgr.units) == 5
    result = mgr.compact_pending(traffic)
    # earliest deadlines (10, 11, 12, 13) -> packets 20, 21, 22, 23 keep
    # packet 24 (deadline 14) is overflow -> drop
    assert result["overflow_drops"] == [(24, 4)], result["overflow_drops"]
    assert int(result["fixed_mask"][0].sum()) == 4
    print(f"transmission.py: overflow safe ok -- dropped "
          f"{result['overflow_drops']}")

    # --- compaction overflow debug mode: must raise ---
    cfg_dbg = Config()
    cfg_dbg.debug = True
    mgr = TransmissionManager(cfg_dbg)
    mgr.reset()
    traffic = TrafficModel(cfg_dbg)
    traffic.reset()
    for k in range(5):
        p = Packet(packet_id=30 + k, ue_id=k, arrival_slot=0,
                   size=99999, deadline=20)
        traffic.packets[k] = p
        mgr.create_unit(p, rbg_id=0, layer_id=k % cfg_dbg.l_max, b_tx=99999.0)
    mgr.process_slot(np.zeros((cfg_dbg.num_rbg, cfg_dbg.l_max)))
    try:
        mgr.compact_pending(traffic)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "debug mode should assert on overflow"
    print(f"transmission.py: overflow debug mode AssertionError ok")

    print("transmission.py smoke test passed.")
