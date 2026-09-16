"""traffic.py -- random traffic model: per-UE packet queue (Run4) with the
Run3 one-packet model as the exact queue_size=1 special case.

Run3 (queue_size=1): one head-of-line (HOL) packet per UE; an IDLE UE draws a
new packet with Bernoulli(p_arrival). While the HOL is busy no arrivals occur
(self-throttling). This path is kept RNG-exact for reproducibility.

Run4 (queue_size>1): per-UE FIFO queue. EVERY active UE draws a
Bernoulli(p_arrival) arrival EVERY slot; if the queue is full the arrival is
a buffer-overflow drop. Deadlines tick for ALL queued packets; a packet that
expires while waiting in the queue is a deadline miss. Only the HOL packet is
schedulable -- ``packets[u]`` remains the HOL view so env/policy/baselines
keep their existing interface; on HOL removal the next packet promotes.

Opt-in FTP3: each active UE draws a Poisson packet count in each slot. The
existing arrival-intensity values denote mean packets per UE per slot, so
switching models preserves offered mean load. Multiple packets can arrive;
arrivals continue even with a full queue and every rejected packet counts as
overflow. Packet sizes, deadlines and FIFO service retain the same settings.
This implements FTP Model 3's arrival process with the experiment's packet
parameters, rather than claiming a full standardized FTP3 traffic profile.

A packet may be split across several transmission units (see transmission.py).
The packet tracks:
  * size            -- total payload bits
  * committed_bits  -- sum of B_tx of its transmission units (committed)
  * acked_bits      -- sum of B_tx of its ACKed units (delivered)
  * deadline        -- remaining slots until expiry
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import Config


@dataclass
class Packet:
    """One packet (HOL or queued)."""
    packet_id: int
    ue_id: int
    arrival_slot: int
    size: int                      # total payload bits
    deadline: int                  # remaining slots until expiry
    committed_bits: float = 0.0    # sum of B_tx over its transmission units
    acked_bits: float = 0.0        # sum of B_tx over its ACKed units

    @property
    def remaining_backlog(self) -> float:
        """Bits not yet delivered (ACKed)."""
        return self.size - self.acked_bits

    @property
    def uncommitted_backlog(self) -> float:
        """Bits not yet assigned to any transmission unit."""
        return self.size - self.committed_bits

    @property
    def is_complete(self) -> bool:
        return self.acked_bits >= self.size - 1e-6

    def commit(self, bits: float) -> None:
        """Reserve `bits` for a newly created transmission unit."""
        self.committed_bits += bits

    def ack(self, bits: float) -> None:
        """Mark `bits` as delivered (a transmission unit ACKed)."""
        self.acked_bits += bits


class TrafficModel:
    """Per-UE packet queue with selectable Bernoulli or FTP3 Poisson arrivals.

    ``packets[u]`` is ALWAYS the HOL view (queues[u][0] or None) -- the rest
    of the simulator reads/mutates only the HOL, exactly as in Run3.
    """

    def __init__(self, cfg: Config):
        cfg.validate_traffic()
        self.cfg = cfg
        self.queues: list[deque] = [deque() for _ in range(cfg.num_ue)]
        self.packets: list[Packet | None] = [None] * cfg.num_ue
        self._next_id = 0
        self.n_active = cfg.num_ue     # UEs [0, n_active) get traffic; env sets per-episode
        # Level-2 mixed-arrival: per-episode p_arrival override (env sets it
        # each reset when cfg.p_arrival_min/max enabled; None = cfg.p_arrival)
        self.p_arrival_ep: float | None = None

    def _sync_hol(self, u: int) -> None:
        self.packets[u] = self.queues[u][0] if self.queues[u] else None

    def reset(self) -> None:
        """Episode start: all UEs idle, no packets (spec refinement #10)."""
        self.queues = [deque() for _ in range(self.cfg.num_ue)]
        self.packets = [None] * self.cfg.num_ue
        self._next_id = 0

    def _new_packet(self, u: int, slot: int, rng: np.random.Generator) -> Packet:
        cfg = self.cfg
        size = int(rng.integers(cfg.packet_size_min, cfg.packet_size_max + 1))
        deadline = int(rng.integers(cfg.deadline_min, cfg.deadline_max + 1))
        pkt = Packet(packet_id=self._next_id, ue_id=u, arrival_slot=slot,
                     size=size, deadline=deadline)
        self._next_id += 1
        return pkt

    def generate_arrivals(self, slot: int, rng: np.random.Generator):
        """Return (admitted_ue_list, n_buffer_overflow) for this slot.

        FTP3 can admit multiple packets for one UE: the returned list contains
        its UE index once per admitted packet, preserving arrival accounting.

        Bernoulli queue_size == 1: the Run3 path, RNG-EXACT -- the draw
        happens ONLY when the UE is idle (short-circuit preserved), so the
        downstream RNG stream is bit-identical to the legacy model.
        queue_size > 1: every active UE draws EVERY slot; a draw landing on
        a full queue is counted as a buffer-overflow drop.
        """
        cfg = self.cfg
        p = self.p_arrival_ep if self.p_arrival_ep is not None else cfg.p_arrival
        new_ues, n_overflow = [], 0
        if cfg.traffic_model == "ftp3":
            for u in range(self.n_active):
                n_arrivals = int(rng.poisson(p))
                n_admitted = min(n_arrivals, cfg.queue_size - len(self.queues[u]))
                for _ in range(n_admitted):
                    self.queues[u].append(self._new_packet(u, slot, rng))
                    new_ues.append(u)
                if n_admitted:
                    self._sync_hol(u)
                n_overflow += n_arrivals - n_admitted
            return new_ues, n_overflow
        if cfg.queue_size <= 1:
            for u in range(self.n_active):
                if self.packets[u] is None and rng.random() < p:
                    self.queues[u].append(self._new_packet(u, slot, rng))
                    self._sync_hol(u)
                    new_ues.append(u)
        else:
            for u in range(self.n_active):
                if rng.random() < p:
                    if len(self.queues[u]) < cfg.queue_size:
                        self.queues[u].append(self._new_packet(u, slot, rng))
                        self._sync_hol(u)
                        new_ues.append(u)
                    else:
                        n_overflow += 1
        return new_ues, n_overflow

    def decrement_deadlines(self) -> None:
        """Advance time: EVERY queued packet loses one slot of deadline."""
        for q in self.queues:
            for p in q:
                p.deadline -= 1

    def pop_expired_queued(self) -> list[int]:
        """Remove expired NON-HOL packets (they hold no transmission units).

        Returns the UE index once per removed packet (for miss accounting).
        HOL expiry is handled by the env (it must also purge tx units).
        """
        missed_ues = []
        for u, q in enumerate(self.queues):
            if len(q) <= 1:
                continue
            hol = q[0]
            survivors = [p for p in list(q)[1:] if p.deadline > 0]
            n_removed = (len(q) - 1) - len(survivors)
            if n_removed:
                missed_ues.extend([u] * n_removed)
                self.queues[u] = deque([hol] + survivors)
                self._sync_hol(u)
        return missed_ues

    def remove_packet(self, u: int) -> None:
        """Remove the HOL packet (completed / missed / dropped); the next
        queued packet (if any) promotes to HOL."""
        if self.queues[u]:
            self.queues[u].popleft()
        self._sync_hol(u)

    def active_ues(self) -> np.ndarray:
        """Boolean mask [K] of UEs that currently have a (HOL) packet."""
        return np.array([p is not None for p in self.packets], dtype=bool)

    # ---- queue-state accessors (Run4 observation fields) ----
    def queue_lens(self) -> np.ndarray:
        """[K] number of packets currently queued (incl. HOL)."""
        return np.array([len(q) for q in self.queues], dtype=np.int64)

    def queue_bits(self) -> np.ndarray:
        """[K] total remaining (un-ACKed) bits over the whole queue."""
        return np.array([sum(p.remaining_backlog for p in q)
                         for q in self.queues], dtype=np.float64)

    def next_deadlines(self) -> np.ndarray:
        """[K] deadline of the packet BEHIND the HOL (0 if none)."""
        return np.array([q[1].deadline if len(q) > 1 else 0
                         for q in self.queues], dtype=np.float64)


if __name__ == "__main__":
    # --- legacy path (queue_size=1): RNG-exact vs the Run3 model ---
    cfg = Config()
    tm = TrafficModel(cfg)
    tm.reset()
    rng = np.random.default_rng(0)
    total_arrivals = 0
    for slot in range(50):
        new, ovf = tm.generate_arrivals(slot, rng)
        assert ovf == 0
        total_arrivals += len(new)
        tm.decrement_deadlines()
    print(f"traffic.py [legacy q=1]: {total_arrivals} arrivals / 50 slots, "
          f"{int(tm.active_ues().sum())} active")

    # commit / ack bookkeeping unchanged
    u = int(np.argmax(tm.active_ues()))
    p = tm.packets[u]
    p.commit(3000.0); assert p.uncommitted_backlog == p.size - 3000.0
    p.ack(3000.0);    assert p.remaining_backlog == p.size - 3000.0
    p.commit(p.uncommitted_backlog); p.ack(p.size - p.acked_bits)
    assert p.is_complete

    # --- queue mode (queue_size=8) ---
    cfg8 = Config(queue_size=8, p_arrival=0.9, deadline_min=3, deadline_max=6)
    tm = TrafficModel(cfg8)
    tm.reset()
    rng = np.random.default_rng(1)
    ovf_total = 0
    for slot in range(40):
        new, ovf = tm.generate_arrivals(slot, rng)
        ovf_total += ovf
        tm.decrement_deadlines()
        # HOL expiry (env's job normally): emulate
        for u in range(cfg8.num_ue):
            while tm.packets[u] is not None and tm.packets[u].deadline <= 0:
                tm.remove_packet(u)
        missed_q = tm.pop_expired_queued()
    ql = tm.queue_lens(); qb = tm.queue_bits(); nd = tm.next_deadlines()
    assert ql.max() <= cfg8.queue_size
    assert (qb >= 0).all()
    # HOL view consistency
    for u in range(cfg8.num_ue):
        assert (tm.packets[u] is None) == (len(tm.queues[u]) == 0)
        if tm.packets[u] is not None:
            assert tm.packets[u] is tm.queues[u][0]
            assert tm.packets[u].deadline > 0
        for pkt in tm.queues[u]:
            assert pkt.deadline > 0            # no expired survivors
    print(f"traffic.py [queue q=8]: qlen max {ql.max()}, overflow {ovf_total}, "
          f"queue_bits mean {qb.mean():.0f}, HOL-view consistent")
    print("traffic.py smoke test passed.")
