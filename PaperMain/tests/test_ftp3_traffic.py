"""Arrival-process statistics, packet accounting and frozen Bernoulli replay."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from traffic import TrafficModel


class ScriptedArrivalRng:
    """Force offered counts and packet attributes independently of the model."""

    def __init__(self, counts, attributes=()):
        self.counts = iter(counts)
        self.attributes = iter(attributes)
        self.means = []
        self.attribute_bounds = []

    def poisson(self, mean):
        self.means.append(mean)
        return next(self.counts)

    def integers(self, low, high):
        self.attribute_bounds.append((low, high))
        value = next(self.attributes)
        if not low <= value < high:
            raise AssertionError(f"test attribute {value} outside [{low}, {high})")
        return value


class FTP3TrafficTests(unittest.TestCase):
    def make_traffic(self, **changes):
        cfg = Config(**dict(dict(traffic_model="ftp3", num_ue=1, queue_size=8,
                                  p_arrival=.22, packet_size_min=4000,
                                  packet_size_max=12000, deadline_min=3,
                                  deadline_max=12), **changes))
        return TrafficModel(cfg)

    def test_multiple_arrivals_keep_packet_parameters_and_hol_order(self):
        tm = self.make_traffic()
        rng = ScriptedArrivalRng([3], [4000, 3, 12000, 12, 8123, 7])
        admitted, overflow = tm.generate_arrivals(19, rng)
        self.assertEqual((admitted, overflow), ([0, 0, 0], 0))
        self.assertEqual(rng.means, [.22])
        self.assertEqual(rng.attribute_bounds, [(4000, 12001), (3, 13)] * 3)
        packets = list(tm.queues[0])
        self.assertEqual([(p.packet_id, p.ue_id, p.arrival_slot, p.size, p.deadline)
                          for p in packets],
                         [(0, 0, 19, 4000, 3), (1, 0, 19, 12000, 12),
                          (2, 0, 19, 8123, 7)])
        self.assertIs(tm.packets[0], packets[0])
        tm.remove_packet(0)
        self.assertIs(tm.packets[0], packets[1])
        np.testing.assert_array_equal(tm.queue_bits(), [20123])

    def test_counts_every_overflow_and_keeps_existing_hol(self):
        tm = self.make_traffic(queue_size=2)
        first = ScriptedArrivalRng([1], [5000, 8])
        tm.generate_arrivals(0, first)
        hol = tm.packets[0]
        second = ScriptedArrivalRng([5], [6000, 9])
        self.assertEqual(tm.generate_arrivals(1, second), ([0], 4))
        self.assertIs(tm.packets[0], hol)
        self.assertEqual(tm.generate_arrivals(2, ScriptedArrivalRng([7])), ([], 7))
        self.assertEqual(len(tm.queues[0]), 2)
        self.assertEqual(tm._next_id, 2)

    def test_single_packet_buffer_does_not_self_throttle_ftp3(self):
        tm = self.make_traffic(queue_size=1)
        self.assertEqual(tm.generate_arrivals(0, ScriptedArrivalRng([4], [5000, 9])),
                         ([0], 3))
        self.assertEqual(tm.generate_arrivals(1, ScriptedArrivalRng([6])), ([], 6))
        self.assertEqual(tm.packets[0].arrival_slot, 0)

    def test_only_active_ues_receive_traffic_and_episode_rate_is_mean(self):
        tm = self.make_traffic(num_ue=4)
        tm.n_active = 2
        tm.p_arrival_ep = .5
        rng = ScriptedArrivalRng([2, 1], [4000, 3, 4001, 4, 4002, 5])
        self.assertEqual(tm.generate_arrivals(0, rng), ([0, 0, 1], 0))
        self.assertEqual(rng.means, [.5, .5])  # no -log(1-p) conversion
        np.testing.assert_array_equal(tm.queue_lens(), [2, 1, 0, 0])
        tm.n_active = 0
        self.assertEqual(tm.generate_arrivals(1, ScriptedArrivalRng([])), ([], 0))

    def test_zero_rate_produces_no_packets(self):
        tm = self.make_traffic(p_arrival=0)
        rng = np.random.default_rng(428)
        for slot in range(20):
            self.assertEqual(tm.generate_arrivals(slot, rng), ([], 0))
        self.assertIsNone(tm.packets[0])

    def test_poisson_arrival_statistics_and_multiple_arrival_probability(self):
        # Leave the buffer full so no packet-attribute draws or services enter
        # this independent measurement of offered (admitted + rejected) counts.
        mean = .35
        tm = self.make_traffic(queue_size=1, p_arrival=mean)
        tm.generate_arrivals(0, ScriptedArrivalRng([1], [4000, 3]))
        rng = np.random.default_rng(723987)
        counts = np.array([tm.generate_arrivals(slot, rng)[1]
                           for slot in range(1, 60001)])
        self.assertLess(abs(counts.mean() - mean), .012)
        self.assertLess(abs(counts.var() - mean), .020)
        self.assertLess(abs(np.mean(counts == 0) - math.exp(-mean)), .010)
        expected_multiple = 1 - math.exp(-mean) * (1 + mean)
        self.assertGreater(np.count_nonzero(counts >= 2), 0)
        self.assertLess(abs(np.mean(counts >= 2) - expected_multiple), .006)

    def test_poisson_mean_can_exceed_one(self):
        tm = self.make_traffic(p_arrival=3.2, p_arrival_min=2., p_arrival_max=4.)
        self.assertEqual(tm.generate_arrivals(0, ScriptedArrivalRng([0])), ([], 0))

    def test_simultaneous_arrivals_retain_deadline_expiry_and_fifo_promotion(self):
        tm = self.make_traffic(deadline_min=1)
        tm.generate_arrivals(0, ScriptedArrivalRng([3], [4000, 4, 5000, 1, 6000, 3]))
        tm.decrement_deadlines()
        self.assertEqual(tm.pop_expired_queued(), [0])
        self.assertEqual([(p.packet_id, p.deadline) for p in tm.queues[0]],
                         [(0, 3), (2, 2)])
        tm.remove_packet(0)
        self.assertEqual(tm.packets[0].packet_id, 2)
        tm.remove_packet(0)
        self.assertIsNone(tm.packets[0])

    def test_reset_removes_all_packets_and_restarts_ids(self):
        tm = self.make_traffic()
        tm.generate_arrivals(5, ScriptedArrivalRng([2], [4000, 3, 5000, 4]))
        tm.reset()
        np.testing.assert_array_equal(tm.queue_lens(), [0])
        self.assertIsNone(tm.packets[0])
        tm.generate_arrivals(0, ScriptedArrivalRng([1], [6000, 5]))
        self.assertEqual(tm.packets[0].packet_id, 0)

    def test_invalid_model_rates_and_capacity_fail_at_construction(self):
        changes = [{"traffic_model": value} for value in ("poisson", "FTP3", "", None)]
        changes += [{name: value}
                    for name in ("p_arrival", "p_arrival_min", "p_arrival_max")
                    for value in (-.01, float("nan"), float("inf"), True, ".2")]
        changes += [{"p_arrival_min": .5, "p_arrival_max": .2}]
        changes += [{"queue_size": value} for value in (0, -1, 1.5, True)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make_traffic(**change)
        with self.assertRaises(ValueError):
            self.make_traffic(traffic_model="bernoulli", p_arrival=1.1)

    def test_config_describes_arrival_process_and_units(self):
        self.assertEqual(Config().traffic_model, "bernoulli")
        self.assertIn("bernoulli, p_arrival=", Config().describe())
        self.assertIn("ftp3, mean_arrivals_per_ue_slot=0.22",
                      self.make_traffic().cfg.describe())

    def test_default_bernoulli_packet_history_and_rng_match_frozen_prechange_run(self):
        # SHA256 fixtures were captured from the unchanged traffic.py BEFORE
        # introducing FTP3. Cover idle-only Run3 and every-slot Run4, payloads,
        # deadlines, overflow, promotion, mixed arrival rate and final RNG state.
        expected = {
            1: "befea0c8b36d0b68c01d9e0ed455df64a35563cc007637ffd312204fa1e79259",
            8: "1128935df7f2db53e847e7eb8a0d9448c54cc9e6c143d46626be9ac1960d428f",
        }
        for qsize, digest in expected.items():
            with self.subTest(queue_size=qsize):
                cfg = Config(num_ue=5, queue_size=qsize, p_arrival=.37,
                             packet_size_min=4000, packet_size_max=12000,
                             deadline_min=3, deadline_max=12)
                tm = TrafficModel(cfg)
                tm.n_active = 4
                rng = np.random.default_rng(349289)
                history = []
                for slot in range(150):
                    if slot == 50:
                        tm.p_arrival_ep = .15
                    if slot == 100:
                        tm.p_arrival_ep = .5
                    new, overflow = tm.generate_arrivals(slot, rng)
                    tm.decrement_deadlines()
                    for u in range(cfg.num_ue):
                        packet = tm.packets[u]
                        if packet is not None and (packet.deadline <= 0 or (slot + u) % 7 == 0):
                            tm.remove_packet(u)
                    expired = tm.pop_expired_queued()
                    history.append([new, overflow, expired,
                                    [[asdict(packet) for packet in q] for q in tm.queues]])
                payload = {"history": history, "rng": rng.bit_generator.state}
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
