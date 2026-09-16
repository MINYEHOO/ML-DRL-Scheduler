"""CQI4 calibration regression cases with analytic, NumPy-only channels.

These fixtures deliberately vary the true channel independently of feedback:
prediction determines eligibility; realization determines success or failure.
No channel simulator, saved artifact, GPU or training run is required.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from la_planner import SlotAllocationPlanner
from transmission import TransmissionUnit
from calibration import cqi4


def analytic_config(**overrides):
    # One RE and one watt make the independent capacity oracle log2(1+SINR).
    defaults = dict(num_ue=5, num_rbg=1, l_max=4, num_bs_ant=4,
                    num_rb_per_rbg=1, num_sc_per_rb=1, num_sym_per_rb=1,
                    p_total=1.0, eta_data=1.0, beta_rate=1.0,
                    b_tx_epsilon=0.01, la_beta=1.0, la_beta_by_depth=(),
                    la_mode='post_rzf', decode_order='rbg_major',
                    cqi_mode='nr4bit', episode_len_main=1)
    defaults.update(overrides)
    return Config(**defaults)


class StaticFeedbackEnvironment:
    """Only the read-only channel/feedback interface the sampler needs."""

    def __init__(self, cfg, h_hat, h_true=None, cqi=None):
        self.cfg = cfg
        self.h_hat_slot = np.asarray(h_hat, dtype=complex).copy()
        self.h_true_slot = (self.h_hat_slot.copy() if h_true is None
                            else np.asarray(h_true, dtype=complex).copy())
        if cqi is None:
            cqi = np.log2(1.0 + np.sum(abs(self.h_hat_slot) ** 2, axis=-1))
        self.csi = SimpleNamespace(cqi_fb=np.asarray(cqi, dtype=float).copy())
        self.noise_var = 1.0
        self.episode_indices = []
        self.step_count = 0

    def reset(self, episode_idx):
        self.episode_indices.append(episode_idx)

    def step(self, action):
        # The channel-only experiment must never schedule traffic.
        np.testing.assert_array_equal(action, np.zeros((self.cfg.num_rbg,
                                                      self.cfg.l_max)))
        self.step_count += 1


def orthogonal_environment(cfg=None, *, outage=False):
    cfg = analytic_config() if cfg is None else cfg
    h = np.zeros((cfg.num_ue, cfg.num_rbg, cfg.num_bs_ant), dtype=complex)
    for u in range(min(4, cfg.num_ue, cfg.num_bs_ant)):
        h[u, :, u] = np.sqrt(10.0)
    actual = np.zeros_like(h) if outage else h.copy()
    return StaticFeedbackEnvironment(cfg, h, actual)


class RuntimeClosureOracle(unittest.TestCase):
    def test_dropped_zero_feedback_stream_releases_power_before_measurement(self):
        cfg = analytic_config(num_ue=2)
        h = np.zeros((2, 1, 4), dtype=complex)
        h[0, 0, 0] = np.sqrt(10.0)
        planner = SlotAllocationPlanner(cfg, h, 1.0, 1.0,
                                        np.full(2, np.inf))
        closed = planner.solve_closure(0, [], [0, 1])
        self.assertEqual(closed['group'], [0])
        # A nominal two-stream calculation would incorrectly return 5.
        self.assertAlmostEqual(float(closed['sinr'][0]), 10.0, places=12)
        self.assertAlmostEqual(float(closed['caps'][0]), np.log2(11.0), places=12)
        self.assertAlmostEqual(float(np.sum(abs(closed['w'])**2)), 1.0, places=12)

    def test_epsilon_closure_uses_beta_for_final_depth(self):
        cfg = analytic_config(num_ue=2, b_tx_epsilon=0.1,
                              la_beta_by_depth=(0.8, 0.01, 0.7, 0.6))
        h = np.zeros((2, 1, 4), dtype=complex)
        h[0, 0, 0] = np.sqrt(1e9)
        h[1, 0, 1] = 1.0
        planner = SlotAllocationPlanner(cfg, h, 1.0, 1.0,
                                        np.full(2, np.inf))
        closed = planner.solve_closure(0, [], [0, 1])
        self.assertEqual(closed['group'], [0])
        # The weak stream is removed at depth 2; remaining SU uses beta_1.
        self.assertAlmostEqual(float(closed['caps'][0]),
                               0.8 * np.log2(1.0 + 1e9), places=10)


class CQI4Eligibility(unittest.TestCase):
    def test_cqi_zero_or_zero_reconstructed_channel_never_enters_sample_pool(self):
        cfg = analytic_config(num_ue=4)
        h = np.eye(4, dtype=complex)
        h[2] = 0.0
        # UE 1 has zero CQI; UE 2 has zero reconstructed feedback channel;
        # UE 3 is positive but below the runtime's preselection epsilon.
        cqi = np.array([1.0, 0.0, 1.0, cfg.b_tx_epsilon / 2.0])
        actual = cqi4.eligible_users(cqi, h, cfg)
        np.testing.assert_array_equal(actual, [0])

    def test_sampled_groups_use_only_eligible_ues_without_replacement(self):
        population = np.array([1, 3, 5, 7, 9])
        rng = np.random.default_rng(883)
        observed = set()
        for _ in range(100):
            group = cqi4.sample_group(population, 3, rng)
            self.assertEqual(len(group), 3)
            self.assertEqual(len(set(group)), 3)
            self.assertTrue(set(group).issubset(set(population)))
            self.assertEqual(list(group), sorted(group))
            observed.update(group)
        self.assertEqual(observed, set(population))

    def test_insufficient_eligible_population_is_skipped_without_fabricated_failure(self):
        rng = np.random.default_rng(15)
        self.assertIsNone(cqi4.sample_group(np.array([2]), 2, rng))

    def test_nonfinite_feedback_is_a_diagnostic_error(self):
        cfg = analytic_config(num_ue=1)
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.eligible_users(np.array([np.nan]), np.ones((1, 4)), cfg)
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.eligible_users(np.array([1.0]), np.full((1, 4), np.nan), cfg)


class CQI4Measurement(unittest.TestCase):
    def test_su_and_mu_capacity_match_independent_orthogonal_channel_oracle(self):
        cfg = analytic_config(la_beta=0.25,
                              la_beta_by_depth=(0.8, 0.7, 0.6, 0.5))
        for depth in range(1, 5):
            h = np.eye(4, dtype=complex)[:depth] * np.sqrt(10.0)
            measured = cqi4.measure_group(h, h.copy(), cfg, 1.0)
            # Equal stream power is 1/depth; the neutral prediction excludes
            # old beta values, even when supplied in the saved configuration.
            capacity = np.log2(1.0 + 10.0 / depth)
            np.testing.assert_allclose(measured['raw_cap'], capacity, rtol=1e-12)
            np.testing.assert_allclose(measured['mi'], capacity, rtol=1e-12)
            np.testing.assert_allclose(measured['ratio'], 1.0, rtol=1e-12)
        self.assertEqual(cfg.la_beta_by_depth, (0.8, 0.7, 0.6, 0.5))

    def test_real_outage_with_positive_prediction_remains_zero_ratio(self):
        cfg = analytic_config()
        h = np.eye(4, dtype=complex)[:2] * np.sqrt(10.0)
        measured = cqi4.measure_group(h, np.zeros_like(h), cfg, 1.0)
        self.assertTrue(np.all(measured['raw_cap'] > 0.0))
        np.testing.assert_array_equal(measured['mi'], [0.0, 0.0])
        np.testing.assert_array_equal(measured['ratio'], [0.0, 0.0])

    def test_prediction_rate_factor_does_not_scale_realized_mutual_information(self):
        cfg = analytic_config(beta_rate=0.8)
        h = np.eye(4, dtype=complex)[:1] * np.sqrt(10.0)
        measured = cqi4.measure_group(h, h.copy(), cfg, 1.0)
        np.testing.assert_allclose(measured['raw_cap'], 0.8 * np.log2(11.0))
        np.testing.assert_allclose(measured['mi'], np.log2(11.0))
        np.testing.assert_allclose(measured['ratio'], 1.25)

    def test_undefined_zero_over_zero_is_rejected_as_invalid_group(self):
        cfg = analytic_config()
        h = np.zeros((2, 4), dtype=complex)
        h[0, 0] = np.sqrt(10.0)
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.measure_group(h, h.copy(), cfg, 1.0)

    def test_nonfinite_true_channel_is_not_silently_reclassified_as_failure(self):
        cfg = analytic_config()
        h = np.eye(4, dtype=complex)[:1]
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.measure_group(h, np.full_like(h, np.nan), cfg, 1.0)


class CQI4Sampling(unittest.TestCase):
    def collect(self, env, cfg=None):
        cfg = env.cfg if cfg is None else cfg
        with contextlib.redirect_stdout(io.StringIO()):
            return cqi4.collect_samples(env, cfg, 50000, 2, 777)

    def test_collect_excludes_cqi_zero_before_any_random_group_draw(self):
        env = orthogonal_environment()
        original = cqi4.sample_group
        seen = []

        def inspect_pool(eligible, depth, rng):
            seen.append(np.asarray(eligible).copy())
            np.testing.assert_array_equal(eligible, [0, 1, 2, 3])
            return original(eligible, depth, rng)

        with patch.object(cqi4, 'sample_group', side_effect=inspect_pool):
            result = self.collect(env)
        self.assertGreater(len(seen), 0)
        self.assertEqual(env.episode_indices, [50000, 50001])
        self.assertEqual(env.step_count, 2)
        for depth in range(1, 5):
            for episode in range(2):
                # Two draws per depth, each retaining the real sampled depth.
                self.assertEqual(len(result['ratios'][depth][episode]), 2 * depth)
                np.testing.assert_allclose(result['ratios'][depth][episode], 1.0)
                np.testing.assert_allclose(result['caps'][depth][episode],
                                           np.log2(1.0 + 10.0 / depth))
                np.testing.assert_allclose(result['mi'][depth][episode],
                                           np.log2(1.0 + 10.0 / depth))

    def test_true_channel_cannot_change_group_membership_or_predicted_caps(self):
        good = self.collect(orthogonal_environment())
        outage = self.collect(orthogonal_environment(outage=True))
        for depth in range(1, 5):
            for episode in range(2):
                np.testing.assert_array_equal(good['caps'][depth][episode],
                                              outage['caps'][depth][episode])
                self.assertEqual(len(good['ratios'][depth][episode]),
                                 len(outage['ratios'][depth][episode]))
                np.testing.assert_array_equal(outage['ratios'][depth][episode],
                                              np.zeros(2 * depth))
                np.testing.assert_array_equal(outage['mi'][depth][episode],
                                              np.zeros(2 * depth))

    def test_scarce_population_does_not_turn_skipped_groups_into_ratio_zero(self):
        cfg = analytic_config()
        h = np.zeros((5, 1, 4), dtype=complex)
        h[0, 0, 0] = np.sqrt(10.0)
        result = self.collect(StaticFeedbackEnvironment(cfg, h))
        for episode in range(2):
            self.assertEqual(len(result['ratios'][1][episode]), 2)
            for depth in (2, 3, 4):
                self.assertEqual(len(result['ratios'][depth][episode]), 0)


def per_depth(*episodes):
    return {depth: [np.asarray(ep, dtype=float).copy() for ep in episodes]
            for depth in range(1, 5)}


class CQI4FrozenCalibration(unittest.TestCase):
    def test_holdout_matches_actual_runtime_ack_tolerance_in_bits(self):
        target = 5.0
        actual_mi = np.array([target - 0.5e-6, target - 1.5e-6, target])
        runtime = []
        for i, value in enumerate(actual_mi):
            unit = TransmissionUnit(unit_id=i, packet_id=i, ue_id=i,
                                    rbg_id=0, previous_layer_id=0,
                                    current_layer_id=0, b_tx=target,
                                    i_acc=float(value))
            runtime.append(unit.is_acked)
        self.assertEqual(runtime, [True, False, True])
        for supply_actual_mi in (False, True):
            kwargs = ({'realized_mi': per_depth(actual_mi)}
                      if supply_actual_mi else {})
            with self.subTest(supply_actual_mi=supply_actual_mi):
                report = cqi4.summarize_holdout(
                    per_depth(actual_mi / 10.0), per_depth([10.0] * 3),
                    [0.5] * 4, analytic_config(), **kwargs)
                for depth in range(1, 5):
                    stats = report[str(depth)]
                    self.assertEqual(stats['acks'], sum(runtime))
                    self.assertAlmostEqual(stats['first_ack'], 2.0 / 3.0)
                    self.assertAlmostEqual(stats['strict_ratio_first_ack'], 1.0 / 3.0)
                    self.assertEqual(stats['ack_tolerance_bits'], 1e-6)

    def test_supplied_realized_mi_must_match_the_recorded_ratio(self):
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.summarize_holdout(per_depth([0.5]), per_depth([10.0]),
                                  [0.5] * 4, analytic_config(),
                                  realized_mi=per_depth([1.0]))

    def test_quantile_is_rounded_before_holdout_and_holdout_never_fits_beta(self):
        ratios = per_depth([0.123456, 0.223456, 0.323456, 0.423456,
                            0.523456, 0.623456, 0.723456, 0.823456,
                            0.923456, 1.023456])
        fitted = cqi4.fit_beta(ratios)
        np.testing.assert_allclose(fitted['beta_raw'], [0.213456] * 4)
        self.assertEqual(list(fitted['beta_rounded']), [0.2135] * 4)
        beta = list(fitted['beta_rounded'])
        original_beta = beta.copy()
        # 0.21348 passes against raw beta but fails against deployed 0.2135.
        heldout = per_depth([0.21348, 0.2135, 0.9], [0.21348, 0.2135, 0.9])
        caps = per_depth([100.0] * 3, [100.0] * 3)
        with patch.object(cqi4, 'fit_beta', side_effect=AssertionError('holdout retuned beta')):
            report = cqi4.summarize_holdout(heldout, caps, beta, analytic_config())
        self.assertEqual(beta, original_beta)
        for depth in range(1, 5):
            self.assertAlmostEqual(report[str(depth)]['first_ack'], 2.0 / 3.0)
            self.assertEqual(report[str(depth)]['member_count'], 6)
            self.assertEqual(report[str(depth)]['acks'], 4)
            self.assertTrue(np.isfinite(report[str(depth)]['episode_cluster_ci95']).all())

    def test_real_zero_ratios_are_kept_when_fitting_lower_quantile(self):
        # 10% true outages set the 10th percentile to 0.9 with NumPy's
        # documented default linear interpolation; removing them yields 1.0.
        result = cqi4.fit_beta(per_depth([0.0] + [1.0] * 9))
        np.testing.assert_allclose(result['beta_raw'], [0.9] * 4)

    def test_empty_depth_invalid_ratio_and_unusable_beta_fail_closed(self):
        for values in ([], [np.nan, 1.0], [np.inf, 1.0], [-0.1, 1.0],
                       [0.0] * 10, [2.0] * 10):
            with self.subTest(values=values), self.assertRaises(cqi4.CalibrationError):
                cqi4.fit_beta(per_depth(values))

    def test_beta_scaled_epsilon_gate_is_checked_at_deployment_precision(self):
        cfg = analytic_config(b_tx_epsilon=0.1)
        cqi4.validate_membership(per_depth([0.2, 10.0]), [0.5] * 4, cfg)
        # All raw capacities pass epsilon, but one beta-scaled payload fails.
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.validate_membership(per_depth([0.15, 10.0]), [0.5] * 4, cfg)

    def test_holdout_cannot_publish_ack_rate_when_frozen_beta_changes_membership(self):
        cfg = analytic_config(b_tx_epsilon=0.1)
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.summarize_holdout(per_depth([0.0, 1.0]),
                                  per_depth([0.15, 10.0]), [0.5] * 4, cfg)

    def test_holdout_ratio_cap_pairs_must_have_matching_shape(self):
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.summarize_holdout(per_depth([0.0, 1.0]),
                                  per_depth([10.0]), [0.5] * 4,
                                  analytic_config())

    def test_deployment_beta_must_be_finite_positive_and_cover_all_depths(self):
        cfg = analytic_config()
        for beta in ([0.0] * 4, [-0.1] * 4, [np.nan] * 4,
                     [2.0] * 4, [0.5] * 3):
            with self.subTest(beta=beta), self.assertRaises(cqi4.CalibrationError):
                cqi4.validate_membership(per_depth([10.0]), beta, cfg)

    def test_missing_holdout_depth_is_an_error_not_nan(self):
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.summarize_holdout(per_depth([]), per_depth([]),
                                  [0.5] * 4, analytic_config())

    def test_undefined_episode_bootstrap_is_reported_as_error(self):
        # A whole resampled cluster draw can contain only the empty episode.
        # It must not silently become NaN or be discarded to condition the CI.
        with self.assertRaises(cqi4.CalibrationError):
            cqi4.summarize_holdout(per_depth([], [1.0]),
                                  per_depth([], [10.0]), [0.5] * 4,
                                  analytic_config())


if __name__ == '__main__':
    unittest.main()
