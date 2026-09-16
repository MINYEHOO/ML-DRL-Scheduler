"""Profile acceptance guards; no wireless channel generation or GPU use."""
import copy
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration import profile as cp
from config import Config


class CalibrationProfileGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*cp.SCIENCE_SOURCES, 'calibration/profile.py',
                     'calibration/cqi4.py', 'calibration/beta_m_calib_holdout.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixed source for calibration guard test\n')
        self.original_config = json.loads((ROOT / cp.BASE_CONFIG).read_text())
        config_path = self.root / cp.BASE_CONFIG
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(self.original_config))
        self.profile = {
            'schema_version': cp.SCHEMA_VERSION,
            'protocol': cp.PROTOCOL,
            'status': 'validated',
            'config': cp.normalized_config(self.original_config),
            'reference_config_sha256': cp.sha256_file(config_path),
            'source_sha256': cp.source_hashes(self.root),
            'beta_rounded': [1.0, .8, .7, .6],
            'membership_verified_calibration': True,
            'membership_verified_holdout': True,
            'target_coverage': .9,
            'calibration': {'start': 50000, 'episodes': 36, 'slots': 1000,
                            'sampler_seed': 777},
            'holdout': {'start': 70000, 'episodes': 12, 'slots': 1000,
                        'sampler_seed': 20250713, 'beta_rounded': [1.0, .8, .7, .6],
                        'by_depth': {str(m): {'member_count': 1000, 'acks': 900,
                                             'first_ack': .9,
                                             'episode_cluster_ci95': [.88, .92]}
                                     for m in range(1, 5)}},
        }
        self.path = self.root / 'calibration_summary.json'

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, profile=None):
        self.path.write_text(json.dumps(profile if profile is not None else self.profile))
        return cp.load_profile(self.path, self.root)

    def assert_rejected(self, mutate, pattern):
        candidate = copy.deepcopy(self.profile)
        mutate(candidate)
        with self.assertRaisesRegex(ValueError, pattern):
            self.load(candidate)

    def test_complete_profile_is_read_only_and_beta_is_opt_in(self):
        loaded = self.load()
        raw = copy.deepcopy(self.original_config)
        after = cp.apply_profile(raw, loaded)
        self.assertEqual(raw, self.original_config)
        expected = {**raw, 'la_beta': 1., 'la_beta_by_depth': loaded['beta_rounded']}
        self.assertEqual(after, expected)
        self.assertEqual(json.loads((self.root / cp.BASE_CONFIG).read_text()), raw)
        self.assertEqual(json.loads(self.path.read_text()), self.profile)

    def test_dataclass_application_preserves_feature_normalization_and_schedule(self):
        raw = cp.normalized_config(self.original_config)
        # An explicitly chosen OOD or narrow transfer changes no learned scales.
        raw.update(deadline_max=12, num_ue=48, ue_speed_max=20.)
        cfg = Config(**raw)
        before = dataclasses.asdict(cfg)
        actual = cp.apply_profile(cfg, self.load())
        expected = {**before, 'la_beta': 1., 'la_beta_by_depth': (1., .8, .7, .6)}
        self.assertEqual(dataclasses.asdict(actual), expected)
        self.assertEqual(dataclasses.asdict(cfg), before)

    def test_profile_cannot_be_transferred_to_genie_or_different_la_world(self):
        loaded = self.load()
        for changes in ({'pmi_mode': 'genie', 'cqi_mode': 'continuous'},
                        {'cqi_mode': 'continuous'}, {'la_mode': 'legacy'},
                        {'decode_order': 'layer_major'}):
            cfg = {**self.original_config, **changes}
            with self.assertRaisesRegex(ValueError, 'imperfect-CSI post-RZF'):
                cp.apply_profile(cfg, loaded)

    def test_historical_smoke_and_running_results_cannot_be_applied(self):
        for status in ('historical_diagnostic_only', 'diagnostic_only', 'running', 'failed'):
            self.assert_rejected(lambda p: p.update(status=status), 'validated status')

    def test_wrong_schema_and_protocol_are_rejected(self):
        self.assert_rejected(lambda p: p.update(schema_version=2), 'schema/protocol')
        self.assert_rejected(lambda p: p.update(protocol='archived'), 'schema/protocol')

    def test_full_length_and_counts_required_for_both_phases(self):
        for phase in ('calibration', 'holdout'):
            self.assert_rejected(lambda p: p[phase].update(episodes=1), 'complete 1000-slot')
            self.assert_rejected(lambda p: p[phase].update(slots=32), 'complete 1000-slot')
        self.assert_rejected(lambda p: p['config'].update(episode_len_main=32), 'full-length Base')

    def test_fit_and_holdout_must_both_verify_membership(self):
        for field in ('membership_verified_calibration', 'membership_verified_holdout'):
            self.assert_rejected(lambda p: p.update({field: False}), 'membership')
            self.assert_rejected(lambda p: p.pop(field), 'membership')

    def test_holdout_cannot_reuse_calibration_episode_indices(self):
        self.assert_rejected(lambda p: p['holdout'].update(start=50035), 'disjoint')

    def test_holdout_must_use_frozen_rounded_beta(self):
        self.assert_rejected(lambda p: p['holdout'].update(beta_rounded=[1., .8, .7, .61]), 'exact frozen')

    def test_invalid_beta_values_rejected(self):
        for beta in ([], [1., .8, .7], [1., .8, .7, 0.],
                     [1., .8, .7, -1.], [1., .8, .7, float('nan')],
                     [1., .8, .7, float('inf')], [1., .8, .7, 1.6],
                     [1., .8, .7, .60001], [1., .8, .7, True]):
            with self.subTest(beta=beta):
                self.assert_rejected(lambda p: p.update(beta_rounded=beta), 'beta')

    def test_every_final_depth_needs_nonempty_finite_holdout_evidence(self):
        self.assert_rejected(lambda p: p['holdout']['by_depth'].pop('4'), 'every final')
        for changes in ({'member_count': 0}, {'first_ack': float('nan')},
                        {'episode_cluster_ci95': [.92, .88]}, {'acks': 901},
                        {'first_ack': 1.1}):
            self.assert_rejected(lambda p: p['holdout']['by_depth']['4'].update(changes), 'holdout evidence')

    def test_hidden_transfer_metadata_does_not_change_source_hashes(self):
        expected = cp.source_hashes(self.root)
        sidecars = [self.root / 'calibration' / name
                    for name in ('._cqi4.py', '._profile.py', '.transfer.py')]
        for path in sidecars:
            path.write_bytes(bytes.fromhex('00051607') + b' transfer metadata')
        self.assertEqual(cp.source_hashes(self.root), expected)
        self.load()  # Profile remains valid with transfer sidecars present.
        for path in sidecars:
            path.unlink()
        self.assertEqual(cp.source_hashes(self.root), expected)
        support = self.root / 'calibration' / 'new_support.py'
        support.write_text('# genuine calibration support module\n')
        self.assertNotEqual(cp.source_hashes(self.root), expected)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.load()
        support.unlink()
        self.assertEqual(cp.source_hashes(self.root), expected)

    def test_source_changes_including_calibration_support_fail_closed(self):
        for name in ('phy.py', 'calibration/profile.py', 'calibration/cqi4.py'):
            with self.subTest(name=name):
                path = self.root / name
                before = path.read_text()
                path.write_text('# new source\n')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.load()
                path.write_text(before)

    def test_reference_config_and_realized_world_are_bound(self):
        self.assert_rejected(lambda p: p.update(reference_config_sha256='bad'), 'Base configuration')
        for changes in ({'p_csi': .99}, {'seed': 3024}, {'ue_speed_max': 20.}):
            self.assert_rejected(lambda p: p['config'].update(changes), 'full-length Base')

    def test_legacy_beta_is_excluded_from_world_comparison(self):
        candidate = copy.deepcopy(self.profile)
        candidate['config'].update(la_beta=.5, la_beta_by_depth=[.5] * 4)
        self.assertEqual(self.load(candidate), candidate)

    def test_target_is_not_a_claim_of_exact_holdout_ack(self):
        candidate = copy.deepcopy(self.profile)
        candidate['holdout']['by_depth']['4'].update(
            acks=875, first_ack=.875, episode_cluster_ci95=[.85, .898])
        self.assertEqual(self.load(candidate), candidate)


if __name__ == '__main__':
    unittest.main(verbosity=2)
