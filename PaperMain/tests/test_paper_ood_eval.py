"""Meaningful OOD protocol checks, independent-pilot guards and frozen scales."""
import copy
import csv
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_ood_eval as ev
import paper_train as pt


def base_config():
    from config import phase4_queue_config
    cfg = phase4_queue_config()
    cfg.traffic_model = 'bernoulli'
    cfg.ue_speed_min, cfg.ue_speed_max = 5., 40.
    cfg.p_arrival_min, cfg.p_arrival_max = .15, .50
    return pt.normalize_config(dataclasses.asdict(cfg))


class WorldProtocol(unittest.TestCase):
    def test_all_worlds_only_apply_declared_environment_changes(self):
        base = base_config()
        narrow = copy.deepcopy(base)
        narrow.update(n_active_min=24, n_active_max=24, ue_speed_min=22.5,
                      ue_speed_max=22.5, p_arrival_min=.325, p_arrival_max=.325)
        saved = copy.deepcopy((base, narrow))
        for world, changes in ev.WORLDS.items():
            environment, policy = ev.make_configs(base, narrow, world)
            self.assertEqual(environment, {**base, **changes})
            self.assertEqual(policy, {**narrow, 'num_ue': environment['num_ue']})
        self.assertEqual((base, narrow), saved)

    def test_deadline_and_k_normalization_are_separate(self):
        base = base_config()
        environment, policy = ev.make_configs(base, base, 'D26strict')
        self.assertEqual((environment['deadline_min'], environment['deadline_max']), (2, 6))
        self.assertEqual(policy['deadline_max'], 12)
        for world, k in [('K8', 8), ('K48', 48), ('K60', 60)]:
            environment, policy = ev.make_configs(base, base, world)
            self.assertEqual((environment['num_ue'], environment['n_active_min'], environment['n_active_max']), (k, k, k))
            self.assertEqual(policy, {**base, 'num_ue': k})

    def test_v60_preserves_historical_lower_endpoint_and_storm(self):
        base = base_config()
        for world in ('V60max', 'STORM2'):
            environment, _ = ev.make_configs(base, base, world)
            self.assertEqual((environment['ue_speed_min'], environment['ue_speed_max']), (5., 60.))
        environment, _ = ev.make_configs(base, base, 'STORM2')
        self.assertEqual((environment['p_arrival_min'], environment['p_arrival_max'], environment['p_csi']), (.55, .55, .2))

    def test_smoke_shortens_environment_only(self):
        base = base_config()
        environment, policy = ev.make_configs(base, base, 'ID', 32)
        self.assertEqual(environment, {**base, 'debug': False, 'episode_len_main': 32})
        self.assertEqual(policy, base)

    def test_wrong_traffic_and_deadline_scales_rejected(self):
        base = base_config()
        for change in ({'traffic_model': 'ftp3'}, {'deadline_max': 6}):
            with self.assertRaises(ValueError):
                ev.make_configs(base, {**base, **change}, 'ID')

    def test_random_seed_depends_on_world_episode_key_not_shard(self):
        a = ev.random_seed_components(2024, 'ID', 140000, 'sus_random')
        self.assertEqual(a, ev.random_seed_components(2024, 'ID', 140000, 'sus_random'))
        self.assertNotEqual(a, ev.random_seed_components(2024, 'K8', 140000, 'sus_random'))
        self.assertNotEqual(a, ev.random_seed_components(2024, 'ID', 140001, 'sus_random'))
        self.assertNotEqual(a, ev.random_seed_components(2024, 'ID', 140000, 'su_random'))

    def test_history_requires_scanned_selected_runs_and_distinct_purpose(self):
        cfg = base_config()
        selected = dict(run=Path('/runs/16'), training_config=cfg,
                        manifest={'target_updates': 2000}, calibration_profile=None)
        base = {**selected, 'run': Path('/runs/21')}
        args = SimpleNamespace(mode='eval', smoke_slots=None, episode_start=140000, episodes=100)
        document = dict(status='validated', checked_run_names=['16', '21', '22'], scanned_paths=['runs', 'results'],
                        allowed_episode_ranges=[dict(start=140000, episodes=100, purpose='eval')])
        ev.validate_history(document, args, selected, base)
        for field, value in [('checked_run_names', ['21']), ('scanned_paths', []), ('status', 'pending')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.validate_history({**document, field: value}, args, selected, base)
        args.mode = 'pilot'
        with self.assertRaises(ValueError):
            ev.validate_history(document, args, selected, base)


class PilotThresholdProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ev.RUNNER_FILES:
            (self.root / name).write_text('# immutable runner\n')
        self.cfg = base_config()
        self.base = dict(training_config=self.cfg, input_hashes={'runs/21/config.json': 'fixed'},
                         source_provenance={'current_source_sha256': {'policy.py': 'fixed'}})
        self.episodes = list(range(139000, 139008))
        self.args = SimpleNamespace(worlds=['ID', 'D26strict'], episode_start=140000, episodes=100, smoke_slots=None)
        self.document = dict(status='validated', diagnostic_only=False, selection_metric='reward',
                             selection_rule=ev.SELECTION_RULE, pilot_episode_ids=self.episodes,
                             thresholds={'ID': .7, 'D26strict': .6}, pilot_directories=['results/pilot'])
        self.directory = self.root / 'results/pilot'
        self.directory.mkdir(parents=True)
        self.rows = []
        for world in self.args.worlds:
            for threshold in ev.THRESHOLDS:
                for ep in self.episodes:
                    self.rows.append(dict(world=world, scheduler='sus_cqi', mode='pilot', threshold=threshold,
                        episode_idx=ep, reward=100 if world == 'D26strict' else 100 - abs(threshold - .7),
                        slots=self.cfg['episode_len_main']))
        self.record = dict(status='completed', inputs_unchanged=True, model_state_unchanged=True, torch_rng_unchanged=True,
            protocol=dict(mode='pilot', diagnostic_only=False, smoke_slots=None, episode_ids=self.episodes,
                          scheduler_keys=['sus_cqi'], candidate_thresholds=list(ev.THRESHOLDS),
                          base_config=self.cfg, base_input_hashes=self.base['input_hashes'],
                          current_source_sha256=self.base['source_provenance']['current_source_sha256'],
                          runner_sha256={name: pt.sha_file(self.root / name) for name in ev.RUNNER_FILES},
                          worlds=self.args.worlds,
                          configurations={world: {'environment': ev.make_configs(self.cfg, self.cfg, world)[0]} for world in self.args.worlds}))
        self.document_path = self.root / 'results/thresholds.json'
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        with (self.directory / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.record['rows_written'] = len(self.rows)
        self.record['metrics_sha256'] = pt.sha_file(self.directory / 'metrics.csv')
        (self.directory / 'manifest.json').write_text(json.dumps(self.record))
        self.document_path.write_text(json.dumps(self.document))

    def load(self):
        return ev.load_thresholds(self.document_path, self.args, self.base, self.root)

    def test_pilot_recomputed_and_ties_choose_smaller_threshold(self):
        document, hashes = self.load()
        self.assertEqual(document['thresholds'], {'ID': .7, 'D26strict': .6})
        self.assertEqual(len(hashes), 3)

    def test_wrong_threshold_rejected_despite_validated_label(self):
        self.document['thresholds']['ID'] = .8
        self.save()
        with self.assertRaisesRegex(ValueError, 'differ from pilot rewards'):
            self.load()

    def test_missing_duplicate_and_unexpected_rows_rejected(self):
        original = copy.deepcopy(self.rows)
        for rows in (original[:-1], original + [original[0]], [{**original[0], 'scheduler': 'ppo'}] + original[1:]):
            self.rows = rows
            self.save()
            with self.assertRaises(ValueError):
                self.load()

    def test_pilot_test_overlap_rejected(self):
        self.args.episode_start = self.episodes[-1]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.load()

    def test_modified_pilot_metrics_rejected(self):
        with (self.directory / 'metrics.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.load()

    def test_incomplete_diagnostic_or_source_drift_rejected(self):
        original = copy.deepcopy(self.record)
        for key, value in [('status', 'running'), ('model_state_unchanged', False), ('torch_rng_unchanged', False)]:
            self.record = {**original, key: value}
            self.save()
            with self.assertRaises(ValueError):
                self.load()
        self.record = original
        self.record['protocol']['diagnostic_only'] = True
        self.save()
        with self.assertRaises(ValueError):
            self.load()
        self.record['protocol']['diagnostic_only'] = False
        self.record['protocol']['base_config'] = {**self.cfg, 'p_csi': .2}
        self.save()
        with self.assertRaisesRegex(ValueError, 'different common Base'):
            self.load()

    def test_diagnostic_pilot_only_consumed_by_compatible_smoke(self):
        self.document['diagnostic_only'] = True
        self.document['pilot_episode_ids'] = [138000]
        self.record['protocol'].update(diagnostic_only=True, smoke_slots=32, episode_ids=[138000])
        self.rows = [{**row, 'episode_idx': 138000, 'slots': 32} for row in self.rows if row['episode_idx'] == self.episodes[0]]
        for cfg in self.record['protocol']['configurations'].values():
            cfg['environment']['episode_len_main'] = 32
        self.save()
        with self.assertRaisesRegex(ValueError, 'Diagnostic thresholds'):
            self.load()
        self.args.smoke_slots, self.args.episode_start, self.args.episodes = 32, 138001, 1
        self.load()
        self.args.smoke_slots = 16
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            self.load()


if __name__ == '__main__':
    unittest.main()
