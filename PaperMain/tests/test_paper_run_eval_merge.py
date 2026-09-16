"""Failure gates and statistical-unit checks for paired live-run evaluation merges."""
from pathlib import Path
import copy
import csv
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_run_eval_merge import (COUNT_COLUMNS, MEAN_METRICS, REQUIRED_PROTOCOL,
    build_summary, load_shards, merge_shards, render_markdown, sha_bytes)


class RunEvaluationMerge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = dict.fromkeys(REQUIRED_PROTOCOL, 'fixed')
        self.protocol.update(name='paper-live-run-id-v1', episode_ids=[120000, 120001, 120002, 120003],
                             scheduler_keys=['ppo', 'sus_cqi'],
                             scheduler_names={'ppo': 'PPO', 'sus_cqi': 'SUS+CQI'},
                             checkpoint_update=999, diagnostic_only=False,
                             checkpoint_sha256='a' * 64,
                             training_config={'seed': 2024}, environment_config={'traffic_model': 'ftp3'},
                             policy_config={'seed': 2024}, source_sha256={'traffic.py': 'b' * 64})
        self.header = ['scheduler', 'scheduler_name', 'episode_idx'] + list(MEAN_METRICS) + list(COUNT_COLUMNS)
        self.shards = [self.root / 'gpu3', self.root / 'gpu5']
        for index, directory in enumerate(self.shards):
            directory.mkdir()
            episodes = self.protocol['episode_ids'][index::2]
            rows = []
            for ep in episodes:
                for key in self.protocol['scheduler_keys']:
                    row = dict.fromkeys(self.header, 0)
                    row.update(scheduler=key, scheduler_name=self.protocol['scheduler_names'][key],
                               episode_idx=ep, goodput_mbps=ep - 120000 + (2 if key == 'ppo' else 0),
                               throughput_mbps=4, reward=ep - 120000,
                               completion_rate=0.5, deadline_miss_rate=0.2, buffer_overflow_rate=0.1,
                               retx_drop_rate=0.1, mu_depth=2, jain=0.8,
                               n_arrivals=9, n_offered=10, n_buffer_overflow=1,
                               n_units_new=10, n_units_first_ack=8,
                               acks_m1=8, units_m1=10, n_comp=5, n_miss_deadline=2, n_retx_drop=1,
                               n_retx_overflow_drop=0, terminal_queued_packets=1)
                    rows.append(row)
            self.write_rows(directory, rows)
            manifest = dict(protocol=copy.deepcopy(self.protocol), status='completed',
                            model_state_sha256_before='c' * 64, model_state_sha256_after='c' * 64,
                            model_state_unchanged=True, torch_rng_unchanged=True,
                            runtime={'python': '3.11', 'packages': {'torch': '2.7'}, 'gpu_name': 'RTX 4090'},
                            execution=dict(shard_index=index, shard_count=2, gpu=(3, 5)[index],
                                           episode_ids=episodes, device='cuda', threads=4),
                            metrics_sha256=sha_bytes((directory / 'metrics.csv').read_bytes()),
                            rows_written=len(rows))
            self.write_manifest(directory, manifest)

    def read_rows(self, path):
        with (path / 'metrics.csv').open(newline='') as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, path, rows):
        with (path / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(rows)

    def write_manifest(self, path, manifest):
        (path / 'manifest.json').write_text(json.dumps(manifest))

    def mutate_manifest(self, index, fn):
        path = self.shards[index]
        manifest = json.loads((path / 'manifest.json').read_text())
        fn(manifest)
        self.write_manifest(path, manifest)

    def mutate_rows(self, index, fn):
        path = self.shards[index]
        rows = self.read_rows(path)
        fn(rows)
        self.write_rows(path, rows)
        self.mutate_manifest(index, lambda m: m.update(
            metrics_sha256=sha_bytes((path / 'metrics.csv').read_bytes()), rows_written=len(rows)))

    def test_complete_merge_uses_episode_not_row_count(self):
        result = merge_shards(self.shards[::-1], self.root / 'combined', 200, 7)
        self.assertEqual((result['episodes'], result['rows']), (4, 8))
        self.assertEqual(result['bootstrap']['statistical_unit'], 'episode')
        self.assertEqual(result['by_scheduler']['ppo']['episodes'], 4)
        self.assertEqual(result['by_scheduler']['ppo']['raw_count_totals']['n_offered'], 40)
        self.assertEqual(result['by_scheduler']['ppo']['raw_count_totals']['terminal_queued_packets'], 4)
        self.assertEqual(set(p.name for p in (self.root / 'combined').iterdir()),
                         {'mergedmetrics.csv', 'summary.json', 'summary.md'})

    def test_paired_bootstrap_constant_gain_has_zero_width(self):
        protocol, rows, _, inputs = load_shards(self.shards)
        result = build_summary(protocol, rows, inputs, 300, 7)
        delta = result['paired_ppo_minus_baseline']['sus_cqi']['goodput_mbps']
        self.assertEqual(delta, {'mean_difference': 2.0, 'ci95': [2.0, 2.0]})
        self.assertEqual(result, build_summary(protocol, rows, inputs, 300, 7))
        self.assertNotEqual(result['by_scheduler']['ppo']['metrics']['goodput_mbps']['ci95'], [3.5, 3.5])

    def test_ack_counts_are_pooled_and_empty_bins_undefined(self):
        def change(rows):
            for r in rows:
                r.update(acks_m1=0, units_m1=0, n_units_new=0, n_units_first_ack=0)
            rows[0].update(acks_m2=1, units_m2=1, n_units_new=1, n_units_first_ack=1)
            rows[2].update(acks_m2=1, units_m2=9, n_units_new=9, n_units_first_ack=1)
        self.mutate_rows(0, change)
        self.mutate_rows(1, lambda rows: [r.update(acks_m1=0, units_m1=0, n_units_new=0,
                                                n_units_first_ack=0) for r in rows])
        protocol, rows, _, inputs = load_shards(self.shards)
        result = build_summary(protocol, rows, inputs, 100, 0)
        depths = result['by_scheduler']['ppo']['first_ack_by_depth']
        self.assertEqual(depths['2'], dict(acks=2, units=10, rate=.2))
        self.assertEqual(depths['1'], dict(acks=0, units=0, rate=None))
        self.assertIn('undefined (0/0)', render_markdown(result))
        json.dumps(result, allow_nan=False)

    def test_partial_shard_rejected_without_output(self):
        self.mutate_manifest(1, lambda m: m.update(status='running'))
        output = self.root / 'combined'
        with self.assertRaisesRegex(ValueError, 'not completed'):
            merge_shards(self.shards, output)
        self.assertFalse(output.exists())

    def test_missing_entire_shard_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing shards'):
            load_shards(self.shards[:1])

    def test_changed_checkpoint_or_config_or_source_rejected(self):
        for field in ('checkpoint_sha256', 'policy_config', 'environment_config', 'source_sha256',
                      'history_audit_sha256'):
            original = json.loads((self.shards[1] / 'manifest.json').read_text())
            self.mutate_manifest(1, lambda m: m['protocol'].update({field: 'changed'}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                load_shards(self.shards)
            self.write_manifest(self.shards[1], original)

    def test_post_completion_csv_corruption_rejected(self):
        with (self.shards[0] / 'metrics.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_shards(self.shards)

    def test_missing_metrics_hash_rejected(self):
        self.mutate_manifest(0, lambda m: m.pop('metrics_sha256'))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_shards(self.shards)

    def test_wrong_modulo_split_rejected(self):
        self.mutate_manifest(1, lambda m: m['execution'].update(episode_ids=[120000, 120002]))
        with self.assertRaisesRegex(ValueError, 'modulo'):
            load_shards(self.shards)

    def test_duplicate_and_missing_pairs_rejected(self):
        self.mutate_rows(0, lambda rows: rows.__setitem__(1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, 'duplicate episode'):
            load_shards(self.shards)

    def test_missing_rows_rejected_even_if_status_and_hash_complete(self):
        self.mutate_rows(1, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'missing 1 planned rows'):
            load_shards(self.shards)

    def test_wrong_scheduler_display_label_rejected(self):
        self.mutate_rows(1, lambda rows: rows[0].update(scheduler_name='Different PPO'))
        with self.assertRaisesRegex(ValueError, 'display name mismatch'):
            load_shards(self.shards)

    def test_undefined_sinr_requires_zero_sample_count_and_stays_blank(self):
        self.header += ['sinr_count', 'mean_sinr_db']
        for index in range(2):
            self.mutate_rows(index, lambda rows: [r.update(sinr_count=0, mean_sinr_db='') for r in rows])
        result = merge_shards(self.shards, self.root / 'empty_sinr', 100, 0)
        with (self.root / 'empty_sinr' / 'mergedmetrics.csv').open() as handle:
            self.assertEqual(next(csv.DictReader(handle))['mean_sinr_db'], '')
        self.assertNotIn('mean_sinr_db', result['by_scheduler']['ppo']['metrics'])
        self.mutate_rows(0, lambda rows: rows[0].update(sinr_count=1))
        with self.assertRaisesRegex(ValueError, 'nonnumeric mean_sinr_db'):
            load_shards(self.shards)

    def test_mutated_model_or_rng_rejected(self):
        original = json.loads((self.shards[1] / 'manifest.json').read_text())
        for changes, message in (({'model_state_unchanged': False}, 'model-state'),
                                 ({'model_state_sha256_after': 'd' * 64}, 'model-state'),
                                 ({'torch_rng_unchanged': False}, 'Torch RNG')):
            self.mutate_manifest(1, lambda m: m.update(changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                load_shards(self.shards)
            self.write_manifest(self.shards[1], original)

    def test_runtime_and_execution_mismatches_rejected(self):
        original = json.loads((self.shards[1] / 'manifest.json').read_text())
        self.mutate_manifest(1, lambda m: m['runtime'].update(python='3.12'))
        with self.assertRaisesRegex(ValueError, 'runtime or loaded-model'):
            load_shards(self.shards)
        self.write_manifest(self.shards[1], original)
        self.mutate_manifest(1, lambda m: m['execution'].update(threads=8))
        with self.assertRaisesRegex(ValueError, 'execution settings'):
            load_shards(self.shards)

    def test_row_metadata_must_match_protocol(self):
        self.header += ['world', 'checkpoint_update', 'train_seed', 'diagnostic_only', 'slots']
        for index in range(2):
            self.mutate_manifest(index, lambda m: m['protocol']['environment_config'].update(
                debug=False, episode_len_main=1000))
            self.mutate_rows(index, lambda rows: [r.update(world='ID', checkpoint_update=999,
                train_seed=2024, diagnostic_only=0, slots=1000) for r in rows])
        load_shards(self.shards)
        original = self.read_rows(self.shards[0])
        for changes in ({'world': 'OOD'}, {'checkpoint_update': 1000}, {'train_seed': 7},
                        {'diagnostic_only': 1}, {'slots': 32}):
            self.mutate_rows(0, lambda rows: rows[0].update(changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_shards(self.shards)
            self.write_rows(self.shards[0], original)

    def test_nan_scalar_and_invalid_ack_counts_rejected(self):
        original = self.read_rows(self.shards[0])
        for changes, message in (({'goodput_mbps': 'nan'}, 'nonfinite'),
                                 ({'acks_m1': '11'}, 'ACK count exceeds'),
                                 ({'n_units_first_ack': '7'}, 'do not sum'),
                                 ({'n_offered': '9'}, 'offered-arrival'),
                                 ({'n_arrivals': '1.2'}, 'invalid count')):
            self.mutate_rows(0, lambda rows: rows[0].update(changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                load_shards(self.shards)
            self.write_rows(self.shards[0], original)

    def test_smoke_requires_explicit_opt_in_and_is_labelled(self):
        for index in range(2):
            self.mutate_manifest(index, lambda m: m['protocol'].update(diagnostic_only=True))
        with self.assertRaisesRegex(ValueError, 'allow-smoke'):
            load_shards(self.shards)
        result = merge_shards(self.shards, self.root / 'smoke', 100, 0, allow_smoke=True)
        self.assertIn('DIAGNOSTIC SMOKE RUN', (self.root / 'smoke' / 'summary.md').read_text())
        self.assertTrue(result['protocol']['diagnostic_only'])

    def test_existing_outputs_are_never_overwritten(self):
        output = self.root / 'combined'
        output.mkdir()
        sentinel = output / 'summary.json'
        sentinel.write_text('original')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            merge_shards(self.shards, output)
        self.assertEqual(sentinel.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
