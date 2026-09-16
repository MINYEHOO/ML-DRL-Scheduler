"""Read-only live evaluation guards and scientific protocol invariants."""
import copy
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_run_eval as ev
import paper_train as pt


class LiveEvaluationGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*pt.SOURCES, 'paper_run_eval.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# frozen source\n')
        from config import phase4_queue_config
        cfg = phase4_queue_config()
        cfg.cqi_mode, cfg.la_mode, cfg.decode_order = 'nr4bit', 'post_rzf', 'rbg_major'
        cfg.la_beta_by_depth = (1.0162, .8509, .7732, .728)
        cfg.p_arrival_min, cfg.p_arrival_max = .15, .5
        cfg.sus_ortho_threshold = .67
        cfg.ue_speed_max = 40.
        path = self.root / 'artifacts/base/config.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dataclasses.asdict(cfg)))
        train_args = pt.parser().parse_args(['--recipe', 'base', '--name', '18_ftp3',
            '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
            '--traffic-model', 'ftp3', '--replay-mode', 'batched'])
        self.run, _, self.manifest = pt.plan(train_args, self.root)
        (self.run / 'ckpt').mkdir(parents=True)
        (self.run / 'ckpt/best.pt').write_bytes(b'frozen best bytes')
        (self.run / 'ckpt/latest.pt').write_bytes(b'frozen latest bytes')
        self.save_manifest()
        (self.run / 'config.json').write_text(json.dumps(self.manifest['config']))
        self.completion = dict(status='completed', returncode=0, target_completed=True, checkpoint_update=999)
        self.status = self.root / 'review/logs/18_ftp3_launch.json'
        self.status.parent.mkdir(parents=True)
        self.save_status()

    def tearDown(self):
        self.tmp.cleanup()

    def save_manifest(self):
        (self.run / 'paper_manifest.json').write_text(json.dumps(self.manifest))

    def save_status(self):
        self.status.write_text(json.dumps(self.completion))

    def args(self, *extra):
        return ev.parser().parse_args(['--run', 'runs/18_ftp3', '--episode-start', '120000',
                                      '--episodes', '100', '--out', 'results/campaign/shard0', *extra])

    def prepared(self, *extra):
        return ev.plan(self.args(*extra), self.root)

    def checkpoint(self):
        return dict(cfg=copy.deepcopy(self.manifest['config']), model={}, update=639,
                    lr_schedule=copy.deepcopy(self.manifest['schedule']))

    def test_full_config_ftp3_and_sus_threshold_preserved(self):
        p = self.prepared()
        self.assertEqual(p['protocol']['training_config'], self.manifest['config'])
        self.assertEqual(p['protocol']['environment_config'], self.manifest['config'])
        self.assertEqual(p['protocol']['policy_config'], self.manifest['config'])
        self.assertEqual(p['protocol']['environment_config']['traffic_model'], 'ftp3')
        self.assertEqual(p['protocol']['environment_config']['sus_ortho_threshold'], .67)
        self.assertFalse(p['out'].exists())

    def test_smoke_only_shortens_environment(self):
        p = self.prepared('--smoke-slots', '32')
        env_cfg = copy.deepcopy(self.manifest['config'])
        env_cfg.update(debug=False, episode_len_main=32)
        self.assertEqual(p['protocol']['environment_config'], env_cfg)
        self.assertEqual(p['protocol']['policy_config'], self.manifest['config'])
        self.assertTrue(p['protocol']['diagnostic_only'])

    def test_shards_complete_disjoint_and_protocol_identical(self):
        a = self.prepared('--shard-index', '0', '--num-shards', '2', '--gpu', '3')
        b = self.prepared('--shard-index', '1', '--num-shards', '2', '--gpu', '5')
        self.assertEqual(a['protocol'], b['protocol'])
        aa, bb = set(a['execution']['episode_ids']), set(b['execution']['episode_ids'])
        self.assertFalse(aa & bb)
        self.assertEqual(aa | bb, set(range(120000, 120100)))
        self.assertEqual(len(aa), 50)
        self.assertEqual(len(bb), 50)

    def test_invalid_shard_metadata_including_bool(self):
        for dims in [(0, 2, 0, 3), (0, 2, 2, 2), (False, 2, 0, 1), (0, True, 0, 1),
                     (0, 2, False, 1), (0, 2, 0, 1.0)]:
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                ev.episode_partition(*dims)

    def test_source_drift_rejected(self):
        (self.root / 'traffic.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            self.prepared()

    def test_existing_output_and_path_escape_rejected(self):
        out = self.root / 'results/campaign/shard0'
        out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'existing output'):
            self.prepared()
        with self.assertRaisesRegex(ValueError, 'below PaperMain/results'):
            self.prepared('--out', '../outside')

    def test_run_config_drift_rejected(self):
        cfg = copy.deepcopy(self.manifest['config'])
        cfg['traffic_model'] = 'bernoulli'
        (self.run / 'config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'config.json differs'):
            self.prepared()

    def test_incomplete_or_malformed_completion_rejected(self):
        for key, val in [('status', 'running'), ('returncode', False), ('returncode', 1),
                         ('target_completed', 1), ('checkpoint_update', 998), ('checkpoint_update', 999.0)]:
            with self.subTest(key=key, value=val):
                old = self.completion[key]
                self.completion[key] = val
                self.save_status()
                with self.assertRaises(ValueError):
                    self.prepared()
                self.completion[key] = old
        self.save_status()

    def test_training_and_selection_overlap_rejected(self):
        for start in (0, 999, 10000, 10002):
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'overlap'):
                self.prepared('--episode-start', str(start), '--episodes', '1')
        self.prepared('--episode-start', '10003', '--episodes', '1')

    def test_frozen_checkpoint_must_match_selected_bytes(self):
        frozen = self.root / 'results/campaign/inputs/best.pt'
        frozen.parent.mkdir(parents=True)
        frozen.write_bytes((self.run / 'ckpt/best.pt').read_bytes())
        self.assertEqual(self.prepared('--checkpoint-path', str(frozen))['checkpoint'], frozen.resolve())
        frozen.write_bytes(b'not the same checkpoint')
        with self.assertRaisesRegex(ValueError, 'Frozen checkpoint copy differs'):
            self.prepared('--checkpoint-path', str(frozen))

    def test_history_audit_range_and_purpose(self):
        doc = dict(status='validated', allowed_episode_ranges=[
            dict(start=119000, episodes=2, purpose='smoke'), dict(start=120000, episodes=100, purpose='id')])
        ev.validate_history_audit(doc, 120000, 100, False)
        ev.validate_history_audit(doc, 119000, 2, True)
        for args in [(120000, 101, False), (119000, 2, False), (120000, 1, True)]:
            with self.assertRaises(ValueError):
                ev.validate_history_audit(doc, *args)
        doc['allowed_episode_ranges'][0]['start'] = True
        with self.assertRaises(ValueError):
            ev.validate_history_audit(doc, 120000, 100, False)

    def test_history_audit_snapshot_bound_to_protocol(self):
        audit = self.root / 'results/campaign/inputs/history.json'
        audit.parent.mkdir(parents=True)
        audit.write_text(json.dumps(dict(status='validated', allowed_episode_ranges=[
            dict(start=120000, episodes=100, purpose='id')])))
        p = self.prepared('--history-audit', str(audit))
        self.assertEqual(p['protocol']['history_audit_sha256'], pt.sha_file(audit))

    def test_checkpoint_config_is_exact_and_metadata_types_strict(self):
        p = self.prepared()
        ck = self.checkpoint()
        ev.validate_checkpoint(ck, p)
        self.assertEqual(p['protocol']['checkpoint_update'], 639)
        for field, val in [('traffic_model', 'bernoulli'), ('deadline_max', 11), ('seed', 2024.0)]:
            bad = copy.deepcopy(ck)
            bad['cfg'][field] = val
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.validate_checkpoint(bad, self.prepared())
        for update in (True, 639.0, -1, 1000):
            bad = copy.deepcopy(ck)
            bad['update'] = update
            with self.subTest(update=update), self.assertRaises(ValueError):
                ev.validate_checkpoint(bad, self.prepared())

    def test_checkpoint_lr_schedule_mismatch_rejected(self):
        ck = self.checkpoint()
        ck['lr_schedule']['lr_decay_updates'] = 2000
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            ev.validate_checkpoint(ck, self.prepared())

    def test_calibration_beta_checked_against_saved_profile(self):
        # A malformed record is rejected before it can bypass strict beta/source validation.
        self.manifest['calibration_profile'] = dict(document={}, document_sha256='0' * 64)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'profile document changed'):
            self.prepared()

    def test_profile_reference_without_profile_rejected(self):
        self.manifest['calibration_reference_root'] = 'provenance/no_such_snapshot'
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.prepared()

    def test_model_hash_includes_normalizer_buffers(self):
        import torch
        model = torch.nn.Linear(2, 1)
        model.register_buffer('ret_std', torch.ones(()))
        before = ev.model_state_hash(model)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            model(torch.ones(1, 2))
        self.assertEqual(ev.model_state_hash(model), before)
        model.ret_std.add_(1)
        self.assertNotEqual(ev.model_state_hash(model), before)

    def test_packet_accounting_includes_boundary_censoring(self):
        ep = dict(n_arrivals=20, n_comp=8, n_miss_deadline=5, n_retx_drop=2, n_retx_overflow_drop=1)
        ev.validate_packet_accounting(ep, 4)
        with self.assertRaisesRegex(ValueError, 'packet accounting'):
            ev.validate_packet_accounting(ep, 3)

    def test_header_has_pooled_ack_and_censoring_counts(self):
        self.assertEqual(len(ev.HEADER), len(set(ev.HEADER)))
        for key in ('n_offered', 'n_arrivals', 'n_buffer_overflow', 'n_comp',
                    'n_miss_deadline', 'n_retx_drop', 'n_retx_overflow_drop', 'terminal_queued_packets'):
            self.assertIn(key, ev.HEADER)
        for depth in range(1, 5):
            self.assertIn(f'acks_m{depth}', ev.HEADER)
            self.assertIn(f'units_m{depth}', ev.HEADER)


if __name__ == '__main__':
    unittest.main()
