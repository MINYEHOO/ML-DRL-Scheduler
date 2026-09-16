"""OOD input provenance tests, including the exact reviewed legacy bridge."""
import copy
import dataclasses
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_ood_inputs as oi
import paper_train as pt
import paper_train_variants as pv
from calibration import profile as cp


class OODInputGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        names = set(pt.SOURCES) | set(cp.source_hashes(ROOT)) | {
            'paper_ood_inputs.py', 'paper_train_variants.py',
            'artifacts/base/config.json', 'artifacts/narrow_mean/config.json'}
        for name in names:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        shutil.copytree(ROOT / oi.FROZEN_REFERENCE, self.root / oi.FROZEN_REFERENCE)
        self.frozen = self.root / oi.FROZEN_REFERENCE
        base = self.frozen / 'artifacts/base/config.json'
        beta = [1.0162, .8509, .7732, .728]
        document = dict(
            schema_version=cp.SCHEMA_VERSION, protocol=cp.PROTOCOL, status='validated',
            reference_config_sha256=cp.sha256_file(base),
            config=cp.normalized_config(json.loads(base.read_text())),
            source_sha256=cp.source_hashes(self.frozen), beta_rounded=beta,
            membership_verified_calibration=True, membership_verified_holdout=True,
            calibration=dict(start=50000, episodes=36, slots=1000, sampler_seed=777),
            holdout=dict(start=70000, episodes=12, slots=1000, sampler_seed=20250713,
                         beta_rounded=beta,
                         by_depth={str(m): dict(member_count=1000, acks=900, first_ack=.9,
                                               episode_cluster_ci95=[.88, .92]) for m in range(1, 5)}))
        self.profile = self.root / 'profile.json'
        self.profile.write_text(json.dumps(document))
        self.runtime = dict(python='3.11.0rc1', system='Linux', machine='x86_64',
                            packages={'torch': '2.7.1+cu118'}, torch_cuda='11.8',
                            gpu_name='NVIDIA GeForce RTX 4090')
        self.run, self.manifest = self.create_run('base', '21_base')

    def tearDown(self):
        self.tmp.cleanup()

    def create_run(self, recipe, name, legacy=False):
        args = pv.parser().parse_args([
            '--recipe', recipe, '--name', name, '--seed', '2024', '--num-updates', '2000',
            '--lr-final', '0', '--lr-decay-updates', '2000', '--replay-mode', 'batched',
            '--traffic-model', 'bernoulli', '--calibration-profile', str(self.profile),
            '--calibration-reference-root', oi.FROZEN_REFERENCE])
        run, _, manifest = pv.plan(args, self.root)
        manifest['runtime'] = copy.deepcopy(self.runtime)
        if legacy:
            manifest['source_sha256'] = pt.source_hashes(self.frozen)
            manifest['config'].pop('traffic_model')
            manifest.pop('calibration_reference_root')
            manifest.pop('traffic_origin')
        (run / 'ckpt').mkdir(parents=True)
        (run / 'ckpt/best.pt').write_bytes(b'unchanged checkpoint fixture')
        (run / 'ckpt/latest.pt').write_bytes(b'unchanged latest fixture')
        status = self.root / 'review/logs' / f'{name}_launch.json'
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(dict(status='completed', returncode=0,
                                         target_completed=True, checkpoint_update=1999)))
        self.save(run, manifest)
        return run, manifest

    @staticmethod
    def save(run, manifest):
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        stamped = {**manifest['config'], **OODInputGuards.metadata(manifest)}
        (run / 'config.json').write_text(json.dumps(stamped))

    @staticmethod
    def metadata(manifest):
        stamp = 'sha256:' + hashlib.sha256(pt.canonical(manifest['source_sha256']).encode()).hexdigest()
        return {'git_hash': stamp, 'git_dirty_py': False}

    def prepare(self, run=None):
        return oi.validate_run(self.root, run or self.run)

    def checkpoint(self, manifest=None):
        import torch
        m = manifest or self.manifest
        return dict(cfg={**copy.deepcopy(m['config']), **self.metadata(m)}, model={'ret_std': torch.tensor(1.)},
                    update=649, lr_schedule=copy.deepcopy(m['schedule']))

    def test_current_run_preserved_and_read_only(self):
        before = {p: pt.sha_file(p) for p in self.root.rglob('*') if p.is_file()}
        p = self.prepare()
        self.assertEqual(p['training_config'], self.manifest['config'])
        self.assertEqual(p['config_provenance_metadata'], self.metadata(self.manifest))
        self.assertEqual(p['source_provenance']['mode'], 'current-exact')
        oi.assert_inputs_unchanged(self.root, p)
        self.assertEqual(before, {p: pt.sha_file(p) for p in self.root.rglob('*') if p.is_file()})

    def test_exact_legacy_bridge_preserves_original_checkpoint_configuration(self):
        run, m = self.create_run('base', '16_legacy', legacy=True)
        p = self.prepare(run)
        self.assertNotIn('traffic_model', p['raw_training_config'])
        self.assertEqual(p['training_config'], {**m['config'], 'traffic_model': 'bernoulli'})
        self.assertEqual(p['source_provenance']['mode'], 'reviewed-bernoulli-legacy-bridge')
        self.assertEqual(set(p['source_provenance']['reviewed_bridge']), set(oi.REVIEWED_BRIDGE))
        oi.validate_loaded_checkpoint(self.checkpoint(m), p)
        self.assertEqual(p['checkpoint_update'], 649)
        bad = self.checkpoint(m)
        bad['cfg']['traffic_model'] = 'bernoulli'
        with self.assertRaisesRegex(ValueError, 'exact saved'):
            oi.validate_loaded_checkpoint(bad, p)

    def test_legacy_requires_exact_reviewed_current_and_frozen_bytes(self):
        run, _ = self.create_run('base', '16_legacy', legacy=True)
        for path in (self.root / 'traffic.py', self.frozen / 'traffic.py',
                     self.root / 'policy.py', self.frozen / 'phy.py'):
            saved = path.read_bytes()
            path.write_bytes(saved + b'\n# changed\n')
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare(run)
            path.write_bytes(saved)

    def test_reviewed_bernoulli_bridge_preserves_arrivals_queues_and_rng(self):
        import numpy as np
        from config import Config
        from traffic import TrafficModel
        name = '_ood_test_frozen_traffic'
        spec = importlib.util.spec_from_file_location(name, self.frozen / 'traffic.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            for size in (1, 8):
                cfg = Config(num_ue=8, queue_size=size, p_arrival=.5,
                             packet_size_min=4000, packet_size_max=12000,
                             deadline_min=3, deadline_max=12)
                new, old = TrafficModel(cfg), module.TrafficModel(cfg)
                rnew, rold = np.random.default_rng(2024), np.random.default_rng(2024)
                for slot in range(150):
                    # Mix default/episode-specific arrival probabilities,
                    # inactive UEs, full queues, expiry and service removals.
                    for tm in (new, old):
                        tm.n_active = 8 if slot < 100 else 6
                        tm.p_arrival_ep = None if slot < 50 else .9
                    self.assertEqual(new.generate_arrivals(slot, rnew), old.generate_arrivals(slot, rold))
                    self.assertEqual(rnew.bit_generator.state, rold.bit_generator.state)
                    for tm in (new, old):
                        tm.decrement_deadlines()
                        for u in range(cfg.num_ue):
                            if slot % 7 == u:
                                tm.remove_packet(u)
                            while tm.packets[u] is not None and tm.packets[u].deadline <= 0:
                                tm.remove_packet(u)
                    self.assertEqual(new.pop_expired_queued(), old.pop_expired_queued())
                    self.assertEqual([[dataclasses.asdict(p) for p in q] for q in new.queues],
                                     [[dataclasses.asdict(p) for p in q] for q in old.queues])
        finally:
            sys.modules.pop(name, None)

    def test_legacy_defaulting_cannot_hide_any_other_missing_config_field(self):
        run, m = self.create_run('base', '16_legacy', legacy=True)
        m['config'].pop('deadline_max')
        self.save(run, m)
        with self.assertRaisesRegex(ValueError, 'sole traffic_model'):
            self.prepare(run)

    def test_legacy_cannot_change_recipe_or_explicit_traffic(self):
        run, m = self.create_run('base', '16_legacy', legacy=True)
        m['config']['traffic_model'] = 'bernoulli'
        self.save(run, m)
        with self.assertRaisesRegex(ValueError, 'predate'):
            self.prepare(run)

    def test_variant_requires_its_own_source_and_recipe_provenance(self):
        run, m = self.create_run('narrow_mean', '22_narrow')
        p = self.prepare(run)
        self.assertEqual(p['training_config']['ue_speed_max'], 22.5)
        self.assertIsNotNone(p['variant_provenance'])
        m.pop('variant_provenance')
        self.save(run, m)
        with self.assertRaisesRegex(ValueError, 'Variant provenance'):
            self.prepare(run)

    def test_variant_range_change_is_not_accepted_as_training_config(self):
        run, m = self.create_run('narrow_mean', '22_narrow')
        m['config']['ue_speed_min'] = 20.
        self.save(run, m)
        with self.assertRaisesRegex(ValueError, 'permitted recipe'):
            self.prepare(run)

    def test_current_source_drift_rejected(self):
        (self.root / 'policy.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            self.prepare()

    def test_checkpoint_cfg_lr_update_and_finite_model_are_strict(self):
        p = self.prepare()
        oi.validate_loaded_checkpoint(self.checkpoint(), p)
        self.assertEqual(p['checkpoint_update'], 649)
        for field, value in [('seed', 2024.), ('deadline_max', 11), ('traffic_model', 'ftp3')]:
            ck = self.checkpoint()
            ck['cfg'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                oi.validate_loaded_checkpoint(ck, p)
        for update in (True, 649., -1, 2000):
            ck = self.checkpoint()
            ck['update'] = update
            with self.subTest(update=update), self.assertRaises(ValueError):
                oi.validate_loaded_checkpoint(ck, p)
        ck = self.checkpoint()
        ck['lr_schedule']['lr_decay_updates'] = 1000
        with self.assertRaisesRegex(ValueError, 'schedule'):
            oi.validate_loaded_checkpoint(ck, p)
        ck = self.checkpoint()
        ck['model']['ret_std'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'finite tensor'):
            oi.validate_loaded_checkpoint(ck, p)

    def test_config_and_checkpoint_source_stamp_is_verified_without_defaulting(self):
        p = self.prepare()
        original = json.loads((self.run / 'config.json').read_text())
        for field, value in [('git_hash', 'sha256:' + '0' * 64), ('git_dirty_py', True),
                             ('git_dirty_py', 0), ('unknown_metadata', 'not allowed')]:
            bad = {**original, field: value}
            (self.run / 'config.json').write_text(json.dumps(bad))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
            ck = self.checkpoint()
            ck['cfg'][field] = value
            with self.assertRaises(ValueError):
                oi.validate_loaded_checkpoint(ck, p)
        for missing in ('git_hash', 'git_dirty_py', 'deadline_max'):
            bad = dict(original)
            bad.pop(missing)
            (self.run / 'config.json').write_text(json.dumps(bad))
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.prepare()
            ck = self.checkpoint()
            ck['cfg'].pop(missing)
            with self.assertRaises(ValueError):
                oi.validate_loaded_checkpoint(ck, p)
        (self.run / 'config.json').write_text(json.dumps(original))
        oi.validate_loaded_checkpoint(self.checkpoint(), p)

    def test_completion_record_requires_exact_completed_target(self):
        status = self.root / 'review/logs' / f'{self.run.name}_launch.json'
        original = json.loads(status.read_text())
        for field, value in [('status', 'running'), ('returncode', False),
                             ('target_completed', 1), ('checkpoint_update', 1998),
                             ('checkpoint_update', 2000), ('checkpoint_update', 1999.)]:
            status.write_text(json.dumps({**original, field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
        status.write_text(json.dumps(original))

    def test_config_and_beta_drift_rejected(self):
        for field, value in [('deadline_max', 11), ('la_beta_by_depth', [.5] * 4),
                             ('traffic_model', 'ftp3')]:
            m = copy.deepcopy(self.manifest)
            m['config'][field] = value
            self.save(self.run, m)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
        self.save(self.run, self.manifest)
        cfg = copy.deepcopy(self.manifest['config'])
        cfg['debug'] = True
        (self.run / 'config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'config.json differs'):
            self.prepare()

    def test_invalid_calibration_reference_or_missing_profile_rejected(self):
        for field, value in [('calibration_profile', None), ('calibration_reference_root', '../outside')]:
            m = copy.deepcopy(self.manifest)
            m[field] = value
            self.save(self.run, m)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()

    def test_source_runtime_is_exactly_matched(self):
        p = self.prepare()
        oi.validate_runtime(self.runtime, p)
        wrong = {**self.runtime, 'torch_cuda': '12.1'}
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            oi.validate_runtime(wrong, p)

    def test_changed_inputs_at_exit_and_symlink_checkpoint_rejected(self):
        p = self.prepare()
        ck = self.run / 'ckpt/best.pt'
        ck.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            oi.assert_inputs_unchanged(self.root, p)
        ck.unlink()
        ck.symlink_to(self.run / 'ckpt/latest.pt')
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
