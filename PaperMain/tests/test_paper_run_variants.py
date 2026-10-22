"""Variant dispatch retains the live-run completion and provenance gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_model_loader as loader
import paper_run_eval as ev
import paper_train as pt
from test_paper_model_loader import RECIPES, TRAINERS, copy_sources, manifest_for


class LiveVariantGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_sources(self.root)
        self.recipes_before = copy.deepcopy(pt.RECIPES)
        self.manifests = {}
        for recipe in RECIPES:
            manifest = manifest_for(self.root, recipe)
            self.manifests[recipe] = manifest
            run = self.root / 'runs' / recipe
            (run / 'ckpt').mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps(manifest['config']))
            (run / 'ckpt/best.pt').write_bytes(b'tensor loading tested separately')
            (run / 'ckpt/latest.pt').write_bytes(b'tensor loading tested separately')
            self.save(recipe)
            status = self.root / 'review/logs' / f'{recipe}_launch.json'
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps(dict(status='completed', returncode=0,
                                             target_completed=True, checkpoint_update=999)))

    def tearDown(self):
        self.assertEqual(pt.RECIPES, self.recipes_before)
        self.tmp.cleanup()

    def save(self, recipe):
        path = self.root / 'runs' / recipe / 'paper_manifest.json'
        path.write_text(json.dumps(self.manifests[recipe]))

    def prepared(self, recipe, *extra):
        args = ev.parser().parse_args([
            '--run', f'runs/{recipe}', '--episode-start', '140000', '--episodes', '4',
            '--out', f'results/{recipe}', '--device', 'cpu', *extra])
        return ev.plan(args, self.root)

    def test_core_raw_and_both_phase_recipes_without_global_registration(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                p = self.prepared(recipe)
                self.assertEqual(p['protocol']['training_config'], self.manifests[recipe]['config'])
                self.assertEqual(p['protocol']['environment_config'], self.manifests[recipe]['config'])
                self.assertEqual(p['protocol']['policy_config'], self.manifests[recipe]['config'])
                self.assertFalse(p['out'].exists())
                sources = loader.evaluation_sources(self.root, self.manifests[recipe])
                self.assertIn('paper_model_loader.py', sources)
                if recipe != 'base':
                    module = 'paper_train_raw.py' if recipe == 'raw_invariant' else 'paper_train_rawpi.py'
                    self.assertEqual(sources[module], pt.sha_file(self.root / module))

    def test_unknown_recipe_and_variant_disguised_as_stock_rejected(self):
        for recipe_name in ('unknown', 'base'):
            self.manifests['raw_phase_inv']['recipe'] = recipe_name
            self.save('raw_phase_inv')
            with self.subTest(recipe=recipe_name), self.assertRaises(ValueError):
                self.prepared('raw_phase_inv')

    def test_missing_or_modified_architecture_provenance_rejected(self):
        original = copy.deepcopy(self.manifests['raw_phase_inv'])
        for corruption in ('missing', 'architecture', 'hash'):
            manifest = copy.deepcopy(original)
            if corruption == 'missing':
                manifest.pop('variant_provenance')
            elif corruption == 'architecture':
                manifest['variant_provenance']['architecture']['encoder_input'] = 73
            else:
                manifest['variant_provenance']['architecture_sha256'] = '0' * 64
            self.manifests['raw_phase_inv'] = manifest
            self.save('raw_phase_inv')
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.prepared('raw_phase_inv')

    def test_variant_module_drift_is_rejected(self):
        path = self.root / 'paper_train_rawpi.py'
        path.write_text(path.read_text() + '\n# synthetic drift\n')
        with self.assertRaises(ValueError):
            self.prepared('raw_phase_inv')

    def test_variant_config_cannot_hide_environment_changes(self):
        manifest = self.manifests['raw_phase_inv']
        manifest['config']['ue_speed_max'] = 60.
        self.save('raw_phase_inv')
        (self.root / 'runs/raw_phase_inv/config.json').write_text(json.dumps(manifest['config']))
        with self.assertRaises(ValueError):
            self.prepared('raw_phase_inv')

    def test_frozen_recipe_hash_and_lr6_learning_rate_remain_enforced(self):
        path = self.root / 'artifacts/raw_phase_inv_lr6/config.json'
        document = json.loads(path.read_text())
        document['ppo_learning_rate'] = 3e-4
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            self.prepared('raw_phase_inv_lr6')

    def test_running_variant_cannot_be_evaluated_even_in_smoke_mode(self):
        path = self.root / 'review/logs/raw_phase_inv_launch.json'
        path.write_text(json.dumps(dict(status='running', target_completed=False)))
        for extra in ((), ('--smoke-slots', '40')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.prepared('raw_phase_inv', *extra)

    def test_variant_checkpoint_schedule_and_config_mismatch_rejected(self):
        manifest = self.manifests['raw_phase_inv_lr6']
        checkpoint = dict(cfg=copy.deepcopy(manifest['config']), model={}, update=999,
                          lr_schedule=copy.deepcopy(manifest['schedule']))
        ev.validate_checkpoint(checkpoint, self.prepared('raw_phase_inv_lr6'))
        for corruption in ('cfg', 'schedule', 'update'):
            bad = copy.deepcopy(checkpoint)
            if corruption == 'cfg':
                bad['cfg']['ppo_learning_rate'] = 3e-4
            elif corruption == 'schedule':
                bad['lr_schedule']['lr_decay_updates'] = 2000
            else:
                bad['update'] = 1000
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                ev.validate_checkpoint(bad, self.prepared('raw_phase_inv_lr6'))

    def test_episode_separation_and_smoke_config_preserved(self):
        for start in (0, 999, 10000):
            with self.subTest(start=start), self.assertRaises(ValueError):
                self.prepared('raw_phase_inv', '--episode-start', str(start))
        p = self.prepared('raw_phase_inv', '--smoke-slots', '40')
        self.assertTrue(p['protocol']['diagnostic_only'])
        self.assertEqual(p['protocol']['environment_config']['episode_len_main'], 40)
        self.assertEqual(p['protocol']['policy_config']['episode_len_main'], 1000)

    def test_archived_variant_requires_original_manifest_and_keeps_training_config(self):
        import paper_eval as archived
        with mock.patch.object(archived, 'ROOT', self.root):
            for recipe in RECIPES[1:]:
                with self.subTest(recipe=recipe):
                    with self.assertRaisesRegex(ValueError, 'paper_manifest.json'):
                        archived.archive_manifest(recipe)
                    path = self.root / 'artifacts' / recipe / 'paper_manifest.json'
                    path.write_text(json.dumps(self.manifests[recipe]))
                    actual = archived.archive_manifest(recipe)
                    self.assertEqual(actual, self.manifests[recipe])
                    env, cfg = archived.make_configs(recipe, 'K8', smoke_slots=40)
                    self.assertEqual(cfg.num_ue, 8)
                    self.assertEqual(cfg.ppo_learning_rate,
                                     self.manifests[recipe]['config']['ppo_learning_rate'])
                    self.assertEqual(cfg.episode_len_main, 1000)
                    self.assertEqual(env.episode_len_main, 40)
                    corrupted = copy.deepcopy(actual)
                    corrupted['variant_provenance']['architecture_sha256'] = '0' * 64
                    path.write_text(json.dumps(corrupted))
                    with self.assertRaises(ValueError):
                        archived.archive_manifest(recipe)

    def test_legacy_archived_base_does_not_require_new_manifest(self):
        import paper_eval as archived
        with mock.patch.object(archived, 'ROOT', self.root):
            manifest = archived.archive_manifest('base')
            self.assertEqual(manifest['recipe'], 'base')
            self.assertEqual(manifest['config']['seed'], 2024)


class OODVariantGuards(unittest.TestCase):
    """Reuse the existing reviewed-calibration fixture, adding actual variants."""
    def setUp(self):
        import test_paper_ood_inputs as existing
        self.fixture = existing.OODInputGuards()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        copy_sources(self.root)

    def create(self, recipe):
        import paper_ood_inputs as oi
        trainer = TRAINERS[recipe]
        args = trainer.parser().parse_args([
            '--recipe', recipe, '--name', recipe, '--seed', '2024',
            '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
            '--traffic-model', 'bernoulli', '--replay-mode', 'batched',
            '--calibration-profile', str(self.fixture.profile),
            '--calibration-reference-root', oi.FROZEN_REFERENCE])
        run, _, manifest = trainer.plan(args, self.root)
        manifest['runtime'] = copy.deepcopy(self.fixture.runtime)
        (run / 'ckpt').mkdir(parents=True)
        (run / 'ckpt/best.pt').write_bytes(b'fixture; tensors tested separately')
        self.fixture.save(run, manifest)
        status = self.root / 'review/logs' / f'{recipe}_launch.json'
        status.write_text(json.dumps(dict(status='completed', returncode=0,
                                         target_completed=True, checkpoint_update=999)))
        return run, manifest

    def test_raw_and_phase_variants_keep_ood_checkpoint_and_source_guards(self):
        import paper_ood_inputs as oi
        for recipe in RECIPES[1:]:
            with self.subTest(recipe=recipe):
                run, manifest = self.create(recipe)
                prepared = oi.validate_run(self.root, run)
                self.assertEqual(prepared['training_config'], manifest['config'])
                self.assertEqual(prepared['variant_provenance'], manifest['variant_provenance'])
                module = 'paper_train_raw.py' if recipe == 'raw_invariant' else 'paper_train_rawpi.py'
                self.assertIn(module, prepared['input_hashes'])
                # The new loader is evaluator code; OOD records it with its
                # runners rather than as a training input.
                import paper_ood_eval as ood
                self.assertIn('paper_model_loader.py', ood.RUNNER_FILES)
                ck = self.fixture.checkpoint(manifest)
                oi.validate_loaded_checkpoint(ck, prepared)
                ck['cfg']['seed'] = 3024
                with self.assertRaises(ValueError):
                    oi.validate_loaded_checkpoint(ck, prepared)
                oi.assert_inputs_unchanged(self.root, prepared)
                source = self.root / module
                original = source.read_bytes()
                source.write_bytes(original + b'\n# unexpected drift\n')
                with self.assertRaises(ValueError):
                    oi.assert_inputs_unchanged(self.root, prepared)
                source.write_bytes(original)

    def test_ood_variant_rejects_incomplete_source(self):
        import paper_ood_inputs as oi
        run, _ = self.create('raw_phase_inv')
        status = self.root / 'review/logs/raw_phase_inv_launch.json'
        status.write_text(json.dumps(dict(status='running', target_completed=False)))
        with self.assertRaises(ValueError):
            oi.validate_run(self.root, run)


if __name__ == '__main__':
    unittest.main()
