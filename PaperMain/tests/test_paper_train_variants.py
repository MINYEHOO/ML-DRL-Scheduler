"""Protected variant planning and provenance tests; no GPU/channel generation."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_train as pt
import paper_train_variants as pv


class VariantGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*pt.SOURCES, 'paper_train_variants.py',
                     'artifacts/base/config.json', 'artifacts/narrow_mean/config.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.base_file = self.root / 'artifacts/base/config.json'
        self.variant_file = self.root / 'artifacts/narrow_mean/config.json'
        self.source_hashes = pt.source_hashes(self.root)
        self.recipes = copy.deepcopy(pt.RECIPES)

    def tearDown(self):
        self.assertEqual(pt.RECIPES, self.recipes)
        self.tmp.cleanup()

    def args(self, *argv):
        return pv.parser().parse_args(list(argv))

    def fresh(self, **overrides):
        argv = ['--recipe', 'narrow_mean', '--name', 'variant', '--seed', '2024',
                '--num-updates', '2000', '--lr-final', '0', '--lr-decay-updates', '2000',
                '--replay-mode', 'batched', '--traffic-model', 'bernoulli']
        args = self.args(*argv)
        for key, value in overrides.items():
            setattr(args, key, value)
        return pv.plan(args, self.root)

    def save(self, **overrides):
        run, _, manifest = self.fresh(**overrides)
        checkpoint = run / 'ckpt/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('payload checked by integration smoke')
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        return run, checkpoint, manifest

    def resume(self, checkpoint, *argv):
        return pv.plan(self.args('--recipe', 'narrow_mean', '--resume', str(checkpoint), *argv), self.root)

    def profile(self):
        from calibration import profile as cp
        for name in (*cp.SCIENCE_SOURCES, 'calibration/cqi4.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        document = dict(
            schema_version=cp.SCHEMA_VERSION, protocol=cp.PROTOCOL, status='validated',
            reference_config_sha256=cp.sha256_file(self.base_file),
            config=cp.normalized_config(json.loads(self.base_file.read_text())),
            source_sha256=cp.source_hashes(self.root), beta_rounded=[1.0, .8, .7, .6],
            membership_verified_calibration=True, membership_verified_holdout=True,
            calibration=dict(start=50000, episodes=36, slots=1000, sampler_seed=777),
            holdout=dict(start=70000, episodes=12, slots=1000, sampler_seed=20250713,
                         beta_rounded=[1.0, .8, .7, .6],
                         by_depth={str(m): dict(member_count=1000, acks=900, first_ack=.9,
                                               episode_cluster_ci95=[.88, .92]) for m in range(1, 5)}))
        path = self.root / 'profile.json'
        path.write_text(json.dumps(document))
        return path

    def test_fresh_pair_differs_only_in_six_ranges_and_keeps_core_hashes(self):
        profile = self.profile()
        _, _, variant = self.fresh(calibration_profile=profile)
        args = self.args('--recipe', 'base', '--name', 'base', '--seed', '2024',
                         '--num-updates', '2000', '--lr-final', '0', '--lr-decay-updates', '2000',
                         '--replay-mode', 'batched', '--traffic-model', 'bernoulli',
                         '--calibration-profile', str(profile))
        _, _, base = pt.plan(args, self.root)
        differences = {key for key in base['config'] if base['config'][key] != variant['config'][key]}
        self.assertEqual(differences, set(pv.MEAN_RANGES))
        for key, value in pv.MEAN_RANGES.items():
            self.assertEqual(variant['config'][key], value)
        self.assertEqual(base['schedule'], variant['schedule'])
        self.assertEqual(variant['source_sha256'], self.source_hashes)
        self.assertEqual(pt.source_hashes(self.root), self.source_hashes)
        self.assertEqual(variant['config']['la_beta_by_depth'], [1.0, .8, .7, .6])
        self.assertIn('no NARROW-specific holdout validation', variant['calibration_application'])
        self.assertFalse((self.root / 'runs').exists())

    def test_resume_restores_smoke_config_lr_and_provenance(self):
        run, checkpoint, saved = self.save(smoke_slots=32, num_updates=1)
        _, _, restored = self.resume(checkpoint, '--num-updates', '2')
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['schedule'], saved['schedule'])
        self.assertEqual(restored['variant_provenance'], saved['variant_provenance'])
        self.assertEqual(restored['target_updates'], 2)
        self.assertEqual(json.loads((run / 'paper_manifest.json').read_text()), saved)

    def test_variant_wrapper_changed_rejects_resume(self):
        _, checkpoint, _ = self.save()
        with (self.root / 'paper_train_variants.py').open('a') as f:
            f.write('\n# changed implementation\n')
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_core_source_changed_still_rejects_resume(self):
        _, checkpoint, _ = self.save()
        (self.root / 'traffic.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'training source changed'):
            self.resume(checkpoint)

    def test_missing_variant_provenance_rejects_resume(self):
        run, checkpoint, saved = self.save()
        del saved['variant_provenance']
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_extra_recipe_change_rejected_before_run_writes(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['queue_size'] = 9
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'only in the six'):
            self.fresh()
        self.assertFalse((self.root / 'runs').exists())

    def test_wrong_mean_rejected(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['p_arrival_min'] = recipe['p_arrival_max'] = .30
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'only in the six'):
            self.fresh()

    def test_changed_base_ranges_rejected(self):
        base = json.loads(self.base_file.read_text())
        base['ue_speed_max'] = 30.0
        self.base_file.write_text(json.dumps(base))
        with self.assertRaisesRegex(ValueError, 'Base ranges differ'):
            self.fresh()

    def test_base_recipe_hash_tracked_on_resume_even_whitespace(self):
        _, checkpoint, _ = self.save()
        self.base_file.write_text(self.base_file.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_disallowed_runtime_change_rejected_even_consistent_saved_configs(self):
        run, checkpoint, saved = self.save()
        saved['config']['queue_size'] += 1
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        (run / 'config.json').write_text(json.dumps(saved['config']))
        with self.assertRaisesRegex(ValueError, 'differs beyond the permitted'):
            self.resume(checkpoint)

    def test_ftp3_override_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original Bernoulli'):
            self.fresh(traffic_model='ftp3')

    def test_resume_lr_changes_rejected_by_original_guard(self):
        _, checkpoint, _ = self.save()
        with self.assertRaisesRegex(ValueError, 'LR schedule cannot change'):
            self.resume(checkpoint, '--lr-final', '0', '--lr-decay-updates', '1000')

    def test_registration_restores_original_module_state_on_exception(self):
        original = pt.RECIPES
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with pv.registered_recipes():
                self.assertIn('narrow_mean', pt.RECIPES)
                self.assertEqual(pt.source_hashes(self.root), self.source_hashes)
                raise RuntimeError('sentinel')
        self.assertIs(pt.RECIPES, original)

    def test_main_uses_original_trainer_and_restores_hooks_on_error(self):
        original_plan, original_recipes = pt.plan, pt.RECIPES
        def driver(argv):
            self.assertIs(pt.plan, pv.plan)
            self.assertIn('narrow_mean', pt.RECIPES)
            self.assertEqual(argv, ['--recipe', 'narrow_mean'])
            raise RuntimeError('driver sentinel')
        with patch.object(pt, 'main', side_effect=driver):
            with self.assertRaisesRegex(RuntimeError, 'driver sentinel'):
                pv.main(['--recipe', 'narrow_mean'])
        self.assertIs(pt.plan, original_plan)
        self.assertIs(pt.RECIPES, original_recipes)

    def test_standard_base_passes_through_without_extension_metadata(self):
        args = self.args('--recipe', 'base', '--name', 'base')
        core = pt.plan(args, self.root)
        wrapped = pv.plan(args, self.root)
        self.assertEqual(core, wrapped)
        self.assertNotIn('variant_provenance', wrapped[2])


if __name__ == '__main__':
    unittest.main()
