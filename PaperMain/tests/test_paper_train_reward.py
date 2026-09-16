"""Protected reward_c2m6 recipe: planning and provenance guards (no GPU)."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_train as pt
import paper_train_reward as pw


class RewardRecipeGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*pt.SOURCES, 'paper_train_reward.py',
                     'artifacts/base/config.json', 'artifacts/reward_c2m6/config.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.base_file = self.root / 'artifacts/base/config.json'
        self.variant_file = self.root / 'artifacts/reward_c2m6/config.json'
        self.source_hashes = pt.source_hashes(self.root)
        self.recipes = copy.deepcopy(pt.RECIPES)

    def tearDown(self):
        self.assertEqual(pt.RECIPES, self.recipes)
        self.tmp.cleanup()

    def args(self, *argv):
        return pw.parser().parse_args(list(argv))

    def fresh(self, **overrides):
        argv = ['--recipe', 'reward_c2m6', '--name', 'rw', '--seed', '2024',
                '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                '--replay-mode', 'batched', '--traffic-model', 'bernoulli']
        args = self.args(*argv)
        for key, value in overrides.items():
            setattr(args, key, value)
        return pw.plan(args, self.root)

    def save(self, **overrides):
        run, _, manifest = self.fresh(**overrides)
        checkpoint = run / 'ckpt/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('payload checked by integration smoke')
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        return run, checkpoint, manifest

    def resume(self, checkpoint, *argv):
        return pw.plan(self.args('--recipe', 'reward_c2m6', '--resume', str(checkpoint), *argv), self.root)

    def test_differs_from_base_only_in_the_two_weights(self):
        _, _, variant = self.fresh()
        args = self.args('--recipe', 'base', '--name', 'base', '--seed', '2024',
                         '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                         '--replay-mode', 'batched', '--traffic-model', 'bernoulli')
        _, _, base = pt.plan(args, self.root)
        differences = {k for k in base['config'] if base['config'][k] != variant['config'][k]}
        self.assertEqual(differences, {'lambda_c', 'lambda_m'})
        self.assertEqual(variant['config']['lambda_c'], 2.0)
        self.assertEqual(variant['config']['lambda_m'], 6.0)
        self.assertEqual(variant['config']['lambda_s'], base['config']['lambda_s'])
        self.assertEqual(base['schedule'], variant['schedule'])
        self.assertEqual(variant['source_sha256'], self.source_hashes)
        self.assertEqual(pt.source_hashes(self.root), self.source_hashes)
        self.assertEqual(variant['variant_provenance']['changed_ranges'],
                         {'lambda_c': {'base': 1.0, 'variant': 2.0}, 'lambda_m': {'base': 2.0, 'variant': 6.0}})
        self.assertFalse((self.root / 'runs').exists())

    def test_default_horizon_1000_linear_decay(self):
        _, _, m = pw.plan(self.args('--recipe', 'reward_c2m6', '--name', 'rw', '--seed', '2024',
                                    '--replay-mode', 'batched'), self.root)
        self.assertEqual(m['target_updates'], 1000)
        self.assertEqual(m['schedule'], {'lr_final': 0.0, 'lr_decay_updates': 1000})

    def test_resume_restores_config_and_provenance(self):
        run, checkpoint, saved = self.save(smoke_slots=32, num_updates=1)
        _, _, restored = self.resume(checkpoint, '--num-updates', '2')
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['variant_provenance'], saved['variant_provenance'])
        self.assertEqual(restored['target_updates'], 2)

    def test_module_changed_rejects_resume(self):
        _, checkpoint, _ = self.save()
        with (self.root / 'paper_train_reward.py').open('a') as f:
            f.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_core_source_changed_rejects_resume(self):
        _, checkpoint, _ = self.save()
        (self.root / 'env.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'training source changed'):
            self.resume(checkpoint)

    def test_extra_change_rejected(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['lambda_s'] = 0.5
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'only in lambda_c and lambda_m'):
            self.fresh()
        self.assertFalse((self.root / 'runs').exists())

    def test_wrong_weights_rejected(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['lambda_m'] = 4.0
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'only in lambda_c and lambda_m'):
            self.fresh()

    def test_changed_base_weights_rejected(self):
        base = json.loads(self.base_file.read_text())
        base['lambda_m'] = 3.0
        self.base_file.write_text(json.dumps(base))
        with self.assertRaisesRegex(ValueError, 'Base reward weights differ'):
            self.fresh()

    def test_ftp3_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Bernoulli'):
            self.fresh(traffic_model='ftp3')


if __name__ == '__main__':
    unittest.main()
