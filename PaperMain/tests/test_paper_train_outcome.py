"""Protected reward_outcome recipe: planning/provenance guards (no GPU) and the
reward identity of the outcome-only env wrapper (CPU, short episode)."""
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
import paper_train_outcome as po


class OutcomeRecipeGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*pt.SOURCES, 'paper_train_outcome.py',
                     'artifacts/base/config.json', 'artifacts/reward_outcome/config.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.base_file = self.root / 'artifacts/base/config.json'
        self.variant_file = self.root / 'artifacts/reward_outcome/config.json'
        self.source_hashes = pt.source_hashes(self.root)
        self.recipes = copy.deepcopy(pt.RECIPES)

    def tearDown(self):
        self.assertEqual(pt.RECIPES, self.recipes)
        self.tmp.cleanup()

    def args(self, *argv):
        return po.parser().parse_args(list(argv))

    def fresh(self, **overrides):
        argv = ['--recipe', 'reward_outcome', '--name', 'oc', '--seed', '2024',
                '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                '--replay-mode', 'batched', '--traffic-model', 'bernoulli']
        args = self.args(*argv)
        for key, value in overrides.items():
            setattr(args, key, value)
        return po.plan(args, self.root)

    def save(self, **overrides):
        run, _, manifest = self.fresh(**overrides)
        checkpoint = run / 'ckpt/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('payload checked by integration smoke')
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        return run, checkpoint, manifest

    def resume(self, checkpoint, *argv):
        return po.plan(self.args('--recipe', 'reward_outcome', '--resume', str(checkpoint), *argv), self.root)

    def test_differs_from_base_only_in_lambda_s_and_lambda_m(self):
        _, _, variant = self.fresh()
        args = self.args('--recipe', 'base', '--name', 'base', '--seed', '2024',
                         '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                         '--replay-mode', 'batched', '--traffic-model', 'bernoulli')
        _, _, base = pt.plan(args, self.root)
        differences = {k for k in base['config'] if base['config'][k] != variant['config'][k]}
        self.assertEqual(differences, {'lambda_s', 'lambda_m'})
        self.assertEqual(variant['config']['lambda_s'], 0.0)
        self.assertEqual(variant['config']['lambda_m'], 1.0)
        self.assertEqual(variant['config']['lambda_c'], 1.0)
        self.assertEqual(variant['config']['eta_d'], base['config']['eta_d'])
        self.assertEqual(variant['variant_provenance']['reward'], po.REWARD)
        self.assertEqual(variant['source_sha256'], self.source_hashes)
        self.assertFalse((self.root / 'runs').exists())

    def test_default_horizon_1000_linear_decay(self):
        _, _, m = po.plan(self.args('--recipe', 'reward_outcome', '--name', 'oc', '--seed', '2024',
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
        with (self.root / 'paper_train_outcome.py').open('a') as f:
            f.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_extra_change_rejected(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['lambda_c'] = 2.0
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'only in lambda_s and lambda_m'):
            self.fresh()

    def test_ftp3_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Bernoulli'):
            self.fresh(traffic_model='ftp3')


class OutcomeEnv(unittest.TestCase):
    """reward == lambda_c*completed_bits/b_norm - lambda_m*n_fail - LAMBDA_W*wasted/b_norm
    every slot, and the episode counters agree with the per-slot info."""

    @classmethod
    def setUpClass(cls):
        cls.Env = po.install()
        import train_phase2 as T
        import baselines as B
        from config import Config
        assert T.SchedulerEnv is cls.Env
        raw = json.loads((ROOT / 'artifacts/reward_outcome/config.json').read_text())
        raw = {k: v for k, v in raw.items() if k not in ('git_hash', 'git_dirty_py')}
        for k in ('la_beta_by_depth', 'ue_speed_mix'):
            raw[k] = tuple(raw.get(k) or ())
        raw.update(episode_len_main=120, episode_len_debug=120)
        cls.cfg = Config(**raw)
        cls.sched = [s for s in B.all_baselines(cls.cfg) if s.name == 'SUS+CQI'][0]

    @classmethod
    def tearDownClass(cls):
        # the patch is process-global by design; undo it so other recipe test
        # modules in the same unittest process see the stock env again
        po.uninstall()

    def test_lambda_s_must_be_zero(self):
        from config import Config
        import dataclasses
        bad = Config(**{**dataclasses.asdict(self.cfg), 'lambda_s': 1.0})
        with self.assertRaisesRegex(ValueError, 'lambda_s = 0'):
            self.Env(bad)

    def test_per_slot_identity_and_episode_sums(self):
        env = self.Env(self.cfg)
        env.reset(90006)
        tot = comp = waste = 0.0
        done = False
        while not done:
            _, r, done, info = env.step(self.sched.schedule(env))
            expect = (self.cfg.lambda_c * info['completed_bits'] / self.cfg.b_norm
                      - self.cfg.lambda_m * info['n_fail']
                      - po.LAMBDA_W * info['wasted_bits'] / self.cfg.b_norm)
            self.assertAlmostEqual(r, expect, places=9)
            self.assertGreaterEqual(info['wasted_bits'], 0.0)
            tot += r; comp += info['completed_bits']; waste += info['wasted_bits']
        self.assertAlmostEqual(tot, env.ep['reward'], places=6)
        self.assertAlmostEqual(comp, env.ep['completed_bits'], places=6)
        self.assertAlmostEqual(waste, env.ep['wasted_bits'], places=6)
        self.assertLessEqual(waste, env.ep['acked_bits'] - env.ep['completed_bits'] + 1e-6)

    def test_validation_baselines_extended(self):
        import baselines as B
        names = [s.name for s in B.all_baselines(self.cfg)]
        self.assertEqual(names[-2:], ['SUS+CQI-Feasible', 'SUS-RPS'])


if __name__ == '__main__':
    unittest.main()
