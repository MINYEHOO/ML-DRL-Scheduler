"""CPU-only checkpoint dispatch and phase invariance; no training or channel simulation.

Weights below are synthetic fixtures, not Run 30/34/35 performance evidence.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_model_loader as loader
import paper_train as pt
import paper_train_raw as raw
import paper_train_rawpi as rawpi
import policy
from config import Config
from test_eval_actions import make_obs, reserve

RECIPES = ('base', 'raw_invariant', 'raw_phase_inv', 'raw_phase_inv_lr6')
TRAINERS = {'base': pt, 'raw_invariant': raw,
            'raw_phase_inv': rawpi, 'raw_phase_inv_lr6': rawpi}


def copy_sources(root):
    names = [*pt.SOURCES, 'paper_train_raw.py', 'paper_train_rawpi.py',
             'paper_model_loader.py', 'paper_run_eval.py']
    names += [f'artifacts/{recipe}/config.json' for recipe in RECIPES]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)


def manifest_for(root, recipe):
    trainer = TRAINERS[recipe]
    args = trainer.parser().parse_args([
        '--recipe', recipe, '--name', recipe, '--seed', '2024',
        '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
        '--traffic-model', 'bernoulli', '--replay-mode', 'batched',
        '--device', 'cpu', '--threads', '1'])
    return trainer.plan(args, root)[2]


def native_model(manifest):
    """Instantiate the frozen trainer architecture without calling install()."""
    recipe = manifest['recipe']
    enc, tail = policy.build_encoder_input, policy.build_value_tail
    if recipe == 'raw_invariant':
        cls, tail = raw._build()
    elif recipe in rawpi.RECIPES:
        cls, enc, tail = rawpi._build()
    else:
        cls = policy.ActorCritic
    with torch.random.fork_rng():
        torch.manual_seed(567)
        model = cls(Config(**manifest['config'])).eval().requires_grad_(False)
    model.ret_mean.fill_(2.75)
    model.ret_std.fill_(1.125)
    model.ret_count.fill_(43.)
    return model, enc, tail


class ModelLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        copy_sources(cls.root)
        cls.manifests = {r: manifest_for(cls.root, r) for r in RECIPES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.stock = (policy.ActorCritic, policy.build_encoder_input, policy.build_value_tail)
        self.recipes_before = copy.deepcopy(pt.RECIPES)

    def tearDown(self):
        self.assertEqual(self.stock, (policy.ActorCritic, policy.build_encoder_input,
                                     policy.build_value_tail))
        self.assertFalse(raw._INSTALLED)
        self.assertFalse(rawpi._INSTALLED)
        self.assertEqual(pt.RECIPES, self.recipes_before)

    def fixture(self, recipe):
        manifest = copy.deepcopy(self.manifests[recipe])
        native, enc, tail = native_model(manifest)
        ck = dict(model=copy.deepcopy(native.state_dict()), cfg=copy.deepcopy(manifest['config']),
                  update=999, lr_schedule=copy.deepcopy(manifest['schedule']))
        return manifest, ck, native, enc, tail

    def load(self, manifest, ck, **kwargs):
        return loader.load_model(manifest, ck, root=self.root, **kwargs)

    def test_all_variants_match_native_actions_values_and_state(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                manifest, ck, native, enc, tail = self.fixture(recipe)
                model = self.load(manifest, ck)
                self.assertEqual(set(model.state_dict()), set(ck['model']))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, ck['model'][name], rtol=0, atol=0)
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                obs = make_obs(model.cfg)
                reserve(obs, 0, 1, 0)
                obs['cqi_fb'][1] = 0.
                with mock.patch.object(policy, 'build_encoder_input', enc), \
                        mock.patch.object(policy, 'build_value_tail', tail):
                    expected_action = native.deterministic_action(obs)
                    expected_value = native.state_value(obs)
                np.testing.assert_array_equal(model.deterministic_action(obs), expected_action)
                self.assertEqual(model.state_value(obs), expected_value)
                np.testing.assert_array_equal(model.decode(obs, deterministic=True)['action_sequence'],
                                              expected_action)

    def test_mixed_models_are_independent_and_do_not_mutate_rng_or_buffers(self):
        models = {}
        for recipe in RECIPES:
            manifest, ck, *_ = self.fixture(recipe)
            models[recipe] = self.load(manifest, ck)
        observations = {r: make_obs(model.cfg, seed=772) for r, model in models.items()}
        expected = {r: (m.deterministic_action(observations[r]), m.state_value(observations[r]))
                    for r, m in models.items()}
        states = {r: copy.deepcopy(m.state_dict()) for r, m in models.items()}
        rng = torch.get_rng_state().clone()
        for recipe in ('raw_phase_inv', 'base', 'raw_invariant', 'raw_phase_inv_lr6', 'base'):
            m, obs = models[recipe], observations[recipe]
            np.testing.assert_array_equal(m.deterministic_action(obs), expected[recipe][0])
            self.assertEqual(m.state_value(obs), expected[recipe][1])
            self.assertEqual(self.stock, (policy.ActorCritic, policy.build_encoder_input,
                                         policy.build_value_tail))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for recipe, model in models.items():
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, states[recipe][name], rtol=0, atol=0)

    def test_phase_rotation_preserves_features_logits_values_and_actions(self):
        # Each UE/RBG can have its own global antenna phase. The first eight
        # deterministic synthetic observations avoid interpreting argmax ties
        # or floating-point noise as trained-policy performance measurements.
        for recipe in ('raw_phase_inv', 'raw_phase_inv_lr6'):
            manifest, ck, *_ = self.fixture(recipe)
            model = self.load(manifest, ck)
            for seed in range(8):
                with self.subTest(recipe=recipe, seed=seed):
                    obs = make_obs(model.cfg, seed=seed)
                    rotated = copy.deepcopy(obs)
                    rng = np.random.default_rng(100 + seed)
                    phase = rng.uniform(-np.pi, np.pi, obs['direction_fb'].shape[:2])
                    rotated['direction_fb'] *= np.exp(1j * phase[..., None])
                    inputs = [policy.obs_to_tensors(x, torch.device('cpu')) for x in (obs, rotated)]
                    features = [model.encode(x) for x in inputs]
                    torch.testing.assert_close(features[0], features[1], rtol=2e-5, atol=2e-6)
                    logits = []
                    for x, e in zip(inputs, features):
                        logits.append(model._position_logits_and_mask(
                            x, e, 0, 0, set(), np.zeros(model.cfg.num_ue),
                            x['uncommitted'].clone())[0])
                    torch.testing.assert_close(logits[0], logits[1], rtol=2e-5, atol=2e-6)
                    self.assertAlmostEqual(model.state_value(obs), model.state_value(rotated), places=5)
                    np.testing.assert_array_equal(model.deterministic_action(obs),
                                                  model.deterministic_action(rotated))

    def test_inference_exception_restores_features_for_next_model(self):
        manifest, ck, *_ = self.fixture('raw_phase_inv')
        model = self.load(manifest, ck)
        with mock.patch.object(model.encoder, 'forward', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                model.deterministic_action(make_obs(model.cfg))
        self.assertEqual(self.stock, (policy.ActorCritic, policy.build_encoder_input,
                                     policy.build_value_tail))
        manifest, ck, native, *_ = self.fixture('base')
        standard = self.load(manifest, ck)
        obs = make_obs(standard.cfg)
        np.testing.assert_array_equal(standard.deterministic_action(obs), native.deterministic_action(obs))

    def test_wrong_architecture_missing_extra_or_wrong_shape_is_rejected(self):
        manifest, ck, *_ = self.fixture('raw_phase_inv')
        _, stock, *_ = self.fixture('base')
        bad = [copy.deepcopy(ck) for _ in range(4)]
        bad[0]['model'] = stock['model']
        first = next(iter(ck['model']))
        bad[1]['model'].pop(first)
        bad[2]['model']['invented.weight'] = torch.ones(1)
        weight = next(k for k, v in ck['model'].items() if v.ndim == 2)
        bad[3]['model'][weight] = ck['model'][weight][:-1]
        for index, broken in enumerate(bad):
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                self.load(manifest, broken)

    def test_checkpoint_config_and_nonfinite_parameters_are_rejected(self):
        manifest, ck, *_ = self.fixture('raw_phase_inv_lr6')
        for key, value in [('ppo_learning_rate', 3e-4), ('num_ue', 16), ('seed', 2024.0)]:
            bad = copy.deepcopy(ck)
            bad['cfg'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(manifest, bad)
        for value in (float('nan'), float('inf')):
            bad = copy.deepcopy(ck)
            bad['model']['ret_std'].fill_(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(manifest, bad)

    def test_policy_user_count_override_keeps_checkpoint_training_identity(self):
        manifest, ck, *_ = self.fixture('raw_phase_inv')
        cfg = copy.deepcopy(manifest['config'])
        cfg['num_ue'] = 8
        model = self.load(manifest, ck, policy_config=cfg)
        self.assertEqual(model.cfg.num_ue, 8)
        self.assertEqual(model.deterministic_action(make_obs(model.cfg)).shape,
                         (model.cfg.num_rbg, model.cfg.l_max))


if __name__ == '__main__':
    unittest.main()
