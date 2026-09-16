"""Protected raw_invariant recipe: planning/provenance guards (no GPU) and the
model-level invariants the ablation rests on (CPU, a few slots)."""
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
import paper_train_raw as pr


class RawRecipeGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in (*pt.SOURCES, 'paper_train_raw.py',
                     'artifacts/base/config.json', 'artifacts/raw_invariant/config.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.base_file = self.root / 'artifacts/base/config.json'
        self.variant_file = self.root / 'artifacts/raw_invariant/config.json'
        self.source_hashes = pt.source_hashes(self.root)
        self.recipes = copy.deepcopy(pt.RECIPES)

    def tearDown(self):
        self.assertEqual(pt.RECIPES, self.recipes)
        self.tmp.cleanup()

    def args(self, *argv):
        return pr.parser().parse_args(list(argv))

    def fresh(self, **overrides):
        argv = ['--recipe', 'raw_invariant', '--name', 'raw', '--seed', '2024',
                '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                '--replay-mode', 'batched', '--traffic-model', 'bernoulli']
        args = self.args(*argv)
        for key, value in overrides.items():
            setattr(args, key, value)
        return pr.plan(args, self.root)

    def save(self, **overrides):
        run, _, manifest = self.fresh(**overrides)
        checkpoint = run / 'ckpt/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('payload checked by integration smoke')
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        return run, checkpoint, manifest

    def resume(self, checkpoint, *argv):
        return pr.plan(self.args('--recipe', 'raw_invariant', '--resume', str(checkpoint), *argv), self.root)

    def test_config_identical_to_base_and_core_hashes_untouched(self):
        _, _, variant = self.fresh()
        args = self.args('--recipe', 'base', '--name', 'base', '--seed', '2024',
                         '--num-updates', '1000', '--lr-final', '0', '--lr-decay-updates', '1000',
                         '--replay-mode', 'batched', '--traffic-model', 'bernoulli')
        _, _, base = pt.plan(args, self.root)
        self.assertEqual(base['config'], variant['config'])
        self.assertEqual(base['schedule'], variant['schedule'])
        self.assertEqual(variant['source_sha256'], self.source_hashes)
        self.assertEqual(pt.source_hashes(self.root), self.source_hashes)
        self.assertEqual(variant['variant_provenance']['architecture'], pr.ARCHITECTURE)
        self.assertEqual(variant['variant_provenance']['changed_ranges'], {})
        self.assertFalse((self.root / 'runs').exists())

    def test_default_horizon_is_1000_with_linear_decay(self):
        args = self.args('--recipe', 'raw_invariant', '--name', 'raw', '--seed', '2024',
                         '--replay-mode', 'batched')
        _, _, manifest = pr.plan(args, self.root)
        self.assertEqual(manifest['target_updates'], 1000)
        self.assertEqual(manifest['schedule'], {'lr_final': 0.0, 'lr_decay_updates': 1000})

    def test_resume_restores_config_schedule_and_provenance(self):
        run, checkpoint, saved = self.save(smoke_slots=32, num_updates=1)
        _, _, restored = self.resume(checkpoint, '--num-updates', '2')
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['schedule'], saved['schedule'])
        self.assertEqual(restored['variant_provenance'], saved['variant_provenance'])
        self.assertEqual(restored['target_updates'], 2)

    def test_module_changed_rejects_resume(self):
        _, checkpoint, _ = self.save()
        with (self.root / 'paper_train_raw.py').open('a') as f:
            f.write('\n# changed implementation\n')
        with self.assertRaisesRegex(ValueError, 'Variant provenance changed'):
            self.resume(checkpoint)

    def test_core_source_changed_rejects_resume(self):
        _, checkpoint, _ = self.save()
        (self.root / 'policy.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'training source changed'):
            self.resume(checkpoint)

    def test_any_config_drift_from_base_rejected(self):
        recipe = json.loads(self.variant_file.read_text())
        recipe['ue_speed_max'] = 30.0
        self.variant_file.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'identical to frozen Base'):
            self.fresh()
        self.assertFalse((self.root / 'runs').exists())

    def test_ftp3_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Bernoulli'):
            self.fresh(traffic_model='ftp3')


class RawModel(unittest.TestCase):
    """The two properties the ablation claims: batched replay stays exact, and
    the pooled heads are invariant to per-UE PMI phase rotation."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        torch.set_num_threads(2)
        cls.torch, cls.np = torch, np
        cls.Raw = pr.install()
        import policy as P
        from config import Config
        from env import SchedulerEnv
        cls.P = P
        raw = json.loads((ROOT / 'artifacts/base/config.json').read_text())
        raw = {k: v for k, v in raw.items() if k not in ('git_hash', 'git_dirty_py')}
        for k in ('la_beta_by_depth', 'ue_speed_mix'):
            raw[k] = tuple(raw.get(k) or ())
        raw.update(episode_len_main=64, episode_len_debug=64)
        cls.cfg = Config(**raw)
        cls.env = SchedulerEnv(cls.cfg)
        torch.manual_seed(3)
        cls.ac = cls.Raw(cls.cfg).eval()

    def test_encoder_removed_and_heads_sized(self):
        torch = self.torch
        self.assertIsInstance(self.ac.encoder, torch.nn.Identity)
        self.assertEqual(self.ac.score_net.net[0].in_features, 80)
        self.assertEqual(self.ac.no_user_head.net[0].in_features, 17)
        self.assertEqual(self.ac.value_head.net[0].in_features, 60)

    def test_batched_replay_matches_sequential(self):
        obs = self.env.reset(90002)
        slots = []
        for _ in range(4):
            out = self.ac.decode(obs, deterministic=False, emit_context=True)
            slots.append((obs, out))
            obs, _, _, _ = self.env.step(out['action_sequence'])
        bat = self.ac.replay_batch([o['context'] for _, o in slots])
        for j, (ob, o) in enumerate(slots):
            seq = self.ac.replay(ob, o['action_sequence'], o['policy_decision_mask'])
            for k in ('log_prob_sum', 'entropy_sum', 'value'):
                self.assertLess(abs(float(seq[k]) - float(bat[k][j])), 1e-4, k)
            self.assertEqual(int(seq['num_policy_decisions']), int(bat['num_policy_decisions'][j]))
            self.assertEqual(o['context']['nu_aux'].shape[1], pr.NO_USER_AUX)
            self.assertEqual(o['context']['v_tail'].shape[0], 42)

    def test_pooled_heads_phase_invariant(self):
        torch, np, P, cfg = self.torch, self.np, self.P, self.cfg
        ob = self.env.reset(90003)
        dev = self.ac.device
        K = cfg.num_ue
        S = set(ob['initial_S_r'][0])
        isc = np.zeros(K, dtype=np.int64)
        tu = torch.from_numpy(ob['uncommitted'].astype(np.float32))
        def heads(o):
            t = P.obs_to_tensors(o, dev)
            e = self.ac.encoder(P.build_encoder_input(t, cfg))
            lg, _, _, _, nu = self.ac._position_logits_and_mask(t, e, 0, 0, S, isc, tu, return_ctx=True)
            return float(lg[0]), nu, float(self.ac.value(e, t, o['fixed_mask'], o['slot']))
        nu_logit0, nu0, v0 = heads(ob)
        rot = dict(ob)
        phase = np.exp(1j * np.random.default_rng(1).uniform(0, 2 * np.pi, size=(K, 1, 1)))
        rot['direction_fb'] = ob['direction_fb'] * phase
        nu_logit1, nu1, v1 = heads(rot)
        self.assertLess(abs(nu_logit0 - nu_logit1), 1e-4)
        self.assertLess(float((nu0 - nu1).abs().max()), 1e-5)
        self.assertLess(abs(v0 - v1), 1e-3)


if __name__ == '__main__':
    unittest.main()
