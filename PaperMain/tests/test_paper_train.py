"""Manifest/path/schedule guards. No channel generation or GPU allocation."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_train as pt


class PaperTrainGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in pt.SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# frozen training source\n')
        from config import phase4_queue_config
        cfg = phase4_queue_config()
        cfg.cqi_mode = 'nr4bit'
        cfg.la_mode = 'post_rzf'
        cfg.decode_order = 'rbg_major'
        cfg.la_beta_by_depth = (1.0018, .7499, .6592, .6058)
        cfg.p_arrival_min, cfg.p_arrival_max = .15, .50
        cfg.ue_speed_max = 40
        cfg.ppo_entropy_coef = .02
        for recipe in pt.RECIPES:
            raw = dataclasses.asdict(cfg)
            raw['ppo_batched_replay'] = recipe != 'base'
            if recipe == 'narrow':
                raw.update(p_arrival_min=.30, p_arrival_max=.30,
                           ue_speed_min=20, ue_speed_max=20, n_active_min=24, n_active_max=24)
            path = self.root / 'artifacts' / recipe / 'config.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(raw))

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *argv):
        return pt.parser().parse_args(list(argv))

    def saved_run(self, recipe='lrann', profile=None, seed=None, schedule=None, replay_mode=None):
        extra = ['--calibration-profile', str(profile)] if profile else []
        if seed is not None:
            extra += ['--seed', str(seed)]
        if schedule is not None:
            extra += ['--lr-final', str(schedule[0]), '--lr-decay-updates', str(schedule[1])]
        if replay_mode is not None:
            extra += ['--replay-mode', replay_mode]
        run, _, manifest = pt.plan(self.args('--recipe', recipe, '--name', 'smoke',
                                             '--smoke-slots', '32', '--num-updates', '1', *extra), self.root)
        ck = run / 'ckpt' / 'latest.pt'
        ck.parent.mkdir(parents=True)
        ck.write_text('checkpoint payload is validated by the integration smoke')
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        return run, ck, manifest

    def profile_file(self):
        from calibration import profile as cp
        for name in (*cp.SCIENCE_SOURCES, 'calibration/cqi4.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text('# frozen calibration source\n')
        doc = dict(schema_version=cp.SCHEMA_VERSION, protocol=cp.PROTOCOL,
                   status='validated', reference_config_sha256=cp.sha256_file(self.root/cp.BASE_CONFIG),
                   config=cp.normalized_config(json.loads((self.root/cp.BASE_CONFIG).read_text())),
                   source_sha256=cp.source_hashes(self.root), beta_rounded=[1., .8, .7, .6],
                   membership_verified_calibration=True, membership_verified_holdout=True,
                   calibration=dict(start=50000, episodes=36, slots=1000, sampler_seed=777),
                   holdout=dict(start=70000, episodes=12, slots=1000, sampler_seed=20250713,
                                beta_rounded=[1., .8, .7, .6],
                                by_depth={str(m):dict(member_count=1000, acks=900, first_ack=.9,
                                                     episode_cluster_ci95=[.88,.92]) for m in range(1,5)}))
        path = self.root / 'profile.json'
        path.write_text(json.dumps(doc))
        return path

    def test_profile_applies_only_to_new_run_and_preserves_recipe(self):
        path = self.profile_file()
        original = (self.root/'artifacts/narrow/config.json').read_bytes()
        _, _, m = pt.plan(self.args('--recipe', 'narrow', '--name', 'new_profile',
                                   '--calibration-profile', str(path), '--seed', '3024'), self.root)
        self.assertEqual(m['config']['la_beta_by_depth'], [1., .8, .7, .6])
        self.assertEqual(m['config']['seed'], 3024)
        self.assertEqual(m['config']['ue_speed_max'], 20)
        self.assertEqual(m['config']['deadline_max'], 12)
        self.assertIn('NARROW', m['calibration_application'])
        self.assertEqual(m['calibration_profile']['file_sha256'], pt.sha_file(path))
        self.assertEqual((self.root/'artifacts/narrow/config.json').read_bytes(), original)
        self.assertFalse((self.root/'runs/new_profile').exists())

    def test_profile_resume_restores_snapshot_when_original_json_is_absent(self):
        path = self.profile_file()
        _, ck, saved = self.saved_run(profile=path)
        path.unlink()
        _, _, restored = pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck),
                                          '--num-updates', '2'), self.root)
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['calibration_profile'], saved['calibration_profile'])
        self.assertEqual(restored['schedule'], saved['schedule'])

    def test_profile_cannot_be_added_or_changed_on_resume(self):
        path = self.profile_file()
        _, ck, _ = self.saved_run()
        with self.assertRaisesRegex(ValueError, 'cannot be added on resume'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck),
                              '--calibration-profile', str(path)), self.root)

    def test_different_profile_rejected_on_profile_resume(self):
        path = self.profile_file()
        _, ck, _ = self.saved_run(profile=path)
        changed = json.loads(path.read_text())
        changed['beta_rounded'][3] = .61
        changed['holdout']['beta_rounded'][3] = .61
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'cannot change on resume'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck),
                              '--calibration-profile', str(path)), self.root)

    def test_same_profile_can_be_explicitly_resupplied_on_resume(self):
        path = self.profile_file()
        _, ck, saved = self.saved_run(profile=path)
        _, _, restored = pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck),
                                          '--calibration-profile', str(path)), self.root)
        self.assertEqual(restored['calibration_profile'], saved['calibration_profile'])

    def test_modified_profile_snapshot_rejected_before_writes(self):
        path = self.profile_file()
        run, ck, saved = self.saved_run(profile=path)
        saved['calibration_profile']['document']['beta_rounded'][3] = .61
        (run/'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'profile document changed'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck)), self.root)

    def test_profile_cannot_calibrate_on_training_or_validation_episodes(self):
        path = self.profile_file()
        doc = json.loads(path.read_text())
        doc['calibration']['start'] = 0
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'overlaps the profile calibration'):
            pt.plan(self.args('--recipe', 'base', '--name', 'leak',
                              '--calibration-profile', str(path)), self.root)
        self.assertFalse((self.root/'runs/leak').exists())

    def test_seed_override_cannot_reuse_calibration_or_holdout_channel_seeds(self):
        path = self.profile_file()
        for seed, phase in ((52024, 'calibration'), (72024, 'holdout'),
                            (42024, 'calibration'), (62024, 'holdout')):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                    ValueError, f'overlaps the profile {phase}'):
                pt.plan(self.args('--recipe', 'base', '--name', 'seed_overlap',
                                  '--seed', str(seed), '--num-updates', '1',
                                  '--calibration-profile', str(path)), self.root)
        self.assertFalse((self.root / 'runs/seed_overlap').exists())

    def test_channel_seed_overlap_uses_half_open_training_boundaries(self):
        path = self.profile_file()
        # Calibration channel seeds are [52024, 52060), holdout [72024, 72036).
        for seed in (52023, 52060, 72023, 72036):
            with self.subTest(seed=seed):
                pt.plan(self.args('--recipe', 'base', '--name', 'boundary',
                                  '--seed', str(seed), '--num-updates', '1',
                                  '--calibration-profile', str(path)), self.root)
        for seed in (52023, 72023):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'overlaps the profile'):
                pt.plan(self.args('--recipe', 'base', '--name', 'cross_boundary',
                                  '--seed', str(seed), '--num-updates', '2',
                                  '--calibration-profile', str(path)), self.root)

    def test_channel_seed_overlap_checks_full_validation_band_boundaries(self):
        path = self.profile_file()
        cfg = json.loads((self.root / 'artifacts/base/config.json').read_text())
        episodes = cfg['ppo_eval_episodes']
        for start, end in ((52024, 52060), (72024, 72036)):
            for seed in (start - 10000 - episodes, end - 10000):
                with self.subTest(seed=seed):
                    pt.plan(self.args('--recipe', 'base', '--name', 'validation_boundary',
                                      '--seed', str(seed), '--num-updates', '1',
                                      '--calibration-profile', str(path)), self.root)
            seed = start - 10000 - episodes + 1
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'overlaps the profile'):
                pt.plan(self.args('--recipe', 'base', '--name', 'validation_overlap',
                                  '--seed', str(seed), '--num-updates', '1',
                                  '--calibration-profile', str(path)), self.root)

    def test_negative_relative_channel_seed_offset_is_valid(self):
        path = self.profile_file()
        for seed in (0, 2024, 3024, 4024, 5024):
            with self.subTest(seed=seed):
                _, _, manifest = pt.plan(self.args('--recipe', 'base', '--name', 'safe_seed',
                                                  '--seed', str(seed),
                                                  '--calibration-profile', str(path)), self.root)
                self.assertEqual(manifest['config']['seed'], seed)

    def test_resume_extension_cannot_cross_a_calibration_channel_seed_boundary(self):
        path = self.profile_file()
        _, checkpoint, _ = self.saved_run(recipe='base', profile=path, seed=52023)
        with self.assertRaisesRegex(ValueError, 'overlaps the profile calibration'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint),
                              '--num-updates', '2'), self.root)

    def test_lrann_short_test_does_not_change_866_update_horizon(self):
        run, ck, m = pt.plan(self.args('--recipe', 'lrann', '--name', 'fresh',
                                      '--smoke-slots', '32', '--num-updates', '1', '--seed', '3024'), self.root)
        self.assertEqual(m['schedule'], {'lr_final': 0.0, 'lr_decay_updates': 866})
        self.assertEqual(m['config']['num_ue'], 32)
        self.assertEqual(m['config']['episode_len_main'], 32)
        self.assertEqual(m['config']['seed'], 3024)
        self.assertTrue(m['config']['ppo_batched_replay'])
        self.assertFalse(run.exists())

    def test_historical_base_stays_sequential_and_fixed_lr(self):
        _, _, m = pt.plan(self.args('--recipe', 'base', '--name', 'fresh'), self.root)
        self.assertFalse(m['config']['ppo_batched_replay'])
        self.assertIsNone(m['schedule']['lr_final'])
        self.assertEqual(m['schedule_origin'], 'recipe')

    def test_replay_defaults_remain_bound_to_each_frozen_recipe(self):
        for recipe, batched in (('base', False), ('lrann', True), ('narrow', True)):
            with self.subTest(recipe=recipe):
                _, _, manifest = pt.plan(self.args('--recipe', recipe, '--name', 'default'), self.root)
                self.assertIs(manifest['config']['ppo_batched_replay'], batched)
                self.assertEqual(manifest['replay_origin'], 'recipe')

    def test_replay_override_changes_only_replay_configuration(self):
        for recipe, mode in (('base', 'batched'), ('lrann', 'sequential'), ('narrow', 'sequential')):
            with self.subTest(recipe=recipe):
                frozen = (self.root / 'artifacts' / recipe / 'config.json').read_bytes()
                _, _, original = pt.plan(self.args('--recipe', recipe, '--name', 'original'), self.root)
                run, _, changed = pt.plan(self.args('--recipe', recipe, '--name', 'changed',
                                                    '--replay-mode', mode), self.root)
                expected = dict(original['config'], ppo_batched_replay=mode == 'batched')
                self.assertEqual(changed['config'], expected)
                self.assertEqual(changed['replay_origin'], 'override')
                self.assertEqual(changed['schedule'], original['schedule'])
                self.assertEqual(changed['target_updates'], original['target_updates'])
                self.assertEqual((self.root / 'artifacts' / recipe / 'config.json').read_bytes(), frozen)
                self.assertFalse(run.exists())

    def test_batched_profile_lr_override_restores_all_saved_settings_on_resume(self):
        path = self.profile_file()
        _, ck, saved = self.saved_run(recipe='base', profile=path, schedule=(0., 1000),
                                      replay_mode='batched')
        path.unlink()
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                                          '--num-updates', '1000'), self.root)
        self.assertTrue(restored['config']['ppo_batched_replay'])
        self.assertEqual(restored['replay_origin'], 'override')
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['calibration_profile'], saved['calibration_profile'])
        self.assertEqual(restored['schedule'], {'lr_final': 0., 'lr_decay_updates': 1000})
        self.assertEqual(restored['target_updates'], 1000)

    def test_resume_accepts_same_replay_override_and_rejects_changed_mode(self):
        _, ck, saved = self.saved_run(recipe='base', replay_mode='batched')
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                                          '--replay-mode', 'batched'), self.root)
        self.assertEqual(restored['config'], saved['config'])
        with self.assertRaisesRegex(ValueError, 'replay mode cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                              '--replay-mode', 'sequential'), self.root)

    def test_legacy_replay_resume_keeps_recipe_guard(self):
        run, ck, saved = self.saved_run(recipe='base')
        saved.pop('replay_origin')
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)
        self.assertFalse(restored['config']['ppo_batched_replay'])
        with self.assertRaisesRegex(ValueError, 'replay mode cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                              '--replay-mode', 'batched'), self.root)
        saved['config']['ppo_batched_replay'] = True
        (run / 'config.json').write_text(json.dumps(saved['config']))
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'replay mode differs from the frozen recipe'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_explicit_recipe_replay_origin_rejects_modified_mode(self):
        run, ck, saved = self.saved_run(recipe='base')
        saved['config']['ppo_batched_replay'] = True
        (run / 'config.json').write_text(json.dumps(saved['config']))
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'replay mode differs from the frozen recipe'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_unknown_replay_origin_is_rejected(self):
        run, ck, saved = self.saved_run(recipe='base', replay_mode='batched')
        saved['replay_origin'] = 'unknown'
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'unknown saved replay origin'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_saved_replay_flag_requires_boolean(self):
        run, ck, saved = self.saved_run(recipe='base', replay_mode='batched')
        for invalid in ('false', 0, 1, None):
            with self.subTest(invalid=invalid):
                saved['config']['ppo_batched_replay'] = invalid
                (run / 'config.json').write_text(json.dumps(saved['config']))
                (run / 'paper_manifest.json').write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, 'ppo_batched_replay must be a boolean'):
                    pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_batched_replay_rejects_unsupported_decode_order_before_writes(self):
        recipe_file = self.root / 'artifacts/base/config.json'
        raw = json.loads(recipe_file.read_text())
        raw.update(la_mode='legacy', decode_order='layer_major')
        recipe_file.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'batched replay requires decode_order=rbg_major'):
            pt.plan(self.args('--recipe', 'base', '--name', 'unsupported',
                              '--replay-mode', 'batched'), self.root)
        self.assertFalse((self.root / 'runs/unsupported').exists())

    def test_base_lr_override_changes_only_schedule_and_total_target(self):
        _, _, original = pt.plan(self.args('--recipe', 'base', '--name', 'original'), self.root)
        run, _, changed = pt.plan(self.args('--recipe', 'base', '--name', 'annealed',
                                           '--num-updates', '1000', '--lr-final', '0',
                                           '--lr-decay-updates', '1000'), self.root)
        self.assertEqual(changed['config'], original['config'])
        self.assertFalse(changed['config']['ppo_batched_replay'])
        self.assertEqual(changed['target_updates'], 1000)
        self.assertEqual(changed['schedule'], {'lr_final': 0., 'lr_decay_updates': 1000})
        self.assertEqual(changed['schedule_origin'], 'override')
        self.assertFalse(run.exists())

    def test_override_is_compatible_with_corrected_profile_and_resume_snapshot(self):
        path = self.profile_file()
        _, ck, saved = self.saved_run(recipe='base', profile=path, schedule=(0., 1000))
        path.unlink()
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                                           '--num-updates', '1000'), self.root)
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['calibration_profile'], saved['calibration_profile'])
        self.assertEqual(restored['schedule'], saved['schedule'])
        self.assertEqual(restored['schedule_origin'], 'override')
        self.assertEqual(restored['target_updates'], 1000)

    def test_override_resume_allows_same_schedule_but_rejects_either_change(self):
        _, ck, saved = self.saved_run(recipe='base', schedule=(0., 1000))
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                                           '--lr-final', '0', '--lr-decay-updates', '1000'), self.root)
        self.assertEqual(restored['schedule'], saved['schedule'])
        for final, horizon in (('0.00001', '1000'), ('0', '999')):
            with self.subTest(final=final, horizon=horizon), self.assertRaisesRegex(
                    ValueError, 'LR schedule cannot change on resume'):
                pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                                  '--lr-final', final, '--lr-decay-updates', horizon), self.root)

    def test_legacy_resume_retains_recipe_guard_and_cannot_add_decay(self):
        run, ck, saved = self.saved_run(recipe='base')
        saved.pop('schedule_origin')
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)
        self.assertEqual(restored['schedule'], {'lr_final': None, 'lr_decay_updates': None})
        with self.assertRaisesRegex(ValueError, 'LR schedule cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                              '--lr-final', '0', '--lr-decay-updates', '1000'), self.root)
        saved['schedule'] = {'lr_final': 0., 'lr_decay_updates': 1000}
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'schedule differs from the frozen recipe'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_lr_override_requires_both_arguments_before_writes(self):
        for extra in (('--lr-final', '0'), ('--lr-decay-updates', '1000')):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'supplied together'):
                pt.plan(self.args('--recipe', 'base', '--name', 'invalid', *extra), self.root)
        self.assertFalse((self.root / 'runs/invalid').exists())

    def test_lr_override_rejects_nonfinite_negative_or_increasing_endpoint(self):
        for final in ('nan', 'inf', '-inf', '-0.0001', '0.000301'):
            with self.subTest(final=final), self.assertRaisesRegex(ValueError, 'finite and between'):
                pt.plan(self.args('--recipe', 'base', '--name', 'invalid',
                                  '--lr-final=' + final, '--lr-decay-updates', '1000'), self.root)
        self.assertFalse((self.root / 'runs/invalid').exists())

    def test_lr_override_validates_horizon_and_accepts_equal_initial_endpoint(self):
        for horizon in ('0', '-1', '10001'):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, '1..10000'):
                pt.plan(self.args('--recipe', 'base', '--name', 'invalid',
                                  '--lr-final', '0', '--lr-decay-updates', horizon), self.root)
        _, _, manifest = pt.plan(self.args('--recipe', 'base', '--name', 'flat',
                                           '--lr-final', '0.0003', '--lr-decay-updates', '1000'), self.root)
        self.assertEqual(manifest['schedule']['lr_final'], manifest['config']['ppo_learning_rate'])

    def test_new_fixed_schedule_cannot_be_changed_on_resume(self):
        _, ck, _ = self.saved_run(recipe='base')
        with self.assertRaisesRegex(ValueError, 'LR schedule cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck),
                              '--lr-final', '0', '--lr-decay-updates', '1000'), self.root)

    def test_unknown_schedule_origin_is_rejected(self):
        run, ck, saved = self.saved_run(recipe='base', schedule=(0., 1000))
        saved['schedule_origin'] = 'unknown'
        (run / 'paper_manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'unknown saved LR schedule origin'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(ck)), self.root)

    def test_narrow_keeps_actual_training_world(self):
        _, _, m = pt.plan(self.args('--recipe', 'narrow', '--name', 'fresh'), self.root)
        self.assertEqual(m['config']['ue_speed_max'], 20)
        self.assertEqual(m['config']['p_arrival_max'], .30)
        self.assertEqual(m['config']['n_active_min'], 24)

    def test_resume_extends_target_but_restores_schedule_and_config(self):
        run, ck, saved = self.saved_run()
        _, _, restored = pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck),
                                           '--num-updates', '2'), self.root)
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['schedule'], saved['schedule'])
        self.assertEqual(restored['target_updates'], 2)
        self.assertEqual(json.loads((run/'paper_manifest.json').read_text())['target_updates'], 1)

    def test_resume_rejects_changed_seed_or_threads(self):
        _, ck, _ = self.saved_run()
        for flag, value in (('--seed', '3024'), ('--threads', '2'), ('--smoke-slots', '64')):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'cannot change'):
                pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck), flag, value), self.root)

    def test_resume_rejects_changed_source(self):
        _, ck, _ = self.saved_run()
        (self.root/'env.py').write_text('# changed source\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck)), self.root)

    def test_resume_rejects_changed_run_config(self):
        run, ck, saved = self.saved_run()
        changed = copy.deepcopy(saved['config'])
        changed['cqi_mode'] = 'continuous'
        (run/'config.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'differs from the manifest'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck)), self.root)

    def test_resume_rejects_changed_schedule(self):
        run, ck, m = self.saved_run()
        m['schedule']['lr_decay_updates'] = 2
        (run/'paper_manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            pt.plan(self.args('--recipe', 'lrann', '--resume', str(ck)), self.root)

    def test_artifact_and_best_checkpoint_resume_are_rejected(self):
        _, ck, _ = self.saved_run()
        for candidate in (self.root/'artifacts/lrann/ckpt/latest.pt', ck.with_name('best.pt')):
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text('do not touch')
            with self.assertRaisesRegex(ValueError, 'artifacts cannot be modified'):
                pt.plan(self.args('--recipe', 'lrann', '--resume', str(candidate)), self.root)
            self.assertEqual(candidate.read_text(), 'do not touch')

    def test_path_escape_and_existing_run_are_rejected(self):
        self.saved_run()
        for name in ('../escape', 'smoke'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pt.plan(self.args('--recipe', 'base', '--name', name), self.root)

    def test_thread_environment_set_before_config_import(self):
        pt.plan(self.args('--recipe', 'base', '--name', 'fresh', '--threads', '3'), self.root)
        self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '3')
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')

    def test_cpu_preflight_never_touches_cuda(self):
        torch = MagicMock()
        pt.preflight_device(torch, 'cpu')
        torch.empty.assert_not_called()
        torch.cuda.is_available.assert_not_called()

    def test_cuda_preflight_rejects_busy_gpu_without_run_writes(self):
        run, _, _ = pt.plan(self.args('--recipe', 'base', '--name', 'fresh'), self.root)
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.empty.side_effect = RuntimeError('CUDA-capable device(s) is/are busy or unavailable')
        with self.assertRaisesRegex(ValueError, 'No run files were written'):
            pt.preflight_device(torch, 'cuda')
        self.assertFalse(run.exists())

    def test_cuda_preflight_synchronizes_real_allocation(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        pt.preflight_device(torch, 'cuda')
        torch.empty.assert_called_once_with(1, device='cuda')
        torch.cuda.synchronize.assert_called_once_with()


if __name__ == '__main__':
    unittest.main(verbosity=2)
