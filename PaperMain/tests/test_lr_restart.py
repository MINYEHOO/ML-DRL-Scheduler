"""Learning-rate continuation guards and arithmetic; no training or GPU allocation."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_train as pt
import test_paper_train as guards


SEGMENT = {
    'lr_final': 0.0,
    'lr_decay_updates': 1000,
    'lr_initial': 1e-5,
    'lr_start_update': 1000,
}


class LearningRateRestartGuards(unittest.TestCase):
    # Reuse the temporary frozen-recipe fixtures without inheriting old tests.
    setUp = guards.PaperTrainGuards.setUp
    tearDown = guards.PaperTrainGuards.tearDown
    args = guards.PaperTrainGuards.args
    saved_run = guards.PaperTrainGuards.saved_run

    def saved_segment(self):
        run, checkpoint, manifest = self.saved_run(recipe='base', schedule=(0., 1000))
        # A new continuation directory is prepared explicitly from its parent;
        # changing an existing run's schedule through CLI must remain forbidden.
        manifest['schedule'] = dict(SEGMENT)
        manifest['target_updates'] = 2000
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        return run, checkpoint, manifest

    def test_legacy_two_field_schedules_remain_valid_and_unchanged(self):
        for schedule in ({'lr_final': None, 'lr_decay_updates': None},
                         {'lr_final': 0., 'lr_decay_updates': 1000},
                         {'lr_final': 3e-4, 'lr_decay_updates': 1}):
            with self.subTest(schedule=schedule):
                original = copy.deepcopy(schedule)
                pt.validate_schedule(schedule, 3e-4)
                self.assertEqual(schedule, original)

    def test_segment_accepts_boundaries_without_changing_configuration(self):
        for changes in ({}, {'lr_start_update': 0},
                        {'lr_start_update': 9999, 'lr_decay_updates': 1},
                        {'lr_start_update': 0, 'lr_decay_updates': 10000},
                        {'lr_initial': 3e-4, 'lr_final': 3e-4}):
            with self.subTest(changes=changes):
                schedule = dict(SEGMENT, **changes)
                original = copy.deepcopy(schedule)
                pt.validate_schedule(schedule, 3e-4)
                self.assertEqual(schedule, original)
        _, _, manifest = self.saved_segment()
        self.assertEqual(manifest['config']['ppo_learning_rate'], 3e-4)

    def test_segment_initial_must_be_positive_finite_and_within_recipe_lr(self):
        for value in (True, False, None, '0.00001', float('nan'),
                      float('inf'), -float('inf'), 0., -1e-5, 0.000301):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pt.validate_schedule(dict(SEGMENT, lr_initial=value), 3e-4)

    def test_endpoint_must_not_exceed_segment_initial(self):
        for value in (True, False, None, '0', float('nan'), float('inf'),
                      -float('inf'), -1e-8, 1.0001e-5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pt.validate_schedule(dict(SEGMENT, lr_final=value), 3e-4)
        pt.validate_schedule(dict(SEGMENT, lr_final=1e-5), 3e-4)

    def test_segment_start_requires_integer_in_range(self):
        for value in (True, False, None, '1000', 1000., float('nan'),
                      float('inf'), -1, 10000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pt.validate_schedule(dict(SEGMENT, lr_start_update=value), 3e-4)

    def test_horizon_requires_integer_and_segment_must_end_by_10000(self):
        for value in (True, False, None, '1000', 1000., float('nan'),
                      float('inf'), -1, 0, 10001, 9001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pt.validate_schedule(dict(SEGMENT, lr_decay_updates=value), 3e-4)
        pt.validate_schedule(dict(SEGMENT, lr_decay_updates=9000), 3e-4)

    def test_saved_schema_rejects_partial_or_unknown_keys(self):
        for key in SEGMENT:
            schedule = dict(SEGMENT)
            schedule.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                pt.validate_schedule(schedule, 3e-4)
        for schedule in (None, [], {}, dict(SEGMENT, unknown=0),
                         dict(SEGMENT, lr_final=None, lr_decay_updates=None)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                pt.validate_schedule(schedule, 3e-4)

    def test_cli_segment_arguments_require_each_other_and_linear_endpoint(self):
        for extra in (('--lr-initial', '0.00001'),
                      ('--lr-start-update', '0'),
                      ('--lr-initial', '0.00001', '--lr-start-update', '0'),
                      ('--lr-final', '0', '--lr-decay-updates', '1000',
                       '--lr-initial', '0.00001'),
                      ('--lr-final', '0', '--lr-decay-updates', '1000',
                       '--lr-start-update', '0')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                pt.plan(self.args('--recipe', 'base', '--name', 'invalid', *extra), self.root)
        self.assertFalse((self.root / 'runs/invalid').exists())

    def test_fresh_nonzero_segment_start_is_rejected_before_writes(self):
        with self.assertRaises(ValueError):
            pt.plan(self.args('--recipe', 'base', '--name', 'invalid',
                              '--lr-final', '0', '--lr-decay-updates', '1000',
                              '--lr-initial', '0.00001', '--lr-start-update', '1000'), self.root)
        self.assertFalse((self.root / 'runs/invalid').exists())

    def test_fresh_zero_start_uses_segment_lr_without_editing_recipe_config(self):
        _, _, original = pt.plan(self.args('--recipe', 'base', '--name', 'original'), self.root)
        run, _, changed = pt.plan(self.args('--recipe', 'base', '--name', 'small_lr',
                                            '--lr-final', '0', '--lr-decay-updates', '1000',
                                            '--lr-initial', '0.00001', '--lr-start-update', '0'), self.root)
        self.assertEqual(changed['schedule'], dict(SEGMENT, lr_start_update=0))
        self.assertEqual(changed['schedule_origin'], 'override')
        self.assertEqual(changed['config'], original['config'])
        self.assertFalse(run.exists())

    def test_direct_driver_rejects_explicit_zero_horizon_before_run_or_environment(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            import train_phase2 as driver
        run_root = self.root / 'direct_driver_runs'
        argv = ['train_phase2.py', '--mode', 'queue', '--run_root', str(run_root),
                '--run_name', 'invalid', '--num_updates', '1',
                '--lr_final', '0', '--lr_decay_updates', '0']
        with patch.object(sys, 'argv', argv), \
                patch.object(driver, 'SchedulerEnv') as environment, \
                patch.object(driver, 'ActorCritic') as actor, \
                self.assertRaisesRegex(ValueError, '1..10000'):
            driver.main()
        environment.assert_not_called()
        actor.assert_not_called()
        self.assertFalse(run_root.exists())

    def test_resume_omission_restores_full_segment_and_preserves_saved_files(self):
        run, checkpoint, saved = self.saved_segment()
        before = (run / 'paper_manifest.json').read_bytes()
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint),
                                          '--num-updates', '2000'), self.root)
        self.assertEqual(restored['schedule'], SEGMENT)
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual((run / 'paper_manifest.json').read_bytes(), before)

    def test_resume_accepts_identical_full_segment(self):
        _, checkpoint, _ = self.saved_segment()
        _, _, restored = pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint),
                                          '--lr-final', '0', '--lr-decay-updates', '1000',
                                          '--lr-initial', '0.00001', '--lr-start-update', '1000'), self.root)
        self.assertEqual(restored['schedule'], SEGMENT)

    def test_resume_rejects_changes_to_any_segment_component(self):
        _, checkpoint, _ = self.saved_segment()
        changes = ({'lr_final': 1e-6}, {'lr_decay_updates': 999},
                   {'lr_initial': 2e-5}, {'lr_start_update': 999})
        for change in changes:
            requested = dict(SEGMENT, **change)
            extra = [item for key, value in requested.items()
                     for item in ('--' + key.replace('_', '-'), str(value))]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'cannot change on resume'):
                pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint), *extra), self.root)

    def test_resume_cannot_silently_drop_segment_or_add_it_to_legacy_schedule(self):
        run, checkpoint, manifest = self.saved_segment()
        with self.assertRaisesRegex(ValueError, 'cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint),
                              '--lr-final', '0', '--lr-decay-updates', '1000'), self.root)
        manifest['schedule'] = {'lr_final': 0., 'lr_decay_updates': 1000}
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'cannot change on resume'):
            pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint),
                              '--lr-final', '0', '--lr-decay-updates', '1000',
                              '--lr-initial', '0.00001', '--lr-start-update', '1000'), self.root)

    def preflight(self, run, checkpoint, manifest, update, saved_schedule='absent'):
        import torch
        payload = {'cfg': manifest['config'], 'update': update, 'model': {}}
        if saved_schedule != 'absent':
            payload['lr_schedule'] = saved_schedule
        manifest = copy.deepcopy(manifest)
        manifest['runtime'] = {'fixture': 'cpu'}
        with patch.object(pt, 'plan', return_value=(run, checkpoint, manifest)), \
                patch.object(pt, 'runtime_info', return_value=manifest['runtime']), \
                patch.object(torch, 'load', return_value=payload), \
                patch.object(torch, 'set_num_threads'), \
                patch('policy.ActorCritic') as actor, \
                contextlib.redirect_stdout(io.StringIO()):
            pt.main(['--recipe', 'base', '--resume', str(checkpoint), '--dry-run'])
            actor.return_value.load_state_dict.assert_called_once_with({}, strict=True)

    def test_preflight_rejects_checkpoint_before_segment_start_without_writes(self):
        run, checkpoint, manifest = self.saved_segment()
        before = (run / 'paper_manifest.json').read_bytes()
        with self.assertRaises(SystemExit):
            self.preflight(run, checkpoint, manifest, update=998)
        self.assertEqual((run / 'paper_manifest.json').read_bytes(), before)

    def test_preflight_accepts_first_segment_update_and_mid_segment_resume(self):
        run, checkpoint, manifest = self.saved_segment()
        for update in (999, 1000, 1499, 1998):
            with self.subTest(update=update):
                self.preflight(run, checkpoint, manifest, update, dict(SEGMENT))

    def test_preflight_allows_legacy_checkpoint_without_schedule_field(self):
        run, checkpoint, manifest = self.saved_segment()
        self.preflight(run, checkpoint, manifest, update=999)
        manifest['schedule'] = {'lr_final': 0., 'lr_decay_updates': 1000}
        self.preflight(run, checkpoint, manifest, update=999)

    def test_preflight_rejects_checkpoint_schedule_mismatch(self):
        run, checkpoint, manifest = self.saved_segment()
        for changed in (None, {'lr_final': 0., 'lr_decay_updates': 1000},
                        dict(SEGMENT, lr_initial=2e-5), dict(SEGMENT, lr_start_update=999)):
            with self.subTest(changed=changed), self.assertRaises(SystemExit):
                self.preflight(run, checkpoint, manifest, update=999, saved_schedule=changed)

    def test_preflight_rejects_boolean_checkpoint_schedule_even_when_equal_to_number(self):
        run, checkpoint, manifest = self.saved_segment()
        # Python dict equality alone treats False as equal to the valid 0.0.
        with self.assertRaises(SystemExit):
            self.preflight(run, checkpoint, manifest, update=999,
                           saved_schedule=dict(SEGMENT, lr_final=False))


class LearningRateRestartArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            from train_phase2 import linear_learning_rate
        cls.lr = staticmethod(linear_learning_rate)

    def test_restart_boundaries_and_linear_midpoint(self):
        for update, expected in ((1000, 1e-5), (1001, 9.99e-6), (1500, 5e-6),
                                 (1999, 1e-8), (2000, 0.), (2001, 0.)):
            with self.subTest(update=update):
                self.assertAlmostEqual(self.lr(1e-5, 0., 1000, update, start_update=1000),
                                       expected, delta=1e-19)

    def test_restart_rejects_updates_before_segment_instead_of_extrapolating(self):
        for update in (-1, 0, 999):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.lr(1e-5, 0., 1000, update, start_update=1000)

    def test_legacy_arithmetic_matches_original_operation_order_exactly(self):
        for initial, final, horizon in ((3e-4, 0., 1000), (3e-4, 0., 866),
                                        (3e-4, 1e-5, 1000), (3e-4, 3e-4, 1000)):
            for update in (0, 1, 500, 865, 866, 999, 1000, 1999):
                with self.subTest(initial=initial, final=final, horizon=horizon, update=update):
                    fraction = min(update / max(horizon, 1), 1.0)
                    expected = initial + (final - initial) * fraction
                    self.assertEqual(self.lr(initial, final, horizon, update), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
