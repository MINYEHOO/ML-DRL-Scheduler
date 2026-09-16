"""Arrival opt-in and explicit calibration-transfer guards; no GPU allocation."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import test_paper_train as fixtures
import paper_train as pt


class FTP3WrapperGuards(unittest.TestCase):
    # Reuse fixture helpers without inheriting/discovering the existing tests.
    setUp = fixtures.PaperTrainGuards.setUp
    tearDown = fixtures.PaperTrainGuards.tearDown
    args = fixtures.PaperTrainGuards.args
    profile_file = fixtures.PaperTrainGuards.profile_file

    def fresh(self, name='ftp3', *extra):
        return pt.plan(self.args('--recipe', 'base', '--name', name, *extra), self.root)

    def save(self, *extra):
        run, _, manifest = self.fresh('saved', '--traffic-model', 'ftp3', *extra)
        checkpoint = run / 'ckpt/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('plan-only checkpoint fixture')
        (run / 'config.json').write_text(json.dumps(manifest['config']))
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        return run, checkpoint, manifest

    def resume(self, checkpoint, *extra):
        return pt.plan(self.args('--recipe', 'base', '--resume', str(checkpoint), *extra), self.root)

    def transfer_fixture(self):
        """Freeze a valid source world, then change only destination arrivals."""
        from calibration import profile as cp
        profile = self.profile_file()
        relative = Path('provenance/bernoulli_reference')
        snapshot = self.root / relative
        names = set(pt.SOURCES) | set(cp.source_hashes(self.root)) | {cp.BASE_CONFIG}
        for name in names:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / name, target)
        for name in ('config.py', 'traffic.py'):
            with (self.root / name).open('a') as source:
                source.write('# optional FTP3 arrival implementation\n')
        return profile, relative, snapshot

    def test_all_recipes_default_to_bernoulli_without_override(self):
        for recipe in pt.RECIPES:
            with self.subTest(recipe=recipe):
                _, _, manifest = pt.plan(self.args('--recipe', recipe, '--name', 'default'), self.root)
                self.assertEqual(manifest['config']['traffic_model'], 'bernoulli')
                self.assertEqual(manifest['traffic_origin'], 'recipe')

    def test_ftp3_changes_only_arrival_model_and_preserves_recipe_bytes(self):
        for recipe in pt.RECIPES:
            with self.subTest(recipe=recipe):
                artifact = self.root / 'artifacts' / recipe / 'config.json'
                original_bytes = artifact.read_bytes()
                _, _, original = pt.plan(self.args('--recipe', recipe, '--name', 'default'), self.root)
                run, _, ftp3 = pt.plan(self.args('--recipe', recipe, '--name', 'ftp3',
                                               '--traffic-model', 'ftp3'), self.root)
                self.assertEqual(ftp3['config'], dict(original['config'], traffic_model='ftp3'))
                self.assertEqual(ftp3['traffic_origin'], 'override')
                for field in ('schedule', 'execution', 'target_updates', 'replay_origin'):
                    self.assertEqual(ftp3[field], original[field])
                self.assertEqual(artifact.read_bytes(), original_bytes)
                self.assertFalse(run.exists())

    def test_explicit_bernoulli_is_available_for_new_runs(self):
        _, _, default = self.fresh('default')
        _, _, explicit = self.fresh('explicit', '--traffic-model', 'bernoulli')
        self.assertEqual(explicit['config'], default['config'])
        self.assertEqual(explicit['traffic_origin'], 'override')

    def test_invalid_traffic_mode_rejected_by_cli(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--recipe', 'base', '--name', 'invalid', '--traffic-model', 'poisson_typo')

    def test_resume_restores_ftp3_without_repeating_flag(self):
        _, checkpoint, saved = self.save('--replay-mode', 'batched', '--lr-final', '0',
                                         '--lr-decay-updates', '1000', '--num-updates', '1000')
        _, _, restored = self.resume(checkpoint, '--num-updates', '2000')
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['schedule'], saved['schedule'])
        self.assertEqual(restored['traffic_origin'], 'override')
        self.assertEqual(restored['target_updates'], 2000)

    def test_resume_accepts_same_explicit_mode(self):
        _, checkpoint, saved = self.save()
        _, _, restored = self.resume(checkpoint, '--traffic-model', 'ftp3')
        self.assertEqual(restored['config'], saved['config'])

    def test_resume_rejects_arrival_model_change(self):
        _, checkpoint, _ = self.save()
        with self.assertRaisesRegex(ValueError, 'traffic.*cannot change'):
            self.resume(checkpoint, '--traffic-model', 'bernoulli')

    def test_resume_rejects_invalid_saved_traffic_origin(self):
        run, checkpoint, manifest = self.save()
        manifest['traffic_origin'] = 'untracked'
        (run / 'paper_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.resume(checkpoint)

    def test_stale_profile_remains_rejected_without_explicit_transfer(self):
        profile, _, _ = self.transfer_fixture()
        with self.assertRaisesRegex(ValueError, 'Calibration source changed'):
            self.fresh('stale', '--traffic-model', 'ftp3', '--calibration-profile', str(profile))
        self.assertFalse((self.root / 'runs/stale').exists())

    def test_explicit_transfer_validates_snapshot_and_records_relative_root(self):
        profile, relative, _ = self.transfer_fixture()
        profile_bytes = profile.read_bytes()
        _, _, manifest = self.fresh('transfer', '--traffic-model', 'ftp3',
                                    '--calibration-profile', str(profile),
                                    '--calibration-reference-root', str(relative))
        self.assertEqual(manifest['calibration_reference_root'], relative.as_posix())
        self.assertEqual(manifest['calibration_profile']['document'], json.loads(profile_bytes))
        self.assertEqual(manifest['config']['la_beta_by_depth'], [1., .8, .7, .6])
        self.assertEqual(manifest['config']['traffic_model'], 'ftp3')
        self.assertIn('transfer', manifest['calibration_application'].lower())
        self.assertIn('ftp3', manifest['calibration_application'].lower())
        self.assertEqual(profile.read_bytes(), profile_bytes)

    def test_transfer_requires_profile_on_new_run(self):
        _, relative, _ = self.transfer_fixture()
        with self.assertRaises(ValueError):
            self.fresh('missing_profile', '--traffic-model', 'ftp3',
                       '--calibration-reference-root', str(relative))

    def test_transfer_rejects_changed_destination_phy(self):
        profile, relative, _ = self.transfer_fixture()
        (self.root / 'phy.py').write_text('# changed physical model\n')
        with self.assertRaisesRegex(ValueError, 'non-traffic science'):
            self.fresh('physics', '--traffic-model', 'ftp3', '--calibration-profile', str(profile),
                       '--calibration-reference-root', str(relative))
        self.assertFalse((self.root / 'runs/physics').exists())

    def test_transfer_rejects_tampered_frozen_traffic_source(self):
        profile, relative, snapshot = self.transfer_fixture()
        (snapshot / 'traffic.py').write_text('# changed old source\n')
        with self.assertRaisesRegex(ValueError, 'Calibration source changed'):
            self.fresh('tampered', '--traffic-model', 'ftp3', '--calibration-profile', str(profile),
                       '--calibration-reference-root', str(relative))

    def test_transfer_rejects_source_inventory_changes(self):
        profile, relative, _ = self.transfer_fixture()
        (self.root / 'calibration/new_algorithm.py').write_text('# new calibration module\n')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            self.fresh('inventory', '--traffic-model', 'ftp3', '--calibration-profile', str(profile),
                       '--calibration-reference-root', str(relative))

    def test_transfer_rejects_reference_outside_provenance(self):
        profile, _, snapshot = self.transfer_fixture()
        outside = self.root / 'reference_outside_provenance'
        shutil.copytree(snapshot, outside)
        for candidate in (outside, self.root, self.root / 'provenance'):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.fresh('outside', '--traffic-model', 'ftp3', '--calibration-profile', str(profile),
                           '--calibration-reference-root', str(candidate))

    def test_transfer_rejects_symlink_escape(self):
        profile, _, snapshot = self.transfer_fixture()
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / 'snapshot'
            shutil.copytree(snapshot, outside)
            alias = self.root / 'provenance/external_snapshot'
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.fresh('escape', '--traffic-model', 'ftp3', '--calibration-profile', str(profile),
                           '--calibration-reference-root', str(alias))

    def test_transfer_resume_restores_saved_profile_after_original_file_removed(self):
        profile, relative, _ = self.transfer_fixture()
        _, checkpoint, saved = self.save('--calibration-profile', str(profile),
                                         '--calibration-reference-root', str(relative))
        profile.unlink()
        _, _, restored = self.resume(checkpoint)
        self.assertEqual(restored['config'], saved['config'])
        self.assertEqual(restored['calibration_profile'], saved['calibration_profile'])
        self.assertEqual(restored['calibration_reference_root'], relative.as_posix())

    def test_transfer_resume_cannot_change_reference(self):
        profile, relative, snapshot = self.transfer_fixture()
        _, checkpoint, _ = self.save('--calibration-profile', str(profile),
                                     '--calibration-reference-root', str(relative))
        copied = self.root / 'provenance/another_reference'
        shutil.copytree(snapshot, copied)
        with self.assertRaisesRegex(ValueError, 'reference cannot change'):
            self.resume(checkpoint, '--calibration-reference-root', str(copied))

    def test_transfer_resume_revalidates_frozen_sources(self):
        profile, relative, snapshot = self.transfer_fixture()
        _, checkpoint, _ = self.save('--calibration-profile', str(profile),
                                     '--calibration-reference-root', str(relative))
        (snapshot / 'config.py').write_text('# snapshot changed after run start\n')
        with self.assertRaisesRegex(ValueError, 'Calibration source changed'):
            self.resume(checkpoint)


if __name__ == '__main__':
    unittest.main()
