"""Detached OOD campaign safety and scientific partition contract tests."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_ood_campaign as campaign
import paper_ood_eval as evaluator


class CampaignGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / 'results/campaign'
        self.folder.mkdir(parents=True)
        self.patches = [patch.object(campaign, 'ROOT', self.root), patch.object(campaign, 'C', self.folder)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    @staticmethod
    def parsed(job):
        return evaluator.parser().parse_args(job['command'][3:])

    def test_gpu_partition_covers_models_and_prior_baselines_once(self):
        jobs = campaign.commands('main', 'eval')
        actual = {(j['gpu'], self.parsed(j).run.name) for j in jobs}
        self.assertEqual(actual, {(3, campaign.RUNS[16]), (4, campaign.RUNS[21]), (5, campaign.RUNS[22])})
        baselines = []
        for job in jobs:
            args = self.parsed(job)
            self.assertEqual(args.gpu, job['gpu'])
            self.assertEqual(args.base_run.name, campaign.RUNS[21])
            self.assertEqual(args.worlds, list(evaluator.WORLDS))
            self.assertEqual((args.episode_start, args.episodes, args.smoke_slots), (140000, 100, None))
            self.assertEqual(args.schedulers.count('ppo'), 1)
            baselines.extend(key for key in args.schedulers if key != 'ppo')
        self.assertEqual(set(baselines), set(evaluator.SCHEDULERS) - {'ppo'})
        self.assertEqual(len(baselines), 6)

    def test_separate_pilot_grid_uses_base_policy_and_no_evaluation_seeds(self):
        worlds = []
        for job in campaign.commands('main', 'pilot'):
            args = self.parsed(job)
            self.assertEqual(args.run, args.base_run)
            self.assertEqual(args.run.name, campaign.RUNS[21])
            self.assertEqual(args.schedulers, ['sus_cqi'])
            self.assertEqual((args.episode_start, args.episodes), (139000, 8))
            self.assertIsNone(args.thresholds)
            worlds.extend(args.worlds)
        self.assertEqual(len(worlds), len(set(worlds)))
        self.assertEqual(set(worlds), set(evaluator.WORLDS))

    def test_smoke_is_shortened_and_output_and_episode_bands_are_separate(self):
        outputs, bands = set(), {}
        for mode in ('smoke', 'main'):
            for phase in ('pilot', 'eval'):
                jobs = campaign.commands(mode, phase)
                for job in jobs:
                    args = self.parsed(job)
                    self.assertNotIn(args.out, outputs)
                    outputs.add(args.out)
                    self.assertEqual(args.smoke_slots, 32 if mode == 'smoke' else None)
                args = self.parsed(jobs[0])
                bands[mode, phase] = set(range(args.episode_start, args.episode_start + args.episodes))
        for k, ids in bands.items():
            for other, values in bands.items():
                if k != other:
                    self.assertFalse(ids & values)

    def test_main_requires_successful_same_input_smoke_before_any_gpu_or_process(self):
        frozen = {'core': {'env.py': 'fixed'}, 'tools': {'paper_run_eval.py': 'fixed'}}
        valid = dict(status='completed', hashes=frozen, cross_gpu_equivalence={'status': 'passed'})
        for mutation in ('failed', 'changed_dependency', 'missing_equivalence'):
            smoke = copy.deepcopy(valid)
            if mutation == 'failed':
                smoke['status'] = 'failed'
            elif mutation == 'changed_dependency':
                smoke['hashes']['tools']['paper_run_eval.py'] = 'different'
            else:
                smoke.pop('cross_gpu_equivalence')
            (self.folder / 'smoke_launch.json').write_text(json.dumps(smoke))
            with self.subTest(mutation=mutation), patch.object(campaign, 'hashes', return_value=frozen), \
                    patch.object(campaign, 'free') as free, patch.object(campaign.subprocess, 'Popen') as start:
                with self.assertRaisesRegex(ValueError, 'matching successful smoke'):
                    campaign.main(['main'])
                free.assert_not_called()
                start.assert_not_called()
                self.assertFalse((self.folder / 'main_launch.json').exists())

    def test_existing_launch_record_is_never_overwritten(self):
        record = self.folder / 'smoke_launch.json'
        record.write_bytes(b'original record')
        with patch.object(campaign, 'hashes', return_value={}), patch.object(campaign, 'free') as free, \
                patch.object(campaign.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'cannot be overwritten'):
                campaign.main(['smoke'])
            free.assert_not_called()
            start.assert_not_called()
        self.assertEqual(record.read_bytes(), b'original record')

    def test_stage_source_drift_stops_before_launch(self):
        with patch.object(campaign, 'hashes', return_value={'revision': 'new'}), \
                patch.object(campaign, 'free') as free, patch.object(campaign.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                campaign.run_phase('main', 'pilot', {'hashes': {'revision': 'old'}}, self.folder / 'state.json')
            free.assert_not_called()
            start.assert_not_called()

    def test_partial_dispatch_failure_terminates_already_started_worker(self):
        child = Mock()
        child.pid = 1234
        child.poll.return_value = None
        state = {'hashes': {'revision': 'fixed'}}
        with patch.object(campaign, 'hashes', return_value=state['hashes']), \
                patch.object(campaign, 'free') as free, \
                patch.object(campaign.subprocess, 'Popen', side_effect=[child, OSError('dispatch failed')]):
            with self.assertRaisesRegex(OSError, 'dispatch failed'):
                campaign.run_phase('smoke', 'pilot', state, self.folder / 'state.json')
        self.assertEqual([c.args[0] for c in free.call_args_list], [3, 4, 5])
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=30)

    def test_shared_evaluation_dependencies_are_bound_to_smoke_hashes(self):
        self.assertTrue(set(evaluator.RUNNER_FILES).issubset(campaign.NEW_FILES))
        self.assertIn('paper_train_variants.py', campaign.NEW_FILES)

    def test_threshold_freeze_recomputes_pilot_and_refuses_overwrite(self):
        from paper_ood_eval import THRESHOLDS
        for job in campaign.commands('smoke', 'pilot'):
            folder = Path(job['out'])
            folder.mkdir(parents=True)
            args = self.parsed(job)
            with (folder / 'metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['world', 'threshold', 'episode_idx', 'scheduler', 'mode', 'reward'])
                writer.writeheader()
                for world in args.worlds:
                    for threshold in THRESHOLDS:
                        writer.writerow(dict(world=world, threshold=threshold, episode_idx=138000,
                                             scheduler='sus_cqi', mode='pilot',
                                             reward=10 if threshold in (.65, .75) else 5))
        with patch.object(campaign, 'completed', return_value={'status': 'completed'}):
            document = campaign.freeze_thresholds('smoke')
            self.assertEqual(document['thresholds'], {world: .65 for world in evaluator.WORLDS})
            self.assertTrue(document['diagnostic_only'])
            self.assertEqual(document['pilot_episode_ids'], [138000])
            before = (self.folder / 'smoke/thresholds.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                campaign.freeze_thresholds('smoke')
            self.assertEqual(before, (self.folder / 'smoke/thresholds.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
