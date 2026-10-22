"""The three-run comparison rejects hidden changes to experimental conditions."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_raw_compare as compare
import paper_train as pt


class RawPhaseComparison(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(compare.DEFAULT_SPEC.read_text())
        base = pt.normalize_config(json.loads((ROOT / 'artifacts/base/config.json').read_text()))
        base.update(seed=2024, episode_len_main=1000, debug=False, traffic_model='bernoulli')
        self.prepared = []
        for run in self.spec['runs']:
            cfg = dict(base, ppo_learning_rate=run['learning_rate'])
            protocol = dict(run=run['run'], training_config=copy.deepcopy(cfg),
                environment_config=copy.deepcopy(cfg), policy_config=copy.deepcopy(cfg),
                checkpoint_kind='best', diagnostic_only=False,
                episode_ids=list(range(150000, 150100)), scheduler_keys=self.spec['scheduler_keys'],
                scheduler_names=compare.ev.SCHEDULERS.copy(), history_audit_sha256='audit-hash',
                source_sha256={'policy.py': 'core-hash'},
                calibration_profile={'document_sha256': 'calibration-hash'},
                calibration_reference_root='provenance/before_ftp3_sources')
            self.prepared.append(dict(protocol=protocol, manifest=dict(recipe=run['recipe'], config=cfg,
                target_updates=1000, schedule=copy.deepcopy(self.spec['schedule']))))

    def test_only_predeclared_lr_difference_is_accepted(self):
        compare.validate_cohort(self.spec, self.prepared)

    def test_environment_reward_seed_and_policy_changes_rejected(self):
        for key, value in [('p_csi', .2), ('traffic_model', 'ftp3'), ('lambda_m', 6.0),
                           ('seed', 2025), ('episode_len_main', 40), ('ppo_no_user_scale', .5)]:
            with self.subTest(key=key):
                prepared = copy.deepcopy(self.prepared)
                for field in ('training_config', 'environment_config', 'policy_config'):
                    prepared[1]['protocol'][field][key] = value
                with self.assertRaisesRegex(ValueError, 'Configurations differ'):
                    compare.validate_cohort(self.spec, prepared)

    def test_wrong_lr_cannot_hide_behind_allowed_difference(self):
        self.prepared[2]['protocol']['training_config']['ppo_learning_rate'] = .0003
        with self.assertRaisesRegex(ValueError, 'learning rate mismatch'):
            compare.validate_cohort(self.spec, self.prepared)

    def test_semantic_numeric_type_changes_rejected(self):
        self.prepared[1]['protocol']['training_config']['episode_len_main'] = 1000.0
        with self.assertRaisesRegex(ValueError, 'Configurations differ'):
            compare.validate_cohort(self.spec, self.prepared)

    def test_budget_schedule_and_identity_rejected(self):
        for key, value in [('target_updates', 2000), ('schedule', {'lr_final': .0001, 'lr_decay_updates': 1000}),
                           ('recipe', 'base')]:
            with self.subTest(key=key):
                prepared = copy.deepcopy(self.prepared)
                prepared[1]['manifest'][key] = value
                with self.assertRaises(ValueError):
                    compare.validate_cohort(self.spec, prepared)

    def test_checkpoint_calibration_source_history_and_episode_mismatch_rejected(self):
        changes = [('checkpoint_kind', 'latest'), ('diagnostic_only', True),
                   ('calibration_profile', {'document_sha256': 'other'}),
                   ('source_sha256', {'policy.py': 'changed'}), ('history_audit_sha256', None),
                   ('episode_ids', list(range(150001, 150101))), ('scheduler_keys', ['ppo'])]
        for key, value in changes:
            with self.subTest(key=key):
                prepared = copy.deepcopy(self.prepared)
                prepared[1]['protocol'][key] = value
                with self.assertRaises(ValueError):
                    compare.validate_cohort(self.spec, prepared)

    def test_same_wrong_episode_band_and_missing_audit_are_rejected(self):
        for key, value in [('episode_ids', list(range(150001, 150101))), ('history_audit_sha256', None),
                           ('scheduler_keys', ['ppo'])]:
            with self.subTest(key=key):
                prepared = copy.deepcopy(self.prepared)
                for item in prepared:
                    item['protocol'][key] = value
                with self.assertRaises(ValueError):
                    compare.validate_cohort(self.spec, prepared)

    def test_environment_or_policy_override_rejected(self):
        for key in ('environment_config', 'policy_config'):
            with self.subTest(key=key):
                prepared = copy.deepcopy(self.prepared)
                prepared[1]['protocol'][key]['p_csi'] = .1
                with self.assertRaisesRegex(ValueError, 'saved'):
                    compare.validate_cohort(self.spec, prepared)

    def test_commands_share_ids_policy_and_separate_outputs(self):
        commands = compare.commands(self.spec, device='cuda', gpu=5)
        self.assertEqual(len(commands), 3)
        outputs = set()
        for run, record in zip(self.spec['runs'], commands):
            self.assertEqual(record['run_id'], run['id'])
            self.assertEqual(len(record['evaluation']), 2)
            self.assertIn('--dry-run', record['dry_run'])
            self.assertIn('--smoke-slots 40', record['smoke'])
            self.assertIn('--episode-start 149000', record['smoke'])
            for shard, command in enumerate(record['evaluation']):
                self.assertIn('--checkpoint best', command)
                self.assertIn('--episode-start 150000 --episodes 100', command)
                self.assertIn(f'--num-shards 2 --shard-index {shard}', command)
                self.assertIn('--device cuda --gpu 5', command)
                self.assertNotIn('--smoke-slots', command)
                out = f"results/raw_phase_30_34_35/run{run['id']}/shard{shard}"
                self.assertIn(out, command)
                outputs.add(out)
            self.assertIn(f"run{run['id']}/merged", record['merge'])
        self.assertEqual(len(outputs), 6)

    def test_smoke_full_overlap_rejected(self):
        self.spec['smoke_episode_start'] = self.spec['episode_start']
        with self.assertRaisesRegex(ValueError, 'Smoke episodes'):
            compare.validate_spec(self.spec)

    def test_latest_requires_final_update_but_best_may_be_earlier(self):
        item = self.prepared[0]
        checkpoint = dict(cfg=copy.deepcopy(item['protocol']['training_config']), model={}, update=639,
                          lr_schedule=copy.deepcopy(self.spec['schedule']))
        compare.ev.validate_checkpoint(checkpoint, item)
        self.spec['checkpoint'] = 'latest'
        item['protocol']['checkpoint_kind'] = 'latest'
        with self.assertRaisesRegex(ValueError, 'completed training target'):
            compare.ev.validate_checkpoint(checkpoint, item)
        checkpoint['update'] = 999
        compare.ev.validate_checkpoint(checkpoint, item)
        checkpoint['cfg']['p_csi'] = .1
        with self.assertRaisesRegex(ValueError, 'Checkpoint configuration'):
            compare.ev.validate_checkpoint(checkpoint, item)


if __name__ == '__main__':
    unittest.main()
