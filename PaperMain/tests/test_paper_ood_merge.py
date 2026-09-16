"""Complete-grid, provenance and paired-statistic tests for the OOD merger."""
import copy
import csv
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_ood_eval as ev
import paper_ood_merge as merge
from config import phase4_queue_config


class OODMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = [self.root / 'results' / f'worker{i}' for i in (16, 21, 22)]
        cfg = phase4_queue_config()
        cfg.seed, cfg.num_ue = 2024, 32
        cfg.debug, cfg.episode_len_main = False, 1000
        cfg.n_active_min, cfg.n_active_max = 16, 32
        cfg.ue_speed_min, cfg.ue_speed_max = 5., 40.
        cfg.p_arrival_min, cfg.p_arrival_max = .15, .50
        cfg.deadline_min, cfg.deadline_max = 3, 12
        cfg.traffic_model = 'bernoulli'
        self.base = dataclasses.asdict(cfg)
        self.worlds, self.episodes = ['ID', 'D26strict', 'K60'], [130000, 130001]
        self.runs = {i: f'runs/{i}_frozen' for i in (16, 21, 22)}
        self.sources = {'policy.py': 'a' * 64}
        self.base_manifest = dict(config=self.base, recipe='base', target_updates=2000, source_sha256=self.sources)
        self.base_bytes = json.dumps(self.base_manifest, indent=2).encode()
        self.base_hashes = {self.runs[21] + '/paper_manifest.json': merge.numeric.sha_bytes(self.base_bytes),
                            self.runs[21] + '/ckpt/best.pt': 'b' * 64}
        groups = [('sus_cqi', 'su_cqi'), ('sus_pf', 'su_pf'), ('sus_random', 'su_random')]
        for path, number, baseline_keys in zip(self.paths, (16, 21, 22), groups):
            path.mkdir(parents=True)
            training = copy.deepcopy(self.base)
            if number == 22:
                training.update(n_active_min=24, n_active_max=24, p_arrival_min=.325, p_arrival_max=.325,
                                ue_speed_min=22.5, ue_speed_max=22.5)
            run_manifest = dict(config=training, recipe='narrow_mean' if number == 22 else 'base',
                                target_updates=2000, source_sha256=self.sources)
            run_bytes = json.dumps(run_manifest, indent=2).encode()
            (path / 'run_manifest.json').write_bytes(run_bytes)
            (path / 'base_manifest.json').write_bytes(self.base_bytes)
            schedulers = ['ppo', *baseline_keys]
            configurations = {}
            for world in self.worlds:
                env, policy = ev.make_configs(self.base, training, world, 32)
                configurations[world] = dict(environment=env, policy=policy, baseline_threshold=.75)
            checkpoint_hash = f'{number:064x}'
            protocol = dict(name='paper-common-bernoulli-ood-v1', mode='eval', run=self.runs[number], base_run=self.runs[21],
                training_config=training, base_config=self.base,
                checkpoint_sha256=checkpoint_hash, checkpoint_update={16:1989,21:649,22:1469}[number],
                selected_input_hashes={self.runs[number] + '/ckpt/best.pt': checkpoint_hash,
                                       self.runs[number] + '/paper_manifest.json': merge.numeric.sha_bytes(run_bytes)},
                base_input_hashes=self.base_hashes, current_source_sha256=self.sources,
                source_provenance=dict(mode='current-exact', current_source_sha256=self.sources, training_source_sha256=self.sources),
                runner_sha256={'paper_ood_eval.py': 'c' * 64}, history_audit_sha256='d' * 64,
                threshold_input_hashes={'results/thresholds.json':'e' * 64},
                worlds=self.worlds, configurations=configurations, scheduler_keys=schedulers,
                scheduler_names={key:ev.SCHEDULERS[key] for key in schedulers}, episode_ids=self.episodes,
                candidate_thresholds=None,
                thresholds=dict(status='validated', selection_metric='reward', selection_rule=ev.SELECTION_RULE,
                                diagnostic_only=True, pilot_episode_ids=[140000], thresholds={world:.75 for world in self.worlds}),
                diagnostic_only=True, smoke_slots=32)
            model_hashes = {ev.config_hash(value['policy']): f'{number+10:064x}' for value in configurations.values()}
            manifest = dict(status='completed', protocol=protocol, runtime={'python':'3.11','packages':{'numpy':'1.26'}},
                execution=dict(device='cuda',gpu={16:3,21:4,22:5}[number],threads=4),
                model_state_unchanged=True, torch_rng_unchanged=True, inputs_unchanged=True,
                model_state_sha256_before=model_hashes,model_state_sha256_after=copy.deepcopy(model_hashes),columns=ev.HEADER)
            rows, raws = [], []
            for world in self.worlds:
                env = configurations[world]['environment']
                for ep in self.episodes:
                    for key in schedulers:
                        row = {key:0 for key in ev.HEADER}
                        k = ep - self.episodes[0]
                        row.update(world=world,scheduler=key,scheduler_name=ev.SCHEDULERS[key],episode_idx=ep,
                            train_seed=2024 if key=='ppo' else '',checkpoint_update=protocol['checkpoint_update'] if key=='ppo' else '',
                            diagnostic_only=1,slots=32,mode='eval',run=Path(protocol['run']).name if key=='ppo' else '',
                            threshold='' if key=='ppo' else .75,environment_seed=2024,
                            policy_config_sha256=ev.config_hash(configurations[world]['policy']) if key=='ppo' else '',
                            n_arrivals=10,n_offered=11,n_buffer_overflow=1,n_comp=5,n_miss_deadline=3,n_retx_drop=1,
                            n_retx_overflow_drop=0,terminal_queued_packets=1,n_units_new=5+5*k,n_units_first_ack=4+3*k,
                            units_m1=5+5*k,acks_m1=4+3*k,completion_rate=.5,deadline_miss_rate=.3,retx_drop_rate=.1,
                            buffer_overflow_rate=1/11,goodput_mbps=number+k if key=='ppo' else 10+k,
                            reward=number*10+k,throughput_mbps=30,mu_depth=2,jain=.8,
                            n_active=60 if world=='K60' else 24,mean_speed_kmh=22.5,arrival_intensity_per_slot=.325,
                            mean_sinr_db='',sinr_count=0)
                        rows.append(row)
                        raws.append(dict(world=world,episode_idx=ep,scheduler=key))
            self.save(path,manifest,rows,raws)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self,path,manifest,rows,raws=None):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=ev.HEADER)
        writer.writeheader();writer.writerows(rows)
        data = stream.getvalue().encode()
        (path/'metrics.csv').write_bytes(data)
        if raws is not None:
            (path/'raw_metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in raws))
        manifest.update(metrics_sha256=merge.numeric.sha_bytes(data),raw_metrics_sha256=merge.numeric.sha_bytes((path/'raw_metrics.jsonl').read_bytes()),
                        rows_written=len(rows),total_rows=len(rows))
        (path/'manifest.json').write_text(json.dumps(manifest))

    def edit(self,index=0):
        path=self.paths[index]
        manifest=json.loads((path/'manifest.json').read_text())
        with (path/'metrics.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        return path,manifest,rows

    def test_complete_smoke_grid_preserves_source_and_renames_policies(self):
        protocol,rows,inputs,runtime=merge.load_inputs(self.paths,True)
        self.assertEqual(len(rows),54)
        self.assertEqual(protocol['scheduler_keys'][:3],['ppo_run16','ppo_run21','ppo_run22'])
        self.assertEqual([item['run_id'] for item in inputs],[16,21,22])
        self.assertEqual(rows[0]['source_run'],'runs/16_frozen')
        self.assertEqual(rows[3]['source_checkpoint_sha256'],'')

    def test_requires_explicit_smoke_and_all_three_workers(self):
        with self.assertRaisesRegex(ValueError,'allow-smoke'):
            merge.load_inputs(self.paths)
        with self.assertRaisesRegex(ValueError,'exactly three'):
            merge.load_inputs(self.paths[:2],True)

    def test_missing_and_duplicate_rows_rejected_even_after_rehash(self):
        path,manifest,rows=self.edit()
        rows[-1]=rows[0]
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            merge.load_inputs(self.paths,True)

    def test_missing_grid_row_rejected_even_with_updated_completion_count(self):
        path,manifest,rows=self.edit()
        self.save(path,manifest,rows[:-1])
        with self.assertRaisesRegex(ValueError,'Missing rows'):
            merge.load_inputs(self.paths,True)

    def test_unfinished_and_mutated_input_proof_rejected(self):
        path,manifest,rows=self.edit()
        manifest['inputs_unchanged']=False
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'preservation proof'):
            merge.load_inputs(self.paths,True)

    def test_model_state_hash_map_must_cover_k_configs(self):
        path,manifest,rows=self.edit()
        manifest['model_state_sha256_before'].pop(next(iter(manifest['model_state_sha256_before'])))
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'preservation proof'):
            merge.load_inputs(self.paths,True)

    def test_deadline_preprocessing_drift_rejected(self):
        path,manifest,rows=self.edit()
        manifest['protocol']['configurations']['D26strict']['policy']['deadline_max']=6
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'preprocessing'):
            merge.load_inputs(self.paths,True)

    def test_common_history_and_runtime_must_match(self):
        path,manifest,rows=self.edit(1)
        manifest['protocol']['history_audit_sha256']='f'*64
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'common environment'):
            merge.load_inputs(self.paths,True)

    def test_pilot_overlap_rejected(self):
        path,manifest,rows=self.edit()
        manifest['protocol']['thresholds']['pilot_episode_ids']=[self.episodes[0]]
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'overlap'):
            merge.load_inputs(self.paths,True)

    def test_checkpoint_metadata_and_baseline_blanks_enforced(self):
        path,manifest,rows=self.edit()
        rows[0]['checkpoint_update']='1988'
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'checkpoint/threshold'):
            merge.load_inputs(self.paths,True)

    def test_count_rate_consistency_and_nan_rejected(self):
        path,manifest,rows=self.edit()
        rows[0]['deadline_miss_rate']='0.2'
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'raw counts'):
            merge.load_inputs(self.paths,True)

    def test_initial_episode_conditions_equal_across_workers(self):
        path,manifest,rows=self.edit()
        rows[0]['n_active']='25'
        self.save(path,manifest,rows)
        with self.assertRaisesRegex(ValueError,'initial episode condition'):
            merge.load_inputs(self.paths,True)

    def test_raw_hash_and_copied_manifest_hash_enforced(self):
        (self.paths[0]/'raw_metrics.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            merge.load_inputs(self.paths,True)

    def test_statistics_are_paired_and_first_ack_is_pooled(self):
        protocol,rows,inputs,runtime=merge.load_inputs(self.paths,True)
        summary,means,paired=merge.build_summary(protocol,rows,inputs,runtime)
        self.assertEqual(len(means),3*9*9)
        self.assertEqual(len(paired),3*21*9)
        difference=summary['by_world']['ID']['paired_differences']['ppo_run16_minus_sus_cqi']['metrics']['goodput_mbps']
        self.assertEqual(difference['mean_difference'],6)
        self.assertEqual(difference['ci95'],[6,6])
        pooled=summary['by_world']['ID']['by_scheduler']['ppo_run16']['first_ack_by_depth']
        self.assertAlmostEqual(pooled['1']['rate'],11/15)
        self.assertIsNone(pooled['2']['rate'])
        result=merge.estimate([1,2,3])
        self.assertAlmostEqual(result['mean'],2)
        self.assertAlmostEqual(result['ci95'][1],4.4841377117)
        self.assertIsNone(merge.estimate([2])['ci95'])

    def test_writes_requested_artifacts_only_after_validation(self):
        out=self.root/'results'/'merged'
        summary=merge.merge(self.paths,out,True,self.root)
        self.assertEqual(summary['rows'],54)
        self.assertEqual({path.name for path in out.iterdir()},
            {'metrics.csv','means_summary.csv','paired_statistics.csv','summary.json','summary.md'})
        with self.assertRaisesRegex(ValueError,'overwrite'):
            merge.merge(self.paths,out,True,self.root)
        with self.assertRaisesRegex(ValueError,'below PaperMain/results'):
            merge.merge(self.paths,self.root/'outside',True,self.root)


if __name__=='__main__':
    unittest.main()
