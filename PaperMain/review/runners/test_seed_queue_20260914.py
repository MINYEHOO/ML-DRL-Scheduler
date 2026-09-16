"""No GPU or training: exercise guards and fail-stop dispatch mechanics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

source = Path(__file__).with_name('launch_seed_replicates_20260914.py')
spec = importlib.util.spec_from_file_location('seed_launcher', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Queues(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        jobs = []
        for seed_i, seed in enumerate(m.SEEDS):
            for gpu in m.GPUS:
                recipe, entry, target = m.EXPECTED[gpu]
                ref = '21_reference' if recipe == 'base' else '22_reference'
                (self.root / 'runs' / ref).mkdir(parents=True, exist_ok=True)
                jobs.append(dict(name=f'{23+seed_i*3+gpu-3}_s{seed}_gpu{gpu}', gpu=gpu,
                                 seed=seed, recipe=recipe, entry=entry, target_updates=target,
                                 reference_run=ref, calibration_profile='results/calibration.json',
                                 calibration_reference_root='provenance/frozen'))
        self.spec = dict(root=str(self.root), campaign_dir='results/campaign', jobs=jobs)

    def prepared(self):
        campaign = self.root / 'results/campaign'
        campaign.mkdir(parents=True)
        (self.root / 'review/logs').mkdir(parents=True)
        jobs = []
        for j in self.spec['jobs']:
            job = dict(j, command=m.command(j), run_dir=str(self.root / 'runs' / j['name']), expected_manifest={})
            folder = campaign / 'jobs' / j['name']
            folder.mkdir(parents=True)
            m.atomic_json(folder / 'status.json', {'status': 'queued'})
            jobs.append(job)
        result = dict(schema_version=1, root=str(self.root), campaign_dir=str(campaign),
                      launcher=str(source.resolve()), python=m.sys.executable, jobs=jobs,
                      spec=self.spec, protected_sha256={})
        path = campaign / 'prepared.json'
        m.atomic_json(path, result)
        return result, path

    def test_exact_six_unique_and_allowed_assignment(self):
        m.validate_spec(self.spec, self.root)
        for mutate in (lambda s: s['jobs'][0].update(gpu=2),
                       lambda s: s['jobs'][0].update(target_updates=2000),
                       lambda s: s['jobs'][1].update(name=s['jobs'][0]['name']),
                       lambda s: s['jobs'][0].update(seed=2024),
                       lambda s: s['jobs'].reverse()):
            s = copy.deepcopy(self.spec)
            mutate(s)
            with self.assertRaises(ValueError):
                m.validate_spec(s, self.root)

    def test_existing_run_and_symlink_rejected(self):
        run = self.root / 'runs' / self.spec['jobs'][0]['name']
        run.mkdir()
        with self.assertRaises(ValueError):
            m.validate_spec(self.spec, self.root)
        run.rmdir()
        run.symlink_to(self.root / 'runs' / '21_reference', target_is_directory=True)
        with self.assertRaises(ValueError):
            m.validate_spec(self.spec, self.root)

    def test_gpu_busy_never_kills(self):
        with patch.object(m.subprocess, 'check_output', return_value='12345') as query:
            with self.assertRaises(ValueError):
                m.free_gpu(3)
        self.assertEqual(query.call_count, 1)

    def test_hash_change_blocks(self):
        f = self.root / 'protected.py'
        f.write_text('original')
        p = {'protected_sha256': {str(f): m.sha(f)}}
        m.verify_static(p)
        f.write_text('changed')
        with self.assertRaises(ValueError):
            m.verify_static(p)

    def run_queue(self, code=0, verification_failure=False):
        prepared, path = self.prepared()
        calls = []
        class Child:
            pid = 777
            def __init__(self, command, **kwargs):
                name = command[command.index('--name') + 1]
                calls.append(name)
                (self_root / 'runs' / name).mkdir()
                self.returncode = code
            def poll(self):
                return self.returncode
        self_root = self.root
        verified = []
        def verify(p, j, returncode):
            verified.append(j['name'])
            m.require(returncode == 0, 'failed child')
            m.require(not verification_failure, 'bad checkpoint')
            return {'checked': True, 'checkpoint_update': j['target_updates'] - 1}
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'free_gpu'), \
                patch.object(m.subprocess, 'Popen', Child), patch.object(m, 'verify_completion', verify), \
                patch.object(m.signal, 'signal'):
            if code or verification_failure:
                with self.assertRaises(ValueError):
                    m.supervise(path, 3)
            else:
                m.supervise(path, 3)
        statuses = [m.read(m.state_path(prepared, j))['status'] for j in prepared['jobs'] if j['gpu'] == 3]
        return prepared, path, calls, statuses, verified

    def test_queue_advances_only_after_completion_verification(self):
        prepared, path, calls, statuses, verified = self.run_queue()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, verified)
        self.assertEqual(statuses, ['completed', 'completed'])
        first = prepared['jobs'][0]
        run = Path(first['run_dir'])
        self.assertTrue((run / 'eval.csv').is_symlink())
        self.assertEqual((run / 'console.log').resolve(), m.state_path(prepared, first).parent / 'console.log')
        # Accidental supervisor invocation cannot restart completed jobs.
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'free_gpu'), \
                patch.object(m.signal, 'signal'), patch.object(m.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                m.supervise(path, 3)
            spawn.assert_not_called()
        self.assertEqual(m.read(m.state_path(prepared, first))['status'], 'completed')
        conventional = self.root / 'review/logs' / (first['name'] + '_launch.json')
        self.assertFalse(conventional.is_symlink())
        state = m.read(conventional)
        self.assertEqual(state['checkpoint_update'], first['target_updates'] - 1)
        self.assertTrue(state['target_completed'])
        self.assertEqual(state['returncode'], 0)

    def test_failed_child_blocks_second_job(self):
        _, _, calls, statuses, _ = self.run_queue(code=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(statuses, ['failed', 'blocked'])

    def test_bad_checkpoint_blocks_second_job(self):
        _, _, calls, statuses, _ = self.run_queue(verification_failure=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(statuses, ['failed', 'blocked'])

    def test_source_failure_prevents_any_gpu_process(self):
        prepared, path = self.prepared()
        prepared['protected_sha256'] = {str(self.root / 'missing'): 'x'}
        m.atomic_json(path, prepared)
        with patch.object(m, 'ROOT', self.root), patch.object(m.subprocess, 'Popen') as spawn:
            with self.assertRaises(FileNotFoundError):
                m.launch(path)
            spawn.assert_not_called()
        self.assertFalse((path.parent / 'launch.json').exists())

    def test_best_checkpoint_provenance_and_finiteness(self):
        class Finite:
            def __init__(self, value):
                self.value = value
            def all(self):
                return self.value
        class Torch:
            @staticmethod
            def isfinite(value):
                return Finite(m.math.isfinite(value))
        expected = {'config': {'seed': 1002024}, 'schedule': {'lr_final': 0, 'lr_decay_updates': 1000}}
        best = {'update': 19, 'cfg': expected['config'], 'lr_schedule': expected['schedule'],
                'model': {'ret_count': 20, 'weight': 0.5}, 'eval_reward': 3.0}
        m.verify_best(best, expected, [9, 19], lambda x: x, Torch)
        for key, value in [('update', 18), ('cfg', {'seed': 2024}), ('eval_reward', float('nan')),
                           ('model', {'ret_count': 20, 'weight': float('inf')})]:
            changed = copy.deepcopy(best)
            changed[key] = value
            with self.assertRaises(ValueError):
                m.verify_best(changed, expected, [9, 19], lambda x: x, Torch)

    def test_progress_uses_ppo_rows_and_link_refuses_overwrite(self):
        prepared, _ = self.prepared()
        job = prepared['jobs'][0]
        run = Path(job['run_dir'])
        (run / 'csv_logs').mkdir(parents=True)
        (run / 'csv_logs/eval_metrics.csv').write_text('update,scheduler,reward\n9,PPO,1\n99,SUS,2\n')
        self.assertEqual(m.progress(run), {'eval_last_update': 9})
        (run / 'eval.csv').write_text('existing result')
        with self.assertRaises(ValueError):
            m.link_outputs(prepared, job)
        self.assertEqual((run / 'eval.csv').read_text(), 'existing result')


if __name__ == '__main__':
    unittest.main()
