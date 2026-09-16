"""Focused recovery tests; no server, GPU, checkpoints, or real child processes."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

source = Path(__file__).with_name('recover_seed_replicates_20260915.py')
spec = importlib.util.spec_from_file_location('seed_recovery', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {'ppo_eval_every': 10, 'ppo_eval_episodes': 3, 'num_ue': 2}

    def test_exact_csv_prefix_preserves_bytes(self):
        data = b'update,reward\r\n' + b''.join(f'{u},1.25\r\n'.encode() for u in range(20))
        output, report = m.csv_prefix(data, 19, 'env_metrics.csv', self.cfg)
        self.assertEqual(output, data)
        self.assertEqual(report['removed_rows'], 0)

    def test_uncommitted_complete_and_partial_tail_only(self):
        prefix = b'update,reward\n0,1\n1,2\n'
        for tail in (b'2,3\n3,4\n', b'2,3\n3,', b'2,'):
            self.assertEqual(m.csv_prefix(prefix + tail, 1, 'env_metrics.csv', self.cfg)[0], prefix)

    def test_missing_or_duplicate_committed_rows_rejected(self):
        for data in (b'update,reward\n0,1\n2,2\n', b'update,reward\n0,1\n0,1\n1,2\n', b'update,reward\n0,1\n1,'):
            with self.assertRaises(ValueError):
                m.csv_prefix(data, 1, 'env_metrics.csv', self.cfg)

    def test_baseline_and_ppo_coverage(self):
        data = b'update,scheduler,reward\n9,PPO,1\n'
        data += ''.join(f'9,{name},0.5\n' for name in sorted(m.BASELINES)).encode()
        data += b'19,PPO,2\n'
        self.assertEqual(m.csv_prefix(data, 19, 'eval_metrics.csv', self.cfg)[0], data)
        with self.assertRaises(ValueError):
            m.csv_prefix(data.replace(b'19,PPO,2\n', b''), 19, 'eval_metrics.csv', self.cfg)
        with self.assertRaises(ValueError):
            m.csv_prefix(data + b'9,SU+PF,0.5\n', 19, 'eval_metrics.csv', self.cfg)

    def test_perue_exact_episode_user_grid(self):
        data = b'update,ep_idx,ue\n' + ''.join(f'{u},{e},{k}\n' for u in (9, 19) for e in range(3) for k in range(2)).encode()
        self.assertEqual(m.csv_prefix(data, 19, 'per_ue_metrics.csv', self.cfg)[0], data)
        with self.assertRaises(ValueError):
            m.csv_prefix(data.replace(b'19,2,1\n', b''), 19, 'per_ue_metrics.csv', self.cfg)

    def test_live_pid_rejected_without_killing(self):
        with patch.object(m.os, 'kill') as check:
            with self.assertRaises(ValueError):
                m.dead_pids([1234])
            check.assert_called_once_with(1234, 0)
        with patch.object(m.os, 'kill', side_effect=ProcessLookupError):
            m.dead_pids([1234])

    def test_unchanged_csv_not_rewritten_and_changed_candidate_rejected(self):
        src = self.root / 'metrics.csv'
        src.write_bytes(b'update,reward\n0,1\n')
        original_mtime = src.stat().st_mtime_ns
        candidate = self.root / 'candidate.csv'
        candidate.write_bytes(src.read_bytes())
        archive = self.root / 'archive.csv'
        archive.write_bytes(src.read_bytes())
        record = {'candidate': str(candidate), 'prefix_sha256': m.l.sha(candidate), 'changed': False}
        plan = {'original_sha256': {str(src): m.l.sha(src)}, 'archive_sha256': {str(archive): m.l.sha(archive)},
                'items': [{'csv': {str(src): record}}]}
        m.apply_csv_prefixes(plan)
        self.assertEqual(src.stat().st_mtime_ns, original_mtime)
        candidate.write_text('tampered')
        with self.assertRaises(ValueError):
            m.apply_csv_prefixes(plan)
        self.assertEqual(src.stat().st_mtime_ns, original_mtime)

    def queue_fixture(self):
        campaign = self.root / 'results/campaign'
        recovery = campaign / 'recovery'
        recovery.mkdir(parents=True)
        (self.root / 'review/logs').mkdir(parents=True)
        jobs = []
        for n, seed in ((23, 1002024), (26, 2002024)):
            name = f'{n}_base_s{seed}_gpu3'
            job = dict(name=name, gpu=3, seed=seed, target_updates=1000, recipe='base', entry='paper_train.py',
                       calibration_profile='results/calibration.json', calibration_reference_root='provenance/frozen',
                       run_dir=str(self.root / 'runs' / name))
            job['command'] = m.l.command(job)
            folder = campaign / 'jobs' / name
            folder.mkdir(parents=True)
            state = {'status': 'recovering' if n == 23 else 'queued', 'name': name, 'run_dir': job['run_dir']}
            m.l.atomic_json(folder / 'status.json', state)
            if n == 23:
                Path(job['run_dir']).mkdir(parents=True)
                (folder / 'console.log').write_text('original log\n')
            jobs.append(job)
        prepared = {'root': str(self.root), 'campaign_dir': str(campaign), 'jobs': jobs, 'python': m.sys.executable}
        first = jobs[0]
        item = {'name': first['name'], 'gpu': 3, 'resume_command': m.resume_command(prepared, first),
                'resume_input_sha256': {}, 'checkpoint': {'resume_update': 889}}
        plan = {'recovery_dir': str(recovery), 'recovery_launcher_sha256': m.l.sha(source), 'items': [item]}
        return plan, prepared

    def exercise_queue(self, code=0, verification_failure=False):
        plan, prepared = self.queue_fixture()
        calls = []
        verified = []
        root = self.root
        class Child:
            pid = 321
            def __init__(self, cmd, **kwargs):
                calls.append(cmd)
                if '--name' in cmd:
                    (root / 'runs' / cmd[cmd.index('--name') + 1]).mkdir()
                self.returncode = code
            def poll(self):
                return self.returncode
        def verify(p, j, result):
            verified.append(j['name'])
            m.l.require(result == 0 and not verification_failure, 'failed completion')
            return {'checkpoint_update': 999}
        patches = (patch.object(m, 'ROOT', self.root), patch.object(m, 'load_plan', return_value=(plan, prepared)),
                   patch.object(m.l, 'verify_static'), patch.object(m.l, 'free_gpu'), patch.object(m.l, 'link_outputs'),
                   patch.object(m.l, 'verify_completion', side_effect=verify), patch.object(m.signal, 'signal'),
                   patch.object(m.subprocess, 'Popen', Child))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            if code or verification_failure:
                with self.assertRaises(ValueError):
                    m.supervise(Path(plan['recovery_dir']) / 'plan.json', 'sha', 3)
            else:
                m.supervise(Path(plan['recovery_dir']) / 'plan.json', 'sha', 3)
                # A second invocation must not rewrite completed records.
                with self.assertRaises(FileExistsError):
                    m.supervise(Path(plan['recovery_dir']) / 'plan.json', 'sha', 3)
        statuses = [m.l.read(m.l.state_path(prepared, j))['status'] for j in prepared['jobs']]
        return plan, prepared, calls, verified, statuses

    def test_resume_then_original_second_command_and_no_relaunch(self):
        plan, prepared, calls, verified, statuses = self.exercise_queue()
        self.assertEqual(len(calls), 2)
        self.assertIn('--resume', calls[0])
        self.assertNotIn('--name', calls[0])
        self.assertEqual(calls[1], prepared['jobs'][1]['command'])
        self.assertEqual(verified, [j['name'] for j in prepared['jobs']])
        self.assertEqual(statuses, ['completed', 'completed'])
        self.assertIn('started_at', m.l.read(m.l.state_path(prepared, prepared['jobs'][1])))
        text = (m.l.state_path(prepared, prepared['jobs'][0]).parent / 'console.log').read_text()
        self.assertTrue(text.startswith('original log\n'))
        self.assertIn('Resume from update 889', text)
        mirror = self.root / 'review/logs' / (prepared['jobs'][0]['name'] + '_launch.json')
        self.assertEqual(m.l.read(mirror)['checkpoint_update'], 999)
        self.assertFalse(mirror.is_symlink())

    def test_partial_dispatch_marks_only_undispatched_jobs(self):
        _, prepared = self.queue_fixture()
        m.mark_undispatched(prepared, {'3': 1234}, RuntimeError('dispatch'))
        self.assertEqual([m.l.read(m.l.state_path(prepared, j))['status'] for j in prepared['jobs']], ['recovering', 'queued'])
        m.mark_undispatched(prepared, {}, RuntimeError('dispatch'))
        self.assertEqual([m.l.read(m.l.state_path(prepared, j))['status'] for j in prepared['jobs']], ['failed', 'blocked'])

    def test_resume_failure_blocks_queued_seed(self):
        _, _, calls, _, statuses = self.exercise_queue(code=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(statuses, ['failed', 'blocked'])

    def test_checkpoint_verification_failure_blocks_queued_seed(self):
        _, _, calls, _, statuses = self.exercise_queue(verification_failure=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(statuses, ['failed', 'blocked'])


if __name__ == '__main__':
    unittest.main()
