"""Provider and agent outcomes retain their meaning when a job is resumed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import catalog
import jobs


class JobResumeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = next(row for row in catalog.load_catalog() if row['id'] == 'TC-FT-001')

    def setUp(self):
        backoff = patch('jobs.PROVIDER_BACKOFF_BASE', 0)
        backoff.start()
        self.addCleanup(backoff.stop)
        # Plans skip cases that need more cores than the host has; a GitHub runner has only 4.
        cpus = patch('jobs.os.cpu_count', return_value=64)
        cpus.start()
        self.addCleanup(cpus.stop)
        discovery = patch('jobs.scheduler_docker_info', return_value=None)
        discovery.start()
        self.addCleanup(discovery.stop)

    def test_normal_exit_and_resume_preserve_provider_agent_and_deferred_outcomes(self):
        # Transient failures exhaust one retry; HTTP 402 and authentication failures stop immediately.
        cases = [
            ('http402', 'provider_quota', 'agent_run.json', True, False, 'provider_error', 1),
            ('http429', 'provider_rate_limit', 'agent_run.json', True, False, 'provider_error', 2),
            ('http500', 'provider_unavailable', 'agent_run.json', True, False, 'provider_error', 2),
            ('preflight401', 'provider_auth', 'preflight.json', False, False, 'provider_error', 1),
            ('preflight500', 'provider_unavailable', 'preflight.json', False, False, 'provider_error', 2),
            ('agent_evaluated', 'agent_error', 'agent_run.json', True, False, 'completed', 1),
            ('agent_failed', 'agent_error', 'agent_run.json', False, False, 'failed', 1),
            ('success', None, 'agent_run.json', True, False, 'completed', 1),
            ('deferred', None, 'agent_run.json', False, True, 'awaiting_judge', 1),
        ]
        for name, kind, filename, evaluate, deferred, expected, attempts in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                document = jobs.plan([self.row], ['mock'], [None], inventory=[])
                document.update(id=name, suite='all@2.2.0')
                path = root / 'jobs' / name / 'job.json'
                launched = []

                def popen(command, cwd, stdout, stderr):
                    run = root / 'runs' / 'mock' / self.row['category'] / self.row['id'] / command[command.index('--run-id') + 1]
                    run.mkdir(parents=True)
                    record = {'error_type': kind}
                    if filename == 'agent_run.json':
                        record.update(status='failed' if kind else 'succeeded', evaluation_deferred=deferred)
                    (run / filename).write_text(json.dumps(record))
                    if evaluate:
                        (run / 'evaluation.json').write_text(json.dumps({
                            'verdict': 'submission_invalid' if kind else 'evaluated', 'scorable': False}))
                    launched.append(run)
                    return SimpleNamespace(poll=lambda: int(bool(kind)), pid=None)

                with patch('catalog.load_catalog', return_value=[self.row]), \
                        patch('jobs.gpu_inventory', return_value=[]), patch('jobs.os.cpu_count', return_value=64), \
                        patch('jobs.time.sleep'), patch('jobs.subprocess.Popen', side_effect=popen):
                    jobs.execute(document, path, root=root, retries=1)
                entry = document['entries'][0]
                self.assertEqual(len(launched), attempts)
                self.assertEqual(entry['state'], expected)
                self.assertEqual(entry['error_type'], kind)
                before = {p: p.read_bytes() for run in launched for p in run.iterdir()}
                run_id, retry_count = entry['run'], entry['retries']

                # Resume the JSON that was actually persisted, without repeating paid work or changing artifacts.
                resumed = json.loads(path.read_text())
                with patch('catalog.load_catalog', return_value=[self.row]), \
                        patch('jobs.gpu_inventory', return_value=[]), patch('jobs.subprocess.Popen') as launch:
                    jobs.execute(resumed, path, root=root, retries=1)
                    launch.assert_not_called()
                entry = resumed['entries'][0]
                self.assertEqual((entry['state'], entry['error_type'], entry['run'], entry['retries']),
                                 (expected, kind, run_id, retry_count))
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_judge_infrastructure_failure_judges_the_collected_submission_again(self):
        failed = {'verdict': 'judge_error', 'scorable': False, 'error': 'Docker archive transfer timed out after 120s',
                  'failure': {'phase': 'judge', 'exception_type': 'DockerError', 'kind': 'judge_infrastructure',
                              'retryable': True}}
        for recovers, expected, judged in ((True, 'completed', 1), (False, 'judge_failed', 2)):
            with self.subTest(recovers=recovers), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                document = jobs.plan([self.row], ['mock'], [None], inventory=[])
                document.update(id='judge', suite='all@2.2.0')
                commands = []

                def popen(command, cwd, stdout, stderr):
                    commands.append(command)
                    if 'judge' in command:
                        if recovers:
                            run = Path(command[command.index('--run') + 1])
                            (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated', 'scorable': True}))
                        return SimpleNamespace(poll=lambda: 0 if recovers else 1, pid=None)
                    run = root / 'runs' / 'mock' / self.row['category'] / self.row['id'] / command[command.index('--run-id') + 1]
                    run.mkdir(parents=True)
                    (run / 'agent_run.json').write_text(json.dumps({'status': 'succeeded', 'error_type': None}))
                    (run / 'evaluation.json').write_text(json.dumps(failed))
                    return SimpleNamespace(poll=lambda: 1, pid=None)

                with patch('catalog.load_catalog', return_value=[self.row]), \
                        patch('jobs.gpu_inventory', return_value=[]), patch('jobs.os.cpu_count', return_value=64), \
                        patch('jobs.time.sleep'), patch('jobs.subprocess.Popen', side_effect=popen):
                    jobs.execute(document, root / 'jobs/judge/job.json', root=root, retries=2)
                entry = document['entries'][0]
                judges = [c for c in commands if 'judge' in c]
                self.assertEqual(len(commands) - len(judges), 1)           # the agent is never run again
                self.assertEqual(len(judges), judged)
                self.assertTrue(all(c[c.index('--run') + 1] == entry['run'] for c in judges))
                self.assertEqual((entry['state'], entry['judge_retries']), (expected, judged))

    def test_resume_recovers_provider_failure_when_scheduler_missed_its_exit(self):
        for state in ('running', 'pending', 'completed'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                document = jobs.plan([self.row], ['mock'], [None], inventory=[])
                document.update(id='missed-exit', suite='all@2.2.0')
                entry = document['entries'][0]
                run = jobs.run_path(root, document, entry)
                run.mkdir(parents=True)
                (run / 'agent_run.json').write_text(json.dumps({'status': 'failed', 'error_type': 'provider_quota'}))
                (run / 'evaluation.json').write_text(json.dumps({'verdict': 'submission_invalid', 'scorable': False}))
                entry.update(state=state, run=str(run), pid=None)
                with patch('catalog.load_catalog', return_value=[self.row]), \
                        patch('jobs.gpu_inventory', return_value=[]), patch('jobs.subprocess.Popen') as launch:
                    jobs.execute(document, root / 'jobs/missed-exit/job.json', root=root)
                    launch.assert_not_called()
                self.assertEqual(entry['state'], 'provider_error')
                self.assertEqual(entry['error_type'], 'provider_quota')
                self.assertNotIn('restarts', entry)


if __name__ == '__main__':
    unittest.main()
