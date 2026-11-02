"""Guard modes which must never become scored, even with otherwise valid image evidence."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import oj_eval

class ScoreabilityTest(unittest.TestCase):
    def test_smoke_does_not_require_a_judge_image_or_decrypt(self):
        with tempfile.TemporaryDirectory() as raw, \
             mock.patch.object(oj_eval,'load_oj',return_value={'judge':{'kind':'harness'}}), \
             mock.patch.object(oj_eval,'validate_submission_dir',return_value=[]), \
             mock.patch('judge_container.DockerStageLauncher.for_case',side_effect=AssertionError('smoke must not start Docker')):
            result=oj_eval.evaluate('TC-FT-001',Path(raw),None,'smoke',backend='docker')
        self.assertEqual(result['verdict'],'smoke_unscored')
        self.assertFalse(result['scorable'])
        self.assertEqual(result['judge_isolation'], 'not_started')
        self.assertEqual(result['judge_backend_requested'], 'docker')

    def test_invalid_submission_does_not_claim_host_execution(self):
        launcher = SimpleNamespace(image='image', locked=False)
        with tempfile.TemporaryDirectory() as raw, \
             mock.patch.object(oj_eval,'load_oj',return_value={'judge':{'kind':'harness'}}), \
             mock.patch.object(oj_eval,'validate_submission_dir',return_value=['invalid report']), \
             mock.patch.object(oj_eval,'_environment_check',return_value={'matches':False}), \
             mock.patch('judge_container.DockerStageLauncher.for_case',return_value=launcher), \
             mock.patch.object(oj_eval,'evaluate_harness') as execute:
            result = oj_eval.evaluate('TC-FT-001',Path(raw),None,backend='docker')
        self.assertEqual(result['verdict'], 'submission_invalid')
        self.assertEqual(result['judge_isolation'], 'not_started')
        self.assertEqual(result['judge_backend_requested'], 'docker')
        execute.assert_not_called()

    def test_pipeline_calibration_is_excluded_even_with_trusted_images(self):
        launcher=SimpleNamespace(image='image@sha256:'+'a'*64,locked=True)
        with tempfile.TemporaryDirectory() as raw, \
             mock.patch.object(oj_eval,'load_oj',return_value={'judge':{'kind':'pending'},'resources':{'leaderboard':'none'}}), \
             mock.patch.object(oj_eval,'validate_submission_dir',return_value=[]), \
             mock.patch.object(oj_eval,'_dependency_attestation',return_value={'trusted':True}), \
             mock.patch.object(oj_eval,'_environment_check',return_value={'matches':True}), \
             mock.patch('judge_container.DockerStageLauncher.for_case',return_value=launcher):
            result=oj_eval.evaluate('TC-FT-001',Path(raw),None,backend='docker')
        self.assertFalse(result['scorable'])
        self.assertIn('pipeline calibration',result['scoreability_reason'])


class BudgetAttestationTest(unittest.TestCase):
    """A budgeted resource that was not metered, or Pro H100 use, can never pass as within budget."""
    def test_pro_unmetered_and_cpu_budgets(self):
        from runtime.process import budget_check
        metered = lambda hours: {'metered': True, 'gpu_busy_hours': hours}
        pro = {'h100_hours': 96.0, 'session_hours': 24.0}
        self.assertIs(budget_check(pro, metered(9999), None)['over_budget'], True)
        self.assertIs(budget_check(pro, metered(90), None)['over_budget'], False)
        self.assertIsNone(budget_check(pro, {'metered': False}, None)['over_budget'])
        self.assertIsNone(budget_check({'gpu_hours': 4}, {'metered': False}, 0.0)['over_budget'])
        self.assertIsNone(budget_check({'cpu_core_hours': 2}, {}, None)['over_budget'])
        self.assertIs(budget_check({'cpu_core_hours': 2}, {}, 3600.0)['over_budget'], False)
        self.assertIs(budget_check({'cpu_core_hours': 2}, {}, 9 * 3600.0)['over_budget'], True)
        self.assertIs(budget_check({'session_hours': 6}, {}, None)['over_budget'], False)

    def test_meter_requires_coverage_and_a_working_final_sample(self):
        from runtime.process import GpuMeter
        for failed, final_ok, expected in ((0, True, True), (1, True, False), (0, False, False)):
            meter = GpuMeter(interval=3600, gpu_ids='0')
            meter.available = True
            calls = iter([[(0.0, 10.0)], [(0.0, 10.0)] if final_ok else ValueError('no GPU')])
            def read():
                value = next(calls)
                if isinstance(value, Exception):
                    raise value
                return value
            meter._read = read
            meter.start()
            meter.samples, meter.failed = 10, failed
            self.assertEqual(meter.stop()['metered'], expected, (failed, final_ok))

    def test_scoring_names_exceeded_and_unattested_budgets(self):
        from agents.scoring import apply_agent_attestation
        for over, reason in ((True, 'development budget exceeded'), (None, 'development budget not attested')):
            result = apply_agent_attestation({'scorable': True}, {'budget': {'over_budget': over}})
            self.assertFalse(result['scorable'])
            self.assertIn(reason, result['scoreability_reason'])


class AgentExitTest(unittest.TestCase):
    """Timeouts keep the container for cgroup accounting; an early SIGKILL (e.g. OOM) is a failure, not a timeout."""
    def test_exit_classification(self):
        from runtime.agent import DockerAgentEnvironment
        env = DockerAgentEnvironment.__new__(DockerAgentEnvironment)
        env.docker = mock.Mock(executable='docker')
        env.cid, env.host_paths = 'cid', {'workspace': Path('.')}
        launch = SimpleNamespace(argv=['agent'], stdin=b'', env={})
        for outcome, elapsed, expected in ((('failed', 137), 1, ('failed', 137)), (('failed', 137), 100, ('timeout', None)),
                                           (('failed', 124), 1, ('timeout', None)), (('timeout', None), 107, ('timeout', None)),
                                           (('succeeded', 0), 1, ('succeeded', 0))):
            clock = iter([0.0, float(elapsed)])
            with mock.patch('sandbox_runner.run_agent', return_value=outcome), \
                    mock.patch('runtime.agent.time.monotonic', side_effect=lambda: next(clock)):
                self.assertEqual(env.execute(launch, {}, 100, None, None), expected)
        env.docker.kill.assert_not_called()


class HarnessIdentityAttestationTest(unittest.TestCase):
    def test_real_harnesses_need_matching_verified_cli_identity(self):
        from agents import HARNESSES
        from agents.scoring import apply_agent_attestation
        record = {'run_mode': 'benchmark', 'isolation': 'container', 'status': 'succeeded',
                  'network': {'mode': 'none'}, 'budget': {'over_budget': False},
                  'sandbox': {'backend': 'docker', 'verified': True, 'resources_enforced': True,
                              'platform': 'linux/amd64', 'image_attestation': {'locked': True},
                              'harness_bundle': {'locked': True, 'lock': 'local'}, 'hardware_matches': True}}
        for harness in HARNESSES:
            record['harness'] = harness
            for proof in (None, {'verified': True, 'requested_version': '1', 'actual_version': '0'},
                          {'verified': False, 'requested_version': '1', 'actual_version': '1'}):
                record['sandbox']['harness_cli_version'] = proof
                result = apply_agent_attestation({'scorable': True}, record)
                self.assertFalse(result['scorable'])
                self.assertIn('actual harness CLI version not verified', result['scoreability_reason'])
            record['sandbox']['harness_cli_version'] = {'verified': True, 'requested_version': '1', 'actual_version': '1'}
            self.assertTrue(apply_agent_attestation({'scorable': True}, record)['scorable'])
            # the CLI must come from the bundle locked on this host
            record['sandbox']['harness_bundle'] = {'locked': False}
            result = apply_agent_attestation({'scorable': True}, record)
            self.assertIn('harness CLI bundle is not the one locked', result['scoreability_reason'])
            record['sandbox']['harness_bundle'] = {'locked': True, 'lock': 'local'}
            record['sandbox']['cleanup'] = {'remaining': {'containers': ['cid'], 'volumes': []}}
            result = apply_agent_attestation({'scorable': True}, record)
            self.assertFalse(result['scorable'])
            self.assertIn('cleanup is incomplete', result['scoreability_reason'])
            del record['sandbox']['cleanup']
