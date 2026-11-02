"""Stopped collection can survive process loss without repeating execution or trusting partial transfers."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from agents import lifecycle, recovery
from agents.runner import _recovery_credentials, recover
from runtime.agent import DockerAgentEnvironment
from runtime.transport import Docker, DockerError, DockerPathMissing


class FakeDocker:
    def __init__(self):
        self.labels = {'abench.run': 'owned', 'abench.run_id': 'retained', 'abench.case': 'TC-FT-001'}
        self.fail, self.running, self.foreign = True, False, False
        self.transfers, self.removed, self.pending_containers = 0, [], []
    def info(self): return {'raw': {'ID': 'original-daemon'}}
    def inspect(self, identifier):
        if identifier in self.removed:
            raise DockerError('No such object')
        labels = dict(self.labels, **({'abench.run': 'foreign'} if self.foreign else {}))
        return {'State': {'Running': self.running}, 'Config': {'Labels': labels}, 'Labels': labels}
    def download(self, container, source, destination, **limits):
        self.transfers += 1
        if source.endswith('.abench'):
            raise DockerPathMissing('absent')
        if self.fail:
            (destination / 'model.py').write_text('partial')
            raise DockerError('transport disconnected')
        (destination / 'model.py').write_text('pass\n')
        return {'bytes': 5}
    def remove(self, identifier):
        self.removed.append(identifier)
        return SimpleNamespace(returncode=0)
    remove_volume = remove
    def start(self, *args): raise AssertionError('agent was restarted')
    def exec(self, *args, **kwargs): raise AssertionError('agent executed during recovery')
    def kill(self, *args): pass


class OperatorRecoveryTests(unittest.TestCase):
    @contextlib.contextmanager
    def collection(self):
        with tempfile.TemporaryDirectory() as output, tempfile.TemporaryDirectory(prefix='abench-agent-') as staging, \
             tempfile.TemporaryDirectory() as registry, patch.dict(os.environ, {'ABENCH_RECOVERY_DIR': registry}):
            result = Path(output)
            (result/'logs').mkdir()
            for name in ('stdout', 'stderr'):
                (result/f'logs/agent_{name}.log').write_bytes(b'')
            yield result, Path(staging), FakeDocker()

    def context(self):
        return {'args': {'case': 'TC-FT-001', 'run_id': 'retained', 'backend': 'docker', 'harness': 'mock',
                         'smoke': True, 'gpus': None, 'defer_evaluation': True},
                'oj': {'submission': {'allowed_top_level': ['model.py']}},
                'endpoint': None, 'net': {'mode': 'none'}, 'attestation': {'locked': True}, 'view': {},
                'cutoff': None, 'papers': None, 'run_config': {'suite': 'all@2.2.0'}, 'catalog': {},
                'restrictions': {}, 'recorded_argv': ['mock'], 'passed': ['EXTRA_SECRET'], 'cli_version': None,
                'budget': {'cpu_core_hours': 1}, 'gpu': {'metered': False}, 'status': 'succeeded', 'code': 0,
                'wall': 12.5, 'started': '2026-01-01T00:00:00Z', 'mode': 'smoke', 'prompt_sha256': 'a' * 64,
                'agent_env_names': ['EXTRA_SECRET'], 'launch_files': [], 'host': {'platform': 'test', 'python': '3.11'},
                'host_cpu_seconds': None, 'credential_sha256': None}

    def environment(self, root, docker):
        env = DockerAgentEnvironment.__new__(DockerAgentEnvironment)
        env.docker, env.host_paths, env.paths = docker, {'root': root}, {}
        env.label, env.cid, env.sidecar, env.image = 'owned', 'container', None, 'test-image'
        env.ids, env.volumes = ['container'], ['volume']
        env.record = {'backend': 'docker', 'verified': True, 'cpu_usage_usec': 900000, 'cpu_baseline_usec': 100000}
        return env

    def test_new_process_finishes_the_original_run_after_all_transfer_retries_failed(self):
        with self.collection() as (result, root, docker):
            env, context = self.environment(root, docker), self.context()
            context['recorded_argv'].append('dummy-private-pass-env-value')
            lifecycle.Attempt(result, 'TC-FT-001', 'retained').phase('collection')
            with patch.dict(os.environ, {'EXTRA_SECRET': 'dummy-private-pass-env-value', 'RUN_STATUS': 'succeeded'}):
                secrets = _recovery_credentials(SimpleNamespace(pass_env=['EXTRA_SECRET', 'RUN_STATUS']), None)
                recovery.save_execution(result, context, env, secrets=secrets)
            self.assertNotIn('dummy-private-pass-env-value', recovery.pending_path(result).read_text())
            with patch('runtime.agent.time.sleep'), self.assertRaises(DockerError):
                env.collect()
            self.assertEqual(docker.removed, [])
            self.assertFalse((result/'agent_run.json').exists())
            self.assertEqual(lifecycle.protected_objects(docker)['containers'], {'container'})
            docker.fail = False
            with patch('containers.Docker', return_value=docker), patch('catalog.check_run_lock'), \
                 patch.object(DockerAgentEnvironment, 'execute', side_effect=AssertionError('executed agent')), \
                 patch.object(DockerAgentEnvironment, 'begin_meter', side_effect=AssertionError('metered again')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recover(result), 0)
            record = json.loads((result/'agent_run.json').read_text())
            self.assertEqual(record['agent_wall_seconds'], 12.5)
            self.assertEqual(record['agent_child_cpu_seconds'], .8)
            self.assertEqual(record['status'], 'succeeded')
            self.assertEqual((result/'submission/model.py').read_text(), 'pass\n')
            self.assertEqual(record['passed_env_names'], ['EXTRA_SECRET'])
            self.assertFalse(recovery.pending_path(result).exists())
            self.assertEqual(lifecycle.protected_objects(docker)['containers'], set())
            self.assertFalse(root.exists())
            self.assertEqual(set(docker.removed), {'container', 'volume'})

    def test_restore_rejects_running_foreign_or_different_daemon_before_any_transfer(self):
        for state in ('running', 'foreign', 'daemon'):
            with self.subTest(state=state), self.collection() as (result, root, docker):
                recovery.save_execution(result, self.context(), self.environment(root, docker))
                docker.running = state == 'running'
                docker.foreign = state == 'foreign'
                if state == 'daemon': docker.info = lambda: {'raw': {'ID': 'other-daemon'}}
                with self.assertRaises(ValueError): recovery.restore(result, docker=docker)
                self.assertEqual(docker.transfers, 0)
                self.assertEqual(docker.removed, [])

    def test_attempt_contains_phase_and_exception_class_without_exception_secret(self):
        with tempfile.TemporaryDirectory() as raw:
            attempt = lifecycle.Attempt(raw, 'case', 'run')
            attempt.phase('environment_setup')
            attempt.finish(error=DockerError('dummy-secret-in-transport-message'))
            document = json.loads((Path(raw)/'attempt.json').read_text())
            self.assertEqual(document['failure']['kind'], 'infrastructure')
            self.assertEqual(document['failure']['exception_type'], 'DockerError')
            self.assertNotIn('dummy-secret', json.dumps(document))
            self.assertEqual(document['phase'], 'environment_setup')

    def test_failed_collection_never_commits_partial_or_invalid_artifacts(self):
        cases = ((DockerError, 2, False), (DockerPathMissing, 1, False),
                 (ValueError, 1, False), (DockerError, 1, True))
        for error, expected_attempts, expired in cases:
            with self.subTest(error=error.__name__, expired=expired), self.collection() as (_, root, docker), \
                 contextlib.ExitStack() as stack:
                env = self.environment(root, docker)
                target = root/'download/submission'
                def fail(container, source, destination, **kwargs):
                    docker.transfers += 1
                    (destination/'partial').write_bytes(b'partial')
                    raise error('transfer failed')
                docker.download = fail
                stack.enter_context(patch('runtime.agent.time.sleep'))
                if expired:
                    stack.enter_context(patch('runtime.agent.time.monotonic', side_effect=[0, 0, 2]))
                with self.assertRaises(error):
                    env._download_artifact('container', '/submission', target, timeout=1, attempts=2)
                self.assertEqual(docker.transfers, expected_attempts)
                self.assertFalse(target.exists())
                self.assertEqual(list(target.parent.glob('.collect-*')), [])

    def test_gc_preserves_pending_artifacts_and_refuses_a_corrupt_registry(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {'ABENCH_RECOVERY_DIR': raw}):
            registry = Path(raw)/'retained.json'
            registry.write_text(json.dumps({'daemon': 'test-daemon', 'containers': ['pending-container'],
                                            'volumes': ['pending-volume']}))
            docker = Docker()
            docker.info = Mock(return_value={'raw': {'ID': 'test-daemon'}})
            docker._list = lambda kind, label: {'container': ['pending-container', 'old-container'],
                'volume': ['pending-volume', 'old-volume'], 'network': []}[kind]
            docker.run = Mock(return_value=SimpleNamespace(returncode=0))
            collected = docker.gc('test-run')
            self.assertEqual(collected['container'], ['old-container'])
            self.assertEqual(collected['volume'], ['old-volume'])
            registry.write_text('invalid json')
            docker.run.reset_mock()
            with self.assertRaises(ValueError):
                docker.gc('test-run')
            docker.run.assert_not_called()
            registry.unlink()
            collected = docker.gc('test-run')
            self.assertIn('pending-container', collected['container'])
            self.assertIn('pending-volume', collected['volume'])

    def test_original_runner_lock_prevents_concurrent_manual_recovery(self):
        with tempfile.TemporaryDirectory() as raw:
            with recovery.run_lock(raw), patch('agents.recovery.restore', side_effect=AssertionError('started collecting')):
                with self.assertRaises(BlockingIOError): recover(raw)

    def test_collect_only_commits_awaiting_judge_and_cannot_be_restarted_as_agent_work(self):
        from agents.runner import _finalize_run
        import jobs
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root/'logs').mkdir()
            for name in ('stdout', 'stderr'): (root/f'logs/agent_{name}.log').write_bytes(b'')
            source = root/'download/submission'; source.mkdir(parents=True)
            (source/'model.py').write_text('pass\n')
            context = self.context(); context['args']['defer_evaluation'] = False
            attempt = lifecycle.Attempt(root, 'TC-FT-001', 'retained')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_finalize_run(root, context, {'submission': source, 'retrieval_log': root/'absent'},
                                              {'backend': 'docker'}, None, attempt, judge=False), 0)
            attempt.finish()
            record = json.loads((root/'agent_run.json').read_text())
            self.assertFalse(record['evaluation_deferred'])
            self.assertEqual(jobs.finished_state(root, record, None)[0], 'judge_failed')
            self.assertFalse((root/'evaluation.json').exists())

if __name__ == '__main__': unittest.main()
