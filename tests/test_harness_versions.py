"""Installed CLI identity is established before authentication, development budget or independent judging."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from agents import HARNESSES, get_agent
from agents.runner import run, verify_harness_version
from catalog import category_of
from runtime.agent import HostSoftEnvironment

OUTPUTS = {'claude': b'2.1.312 (Claude Code)\n', 'codex': b'codex-cli 0.165.0\n',
           'gemini': b'0.66.0\n', 'dsh': b'0.2.1-rc.2\n', 'zcode': b'0.16.9\n'}


class VersionTest(unittest.TestCase):
    def test_real_cli_formats_and_source_version(self):
        for name in HARNESSES:
            agent = get_agent(name)
            self.assertEqual(agent.parse_cli_version(OUTPUTS[name]), agent.requested_cli_version, name)
            self.assertEqual(agent.version_probe(SimpleNamespace()).argv, [name, '--version'])
            for bad in (b'', b'unknown', b'0.0.0 warning', OUTPUTS[name] + b'also 0.0.0'):
                self.assertIsNone(agent.parse_cli_version(bad), (name, bad))
        zcode = get_agent('zcode')
        self.assertEqual(zcode.version, '3.15.1')
        self.assertEqual(zcode.requested_cli_version, '0.16.9')
        self.assertEqual(get_agent('claude').version_probe(SimpleNamespace(claude_bin='/custom/claude')).argv,
                         ['/custom/claude', '--version'])

    def test_non_cli_adapters_need_no_probe(self):
        with tempfile.TemporaryDirectory() as raw:
            environment = Mock()
            for name in ('mock', 'oracle', 'command'):
                agent = get_agent(name)
                self.assertIsNone(agent.requested_cli_version)
                self.assertIsNone(verify_harness_version(agent, SimpleNamespace(), environment, {}, Path(raw)))
            environment.execute.assert_not_called()
            self.assertFalse((Path(raw) / 'harness_version.json').exists())

    def test_host_smoke_probes_the_executable_without_a_credential(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            script = root / 'claude'
            script.write_text('#!/bin/sh\n[ -z "$TEST_KEY" ] || exit 9\nprintf "2.1.312 (Claude Code)\\n"\n')
            script.chmod(0o700)
            args = SimpleNamespace(smoke=True, claude_bin=str(script))
            environment = HostSoftEnvironment(args, {'workspace': root, 'home': root, 'root': root})
            result = verify_harness_version(get_agent('claude'), args, environment,
                                            {'HOME': str(root), 'PATH': '/usr/bin:/bin', 'TEST_KEY': 'secret'}, root)
            self.assertTrue(result['verified'])
            self.assertEqual(result['actual_version'], '2.1.312')
            self.assertEqual(json.loads((root / 'harness_version.json').read_text()), result)
            self.assertNotIn('secret', (root / 'harness_version.json').read_text())

    def test_failures_retain_unverified_evidence(self):
        cases = [('succeeded', 0, b'0.0.1', 'does not match'),
                 ('succeeded', 0, b'could not read version', 'unrecognized'),
                 ('failed', 127, b'', 'did not finish'), ('timeout', None, b'', 'did not finish')]
        for status, code, output, reason in cases:
            with self.subTest(status=status, output=output), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                def execute(probe, env, timeout, out, err):
                    self.assertEqual(probe.argv, ['gemini', '--version'])
                    self.assertNotIn('TEST_KEY', env)
                    self.assertNotIn('HTTPS_PROXY', env)
                    self.assertLessEqual(timeout, 30)
                    out.write_bytes(output)
                    err.write_bytes(b'')
                    return status, code
                with self.assertRaisesRegex(ValueError, reason):
                    verify_harness_version(get_agent('gemini'), SimpleNamespace(), SimpleNamespace(execute=execute),
                                            {'TEST_KEY': 'secret', 'HTTPS_PROXY': 'https://model.invalid'}, root)
                record = json.loads((root / 'harness_version.json').read_text())
                self.assertFalse(record['verified'])
                self.assertEqual(record['requested_version'], '0.66.0')
                self.assertEqual(record['exit_code'], code)
                self.assertIn('stdout_sha256', record)
        with tempfile.TemporaryDirectory() as raw:
            environment = SimpleNamespace(execute=Mock(side_effect=OSError('missing executable')))
            with self.assertRaisesRegex(ValueError, 'OSError'):
                verify_harness_version(get_agent('codex'), SimpleNamespace(), environment, {}, Path(raw))
            self.assertFalse(json.loads((Path(raw) / 'harness_version.json').read_text())['verified'])


class RunnerVersionGateTest(unittest.TestCase):
    """A mislabeled old CLI stops even a formal Docker run before preflight or paid execution."""
    def test_version_gate_precedes_authentication_and_metering(self):
        for backend, smoke in (('docker', False), ('host', True)):
            for output, reached in ((b'0.0.1 (Claude Code)\n', False), (OUTPUTS['claude'], True)):
                with self.subTest(backend=backend, matching=reached), tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    args = SimpleNamespace(case='TC-FT-001', timeout_seconds=5, suite='all@2.2.0', device='cpu',
                        run_id='version-gate', backend=backend, smoke=smoke, harness='claude', network='none',
                        model=None, base_url=None, allow_private_model_endpoint=False, claude_model=None,
                        dry_run=False, gpus=None, hidden_key_file=None, defer_evaluation=True, result_root=str(root),
                        papers_index=None, models_dir=None, image='mislabeled-image:new-cli', pass_env=[],
                        claude_effort=None, claude_max_budget_usd=None, attempt=1, claude_bin='claude')
                    environment = Mock()
                    environment.default_path.return_value = '/opt/abench/harness/bin:/usr/bin:/bin'
                    environment.start.return_value = {'backend': backend, 'verified': True}
                    environment.record = {'cleanup': {'removed': [], 'failed': ['fake-container'],
                                                      'remaining': ['fake-container']}}
                    if not reached:
                        environment.close.side_effect = RuntimeError('cleanup failure must not mask version failure')
                    def construct(args, paths, *rest, **kwargs):
                        environment.paths = paths
                        return environment
                    def execute(probe, env, timeout, out, err):
                        self.assertEqual(probe.argv, ['claude', '--version'])
                        self.assertNotIn('TEST_KEY', env)
                        out.write_bytes(output)
                        err.write_bytes(b'')
                        return 'succeeded', 0
                    environment.execute.side_effect = execute
                    agent = get_agent('claude')
                    endpoint = {'name': 'fake', 'key_env': 'TEST_KEY', 'model': 'fake', 'parameters': {},
                                'base_url': 'https://model.invalid', 'api': 'anthropic'}
                    preflight = Mock(side_effect=RuntimeError('preflight reached'))
                    def view(case, oj, target, **kwargs):
                        target.mkdir(parents=True)
                        return {}
                    import sandbox_runner
                    with contextlib.ExitStack() as stack:
                        for target, replacement in (
                            ('agents.runner.resolve', Mock(return_value=endpoint)),
                            ('agents.runner.DockerAgentEnvironment', Mock(side_effect=construct)),
                            ('agents.runner.HostSoftEnvironment', Mock(side_effect=construct)),
                            ('images.image_attestation', Mock(return_value={'locked': True})),
                            ('catalog.suite_cases', Mock(return_value=([{'id': args.case}], 'all@2.2.0'))),
                            ('catalog.load_catalog', Mock(return_value=[{'id': args.case}])),
                            ('catalog.snapshot', Mock(return_value={'leaderboard': 'none'})),
                            ('sandbox_runner.build_agent_view', Mock(side_effect=view)),
                            ('sandbox_runner.paper_index_info', Mock(return_value=None)),
                            ('sandbox_runner.agent_prompt', Mock(return_value='test prompt'))):
                            stack.enter_context(patch(target, replacement))
                        stack.enter_context(patch.dict(os.environ, {'TEST_KEY': 'secret'}))
                        stack.enter_context(patch.object(agent, 'preflight', preflight))
                        meter = stack.enter_context(patch.object(sandbox_runner, 'GpuMeter'))
                        evaluate = stack.enter_context(patch('oj_eval.evaluate'))
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        expected = 'preflight reached' if reached else 'version verification failed'
                        with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                            run(args)
                        meter.assert_not_called()
                        evaluate.assert_not_called()
                        self.assertEqual(preflight.call_count, int(reached))
                    result = root / 'claude-fake' / category_of(args.case) / args.case / args.run_id   # runs/<harness>-<model>/<category>/<case>/<run>
                    record = json.loads((result / 'harness_version.json').read_text())
                    self.assertEqual(record['verified'], reached)
                    self.assertEqual(json.loads((result / 'cleanup.json').read_text()),
                                     environment.record['cleanup'])
                    self.assertEqual(environment.execute.call_count, 1)
                    environment.close.assert_called_once()

    def test_agent_receives_an_integer_deadline_and_the_run_is_filed_under_harness_and_model(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = SimpleNamespace(case='TC-FT-001', timeout_seconds=5, suite='all@2.2.0', device='cpu',
                run_id='deadline', backend='docker', smoke=False, harness='claude', network='none',
                model=None, base_url=None, allow_private_model_endpoint=False, claude_model=None,
                dry_run=False, gpus=None, hidden_key_file=None, defer_evaluation=True, result_root=str(root),
                papers_index=None, models_dir=None, image='claude-image', pass_env=[],
                claude_effort=None, claude_max_budget_usd=None, attempt=1, claude_bin='claude')
            environment = Mock()
            environment.default_path.return_value = '/opt/abench/harness/bin:/usr/local/cuda/bin:/usr/bin:/bin'
            environment.start.return_value = {'backend': 'docker', 'verified': True}
            environment.record = {'cleanup': {'removed': [], 'failed': [], 'remaining': []}}
            launched, constructed = {}, {}
            def construct(args, paths, *rest, **kwargs):
                environment.paths = paths
                constructed.update(kwargs)
                return environment
            def execute(probe, env, timeout, out, err):
                if probe.argv == ['claude', '--version']:
                    out.write_bytes(OUTPUTS['claude'])
                    err.write_bytes(b'')
                    return 'succeeded', 0
                launched.update(env)
                raise RuntimeError('agent launched')
            environment.execute.side_effect = execute
            endpoint = {'name': 'fake', 'key_env': 'TEST_KEY', 'model': 'fake', 'parameters': {},
                        'base_url': 'https://model.invalid', 'api': 'anthropic'}
            def view(case, oj, target, **kwargs):
                target.mkdir(parents=True)
                return {}
            import sandbox_runner
            with contextlib.ExitStack() as stack:
                for target, replacement in (
                    ('agents.runner.resolve', Mock(return_value=endpoint)),
                    ('agents.runner.DockerAgentEnvironment', Mock(side_effect=construct)),
                    ('images.image_attestation', Mock(return_value={'locked': True})),
                    ('catalog.suite_cases', Mock(return_value=([{'id': args.case}], 'all@2.2.0'))),
                    ('catalog.load_catalog', Mock(return_value=[{'id': args.case}])),
                    ('catalog.snapshot', Mock(return_value={'leaderboard': 'none'})),
                    ('sandbox_runner.build_agent_view', Mock(side_effect=view)),
                    ('sandbox_runner.paper_index_info', Mock(return_value=None)),
                    ('sandbox_runner.agent_prompt', Mock(return_value='test prompt'))):
                    stack.enter_context(patch(target, replacement))
                stack.enter_context(patch.dict(os.environ, {'TEST_KEY': 'secret'}))
                stack.enter_context(patch.object(get_agent('claude'), 'preflight', Mock(return_value=None)))
                stack.enter_context(patch.object(sandbox_runner, 'GpuMeter'))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, 'agent launched'):
                    run(args)
            # Integer Unix seconds, as on main and as the Sandbox Guide's own command parses it
            self.assertRegex(launched['ABENCH_DEADLINE'], r'^[0-9]+$')
            # The agent runs in the task image with its harness bundle first on the image's own PATH
            self.assertEqual(constructed.get('harness'), 'claude')
            self.assertEqual(launched['PATH'], '/opt/abench/harness/bin:/usr/local/cuda/bin:/usr/bin:/bin')
            # As on main, the runner sets no thread counts: the agent sees the libraries' own defaults
            self.assertFalse({'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'} & set(launched))
            self.assertTrue((root / 'claude-fake' / category_of(args.case) / args.case / args.run_id / 'harness_version.json').is_file())

    def test_run_label(self):
        from agents import run_label
        self.assertEqual(run_label('oracle'), 'oracle')
        self.assertEqual(run_label('mock'), 'mock')
        self.assertEqual(run_label('dsh'), 'dsh-' + get_agent('dsh').default_model)
        self.assertEqual(run_label('claude', 'deepseek-flash-anthropic'), 'claude-deepseek-flash-anthropic')


if __name__ == '__main__':
    unittest.main()
