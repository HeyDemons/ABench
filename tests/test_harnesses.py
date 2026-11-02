"""Harness adapters without Docker or credentials: configuration each CLI receives, protocol preflights against the
fake model service, trajectory summaries of recorded CLI output, and the in-container key injection helper.
The real CLIs are exercised by tests/test_harness_contract.py inside their agent images."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
from agents import HARNESSES, get_agent
from agents.models import resolve
from agents.preflight import protocol_preflight
from egress_proxy import decide, policy_for_case
from fake_model_server import FakeModelServer

STREAMS = ROOT / 'tests/fixtures/harness_streams'
PROMPT = 'Solve the ABench task.'


def endpoint(api, model='m', base='https://gateway.example.com/v1', effort='high'):
    return {'name': 'test', 'api': api, 'base_url': base, 'model': model, 'key_env': 'TEST_KEY',
            'parameters': {'effort': effort} if effort else {}}


class HarnessAdapterTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(backend='docker', claude_bin='claude', claude_effort=None, claude_max_budget_usd=None)
        self.paths = {k: Path('/abench/workspace' if k == 'workspace' else '/abench/' + k)
                      for k in ('workspace', 'view', 'submission', 'checkpoint')}
        self.paths['home'] = Path('/abench/workspace/.home')
        self.dns = patch('agents.models.socket.getaddrinfo',
                         return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))])
        self.dns.start()
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.dns.stop()

    def test_defaults_versions_and_official_hosts(self):
        pins = yaml.safe_load((ROOT / 'spec/env/images.yaml').read_text())['harnesses']
        self.assertEqual(set(HARNESSES), set(pins))
        defaults = {'claude': ('claude-opus-5-5', 'api.anthropic.com'), 'codex': ('gpt-6-astra', 'api.openai.com'),
                    'gemini': ('gemini-4-argon', 'generativelanguage.googleapis.com'),
                    'dsh': ('deepseek-flash', 'api.deepseek.com'), 'zcode': ('GLM-5.3', 'api.z.ai')}
        for name in HARNESSES:
            agent = get_agent(name)
            self.assertEqual(agent.version, str(pins[name]['version']))
            ep = resolve(agent)
            self.assertEqual((ep['model'], ep['host']), defaults[name], name)
            self.assertEqual(ep['parameters'], {'effort': 'high'}, name)
            pol = policy_for_case({}, name, ep)
            self.assertEqual(decide(pol, ep['host'], 443), (True, 'model_api'))
            agent.launch(self.args, self.paths, PROMPT, ep)
        with self.assertRaisesRegex(ValueError, 'does not support openai-chat'):
            resolve(get_agent('codex'), 'deepseek-flash-openai')
        with self.assertRaisesRegex(ValueError, 'does not support anthropic'):
            resolve(get_agent('gemini'), 'claude-opus-5-5')

    def test_unsupported_parameters_are_refused(self):
        for name in HARNESSES:
            agent = get_agent(name)
            ep = {**endpoint(agent.protocols[0]), 'parameters': {'silently_ignored': 1}}
            with self.assertRaisesRegex(ValueError, 'does not support model parameters', msg=name):
                agent.launch(self.args, self.paths, PROMPT, ep)

    def test_claude_effort_and_credentials(self):
        launch = get_agent('claude').launch(self.args, self.paths, PROMPT, endpoint('anthropic', base='https://api.anthropic.com'))
        self.assertEqual(launch.argv[launch.argv.index('--effort') + 1], 'high')
        self.assertEqual(launch.env['ANTHROPIC_BASE_URL'], 'https://api.anthropic.com')
        self.assertEqual(get_agent('claude').credential_env({}), ('ANTHROPIC_API_KEY',))

    def test_codex_isolated_provider_configuration(self):
        launch = get_agent('codex').launch(self.args, self.paths, PROMPT, endpoint('openai-responses'))
        config = launch.files['.codex/config.toml']
        for line in ('model = "m"', 'model_provider = "abench"', 'model_reasoning_effort = "high"',
                     'base_url = "https://gateway.example.com/v1"', 'env_key = "TEST_KEY"', 'wire_api = "responses"',
                     'check_for_update_on_startup = false'):
            self.assertIn(line, config)
        self.assertEqual(launch.env['CODEX_HOME'], '/abench/workspace/.home/.codex')
        self.assertEqual(launch.argv[-1], '-')
        self.assertEqual(launch.stdin, PROMPT.encode())
        self.assertIn('--dangerously-bypass-approvals-and-sandbox', launch.argv)
        self.assertEqual(get_agent('codex').credential_env({}), ())

    def test_gemini_settings_thinking_level_and_endpoint(self):
        launch = get_agent('gemini').launch(self.args, self.paths, PROMPT,
                                            endpoint('gemini', 'gemini-4-argon', 'https://generativelanguage.googleapis.com'))
        settings = json.loads(launch.files['.gemini/settings.json'])
        override, = settings['modelConfigs']['customOverrides']
        self.assertEqual(override['match'], {'model': 'gemini-4-argon'})
        self.assertEqual(override['modelConfig']['generateContentConfig']['thinkingConfig']['thinkingLevel'], 'HIGH')
        self.assertFalse(settings['privacy']['usageStatisticsEnabled'])
        self.assertFalse(settings['telemetry']['enabled'])
        self.assertEqual(launch.env['GOOGLE_GEMINI_BASE_URL'], 'https://generativelanguage.googleapis.com')
        self.assertIn('--yolo', launch.argv)
        self.assertEqual(launch.stdin, PROMPT.encode())
        self.assertEqual(get_agent('gemini').credential_env({}), ('GEMINI_API_KEY',))

    def test_dsh_native_deepseek_route_and_pi_ai_gateways(self):
        dsh = get_agent('dsh')
        native = dsh.launch(self.args, self.paths, PROMPT, endpoint('anthropic', 'deepseek-flash', 'https://api.deepseek.com/anthropic'))
        patch_doc = {e['id']: e['config'] for e in yaml.safe_load(native.files['.dsh-abench/patch.yml'])}
        self.assertEqual(patch_doc['llm-deepseek'], {'baseURL': 'https://api.deepseek.com/anthropic', 'apiKeyEnv': 'TEST_KEY',
                                                     'reasoningEffort': 'high'})
        self.assertEqual(patch_doc['agent-default-model'], {'provider': 'deepseek-official', 'model': 'deepseek-flash',
                                                            'reasoningEffort': 'high'})
        self.assertEqual(native.env, {'DSH_TELEMETRY_MODE': 'DISABLED', 'DSH_PERMISSION_MODE': 'danger-full-access'})
        self.assertEqual(native.argv[-2:], ['--json', '-'])
        for api, wire in (('openai-chat', 'openai-completions'), ('anthropic', 'anthropic-messages'),
                          ('openai-responses', 'openai-responses'), ('gemini', 'google-generative-ai')):
            launch = dsh.launch(self.args, self.paths, PROMPT, endpoint(api, 'other-model'))
            doc = {e['id']: e['config'] for e in yaml.safe_load(launch.files['.dsh-abench/patch.yml'])}
            route = doc['llm-pi-ai']['providers']['abench']
            self.assertEqual((route['api'], route['apiKeyEnv'], route['baseURL']), (wire, 'TEST_KEY', 'https://gateway.example.com/v1'))
            self.assertEqual(route['models'][0]['reasoningEfforts'], {'high': 'high'})
            self.assertEqual(doc['agent-default-model']['provider'], 'abench')

    def test_zcode_keyless_template_and_prompt_argument(self):
        zcode = get_agent('zcode')
        launch = zcode.launch(self.args, self.paths, PROMPT, endpoint('openai-chat', 'GLM-5.3', 'https://api.z.ai/api/paas/v4'))
        template = launch.files['.zcode-abench/provider_config.template.json']
        config = json.loads(template)['config']
        rule, = config['providerConfigRules']['providerRules']
        self.assertEqual(rule['config']['api'], {'type': 'openai-chat-completions', 'baseUrl': 'https://api.z.ai/api/paas/v4'})
        self.assertEqual(config['defaultModelSelection'], {'providerId': 'abench', 'modelId': 'GLM-5.3',
                                                           'options': {'reasoningLevel': 'high'}})
        self.assertEqual(template.count('ABENCH_KEY_PLACEHOLDER'), 1)
        self.assertEqual(launch.argv[:2], ['python3', '/abench/control/secret_config.py'])
        self.assertEqual(launch.argv[launch.argv.index('--key-env') + 1], 'TEST_KEY')
        self.assertEqual(launch.argv[launch.argv.index('--prompt') + 1], PROMPT)
        self.assertEqual(launch.env['ZCODE_PERSONAL_PROVIDER_CONFIG_FILE'], '/abench/workspace/.home/.zcode-abench/provider_config.json')
        with self.assertRaisesRegex(ValueError, 'single-argument limit'):
            zcode.launch(self.args, self.paths, 'x' * 200_000, endpoint('openai-chat', 'GLM-5.3'))


class WebToolPolicyTest(unittest.TestCase):
    """Provider-side web tools bypass the logged egress proxy and each case's blocklist, so every launch disables
    them; local fetch tools, which go through the proxy, stay (verified against the real CLIs, see the contract)."""
    def setUp(self):
        self.args = SimpleNamespace(backend='docker', claude_bin='claude', claude_effort=None, claude_max_budget_usd=None)
        self.paths = {'workspace': Path('/w'), 'view': Path('/v'), 'home': Path('/h')}

    def launch(self, name, api, model='m'):
        return get_agent(name).launch(self.args, self.paths, PROMPT, endpoint(api, model))

    def test_each_harness_disables_its_provider_side_tools(self):
        claude = self.launch('claude', 'anthropic')
        self.assertEqual(claude.argv[claude.argv.index('--disallowedTools') + 1], 'WebSearch')
        settings = json.loads(claude.files['.claude/settings.json'])
        self.assertEqual(settings, {'skipWebFetchPreflight': True, 'permissions': {'deny': ['WebSearch']}})
        self.assertIn('web_search = "disabled"', self.launch('codex', 'openai-responses').files['.codex/config.toml'])
        gemini = json.loads(self.launch('gemini', 'gemini').files['.gemini/settings.json'])
        self.assertEqual(gemini['tools']['exclude'], ['google_web_search', 'web_fetch'])
        for model in ('deepseek-flash', 'other'):
            patch_doc = {e['id']: e['config'] for e in yaml.safe_load(self.launch('dsh', 'anthropic', model).files['.dsh-abench/patch.yml'])}
            self.assertEqual(patch_doc['tool-web'], {'search': False, 'fetch': True})
        zcode = self.launch('zcode', 'openai-chat', 'GLM-5.3')
        self.assertEqual(zcode.argv[zcode.argv.index('--disallowed-tools') + 1], 'WebSearch')
        expected = {'claude': ('WebSearch',), 'codex': ('web_search',), 'gemini': ('google_web_search', 'web_fetch'),
                    'dsh': ('web_search',), 'zcode': ('WebSearch',)}
        self.assertEqual({n: get_agent(n).disabled_tools for n in HARNESSES}, expected)

    def test_zcode_gets_its_own_proxy_variables(self):
        self.assertEqual(get_agent('zcode').proxy_env('http://127.0.0.1:3128'),
                         {'ZCODE_HTTP_PROXY': 'http://127.0.0.1:3128', 'ZCODE_NO_PROXY': 'localhost,127.0.0.1'})
        self.assertEqual({n: get_agent(n).proxy_env('x') for n in HARNESSES if n != 'zcode'},
                         {n: {} for n in HARNESSES if n != 'zcode'})


class ErrorClassificationTest(unittest.TestCase):
    """Recorded output of each real CLI against a fake provider answering 401, 402, 429 or 500."""
    EXPECTED = {'401': 'provider_auth', '402': 'provider_quota', '429': 'provider_rate_limit', '500': 'provider_unavailable'}

    def test_recorded_provider_failures(self):
        from agents.errors import run_error_type
        errors = ROOT / 'tests/fixtures/harness_errors'
        for name in HARNESSES:
            for code, kind in self.EXPECTED.items():
                with self.subTest(harness=name, status=code):
                    exit_code = json.loads((errors / f'{name}-{code}.json').read_text())['exit']
                    status = 'timeout' if exit_code == 'timeout' else 'failed'
                    summary = get_agent(name).summarize((errors / f'{name}-{code}.stdout').read_bytes(),
                                                        (errors / f'{name}-{code}.stderr').read_bytes())
                    self.assertEqual(run_error_type(status, summary), kind)

    def test_successful_runs_and_agent_failures(self):
        from agents.errors import run_error_type
        for name in HARNESSES:
            summary = get_agent(name).summarize((STREAMS / f'{name}.jsonl').read_bytes())
            self.assertIsNone(run_error_type('succeeded', summary), name)
        retried = (b'{"type":"turn.started"}\n{"type":"error","message":"Reconnecting... 1/5 (unexpected status 500)"}\n'
                   b'{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n{"type":"turn.completed","usage":{}}\n')
        self.assertIsNone(run_error_type('succeeded', get_agent('codex').summarize(retried)))
        self.assertEqual(run_error_type('timeout', get_agent('codex').summarize(retried)), 'timeout')
        self.assertEqual(run_error_type('failed', {'result': {'is_error': True, 'error': 'tests failed'}}), 'agent_error')
        self.assertEqual(run_error_type('failed', {'result': {'is_error': True, 'error': 'prompt is too long'}}), 'context_length')

    def test_message_classification(self):
        from agents.errors import classify
        cases = {(429, 'You exceeded your current quota (insufficient_quota)'): 'provider_quota',
                 (None, 'unexpected status 403 Forbidden'): 'provider_auth',
                 (529, 'Overloaded'): 'provider_unavailable',
                 (None, 'The model `x` does not exist'): 'model_not_found',
                 (None, 'Error: ENOTFOUND api.example.com'): 'provider_unavailable',
                 (400, 'invalid request body'): None}
        for (status, text), kind in cases.items():
            self.assertEqual(classify(status, text), kind, text)

    def test_scoring_names_provider_errors(self):
        from agents.scoring import apply_agent_attestation
        result = apply_agent_attestation({'scorable': True}, {'status': 'failed', 'error_type': 'provider_quota'})
        self.assertIn('model provider error (provider_quota)', result['scoreability_reason'])
        self.assertNotIn('agent did not finish', result['scoreability_reason'])


class EffortAndCapabilityTest(unittest.TestCase):
    """Effort values are checked before any container starts; each harness publishes one capability record."""
    def setUp(self):
        self.args = SimpleNamespace(backend='docker', claude_bin='claude', claude_effort=None, claude_max_budget_usd=None)
        self.paths = {'workspace': Path('/w'), 'view': Path('/v'), 'home': Path('/h')}

    def launch(self, name, api, model, effort):
        return get_agent(name).launch(self.args, self.paths, PROMPT, endpoint(api, model, effort=effort))

    def test_effort_values_per_harness_and_model(self):
        accepted = (('claude', 'anthropic', 'claude-opus-5-5', 'xhigh'), ('codex', 'openai-responses', 'gpt-6-astra', 'none'),
                    ('gemini', 'gemini', 'gemini-4-argon', 'high'), ('gemini', 'gemini', 'gemini-4-flash', 'minimal'),
                    ('dsh', 'anthropic', 'deepseek-flash', 'max'), ('dsh', 'openai-chat', 'other', 'medium'),
                    ('zcode', 'openai-chat', 'GLM-5.3', 'max'), ('zcode', 'openai-chat', 'glm-4.6', 'enabled'))
        for case in accepted:
            with self.subTest(case=case):
                self.launch(*case)
        refused = (('claude', 'anthropic', 'claude-opus-5-5', 'highest'), ('codex', 'openai-responses', 'gpt-6-astra', 'hgih'),
                   ('gemini', 'gemini', 'gemini-4-argon', 'minimal'), ('gemini', 'gemini', 'gemini-2.5-pro', 'high'),
                   ('dsh', 'anthropic', 'deepseek-flash', 'medium'), ('zcode', 'openai-chat', 'GLM-5.3', 'enabled'))
        for case in refused:
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, 'does not accept effort'):
                self.launch(*case)
        self.args.claude_effort = 'turbo'                 # the CLI-flag override is checked as well
        with self.assertRaisesRegex(ValueError, 'does not accept effort'):
            self.launch('claude', 'anthropic', 'claude-opus-5-5', None)

    def test_capability_records(self):
        for name in HARNESSES:
            record = get_agent(name).capabilities()
            self.assertEqual(record['harness'], name)
            self.assertTrue(record['protocols'] and record['session_logs'] and record['disabled_provider_tools'])
            self.assertIn('effort', record['parameters'])
        self.assertEqual(get_agent('zcode').capabilities()['own_proxy_variables'], ['ZCODE_HTTP_PROXY', 'ZCODE_NO_PROXY'])
        self.assertEqual(get_agent('gemini').capabilities()['credential_variables'], ['GEMINI_API_KEY'])
        self.assertNotIn('--ephemeral', self.launch('codex', 'openai-responses', 'm', None).argv)
        self.assertNotIn('--no-session-persistence', self.launch('claude', 'anthropic', 'm', None).argv)


class NativeLogTest(unittest.TestCase):
    def test_transcripts_are_copied_and_credentials_withheld(self):
        from agents.native_logs import collect_native_logs
        from runtime.agent import HostSoftEnvironment
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            home = root / 'home'
            (home / '.codex/sessions/2026/10/29').mkdir(parents=True)
            (home / '.codex/sessions/2026/10/29/rollout-a.jsonl').write_text('{"type":"session_meta"}\n')
            (home / '.codex/sessions/2026/10/29/rollout-b.jsonl').write_text('{"auth":"sk-secret-value-123"}\n')
            (home / '.codex/config.toml').write_text('not a transcript')
            env = HostSoftEnvironment(SimpleNamespace(smoke=True), {'home': home, 'root': root})
            collected = env.collect(native=get_agent('codex').session_logs)
            record = collect_native_logs(collected['native'], root / 'record', ['sk-secret-value-123'])
            self.assertEqual([f['path'] for f in record['files']], ['.codex/sessions/2026/10/29/rollout-a.jsonl'])
            self.assertEqual(record['withheld'], [{'path': '.codex/sessions/2026/10/29/rollout-b.jsonl',
                                                   'reason': 'contains the model credential'}])
            self.assertFalse((root / 'record/.codex/config.toml').exists())
            self.assertEqual(collect_native_logs(None, root / 'none'), {'files': [], 'withheld': [], 'bytes': 0})


    def test_compressed_transcripts_are_checked_after_decompression(self):
        import gzip, shutil, subprocess
        from agents.native_logs import collect_native_logs
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / 'native/.dsh/sessions/s1'
            source.mkdir(parents=True)
            (source / 'clean.jsonl.gz').write_bytes(gzip.compress(b'{"type":"session"}\n'))
            (source / 'leak.jsonl.gz').write_bytes(gzip.compress(b'{"header":"sk-secret-value-123"}\n'))
            (source / 'broken.jsonl.gz').write_bytes(b'not gzip at all')
            if shutil.which('zstd'):                 # the format DSH writes
                subprocess.run(['zstd', '-q', '-o', str(source / 'leak.jsonl.zstd')], input=b'{"k":"sk-secret-value-123"}',
                               check=True)
            record = collect_native_logs(Path(raw) / 'native', Path(raw) / 'record', ['sk-secret-value-123'])
            self.assertEqual([f['path'] for f in record['files']], ['.dsh/sessions/s1/clean.jsonl.gz'])
            withheld = {w['path']: w['reason'] for w in record['withheld']}
            self.assertEqual(withheld['.dsh/sessions/s1/leak.jsonl.gz'], 'contains the model credential')
            self.assertEqual(withheld['.dsh/sessions/s1/broken.jsonl.gz'], 'compressed and could not be checked for the credential')
            if shutil.which('zstd'):
                self.assertEqual(withheld['.dsh/sessions/s1/leak.jsonl.zstd'], 'contains the model credential')
            self.assertFalse((Path(raw) / 'record/.dsh/sessions/s1/leak.jsonl.gz').exists())

class PreflightTest(unittest.TestCase):
    def test_each_protocol_reaches_the_configured_endpoint_with_its_key(self):
        cases = (('anthropic', '', '/v1/messages', 'x-api-key', 'fake-key'),
                 ('anthropic', '/anthropic', '/anthropic/v1/messages', 'x-api-key', 'fake-key'),
                 ('openai-responses', '/v1', '/v1/responses', 'authorization', 'Bearer fake-key'),
                 ('openai-chat', '/v1', '/v1/chat/completions', 'authorization', 'Bearer fake-key'),
                 ('gemini', '', '/v1beta/models/fake-model:generateContent', 'x-goog-api-key', 'fake-key'))
        with FakeModelServer() as server:
            for api, suffix, path, header, value in cases:
                with self.subTest(api=api, prefix=suffix):
                    configured = endpoint(api, 'fake-model', server.url + suffix)
                    launch = (get_agent('claude').preflight(configured) if api == 'anthropic'
                              else protocol_preflight(configured))
                    proc = subprocess.run(launch.argv, capture_output=True, timeout=20,
                                          env={'PATH': os.environ.get('PATH', ''), 'NO_PROXY': '127.0.0.1',
                                               'TEST_KEY': 'fake-key', **launch.env})
                    self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                    request = server.requests[-1]
                    self.assertEqual((request['path'], request[header]), (path, value))
                    if api != 'gemini':  # Gemini's model is in the URL, checked above.
                        self.assertEqual(request['body']['model'], 'fake-model')
                    self.assertIn('Reply pong', json.dumps(request['body']))
                    self.assertNotIn(b'fake-key', proc.stdout + proc.stderr)

    def test_failures_are_classified_without_counting_against_the_agent(self):
        from agents.preflight import preflight_failure
        for status, kind in ((401, 'provider_auth'), (402, 'provider_quota'), (429, 'provider_rate_limit')):
            with FakeModelServer(fail_status=status) as server:
                launch = protocol_preflight(endpoint('openai-chat', 'fake-model', server.url + '/v1'))
                proc = subprocess.run(launch.argv, capture_output=True, timeout=20,
                                      env={'PATH': os.environ.get('PATH', ''), 'NO_PROXY': '127.0.0.1', 'TEST_KEY': 'k', **launch.env})
                self.assertEqual(proc.returncode, 3)
                self.assertEqual(preflight_failure(proc.stdout)['error_type'], kind)
        with socket.socket() as sock:                     # a port nothing listens on
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        launch = protocol_preflight(endpoint('anthropic', 'm', f'http://127.0.0.1:{port}'))
        proc = subprocess.run(launch.argv, capture_output=True, timeout=20,
                              env={'PATH': os.environ.get('PATH', ''), 'NO_PROXY': '127.0.0.1', 'TEST_KEY': 'k', **launch.env})
        self.assertEqual(preflight_failure(proc.stdout)['error_type'], 'provider_unavailable')
        self.assertEqual(preflight_failure(b'{"abench_preflight_error": {"status": 400, "message": "bad"}}')['error_type'],
                         'provider_rejected')


class SummaryTest(unittest.TestCase):
    """Recorded output of each real CLI completing one shell-tool round against the fake model."""
    def test_recorded_streams(self):
        commands = {'codex': "/bin/sh -lc 'echo abench-tool-probe > abench_tool_probe.txt'"}
        tools = {'claude': 'Bash', 'codex': 'command_execution', 'gemini': 'run_shell_command', 'dsh': 'bash', 'zcode': 'Bash'}
        for name in HARNESSES:
            with self.subTest(name=name):
                summary = get_agent(name).summarize((STREAMS / f'{name}.jsonl').read_bytes())
                self.assertEqual(summary['tool_calls'], {tools[name]: 1})
                self.assertEqual(summary['bash_commands'], [commands.get(name, 'echo abench-tool-probe > abench_tool_probe.txt')])
                self.assertFalse(summary['result']['is_error'])
                self.assertEqual(summary['result']['final_text'], 'pong')
        self.assertIsNone(get_agent('codex').summarize(b''))

    def test_failed_or_unfinished_runs_are_errors(self):
        failures = {'codex': b'{"type":"turn.started"}\n{"type":"turn.failed","error":{"message":"quota"}}\n',
                    'gemini': b'{"type":"init","model":"g"}\n{"type":"result","status":"error","error":{"message":"bad key"}}\n',
                    'dsh': b'{"type":"status","phase":"turn_start","turn":1}\n{"type":"status","phase":"turn_end","reason":{"kind":"aborted"}}\n{"type":"final","text":""}\n',
                    'zcode': b'{"type":"turn.started","payload":{}}\n'}
        for name, raw in failures.items():
            with self.subTest(name=name):
                self.assertTrue(get_agent(name).summarize(raw)['result']['is_error'])


class SecretConfigTest(unittest.TestCase):
    def test_key_reaches_only_the_private_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'template.json').write_text('{"apiKey": "PLACEHOLDER"}')
            out = root / 'private/config.json'
            proc = subprocess.run([sys.executable, str(ROOT / 'tools/agents/secret_config.py'), '--template', str(root / 'template.json'),
                                   '--out', str(out), '--key-env', 'TEST_KEY', '--placeholder', 'PLACEHOLDER', '--',
                                   'cat', str(out)], capture_output=True, env={'PATH': os.environ['PATH'], 'TEST_KEY': 'se"cret'})
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertEqual(json.loads(proc.stdout), {'apiKey': 'se"cret'})
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
            missing = subprocess.run([sys.executable, str(ROOT / 'tools/agents/secret_config.py'), '--template', str(root / 'template.json'),
                                      '--out', str(out), '--key-env', 'TEST_KEY', '--placeholder', 'PLACEHOLDER', '--', 'true'],
                                     capture_output=True, env={'PATH': os.environ['PATH']})
            self.assertNotEqual(missing.returncode, 0)


class HomeFilesTest(unittest.TestCase):
    def test_files_stay_inside_home(self):
        from runtime.agent import HostSoftEnvironment
        with tempfile.TemporaryDirectory() as raw:
            env = HostSoftEnvironment(SimpleNamespace(smoke=True), {'home': Path(raw), 'root': Path(raw)})
            env.write_files({'.codex/config.toml': 'x = 1\n'})
            self.assertEqual((Path(raw) / '.codex/config.toml').read_text(), 'x = 1\n')
            for bad in ('../escape', '/etc/passwd'):
                with self.assertRaises(ValueError):
                    env.write_files({bad: 'x'})


if __name__ == '__main__':
    unittest.main()
