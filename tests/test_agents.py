"""Harness contract, endpoint policy and Unix proxy tests; no real service credential required."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from agents import get_agent
from agents.models import resolve
from egress_proxy import EgressProxy, decide, policy_for_case, summarize_log
from agents.workspace import resource_limits


class ResourceInputTest(unittest.TestCase):
    def test_verified_container_limits_and_shortened_deadline_are_recorded(self):
        import yaml
        oj = yaml.safe_load((ROOT / 'cases/TC-FT-001/case/oj.yaml').read_text())
        sandbox = {'backend': 'docker', 'resources_enforced': True, 'resources': oj['resources'],
                   'inspect': {'resource_enforcement': {'actual': {'cpus': 8_000_000_000,
                               'memory': 16 * 1024**3, 'pids': 1024}}},
                   'probes': {'memory_current_bytes': 1024**3}}
        info = resource_limits(oj, sandbox, 1800)
        self.assertEqual(info['limits']['memory_bytes'], 16 * 1024**3)
        self.assertEqual(info['memory_headroom_bytes_at_start'], 15 * 1024**3)
        self.assertEqual(info['limits']['cpu_cores'], 8)
        self.assertEqual(info['session_seconds'], 1800)
        self.assertEqual(info['development_budget'], oj['agent_budget'])
        sandbox['resources_enforced'] = False
        info = resource_limits(oj, sandbox, 1800)
        self.assertIsNone(info['limits']['memory_bytes'])
        self.assertIsNone(info['memory_headroom_bytes_at_start'])


class AgentsTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(backend='docker', claude_bin='claude', claude_effort=None,
                                    claude_max_budget_usd=None, command='python3 worker.py')
        self.paths = {k: Path('/abench/' + k) for k in ('workspace', 'view', 'submission', 'checkpoint')}
        self.agent = get_agent('claude')
        self.dns = patch('agents.models.socket.getaddrinfo', return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
        self.dns.start()
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.dns.stop()

    def test_endpoint_protocol_override_and_private_target(self):
        ep = resolve(self.agent, 'deepseek-chat-anthropic')
        self.assertEqual(ep['host'], 'api.deepseek.com')
        self.assertEqual(ep['key_env'], 'DEEPSEEK_API_KEY')
        self.assertNotIn('key', ep)
        with self.assertRaises(ValueError):
            resolve(self.agent, 'deepseek-chat-openai')
        for url in ('http://api.example.com', 'https://u:p@api.example.com', 'https://127.0.0.1:8000'):
            with self.assertRaises(ValueError, msg=url):
                resolve(self.agent, base_url=url)
        private = resolve(self.agent, base_url='https://127.0.0.1:8000', allow_private=True)
        pol = policy_for_case({}, 'claude', private)
        self.assertTrue(decide(pol, '127.0.0.1', 8000)[0])
        self.assertFalse(decide(pol, '127.0.0.1', 8001)[0])
        with self.assertRaises(ValueError):
            resolve(self.agent, base_url='https://huggingface.co')
        with self.assertRaises(ValueError):
            resolve(self.agent, base_url='https://data.example.com', oj={'agent_network': {'deny':['data.example.com']}})

    def test_only_configured_model_host_and_source_are_recorded(self):
        with patch.dict(os.environ, {'ANTHROPIC_BASE_URL': 'https://gateway.example.com/anthropic'}):
            ep = resolve(self.agent)                         # default model: the environment may redirect it
            self.assertEqual((ep['source'], ep['host']), ('environment', 'gateway.example.com'))
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                resolve(self.agent, 'claude-sonnet')         # a named configuration is never silently redirected
            self.assertEqual(resolve(self.agent, 'claude-sonnet', base_url='https://gateway.example.com/anthropic')['source'],
                             'command_line')
        with patch.dict(os.environ, {'ANTHROPIC_BASE_URL': 'https://api.deepseek.com/anthropic/'}):
            self.assertEqual(resolve(self.agent, 'deepseek-chat-anthropic')['source'], 'models.yaml')
        ep = resolve(self.agent, base_url='https://api.deepseek.com/anthropic')
        self.assertEqual(ep['source'], 'command_line')
        pol = policy_for_case({}, 'claude', ep)
        self.assertEqual(decide(pol, 'api.deepseek.com', 443), (True,'model_api'))
        self.assertFalse(decide(pol, 'api.anthropic.com', 443)[0])

    def test_adapter_prompt_environment_and_trace_contract(self):
        ep = resolve(self.agent, 'deepseek-chat-anthropic')
        launch = self.agent.launch(self.args, self.paths, 'the exact prompt', ep)
        self.assertEqual(launch.stdin, b'the exact prompt')
        self.assertEqual(launch.env['ANTHROPIC_MODEL'], 'deepseek-chat')
        self.assertEqual(launch.env['ANTHROPIC_BASE_URL'], ep['base_url'])
        self.assertEqual(launch.env['DISABLE_AUTOUPDATER'], '1')
        self.assertIn('--dangerously-skip-permissions', launch.argv)
        with self.assertRaises(ValueError):
            self.agent.launch(self.args, self.paths, 'p', {**ep, 'parameters':{'silently_ignored': True}})
        raw = b'{"type":"assistant","message":{"model":"actual-model","content":[{"type":"tool_use","name":"Bash"}]}}\n{"type":"result","subtype":"success","num_turns":1,"usage":{"input_tokens":5,"output_tokens":2}}\n'
        trace = self.agent.summarize(raw)
        self.assertEqual(trace['actual_models'], ['actual-model'])
        self.assertEqual(trace['usage']['input_tokens'], 5)
        for name in ('mock', 'oracle', 'command'):
            offline = get_agent(name)
            self.assertIsNone(resolve(offline))
            self.assertTrue(offline.launch(self.args, self.paths, 'p').argv)


class UnixProxyTest(unittest.TestCase):
    def test_unix_denial_and_hash_chain_tampering(self):
        with tempfile.TemporaryDirectory(prefix='px-') as raw:
            root=Path(raw)
            proxy=EgressProxy(policy_for_case({},'mock'),root/'log',unix_socket=root/'p.sock').start()
            try:
                client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                client.connect(str(root/'p.sock'))
                client.sendall(b'CONNECT huggingface.co:443 HTTP/1.1\r\nHost: huggingface.co\r\n\r\n')
                self.assertIn(b'403',client.recv(4096))
                client.close()
            finally:
                proxy.stop()
            summary=summarize_log(root/'log')
            self.assertTrue(summary['chain_ok'])
            self.assertEqual(summary['denied'],1)
            row=json.loads((root/'log').read_text())
            row['prev_sha256']='1'*64
            (root/'log').write_text(json.dumps(row)+'\n')
            self.assertFalse(summarize_log(root/'log')['chain_ok'])
