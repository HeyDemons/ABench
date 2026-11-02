"""Real harness CLIs from their bundles in a task image, offline, against the fake model service. Opt-in, dummy
credentials only.

Each harness runs through the runner's own Docker agent lifecycle (network none, uid 1000, configuration written
into HOME). The fake service runs on the container's loopback behind a forward proxy, and the model base URL names a
host only that proxy can reach, as in a real run behind the egress sidecar. The fake answers the first shell-tool
offer with a tool call, so the test checks: protocol preflight, that model traffic honours the proxy settings,
prompt/model/key delivery, that the headless approval mode really runs the tool, that no provider-side web tool is
offered, a parseable trajectory, the pinned CLI version, that a provider failure (402) is classified as such, and
that the CLI's own session transcripts are collected without the credential.

    ABENCH_TEST_IMAGE=<profile image> ABENCH_TEST_HARNESSES=claude,codex,... python3 -m unittest tests.test_harness_contract

Each named harness needs its bundle on this host (./abench images build --agent NAME).
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from agents import get_agent
from agents.errors import run_error_type
from agents.native_logs import collect_native_logs
from agents.models import resolve
from agents.runner import verify_harness_version
from runtime.agent import DockerAgentEnvironment

PROMPT = 'Reply pong for the ABench harness contract.'
KEY = 'abench-fake-contract-key'
PORT, PROXY, FAIL_PORT, FAIL_PROXY = 18080, 18081, 18090, 18091
MODEL_HOST = 'http://model.abench.test'      # resolvable only through the proxy
SUFFIX = {'anthropic': '', 'openai-responses': '/v1', 'openai-chat': '/v1', 'gemini': ''}
MODEL = {'dsh': 'deepseek-flash',          # DSH's default route for DeepSeek models on a Messages endpoint
         'zcode': 'GLM-5.3'}               # a model whose reasoning levels include the default effort high
WAIT = ('import socket, time\nfor port in (%d, %d, %d, %d):\n for _ in range(100):\n  try:\n   socket.create_connection(("127.0.0.1", port), 1).close(); break\n'
        '  except OSError: time.sleep(.1)\n else: raise SystemExit("fake model did not start")' % (PORT, PROXY, FAIL_PORT, FAIL_PROXY))


def offered(request):
    """Tool names and provider tool types a model request offers."""
    tools = (request.get('body') or {}).get('tools') or []
    names = {t.get('name') or (t.get('function') or {}).get('name') or t.get('type') for t in tools}
    names |= {d.get('name') for t in tools for d in t.get('functionDeclarations') or []}
    names |= {k for t in tools for k in t if k in ('googleSearch', 'google_search', 'urlContext', 'url_context')}
    return names - {None}


def harnesses():
    return [n for n in os.environ.get('ABENCH_TEST_HARNESSES', '').split(',') if n]


@unittest.skipUnless(harnesses() and os.environ.get('ABENCH_TEST_IMAGE') and shutil.which('docker'),
                     'set ABENCH_TEST_IMAGE=<profile image> and ABENCH_TEST_HARNESSES=name,... (bundles built on this host)')
class HarnessContract(unittest.TestCase):
    def test_each_harness(self):
        for name in harnesses():
            with self.subTest(harness=name):
                self.check(name, os.environ['ABENCH_TEST_IMAGE'])

    def check(self, name, image):
        agent = get_agent(name)
        with tempfile.TemporaryDirectory(prefix='abench-harness-contract-') as raw:
            root = Path(raw)
            paths = {'root': root, 'view': root / 'view', 'workspace': root / 'workspace', 'control': root / 'control'}
            for key in ('view', 'workspace', 'control'):
                paths[key].mkdir()
            (paths['view'] / 'task.md').write_text('contract task\n')
            for source in (ROOT / 'tests/fake_model_server.py', ROOT / 'tools/agents/secret_config.py'):
                shutil.copyfile(source, paths['control'] / source.name)
            args = SimpleNamespace(case='TC-FT-001', run_id='harness-contract-' + name, gpus=None, network='none',
                                   backend='docker', smoke=True, claude_bin='claude', claude_effort=None,
                                   claude_max_budget_usd=None)
            oj = yaml.safe_load((ROOT / 'cases/TC-FT-001/case/oj.yaml').read_text())
            environment = DockerAgentEnvironment(args, paths, oj, image, None, None, harness=name)
            try:
                self.assertTrue(environment.start()['verified'])
                docker, cid, runpaths = environment.docker, environment.cid, environment.paths
                for port, proxy, extra in ((PORT, PROXY, '--probe shell'), (FAIL_PORT, FAIL_PROXY, '--fail-status 402')):
                    docker.exec(cid, ['sh', '-c', f'python3 /abench/control/fake_model_server.py --port {port} --proxy-port {proxy} '
                                      f'{extra} --record /abench/workspace/.tmp/requests-{port}.jsonl > /dev/null 2>&1 &'])
                docker.exec(cid, ['python3', '-c', WAIT])
                endpoint = resolve(agent)               # the harness default, pointed at the fake service
                endpoint.update(base_url=MODEL_HOST + SUFFIX[endpoint['api']],
                                model=MODEL.get(name, 'fake-model'), key_env='ABENCH_FAKE_KEY')
                proxy = f'http://127.0.0.1:{PROXY}'
                env = {'HOME': str(runpaths['home']), 'TMPDIR': str(runpaths['tmp']), 'LANG': 'C.UTF-8',
                       'PATH': environment.default_path(), 'ABENCH_FAKE_KEY': KEY,
                       **{k: proxy for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')},
                       'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1', **agent.proxy_env(proxy)}
                env.update({n: KEY for n in agent.credential_env(endpoint)})
                launch = agent.launch(args, runpaths, PROMPT, endpoint)
                environment.write_files(launch.files)
                version = verify_harness_version(agent, args, environment, env, root)
                self.assertTrue(version['verified'], version)
                self.assertEqual(version['actual_version'], agent.requested_cli_version)
                status, _ = environment.execute(agent.preflight(endpoint), env, 60, root / 'pre.out', root / 'pre.err')
                self.assertEqual(status, 'succeeded', (root / 'pre.err').read_text())
                status, code = environment.execute(launch, env, 240, root / 'out', root / 'err')
                stdout, stderr = (root / 'out').read_bytes(), (root / 'err').read_text(errors='replace')
                self.assertEqual((status, code), ('succeeded', 0), stderr[-3000:])
                requests = [json.loads(line) for line in
                            docker.exec(cid, ['cat', f'/abench/workspace/.tmp/requests-{PORT}.jsonl']).stdout.splitlines()]
                self.assertTrue(any(r.get('proxy') and r['target'].startswith('model.abench.test') for r in requests))
                requests = [r for r in requests if not r.get('proxy')]
                for request in requests:
                    self.assertFalse(offered(request) & (set(agent.disabled_tools) | {'web_search', 'googleSearch', 'urlContext'}))
                if name == 'claude':                # Claude defers WebSearch; its init event lists every tool
                    init = next(json.loads(line) for line in stdout.splitlines() if b'"subtype":"init"' in line.replace(b' ', b''))
                    self.assertNotIn('WebSearch', init['tools'])
                agent_requests = [r for r in requests if 'Reply pong' not in json.dumps(r['body'])
                                  or PROMPT in json.dumps(r['body'])]
                self.assertTrue(any(PROMPT in json.dumps(r['body']) for r in agent_requests), requests)
                self.assertTrue(all(KEY in (r['authorization'] or '') + (r['x-api-key'] or '') + (r['x-goog-api-key'] or '')
                                    for r in requests if r['method'] == 'POST'))
                self.assertTrue(any(endpoint['model'] in json.dumps(r['body']) + r['path'] for r in agent_requests))
                probe = docker.exec(cid, ['cat', '/abench/workspace/abench_tool_probe.txt'], check=False)
                self.assertEqual(probe.stdout.strip(), 'abench-tool-probe', stderr[-3000:])
                summary = agent.summarize(stdout)
                self.assertFalse(summary['result']['is_error'], summary)
                self.assertGreaterEqual(sum(summary['tool_calls'].values()), 1, summary)
                self.assertNotIn(KEY.encode(), stdout)
                failing = dict(endpoint)
                env.update({k: f'http://127.0.0.1:{FAIL_PROXY}' for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')})
                env.update(agent.proxy_env(f'http://127.0.0.1:{FAIL_PROXY}'))
                launch = agent.launch(args, runpaths, PROMPT, failing)
                status, _ = environment.execute(launch, env, 120, root / 'fail.out', root / 'fail.err')
                summary = agent.summarize((root / 'fail.out').read_bytes(), (root / 'fail.err').read_bytes())
                self.assertEqual(run_error_type(status, summary), 'provider_quota', (status, summary))
                collected = environment.collect(native=agent.session_logs)
                native = collect_native_logs(collected['native'], root / 'native_logs', [KEY])
                self.assertTrue(native['files'], environment.record.get('native_log_transfers'))
                self.assertEqual(native['withheld'], [])
            finally:
                environment.close()


if __name__ == '__main__':
    unittest.main()
