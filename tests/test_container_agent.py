"""Opt-in agent isolation/lifecycle tests using only a local CPU image and no API key."""
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
sys.path.insert(0,str(ROOT/'tools'))
from agents.base import Launch
from runtime.agent import DockerAgentEnvironment
from egress_proxy import policy_for_case, summarize_log


@unittest.skipUnless(os.environ.get('ABENCH_TEST_IMAGE') and shutil.which('docker'),
                     'set ABENCH_TEST_IMAGE to a locally available CPU profile image')
class DockerAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='abench-test-agent-')
        self.root=Path(self.tmp.name)
        self.paths={'root':self.root,'view':self.root/'view','workspace':self.root/'workspace','control':self.root/'control'}
        for name in ('view','workspace','control'):
            self.paths[name].mkdir()
        (self.paths['view']/'task.md').write_text('test task\n')
        self.args=SimpleNamespace(case='TC-FT-001',run_id='integration-test',gpus=None,network='none')
        self.oj=yaml.safe_load((ROOT/'cases/TC-FT-001/case/oj.yaml').read_text())
        # Docker refuses more CPUs than the host has, and a GitHub runner has 4; the checks do not depend on it.
        self.oj['resources']={**self.oj['resources'],'cpus':2}
        self.environment=None

    def tearDown(self):
        if self.environment:
            self.environment.close()
        self.tmp.cleanup()

    def start(self,proxy=False):
        pol=None
        if proxy:
            self.args.network='egress'
            pol=policy_for_case({},'mock')
            for src,name in ((ROOT/'tools/egress_proxy.py','egress_proxy.py'),
                             (ROOT/'tools/runtime/forwarder.py','forwarder.py')):
                shutil.copyfile(src,self.paths['control']/name)
            (self.paths['control']/'policy.json').write_text(json.dumps(pol))
        image=os.environ['ABENCH_TEST_IMAGE']
        self.environment=DockerAgentEnvironment(self.args,self.paths,self.oj,image,image,pol)
        return self.environment.start()

    def execute(self,code,timeout=10,env=None):
        return self.environment.execute(Launch(['python3','-c',code]),{'HOME':'/abench/workspace/.home',**(env or {})},timeout,
                                        self.root/'out.log',self.root/'err.log')

    def test_network_dns_mounts_and_denied_hub_proxy(self):
        rec=self.start(proxy=True)
        self.assertTrue(rec['verified'],rec)
        self.assertTrue(rec['probes']['direct_blocked'])
        self.assertTrue(rec['probes']['dns_blocked'])
        self.assertTrue(rec['probes']['view_readonly'])
        code="""import urllib.request
try:
 urllib.request.urlopen('https://huggingface.co',timeout=5)
 raise RuntimeError('hub unexpectedly reachable')
except urllib.error.URLError as e:
 assert '403' in str(e), str(e)
open('/abench/workspace/submission/result.txt','w').write('blocked')
"""
        status,code=self.execute(code,env={'HTTPS_PROXY':'http://127.0.0.1:3128','NO_PROXY':'localhost,127.0.0.1'})
        self.assertEqual(status,'succeeded',(status,code,(self.root/'err.log').read_text()))
        result=self.environment.collect()
        self.assertEqual((result['submission']/'result.txt').read_text(),'blocked')
        summary=summarize_log(result['egress_log'])
        self.assertTrue(summary['chain_ok'])
        self.assertGreaterEqual(summary['denied'],1)

    def test_timeout_is_accounted_then_killed(self):
        self.start()
        status,_=self.execute('import time; time.sleep(30)',timeout=1)
        self.assertEqual(status,'timeout')
        # collect() reads the container cgroup first, so a timed-out run still has CPU accounting, then kills it
        self.environment.collect()
        self.assertFalse(self.environment.docker.inspect(self.environment.cid)['State']['Running'])
        self.assertIsNotNone(self.environment.record['cpu_usage_usec'])

    def test_collection_rejects_symlink(self):
        self.start()
        self.assertEqual(self.execute("import os; os.symlink('/etc/passwd','/abench/workspace/submission/stolen')")[0],'succeeded')
        with self.assertRaises(ValueError):
            self.environment.collect()

    def test_task_workspace_and_processes_do_not_cross_into_evaluation(self):
        sys.path.insert(0, str(ROOT / 'tools/harness_kit'))
        import abench_stage
        from judge_container import DockerStageLauncher
        from agents.workspace import resource_limits
        record = self.start()
        limits = resource_limits(self.oj, record, 10)['limits']
        # The recorded limits are the ones the container's cgroup enforces.
        code = f"""import pathlib,subprocess,sys
assert {limits['memory_bytes']}==int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
quota,period=map(int,pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert {limits['cpu_cores']}==quota/period
pathlib.Path('/abench/workspace/task-only.txt').write_text('private scratch')
pathlib.Path('/tmp/task-only.txt').write_text('private temporary file')
pathlib.Path('/abench/workspace/submission/answer.txt').write_text('42')
subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'], start_new_session=True,
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
"""
        self.assertEqual(self.execute(code,env={'TASK_ONLY_ENV':'private'})[0], 'succeeded')
        collected = self.environment.collect()
        self.assertFalse(self.environment.docker.inspect(self.environment.cid)['State']['Running'])
        self.environment.close()
        stages = self.root/'harness/stages.py'
        stages.parent.mkdir()
        stages.write_text('''import os
from pathlib import Path
import numpy as np
from abench_stage import Stage
PARAMS = {}
PIPELINE = [{'stage':'check','timeout':30,'finish_cap':10,'threads':1}]
def timed(ctx, state):
    assert not Path('/abench/workspace/task-only.txt').exists()
    assert not Path('/tmp/task-only.txt').exists()
    assert not Path('/abench/control').exists()
    assert 'TASK_ONLY_ENV' not in os.environ
    return int((ctx.submission/'answer.txt').read_text())
STAGES = {'check':Stage(timed=timed,finish=lambda ctx,value:{'answer':np.array([value])})}
''')
        resources = dict(gpus=0, cpus=1, memory='1g', shm_size='64m', pids=128, budget_tier='budget')
        launcher = DockerStageLauncher(os.environ['ABENCH_TEST_IMAGE'], resources)
        try:
            result = abench_stage.run_pipeline(abench_stage.load_module(stages,'phase_boundary'), stages,
                collected['submission'], self.root/'judge', public=None,models=None,held_out=None,
                seed=1,device='cpu',launcher=launcher)
            self.assertEqual(result['status'], 'ok', result)
            self.assertEqual(result['outputs']['check']['answer'].tolist(), [42])
        finally:
            launcher.close_inputs()
