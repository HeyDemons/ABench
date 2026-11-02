"""Catalog selection, frozen suites, resource leases and resumable job behavior."""
import copy
import contextlib
import io
import json
import os
import sys
import tempfile
import shutil
import socket
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import catalog
import jobs
import abench
from agents import run_dir
from types import SimpleNamespace


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = catalog.load_catalog()
        suites = yaml.safe_load((catalog.ROOT / 'spec/suites.yaml').read_text())
        cls.default_suite = 'all@' + str(suites['default_versions']['all'])

    def test_all_profiles_resources_and_classifications(self):
        self.assertEqual(len(self.rows), 73)
        self.assertTrue(all('budget_tier' not in row for row in self.rows))
        self.assertEqual(catalog.catalog_errors(self.rows, check_site=False), [])
        self.assertEqual(catalog.distribution(self.rows), {
            'low': {'low': 11, 'medium': 4, 'high': 0, 'pro': 0},
            'medium': {'low': 1, 'medium': 19, 'high': 10, 'pro': 0},
            'high': {'low': 3, 'medium': 12, 'high': 7, 'pro': 6}})

    def test_shared_filters_and_calibration_membership(self):
        low = catalog.select_cases(self.rows, difficulty='low', resources='low')
        self.assertEqual(len(low), 11)
        self.assertEqual(sum(r['leaderboard'] == 'none' for r in self.rows), 2)
        with self.assertRaises(ValueError):
            catalog.select_cases(self.rows, cases='NOT-A-CASE')

    def test_suite_freezes_case_spec_and_profile(self):
        selected, name = catalog.suite_cases(rows=self.rows)
        self.assertEqual(len(selected), 73)
        self.assertEqual(name, self.default_suite)
        tampered = copy.deepcopy(self.rows)
        tampered[0]['environment']['version'] = 'future'
        with self.assertRaisesRegex(ValueError, 'frozen suite'):
            catalog.suite_cases(rows=tampered)

    def test_frozen_data_changes_do_not_change_any_case_suite_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ('index.yaml', 'verify.csv', 'spec/areas.yaml', 'spec/categories.yaml', 'spec/suites.yaml'):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(catalog.ROOT / relative, target)
            for row in self.rows:
                source = catalog.spec_path(row['id'])
                target = root / source.relative_to(catalog.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                oj = copy.deepcopy(row['oj'])
                oj['data'].update(public=[{'path': 'new/train.npz', 'sha256': 'a' * 64}],
                                  hidden={'path': 'new/hidden.enc', 'sha256_plain': 'b' * 64}, synthetic=True)
                target = root / 'cases' / row['id'] / 'case/oj.yaml'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(yaml.safe_dump(oj))
                shutil.copytree(catalog.ROOT / 'cases' / row['id'] / 'harness', root / 'cases' / row['id'] / 'harness',
                                ignore=shutil.ignore_patterns('__pycache__'))
                self.assertEqual(catalog.spec_hash(row['id'], root), row['spec_sha256'])
                self.assertEqual(catalog.judge_hash(row['id'], root), row['judge_sha256'])
            selected, identity = catalog.suite_cases(root=root)
            self.assertEqual(len(selected), 73)
            self.assertEqual(identity, self.default_suite)

    def _tree(self, root):
        for relative in ('index.yaml', 'verify.csv', 'spec/areas.yaml', 'spec/categories.yaml', 'spec/suites.yaml'):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(catalog.ROOT / relative, target)
        for row in self.rows:
            source = catalog.spec_path(row['id'])
            target = root / source.relative_to(catalog.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            for part in ('case/oj.yaml', 'harness'):
                src, dst = catalog.ROOT / 'cases' / row['id'] / part, root / 'cases' / row['id'] / part
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
                else:
                    shutil.copyfile(src, dst)

    def test_case_lock_covers_the_effective_oj_config_and_the_executed_kit(self):
        """A seed change in the generated oj.yaml or a changed kit copy leaves the suite identity unchanged, but not
        the case lock that a run is judged against."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            before = catalog.case_lock('DS-GBM-001', root)
            oj_path = root / 'cases/DS-GBM-001/case/oj.yaml'
            oj = yaml.safe_load(oj_path.read_text())
            oj['judge']['seeds'] = [999999]
            oj_path.write_text(yaml.safe_dump(oj))
            kit = root / 'cases/DS-GBM-001/harness/abench_stage.py'
            kit.write_text(kit.read_text() + '\n# changed protocol\n')
            after = catalog.case_lock('DS-GBM-001', root)
            self.assertEqual(sorted(k for k in before if before[k] != after[k]), ['harness_sha256', 'oj_sha256'])
            record = {'case_id': 'DS-GBM-001', 'run_config': {'suite': 'all@2.2.0', 'run_lock': {'case': before}}}
            catalog.suite_cases('all@2.2.0', root, only={'DS-GBM-001'})        # the suite identity still holds
            with self.assertRaisesRegex(ValueError, r'changed since the run started \(harness_sha256, oj_sha256\)'):
                catalog.check_run_lock(record, root)

    def test_judge_code_change_blocks_only_that_case_until_a_new_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            metrics = root / 'cases/A-LIN-002/harness/metrics.py'
            metrics.write_text(metrics.read_text() + '\n# changed scoring\n')
            (root / 'cases/A-LIN-002/harness/README.md').write_text('documentation does not count')
            with self.assertRaisesRegex(ValueError, 'A-LIN-002 differs'):
                catalog.suite_cases(root=root)
            with self.assertRaisesRegex(ValueError, 'A-LIN-002 differs'):
                catalog.suite_cases(root=root, only={'A-LIN-002'})
            selected, _ = catalog.suite_cases(root=root, only={'TC-FT-001'})
            self.assertEqual([r['id'] for r in selected], ['TC-FT-001'])
            with self.assertRaisesRegex(ValueError, 'does not contain'):
                catalog.suite_cases('calibration', root=root, only={'A-LIN-002'})
            result = catalog.freeze_suite('all@3.0.0', root, default=True)
            self.assertEqual(result, {'suite': 'all@3.0.0', 'cases': 73, 'default': True})
            self.assertEqual(catalog.suite_cases(root=root)[1], 'all@3.0.0')
            frozen = yaml.safe_load((root / 'spec/suites.yaml').read_text())
            self.assertRegex(frozen['suites']['all@3.0.0']['created_at'], r'^\d{4}-\d{2}-\d{2}$')
            with self.assertRaisesRegex(ValueError, 'semantic version'):
                catalog.freeze_suite('all@next', root)
            self.assertEqual(len(catalog.suite_cases('all@2.2.0', root=root, only={'TC-FT-001'})[0]), 1)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                catalog.freeze_suite('all@3.0.0', root)
            calibration = catalog.freeze_suite('calibration@1.0.0', root)
            self.assertEqual((calibration['cases'], calibration['default']), (3, False))
            self.assertEqual(catalog.suite_cases('calibration', root=root)[1], 'calibration@2.2.0')


class CategoryTests(unittest.TestCase):
    """Runs are organized by the five categories of the published dataset; every track and case has exactly one."""
    def test_every_case_has_one_category_and_runs_are_filed_by_it(self):
        rows = catalog.load_catalog()
        counts = {name: sum(r['category'] == name for r in rows) for name in catalog.category_names()}
        self.assertEqual(counts, {'llm_opt': 18, 'llm_post': 16, 'cv': 14, 'data_science': 19, 'pro': 6})
        self.assertEqual(catalog.category_of('IC-RES-002'), 'data_science')
        self.assertEqual(catalog.category_of('X-PPO-001'), 'llm_post')
        self.assertEqual(run_dir('runs', 'oracle', None, 'A-NORM-005', 'r1'), Path('runs/oracle/llm_opt/A-NORM-005/r1'))
        self.assertEqual([r['id'] for r in catalog.select_cases(rows, category='pro')],
                         [r['id'] for r in rows if r['id'].startswith('PRO-')])
        with self.assertRaisesRegex(ValueError, 'unknown case'):
            catalog.category_of('NOPE-001')

    def test_jobs_run_category_by_category(self):
        rows = [r for r in catalog.load_catalog() if r['id'] in ('PRO-SPD-004', 'TC-FT-001', 'A-LIN-002', 'C-QNT-003')]
        planned = jobs.plan(rows, ['mock'], [None], inventory=[], allow_shared=True)
        self.assertEqual([e['case'] for e in planned['entries']], ['A-LIN-002', 'C-QNT-003', 'TC-FT-001', 'PRO-SPD-004'])
        self.assertEqual([e['catalog']['category'] for e in planned['entries']], ['llm_opt', 'llm_post', 'data_science', 'pro'])


class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by = {r['id']: r for r in catalog.load_catalog()}

    def setUp(self):
        # Plans skip cases that need more cores than the host has; a GitHub runner has only 4.
        cpus = patch('jobs.os.cpu_count', return_value=64)
        cpus.start()
        self.addCleanup(cpus.stop)
        discovery = patch('jobs.scheduler_docker_info', return_value=None)
        discovery.start()
        self.addCleanup(discovery.stop)

    def test_matrix_estimate_and_skip_missing_hardware(self):
        plan = jobs.plan([self.by['TC-FT-001'], self.by['PRO-SPD-004']], ['mock', 'oracle'], [None, 'm'], 2, inventory=[])
        self.assertEqual(len(plan['entries']), 16)
        self.assertEqual(sum(e['state'] == 'skipped' for e in plan['entries']), 8)
        self.assertAlmostEqual(plan['estimate']['h100_hours'], (96 + jobs.evaluation_hours(self.by['PRO-SPD-004']) * 4) * 8)
        self.assertEqual(plan['runnable_estimate']['h100_hours'], 0)

    @patch('agents.models.socket.getaddrinfo',
           return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))])
    def test_harness_model_matrix_keeps_protocol_compatible_pairs(self, _dns):
        output = io.StringIO()
        with patch('jobs.gpu_inventory', return_value=[]), contextlib.redirect_stdout(output):
            self.assertEqual(abench.jobs_command(['run', '--cases', 'TC-FT-001', '--agent', 'claude,codex,dsh',
                                                  '--model', 'gpt-6-astra,deepseek-flash-anthropic', '--dry-run']), 0)
        doc = json.loads(output.getvalue())
        self.assertEqual(sorted((e['agent'], e['model']) for e in doc['entries']),
                         [('claude', 'deepseek-flash-anthropic'), ('codex', 'gpt-6-astra'),
                          ('dsh', 'deepseek-flash-anthropic'), ('dsh', 'gpt-6-astra')])
        self.assertEqual(sorted((x['agent'], x['model']) for x in doc['skipped_combinations']),
                         [('claude', 'gpt-6-astra'), ('codex', 'deepseek-flash-anthropic')])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            abench.jobs_command(['run', '--cases', 'TC-FT-001', '--agent', 'codex', '--model', 'claude-opus-5-5', '--dry-run'])

    def test_matrix_refuses_effort_values_before_running(self):
        output = io.StringIO()
        with patch('jobs.gpu_inventory', return_value=[]), contextlib.redirect_stdout(output):
            self.assertEqual(abench.jobs_command(['run', '--cases', 'TC-FT-001', '--agent', 'zcode,dsh',
                                                  '--model', 'deepseek-flash-openai', '--dry-run']), 0)
        doc = json.loads(output.getvalue())
        self.assertEqual([(e['agent'], e['model']) for e in doc['entries']], [('dsh', 'deepseek-flash-openai')])
        skipped, = doc['skipped_combinations']
        self.assertEqual(skipped['agent'], 'zcode')
        self.assertIn('does not accept effort', skipped['reason'])

    def test_other_gpu_requires_explicit_override_and_idle(self):
        inventory = [{'index': '0', 'class': 'other', 'idle': True}, {'index': '1', 'class': 'other', 'idle': False}]
        row = self.by['A-LIN-002']
        self.assertIsNotNone(jobs.skip_reason(row, inventory, cpu_count=64))
        self.assertIsNone(jobs.skip_reason(row, inventory, cpu_count=64, allow_nonreference=True))
        self.assertEqual(jobs.compatible_gpus(row, inventory, True), ['0'])

    def test_inventory_records_shared_memory_and_load_evidence(self):
        result = SimpleNamespace(stdout='2, GPU-test, NVIDIA RTX 6000 Ada Generation, 49140, 4096, 0\n')
        with patch('jobs.subprocess.run', return_value=result):
            gpu, = jobs.gpu_inventory()
        self.assertEqual(gpu['used_memory_mb'], 4096)
        self.assertEqual(gpu['free_memory_mb'], 45044)
        self.assertEqual(gpu['utilization'], 0)
        self.assertFalse(gpu['idle'])
        self.assertEqual(gpu['class'], 'other')

    def test_shared_nonreference_gpu_needs_both_explicit_overrides_and_enough_free_memory(self):
        row = self.by['A-LIN-002']
        gpu = {'index': '2', 'class': 'other', 'idle': False, 'free_memory_mb': 45044, 'utilization': 0}
        self.assertFalse(jobs.available_for_case(row, gpu))
        self.assertTrue(jobs.available_for_case(row, gpu, allow_shared=True))
        self.assertEqual(jobs.compatible_gpus(row, [gpu], allow_nonreference=True), [])
        self.assertEqual(jobs.compatible_gpus(row, [gpu], allow_shared=True), [])
        self.assertEqual(jobs.compatible_gpus(row, [gpu], allow_nonreference=True, allow_shared=True), ['2'])
        for memory, utilization, expected in ((24576, 10, True), (24575, 0, False), (45044, 11, False)):
            with self.subTest(memory=memory, utilization=utilization):
                changed = dict(gpu, free_memory_mb=memory, utilization=utilization)
                self.assertEqual(jobs.available_for_case(row, changed, allow_shared=True), expected)

    def test_shared_mode_keeps_cpu_behavior_and_rejects_pro_sharing(self):
        cpu, pro = self.by['TC-FT-001'], self.by['PRO-SPD-004']
        gpus = [{'index': str(i), 'class': 'h100', 'idle': True, 'free_memory_mb': 80000, 'utilization': 0}
                for i in range(4)]
        self.assertTrue(jobs.available_for_case(cpu, {}, allow_shared=True))
        self.assertIsNone(jobs.skip_reason(cpu, [], allow_shared=True, cpu_count=64))
        self.assertIsNone(jobs.skip_reason(pro, gpus, cpu_count=64))
        self.assertFalse(jobs.available_for_case(pro, gpus[0], allow_shared=True))
        self.assertIn('exclusive', jobs.skip_reason(pro, gpus, allow_shared=True, cpu_count=64))

    def test_pro_on_another_gpu_model_needs_the_override_and_an_exclusive_node(self):
        """Any GPU model is scored for Pro as for Budget, so the scheduler, runner and judge agree."""
        pro = self.by['PRO-SPD-004']
        card = lambda i: {'index': str(i), 'class': 'reference24', 'idle': True, 'free_memory_mb': 24000, 'utilization': 0}
        self.assertIn('needs 4', jobs.skip_reason(pro, [card(i) for i in range(4)], cpu_count=64))
        self.assertIsNone(jobs.skip_reason(pro, [card(i) for i in range(4)], allow_nonreference=True, cpu_count=64))
        self.assertEqual(jobs.compatible_gpus(pro, [card(i) for i in range(4)], True), ['0', '1', '2', '3'])
        self.assertIn('exclusive', jobs.skip_reason(pro, [card(i) for i in range(8)], allow_nonreference=True,
                                                    cpu_count=64))

    def test_shared_gpu_pool_still_leases_one_gpu_to_one_abench_child(self):
        row = self.by['A-LIN-002']
        gpu = {'index': '2', 'class': 'other', 'idle': False, 'free_memory_mb': 45044, 'utilization': 0}
        with tempfile.TemporaryDirectory() as tmp, patch('jobs.os.cpu_count', return_value=64):
            pool = jobs.ResourcePool([gpu], allow_nonreference=True, lock_dir=Path(tmp), allow_shared=True)
            self.assertEqual(pool.acquire('a', row), ['2'])
            self.assertIsNone(pool.acquire('b', row))
            pool.release('a')
            self.assertEqual(pool.acquire('b', row), ['2'])
            pool.release('b')

    def test_shared_cli_planning_and_resume_preserve_explicit_policy(self):
        gpu = {'index': '2', 'class': 'other', 'idle': False, 'free_memory_mb': 45044, 'utilization': 0}
        output = io.StringIO()
        with patch('jobs.gpu_inventory', return_value=[gpu]), contextlib.redirect_stdout(output):
            self.assertEqual(abench.jobs_command(['run', '--cases', 'A-LIN-002', '--agent', 'mock', '--dry-run',
                              '--allow-shared-gpu', '--allow-nonreference-gpu']), 0)
        document = json.loads(output.getvalue())
        self.assertTrue(document['allow_shared_gpu'])
        self.assertTrue(document['allow_nonreference_gpu'])
        self.assertEqual(document['entries'][0]['state'], 'pending')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.json'
            path.write_text(json.dumps(document))
            with patch('jobs.execute', return_value=document) as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(abench.jobs_command(['resume', str(path)]), 0)
                self.assertTrue(execute.call_args.kwargs['allow_shared'])
                self.assertTrue(execute.call_args.kwargs['allow_nonreference'])

    def test_resource_pool_excludes_concurrent_and_cross_job_gpu(self):
        row = self.by['A-LIN-002']
        inventory = [{'index': '0', 'class': 'reference24', 'idle': True}]
        with tempfile.TemporaryDirectory() as tmp, patch('jobs.os.cpu_count', return_value=64):
            first = jobs.ResourcePool(inventory, lock_dir=Path(tmp))
            second = jobs.ResourcePool(inventory, lock_dir=Path(tmp))
            self.assertEqual(first.acquire('a', row), ['0'])
            self.assertIsNone(first.acquire('b', row))
            self.assertIsNone(second.acquire('c', row))
            first.release('a')
            self.assertEqual(second.acquire('c', row), ['0'])
            second.release('c')

    def test_cpu_pool_accounts_for_cores(self):
        with tempfile.TemporaryDirectory() as tmp, patch('jobs.os.cpu_count', return_value=8):
            pool = jobs.ResourcePool([], lock_dir=Path(tmp))
            row = self.by['TC-FT-001']
            self.assertEqual(pool.acquire('a', row), [])
            self.assertIsNone(pool.acquire('b', row))
            pool.release('a')
            self.assertEqual(pool.acquire('b', row), [])
            pool.release('b')

    def test_only_infrastructure_failures_retry(self):
        self.assertTrue(jobs.infrastructure_failure(1, 'Cannot connect to the Docker daemon'))
        self.assertFalse(jobs.infrastructure_failure(1, 'training process exited with an error'))
        self.assertFalse(jobs.infrastructure_failure(0, 'docker daemon'))
        self.assertFalse(jobs.infrastructure_failure(1, 'docker daemon', {'error_type': 'agent'}))

    def test_resume_skips_completed_evaluation_and_report_excludes_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.by['TC-FT-001']
            doc = jobs.plan([row], ['mock'], [None], inventory=[])
            doc.update(id='resume', suite='all@2.2.0')
            entry = doc['entries'][0]
            run = root / 'runs' / 'mock' / row['category'] / row['id'] / ('job-resume-' + entry['id'])
            run.mkdir(parents=True)
            (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated'}))
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]), patch('jobs.subprocess.Popen') as launch:
                result = jobs.execute(doc, root / 'jobs/resume/job.json', root=root)
                launch.assert_not_called()
            self.assertEqual(result['entries'][0]['state'], 'completed')
            report = jobs.report(result)
            self.assertEqual(report['excluded_calibration'], ['TC-FT-001'])
            self.assertEqual(dict(report['groups']['area']), {})

    def _fake_runner(self, outcomes):
        """Popen stand-in: each call consumes (returncode, log text, write evaluation) and records its run id."""
        launched = []
        def popen(command, cwd, stdout, stderr):
            code, text, evaluate = outcomes.pop(0)
            stdout.write(text.encode())
            stdout.flush()
            # The runner's layout: <result root>/<harness>-<model>/<category>/<case>/<run id>
            run = run_dir(command[command.index('--result-root') + 1], command[command.index('--agent') + 1], None,
                          command[command.index('--case') + 1], command[command.index('--run-id') + 1])
            run.mkdir(parents=True, exist_ok=True)
            (run / 'logs').mkdir(exist_ok=True)
            if evaluate:
                (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated'}))
            launched.append(run.name)
            return SimpleNamespace(poll=lambda: code, pid=None, terminate=lambda: None, wait=lambda timeout=None: code)
        return popen, launched

    def _job(self, root, job_id):
        row = self.by['TC-FT-001']
        doc = jobs.plan([row], ['mock'], [None], inventory=[])
        doc.update(id=job_id, suite='all@2.2.0')
        return row, doc, root / 'jobs' / job_id / 'job.json'

    def test_interrupted_attempt_restarts_in_a_fresh_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, doc, path = self._job(root, 'interrupted')
            entry = doc['entries'][0]
            partial = jobs.run_path(root, doc, entry)
            (partial / 'logs').mkdir(parents=True)                  # a runner killed after creating its record
            entry.update(state='running', run=str(partial), pid=None)
            popen, launched = self._fake_runner([(0, '', True)])
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]), \
                    patch('jobs.os.cpu_count', return_value=64), patch('jobs.subprocess.Popen', side_effect=popen):
                jobs.execute(doc, path, root=root)
            self.assertEqual(launched, [partial.name + '-restart1'])
            self.assertEqual(entry['state'], 'completed')
            self.assertEqual(entry['interrupted_runs'], [str(partial)])
            self.assertTrue((partial / 'logs').is_dir())

    def test_infrastructure_retry_of_the_last_entry_runs_in_the_same_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, doc, path = self._job(root, 'retry')
            popen, launched = self._fake_runner([(1, 'Cannot connect to the Docker daemon', False), (0, '', True)])
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]), \
                    patch('jobs.os.cpu_count', return_value=64), patch('jobs.subprocess.Popen', side_effect=popen):
                jobs.execute(doc, path, root=root)
            entry = doc['entries'][0]
            self.assertEqual(len(launched), 2)
            self.assertTrue(launched[1].endswith('-retry1'))
            self.assertEqual(entry['state'], 'completed')
            self.assertNotIn('waiting_for_resources', doc)

    def test_deferred_runs_await_judging_instead_of_failing_or_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, doc, path = self._job(root, 'deferred')
            entry = doc['entries'][0]
            run = jobs.run_path(root, doc, entry)
            run.mkdir(parents=True)
            (run / 'agent_run.json').write_text(json.dumps({'evaluation_deferred': True}))
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]), \
                    patch('jobs.subprocess.Popen') as launch:
                jobs.execute(doc, path, root=root)
                launch.assert_not_called()
            self.assertEqual(entry['state'], 'awaiting_judge')
            (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated'}))
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]):
                jobs.execute(doc, path, root=root)
            self.assertEqual(entry['state'], 'completed')

    def test_provider_failures_retry_or_stop_without_counting_as_agent_failures(self):
        for kind, where, launches, state in (('provider_rate_limit', 'agent_run.json', 2, 'completed'),
                                              ('provider_unavailable', 'preflight.json', 2, 'completed'),
                                              ('provider_quota', 'agent_run.json', 1, 'provider_error'),
                                              ('provider_auth', 'preflight.json', 1, 'provider_error')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                row, doc, path = self._job(root, 'provider')
                calls = []
                def popen(command, cwd, stdout, stderr):
                    run = run_dir(command[command.index('--result-root') + 1], command[command.index('--agent') + 1], None,
                                  row['id'], command[command.index('--run-id') + 1])
                    run.mkdir(parents=True, exist_ok=True)
                    calls.append(run.name)
                    if len(calls) == 1:
                        (run / where).write_text(json.dumps({'error_type': kind}))
                        code = 1
                    else:
                        (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated'}))
                        code = 0
                    return SimpleNamespace(poll=lambda: code, pid=None, terminate=lambda: None, wait=lambda timeout=None: code)
                with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]), \
                        patch('jobs.os.cpu_count', return_value=64), patch('jobs.subprocess.Popen', side_effect=popen):
                    jobs.execute(doc, path, root=root)
                entry = doc['entries'][0]
                self.assertEqual((len(calls), entry['state']), (launches, state))
                if state == 'provider_error':
                    self.assertEqual(entry['error_type'], kind)

    def test_busy_gpu_is_waited_for_not_skipped(self):
        row = self.by['A-LIN-002']
        busy = [{'index': '0', 'class': 'reference24', 'idle': False, 'free_memory_mb': 1000, 'utilization': 90}]
        self.assertIsNone(jobs.skip_reason(row, busy, cpu_count=64))
        self.assertIsNotNone(jobs.skip_reason(row, [], cpu_count=64))
        doc = jobs.plan([row], ['mock'], [None], inventory=busy)
        self.assertEqual(doc['entries'][0]['state'], 'pending')
        with tempfile.TemporaryDirectory() as tmp:
            doc.update(id='busy', suite='all@2.2.0')
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=busy), \
                    patch('jobs.os.cpu_count', return_value=64), patch('jobs.subprocess.Popen') as launch:
                jobs.execute(doc, Path(tmp) / 'jobs/busy/job.json', root=Path(tmp))
                launch.assert_not_called()
        self.assertEqual(doc['entries'][0]['state'], 'pending')
        self.assertTrue(doc['waiting_for_resources'])

    def test_one_job_is_executed_by_one_process(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, doc, path = self._job(root, 'locked')
            path.parent.mkdir(parents=True)
            with (path.parent / 'job.lock').open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, 'already being executed'):
                        jobs.execute(doc, path, root=root)

    def test_pipeline_estimate_counts_actual_limits_seeds_and_baseline(self):
        row = copy.deepcopy(self.by['A-LIN-002'])
        row['oj']['judge']['seeds'] = [1, 2, 3]
        row['oj']['judge']['baseline'] = 'harness/baselines/gla'
        self.assertAlmostEqual(jobs.evaluation_hours(row), (1620 + 10) * 3 * 2 / 3600)
        self.assertAlmostEqual(jobs.estimate(row)['reference_gpu_hours'], 6 + (1620 + 10) * 3 * 2 / 3600)

    def test_report_retains_primary_interval_and_baseline_ratio(self):
        row = self.by['A-LIN-002']
        doc = jobs.plan([row], ['mock'], [None], inventory=[])
        doc.update(id='report', suite='all@2.2.0')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'evaluation.json').write_text(json.dumps({'verdict': 'evaluated',
                'primary': {'metric': 'quality', 'goal': 'max'}, 'metrics': {'quality': .9},
                'primary_ci95': [.8, .95], 'baseline': {'metrics': {'quality': .75}}}))
            doc['entries'][0].update(run=tmp, state='completed')
            result = jobs.report(doc)['groups']['area'][row['area']]['results'][0]
            self.assertEqual(result['primary_ci95'], [.8, .95])
            self.assertAlmostEqual(result['relative_to_baseline'], 1.2)

    def test_gpu_budget_stops_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.by['A-LIN-002']
            gpu = [{'index': '0', 'class': 'reference24', 'idle': True}]
            doc = jobs.plan([row], ['mock'], [None], inventory=gpu)
            doc.update(id='budget', suite='all@2.2.0')
            with patch('catalog.load_catalog', return_value=[row]), patch('jobs.gpu_inventory', return_value=gpu), patch('jobs.os.cpu_count', return_value=64), patch('jobs.subprocess.Popen') as launch:
                jobs.execute(doc, Path(tmp) / 'job.json', root=Path(tmp), max_gpu_hours=0)
                launch.assert_not_called()
            self.assertEqual(doc['entries'][0]['skip_reason'], 'max GPU hours would be exceeded')


class EnvFileTests(unittest.TestCase):
    def test_env_sets_unset_names_literally_and_errors_never_echo_values(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / '.env'
            self.assertEqual(abench.load_env(path), [])
            path.write_text('# keys\n\nABENCH_T_A=sk-1#x $HOME\nexport ABENCH_T_B="quoted value"\nABENCH_T_C=\n'
                            'ABENCH_T_SHELL=file\n')
            with patch.dict(os.environ, {'ABENCH_T_SHELL': 'shell'}):
                self.assertEqual(abench.load_env(path), ['ABENCH_T_A', 'ABENCH_T_B'])
                self.assertEqual(os.environ['ABENCH_T_A'], 'sk-1#x $HOME')
                self.assertEqual(os.environ['ABENCH_T_B'], 'quoted value')
                self.assertNotIn('ABENCH_T_C', os.environ)
                self.assertEqual(os.environ['ABENCH_T_SHELL'], 'shell')
            path.write_text('ABENCH_T_A=ok\nsk-secret-value\n')
            with patch.dict(os.environ, {}), self.assertRaises(ValueError) as caught:
                abench.load_env(path)
            self.assertIn(':2:', str(caught.exception))
            self.assertNotIn('secret', str(caught.exception))

    def test_env_is_kept_out_of_git_and_docker_builds(self):
        self.assertIn('/.env', (catalog.ROOT / '.gitignore').read_text().split())
        self.assertIn('.env', (catalog.ROOT / '.dockerignore').read_text().split())


class DoctorTests(unittest.TestCase):
    def test_doctor_paper_index_must_be_a_searchable_snapshot(self):
        import sqlite3
        try:
            with contextlib.closing(sqlite3.connect(':memory:')) as probe:
                probe.execute('CREATE VIRTUAL TABLE t USING fts5(a)')
        except sqlite3.OperationalError:
            self.skipTest('SQLite without FTS5')
        sys.path.insert(0, str(catalog.ROOT / 'tools/paper_search'))
        import build_paper_index
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            missing = abench.paper_index_check(str(root / 'none.sqlite'))
            self.assertFalse(missing['passed'])
            self.assertIn('no paper index', missing['detail'])
            snapshot = root / 'snapshot.jsonl'
            snapshot.write_text(''.join(json.dumps({
                'id': f'2101.0000{i}', 'title': f'Representation learning {i}', 'authors': 'A. Author',
                'abstract': 'Deep learning for tables.', 'categories': 'cs.LG',
                'versions': [{'version': 'v1', 'created': 'Mon, 4 Jan 2021 10:00:00 GMT'}]}) + '\n' for i in range(3)))
            index = root / 'papers.sqlite'
            build_paper_index.build(snapshot, index, '2026-01-01', None)
            small = abench.paper_index_check(str(index))
            self.assertEqual((small['passed'], small['papers']), (False, 3))
            self.assertIn('test index', small['detail'])
            with patch.object(abench, 'PAPER_INDEX_MIN_PAPERS', 3):
                self.assertTrue(abench.paper_index_check(str(index))['passed'])
            with patch.dict(os.environ, {'ABENCH_PAPERS_INDEX': str(index)}):
                self.assertEqual(abench.paper_index_check()['source'], 'environment')
            broken = root / 'broken.sqlite'
            broken.write_bytes(b'not a database')
            self.assertIn('unreadable', abench.paper_index_check(str(broken))['detail'])


class CalibrateTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home)
        self.key = self.home / 'hidden.key'
        self.key.write_text('0' * 64 + '\n')
        row = next(r for r in catalog.load_catalog() if r['id'] == 'DS-PIPE-002')
        self.row = copy.deepcopy(row)
        self.calls = []
        for target in (patch('abench.load_catalog', lambda: [self.row]), patch('abench.Path.home', lambda: self.home),
                       patch('abench.call', lambda tool, arguments, root=abench.ROOT, env=None:
                             self.calls.append((tool, arguments, root, env)) or 0)):
            target.start()
            self.addCleanup(target.stop)

    def freeze(self, synthetic):
        self.row['oj']['data'] = {**(self.row['oj'].get('data') or {}), 'synthetic': synthetic,
                                  'hidden': {'sha256_plain': 'a' * 64}}

    def test_unfrozen_data_is_refused_without_copying_or_preparing(self):
        self.row['oj']['data'] = {**(self.row['oj'].get('data') or {}), 'hidden': {'sha256_plain': 'pending'}}
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            abench.calibrate(['DS-PIPE-002', '--hidden-key-file', str(self.key)])
        self.assertIn('tools/prepare/run.py --case DS-PIPE-002', err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            abench.calibrate(['DS-PIPE-002', '--data', 'synthetic'])
        self.assertEqual(self.calls, [])
        self.assertFalse((self.home / '.abench').exists())

    def test_oracle_runs_in_this_checkout_with_the_key_path(self):
        self.freeze(synthetic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(abench.calibrate(['DS-PIPE-002', '--hidden-key-file', str(self.key), '--run-id', 'cal-1']), 0)
        [(tool, arguments, root, env)] = self.calls
        self.assertEqual((tool, root), ('sandbox_runner.py', abench.ROOT))
        option = lambda name: arguments[arguments.index(name) + 1]
        self.assertEqual(option('--agent'), 'oracle')
        self.assertEqual(option('--hidden-key-file'), str(self.key.resolve()))
        self.assertEqual(option('--result-root'), str(abench.ROOT / 'runs'))
        self.assertEqual(option('--run-id'), 'cal-1')
        self.assertEqual(env['ABENCH_ALLOW_SYNTHETIC'], '1')
        self.assertFalse((self.home / '.abench').exists())

    def test_missing_key_is_refused_not_generated(self):
        self.freeze(synthetic=False)
        missing = self.home / 'absent.key'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            abench.calibrate(['DS-PIPE-002', '--hidden-key-file', str(missing)])
        self.assertFalse(missing.exists())
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
