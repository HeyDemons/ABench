"""Deferred CLI judging preserves immutable records and the original agent's scoring restrictions."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import abench
import catalog
import oj_eval
from agents.scoring import apply_agent_attestation


class DeferredJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.submission = self.run / 'submission'
        self.submission.mkdir()
        (self.submission / 'model.py').write_text('def predict(x): return x\n')
        self.key = self.root / 'hidden.key'
        self.key.write_text('00' * 32 + '\n')
        self.key.chmod(0o600)
        self.record = {'case_id': 'DS-GBM-001', 'run_mode': 'benchmark', 'isolation': 'container',
                       'status': 'succeeded', 'network': {'mode': 'none'}, 'budget': {'over_budget': False},
                       'catalog': {'leaderboard': 'all'},
                       'run_config': {'suite': catalog.suite_definition('all')[1],
                                      'run_lock': {'case': catalog.case_lock('DS-GBM-001')}},
                       'sandbox': {'backend': 'docker', 'verified': True, 'resources_enforced': True,
                                   'platform': 'linux/amd64', 'image_attestation': {'locked': True},
                                   'allocated_gpus': ['2'], 'hardware_matches':True},
                       'submission': {'tree_sha256': oj_eval.tree_hash(self.submission)}}
        self.save_record()

    def tearDown(self):
        self.temp.cleanup()

    def save_record(self):
        (self.run / 'agent_run.json').write_text(json.dumps(self.record))

    def invoke(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return abench.main(['judge', '--run', str(self.run), '--hidden-key-file', str(self.key), *extra])

    def test_legacy_deferred_judge_succeeds_once_and_preserves_placeholder(self):
        placeholder = {'verdict': 'not_evaluated', 'scorable': False, 'note': '--defer-evaluation'}
        (self.run / 'evaluation.json').write_text(json.dumps(placeholder))
        result = {'verdict': 'evaluated', 'scorable': True, 'metrics': {'accuracy': .8}}
        with patch('oj_eval.evaluate', return_value=result) as evaluate:
            self.assertEqual(self.invoke(), 0)
            args, kwargs = evaluate.call_args
            self.assertEqual(args[3], 'benchmark')
            self.assertEqual(kwargs['gpus'], ['2'])
            self.assertEqual(kwargs['backend'], 'docker')
            frozen = (self.run / 'evaluation.json').read_bytes()
            self.assertEqual(json.loads((self.run / 'evaluation.deferred.json').read_text()), placeholder)
            final = json.loads(frozen)
            self.assertEqual(final['metrics'], {'accuracy': .8})
            self.assertEqual(len(final['agent_run_sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.invoke()
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual((self.run / 'evaluation.json').read_bytes(), frozen)

    def test_new_deferred_run_has_no_placeholder_and_honors_explicit_gpu(self):
        with patch('oj_eval.evaluate', return_value={'verdict': 'evaluated', 'scorable': True}) as evaluate:
            self.assertEqual(self.invoke('--gpus', '3'), 0)
            self.assertEqual(evaluate.call_args.kwargs['gpus'], '3')
        self.assertFalse((self.run / 'evaluation.deferred.json').exists())

    def test_smoke_mode_never_becomes_benchmark(self):
        self.record.update(run_mode='smoke', isolation='host_soft')
        self.record['sandbox'].update(backend='host')
        self.save_record()
        with patch('oj_eval.evaluate', return_value={'verdict': 'smoke_unscored', 'scorable': True}) as evaluate:
            self.assertEqual(self.invoke('--backend', 'docker'), 0)
            self.assertEqual(evaluate.call_args.args[3], 'smoke')
            self.assertIsNone(evaluate.call_args.args[2])
        self.assertFalse(json.loads((self.run / 'evaluation.json').read_text())['scorable'])

    def test_open_network_and_original_host_isolation_cannot_be_promoted(self):
        for restriction in ('open', 'host'):
            with self.subTest(restriction=restriction):
                if restriction == 'open':
                    self.record['network']['mode'] = 'open'
                else:
                    self.record['network']['mode'] = 'none'
                    self.record['isolation'] = 'host_soft'
                    self.record['sandbox']['backend'] = 'host'
                self.save_record()
                with patch('oj_eval.evaluate', return_value={'verdict': 'evaluated', 'scorable': True}):
                    self.assertEqual(self.invoke('--backend', 'docker'), 0)
                output = self.run / 'evaluation.json'
                self.assertFalse(json.loads(output.read_text())['scorable'])
                output.unlink()

    def test_judging_needs_the_case_lock_the_run_started_with(self):
        """A deferred judge never scores an older run with a changed case or an unknown suite."""
        changed = copy.deepcopy(self.record)
        changed['run_config']['run_lock']['case']['judge_sha256'] = '0' * 64      # metrics.py changed since the run
        reseeded = copy.deepcopy(self.record)
        reseeded['run_config']['run_lock']['case']['oj_sha256'] = '0' * 64        # e.g. oj.yaml seeds edited by hand
        for alteration, message in ((changed, 'changed since the run started .judge_sha256.'),
                                    (reseeded, 'changed since the run started .oj_sha256.'),
                                    ({**self.record, 'run_config': {'suite': 'all@1999.01'}}, 'unknown suite'),
                                    ({**self.record, 'run_config': {}}, 'names no suite')):
            with self.subTest(message=message):
                self.record = alteration
                self.save_record()
                with patch('oj_eval.evaluate') as evaluate:
                    with self.assertRaisesRegex(ValueError, message):
                        self.invoke()
                    evaluate.assert_not_called()
                self.assertFalse((self.run / 'evaluation.json').exists())

    def test_records_from_before_the_lock_are_checked_against_their_suite(self):
        del self.record['run_config']['run_lock']
        self.save_record()
        with patch('oj_eval.evaluate', return_value={'verdict': 'evaluated', 'scorable': True}) as evaluate:
            self.assertEqual(self.invoke(), 0)
            evaluate.assert_called_once()

    def test_only_an_infrastructure_judge_failure_is_judged_again(self):
        out = self.run / 'evaluation.json'
        final = {'verdict': 'judge_error', 'scorable': False, 'error': 'judge container configuration rejected',
                 'failure': {'phase': 'judge', 'exception_type': 'RuntimeError', 'kind': 'judge_error',
                             'retryable': False}}
        out.write_text(json.dumps(final))
        with patch('oj_eval.evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.invoke()
            evaluate.assert_not_called()
        failed = {'verdict': 'judge_error', 'scorable': False, 'error': 'Docker archive transfer timed out after 120s',
                  'failure': {'phase': 'judge', 'exception_type': 'DockerError', 'kind': 'judge_infrastructure',
                              'retryable': True}}
        out.write_text(json.dumps(failed))
        with patch('oj_eval.evaluate', return_value={'verdict': 'evaluated', 'scorable': True}):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(json.loads((self.run / 'evaluation.judge-error-1.json').read_text()), failed)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'evaluated')
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.invoke()

    def test_an_explicit_rejudge_supersedes_a_final_evaluation_and_keeps_it(self):
        out = self.run / 'evaluation.json'
        final = {'verdict': 'submission_invalid', 'scorable': True, 'errors': ['old report rule']}
        out.write_text(json.dumps(final))
        with patch('oj_eval.evaluate', return_value={'verdict': 'evaluated', 'scorable': True}):
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.invoke()
            self.assertEqual(self.invoke('--rejudge', 'report rule relaxed'), 0)
        self.assertEqual(json.loads((self.run / 'evaluation.superseded-1.json').read_text()), final)
        current = json.loads(out.read_text())
        self.assertEqual(current['supersedes'], {'record': 'evaluation.superseded-1.json',
                                                 'verdict': 'submission_invalid', 'reason': 'report rule relaxed'})

    def test_tampered_submission_is_rejected_before_evaluation_or_archive(self):
        placeholder = {'verdict': 'not_evaluated'}
        out = self.run / 'evaluation.json'
        out.write_text(json.dumps(placeholder))
        (self.submission / 'model.py').write_text('tampered\n')
        with patch('oj_eval.evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'tree hash'):
                self.invoke()
            evaluate.assert_not_called()
        self.assertEqual(json.loads(out.read_text()), placeholder)
        self.assertFalse((self.run / 'evaluation.deferred.json').exists())

    def test_development_hardware_restriction_survives_reference_judge(self):
        for flags in ({'hardware_matches':False},{'validation_cpu_override':True},
                      {'shared_gpu_validation':True},{'hardware_matches':None}):
            with self.subTest(flags=flags):
                record=copy.deepcopy(self.record)
                record['sandbox'].update(flags)
                checked=apply_agent_attestation({'scorable':True,'judge_platform':'linux/amd64'},record)
                self.assertFalse(checked['scorable'])
                reason = checked['scoreability_reason']
                self.assertNotIn('non-reference', reason)
                if flags.get('shared_gpu_validation'):
                    self.assertIn('exclusive GPU allocation is not guaranteed', reason)
                elif flags.get('validation_cpu_override'):
                    self.assertIn('CPU validation override', reason)
                    # The judge gives the same reason; it is listed once.
                    judged = {'scorable':False,'judge_platform':'linux/amd64',
                              'scoreability_reason':'synthetic data; CPU validation override of a GPU case'}
                    reason = apply_agent_attestation(judged,record)['scoreability_reason']
                    self.assertEqual(reason.count('CPU validation override'), 1, reason)
                else:
                    self.assertIn('resource allocation', reason)

    def test_non_reference_gpu_model_is_reported_not_penalised(self):
        record=copy.deepcopy(self.record)
        record['sandbox'].update(gpu_class_matches=False,
                                 gpu_inventory=[{'index':'2','name':'NVIDIA RTX 6000 Ada Generation','class':'other','memory_mb':49140.}])
        checked=apply_agent_attestation({'scorable':True,'judge_platform':'linux/amd64'},record)
        self.assertTrue(checked['scorable'])
        self.assertEqual(checked['agent_hardware'],{'gpu_class_matches':False,'gpus':[
            {'name':'NVIDIA RTX 6000 Ada Generation','class':'other','memory_mb':49140.}]})

    def test_shared_guard_preserves_judge_restrictions_and_checks_egress_budget(self):
        result = {'verdict': 'evaluated', 'scorable': False, 'scoreability_reason': 'judge lock missing'}
        apply_agent_attestation(result, self.record)
        self.assertFalse(result['scorable'])
        self.assertEqual(result['scoreability_reason'], 'judge lock missing')
        for alteration in ('budget', 'egress', 'platform', 'unlocked', 'failed'):
            record = copy.deepcopy(self.record)
            if alteration == 'budget':
                record['budget']['over_budget'] = True
            elif alteration == 'egress':
                record['network']['mode'] = 'egress'
            elif alteration == 'platform':
                record['sandbox']['platform'] = 'linux/arm64'
            elif alteration == 'failed':
                record['status'] = 'failed'
            else:
                record['sandbox']['image_attestation']['locked'] = False
            checked = apply_agent_attestation({'scorable': True}, record)
            self.assertFalse(checked['scorable'], alteration)


if __name__ == '__main__':
    unittest.main()
