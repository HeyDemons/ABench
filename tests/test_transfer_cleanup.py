"""Exercise blocked real pipes and failed Docker deletions without a Docker daemon."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from containers import Docker, DockerError, DockerPathMissing, DockerTimeout
from runtime.agent import DockerAgentEnvironment
from runtime.cleanup import cleanup_objects, shielded, sigterm_interrupts
from judge_container import DockerStageSession


class ArchiveDeadlineTest(unittest.TestCase):
    def script(self, root, program):
        executable = root / 'fake-docker'
        executable.write_text(f'#!{sys.executable}\n{program}\n')
        executable.chmod(0o700)
        return Docker(str(executable))

    def test_deadline_covers_blocked_upload_and_download(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / 'source'
            source.mkdir()
            (source / 'large').write_bytes(b'x' * 2 * 1024**2)
            docker = self.script(root, 'import time; time.sleep(30)')
            for operation in (lambda: docker.upload('cid', source, '/target', timeout=.15),
                              lambda: docker.download('cid', '/source', root / 'output', timeout=.15)):
                started = time.monotonic()
                with self.assertRaisesRegex(DockerError, 'timed out'):
                    operation()
                self.assertLess(time.monotonic() - started, 3)

    def test_large_stderr_cannot_block_a_transfer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            docker = self.script(root, "import sys; sys.stderr.write('daemon failed\\n' * 100000); sys.exit(1)")
            with self.assertRaisesRegex(DockerError, 'daemon failed'):
                docker.download('cid', '/artifact', root / 'out', timeout=2)

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires POSIX process groups')
    def test_deadline_kills_descendants_after_the_cli_parent_exits(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            docker = self.script(root, 'import os,time\nif os.fork(): os._exit(0)\ntime.sleep(30)')
            started = time.monotonic()
            with self.assertRaisesRegex(DockerError, 'timed out'):
                docker.download('cid', '/artifact', root / 'out', timeout=.25)
            self.assertLess(time.monotonic() - started, 3)

    def test_missing_artifact_is_distinct_from_daemon_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for message, error in (('Error response from daemon: Could not find the file /artifact in container cid', DockerPathMissing),
                                   ('Cannot connect to the Docker daemon', DockerError)):
                docker = self.script(root, f'import sys; sys.stderr.write({message!r}); sys.exit(1)')
                with self.assertRaises(error) as caught:
                    docker.download('cid', '/artifact', root / 'out', timeout=2)
                if error is DockerError:
                    self.assertNotIsInstance(caught.exception, DockerPathMissing)


class CleanupEvidenceTest(unittest.TestCase):
    def test_timed_out_create_keeps_a_name_for_late_daemon_completion(self):
        docker = Docker()
        with unittest.mock.patch('runtime.transport.subprocess.run',
                                 side_effect=subprocess.TimeoutExpired(['docker', 'create'], 120)):
            with self.assertRaises(DockerTimeout):
                docker.create('image', ['true'], 'owned-run')
        self.assertEqual(len(docker.pending_containers), 1)
        name = docker.pending_containers[0]
        self.assertTrue(name.startswith('abench-'))
        docker.run = Mock()
        # Real `docker rm -f` exits 0 for an absent name: while the daemon may still be creating it, that absence
        # must not count as removal.
        docker.run.return_value = SimpleNamespace(returncode=0, stderr=f'Error response from daemon: No such container: {name}')
        containers, volumes, record = [], [], {}
        cleanup_objects(docker, containers, volumes, record)
        self.assertEqual(record['cleanup']['remaining']['containers'], [name])
        self.assertEqual(docker.pending_containers, [name])
        docker.run.return_value = SimpleNamespace(returncode=0, stderr='')     # it appeared and was removed
        cleanup_objects(docker, containers, volumes, record)
        self.assertEqual(docker.pending_containers, [])
        self.assertEqual(record['cleanup']['remaining']['containers'], [])
        self.assertIn(name, record['cleanup']['removed']['containers'])

    def test_refused_create_leaves_nothing_pending(self):
        docker = Docker()
        docker.run = Mock(side_effect=DockerError('docker create failed: No such image: missing'))
        with self.assertRaisesRegex(DockerError, 'No such image'):
            docker.create('missing', ['true'], 'owned-run')
        self.assertEqual(docker.pending_containers, [])
        self.assertEqual(docker.run.call_args.args[0][1:3][0], '--name')

    def test_successful_create_clears_uncertainty_and_external_names_are_refused(self):
        docker = Docker()
        docker.run = Mock(return_value=SimpleNamespace(stdout='cid\n'))
        self.assertEqual(docker.create('image', ['true'], 'owned-run'), 'cid')
        self.assertEqual(docker.pending_containers, [])
        for option in (['--name', 'foreign'], ['--name=foreign']):
            with self.assertRaises(ValueError):
                docker.create('image', ['true'], 'owned-run', option)

    def test_artifacts_are_not_collected_from_a_running_agent_after_failed_stop(self):
        environment = DockerAgentEnvironment.__new__(DockerAgentEnvironment)
        environment.cid, environment.record = 'cid', {}
        environment.docker = Mock()
        environment.docker.inspect.return_value = {'State': {'Running': True}}
        environment.docker.exec.return_value = SimpleNamespace(stdout='usage_usec 10\nMEMORY\n20\nEVENTS\noom_kill 0')
        with self.assertRaisesRegex(DockerError, 'still running'):
            environment.collect()
        environment.docker.download.assert_not_called()

    def test_judge_shutdown_failures_do_not_replace_original_stage_error(self):
        session = DockerStageSession.__new__(DockerStageSession)
        session.docker = Mock()
        session.docker.kill.side_effect = DockerError('daemon disconnected')
        session.docker.remove.side_effect = DockerError('still disconnected')
        session.docker.remove_volume.side_effect = DockerError('still disconnected')
        session.proc = Mock()
        session.proc.poll.return_value = None
        session.proc.kill.side_effect = OSError('cannot reap local client')
        session.container, session.input_volume, session.helpers, session.record = 'cid', 'volume', [], {}
        session.stopped = None
        with self.assertRaisesRegex(ValueError, 'original stage error'):
            try:
                raise ValueError('original stage error')
            finally:
                session.close()
        self.assertIn('cannot reap', session.record['reap_error'])
        self.assertEqual(session.record['cleanup']['remaining']['containers'], ['cid'])

    def test_failures_are_retained_and_other_objects_are_still_attempted(self):
        docker = Mock()
        docker.remove.side_effect = [SimpleNamespace(returncode=1, stderr='busy'), DockerError('disconnected'),
                                     SimpleNamespace(returncode=0)]
        docker.remove_volume.return_value = SimpleNamespace(returncode=0)
        containers, volumes, record = ['good', 'disconnected', 'busy'], ['volume'], {}
        report = cleanup_objects(docker, containers, volumes, record)
        self.assertEqual(set(containers), {'busy', 'disconnected'})
        self.assertEqual(volumes, [])
        self.assertEqual(report['containers'], 1)
        self.assertEqual(report['volumes'], 1)
        self.assertEqual(report['removed']['containers'], ['good'])
        self.assertEqual(len(report['failed']), 2)
        docker.remove.side_effect = None
        docker.remove.return_value = SimpleNamespace(returncode=0)
        cleanup_objects(docker, containers, volumes, record)
        self.assertEqual(containers, [])
        self.assertEqual(report['containers'], 3)
        self.assertEqual(report['failed'], [])
        cleanup_objects(docker, containers, volumes, record)
        self.assertEqual(report['containers'], 3)

    def test_agent_close_preserves_the_original_error_and_remaining_ids(self):
        environment = DockerAgentEnvironment.__new__(DockerAgentEnvironment)
        environment.docker = Mock()
        environment.docker.remove.side_effect = DockerError('daemon unavailable')
        environment.docker.remove_volume.return_value = SimpleNamespace(returncode=0)
        environment.ids, environment.volumes, environment.record = ['cid'], ['volume'], {}
        with self.assertRaisesRegex(ValueError, 'original agent error'):
            try:
                raise ValueError('original agent error')
            finally:
                environment.close()
        self.assertEqual(environment.ids, ['cid'])
        self.assertEqual(environment.record['cleanup']['containers'], 0)
        self.assertEqual(environment.record['cleanup']['remaining']['containers'], ['cid'])

    def test_judge_collection_suppresses_only_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as raw:
            for error in (DockerPathMissing('absent'), DockerError('daemon disconnected')):
                session = DockerStageSession.__new__(DockerStageSession)
                session.docker = Mock()
                session.docker.inspect.return_value = {'State': {'Running': False}}
                session.docker.download.side_effect = error
                session.docker.remove.return_value = SimpleNamespace(returncode=0)
                session.docker.remove_volume.return_value = SimpleNamespace(returncode=0)
                session.container, session.input_volume, session.helpers = 'cid', 'volume', []
                session.stopped = None
                session.stage, session.artifacts, session.work, session.record = 'predict', (), Path(raw), {}
                if isinstance(error, DockerPathMissing):
                    session.collect()
                else:
                    with self.assertRaisesRegex(DockerError, 'daemon disconnected'):
                        session.collect()
                self.assertEqual(session.record['cleanup']['containers'], 1)
                self.assertEqual(session.record['cleanup']['volumes'], 1)

    def test_judge_client_exit_does_not_mean_container_exit(self):
        session = DockerStageSession.__new__(DockerStageSession)
        session.docker = Mock()
        session.docker.inspect.side_effect = [{'State': {'Running': True}}, {'State': {'Running': False}}]
        session.proc = Mock()
        session.proc.poll.return_value = 1  # Docker client disconnected, container is still running.
        session.container, session.input_volume, session.helpers, session.record = 'cid', None, [], {}
        session.stopped = None
        session.close()
        session.docker.kill.assert_called_once_with('cid')
        self.assertTrue(session.record['stopped_before_collection'])
        self.assertEqual(session.stopped, 'cid')         # collect() does not ask Docker again

    def test_judge_refuses_to_download_from_running_container(self):
        session = DockerStageSession.__new__(DockerStageSession)
        session.docker = Mock()
        session.docker.inspect.return_value = {'State': {'Running': True}}
        session.docker.remove.return_value = SimpleNamespace(returncode=0)
        session.container, session.input_volume, session.helpers, session.record = 'cid', None, [], {}
        session.stopped = None
        with self.assertRaisesRegex(DockerError, 'still running'):
            session.collect()
        session.docker.download.assert_not_called()

    def test_oversized_stage_logs_are_skipped_but_outputs_keep_their_cap(self):
        from judge_container import LOG_MAX_BYTES
        def download(container, source, destination, **limits):
            if limits['max_bytes'] == LOG_MAX_BYTES:
                raise ValueError('archive exceeds byte limit')
        with tempfile.TemporaryDirectory() as raw:
            session = DockerStageSession.__new__(DockerStageSession)
            session.docker = Mock()
            session.docker.inspect.return_value = {'State': {'Running': False}}
            session.docker.download.side_effect = download
            session.docker.remove.return_value = SimpleNamespace(returncode=0)
            session.container, session.input_volume, session.helpers = 'cid', None, []
            session.stopped = None
            session.stage, session.artifacts, session.work, session.record = 'predict', (), Path(raw), {}
            session.collect()
            self.assertEqual([x['name'] for x in session.record['logs_not_collected']], ['_stdout.txt', '_stderr.txt'])
            caps = {call.args[1].rsplit('/', 1)[1]: call.kwargs['max_bytes'] for call in session.docker.download.call_args_list}
            self.assertEqual(caps['predict.out.npz'], 8 * 1024**3)
            session.container = 'next-cid'  # collect() removed the previous stage container.
            session.docker.download.side_effect = ValueError('archive exceeds byte limit')
            with self.assertRaisesRegex(ValueError, 'byte limit'):          # an oversized output is still an error
                session.collect()


class CleanupShieldTest(unittest.TestCase):
    """Stopping a run (Ctrl-C, or SIGTERM from a job) never interrupts the removal of its Docker objects."""
    def setUp(self):
        self.received = []
        self.previous = {sig: signal.signal(sig, lambda number, frame: self.received.append(number))
                         for sig in (signal.SIGINT, signal.SIGTERM)}

    def tearDown(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def test_signals_during_cleanup_are_held_and_delivered_once_after_it(self):
        removed = []
        docker = Mock(pending_containers=[])
        def remove(identifier):
            os.kill(os.getpid(), signal.SIGINT)                   # Ctrl-C, then a job's SIGTERM, mid-cleanup
            os.kill(os.getpid(), signal.SIGTERM)
            removed.append(identifier)
            return SimpleNamespace(returncode=0)
        docker.remove.side_effect = docker.remove_volume.side_effect = remove
        record = {}
        cleanup_objects(docker, ['c1', 'c2'], ['v1'], record)
        self.assertEqual(removed, ['c2', 'c1', 'v1'])
        self.assertEqual(record['cleanup']['remaining'], {'containers': [], 'volumes': []})
        self.assertEqual(self.received, [signal.SIGINT])         # the first one, after the cleanup

    def test_held_signal_reaches_the_handler_restored_after_the_cleanup(self):
        after = []
        with shielded(restore={signal.SIGTERM: lambda number, frame: after.append(number)}):
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertEqual(after, [])
        self.assertEqual((after, self.received), ([signal.SIGTERM], []))

    def test_sigterm_runs_finally_blocks(self):
        collector = signal.getsignal(signal.SIGTERM)
        cleaned = []
        with self.assertRaises(KeyboardInterrupt):
            with sigterm_interrupts():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(5)
                finally:
                    cleaned.append(True)
        self.assertEqual(cleaned, [True])
        self.assertIs(signal.getsignal(signal.SIGTERM), collector)


class SharedJudgeInputTest(unittest.TestCase):
    """One read-only input volume per (harness, stage runner, submission) and evaluation, removed at its end."""
    def test_stages_and_seeds_share_one_upload(self):
        from judge_container import DockerStageLauncher
        launcher = DockerStageLauncher.__new__(DockerStageLauncher)
        launcher.docker, launcher.image, launcher.labels = Mock(), 'image', {'abench.run_id': 'r1'}
        launcher.inputs, launcher.input_record = {}, {}
        launcher.input_helpers, launcher.input_volumes, launcher.input_failures = [], [], []
        launcher.docker.volume.side_effect = lambda name, run_id, labels: name
        launcher.docker.create.side_effect = ['helper-1', 'helper-2']
        launcher.docker.remove.return_value = launcher.docker.remove_volume.return_value = SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for d in ('harness', 'submission', 'baseline'):
                (root / d).mkdir()
                (root / d / 'x.py').write_text('pass\n')
            (root / 'kit.py').write_text('pass\n')
            first = launcher.input_volume(root / 'harness', root / 'kit.py', root / 'submission')
            self.assertEqual(launcher.input_volume(root / 'harness', root / 'kit.py', root / 'submission'), first)
            other = launcher.input_volume(root / 'harness', root / 'kit.py', root / 'baseline')
        self.assertNotEqual(first, other)
        self.assertEqual(launcher.docker.upload.call_count, 2)
        self.assertEqual(launcher.input_record['cleanup']['removed']['containers'], ['helper-1', 'helper-2'])
        launcher.close_inputs()
        self.assertEqual(sorted(launcher.input_record['cleanup']['removed']['volumes']), sorted([first, other]))
        self.assertEqual(launcher.inputs, {})
        self.assertEqual(launcher.input_record['cleanup']['remaining'], {'containers': [], 'volumes': []})

    def test_objects_that_could_not_be_removed_stay_pending_and_reported(self):
        from containers import DockerError
        from judge_container import DockerStageLauncher
        launcher = DockerStageLauncher.__new__(DockerStageLauncher)
        launcher.docker, launcher.image, launcher.labels = Mock(), 'image', {'abench.run_id': 'r1'}
        launcher.inputs, launcher.input_record = {}, {}
        launcher.input_helpers, launcher.input_volumes, launcher.input_failures = [], [], []
        launcher.docker.volume.side_effect = lambda name, run_id, labels: name
        launcher.docker.create.return_value = 'helper-1'
        busy, ok = SimpleNamespace(returncode=1, stderr='device or resource busy'), SimpleNamespace(returncode=0)
        launcher.docker.remove.side_effect = [busy, busy, ok]
        launcher.docker.remove_volume.side_effect = [busy, ok]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for d in ('harness', 'submission'):
                (root / d).mkdir()
            (root / 'kit.py').write_text('pass\n')
            with self.assertRaisesRegex(DockerError, 'helper could not be removed'):
                launcher.input_volume(root / 'harness', root / 'kit.py', root / 'submission')
        self.assertEqual(launcher.inputs, {})                     # a failed input is never reused
        volume = launcher.input_volumes[0]
        report = launcher.input_record['cleanup']
        self.assertEqual(report['remaining'], {'containers': ['helper-1'], 'volumes': [volume]})
        launcher.close_inputs()                                   # both removals fail this time
        self.assertEqual(report['remaining'], {'containers': ['helper-1'], 'volumes': [volume]})
        self.assertEqual([f['id'] for f in report['failed']], ['helper-1', 'helper-1', volume])
        launcher.close_inputs()                                   # the retry succeeds; the failures stay recorded
        self.assertEqual(report['remaining'], {'containers': [], 'volumes': []})
        self.assertEqual((report['removed']['containers'], report['removed']['volumes']), (['helper-1'], [volume]))
        self.assertEqual(len(report['failed']), 3)
        launcher.close_inputs()
        self.assertEqual(launcher.docker.remove.call_count + launcher.docker.remove_volume.call_count, 5)


if __name__ == '__main__':
    unittest.main()
