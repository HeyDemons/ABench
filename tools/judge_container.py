"""Fresh Docker transport for each judge stage. Labels and keys never enter this module."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from containers import Docker, DockerError, DockerPathMissing
from container_policy import create_options, host_scaled, resources_for, verify_inspect
from runtime.cleanup import cleanup_objects, shielded, stop_container

LOG_MAX_BYTES = 64 * 1024**2      # a stage log only feeds the record's tails; outputs keep their own, larger cap


class DockerStageLauncher:
    def __init__(self, image, resources, *, gpus=None, locked=False, docker=None, run_id=None):
        self.image, self.resources, self.gpus, self.locked = image, resources, gpus, locked
        self.lock = 'published' if locked else None
        # The agent run's id, so `abench gc --run-id` also finds this run's judge objects.
        self.labels = {'abench.run_id': run_id} if run_id else {}
        self.docker = docker or Docker()
        self.docker_info = self.docker.info()
        self.resources = host_scaled(self.resources, self.docker_info)
        self.platform = self.docker_info['platform']
        self.resources_enforced = self.docker_info.get('resources_enforced', True)
        actual_image = self.docker.inspect(image)
        self.image_id = actual_image.get('Id')
        self.digest = next(iter(actual_image.get('RepoDigests') or []), self.image_id)
        self.records = []
        self.inputs = {}             # (harness, stage runner, submission) -> read-only input volume of this evaluation
        # Input objects not yet confirmed removed; a failed removal keeps them listed for close_inputs to retry.
        self.input_helpers, self.input_volumes, self.input_failures = [], [], []
        self.input_record = {}
        self.hardware_matches = self._hardware_matches()

    @classmethod
    def for_case(cls, oj, *, gpus=None, image=None, run_id=None):
        from images import resolve_image
        image = image or os.environ.get('ABENCH_JUDGE_IMAGE')
        # The case's pinned profile version, not merely the current definition of the profile.
        profile = oj.get('environment') or oj.get('profile')
        try:
            # Strict resolution reads only images.lock.yaml; an ABENCH_IMAGE_* override, even one naming a
            # digest, never counts as a published lock.
            published = resolve_image(profile)
        except (ValueError, KeyError, FileNotFoundError):
            published = None
        try:
            resolved = published or resolve_image(profile, allow_unlocked=True)
        except (ValueError, KeyError, FileNotFoundError):
            if not image:
                raise
            resolved = None
        reference = image or resolved
        locked = bool(published and reference == published)
        resources = resources_for(oj)
        cpu_override = os.environ.get('ABENCH_CALIBRATION_CPU') == '1' and int(resources.get('gpus',0)) > 0
        if cpu_override:
            resources = {**resources,'gpus':0}
        launcher = cls(reference, resources, gpus=gpus, locked=locked, run_id=run_id)
        if not locked:
            from images import local_lock
            entry = local_lock(profile, platform=launcher.platform)
            if entry and entry.get('image_id') == launcher.image_id:
                # A host-local lock names this exact image ID: it scores, recorded as unpublished.
                launcher.locked, launcher.lock = True, 'local'
        if cpu_override:
            launcher.hardware_matches = False
            launcher.hardware_reason = 'CPU validation override of a GPU case'
        return launcher

    def _hardware_matches(self):
        """Hardware for scoring, judged with the scheduler's own inventory: the case's GPU count, idle when judging
        starts (a busy or shared card distorts timed metrics, e.g. a deferred judge on a card that is now in use),
        and for Pro the exclusive 4-GPU node. Sets hardware_reason when not. A GPU model other than the case's
        reference class does not affect scoring; gpu_class_matches reports it."""
        self.hardware_reason = None
        self.gpu_class_matches = True
        count = int(self.resources.get('gpus', 0))
        if not count:
            return True
        ids = [str(i) for i in (self.gpus.split(',') if isinstance(self.gpus, str) else list(self.gpus or []))]
        from jobs import gpu_inventory
        inventory = {g['index']: g for g in gpu_inventory()}
        self.gpu_inventory = [inventory.get(i) for i in ids]
        if len(ids) != count or any(i not in inventory for i in ids):
            self.hardware_reason = 'allocated GPUs do not match the case resources'
            return False
        self.gpu_class_matches = all(inventory[i]['class'] == self.resources.get('gpu_class') for i in ids)
        if not all(inventory[i]['idle'] for i in ids):
            self.hardware_reason = 'judge GPU was in use when judging started'
        elif self.resources.get('budget_tier') == 'pro' and len(inventory) != 4:
            self.hardware_reason = 'Pro requires an exclusive node with 4 GPUs'
        return self.hardware_reason is None

    def input_volume(self, harness, kit, submission):
        """The case harness, the stage runner and the submission, uploaded once per evaluation and mounted read-only
        into every fresh stage container of it, so no stage can change them for a later stage or seed."""
        key = (str(harness), str(kit), str(submission))
        if key not in self.inputs:
            run_id = 'judge-input-' + uuid.uuid4().hex
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='abench-stage-input-') as raw:
                bundle = Path(raw)
                shutil.copytree(harness, bundle / 'harness', ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copyfile(kit, bundle / 'harness/abench_stage.py')
                shutil.copytree(submission, bundle / 'submission', ignore=shutil.ignore_patterns('__pycache__'))
                volume = self.docker.volume('abench-' + run_id, run_id, self.labels)
                self.input_volumes.append(volume)      # removed by close_inputs, also after a failed upload
                try:
                    self.input_helpers.append(self.docker.create(
                        self.image, ['true'], run_id,
                        ['--network', 'none', '--mount', f'type=volume,src={volume},dst=/abench/view'], self.labels))
                    self.docker.upload(self.input_helpers[-1], bundle, '/abench/view')
                finally:
                    self._remove_inputs([])
            if self.input_helpers:
                raise DockerError('judge input helper could not be removed')
            self.inputs[key] = volume
            self.input_record.setdefault('upload_seconds', []).append(round(time.perf_counter() - started, 4))
        return self.inputs[key]

    def close_inputs(self):
        """Remove the evaluation's input volumes once all of its stages are done (also after a failure). What cannot
        be removed stays listed, so calling this again retries it and the record keeps reporting it."""
        self.inputs.clear()
        self._remove_inputs(self.input_volumes)

    def _remove_inputs(self, volumes):
        report = cleanup_objects(self.docker, self.input_helpers, volumes, self.input_record)
        # cleanup_objects reports only this attempt: keep every failure and list all input objects still present.
        self.input_failures.extend(report['failed'])
        report['failed'] = list(self.input_failures)
        report['remaining'] = {'containers': list(self.input_helpers), 'volumes': list(self.input_volumes)}

    def start(self, **config):
        session = DockerStageSession(self, **config)
        return session


class DockerStageSession:
    def __init__(self, launcher, *, cmd, work, env, stages_path, submission, public, models, artifacts=()):
        self.launcher, self.docker, self.work = launcher, launcher.docker, Path(work)
        self.run_id = 'judge-' + uuid.uuid4().hex
        self.container = self.input_volume = None
        self.stopped = None             # the container close() confirmed stopped
        self.helpers = []
        self.proc = None
        self.record = {}
        self.artifacts = artifacts
        self.stage = cmd[cmd.index('--stage') + 1]
        launcher.records.append(self.record)
        try:
            # Shared by the stages and seeds of this evaluation and owned by the launcher (close_inputs).
            with self._timed('inputs'):
                inputs = launcher.input_volume(stages_path.parent, cmd[1], submission)
            mounts = [f'type=volume,src={inputs},dst=/abench/view,readonly']
            translations = {str(stages_path.parent): '/abench/view/harness', str(submission): '/abench/view/submission',
                            str(work): '/abench/workspace'}
            cache_start = len(getattr(self.docker, 'cache_records', []))
            with self._timed('data_caches'):
                for source, target in ((public, '/abench/data'), (models, '/abench/models')):
                    if source is not None:
                        volume = self.docker.cached_volume(source, launcher.image)
                        mounts.append(f'type=volume,src={volume},dst={target},readonly')
                        translations[str(source)] = target
            self.record['data_caches'] = getattr(self.docker, 'cache_records', [])[cache_start:]
            def translate(value):
                for host, container in sorted(translations.items(), key=lambda x: -len(x[0])):
                    if value == host or value.startswith(host + '/'):
                        return container + value[len(host):]
                return value
            args = ['python3', '/abench/view/harness/abench_stage.py', *[translate(str(a)) for a in cmd[2:]]]
            options = create_options(launcher.resources, launcher.gpus, mounts=mounts,docker_info=launcher.docker_info)
            options += ['-i', '--workdir', '/abench/workspace']
            for key, val in env.items():
                if key in ('PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES'):
                    continue
                options += ['--env', key + '=' + translate(str(val))]
            with self._timed('create'):
                self.container = self.docker.create(launcher.image, args, self.run_id, options, launcher.labels)
            with self._timed('upload'):
                self.docker.upload(self.container, self.work, '/abench/workspace')
            with self._timed('verify'):
                checked = verify_inspect(self.docker.inspect(self.container), image=launcher.image,
                                         mounts=mounts, resources=launcher.resources, gpus=launcher.gpus, image_id=launcher.image_id, resources_enforced=launcher.resources_enforced)
            if not checked['verified']:
                raise RuntimeError('judge container configuration rejected: ' + ', '.join(checked['failures']))
            self.record.update(container=self.container, image=launcher.image, inspect=checked)
            with self._timed('attach'):
                self.proc = self.docker.attach(self.container)
            self.protocol_fd = self.proc.stdout.fileno()
        except BaseException:
            self._remove()
            raise

    @contextmanager
    def _timed(self, step):
        """Wall time of one Docker step of this stage, recorded to find bottlenecks; never part of a metric."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record.setdefault('seconds', {})[step] = round(time.perf_counter() - started, 4)

    def stage_post(self, files):
        with self._timed('post'):
            self.docker.exec(self.container, ['python3', '-c',
                "import os,shutil; p='/abench/workspace/post'; "
                "os.unlink(p) if os.path.islink(p) else shutil.rmtree(p,ignore_errors=True); os.mkdir(p,0o755)"])
            with tempfile.TemporaryDirectory(prefix='abench-stage-post-') as raw:
                for name, source in files.items():
                    shutil.copyfile(source, Path(raw) / name)
                self.docker.upload(self.container, Path(raw), '/abench/workspace/post')

    @shielded()
    def close(self):
        if self.container:
            try:
                state = stop_container(self.docker, self.container)
                self.stopped = self.container
                self.record['stopped_before_collection'] = True
                self.record['container_oom_killed'] = bool(state.get('OOMKilled'))
            except Exception as exc:
                self.record['stop_error'] = str(exc)
                self._remove()
                if self.proc and self.proc.poll() is None:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
                    except Exception as reap_error:
                        self.record['reap_error'] = str(reap_error)
                return  # collection independently refuses a missing or still-running container
        if self.proc and self.proc.poll() is None:
            try:
                self.docker.kill(self.container)
                self.proc.wait(timeout=30)
            except Exception as exc:
                self.record['stop_error'] = str(exc)
                self._remove()
                if self.proc.poll() is None:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
                    except Exception as reap_error:
                        self.record['reap_error'] = str(reap_error)

    def collect(self):
        try:
            if self.container is None:
                raise DockerError('judge container removed before output collection')
            if self.stopped != self.container:
                stop_container(self.docker, self.container)
                self.record['stopped_before_collection'] = True
            started = time.perf_counter()
            stage = self.stage
            names = [f'{stage}.out.npz', f'{stage}.meta.json', '_stdout.txt', '_stderr.txt', *self.artifacts]
            with tempfile.TemporaryDirectory(prefix='abench-stage-output-') as raw:
                for name in names:
                    path = Path(name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('unsafe stage artifact path')
                    destination = Path(raw) / path.parent
                    destination.mkdir(parents=True, exist_ok=True)
                    log = name in ('_stdout.txt', '_stderr.txt')
                    try:
                        self.docker.download(self.container, '/abench/workspace/' + name, destination,
                                             contents=False, max_bytes=LOG_MAX_BYTES if log else 8 * 1024**3)
                    except DockerPathMissing:
                        # Only a confirmed missing artifact is represented by the stage's exit status.
                        pass
                    except ValueError as exc:
                        if not (log and 'byte limit' in str(exc)):
                            raise
                        self.record.setdefault('logs_not_collected', []).append(
                            {'name': name, 'reason': f'over {LOG_MAX_BYTES} bytes'})
                for item in Path(raw).iterdir():
                    target = self.work / item.name
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists():
                        target.unlink()
                    shutil.move(str(item), target)
            self.record.setdefault('seconds', {})['download'] = round(time.perf_counter() - started, 4)
            return {'sandbox': self.record}
        finally:
            with self._timed('remove'):
                self._remove()

    def _remove(self):
        containers = ([self.container] if self.container else []) + self.helpers
        volumes = [self.input_volume] if self.input_volume else []
        cleanup_objects(self.docker, containers, volumes, self.record)
        if self.container not in containers:
            self.container = None
        self.helpers = [cid for cid in containers if cid != self.container]
        if self.input_volume not in volumes:
            self.input_volume = None
