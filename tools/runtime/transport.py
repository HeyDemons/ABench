"""Docker CLI transport. Host files cross the boundary only as validated tar archives."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import uuid

class DockerError(RuntimeError):
    pass

class DockerTimeout(DockerError):
    """The Docker CLI did not answer in time; the daemon may still complete the request."""
    pass

class DockerPathMissing(DockerError):
    """Docker reached the container, but the requested artifact does not exist."""
    pass

def safe_extract(archive, destination, max_bytes=2 * 1024**3, max_members=100000):
    """Extract regular files/directories only, bounding both declared and copied bytes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO(archive) if isinstance(archive, bytes) else archive
    total = count = 0
    with tarfile.open(fileobj=stream, mode='r|*') as tf:
        seen = set()
        for member in tf:
            count += 1
            path = PurePosixPath(member.name)
            if (count > max_members or path.is_absolute() or '..' in path.parts
                    or not (member.isfile() or member.isdir()) or member.size < 0):
                raise ValueError(f'unsafe archive member: {member.name}')
            if not path.parts or str(path) == '.':
                continue
            if str(path) in seen:
                raise ValueError(f'duplicate archive member: {member.name}')
            seen.add(str(path))
            total += member.size
            if total > max_bytes:
                raise ValueError('archive exceeds byte limit')
            target = destination.joinpath(*path.parts)
            for parent in [target, *target.parents]:
                if parent == destination.parent:
                    break
                if parent.is_symlink():
                    raise ValueError('archive target traverses existing symlink')
            if hasattr(tarfile, 'data_filter'):
                try:
                    member = tarfile.data_filter(member, str(destination))
                except tarfile.FilterError as exc:
                    raise ValueError(f'unsafe archive member: {member.name}') from exc
                if member is None:
                    raise ValueError('archive member rejected by data filter')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                with target.open('xb') as out:
                    remaining = member.size
                    while remaining:
                        block = src.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('truncated archive member')
                        out.write(block)
                        remaining -= len(block)
                target.chmod(member.mode & 0o777 & ~0o022)
    return {'members': count, 'bytes': total}

def _write_tar(source, stream):
    source = Path(source)
    if source.is_symlink():
        raise ValueError(f'unsafe upload root: {source}')
    with tarfile.open(fileobj=stream, mode='w|') as tf:
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f'unsafe upload member: {path}')
            name = path.relative_to(source).as_posix() if source.is_dir() else path.name
            info = tf.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 1000
            info.mtime = 0
            info.uname = info.gname = 'abench'
            info.mode &= 0o777
            with path.open('rb') if path.is_file() else io.BytesIO() as data:
                tf.addfile(info, data if path.is_file() else None)

def make_tar(source):
    buf = io.BytesIO()
    _write_tar(source, buf)
    return buf.getvalue()

def content_hash(source):
    class HashSink:
        def __init__(self): self.digest = hashlib.sha256()
        def write(self, data): self.digest.update(data); return len(data)
    sink = HashSink()
    _write_tar(source, sink)
    return sink.digest.hexdigest()

class Docker:
    def __init__(self, executable='docker'):
        self.executable = executable
        self.pending_containers = []
    def run(self, args, *, input=None, check=True, text=True, timeout=120, env=None):
        try:
            # Own session: a terminal Ctrl-C reaches only ABench, which stops these commands itself, so it cannot
            # abort a shielded `docker rm` halfway.
            result = subprocess.run([self.executable, *map(str, args)], input=input, capture_output=True,
                                    text=text, timeout=timeout, env=env, start_new_session=True)
        except subprocess.TimeoutExpired as exc:
            raise DockerTimeout(str(exc)) from exc
        except OSError as exc:
            raise DockerError(str(exc)) from exc
        if check and result.returncode:
            stderr = result.stderr if text else result.stderr.decode(errors='replace')
            raise DockerError(f'docker {args[0]} failed: {stderr.strip()}')
        return result
    def run_to_file(self, args, path, *, timeout=120):
        """Run a Docker command with its stdout written to path, not held in memory; returns (returncode, stderr)."""
        with open(path, 'wb') as out:
            try:
                result = subprocess.run([self.executable, *map(str, args)], stdout=out, stderr=subprocess.PIPE,
                                        timeout=timeout, start_new_session=True)
            except subprocess.TimeoutExpired as exc:
                raise DockerTimeout(str(exc)) from exc
            except OSError as exc:
                raise DockerError(str(exc)) from exc
        return result.returncode, result.stderr.decode(errors='replace')
    def info(self):
        data = json.loads(self.run(['info', '--format', '{{json .}}']).stdout)
        return {'rootless': any('rootless' in x for x in data.get('SecurityOptions', [])),
                'desktop': 'docker desktop' in data.get('OperatingSystem', '').lower(),
                'platform': 'linux/' + {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(data.get('Architecture'), data.get('Architecture', 'unknown')),
                'version': data.get('ServerVersion'), 'cgroup_version': data.get('CgroupVersion'),
                'resources_enforced': all(data.get(key) for key in ('MemoryLimit','CpuCfsQuota','PidsLimit')) and (not any('rootless' in x for x in data.get('SecurityOptions', [])) or data.get('CgroupDriver') == 'systemd' and str(data.get('CgroupVersion')) == '2'),
                'raw': data}
    @staticmethod
    def _labels(run_id, labels=None):
        """abench.run identifies the object's owner; abench.run_id is the operator-visible run id (see gc)."""
        args = ['--label', f'abench.run={run_id}']
        for key, value in (labels or {}).items():
            args += ['--label', f'{key}={value}']
        return args
    def create(self, image, command, run_id, options=None, labels=None):
        if any(str(option).split('=', 1)[0] == '--name' for option in (options or [])):
            raise ValueError('container names are allocated by the transport')
        # A daemon may complete creation after the client times out. Keep a known, task-owned name so cleanup
        # can remove that object, or report it as uncertain, even when no container ID was returned.
        name = 'abench-' + uuid.uuid4().hex
        self.pending_containers.append(name)
        try:
            result = self.run(['create', '--name', name, *self._labels(run_id, labels), *(options or []), image, *command])
        except DockerTimeout:
            raise
        except DockerError:
            self.pending_containers.remove(name)     # the daemon answered with an error: nothing was created
            raise
        cid = result.stdout.strip()
        if not cid:
            raise DockerError(f'container creation returned no ID: {name}')
        self.pending_containers.remove(name)
        return cid
    def start(self, container):
        self.run(['start', container])
    def attach(self, container):
        return subprocess.Popen([self.executable, 'start', '-ai', container], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def exec(self, container, command, *, user=None, input=None, env=None, timeout=120, check=True, text=True):
        args = ['exec', '-i']
        if user:
            args += ['--user', str(user)]
        for key in (env or {}):
            args += ['-e', key]
        return self.run([*args, container, *command], input=input, timeout=timeout, check=check,
                        text=text, env={**os.environ, **(env or {})})
    def inspect(self, container):
        return json.loads(self.run(['inspect', container]).stdout)[0]
    def upload(self, container, source, destination, *, timeout=7200):
        from runtime.transfers import copy_archive
        return copy_archive([self.executable, 'cp', '-', f'{container}:{destination}'],
                            lambda process: _write_tar(source, process.stdin), upload=True, timeout=timeout)
    def download(self, container, source, destination, *, contents=True, timeout=120, **limits):
        from runtime.transfers import copy_archive
        source_path = source.rstrip('/') + ('/.' if contents else '')
        return copy_archive([self.executable, 'cp', f'{container}:{source_path}', '-'],
                            lambda process: safe_extract(process.stdout, destination, **limits),
                            upload=False, timeout=timeout)
    def kill(self, container):
        self.run(['kill', container], check=False)
    def remove(self, container):
        return self.run(['rm', '-f', '-v', container], check=False)
    def volume(self, name, run_id, labels=None):
        return self.run(['volume', 'create', *self._labels(run_id, labels), name]).stdout.strip()
    def remove_volume(self, name):
        return self.run(['volume', 'rm', name], check=False)
    def _list(self, kind, label):
        command = ['ps', '-aq'] if kind == 'container' else [kind, 'ls', '-q']
        return self.run([*command, '--filter', f'label={label}']).stdout.split()
    def gc(self, run_id=None, *, caches=False, min_age=600.0):
        """Remove ABench Docker objects.

        With run_id: every container, network and volume of that run, matched by the operator-visible run id
        (abench.run_id) or by the exact owner label. Without: only leftovers of crashed runs, i.e. containers that
        are not running and volumes/networks no container uses, all older than min_age seconds so that an active
        run between creating and starting an object is never touched. Content caches (abench-data-*) are shared by
        runs and kept unless caches=True."""
        import re
        from datetime import datetime, timezone
        def age(stamp):
            # Docker RFC 3339 times, e.g. 2026-10-29T02:03:04.123456789Z or 2026-10-29T10:03:04+08:00.
            # Anything unparsable counts as new, so it is never removed by mistake.
            match = re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?(Z|[+-]\d\d:\d\d)', stamp or '')
            if not match:
                return 0.0
            moment = datetime.fromisoformat(match[1] + ('+00:00' if match[2] == 'Z' else match[2]))
            return (datetime.now(timezone.utc) - moment).total_seconds()
        from agents.lifecycle import protected_objects
        retained = protected_objects(self)
        removed = {}
        for kind in ['container', 'network', 'volume']:
            if run_id:
                ids = list(dict.fromkeys(self._list(kind, f'abench.run_id={run_id}') + self._list(kind, f'abench.run={run_id}')))
            else:
                ids = []
                for oid in self._list(kind, 'abench.run'):
                    meta = json.loads(self.run([kind, 'inspect', oid], check=False).stdout or '[{}]')[0]
                    labels = ((meta.get('Config') or {}).get('Labels') if kind == 'container' else meta.get('Labels')) or {}
                    if (labels.get('abench.cache') or str(labels.get('abench.run', '')).startswith('cache-')) and not caches:
                        continue
                    if kind == 'container':
                        state = meta.get('State') or {}
                        if state.get('Running') or state.get('Restarting') or state.get('Paused'):
                            continue
                        stamp = state.get('FinishedAt') if state.get('Status') in ('exited', 'dead') else meta.get('Created')
                    else:
                        stamp = meta.get('Created') or meta.get('CreatedAt')
                    if age(stamp) >= min_age:
                        ids.append(oid)
            gone = []
            for oid in ids:
                if oid in retained.get(kind + 's', ()):
                    continue
                result = self.run(['rm', '-f', '-v', oid] if kind == 'container' else [kind, 'rm', oid], check=False)
                if result.returncode == 0:       # an in-use volume or network is refused by Docker and kept
                    gone.append(oid)
            removed[kind] = gone
        return removed
    cleanup = gc
    def cached_volume(self, source, image):
        from runtime.cache import for_docker
        return for_docker(self).directory(source, image)

    def cached_file_volume(self, sources, image):
        from runtime.cache import for_docker
        return for_docker(self).files(sources, image)
