"""Resume only stopped-container collection; never launch or meter an agent again."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import shutil
import stat

from agents.lifecycle import atomic_json, read_json, register_recovery, unregister_recovery
from file_hashes import sha256_file


def _missing_object(error):
    return bool(re.search(r'\bno such\s+(?:container|object|volume)\b', str(error), re.I))


@contextmanager
def run_lock(result):
    result = Path(result)
    result.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (result / '.operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pending_path(result):
    return Path(result) / 'collection_pending.json'


def _clean_values(document, secrets):
    if isinstance(document, dict):
        return {key: _clean_values(value, secrets) for key, value in document.items()}
    if isinstance(document, list):
        return [_clean_values(value, secrets) for value in document]
    if isinstance(document, str):
        for value in secrets:
            if value:
                document = document.replace(value, '<credential withheld>')
    return document


def save_execution(result, context, environment, *, secrets=()):
    """Commit execution state only after shutdown; context contains names, not environment values."""
    environment.prepare_collection()
    daemon = environment.docker.info()['raw'].get('ID')
    if not daemon:
        raise ValueError('collection recovery requires a verified Docker daemon identity')
    snapshot = {'daemon': daemon, 'case': context['args']['case'], 'run_id': context['args']['run_id'],
                'label': environment.label, 'cid': environment.cid, 'sidecar': environment.sidecar,
                'ids': list(environment.ids), 'volumes': list(environment.volumes),
                'image': environment.image, 'root': str(environment.host_paths['root']),
                'record': environment.record}
    # Never persist launch.env, authentication stdin, pass-env values, hidden keys or their filenames.
    context = _clean_values(context, secrets)
    context['stdout_sha256'] = sha256_file(Path(result) / 'logs/agent_stdout.log')
    context['stderr_sha256'] = sha256_file(Path(result) / 'logs/agent_stderr.log')
    register_recovery(result, snapshot)        # write-ahead GC protection, including the pending-file commit gap
    try:
        atomic_json(pending_path(result), {'schema': 'abench.collection_pending/v1', 'execution_complete': True,
                                          'context': context, 'environment': _clean_values(snapshot, secrets)})
    except BaseException:
        unregister_recovery(snapshot)
        raise
    return snapshot


def collected(result, environment, artifacts):
    pending = read_json(pending_path(result))
    root = Path(pending['environment']['root'])
    pending['collected'] = {name: str(path.relative_to(root)) for name, path in artifacts.items() if path is not None}
    pending['collection_complete'] = True
    receipt = root / '.collection-receipts.json'
    pending['receipts_sha256'] = sha256_file(receipt)
    pending['environment']['record'] = environment.record
    atomic_json(pending_path(result), pending)


def cleanup_state(result, environment):
    pending = read_json(pending_path(result))
    pending['environment'].update(ids=list(environment.ids), volumes=list(environment.volumes), record=environment.record)
    atomic_json(pending_path(result), pending)


def collected_paths(result):
    pending = read_json(pending_path(result))
    if not pending.get('collection_complete'):
        return None
    root = Path(pending['environment']['root'])
    receipt = root / '.collection-receipts.json'
    if receipt.is_symlink() or sha256_file(receipt) != pending['receipts_sha256']:
        raise ValueError('collection receipts changed before recovery')
    for row in read_json(receipt).values():
        target = Path(row['destination'])
        if not target.is_relative_to(root) or target.is_symlink() or not target.is_dir():
            raise ValueError('invalid collection receipt path')
        members = []
        for path in sorted(target.rglob('*')):
            if path.is_symlink():
                raise ValueError('collected artifact contains a symlink')
            metadata = path.stat()
            members.append([path.relative_to(target).as_posix(), metadata.st_mode, metadata.st_size,
                            metadata.st_mtime_ns, sha256_file(path) if path.is_file() else None])
        if members != row['members']:
            raise ValueError('collected artifact changed before recovery')
    paths = {}
    for name, relative in pending['collected'].items():
        path = root / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('invalid recovered artifact path')
        paths[name] = path
    return paths


def restore(result, *, docker=None, discarding=False):
    from containers import Docker
    from runtime.agent import DockerAgentEnvironment, CONTAINER_PATHS
    from runtime.cleanup import stop_container
    result = Path(result)
    pending = read_json(pending_path(result))
    if pending.get('schema') != 'abench.collection_pending/v1' or pending.get('execution_complete') is not True:
        raise ValueError('incomplete collection recovery state')
    snapshot, context = pending['environment'], pending['context']
    root = Path(snapshot['root'])
    if root.is_symlink() or not root.is_absolute() or not root.name.startswith('abench-agent-'):
        raise ValueError('invalid private collection directory')
    if root.exists():
        metadata = root.stat()
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError('private collection directory ownership or permissions changed')
    elif not discarding:
        raise ValueError('private collection directory is missing')
    for stream in ('stdout', 'stderr'):
        path = result / ('logs/agent_' + stream + '.log')
        if not discarding and (path.is_symlink() or sha256_file(path) != context[stream + '_sha256']):
            raise ValueError('execution evidence changed before collection recovery')
    docker = docker or Docker()
    if docker.info()['raw'].get('ID') != snapshot['daemon']:
        raise ValueError('collection recovery must use the original Docker daemon')
    existing = []
    for identifier in snapshot['ids']:
        try:
            inspected = docker.inspect(identifier)
        except Exception as error:
            from containers import DockerError
            if (discarding or pending.get('collection_complete')) and isinstance(error, DockerError) and _missing_object(error):
                continue                         # cleanup completed before its process stopped writing the result
            raise
        labels = (inspected.get('Config') or {}).get('Labels') or {}
        if (labels.get('abench.run') != snapshot['label'] or labels.get('abench.run_id') != snapshot['run_id']
                or labels.get('abench.case') != snapshot['case']):
            raise ValueError('collection container ownership changed')
        if inspected.get('State', {}).get('Running') is not False:
            raise ValueError('collection recovery refuses a running container')
        stop_container(docker, identifier)
        existing.append(identifier)
    snapshot['ids'] = existing
    volumes = []
    for identifier in snapshot['volumes']:
        try:
            inspected = docker.inspect(identifier)
        except Exception as error:
            from containers import DockerError
            if (discarding or pending.get('collection_complete')) and isinstance(error, DockerError) and _missing_object(error):
                continue
            raise
        labels = inspected.get('Labels') or {}
        if (labels.get('abench.run') != snapshot['label'] or labels.get('abench.run_id') != snapshot['run_id']
                or labels.get('abench.case') != snapshot['case']):
            raise ValueError('collection volume ownership changed')
        volumes.append(identifier)
    snapshot['volumes'] = volumes
    environment = DockerAgentEnvironment.__new__(DockerAgentEnvironment)
    environment.docker, environment.host_paths, environment.paths = docker, {'root': root}, dict(CONTAINER_PATHS)
    for name in ('image', 'label', 'cid', 'sidecar', 'ids', 'volumes', 'record'):
        setattr(environment, name, snapshot[name])
    return context, environment, snapshot


def finish(result, snapshot):
    """Forget retained state only after cleanup was verified; raw authoritative records remain immutable."""
    cleanup = snapshot['record'].get('cleanup') or {}
    if any((cleanup.get('remaining') or {}).values()):
        raise RuntimeError('collection cleanup is incomplete; retained recovery registration kept')
    root = Path(snapshot['root'])
    if root.exists():
        from agents.workspace import make_writable
        make_writable(root)
        shutil.rmtree(root)
    unregister_recovery(snapshot)
    pending_path(result).unlink(missing_ok=True)


def discard(result):
    """Explicit operator discard, with the same daemon/ownership/stopped guards as recovery."""
    _, environment, snapshot = restore(result, discarding=True)
    environment.close()
    atomic_json(Path(result) / 'cleanup.json', environment.record.get('cleanup') or {})
    finish(result, snapshot)


def discard_registered(record, *, docker=None):
    """Discard a write-ahead registration whose process died before committing its pending context."""
    from containers import Docker, DockerError
    from runtime.cleanup import cleanup_objects
    from agents.workspace import make_writable
    docker = docker or Docker()
    if docker.info()['raw'].get('ID') != record['daemon']:
        raise ValueError('retained collection belongs to another Docker daemon')
    containers, volumes = [], []
    for kind, output in (('containers', containers), ('volumes', volumes)):
        for identifier in record[kind]:
            try:
                inspected = docker.inspect(identifier)
            except DockerError as error:
                if _missing_object(error):
                    continue
                raise
            labels = ((inspected.get('Config') or {}).get('Labels') if kind == 'containers' else inspected.get('Labels')) or {}
            if any(labels.get('abench.' + key) != record[key] for key in ('case', 'run_id')) or labels.get('abench.run') != record['label']:
                raise ValueError('retained collection object ownership changed')
            if kind == 'containers' and inspected.get('State', {}).get('Running') is not False:
                raise ValueError('discard refuses running retained objects')
            output.append(identifier)
    root = Path(record['root'])
    if root.exists():
        metadata = root.lstat()
        if (not root.is_absolute() or root.is_symlink() or not root.name.startswith('abench-agent-') or
                not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077):
            raise ValueError('private collection directory ownership or permissions changed')
    state = {}
    cleanup_objects(docker, containers, volumes, state)
    atomic_json(Path(record['run']) / 'cleanup.json', state['cleanup'])
    if any(state['cleanup']['remaining'].values()):
        raise RuntimeError('retained collection cleanup incomplete')
    if root.exists():
        make_writable(root)
        shutil.rmtree(root)
    unregister_recovery(record)
