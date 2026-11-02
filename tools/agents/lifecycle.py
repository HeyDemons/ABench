"""Private operator phase records and atomic writes; never mounted into an agent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError('operator state must not be a symlink')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('operator state must not be a symlink')
    return json.loads(path.read_text())


def attempt_failure(error, phase):
    # Exception classes, rather than traceback text, decide new-run infrastructure retries.
    from agents.errors import judge_failure
    classified = judge_failure(error)
    retryable = classified['retryable']
    return {'phase': phase, 'exception_type': type(error).__name__,
            'kind': 'infrastructure' if retryable else ('interrupted' if isinstance(error, KeyboardInterrupt) else 'runner_error'),
            'retryable': retryable}


class Attempt:
    def __init__(self, result, case, run_id, *, resume=False):
        self.path = Path(result) / 'attempt.json'
        self.document = read_json(self.path) if resume and self.path.is_file() else {
            'schema': 'abench.attempt/v1', 'case_id': case, 'run_id': run_id,
            'started_at': time.time(), 'phases': []}
        self.document.update(pid=os.getpid(), status='running')
        self.phase(self.document.get('phase', 'prepare'))

    def phase(self, phase):
        now = time.time()
        phases = self.document['phases']
        if phases and 'finished_at' not in phases[-1]:
            phases[-1]['finished_at'] = now
        phases.append({'phase': phase, 'started_at': now})
        self.document.update(phase=phase, updated_at=now)
        atomic_json(self.path, self.document)

    def finish(self, *, error=None, pending=False, cleanup=None):
        now = time.time()
        self.document.update(status='collection_pending' if pending else ('failed' if error else 'finished'), updated_at=now)
        if self.document['phases']:
            self.document['phases'][-1]['finished_at'] = now
        if error:
            self.document['failure'] = attempt_failure(error, self.document['phase'])
        else:
            self.document.pop('failure', None)
        if cleanup is not None:
            # Cleanup diagnostic details already live in cleanup.json; no exception strings/secrets here.
            self.document['cleanup_remaining'] = cleanup.get('remaining')
        atomic_json(self.path, self.document)


def registry_root():
    return Path(os.environ.get('ABENCH_RECOVERY_DIR', str(Path.home() / '.abench/recovery'))).expanduser()


def registry_path(snapshot):
    key = hashlib.sha256((snapshot['daemon'] + '\0' + snapshot['label']).encode()).hexdigest()
    return registry_root() / (key + '.json')


def register_recovery(result, snapshot):
    atomic_json(registry_path(snapshot), {'schema': 'abench.collection_registry/v1',
                'run': str(Path(result).resolve()), 'run_id': snapshot['run_id'], 'daemon': snapshot['daemon'],
                'label': snapshot['label'], 'case': snapshot['case'], 'root': snapshot['root'],
                'containers': snapshot['ids'], 'volumes': snapshot['volumes']})


def unregister_recovery(snapshot):
    registry_path(snapshot).unlink(missing_ok=True)


def protected_objects(docker):
    """GC skips retained stopped objects on this daemon; stale registrations are harmless and removable."""
    protected = {'containers': set(), 'volumes': set()}
    paths = list(registry_root().glob('*.json')) if registry_root().is_dir() else []
    if not paths:
        return protected
    daemon = docker.info()['raw'].get('ID')
    for path in paths:
        record = read_json(path)
        if record.get('daemon') == daemon:
            for kind in protected:
                protected[kind].update(record.get(kind) or [])
    return protected
