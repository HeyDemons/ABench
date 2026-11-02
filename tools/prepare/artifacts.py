"""Private operator artifacts: shared downloads and recoverable, case-local publication."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from file_hashes import sha256_file, tree_hash


def atomic_text(path, text, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def cached_download(url, destination, expected, fetch, check, cache=None):
    """Single-flight an immutable source; verify bytes on every hit and copy into the writable raw tree."""
    destination = Path(destination)
    identity = {'url': url, 'expected': expected}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache or os.environ.get('ABENCH_DOWNLOAD_CACHE', Path.home() / '.cache/abench/downloads'))
    directory = cache / digest
    with file_lock(cache / (digest + '.lock')):
        data, record = directory / 'data', directory / 'record.json'
        valid = False
        try:
            metadata = json.loads(record.read_text())
            valid = (isinstance(metadata, dict) and metadata.get('source') == identity
                     and data.is_file() and not data.is_symlink()
                     and sha256_file(data) == metadata.get('sha256'))
            if valid:
                check(data)
        except (OSError, ValueError):
            valid = False
        fetched = not valid
        if not valid:
            cache.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=digest + '-', suffix='.partial', dir=cache))
            try:
                # Keep the raw extension: fetch rejects HTML pages masquerading as dataset files.
                staged = staging / destination.name
                fetch(url, staged)
                check(staged)
                metadata = {'schema': 'abench.download/v1', 'source': identity, 'sha256': sha256_file(staged)}
                staged.rename(staging / 'data')
                (staging / 'data').chmod(0o444)
                atomic_text(staging / 'record.json', json.dumps(metadata, sort_keys=True) + '\n')
                if directory.exists():
                    shutil.rmtree(directory)
                staging.rename(directory)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            temporary = destination.with_name('.' + destination.name + '-' + uuid.uuid4().hex + '.partial')
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            with os.fdopen(descriptor, 'wb') as output, data.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
            check(temporary)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return fetched


def immutable_source(url, expected):
    return bool(any(expected.get(key) for key in ('sha256', 'md5', 'git_blob_sha1'))
                or re.search(r'/(?:resolve|raw)/[0-9a-f]{40}/', url)
                or re.search(r'raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/', url))


def verify_frozen(root, lock, key_file):
    """Verify staged files and the authenticated hidden plaintext without retaining another plaintext copy."""
    import hidden_store
    root = Path(root)
    for item in lock['public']:
        if sha256_file(root / item['path']) != item['sha256']:
            raise ValueError('staged public hash mismatch')
    for kind in ('models', 'judge_models'):
        for item in lock.get(kind, []):
            if tree_hash(root / item['path']) != item['tree_sha256']:
                raise ValueError('staged model hash mismatch')
    class Discard:
        def write(self, data):
            return len(data)
    with (root / lock['hidden']['path']).open('rb') as encrypted:
        verified = hidden_store.decrypt_stream(hidden_store.read_key(key_file), encrypted, Discard())
    if verified['sha256_plain'] != lock['hidden']['sha256_plain']:
        raise ValueError('staged hidden hash mismatch')


def remove(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        # Copytree preserves model directory permissions; failed generations can contain read-only trees.
        path.chmod(path.stat().st_mode | 0o700)
        for parent, directories, _ in os.walk(path, followlinks=False):
            for name in directories:
                child = Path(parent) / name
                if not child.is_symlink():
                    child.chmod(child.stat().st_mode | 0o700)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


class Publication:
    """Journal multiple renames; absent old locks/OJ fail closed until all inputs are installed.

    A crash is rolled back when the next operator enters this case. Readers do not acquire this lock;
    they must not run a case being prepared, and their existing content checks remain authoritative.
    """
    def __init__(self, root, case):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', case):
            raise ValueError('invalid case identifier')
        self.root, self.case = Path(root), case
        self.parent = self.root / 'sandbox/.prepare'
        self.directory = self.parent / case
        self.journal = self.directory / 'journal.json'
        self.staging = self.directory / 'new'

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = file_lock(self.parent / (self.case + '.lock'))
        self.lock.__enter__()
        try:
            self.recover()
            self.staging.mkdir(parents=True, mode=0o700)
            return self
        except BaseException:
            self.lock.__exit__(*__import__('sys').exc_info())
            raise

    def __exit__(self, *exception):
        try:
            self.recover()
        finally:
            self.lock.__exit__(*exception)

    def recover(self):
        if self.journal.exists():
            record = json.loads(self.journal.read_text())
            if not record.get('committed'):
                # Restore data first and publish the old lock/OJ only after all old inputs are back.
                for item in reversed(record['items']):
                    target, backup = Path(item['target']), Path(item['backup'])
                    if backup.exists():
                        remove(target)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        backup.replace(target)
                    elif not item['had_old']:
                        remove(target)
            for item in record['items']:
                remove(item['backup'])
                if item['staged'] is not None:
                    remove(item['staged'])
        remove(self.directory)

    def publish(self, mappings, before_commit=None, last=()):
        """mappings are (target, staged or None); lock/OJ invalidations must precede data entries."""
        items = []
        for index, mapping in enumerate(mappings):
            target, staged = mapping[:2]
            target = Path(target)
            items.append({'target': str(target), 'staged': str(staged) if staged is not None else None,
                          'backup': str(mapping[2] if len(mapping) > 2 else self.directory / 'old' / str(index)),
                          'had_old': target.exists()})
        record = {'schema': 'abench.publication/v1', 'items': items, 'committed': False}
        atomic_text(self.journal, json.dumps(record) + '\n')
        for item in items:
            target, backup = Path(item['target']), Path(item['backup'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if item['had_old']:
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
        for item in sorted(items, key=lambda value: value['target'] in {str(path) for path in last}):
            target = Path(item['target'])
            if item['staged'] is not None:
                Path(item['staged']).replace(target)
        if before_commit:
            before_commit()
        record['committed'] = True
        atomic_text(self.journal, json.dumps(record) + '\n')
