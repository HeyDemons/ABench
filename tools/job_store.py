"""Checksummed per-entry journal with compatible JSON checkpoints and crash replay.

The executor owns job.lock. Each event is fsynced before proceeding; a truncated
last event is ignored, while corruption in a complete event fails closed. A JSON
checkpoint is replaced before resetting its journal, so either crash order retains
all committed entry updates without rewriting the full matrix on every transition.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from agents.lifecycle import atomic_json


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def _replay(path, document, checkpoint_sha256):
    journal = path.with_suffix('.journal.jsonl')
    if not journal.is_file():
        return document
    if journal.is_symlink():
        raise ValueError('job journal must not be a symlink')
    with journal.open('rb') as stream:
        header = stream.readline()
        if not header.endswith(b'\n'):
            return document
        first = json.loads(header)
        if first.get('schema') != 'abench.job_journal/v1':
            raise ValueError('unsupported job journal')
        if first.get('checkpoint_sha256') != checkpoint_sha256:
            return document                  # a newer checkpoint replaced the journal's base before a crash
        entries = {entry['id']: entry for entry in document['entries']}
        for line in stream:
            if not line.endswith(b'\n'):
                break                       # interrupted final append, never a committed transition
            event = json.loads(line)
            payload = event['payload']
            if hashlib.sha256(_bytes(payload)).hexdigest() != event.get('sha256'):
                raise ValueError('job journal checksum mismatch')
            if any(entry['id'] not in entries for entry in payload['entries']):
                raise ValueError('job journal names an unknown entry')
            document.update(payload['header'])
            for updated in payload['entries']:
                entry = entries[updated['id']]
                entry.clear()
                entry.update(updated)
    return document


def load(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('job checkpoint must not be a symlink')
    raw = path.read_bytes()
    return _replay(path, json.loads(raw), hashlib.sha256(raw).hexdigest())


class Journal:
    def __init__(self, path, document):
        self.path = Path(path)
        self.journal = self.path.with_suffix('.journal.jsonl')
        if self.path.exists():
            latest = load(self.path)         # job.lock is held; the caller's pre-lock snapshot may be stale
            if latest.get('id') != document.get('id'):
                raise ValueError('job checkpoint identity differs from the requested job')
            existing = {entry['id']: entry for entry in document['entries']}
            entries = []
            for updated in latest['entries']:
                entry = existing.get(updated['id'], {})
                entry.clear()
                entry.update(updated)
                entries.append(entry)
            document.clear()
            document.update(latest)
            document['entries'] = entries
        self.checkpoint(document)

    def checkpoint(self, document):
        atomic_json(self.path, document)
        # Atomic JSON is also a valid single JSONL header line after compact serialization below.
        header = {'schema': 'abench.job_journal/v1',
                  'checkpoint_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=self.path.parent, delete=False) as output:
            output.write(_bytes(header) + b'\n')
            output.flush()
            os.fsync(output.fileno())
            temporary = Path(output.name)
        try:
            if self.journal.is_symlink():
                raise ValueError('job journal must not be a symlink')
            temporary.replace(self.journal)
        finally:
            temporary.unlink(missing_ok=True)

    def update(self, document, entries=()):
        payload = {'header': {key: value for key, value in document.items() if key != 'entries'},
                   'entries': list(entries)}
        event = {'payload': payload, 'sha256': hashlib.sha256(_bytes(payload)).hexdigest()}
        fd = os.open(self.journal, os.O_WRONLY | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'ab') as stream:
            stream.write(_bytes(event) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
