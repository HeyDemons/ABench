#!/usr/bin/env python3
"""Export the reviewed working tree as a clean source snapshot, without Git history or private runtime state."""
from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from urllib.parse import urlsplit

from redact_records import Redactor, _dotenv_secrets, KEY_SHAPE, URL

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'PUBLIC-SNAPSHOT.json'
PRIVATE_PARTS = {'.git', '.abench', '.recovery', '.venv', '__pycache__', 'node_modules',
                 '.ssh', '.aws', '.kube', '.config', '.cache', '.docker', '.gnupg', '.pki',
                 '.codex', '.claude', '.gemini', '.dsh', '.zcode', '.npm'}
PRIVATE_PREFIXES = ('runs/', 'jobs/', 'exports/', 'sandbox/data/', 'sandbox/hidden/',
                    'sandbox/models/', 'sandbox/judge_models/', 'sandbox/.prepare/')
PRIVATE_FILES = {'spec/env/images.local.lock.yaml', 'spec/env/.image-locks.lock'}
PRIVATE_NAMES = {'.netrc', '.npmrc', '.pypirc', '.git-credentials', 'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519'}
PRIVATE_PEM = re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----(?:\s|\\[rn])*[A-Za-z0-9+/=]{32,}')
PERSONAL_PATH = re.compile(r'/(?:Users|home)/(?!abench(?:/|\b)|operator(?:/|\b)|user(?:/|\b)|example(?:/|\b)|agent(?:/|\b))[^/\s"\'\\]+')
DEPLOYMENT_HOST = re.compile(r'\ba6000-\d+\b')


def _private(name):
    path = PurePosixPath(name)
    return (path.is_absolute() or '..' in path.parts or bool(set(path.parts) & PRIVATE_PARTS)
            or name.startswith(PRIVATE_PREFIXES) or name in PRIVATE_FILES
            or path.name in PRIVATE_NAMES
            or path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example'
            or path.suffix.lower() in ('.key', '.enc', '.pem', '.p12', '.pfx', '.pyc')
            or path.name.endswith(('.tar', '.tar.gz', '.tgz', '.zip')))


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def export(root, destination, *, include_untracked=False, secrets=()):
    root, destination = Path(root).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('snapshot output already exists')
    if destination.resolve().is_relative_to(root / '.git'):
        raise ValueError('snapshot output must be outside Git metadata')
    arguments = ['git', '-C', str(root), 'ls-files', '-z', '--cached']
    if include_untracked:
        arguments += ['--others', '--exclude-standard']
    selected = subprocess.run(arguments, capture_output=True, check=True).stdout.decode().split('\0')
    known = [value.encode('utf-8') for value in Redactor(secrets=secrets).secrets if len(value.encode('utf-8')) >= 8]
    overlap = max([16384, *(len(value) for value in known)])
    entries, stamps = {}, {}
    for name in sorted(set(selected)):
        if not name or _private(name):
            continue
        if name == MANIFEST:
            raise ValueError('reserved public snapshot manifest already exists')
        source = root / name
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != root):
            raise ValueError('public snapshots do not include source symlinks')
        if not source.is_file():
            continue
        before, digest, tail, byte_tail = _stamp(source), hashlib.sha256(), '', b''
        decoder = codecs.getincrementaldecoder('utf-8')(errors='ignore')
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                text = tail + decoder.decode(block)
                if KEY_SHAPE.search(text) or PRIVATE_PEM.search(text):
                    raise ValueError('credential shape detected in source; snapshot was not published')
                if any(secret in byte_tail + block for secret in known):
                    raise ValueError('known credential detected in source; snapshot was not published')
                for url in URL.findall(text):
                    try:
                        parsed = urlsplit(url)
                    except ValueError:
                        if '@' in url:
                            raise ValueError('URL credentials detected in source; snapshot was not published') from None
                        continue
                    hostname = parsed.hostname or ''
                    example = hostname in ('example.com', 'example.org', 'example.net') or hostname.endswith(
                        ('.example', '.example.com', '.example.org', '.example.net', '.invalid'))
                    if parsed.username and not (name.startswith('tests/') and example):
                        raise ValueError('URL credentials detected in source; snapshot was not published')
                if not name.startswith('tests/') and (PERSONAL_PATH.search(text) or DEPLOYMENT_HOST.search(text)):
                    raise ValueError('deployment identity detected in source; snapshot was not published')
                tail = text[-overlap:]
                byte_tail = (byte_tail + block)[-overlap:]
        if source.is_symlink() or _stamp(source) != before:
            raise ValueError('source changed while preparing the public snapshot')
        entries[name] = {'sha256': digest.hexdigest(), 'bytes': before[3], 'mode': before[2] & 0o777}
        stamps[name] = before
    manifest = {'schema': 'abench.public_source_snapshot/v1', 'source': 'reviewed_working_tree',
                'git_history_included': False, 'private_runtime_state_included': False, 'files': entries}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.public-source-', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as output, gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as compressed, \
                tarfile.open(fileobj=compressed, mode='w|') as archive:
            for name, entry in entries.items():
                source = root / name
                if source.is_symlink() or _stamp(source) != stamps[name]:
                    raise ValueError('source changed before public snapshot transfer')
                info = tarfile.TarInfo(name)
                info.size, info.mode = entry['bytes'], entry['mode']
                with source.open('rb') as stream:
                    archive.addfile(info, stream)
                if source.is_symlink() or _stamp(source) != stamps[name]:
                    raise ValueError('source changed during public snapshot transfer')
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(manifest_bytes), 0o644
            archive.addfile(info, io.BytesIO(manifest_bytes))
        for name, before in stamps.items():
            source = root / name
            if source.is_symlink() or _stamp(source) != before:
                raise ValueError('working tree changed while creating the public snapshot')
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'files': len(entries), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--include-untracked', action='store_true', help='include new, non-ignored source files')
    args = parser.parse_args(argv)
    try:
        summary = export(ROOT, args.out, include_untracked=args.include_untracked, secrets=_dotenv_secrets())
    except (OSError, ValueError, subprocess.SubprocessError):
        parser.exit(1, 'public snapshot could not be created; inspect source identities, credentials and output location\n')
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
