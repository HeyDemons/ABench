"""Harness CLI bundles: one pinned harness CLI with its own Node.js, built once per host and mounted read-only into
the task container.

A run starts the case's profile image (the task environment) and the selected adapter mounts its bundle at
/opt/abench/harness, writes its configuration and starts the CLI there; nobody builds an image per harness. A bundle
is built in a throwaway Node container with network access (the agent container never has it): npm CLIs install
by exact version, source-built CLIs (ZCode) build from their tagged commit. Entry points become small wrapper scripts
and the tree holds regular files only, so it can be a content-addressed cache volume like the case data.

  ./abench images build --agent dsh            # build (or reuse) the dsh bundle in ~/.abench/harness/
  ./abench images lock --local --agent dsh     # record its content digest for scoring on this host

The bundle identity covers the harness definition in spec/env/images.yaml, the Node.js version, the build script
and the platform. A run records the mounted volume's content digest; the run scores only when that digest is the
one locked for this host (spec/env/images.local.lock.yaml, key harness-<name>@<identity>).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[1]
MOUNT = '/opt/abench/harness'
SCHEMA = 1


def home() -> Path:
    return Path(os.environ.get('ABENCH_HARNESS_DIR') or Path.home() / '.abench' / 'harness').expanduser()


def definition(name: str) -> dict:
    from images import manifest
    harnesses = manifest()['harnesses']
    if name not in harnesses:
        raise ValueError(f'unknown harness {name!r}; spec/env/images.yaml defines {", ".join(harnesses)}')
    return harnesses[name]


def script(name: str) -> str:
    """Shell script run in the builder container; writes the bundle as a tar stream to stdout."""
    spec = definition(name)
    lines = ['set -eu', f'P={MOUNT}', 'mkdir -p "$P/bin" "$P/lib"', 'cp /usr/local/bin/node "$P/bin/node"']
    source = spec.get('source')
    if source:
        lines += ['export COREPACK_ENABLE_PROJECT_SPEC=0 ELECTRON_SKIP_BINARY_DOWNLOAD=1 CI=1',
                  f'corepack enable >&2 && corepack prepare pnpm@{source["pnpm"]} --activate >&2',
                  f'git clone -q --depth 1 --branch {source["tag"]} {source["repository"]} /src >&2',
                  f'test "$(git -C /src rev-parse HEAD)" = "{source["commit"]}"',
                  'cd /src && pnpm install --frozen-lockfile --filter "@zcode/cli..." >&2 && '
                  'pnpm -r --filter "@zcode/cli..." run build >&2',
                  f'cp -r /src/apps/zcode-cli/packages/cli/dist "$P/lib/{name}" && rm -f "$P/lib/{name}"/*.map',
                  f'printf \'#!/bin/sh\\nexec %s/bin/node %s/lib/{name}/zcode.cjs "$@"\\n\' "$P" "$P" > "$P/bin/{name}"',
                  f'chmod 755 "$P/bin/{name}"']
    else:
        lines += [f'HOME=/root npm install -g --prefix "$P" {spec["npm"]}@{spec["version"]} >&2',
                  # bin/ entries are symlinks into lib/node_modules: replace them with wrappers that put this bundle's
                  # node first on PATH (for "#!/usr/bin/env node" entry points) and run the real entry point
                  'for f in "$P"/bin/*; do [ -L "$f" ] || continue; t=$(readlink -f "$f"); rm "$f"; '
                  'printf \'#!/bin/sh\\nPATH=%s/bin:$PATH exec %s "$@"\\n\' "$P" "$t" > "$f"; chmod 755 "$f"; done']
    lines += ['rm -rf /root/.npm',
              # regular files only (the cache refuses links): copy what symlinks point to, and store a file reached
              # through several symlinks as a file each time instead of a hard link
              'cd "$P" && tar -ch --hard-dereference --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner .']
    return '\n'.join(lines) + '\n'


def identity(name: str, platform: str) -> str:
    from images import manifest
    return hashlib.sha256(json.dumps({'schema': SCHEMA, 'harness': name, 'definition': definition(name),
                                      'node': manifest()['node'], 'platform': platform, 'script': script(name)},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def key(name: str, platform: str) -> str:
    return f'harness-{name}@{identity(name, platform)[:12]}'


def directory(name: str, platform: str) -> Path:
    return home() / f'{name}-{identity(name, platform)[:12]}'


def content_digest(path: Path) -> str:
    """The digest the run's cache volume of this directory gets (runtime.cache): names, sizes, bytes and modes."""
    from runtime.cache import _digest, _file_hash
    entries = []
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError(f'bundle member is not a regular file or directory: {p}')
        mode = p.lstat().st_mode & 0o777
        rel = p.relative_to(path).as_posix()
        entries.append([rel, 'dir', mode] if p.is_dir() else [rel, 'file', p.stat().st_size, _file_hash(p), mode])
    return _digest(entries)


def _record(target: Path, name: str, platform: str) -> dict | None:
    """The metadata of a finished build, or None when it is missing, unreadable or names another bundle."""
    try:
        info = json.loads(target.with_name(target.name + '.json').read_text())
    except (OSError, ValueError):
        return None
    if not target.is_dir() or info.get('key') != key(name, platform) or info.get('directory') != str(target):
        return None
    return info


def build(name: str, docker=None, platform: str | None = None) -> dict:
    """Build the bundle unless this host already has it. Builds of one bundle are serialized across processes; each
    stages in its own directory, and the metadata is written last and atomically, so a reader sees either a finished
    bundle or none. A finished build is reused, never rebuilt in place."""
    import fcntl
    import tempfile
    import uuid
    from images import manifest
    from runtime.transport import Docker, safe_extract
    docker = docker or Docker()
    platform = platform or docker.info()['platform']
    target = directory(name, platform)
    home().mkdir(parents=True, exist_ok=True)
    with open(home() / f'{target.name}.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        info = _record(target, name, platform)
        if info:
            return {**info, 'reused': True}
        if target.exists():                      # an interrupted publication: no valid metadata
            shutil.rmtree(target)
        container = f'abench-bundle-{name}-{uuid.uuid4().hex[:12]}'
        args = ['run', '--rm', '--name', container, '--label', f'abench.bundle={key(name, platform)}',
                '--platform', platform]
        if os.environ.get('ABENCH_BUILD_NETWORK'):
            args += ['--network', os.environ['ABENCH_BUILD_NETWORK']]
        proxy = os.environ.get('ABENCH_BUILD_PROXY')
        for var in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'):
            value = proxy if proxy and var != 'NO_PROXY' else os.environ.get(var)
            if value:
                args += ['-e', f'{var}={value}', '-e', f'{var.lower()}={value}']
        image = f"node:{manifest()['node']}-bookworm"
        args += [image, 'sh', '-c', script(name)]
        staging = Path(tempfile.mkdtemp(prefix=target.name + '.', suffix='.partial', dir=home()))
        started = time.monotonic()
        try:
            try:
                code, stderr = docker.run_to_file(args, staging / 'bundle.tar', timeout=7200)
            finally:
                docker.run(['rm', '-f', container], check=False)     # stopped and removed even after a timeout
            if code:
                raise RuntimeError(f'building the {name} bundle failed: {stderr[-2000:]}')
            tree = staging / 'tree'
            with open(staging / 'bundle.tar', 'rb') as archive:
                safe_extract(archive, tree, max_bytes=8 * 1024**3)
            for p in tree.rglob('*'):
                if p.is_symlink():
                    raise RuntimeError(f'bundle contains a link: {p}')
            info = {'schema': SCHEMA, 'harness': name, 'version': str(definition(name)['version']), 'platform': platform,
                    'identity': identity(name, platform), 'key': key(name, platform), 'directory': str(target),
                    'content_sha256': content_digest(tree), 'builder_image': image,
                    'seconds': round(time.monotonic() - started, 1)}
            tree.rename(target)
            pending = staging / 'record.json'
            pending.write_text(json.dumps(info, indent=2) + '\n')
            os.replace(pending, target.with_name(target.name + '.json'))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {**info, 'reused': False}


def find(name: str, platform: str) -> dict:
    """The built bundle of this host, or a ValueError naming the command that builds it."""
    info = _record(directory(name, platform), name, platform)
    if info is None:
        raise ValueError(f'no {name} bundle for {platform} on this host; build it with: ./abench images build --agent {name}')
    return info


def lock(name: str, docker=None, platform: str | None = None) -> dict:
    """Record the content digest of this host's bundle as the one runs may score with."""
    from images import update_local_lock
    from runtime.transport import Docker
    platform = platform or (docker or Docker()).info()['platform']
    info = find(name, platform)
    actual = content_digest(Path(info['directory']))
    if actual != info['content_sha256']:
        raise ValueError(f'{info["directory"]} changed since it was built; delete it and build again')
    entry = {'content_sha256': actual, 'version': info['version'],
             'locked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    previous = update_local_lock(info['key'], platform, entry)
    return {'key': info['key'], 'platform': platform, 'locked_local': actual,
            'replaced': (previous or {}).get('content_sha256') if previous else None}


def attestation(name: str, platform: str, digest: str | None) -> dict:
    """Whether the mounted bundle is the one locked on this host."""
    from images import local_lock_entries
    locked = (local_lock_entries().get(key(name, platform)) or {}).get(platform) or {}
    ok = bool(digest) and locked.get('content_sha256') == digest
    return {'key': key(name, platform), 'content_sha256': digest, 'locked': ok, 'lock': 'local' if ok else None,
            **({} if ok else {'reason': 'harness bundle content is not the one locked on this host'})}
