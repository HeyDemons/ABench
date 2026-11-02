#!/usr/bin/env python3
"""Release workflow: publish the profile and egress images and collect platform locks.

Requires a configured buildx builder and authenticated registry. This command performs publication.
Every image version is immutable: a version that is already published and locked is skipped, so bumping one
profile (spec/env/profiles.yaml) publishes only the new versions, and their digests are added
to spec/env/images.lock.yaml next to the existing ones. Review the uploaded locks and commit them before using the
release for scoring.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import yaml
from images import ROOT, env_lock_path, image_version, manifest, profiles

LOCK = ROOT / 'spec/env/images.lock.yaml'


def command(arguments, capture=False):
    return subprocess.run(arguments, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def published(tag):
    return subprocess.run(['docker', 'buildx', 'imagetools', 'inspect', tag],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def platform_locks(tag, namespace, key, platforms):
    raw = command(['docker', 'buildx', 'imagetools', 'inspect', '--raw', tag], capture=True)
    entries = json.loads(raw).get('manifests') or []
    locks = {}
    for platform in platforms:
        matches = [entry for entry in entries if
                   entry.get('platform', {}).get('os', '') + '/' + entry.get('platform', {}).get('architecture', '') == platform]
        if entries and not matches:
            raise RuntimeError(f'{tag} has no published manifest for {platform}')
        digest = matches[0]['digest'] if matches else 'sha256:' + hashlib.sha256(raw.encode()).hexdigest()
        locks[platform] = {'digest': digest, 'reference': f'{namespace}/{key}@{digest}'}
    return locks


def main():
    config, definitions = manifest(), profiles()
    lock = yaml.safe_load(LOCK.read_text()) or {}
    lock['schema'] = 'abench.images_lock/v1'
    lock['images'] = lock.get('images') or {}
    done = set()

    def save():
        LOCK.write_text(yaml.safe_dump(lock, sort_keys=False))

    def release(key, version, dockerfile, platforms, build_args=(), recipe_sha256=None):
        """Build and push key:version unless it is already published; returns whether it was built now."""
        tag = f'{config["namespace"]}/{key}:{version}'
        if published(tag):
            existing = lock['images'].get(f'{key}@{version}') or {}
            if not all(platform in existing for platform in platforms):
                raise RuntimeError(f'{tag} is published but missing from images.lock.yaml; restore its lock entry '
                                   'instead of republishing an immutable version')
            if recipe_sha256 and any(not isinstance(existing[p], dict) or
                                     existing[p].get('recipe_sha256') != recipe_sha256 for p in platforms):
                raise RuntimeError(f'{tag} packaging identity is missing or mismatched in images.lock.yaml; '
                                   'restore the verified lock or bump the packaging revision')
            return False
        command(['docker', 'buildx', 'build', '--push', '--platform', ','.join(platforms), '-t', tag,
                 '-f', dockerfile, *build_args, '.'])
        lock['images'][f'{key}@{version}'] = platform_locks(tag, config['namespace'], key, platforms)
        if recipe_sha256:
            for entry in lock['images'][f'{key}@{version}'].values():
                entry['recipe_sha256'] = recipe_sha256
        save()
        return True

    def profile(name):
        if name in done:
            return
        definition = definitions[name]
        if definition.get('extends'):
            profile(definition['extends'])
        platforms = config['images'][name]['platforms']
        version = image_version(name)
        tag = f'{config["namespace"]}/{name}:{version}'
        if release(name, version, f'docker/profiles/{name}/Dockerfile', platforms):
            # Only reference-amd64 is used for scoring; lock every published platform environment separately.
            for platform in platforms:
                reference = lock['images'][f'{name}@{version}'][platform]['reference']
                raw = command(['docker', 'run', '--rm', '--platform', platform, '--network', 'none', reference,
                               'python3', '/abench/tools/record_environment.py', '--profile', name], capture=True)
                out = env_lock_path(name, version, platform)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(json.loads(raw), indent=2) + '\n')
        done.add(name)

    for name in definitions:
        profile(name)
    # The sidecar executes proxy code uploaded from the run's tools tree.
    egress = config['images']['egress']
    version = str(egress['version'])
    tag = f'{config["namespace"]}/egress:{version}'
    if published(tag):
        if f'egress@{version}' not in lock['images']:
            raise RuntimeError(f'{tag} is published but missing from images.lock.yaml')
    else:
        command(['docker', 'buildx', 'imagetools', 'create', '--tag', tag, egress['source']])
        document = json.loads(command(['docker', 'buildx', 'imagetools', 'inspect', '--raw', tag], capture=True))
        entries = {}
        for entry in document['manifests']:
            p = entry.get('platform') or {}
            platform = p.get('os', '') + '/' + p.get('architecture', '')
            if platform in egress['platforms']:
                entries[platform] = {'digest': entry['digest'], 'reference': config['namespace'] + '/egress@' + entry['digest']}
        lock['images'][f'egress@{version}'] = entries
        save()
    save()


if __name__ == '__main__':
    main()
