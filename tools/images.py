"""Generate profile Dockerfiles, manage immutable image references and environment locks."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import yaml
from containers import Docker, DockerError
ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r'^sha256:[0-9a-f]{64}$')
BUILD_LABEL = 'abench.image.build_inputs_sha256'


def _hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def manifest():
    return yaml.safe_load((ROOT/'spec/env/images.yaml').read_text())
def profiles():
    return yaml.safe_load((ROOT/'spec/env/profiles.yaml').read_text())['profiles']
def common_system_packages():
    """Command-line tools every root profile installs (agent and judge alike, as on the original host)."""
    return list(yaml.safe_load((ROOT/'spec/env/profiles.yaml').read_text()).get('system_packages') or [])
OFFLINE_AGENTS = ('mock', 'oracle', 'command')

def image_key(profile, agent=None):
    """Image name of a profile. Every agent runs in its case's profile image; a harness CLI comes from its bundle
    (tools/harness_bundles.py), so the agent argument does not change the image."""
    return profile

def image_version(profile, agent=None, profile_version=None):
    """Immutable version of an image: its profile's version (spec/env/profiles.yaml), or the image's own version in
    spec/env/images.yaml (egress). Versions are semantic versions, never dates."""
    m=manifest()
    if profile not in profiles():
        return str((m['images'].get(profile) or {})['version'])
    return str(profile_version or profiles()[profile]['version'])


def lock_entries():
    return yaml.safe_load((ROOT/'spec/env/images.lock.yaml').read_text()).get('images') or {}

def local_lock_entries():
    """Host-local locks written by `images lock --local`: the image ID of a locally built image, accepted for scoring
    and recorded as unpublished (lock: local). A published digest in images.lock.yaml always takes precedence."""
    path=local_lock_path()
    return ((yaml.safe_load(path.read_text()) or {}).get('images') or {}) if path.is_file() else {}

def local_lock_path():
    return ROOT/'spec/env/images.local.lock.yaml'

@contextmanager
def lock_transaction(path):
    """Serialize local image, harness-bundle and published-lock updates in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / '.image-locks.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _atomic_lock_write(path, text, mode):
    path = Path(path)
    if path.exists():
        mode = path.stat().st_mode & 0o777
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_local_lock(entries):
    """Write a complete local snapshot; callers merging entries must hold lock_transaction()."""
    _atomic_lock_write(local_lock_path(),
        '# Host-local locks (`./abench images lock --local`): image IDs of locally built images and content digests of\n'
        '# harness bundles, accepted for scoring and recorded as unpublished (lock: local). Published digests\n'
        '# (images.lock.yaml) take precedence.\n'+yaml.safe_dump({'schema':'abench.images_local_lock/v1','images':entries},sort_keys=False), 0o600)


def update_local_lock(key, platform, entry, *, require_unpublished=False):
    """Merge one image or bundle lock without losing another concurrent writer's entries."""
    with lock_transaction(local_lock_path()):
        if require_unpublished and (lock_entries().get(key) or {}).get(platform):
            raise ValueError(f'{key} is published; its registry digest is the lock')
        entries = local_lock_entries()
        previous = (entries.get(key) or {}).get(platform)
        entries.setdefault(key, {})[platform] = entry
        save_local_lock(entries)
    return previous

def local_lock(profile, agent=None, platform=None):
    """The host-local lock entry of an image version (same version naming as resolve_image), or None."""
    entries=local_lock_entries()
    if not entries:
        return None
    version=None
    if isinstance(profile,dict):
        version=profile.get('version');profile=profile.get('profile') or profile.get('name')
    if '@' in profile:
        profile,version=profile.split('@',1)
    requested=image_version(profile,None,version) if version is not None and profile in profiles() else (
        str(version) if version is not None else image_version(profile))
    platform=platform or Docker().info()['platform']
    return (entries.get(f'{image_key(profile)}@{requested}') or {}).get(platform)

def known_versions(profile):
    """Versions a case may pin: the current definition plus every published (locked) version."""
    versions={str(profiles()[profile]['version'])}
    versions|={key.split('@',1)[1] for key in lock_entries() if key.split('@',1)[0]==profile}
    return versions

def env_lock_path(profile, version, platform='linux/amd64'):
    """Environment snapshot (record_environment.py) of one published profile version on one platform."""
    suffix='' if platform=='linux/amd64' else '-'+platform.replace('/','-')
    return ROOT/'spec/env/lock'/f'{profile}@{version}{suffix}.json'

def resolve_image(profile, agent=None, platform=None, allow_unlocked=False, version=None):
    if isinstance(profile,dict):
        version=profile.get('version',version);profile=profile.get('profile') or profile.get('name')
    if '@' in profile:
        profile,version=profile.split('@',1)
    key=image_key(profile,agent)
    current=image_version(profile,agent)
    requested=current if version is None else (image_version(profile,None,version) if profile in profiles() else str(version))
    platform=platform or Docker().info()['platform']
    entry=(lock_entries().get(f'{key}@{requested}') or {}).get(platform)
    if isinstance(entry,dict):
        reference=entry.get('reference');digest=entry.get('digest')
    else:
        reference=None;digest=entry
    if digest and DIGEST.fullmatch(digest):
        if reference and not reference.endswith('@'+digest):
            raise ValueError(f'invalid image reference for {key}@{requested} on {platform}')
        return reference or f"{manifest()['namespace']}/{key}@{digest}"
    if requested!=current:
        # Only the current definition can be built locally; older versions exist only as published digests.
        raise ValueError(f'unknown profile version: {profile}@{version} (neither current nor published)')
    override=os.environ.get('ABENCH_IMAGE_'+key.upper().replace('-','_'))
    if allow_unlocked:
        return override or ((manifest()['images'].get(key) or {}).get('source')) or f"{manifest()['namespace']}/{key}:{requested}"
    raise ValueError(f'image {key}@{requested} on {platform} has no published digest; build/push/lock it first')

def image_attestation(image, profile=None, agent=None, platform=None):
    docker=Docker();platform=platform or docker.info()['platform']
    actual=docker.inspect(image)
    digests=actual.get('RepoDigests') or []
    locked=False
    try:
        expected=resolve_image(profile,agent,platform) if profile else image
        locked='@sha256:' in expected and (expected in digests or image==expected)
    except ValueError:
        expected=None
    lock='published' if locked else None
    local=local_lock(profile,agent,platform) if profile and not locked else None
    if local and local.get('image_id')==actual.get('Id'):
        # A host-local lock names this exact image ID: it scores, and the record says the image is unpublished.
        locked,lock,expected=True,'local',local['image_id']
    return {'locked':locked,'lock':lock,'matches':locked,'digest':next((d.split('@',1)[1] for d in digests),actual.get('Id')),
            'reference':image,'expected':expected,'platform':platform,'image_id':actual.get('Id')}

def generate(check=False):
    generated={}; ps=profiles();m=manifest()
    for name,p in ps.items():
        parent=p.get('extends')
        base=f"{m['namespace']}/{parent}:{image_version(parent)}" if parent else p['base']
        lines=['# Generated by tools/images.py; edit spec/env/profiles.yaml.']
        if p.get('system') and name=='gpu-torch-cuda-dev':
            lines += [f"FROM {p['base']} AS cuda"]
        lines += [f'FROM {base}', 'USER root']
        if name=='gpu-torch-cuda-dev':
            lines += ['COPY --from=cuda /usr/local/cuda /usr/local/cuda', 'ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH', 'RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*']
        if not parent:
            lines += ['RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tar coreutils libgomp1 && rm -rf /var/lib/apt/lists/*',
                      'RUN useradd --uid 1000 --create-home abench && mkdir -p /abench/view /abench/workspace /abench/models /abench/papers /abench/tools /run/abench-egress && chown -R 1000:1000 /abench /run/abench-egress']
        packages=(common_system_packages() if not parent else [])+list(p.get('system_packages') or [])
        if packages:
            lines += ['RUN apt-get update && apt-get install -y --no-install-recommends '+' '.join(packages)+
                      ' && rm -rf /var/lib/apt/lists/*']
        lines += [f'COPY docker/profiles/{name}/requirements.lock /tmp/requirements.lock',
                  'RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.lock',
                  'COPY tools/record_environment.py /abench/tools/record_environment.py',
                  'COPY tools/runtime/forwarder.py /abench/tools/forwarder.py']
        for key,value in (p.get('env') or {}).items():
            lines += [f'ENV {key}={value}']
        lines += ['ENV HOME=/abench/workspace/.home TMPDIR=/abench/workspace/.tmp PYTHONDONTWRITEBYTECODE=1',
                  'USER 1000:1000', 'WORKDIR /abench/workspace', 'CMD ["sleep", "infinity"]']
        generated[ROOT/f'docker/profiles/{name}/Dockerfile']='\n'.join(lines)+'\n'
        lock_path=ROOT/f'docker/profiles/{name}/requirements.lock'
        inputs=''.join(f'{k}=={v}\n' for k,v in p['packages'].items())
        header='# input-sha256: '+hashlib.sha256(inputs.encode()).hexdigest()+'\n'
        existing=lock_path.read_text() if lock_path.exists() else ''
        generated[lock_path]=existing if existing.startswith(header) else header+inputs
    bad=[]
    for path,content in generated.items():
        if check:
            if not path.exists() or path.read_text()!=content:bad.append(str(path.relative_to(ROOT)))
        elif not path.exists() or path.read_text()!=content:
            path.parent.mkdir(parents=True,exist_ok=True)
            # Concurrent builds may generate the same files; readers must never see a partial Dockerfile.
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
                output.write(content)
                temporary = Path(output.name)
            temporary.chmod(0o644)
            temporary.replace(path)
    if bad:raise ValueError('stale generated images: '+', '.join(bad))
    return len(generated)

def _inspect_if_present(docker, reference):
    try:
        return docker.inspect(reference)
    except DockerError:
        return None


def build_inputs(profile, platform, docker):
    """Only files consumed by the Dockerfile enter this identity; case files and run configuration do not."""
    dockerfile = ROOT / f'docker/profiles/{profile}/Dockerfile'
    inputs = {'schema': 1, 'image': image_key(profile) + '@' + image_version(profile),
              'platform': platform, 'dockerfile': dockerfile.read_text()}
    definition = profiles()[profile]
    references = ([f"{manifest()['namespace']}/{definition['extends']}:{image_version(definition['extends'])}"]
                  if definition.get('extends') else [definition['base']])
    if profile == 'gpu-torch-cuda-dev':
        references.append(definition['base'])
    paths = [f'docker/profiles/{profile}/requirements.lock', 'tools/record_environment.py',
             'tools/runtime/forwarder.py']
    inputs['files'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    inputs['parents'] = []
    for reference in references:
        actual = _inspect_if_present(docker, reference)
        inputs['parents'].append({'reference': reference, 'image_id': (actual or {}).get('Id')})
    return inputs


def build_cache_scope(docker, info):
    """Keep readiness and interprocess locks separate for unrelated daemons/contexts."""
    identity = {'daemon_id': (info.get('raw') or {}).get('ID'), 'executable': docker.executable,
                'context': os.environ.get('DOCKER_CONTEXT'), 'host': os.environ.get('DOCKER_HOST'),
                'config': os.environ.get('DOCKER_CONFIG', str(Path.home() / '.docker'))}
    # An implicit active context is not represented by DOCKER_CONTEXT.
    if not identity['context'] and not identity['host']:
        result = docker.run(['context', 'show'], check=False)
        identity['context'] = result.stdout.strip() if result.returncode == 0 else None
    return _hash(identity)


@contextmanager
def build_lock(scope, identity):
    root = Path(os.environ.get('ABENCH_IMAGE_BUILD_CACHE',
                               str(Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'abench/images')))
    directory = root / scope
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / f'{identity}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory / f'{identity}.json'


def _environment_lock_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _verify_build_environment(docker, tag, profile, lock_path):
    """Check the built image against the profile lock, creating it only for a new profile."""
    result = docker.run(['run', '--rm', '--network', 'none', tag, 'python3',
                         '/abench/tools/record_environment.py', '--profile', profile])
    environment = json.loads(result.stdout)
    if lock_path.exists():
        from record_environment import compare
        differences = compare(json.loads(lock_path.read_text()), environment)
        if differences:
            raise ValueError('published profile environment changed; bump profile version: ' + str(differences))
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(environment, indent=2) + '\n')
    return environment


def _build(profile, platform, docker, info, tag):
    generate()
    inputs = build_inputs(profile, platform, docker)
    identity = _hash(inputs)
    scope = build_cache_scope(docker, info)
    lock_path = env_lock_path(profile, image_version(profile), platform)
    with build_lock(scope, identity) as ready:
        actual = _inspect_if_present(docker, tag)
        if actual and (actual.get('Config', {}).get('Labels') or {}).get(BUILD_LABEL) == identity and ready.is_file():
            try:
                cached = json.loads(ready.read_text())
            except (OSError, ValueError):
                cached = {}
            if (cached.get('image_id') == actual.get('Id') and cached.get('build_inputs_sha256') == identity
                    and cached.get('environment_lock_sha256') == _environment_lock_digest(lock_path)
                    and cached.get('environment')):
                return {'image': tag, 'environment': cached['environment'], 'reused': True,
                        'build_inputs_sha256': identity}
        ready.unlink(missing_ok=True)
        if lock_entries().get(f'{image_key(profile)}@{image_version(profile)}', {}).get(platform):
            raise ValueError('image version already published; use its locked digest or bump the version before rebuilding')
        args = ['build', '--platform', platform, '-t', tag, '--label', f'{BUILD_LABEL}={identity}']
        # The buildx token provider runs on the host. Keep its standard proxy environment unchanged;
        # rootless build containers may need a different, namespace-reachable proxy address.
        build_proxy = os.environ.get('ABENCH_BUILD_PROXY')
        for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            if build_proxy and name in ('HTTP_PROXY', 'HTTPS_PROXY'):
                args += ['--build-arg', f'{name}={build_proxy}']
            elif os.environ.get(name):
                args += ['--build-arg', name]
        if os.environ.get('ABENCH_BUILD_NETWORK'):
            args += ['--network', os.environ['ABENCH_BUILD_NETWORK']]
        args += ['-f', str(ROOT / f'docker/profiles/{profile}/Dockerfile'), str(ROOT)]
        docker.run(args, timeout=7200)
        now = _verify_build_environment(docker, tag, profile, lock_path)
        actual = docker.inspect(tag)
        if (actual.get('Config', {}).get('Labels') or {}).get(BUILD_LABEL) != identity:
            raise DockerError('built image identity changed before environment verification completed')
        cached = {'image_id': actual['Id'], 'build_inputs_sha256': identity,
                  'environment_lock_sha256': _environment_lock_digest(lock_path), 'environment': now}
        with tempfile.NamedTemporaryFile(mode='w', dir=ready.parent, delete=False) as output:
            json.dump(cached, output)
            temporary = Path(output.name)
        temporary.replace(ready)
        return {'image': tag, 'environment': now, 'reused': False, 'build_inputs_sha256': identity}

def manage(action, profile='cpu-sci', agent=None, platform=None, docker=None, local=False):
    """Profile images, or with a harness name (--agent) that harness's CLI bundle (tools/harness_bundles.py)."""
    docker=docker or Docker()
    if agent and agent not in OFFLINE_AGENTS:
        return _manage_bundle(action, agent, platform, docker, local)
    m=manifest();info=docker.info();platform=platform or info['platform'];key=image_key(profile)
    version=image_version(profile);tag=f"{m['namespace']}/{key}:{version}"
    if platform not in m['images'][profile]['platforms']:
        raise ValueError(f'{profile} does not support platform {platform}')
    if action=='lock' and local:
        if (lock_entries().get(f'{key}@{version}') or {}).get(platform):
            raise ValueError(f'{key}@{version} is published; its registry digest is the lock')
        reference=resolve_image(profile,agent,platform,allow_unlocked=True)
        image_id=docker.inspect(reference)['Id']
        import datetime
        entry={'image_id':image_id,'reference':reference,
               'locked_at':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        previous=update_local_lock(f'{key}@{version}',platform,entry,require_unpublished=True)
        return {'locked_local':image_id,'reference':reference,'platform':platform,
                'replaced':previous.get('image_id') if previous and previous.get('image_id')!=image_id else None}
    if action=='build':
        return _build(profile, platform, docker, info, tag)
    if action=='push':
        if docker.run(['manifest','inspect',tag],check=False).returncode == 0:
            raise ValueError('profile version already published; bump version before pushing')
        docker.run(['push',tag],timeout=7200);return {'pushed':tag}
    if action=='lock':
        actual=docker.inspect(tag);refs=actual.get('RepoDigests') or []
        if not refs:raise ValueError('image must be pushed before locking; local image ID is not a registry digest')
        ref=next((r for r in refs if r.startswith(f"{m['namespace']}/{key}@")),None)
        if not ref:raise ValueError(f"no registry digest for {m['namespace']}/{key}")
        path=ROOT/'spec/env/images.lock.yaml'
        with lock_transaction(path):
            lock=yaml.safe_load(path.read_text())
            lock['images']=lock.get('images') or {}
            previous=(lock['images'].get(f'{key}@{version}') or {}).get(platform)
            if previous and previous.get('digest') != ref.split('@')[1]:
                raise ValueError(f'{key}@{version} is already locked to another digest; versions are immutable, bump the version')
            lock['images'].setdefault(f'{key}@{version}',{})[platform]={'digest':ref.split('@')[1],'reference':ref}
            _atomic_lock_write(path,yaml.safe_dump(lock,sort_keys=False),0o644)
        return {'locked':ref,'platform':platform}
    if action=='verify' and not (lock_entries().get(f'{key}@{version}') or {}).get(platform) and local_lock(profile,agent,platform):
        attestation=image_attestation(resolve_image(profile,agent,platform,allow_unlocked=True),profile,agent,platform)
        if not attestation['locked']:
            raise ValueError(f'{key}@{version}: local image ID differs from images.local.lock.yaml; lock it again after a rebuild')
        return attestation
    image=resolve_image(profile,agent,platform)
    if action=='pull':docker.run(['pull','--platform',platform,image],timeout=7200)
    if action in ['verify','pull']:return image_attestation(image,profile,agent,platform)
    raise ValueError(action)

def _manage_bundle(action, agent, platform, docker, local):
    import harness_bundles
    platform=platform or docker.info()['platform']
    if action=='build':
        return harness_bundles.build(agent, docker, platform)
    if action=='lock' and local:
        return harness_bundles.lock(agent, docker, platform)
    if action=='verify':
        info=harness_bundles.find(agent, platform)
        return harness_bundles.attestation(agent, platform, harness_bundles.content_digest(Path(info['directory'])))
    raise ValueError(f'harness bundles are built and locked on each host: use build, lock --local or verify with --agent {agent}')

def resolve_requirements(profile):
    """Resolve and pin transitive wheels on the intended image platform before publishing."""
    import tempfile,sys
    p=profiles()[profile]
    inputs=''.join(f'{k}=={v}\n' for k,v in p['packages'].items())
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'input.txt';report=Path(directory)/'report.json'
        source.write_text(inputs)
        command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed',
                 '--report',str(report),'-r',str(source)]
        ancestor=p.get('extends')
        while ancestor:
            constraint=ROOT/f'docker/profiles/{ancestor}/requirements.lock'
            command += ['-c',str(constraint)]
            ancestor=profiles()[ancestor].get('extends')
        subprocess.run(command,check=True)
        packages={x['metadata']['name'].lower():x['metadata']['version'] for x in json.loads(report.read_text())['install']}
    header='# input-sha256: '+hashlib.sha256(inputs.encode()).hexdigest()+'\n'
    path=ROOT/f'docker/profiles/{profile}/requirements.lock'
    path.write_text(header+'# Resolved transitive dependencies; regenerate with images resolve.\n'+''.join(f'{k}=={v}\n' for k,v in sorted(packages.items())))
    return {'profile':profile,'packages':len(packages),'lock':str(path)}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['generate','resolve','pull','build','push','lock','verify'])
    ap.add_argument('--profile',default='cpu-sci');ap.add_argument('--agent');ap.add_argument('--platform');ap.add_argument('--check',action='store_true')
    ap.add_argument('--local',action='store_true',help='lock: record the local image ID instead of a registry digest (scores, recorded as unpublished)')
    a=ap.parse_args(argv)
    if a.local and a.action!='lock':ap.error('--local applies to lock only')
    try:
        if a.action=='generate':result=generate(a.check)
        elif a.action=='resolve':result=resolve_requirements(a.profile)
        else:result=manage(a.action,a.profile,a.agent,a.platform,local=a.local)
    except (ValueError,DockerError) as exc:ap.error(str(exc))
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
