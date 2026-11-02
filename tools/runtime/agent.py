"""Agent lifecycle with separate host and container paths and tar-only transport."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from containers import Docker, DockerError, DockerPathMissing
from container_policy import create_options, verify_inspect, resources_for, host_scaled
from runtime.cleanup import cleanup_objects, stop_container

CONTAINER_PATHS = {k: Path(v) for k, v in {
    'view': '/abench/view', 'workspace': '/abench/workspace', 'home': '/abench/workspace/.home',
    'tmp': '/abench/workspace/.tmp', 'model': '/abench/workspace/model_store',
    'checkpoint': '/abench/workspace/checkpoint', 'submission': '/abench/workspace/submission',
    'retrieval_log': '/abench/workspace/.abench/retrieval_log.jsonl'}.items()}

def _home_files(files):
    """Validated (relative path, content) pairs of harness configuration written into the agent HOME."""
    from pathlib import PurePosixPath
    for name, content in sorted((files or {}).items()):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError(f'harness file must be relative to HOME: {name}')
        yield path, content

NATIVE_LOG_BYTES = 512 * 1024**2       # cap on a run's harness session transcripts

class AgentEnvironment:
    def start(self): raise NotImplementedError
    def write_files(self, files): raise NotImplementedError
    def execute(self, launch, env, timeout, out, err): raise NotImplementedError
    def collect(self, native=()): raise NotImplementedError
    def close(self): raise NotImplementedError

class HostSoftEnvironment(AgentEnvironment):
    def __init__(self, args, paths, **kwargs):
        if not args.smoke:
            raise ValueError('host backend requires --smoke')
        self.args, self.paths = args, paths
        self.policy = kwargs.get('policy')
        self.proxy = None
        self.record = {'backend': 'host', 'verified': False, 'reason': 'host_soft smoke only'}
    def start(self):
        if self.policy:
            from egress_proxy import EgressProxy
            self.proxy = EgressProxy(self.policy,self.paths['root']/'egress.jsonl',
                                     upstream=os.environ.get('ABENCH_EGRESS_UPSTREAM')).start()
        return self.record
    def write_files(self, files):
        for path, content in _home_files(files):
            target = self.paths['home'].joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    def execute(self, launch, env, timeout, out, err):
        from sandbox_runner import run_agent
        values = {**env, **launch.env}
        if self.proxy:
            values.update({k:f'http://127.0.0.1:{self.proxy.port}' for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy')})
        return run_agent(launch.argv, self.paths['workspace'], values, launch.stdin,
                         timeout=timeout, out_path=out, err_path=err)
    def collect(self, native=()):
        paths = dict(self.paths)
        paths['native'] = self.paths['root'] / 'native'
        for path, _ in _home_files({rel: '' for rel in native}):
            source = self.paths['home'].joinpath(*path.parts)
            if source.is_dir() and not source.is_symlink():
                shutil.copytree(source, paths['native'].joinpath(*path.parts), symlinks=True, dirs_exist_ok=True)
        if self.proxy:
            self.record['egress_summary'] = self.proxy.stop()
            self.proxy = None
            paths['egress_log'] = self.paths['root']/'egress.jsonl'
        return paths
    def close(self):
        if self.proxy:
            self.proxy.stop(); self.proxy = None

class DockerAgentEnvironment(AgentEnvironment):
    def __init__(self, args, paths, oj, image, sidecar_image=None, policy=None, harness=None):
        self.args, self.host_paths, self.oj = args, paths, oj
        self.paths = dict(CONTAINER_PATHS)
        self.docker, self.image = Docker(shutil.which("docker") or "docker"), image
        self.sidecar_image, self.policy = sidecar_image, policy
        self.harness, self._bundle = harness, None   # the harness whose CLI bundle is mounted (tools/harness_bundles.py)
        self.ids, self.volumes = [], []
        self.cid = self.sidecar = None
        # Object ownership is unique even when different cases share a human run id.
        self.label = f'{args.case}-{args.run_id}-{uuid.uuid4().hex[:8]}'
        self.labels = {'abench.run_id': args.run_id, 'abench.case': args.case}
        self.record = {'backend': 'docker', 'image_digest': image, 'run_label': self.label,'allocated_gpus':args.gpus}

    def _volume(self, suffix):
        name = self.docker.volume('abench-' + uuid.uuid4().hex + '-' + suffix, self.label, self.labels)
        self.volumes.append(name)
        return name

    def _create(self, image, cmd, options):
        cid = self.docker.create(image, cmd, self.label, options, self.labels)
        self.ids.append(cid)
        return cid

    def _discard_helper(self, cid):
        result = self.docker.remove(cid)
        if result.returncode:
            raise DockerError(f'helper container deletion failed: {cid}: {result.stderr}')
        self.ids.remove(cid)

    def _initialize_volumes(self, view, work, control):
        d = self.docker
        # Initialization helpers are offline, discarded, and never execute submitted code.
        init = self._create(self.image, ['sleep','infinity'], ['--network','none', '--user','0:0',
            '--mount',f'type=volume,src={view},dst=/abench/view',
            '--mount',f'type=volume,src={work},dst=/abench/workspace',
            '--mount',f'type=volume,src={control},dst=/abench/control'])
        d.start(init)
        d.exec(init, ['mkdir','-p', '/abench/workspace/.home', '/abench/workspace/.tmp',
                     '/abench/workspace/checkpoint','/abench/workspace/submission','/abench/workspace/model_store',
                     '/abench/workspace/.abench'])
        # Copy the view and cache large data independently.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='abench-small-view-') as raw:
            small = Path(raw) / 'view'
            shutil.copytree(self.host_paths['view'], small, ignore=shutil.ignore_patterns('data'))
            small.chmod(0o755)
            (small/'data').mkdir()
            d.upload(init, small, '/abench/view')
        d.upload(init, self.host_paths['control'], '/abench/control')
        d.exec(init, ['chown','-R','1000:1000','/abench/workspace','/abench/control'])
        self._discard_helper(init)

    def _start_sidecar(self, control, info, mounts):
        d = self.docker
        sock = self._volume('socket')
        side_mounts = [f'type=volume,src={sock},dst=/socket',
                       f'type=volume,src={control},dst=/control,readonly']
        side_resources = {'cpus':1,'memory':'1g','shm_size':'64m','pids':512,'gpus':0}
        side_options = create_options(side_resources,network='bridge',mounts=side_mounts,docker_info=info)
        side = self._create(self.sidecar_image,
            ['python3','/control/egress_proxy.py','serve','--policy','/control/policy.json',
             '--socket','/socket/proxy.sock','--log','/tmp/egress.jsonl'],
            [*side_options,*(['-e','ABENCH_EGRESS_UPSTREAM'] if os.environ.get('ABENCH_EGRESS_UPSTREAM') else [])])
        # Named volumes default root-owned; prepare the socket directory before dropping privilege.
        helper = self._create(self.image,['chown','1000:1000','/socket'],
            ['--network','none','--user','0:0','--mount',f'type=volume,src={sock},dst=/socket'])
        d.start(helper)
        d.run(['wait',helper])
        self._discard_helper(helper)
        d.start(side)
        self.sidecar = side
        mounts.append(f'type=volume,src={sock},dst=/run/abench-egress,readonly')
        from images import image_attestation
        checked = verify_inspect(d.inspect(side),image=self.sidecar_image,network='bridge',
                                 mounts=side_mounts,resources=side_resources,image_id=d.inspect(self.sidecar_image)['Id'],resources_enforced=info['resources_enforced'])
        self.record['sidecar'] = {'image_digest':self.sidecar_image,'inspect':checked,
                                 'image_attestation':image_attestation(self.sidecar_image,'egress')}
        if not checked['verified']:
            raise DockerError('sidecar policy verification failed')

    def start(self):
        d = self.docker
        info = d.info()
        self.record.update({k: info[k] for k in ('rootless', 'desktop', 'platform', 'resources_enforced')})
        resources = resources_for(self.oj)
        if os.environ.get('ABENCH_CALIBRATION_CPU') == '1':
            resources = {**resources,'gpus':0}
            self.record['validation_cpu_override'] = True
        resources = host_scaled(resources, info)
        self.record['resources'] = resources
        view, work, control = (self._volume(k) for k in ('view','workspace','control'))
        mounts = [f'type=volume,src={view},dst=/abench/view,readonly',
                  f'type=volume,src={work},dst=/abench/workspace',
                  f'type=volume,src={control},dst=/abench/control,readonly']
        self._initialize_volumes(view, work, control)
        if 'public_sources' in self.host_paths:
            vol = d.cached_file_volume(self.host_paths['public_sources'], self.image)
            mounts.append(f'type=volume,src={vol},dst=/abench/view/data,readonly')
        for src, dest in ((None if 'public_sources' in self.host_paths else self.host_paths['view'] / 'data','/abench/view/data'),
                          (self.host_paths.get('models'),'/abench/models'),
                          (self.host_paths.get('papers'),'/abench/papers')):
            if src and src.is_dir():
                vol = d.cached_volume(src, self.image)
                mounts.append(f'type=volume,src={vol},dst={dest},readonly')
        bundle = self.bundle()
        if bundle:
            import harness_bundles
            vol = d.cached_volume(Path(bundle['directory']), self.image)
            mounts.append(f'type=volume,src={vol},dst={harness_bundles.MOUNT},readonly')
            digest = next((r.get('digest') for r in reversed(getattr(d, 'cache_records', [])) if r.get('volume') == vol), None)
            self.record['harness_bundle'] = {'harness': bundle['harness'], 'version': bundle['version'],
                                             'volume': vol, 'mount': harness_bundles.MOUNT,
                                             **harness_bundles.attestation(bundle['harness'], info['platform'], digest)}
        self.record['data_caches'] = getattr(d, 'cache_records', [])
        if self.policy:
            self._start_sidecar(control, info, mounts)
        net = 'bridge' if self.args.network == 'open' else 'none'
        self.mounts = mounts
        options = create_options(resources, self.args.gpus, network=net, mounts=mounts,docker_info=info)
        self.cid = self._create(self.image, ['sleep','infinity'], options)
        d.start(self.cid)
        self.record['inspect'] = verify_inspect(d.inspect(self.cid), image=self.image, network=net,
                                              mounts=mounts, resources=resources, gpus=self.args.gpus, image_id=d.inspect(self.image)['Id'],resources_enforced=info['resources_enforced'])
        probe = '''import socket, json, os
r={}
try:
 socket.create_connection(('1.1.1.1',443),timeout=1).close(); r['direct_blocked']=False
except OSError: r['direct_blocked']=True
try:
 socket.getaddrinfo('example.com',443); r['dns_blocked']=False
except OSError: r['dns_blocked']=True
try:
 open('/abench/view/.write_probe','w').close(); r['view_readonly']=False
except OSError: r['view_readonly']=True
r['uid']=os.getuid()
try:
 r['memory_current_bytes']=int(open('/sys/fs/cgroup/memory.current').read())
except (OSError,ValueError): r['memory_current_bytes']=None
print(json.dumps(r))
'''
        probes = json.loads(d.exec(self.cid, ['python3','-c',probe]).stdout)
        self.record['probes'] = probes
        self.record['verified'] = (self.record['inspect']['verified'] and probes['uid'] == 1000
            and probes['view_readonly'] and (net != 'none' or probes['direct_blocked'] and probes['dns_blocked']))
        if not self.record['verified']:
            raise DockerError('container policy/probe verification failed: ' + json.dumps(self.record))
        if self.sidecar:
            forward = d.exec(self.cid, ['sh','-c','python3 /abench/control/forwarder.py --socket /run/abench-egress/proxy.sock >/tmp/forwarder.log 2>&1 &'])
            # Check socket availability before any paid authentication request.
            ready = False
            for _ in range(100):
                check = d.exec(self.cid,['python3','-c',"import socket; s=socket.socket(socket.AF_UNIX); s.connect('/run/abench-egress/proxy.sock')"],check=False)
                if check.returncode == 0:
                    ready = True; break
                if not d.inspect(self.sidecar)['State']['Running']:
                    break
                time.sleep(.1)
            if not ready:
                raise DockerError('egress sidecar did not become ready')
        return self.record

    def bundle(self):
        """This host's built bundle of the harness, or a ValueError naming the command that builds it."""
        if self.harness and self._bundle is None:
            import harness_bundles
            self._bundle = harness_bundles.find(self.harness, self.docker.info()['platform'])
        return self._bundle

    def default_path(self):
        """PATH of the task image (a CUDA profile adds its toolkit), with the harness bundle's bin first."""
        env = (self.docker.inspect(self.image).get('Config') or {}).get('Env') or []
        image_path = next((e.split('=', 1)[1] for e in env if e.startswith('PATH=')),
                          '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
        if self.bundle():
            import harness_bundles
            return harness_bundles.MOUNT + '/bin:' + image_path
        return image_path

    def write_files(self, files):
        import tempfile
        pairs = list(_home_files(files))
        if not pairs:
            return
        with tempfile.TemporaryDirectory(prefix='abench-home-files-') as raw:
            for path, content in pairs:
                target = Path(raw).joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            # Tar members carry uid 1000, so the agent owns its configuration.
            self.docker.upload(self.cid, raw, str(self.paths['home']))

    def begin_meter(self):
        import re
        raw = self.docker.exec(self.cid,['cat','/sys/fs/cgroup/cpu.stat'],check=False).stdout
        match = re.search(r'usage_usec (\d+)',raw)
        self.record['cpu_baseline_usec'] = int(match[1]) if match else None

    def execute(self, launch, env, timeout, out, err):
        from sandbox_runner import run_agent
        values = {**env, **launch.env}
        argv = [self.docker.executable,'exec','-i','--workdir','/abench/workspace']
        for key in values:
            argv += ['-e',key]
        argv += [self.cid,'timeout','--signal=TERM','--kill-after=5s',str(timeout),*launch.argv]
        started = time.monotonic()
        status, code = run_agent(argv,self.host_paths['workspace'],{**os.environ, **values, 'DOCKER_CONFIG':os.environ.get('DOCKER_CONFIG',str(Path.home()/'.docker'))},launch.stdin,
                                timeout=timeout+7,out_path=out,err_path=err)
        # 124 is coreutils timeout; 137 is timeout's KILL escalation only at the deadline, otherwise a kill
        # such as the container OOM killer. The container keeps running so collect() can read its cgroup.
        if status == 'timeout' or code == 124 or code == 137 and time.monotonic() - started >= timeout:
            return 'timeout', None
        return status, code

    def prepare_collection(self):
        """Freeze execution and its accounting before creating any resumable collection state."""
        d = self.docker
        if d.inspect(self.cid)['State']['Running']:
            import re
            usage = d.exec(self.cid,['sh','-c','cat /sys/fs/cgroup/cpu.stat; echo MEMORY; cat /sys/fs/cgroup/memory.peak; '
                                     'echo EVENTS; cat /sys/fs/cgroup/memory.events'],check=False)
            self.record['cgroup'] = usage.stdout
            for field, pattern in (('cpu_usage_usec', r'usage_usec (\d+)'), ('memory_peak_bytes', r'MEMORY\n(\d+)'),
                                   ('oom_kills', r'oom_kill (\d+)')):
                match = re.search(pattern, usage.stdout)
                self.record[field] = int(match[1]) if match else None
        else:
            self.record.setdefault('cpu_usage_usec', None)
        # Stop every child/background process before collecting mutable artifacts.
        state = stop_container(d, self.cid)
        self.record['stopped_before_collection'] = True
        self.record['container_oom_killed'] = bool(state.get('OOMKilled'))
        if getattr(self, 'sidecar', None):
            # Drain proxy connections once the agent is stopped; no retained process may continue a session.
            if d.inspect(self.sidecar)['State']['Running']:
                d.run(['stop','--time','5',self.sidecar],check=False)
            stop_container(d, self.sidecar)

    def collect(self, native=()):
        self.prepare_collection()
        d = self.docker
        target = self.host_paths['root'] / 'download'
        self._download_artifact(self.cid,'/abench/workspace/submission',target / 'submission',max_bytes=32*1024**3)
        # Download only declared artifacts, not arbitrary HOME/temp content.
        try:
            self._download_artifact(self.cid,'/abench/workspace/.abench',target / '.abench')
        except DockerPathMissing:
            pass
        # The harness's own session transcripts (declared HOME directories only), bounded; evidence, never scored.
        transfers, budget = {}, NATIVE_LOG_BYTES
        for path, _ in _home_files({rel: '' for rel in native}):
            try:
                copied = self._download_artifact(self.cid, str(self.paths['home'].joinpath(*path.parts)),
                                                target / 'native' / Path(*path.parts), max_bytes=budget, max_members=20000)
                budget -= copied.get('bytes', 0) if isinstance(copied, dict) else 0
                transfers[path.as_posix()] = 'collected'
            except DockerPathMissing:
                transfers[path.as_posix()] = 'absent'
            except (ValueError, DockerError) as error:   # symlink, size limit or timeout: keep the run, drop the evidence
                shutil.rmtree(target / 'native' / Path(*path.parts), ignore_errors=True)
                transfers[path.as_posix()] = f'rejected: {type(error).__name__}'
        if transfers:
            self.record['native_log_transfers'] = transfers
        if self.sidecar:
            self._download_artifact(self.sidecar, '/tmp/egress.jsonl', target / 'egress', contents=False)
        return {'submission':target / 'submission','retrieval_log':target / '.abench/retrieval_log.jsonl',
                'egress_log':target / 'egress/egress.jsonl','native':target / 'native'}

    def _download_artifact(self, container, source, destination, *, timeout=120, attempts=3, **limits):
        """Retry transport failures after shutdown, committing only a complete validated archive.

        One deadline covers all attempts. Archive rejection and absent paths are final; a failed transfer
        cannot leave partial files for the next attempt or replace a previously collected artifact.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        from agents.lifecycle import read_json, atomic_json
        root = getattr(self, 'host_paths', {}).get('root')
        receipt_path = root / '.collection-receipts.json' if root else None
        receipts = read_json(receipt_path) if receipt_path and receipt_path.is_file() else {}
        key = container + '\0' + source
        def members():
            from file_hashes import sha256_file
            rows = []
            for path in sorted(destination.rglob('*')):
                if path.is_symlink():
                    raise ValueError('collected artifact contains a symlink')
                metadata = path.stat()
                rows.append([path.relative_to(destination).as_posix(), metadata.st_mode, metadata.st_size,
                             metadata.st_mtime_ns, sha256_file(path) if path.is_file() else None])
            return rows
        if destination.exists() or destination.is_symlink():
            previous = receipts.get(key)
            if (previous and not destination.is_symlink() and previous.get('destination') == str(destination)
                    and previous.get('members') == members()):
                return previous.get('result')
            # A rename can finish just before a crash writing its receipt. Re-transfer it rather than trusting
            # an uncommitted or altered directory merely because it exists.
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError('unsafe artifact destination')
            shutil.rmtree(destination)
        deadline = time.monotonic() + timeout
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DockerError('artifact collection deadline exceeded')
            with tempfile.TemporaryDirectory(prefix='.collect-', dir=destination.parent) as raw:
                staging = Path(raw) / 'artifact'
                staging.mkdir()
                try:
                    result = self.docker.download(container, source, staging, timeout=remaining, **limits)
                except DockerPathMissing:
                    raise
                except DockerError:
                    if attempt + 1 == attempts or deadline <= time.monotonic():
                        raise
                else:
                    if destination.exists() or destination.is_symlink():
                        raise ValueError('artifact destination already exists')
                    staging.rename(destination)
                    receipts[key] = {'destination': str(destination), 'members': members(), 'result': result}
                    if receipt_path:
                        atomic_json(receipt_path, receipts)
                    if attempt:
                        self.record.setdefault('collection_retries', {})[source] = attempt
                    return result
            time.sleep(min(.1 * (attempt + 1), max(0, deadline - time.monotonic())))

    def close(self):
        if self.ids or self.volumes or getattr(self.docker, 'pending_containers', []) or 'cleanup' in self.record:
            cleanup_objects(self.docker, self.ids, self.volumes, self.record)
