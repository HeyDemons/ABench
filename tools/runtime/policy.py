"""Resource limits and verification of the configuration actually applied by Docker."""
from pathlib import Path
import re
import yaml
ROOT = Path(__file__).resolve().parents[2]

def policy():
    return yaml.safe_load((ROOT / 'spec/container_policy.yaml').read_text())
def bytes_value(value):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([kmgt]?)(?:i?b)?', str(value).lower())
    if not match:
        raise ValueError(f'invalid memory quantity: {value}')
    return int(float(match[1]) * 1024 ** ('kmgt'.index(match[2]) + 1 if match[2] else 0))
def resources_for(oj):
    env = oj.get('environment') or {}
    profile = env.get('profile') or oj.get('profile')
    if isinstance(profile, dict):
        profile = profile.get('name')
    profile = str(profile).split('@')[0]
    declared = oj.get('resources')
    if not isinstance(declared, dict):
        raise ValueError('case must declare resources')
    for key in ['budget_tier', 'hardware', 'wall_clock_limit_h', 'leaderboard']:
        if key not in declared:
            raise ValueError(f'missing required case resource: {key}')
    if not declared['hardware'].get('reference'):
        raise ValueError('missing hardware.reference')
    p = policy()
    if profile not in p['defaults']:
        raise ValueError(f'unknown profile: {profile}')
    result = {**p['defaults'][profile], **declared}
    for key, limit in p['limits'].items():
        value = result[key]
        actual, cap = (bytes_value(value), bytes_value(limit)) if key in ['memory','shm_size'] else (float(value),float(limit))
        if actual <= 0 and key != 'gpus' or actual < 0 or actual > cap:
            raise ValueError(f'{key} exceeds container policy: {value}')
    return result

def host_scaled(resources, docker_info=None):
    """All CPUs and nearly all memory of the Docker host (container policy host_scale), with /dev/shm and process
    limits to match, as on the original host runner, which set none. GPUs stay as declared. Without the host's
    totals the declared values stay."""
    scale = policy().get('host_scale')
    raw = (docker_info or {}).get('raw') or {}
    total, cpus = int(raw.get('MemTotal') or 0), int(raw.get('NCPU') or 0)
    if not scale or total <= 0 or cpus <= 0:
        return resources
    mib = 1024 ** 2   # whole MiB: the kernel rounds memory.max down to a page, and records must equal the cgroup
    memory = int(total * float(scale['memory_fraction'])) // mib * mib
    return {**resources, 'cpus': cpus, 'memory': memory, 'shm_size': int(memory * float(scale['shm_fraction'])) // mib * mib,
            'pids': int(scale['pids'])}

def gpu_mode(info=None):
    """Prefer Docker's native CDI inventory; explicit operator selection wins."""
    import os
    selected = os.environ.get('ABENCH_GPU_MODE', 'auto')
    if selected not in ('auto', '', 'cdi', 'gpus', 'nvidia'):
        raise ValueError('ABENCH_GPU_MODE must be auto, cdi, or gpus')
    if selected not in ('auto', ''):
        return 'gpus' if selected == 'nvidia' else selected
    if info is None:
        from containers import Docker
        info = Docker().info()
    raw = info.get('raw', info)
    native = [device.get('ID', '') for device in raw.get('DiscoveredDevices') or []
              if device.get('Source', 'cdi') == 'cdi']
    return 'cdi' if any(name.startswith('nvidia.com/gpu=') and not name.endswith('=all')
                        for name in native) else 'gpus'

def gpu_options(gpus, info=None):
    ids = [str(x) for x in gpus.split(',') if x] if isinstance(gpus, str) else list(map(str, gpus or []))
    if not ids:
        return []
    if gpu_mode(info) == 'cdi':
        return [item for gid in ids for item in ('--device', f'nvidia.com/gpu={gid}')]
    return ['--gpus', '"device=' + ','.join(ids) + '"']

def create_options(resources, gpus=None, network='none', mounts=None, docker_info=None):
    if network not in ['none', 'bridge']:
        raise ValueError('unsupported container network')
    args = ['--network', network, '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', '1000:1000', '--init', '--pids-limit', str(resources['pids']),
            '--memory', str(resources['memory']), '--cpus', str(resources['cpus']),
            '--shm-size', str(resources['shm_size'])]
    gpu_ids = [str(x) for x in gpus.split(',') if x] if isinstance(gpus,str) else list(map(str,gpus or []))
    if len(gpu_ids) != int(resources.get('gpus',0)):
        raise ValueError('allocated GPU count does not match case resources')
    args += gpu_options(gpu_ids, docker_info)
    for mount in mounts or []:
        args += ['--mount', mount]
    return args

def verify_inspect(inspected, image=None, network='none', mounts=None, resources=None, gpus=None, image_id=None, resources_enforced=True):
    host, cfg = inspected.get('HostConfig',{}), inspected.get('Config',{})
    failures = []
    checks = {'network': host.get('NetworkMode') == network,
              'user': cfg.get('User') == '1000:1000',
              'capabilities': 'ALL' in [x.upper() for x in host.get('CapDrop') or []] and not host.get('CapAdd'),
              'no_new_privileges': any(x.split(':')[0] == 'no-new-privileges' for x in host.get('SecurityOpt') or []),
              'unprivileged': not host.get('Privileged'), 'init': host.get('Init') is True,
              'no_bind_mounts': all(x.get('Type') == 'volume' for x in inspected.get('Mounts',[])),
              'pid_namespace': host.get('PidMode','') in ['', 'private'],
              'ipc_namespace': host.get('IpcMode','') in ['', 'private'],
              'user_namespace': host.get('UsernsMode','') != 'host',
              'cgroup_namespace': host.get('CgroupnsMode','') != 'host',
              'security_profile': not any('unconfined' in x for x in host.get('SecurityOpt') or [])}
    if image_id:
        checks['image_identity'] = inspected.get('Image') == image_id
    expected_gpu_ids = set(gpus.split(',') if isinstance(gpus,str) else map(str,gpus or []))
    device_requests = host.get('DeviceRequests') or []
    devices = host.get('Devices') or []
    cdi = host.get('CDIDevices') or []
    actual_gpu_ids = set()
    valid_devices = True
    for request in device_requests:
        if int(request.get('Count',0)) != 0 or not request.get('DeviceIDs'):
            valid_devices = False
        ids = list(map(str,request.get('DeviceIDs') or []))
        if request.get('Driver') == 'cdi':
            if request.get('Capabilities') or any(not name.startswith('nvidia.com/gpu=') for name in ids):
                valid_devices = False
            actual_gpu_ids.update(name.split('=',1)[-1] for name in ids)
        else:
            actual_gpu_ids.update(ids)
            if request.get('Driver','') not in ['', 'nvidia'] or not all(cap == ['gpu'] for cap in request.get('Capabilities') or []):
                valid_devices = False
    for device in devices + cdi:
        name = device if isinstance(device,str) else (device.get('Name') or device.get('PathOnHost') or '')
        if not name.startswith('nvidia.com/gpu='):
            valid_devices = False
        else:actual_gpu_ids.add(name.split('=',1)[1])
    checks['gpu_assignment'] = valid_devices and actual_gpu_ids == expected_gpu_ids
    if resources:
        checks['gpu_count'] = len(expected_gpu_ids) == int(resources.get('gpus',0))
    if image:
        checks['image'] = cfg.get('Image') == image
    if mounts is not None:
        actual = {(m['Destination'], bool(m.get('RW')), m.get('Name')) for m in inspected.get('Mounts',[])}
        expected = set()
        for m in mounts:
            if isinstance(m,str):
                fields = dict(x.split('=',1) for x in m.split(',') if '=' in x)
                expected.add((fields.get('dst') or fields.get('target') or fields.get('destination'),
                              'readonly' not in m.split(','), fields.get('src') or fields.get('source')))
            else:
                expected.add((m['destination'], not m.get('readonly',False), m.get('source') or m.get('name')))
        checks['mounts'] = actual == expected
    enforcement = {'enforced': bool(resources_enforced), 'requested': {}, 'actual': {}}
    if resources:
        enforcement['requested'] = {'memory': bytes_value(resources['memory']),
                                   'cpus': int(float(resources['cpus'])*10**9), 'pids': int(resources['pids'])}
        enforcement['actual'] = {'memory': host.get('Memory'), 'cpus': host.get('NanoCpus'), 'pids': host.get('PidsLimit')}
        if resources_enforced:
            checks.update({key: enforcement['actual'][key] == value for key,value in enforcement['requested'].items()})
        checks['shm_size'] = host.get('ShmSize') == bytes_value(resources['shm_size'])
    failures = [key for key,value in checks.items() if not value]
    return {'verified': not failures, 'checks': checks, 'failures': failures, 'resource_enforcement': enforcement}
