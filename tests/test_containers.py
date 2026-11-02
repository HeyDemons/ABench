import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from containers import safe_extract, make_tar, content_hash
from container_policy import create_options, verify_inspect, resources_for
class ArchiveTest(unittest.TestCase):
    def archive(self,name,kind=tarfile.REGTYPE,size=0):
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w') as tar:
            info=tarfile.TarInfo(name);info.type=kind;info.size=size;info.linkname='/tmp/escape'
            tar.addfile(info,io.BytesIO(b'x'*size) if size else None)
        return out.getvalue()
    def test_reject_members(self):
        for name,kind in [('/absolute',tarfile.REGTYPE),('../escape',tarfile.REGTYPE),('a/../../x',tarfile.REGTYPE),('link',tarfile.SYMTYPE),('hard',tarfile.LNKTYPE),('dev',tarfile.CHRTYPE),('fifo',tarfile.FIFOTYPE)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as td:
                with self.assertRaises(ValueError):safe_extract(self.archive(name,kind),td)
    def test_size_and_count(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):safe_extract(self.archive('x',size=4),td,max_bytes=3)
            with self.assertRaises(ValueError):safe_extract(self.archive('x'),td,max_members=0)
    def test_existing_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'x').symlink_to('/tmp')
            with self.assertRaises(ValueError):safe_extract(self.archive('x/y'),td)
    def test_roundtrip_and_stable_content_hash(self):
        with tempfile.TemporaryDirectory() as td,tempfile.TemporaryDirectory() as out:
            p=Path(td)/'x';p.write_text('hello')
            first=content_hash(td);p.touch();self.assertEqual(first,content_hash(td))
            safe_extract(make_tar(td),out)
            self.assertEqual((Path(out)/'x').read_text(),'hello')
class PolicyTest(unittest.TestCase):
    def guide(self, case):
        import yaml
        from validate_task_statement import sections
        root = Path(__file__).resolve().parents[1]/'cases'/case/'case'
        return sections((root/'agent/task.md').read_text())['Sandbox Guide'], yaml.safe_load((root/'oj.yaml').read_text())

    def test_every_sandbox_guide_states_the_allocation_of_oj_yaml(self):
        from validate_task_statement import allocation_issues
        for path in sorted((Path(__file__).resolve().parents[1]/'cases').glob('*/case/oj.yaml')):
            with self.subTest(case=path.parts[-3]):
                self.assertEqual(allocation_issues(*self.guide(path.parts[-3])), [])

    def test_a_guide_that_disagrees_with_oj_yaml_is_reported(self):
        from validate_task_statement import allocation_issues
        guide, oj = self.guide('TC-FT-001')
        for old, new, field in [('8 CPU cores', '4 CPU cores', 'CPU cores'), ('at most 3 hours', 'at most 2 hours', 'session'),
                                ('capped at 8 CPU-core-hours', 'capped at 3 CPU-core-hours', 'agent_budget.cpu_core_hours'),
                                ('no GPU', 'one GPU', 'GPUs'), ('no GPU', 'no GPU, 2 GiB RAM', 'RAM bytes'),
                                ('capped at 8 CPU-core-hours', 'capped at an unspecified amount', 'cannot read')]:
            with self.subTest(field=field):
                self.assertIn(old, guide)
                issues = allocation_issues(guide.replace(old, new), oj)
                self.assertTrue(issues and all(field in issue for issue in issues), issues)

    def test_gpu_count_comes_from_the_hardware_label_and_vram_is_not_ram(self):
        from validate_task_statement import allocation_issues
        guide, oj = self.guide('PRO-ARL-001')
        self.assertEqual(allocation_issues(guide.replace('in one node', 'in one node, one GPU per rank'), oj), [])
        guide, oj = self.guide('IC-SPD-001')
        self.assertIn('24GB reference GPU', guide)          # VRAM, not a container RAM statement
        self.assertEqual(allocation_issues(guide, oj), [])
        self.assertIn('GPUs', allocation_issues(guide.replace('One 24GB reference GPU', '2 24GB reference GPUs'), oj)[0])

    def test_secure_flags(self):
        args=create_options({'cpus':8,'memory':'16g','pids':1024,'shm_size':'2g','gpus':0})
        for flag in ['--network','--cap-drop','--security-opt','--user','--init','--pids-limit','--memory','--cpus']:
            self.assertIn(flag,args)
        self.assertIn('none',args);self.assertIn('1000:1000',args)
    def test_cpu_memory_shm_and_processes_scale_to_the_host_as_on_the_original_runner(self):
        from container_policy import host_scaled, bytes_value
        declared=resources_for(self.guide('DS-GBM-001')[1])
        info={'raw':{'MemTotal':100*1024**3,'NCPU':64}}
        scaled=host_scaled(declared,info)
        self.assertEqual((scaled['cpus'],scaled['memory'],scaled['shm_size'],scaled['pids']),(64,90*1024**3,45*1024**3,65536))
        self.assertEqual(scaled['gpus'],declared['gpus'])
        self.assertEqual(host_scaled(declared,{}),declared)
        odd=host_scaled(declared,{'raw':{'MemTotal':100*1024**3+12345,'NCPU':4}})
        self.assertEqual((odd['memory']%2**20,odd['shm_size']%2**20),(0,0))
        args=create_options(scaled)
        self.assertEqual(args[args.index('--memory')+1],str(90*1024**3))
        inspected={'HostConfig':{'Memory':int(args[args.index('--memory')+1]),'ShmSize':bytes_value(args[args.index('--shm-size')+1]),
                                 'PidsLimit':int(args[args.index('--pids-limit')+1]),'NanoCpus':int(float(scaled['cpus'])*1e9)}}
        checks=verify_inspect(inspected,resources=scaled)['checks']
        self.assertTrue(all(checks[k] for k in ('memory','shm_size','pids','cpus')))
    def test_reject_missing_resources(self):
        with self.assertRaises(ValueError):resources_for({'environment':{'profile':'cpu-sci'}})
    def test_inspect_fail_closed(self):
        result=verify_inspect({'Config':{'User':'0'},'HostConfig':{'NetworkMode':'host','Privileged':True},'Mounts':[{'Type':'bind'}]})
        self.assertFalse(result['verified']);self.assertIn('network',result['failures']);self.assertIn('no_bind_mounts',result['failures'])
if __name__=='__main__':unittest.main()

class GPUInspectTest(unittest.TestCase):
    def base(self):
        return {'Config':{'User':'1000:1000','Image':'locked@sha256:abc'},
                'Image':'sha256:actual','HostConfig':{'NetworkMode':'none','CapDrop':['ALL'],
                'SecurityOpt':['no-new-privileges'],'Init':True,'IpcMode':'private'},'Mounts':[]}
    def test_exact_gpu_selection(self):
        inspected=self.base()
        inspected['HostConfig']['DeviceRequests']=[{'Count':0,'DeviceIDs':['2'],'Capabilities':[['gpu']]}]
        self.assertTrue(verify_inspect(inspected,gpus=['2'])['verified'])
        self.assertFalse(verify_inspect(inspected,gpus=['3'])['verified'])
        inspected['HostConfig']['DeviceRequests'][0]['Count']=-1
        self.assertFalse(verify_inspect(inspected,gpus=['2'])['verified'])

    def test_mount_must_be_the_expected_volume_not_only_the_expected_path(self):
        inspected = self.base()
        inspected['Mounts'] = [{'Type': 'volume', 'Name': 'judge-input', 'Destination': '/abench/view', 'RW': False}]
        mounts = ['type=volume,src=judge-input,dst=/abench/view,readonly']
        self.assertTrue(verify_inspect(inspected, mounts=mounts)['verified'])
        inspected['Mounts'][0]['Name'] = 'agent-workspace'
        self.assertFalse(verify_inspect(inspected, mounts=mounts)['checks']['mounts'])
    def test_cdi_selection_and_extra_device(self):
        inspected=self.base();inspected['HostConfig']['Devices']=[{'PathOnHost':'nvidia.com/gpu=2'}]
        self.assertTrue(verify_inspect(inspected,gpus=['2'])['verified'])
        inspected['HostConfig']['Devices'].append({'PathOnHost':'/dev/sda'})
        self.assertFalse(verify_inspect(inspected,gpus=['2'])['verified'])
    def test_actual_docker29_cdi_request_shape(self):
        inspected=self.base()
        request={'Driver':'cdi','Count':0,'DeviceIDs':['nvidia.com/gpu=2'],'Capabilities':None}
        inspected['HostConfig']['DeviceRequests']=[request]
        self.assertTrue(verify_inspect(inspected,gpus=['2'])['verified'])
        self.assertFalse(verify_inspect(inspected,gpus=['3'])['verified'])
        request['DeviceIDs']=['other.com/gpu=2']
        self.assertFalse(verify_inspect(inspected,gpus=['2'])['verified'])

    def test_host_namespace_and_changed_image(self):
        inspected=self.base();inspected['HostConfig']['PidMode']='host'
        self.assertFalse(verify_inspect(inspected)['verified'])
        inspected=self.base()
        self.assertFalse(verify_inspect(inspected,image_id='sha256:other')['verified'])

class GPUSelectionTest(unittest.TestCase):
    def test_native_cdi_inventory_selects_only_requested_devices(self):
        from unittest.mock import patch
        info={'DiscoveredDevices':[{'Source':'cdi','ID':'nvidia.com/gpu=2'},
                                   {'Source':'cdi','ID':'nvidia.com/gpu=3'}]}
        resources={'cpus':8,'memory':'16g','pids':1024,'shm_size':'2g','gpus':1}
        with patch.dict('os.environ',{'ABENCH_GPU_MODE':'auto'}):
            args=create_options(resources,['2'],docker_info=info)
        self.assertIn('nvidia.com/gpu=2',args)
        self.assertNotIn('nvidia.com/gpu=3',args)
        self.assertNotIn('--gpus',args)
    def test_explicit_override_and_legacy_fallback(self):
        from unittest.mock import patch
        from container_policy import gpu_mode,gpu_options
        native={'raw':{'DiscoveredDevices':[{'Source':'cdi','ID':'nvidia.com/gpu=2'}]}}
        with patch.dict('os.environ',{'ABENCH_GPU_MODE':'gpus'}):
            self.assertEqual(gpu_mode(native),'gpus')
            self.assertEqual(gpu_options(['2'],native),['--gpus','"device=2"'])
        with patch.dict('os.environ',{'ABENCH_GPU_MODE':'auto'}):
            self.assertEqual(gpu_mode({}),'gpus')
            self.assertEqual(gpu_mode(native),'cdi')
        with patch.dict('os.environ',{'ABENCH_GPU_MODE':'cdi'}):
            self.assertEqual(gpu_options(['2'],{}),['--device','nvidia.com/gpu=2'])
class GarbageCollectionTest(unittest.TestCase):
    """abench gc: a run id finds every object of the run; without one only stale, unused leftovers go."""
    def fake(self, objects):
        import json
        from types import SimpleNamespace
        from containers import Docker
        removed = []
        class Fake(Docker):
            def run(self, args, check=True, **kwargs):
                if args[0] == 'ps' or args[1] == 'ls':
                    kind = 'container' if args[0] == 'ps' else args[0]
                    key, _, value = args[-1].removeprefix('label=').partition('=')
                    ids = [o['id'] for o in objects if o['kind'] == kind and key in o['labels']
                           and (not value or o['labels'][key] == value)]
                    return SimpleNamespace(returncode=0, stdout='\n'.join(ids))
                if args[-2] == 'inspect' or args[1] == 'inspect':
                    o = next(o for o in objects if o['id'] == args[-1])
                    meta = ({'Config': {'Labels': o['labels']}, 'State': o.get('state', {}), 'Created': o['created']}
                            if o['kind'] == 'container' else {'Labels': o['labels'], 'CreatedAt': o['created']})
                    return SimpleNamespace(returncode=0, stdout=json.dumps([meta]))
                oid = args[-1]
                if next(o for o in objects if o['id'] == oid).get('in_use'):
                    return SimpleNamespace(returncode=1, stdout='')
                removed.append(oid)
                return SimpleNamespace(returncode=0, stdout='')
        return Fake(), removed
    def test_run_id_and_stale_cleanup(self):
        old, new = '2020-01-01T00:00:00.123456789Z', '2999-01-01T00:00:00+08:00'
        run = {'abench.run': 'TC-FT-001-r1-0a1b2c3d', 'abench.run_id': 'r1'}
        objects = [
            {'id': 'agent', 'kind': 'container', 'labels': run, 'created': old, 'state': {'Running': True}},
            {'id': 'judge', 'kind': 'container', 'labels': {'abench.run': 'judge-x', 'abench.run_id': 'r1'}, 'created': old,
             'state': {'Status': 'exited', 'FinishedAt': old}},
            {'id': 'view', 'kind': 'volume', 'labels': run, 'created': old},
            {'id': 'crashed', 'kind': 'container', 'labels': {'abench.run': 'judge-y'}, 'created': old,
             'state': {'Status': 'exited', 'FinishedAt': old}},
            {'id': 'collecting', 'kind': 'container', 'labels': {'abench.run': 'judge-z'}, 'created': old,
             'state': {'Status': 'exited', 'FinishedAt': new}},
            {'id': 'uploading', 'kind': 'container', 'labels': {'abench.run': 'judge-w'}, 'created': new,
             'state': {'Status': 'created'}},
            {'id': 'orphan', 'kind': 'volume', 'labels': {'abench.run': 'gone'}, 'created': old},
            {'id': 'mounted', 'kind': 'volume', 'labels': {'abench.run': 'live'}, 'created': old, 'in_use': True},
            {'id': 'abench-data-1', 'kind': 'volume', 'labels': {'abench.run': 'cache-1', 'abench.cache': '1'}, 'created': old},
            {'id': 'abench-data-2', 'kind': 'volume', 'labels': {'abench.run': 'cache-2'}, 'created': old},
        ]
        docker, removed = self.fake(objects)
        result = docker.gc('r1')
        self.assertEqual(set(removed), {'agent', 'judge', 'view'})
        self.assertEqual(result['volume'], ['view'])
        docker, removed = self.fake([o for o in objects if o['id'] not in ('agent', 'judge', 'view')])
        docker.gc()
        self.assertEqual(set(removed), {'crashed', 'orphan'})
        docker, removed = self.fake([o for o in objects if o['id'] not in ('agent', 'judge', 'view')])
        docker.gc(caches=True)
        self.assertEqual(set(removed), {'crashed', 'orphan', 'abench-data-1', 'abench-data-2'})
class JudgeHardwareTest(unittest.TestCase):
    """The judge scores on idle cards of the case's GPU count, and for Pro on an exclusive 4-GPU node; a GPU model
    other than the reference class is reported, not penalised."""
    def check(self, resources, gpus, inventory, class_matches=False):
        from unittest.mock import patch
        from judge_container import DockerStageLauncher
        launcher = DockerStageLauncher.__new__(DockerStageLauncher)
        launcher.resources, launcher.gpus = resources, gpus
        with patch('jobs.gpu_inventory', return_value=inventory):
            result = launcher._hardware_matches(), launcher.hardware_reason
        return result + (launcher.gpu_class_matches,) if class_matches else result
    def test_class_idle_and_pro_node(self):
        card = lambda i, cls='reference24', idle=True: {'index': str(i), 'class': cls, 'idle': idle}
        budget = {'gpus': 1, 'gpu_class': 'reference24', 'budget_tier': 'budget'}
        pro = {'gpus': 4, 'gpu_class': 'h100', 'budget_tier': 'pro'}
        self.assertEqual(self.check(budget, '0', [card(0)], class_matches=True), (True, None, True))
        self.assertEqual(self.check({'gpus': 0}, None, []), (True, None))
        self.assertIn('in use', self.check(budget, '0', [card(0, idle=False)])[1])
        self.assertEqual(self.check(budget, '0', [card(0, 'other')], class_matches=True), (True, None, False))
        self.assertIn('in use', self.check(budget, '0', [card(0, 'other', idle=False)])[1])
        self.assertIn('do not match', self.check(budget, '3', [card(0)])[1])
        self.assertEqual(self.check(pro, '0,1,2,3', [card(i, 'h100') for i in range(4)]), (True, None))
        self.assertIn('exclusive', self.check(pro, '0,1,2,3', [card(i, 'h100') for i in range(8)])[1])
        self.assertEqual(self.check(pro, '0,1,2,3', [card(i) for i in range(4)], class_matches=True), (True, None, False))
class ResourceEnforcementTest(unittest.TestCase):
    def test_missing_rootless_cgroups_are_explicit_not_misreported(self):
        inspected=GPUInspectTest().base()
        inspected['HostConfig'].update(Memory=0,NanoCpus=0,PidsLimit=0,ShmSize=64*1024**2)
        resources={'gpus':0,'cpus':1,'memory':'1g','pids':128,'shm_size':'64m'}
        self.assertFalse(verify_inspect(inspected,resources=resources)['verified'])
        result=verify_inspect(inspected,resources=resources,resources_enforced=False)
        self.assertTrue(result['verified'])
        self.assertFalse(result['resource_enforcement']['enforced'])
        self.assertEqual(result['resource_enforcement']['actual']['memory'],0)
        self.assertEqual(result['resource_enforcement']['requested']['memory'],1024**3)
