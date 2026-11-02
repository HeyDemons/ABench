"""Image packaging identity and concurrent local builds, without a Docker daemon or network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import images


SNAPSHOT = {'schema': 'abench.environment/v1', 'profile': 'cpu-sci',
            'software': {'python': '3.11.13', 'implementation': 'CPython',
                         'distributions': {'numpy': '2.2.6'}, 'torch': None},
            'hardware': {'gpus': []}}


class FakeDocker:
    executable = 'fake-docker'

    def __init__(self, daemon_id='test-daemon'):
        self.daemon_id = daemon_id
        self.images = {'python:3.11-test': {'Id': 'sha256:base'}}
        self.mutex = threading.Lock()
        self.builds = []
        self.parent_tags = []
        self.parent_tag_removals = []
        self.build_parent_ids = []
        self.client_proxy_environments = []
        self.build_failure = self.remove_failure = None
        self.environment_reads = 0
        self.snapshot = deepcopy(SNAPSHOT)

    def info(self):
        return {'platform': 'linux/amd64', 'raw': {'ID': self.daemon_id}}

    def inspect(self, reference):
        with self.mutex:
            if reference not in self.images:
                raise images.DockerError('image not found')
            return deepcopy(self.images[reference])

    def run(self, args, **kwargs):
        if args[:2] == ['context', 'show']:
            return SimpleNamespace(returncode=0, stdout='test-context\n')
        if args[0] == 'tag':
            with self.mutex:
                source = next(item for item in self.images.values() if item['Id'] == args[1])
                self.images[args[2]] = deepcopy(source)
                self.parent_tags.append((args[1], args[2]))
            return SimpleNamespace(returncode=0, stdout='')
        if args[:2] == ['image', 'rm']:
            with self.mutex:
                self.parent_tag_removals.append(args[2])
                if self.remove_failure:
                    return SimpleNamespace(returncode=1, stdout='', stderr=self.remove_failure)
                self.images.pop(args[2])
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if args[0] == 'build':
            with self.mutex:
                self.builds.append(list(args))
                self.client_proxy_environments.append({name: os.environ.get(name)
                                                       for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')})
                index = len(self.builds)
                parent = next((item.split('=', 1)[1] for item in args if item.startswith('PROFILE_IMAGE=')), None)
                if parent:
                    if parent.startswith('sha256:'):
                        raise images.DockerError('BuildKit does not accept a bare image ID in FROM')
                    self.build_parent_ids.append(self.images[parent]['Id'])
            if self.build_failure:
                raise self.build_failure
            time.sleep(.05)  # Give simultaneous requests time to reach the interprocess file lock.
            tag = args[args.index('-t') + 1]
            key, value = args[args.index('--label') + 1].split('=', 1)
            with self.mutex:
                self.images[tag] = {'Id': f'sha256:build{index}', 'Config': {'Labels': {key: value}}}
            return SimpleNamespace(returncode=0, stdout='')
        if args[0] == 'run':
            with self.mutex:
                self.environment_reads += 1
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.snapshot))
        raise AssertionError(args)


class ImageBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'repo'
        (self.root / 'spec/env').mkdir(parents=True)
        (self.root / 'tools/runtime').mkdir(parents=True)
        (self.root / 'tools/record_environment.py').write_text('# environment recorder\n')
        (self.root / 'tools/runtime/forwarder.py').write_text('# forwarder\n')
        self.config = {'namespace': 'registry.example/abench', 'node': '24.14.0',
                       'harnesses': {'claude': {'version': '2.1.312', 'npm': '@anthropic-ai/claude-code'}},
                       'images': {'cpu-sci': {'platforms': ['linux/amd64']},
                                  'egress': {'version': '1.0.0', 'source': 'python:3.11-test',
                                             'platforms': ['linux/amd64']}}}
        (self.root / 'spec/env/images.yaml').write_text(yaml.safe_dump(self.config))
        (self.root / 'spec/env/profiles.yaml').write_text(yaml.safe_dump({'profiles': {
            'cpu-sci': {'version': '1.0.0', 'base': 'python:3.11-test', 'packages': {'numpy': '2.2.6'}}}}))
        (self.root / 'spec/env/images.lock.yaml').write_text('images: {}\n')
        self.root_patch = patch.object(images, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.env_patch = patch.dict(os.environ, {'ABENCH_IMAGE_BUILD_CACHE': str(Path(self.tmp.name) / 'cache')})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        images.generate()

    def write_manifest(self, config):
        (self.root / 'spec/env/images.yaml').write_text(yaml.safe_dump(config))

    def test_same_concurrent_requests_build_and_verify_only_once(self):
        docker = FakeDocker()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: images.manage('build', docker=docker), range(4)))
        self.assertEqual(len(docker.builds), 1)
        self.assertEqual(docker.environment_reads, 1)
        self.assertEqual(sum(not result['reused'] for result in results), 1)
        self.assertEqual(len({result['build_inputs_sha256'] for result in results}), 1)
        (self.root / 'case').mkdir()
        (self.root / 'case/task.md').write_text('Changed runtime prompt')
        self.assertTrue(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 1)
        (self.root / 'tools/runtime/forwarder.py').write_text('# changed image content\n')
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_unvalidated_or_retagged_images_never_hit_success_cache(self):
        docker = FakeDocker()
        tag = 'registry.example/abench/cpu-sci:1.0.0'
        docker.images[tag] = {'Id': 'sha256:old', 'Config': {'Labels': {}}}
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        docker.images[tag]['Id'] = 'sha256:replaced'
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_failed_environment_check_has_no_success_marker(self):
        docker = FakeDocker()
        path = images.env_lock_path('cpu-sci', '1.0.0')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(SNAPSHOT))
        docker.snapshot['software']['distributions']['numpy'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'profile environment changed; bump profile version'):
            images.manage('build', docker=docker)
        self.assertEqual(list((Path(self.tmp.name) / 'cache').rglob('*.json')), [])
        docker.snapshot = deepcopy(SNAPSHOT)
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        self.assertTrue(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_actual_parent_identity_and_daemon_separate_build_reuse(self):
        docker = FakeDocker()
        first = images.manage('build', docker=docker)
        docker.images['python:3.11-test']['Id'] = 'sha256:base-two'          # the base tag now names another image
        second = images.manage('build', docker=docker)
        self.assertEqual(first['image'], second['image'])
        self.assertNotEqual(first['build_inputs_sha256'], second['build_inputs_sha256'])
        self.assertFalse(images.manage('build', docker=FakeDocker('another-daemon'))['reused'])

    def test_a_harness_name_builds_its_bundle_not_an_image(self):
        docker = FakeDocker()
        with patch('harness_bundles.build', return_value={'key': 'harness-claude@x'}) as build:
            self.assertEqual(images.manage('build', agent='claude', docker=docker), {'key': 'harness-claude@x'})
        build.assert_called_once_with('claude', docker, 'linux/amd64')
        self.assertEqual(docker.builds, [])
        self.assertEqual(images.manage('build', agent='mock', docker=docker)['image'],
                         'registry.example/abench/cpu-sci:1.0.0')                # offline agents use the profile

    def test_build_proxy_override_preserves_host_proxy_and_does_not_change_content_identity(self):
        docker = FakeDocker()
        host = {'HTTP_PROXY': 'http://127.0.0.1:8080', 'HTTPS_PROXY': 'http://127.0.0.1:8080',
                'NO_PROXY': 'localhost,127.0.0.1'}
        with patch.dict(os.environ, {**host, 'ABENCH_BUILD_PROXY': 'http://10.0.2.2:8080',
                                    'ABENCH_BUILD_NETWORK': 'host'}):
            version = images.image_version('cpu-sci', 'claude')
            first = images.manage('build', docker=docker)
            args = docker.builds[0]
            self.assertIn('HTTP_PROXY=http://10.0.2.2:8080', args)
            self.assertIn('HTTPS_PROXY=http://10.0.2.2:8080', args)
            self.assertIn('NO_PROXY', args)
            self.assertEqual(docker.client_proxy_environments, [host])
            self.assertEqual({name: os.environ[name] for name in host}, host)
            with patch.dict(os.environ, {'ABENCH_BUILD_PROXY': 'http://another-build-proxy:8080',
                                        'HTTPS_PROXY': 'http://another-host-proxy:8080'}):
                (self.root / 'task.md').write_text('An unrelated runtime prompt')
                second = images.manage('build', docker=docker)
                self.assertTrue(second['reused'])
                self.assertEqual(second['build_inputs_sha256'], first['build_inputs_sha256'])
                self.assertEqual(images.image_version('cpu-sci', 'claude'), version)
        self.assertEqual(len(docker.builds), 1)

    def test_default_proxy_build_arguments_keep_existing_environment_passthrough(self):
        docker = FakeDocker()
        host = {'HTTP_PROXY': 'http://host-proxy:8080', 'HTTPS_PROXY': 'http://host-proxy:8080',
                'NO_PROXY': 'localhost,127.0.0.1'}
        with patch.dict(os.environ, {**host, 'ABENCH_BUILD_PROXY': ''}):
            images.manage('build', docker=docker)
        args = docker.builds[0]
        self.assertEqual(docker.client_proxy_environments, [host])
        for name in host:
            self.assertIn(name, args)
            self.assertFalse(any(item.startswith(name + '=') for item in args))


if __name__ == '__main__':
    unittest.main()
