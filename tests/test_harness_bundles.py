"""Harness CLI bundles: built once per host, mounted read-only into the task container, locked by content."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import harness_bundles  # noqa: E402
import images  # noqa: E402

PLATFORM = 'linux/amd64'


def bundle_tar(files):
    """A tar stream like the builder's: path -> (bytes, mode)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as tf:
        for name, (data, mode) in files.items():
            info = tarfile.TarInfo('./' + name)
            info.size, info.mode = len(data), mode
            tf.addfile(info, io.BytesIO(data))
    return raw.getvalue()


class FakeDocker:
    def __init__(self, stream=b'', code=0, delay=0.0, fail=None):
        self.stream, self.code, self.delay, self.fail = stream, code, delay, fail
        self.calls, self.removed = [], []

    def info(self):
        return {'platform': PLATFORM}

    def run_to_file(self, args, path, timeout=None):
        import time
        self.calls.append(list(args))
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        Path(path).write_bytes(self.stream)
        return self.code, 'npm ERR! offline'

    def run(self, args, **kwargs):
        assert args[:2] == ['rm', '-f'], args
        self.removed.append(args[2])
        return SimpleNamespace(returncode=0, stdout='', stderr='')


class BundleTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / 'harness'
        self.lock = Path(tmp.name) / 'images.local.lock.yaml'
        for p in (patch.dict(os.environ, {'ABENCH_HARNESS_DIR': str(self.home)}),
                  patch.object(images, 'local_lock_path', return_value=self.lock)):
            p.start()
            self.addCleanup(p.stop)

    def test_build_scripts_pin_each_cli_and_wrap_its_entry_points(self):
        manifest = images.manifest()
        for name, spec in manifest['harnesses'].items():
            script = harness_bundles.script(name)
            self.assertIn('cp /usr/local/bin/node "$P/bin/node"', script)
            self.assertIn('tar -ch --hard-dereference', script)      # links become regular files, never hard links
            if spec.get('source'):
                self.assertIn(spec['source']['commit'], script)
                self.assertIn(f'"$P/bin/{name}"', script)
            else:
                self.assertIn(f"npm install -g --prefix \"$P\" {spec['npm']}@{spec['version']}", script)
                self.assertIn('readlink -f', script)
        keys = {harness_bundles.key(name, PLATFORM) for name in manifest['harnesses']}
        self.assertEqual(len(keys), len(manifest['harnesses']))
        self.assertNotEqual(harness_bundles.identity('dsh', PLATFORM), harness_bundles.identity('dsh', 'linux/arm64'))
        changed = json.loads(json.dumps(manifest))
        changed['harnesses']['dsh']['version'] = '9.9.9'
        with patch.object(images, 'manifest', return_value=changed):
            self.assertNotEqual(harness_bundles.key('dsh', PLATFORM), next(k for k in keys if k.startswith('harness-dsh@')))

    def test_a_built_bundle_is_reused_locked_by_content_and_attested(self):
        stream = bundle_tar({'bin/node': (b'node', 0o755), 'bin/dsh': (b'#!/bin/sh\nexec x\n', 0o755),
                             'lib/node_modules/@deepseek-ai/dsh/index.js': (b'console.log(1)', 0o644)})
        docker = FakeDocker(stream)
        with self.assertRaisesRegex(ValueError, 'images build --agent dsh'):
            harness_bundles.find('dsh', PLATFORM)
        built = harness_bundles.build('dsh', docker, PLATFORM)
        self.assertFalse(built['reused'])
        self.assertEqual(docker.calls[0][:2], ['run', '--rm'])
        self.assertEqual(docker.removed, [docker.calls[0][docker.calls[0].index('--name') + 1]])
        self.assertEqual((Path(built['directory']) / 'bin/dsh').stat().st_mode & 0o777, 0o755)
        self.assertTrue(harness_bundles.build('dsh', docker, PLATFORM)['reused'])
        self.assertEqual(len(docker.calls), 1)
        self.assertEqual(harness_bundles.find('dsh', PLATFORM)['content_sha256'], built['content_sha256'])
        # the digest equals the one the run's cache volume of this directory gets
        from runtime.cache import VolumeCache
        with patch.object(VolumeCache, '_volume', lambda self, digest, *a, **kwargs: digest):
            self.assertEqual(VolumeCache(None).directory(Path(built['directory']), 'image'), built['content_sha256'])
        self.assertFalse(harness_bundles.attestation('dsh', PLATFORM, built['content_sha256'])['locked'])
        locked = harness_bundles.lock('dsh', docker, PLATFORM)
        self.assertEqual(locked['locked_local'], built['content_sha256'])
        self.assertEqual(harness_bundles.attestation('dsh', PLATFORM, built['content_sha256'])['lock'], 'local')
        self.assertFalse(harness_bundles.attestation('dsh', PLATFORM, 'sha256-of-something-else')['locked'])
        (Path(built['directory']) / 'bin/dsh').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'changed since it was built'):
            harness_bundles.lock('dsh', docker, PLATFORM)

    def test_concurrent_builds_share_one_builder_and_corrupt_metadata_is_rebuilt(self):
        from concurrent.futures import ThreadPoolExecutor
        stream = bundle_tar({'bin/node': (b'node', 0o755)})
        docker = FakeDocker(stream, delay=0.2)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: harness_bundles.build('dsh', docker, PLATFORM), range(3)))
        self.assertEqual(len(docker.calls), 1)
        self.assertEqual(sorted(r['reused'] for r in results), [False, True, True])
        self.assertEqual([p.name for p in self.home.iterdir() if p.name.endswith('.partial')], [])
        record = Path(results[0]['directory'] + '.json')
        record.write_text('{"truncated')                         # a crash while publishing metadata
        with self.assertRaisesRegex(ValueError, 'no dsh bundle'):
            harness_bundles.find('dsh', PLATFORM)
        self.assertFalse(harness_bundles.build('dsh', docker, PLATFORM)['reused'])
        self.assertEqual(len(docker.calls), 2)

    def test_a_timed_out_builder_is_removed_and_leaves_nothing(self):
        from runtime.transport import DockerTimeout
        docker = FakeDocker(fail=DockerTimeout('builder timed out'))
        with self.assertRaises(DockerTimeout):
            harness_bundles.build('dsh', docker, PLATFORM)
        self.assertEqual(len(docker.removed), 1)
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), [p.name for p in self.home.iterdir() if p.suffix == '.lock'])

    def test_failed_or_unsafe_builds_leave_no_bundle(self):
        with self.assertRaisesRegex(RuntimeError, 'npm ERR! offline'):
            harness_bundles.build('dsh', FakeDocker(code=1), PLATFORM)
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as tf:
            link = tarfile.TarInfo('./bin/dsh')
            link.type, link.linkname = tarfile.SYMTYPE, '/etc/passwd'
            tf.addfile(link)
        with self.assertRaises(Exception):
            harness_bundles.build('dsh', FakeDocker(raw.getvalue()), PLATFORM)
        with self.assertRaisesRegex(ValueError, 'no dsh bundle'):
            harness_bundles.find('dsh', PLATFORM)


if __name__ == '__main__':
    unittest.main()
