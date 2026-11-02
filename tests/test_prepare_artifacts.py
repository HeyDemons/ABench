"""Download single-flight and recoverable publication preserve the original frozen data format."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare import run, batch, sources
from prepare.common import Frozen


class DownloadCacheTests(unittest.TestCase):
    def test_two_cases_share_a_download_but_writable_raw_files_are_independent(self):
        body = b'fixed release content'
        source = {'name': 'a.bin', 'url': 'https://example.org/a.bin', 'sha256': hashlib.sha256(body).hexdigest()}
        calls = []
        def fetch(url, target):
            calls.append(url)
            time.sleep(.03)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {'ABENCH_DOWNLOAD_CACHE': str(root / 'cache')}), \
                 patch.object(batch, 'sources', return_value=[source]), patch.object(batch, 'fetch', side_effect=fetch):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda case: batch.download(case, root / case), ['A', 'B']))
                self.assertEqual(len(calls), 1)
                self.assertEqual(sum(result['fetched'] for result in results), 1)
                self.assertNotEqual((root / 'A/a.bin').stat().st_ino, (root / 'B/a.bin').stat().st_ino)
                (root / 'A/a.bin').write_bytes(b'changed by operator')
                batch.download('C', root / 'C')
                self.assertEqual((root / 'C/a.bin').read_bytes(), body)
                cached = next((root / 'cache').glob('*/data'))
                cached.chmod(0o600)
                cached.write_bytes(b'tampered cache')
                batch.download('D', root / 'D')
                self.assertEqual(len(calls), 2)
                self.assertEqual((root / 'D/a.bin').read_bytes(), body)

    def test_failed_download_never_publishes_a_cache_or_partial_raw_file(self):
        body = b'expected'
        source = {'name': 'a.bin', 'url': 'https://example.org/a.bin', 'sha256': hashlib.sha256(body).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def fetch(url, target): target.write_bytes(b'bad')
            with patch.dict(os.environ, {'ABENCH_DOWNLOAD_CACHE': str(root / 'cache')}), \
                 patch.object(batch, 'sources', return_value=[source]), patch.object(batch, 'fetch', side_effect=fetch):
                with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                    batch.download('A', root / 'A')
            self.assertFalse((root / 'A/a.bin').exists())
            self.assertFalse(list((root / 'cache').glob('*/record.json')))
            self.assertFalse(list((root / 'cache').glob('*.partial')))


class SourceManifestTests(unittest.TestCase):
    def test_explicit_manifest_is_resolved_before_the_first_freeze_process(self):
        body = b'weights'
        revision = 'a' * 40
        source = {'name': 'human description', 'raw_path': 'weights', 'note': 'RAW/obsolete/',
                  'url': f'https://huggingface.co/o/r/tree/{revision}',
                  'files': {'model.bin': {'raw_path': 'weights/final.bin', 'sha256': hashlib.sha256(body).hexdigest()}}}
        self.assertEqual(sources.files(source), [
            (f'https://huggingface.co/o/r/resolve/{revision}/model.bin', 'weights/final.bin', {'sha256': hashlib.sha256(body).hexdigest()})])
        with self.assertRaisesRegex(ValueError, 'raw_path'):
            sources.files({'url': 'https://example.org/a', 'raw_path': '../outside'})
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / 'final.bin').write_bytes(body)
            module = SimpleNamespace(SOURCES=[source])
            def process(*args, **kwargs):
                self.assertEqual((raw / 'weights/final.bin').read_bytes(), body)
                return SimpleNamespace(returncode=0)
            with patch.object(batch, 'module_for', return_value=module), \
                 patch.object(batch.subprocess, 'run', side_effect=process) as execute:
                batch.freeze('X-TEST', raw, raw / 'key', False)
            execute.assert_called_once()


class FrozenPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'repo'
        self.root.mkdir()
        self.key = self.directory / 'key'
        self.key.write_text('0' * 64)
        self.case = 'X-TEST'
        self.case_dir = self.root / 'cases' / self.case / 'case'
        self.case_dir.mkdir(parents=True)
        self.patch = patch.object(run, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def data(self, value):
        model = self.directory / ('model-' + str(value))
        model.mkdir(exist_ok=True)
        (model / 'weights.bin').write_bytes(bytes([value]) * 11)
        return Frozen(public={'train.npz': {'x': np.arange(5) + value}},
                      hidden={'y': np.arange(3) + value}, sources_sha256={'raw': 'same'}, models={'model': model})

    def snapshot(self):
        return {str(path.relative_to(self.directory)): path.read_bytes()
                for base in (self.root / 'sandbox/data', self.root / 'sandbox/hidden', self.root / 'sandbox/models',
                             self.root / 'cases', self.directory / 'seeds')
                for path in base.rglob('*') if path.is_file()}

    def initial(self):
        with patch.dict(os.environ, {'ABENCH_TEST_SEED': '100'}):
            lock = run.freeze(self.case, self.data(1), self.key, False, seed_env='ABENCH_TEST_SEED')
        lock['activation'] = 'active'
        (self.case_dir / 'data_lock.yaml').write_text(yaml.safe_dump(lock))
        (self.case_dir / 'oj.yaml').write_text('old active OJ\n')
        return lock

    def test_generator_failure_restores_data_models_active_lock_oj_and_seed(self):
        self.initial()
        before = self.snapshot()
        def fail(*args, **kwargs):
            self.assertFalse((self.case_dir / 'oj.yaml').exists())
            self.assertEqual(yaml.safe_load((self.case_dir / 'data_lock.yaml').read_text())['activation'], 'awaiting_data')
            (self.case_dir / 'oj.yaml').write_text('partially generated OJ')
            raise subprocess.CalledProcessError(1, args[0])
        with patch.dict(os.environ, {'ABENCH_TEST_SEED': '200'}), patch.object(run.subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                run.freeze(self.case, self.data(2), self.key, False, regenerate=True, seed_env='ABENCH_TEST_SEED')
        self.assertEqual(before, self.snapshot())
        self.assertFalse((self.root / 'sandbox/.prepare' / self.case).exists())
        self.assertEqual(list((self.directory / 'seeds').glob('.*')), [])

    def test_concurrent_freezes_serialize_case_publication(self):
        original, busy, peak, mutex = run._write_frozen, [0], [0], threading.Lock()
        def build(*args, **kwargs):
            with mutex:
                busy[0] += 1
                peak[0] = max(peak[0], busy[0])
            try:
                time.sleep(.03)
                return original(*args, **kwargs)
            finally:
                with mutex: busy[0] -= 1
        data = [self.data(1), self.data(2)]
        with patch.object(run, '_write_frozen', side_effect=build), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda value: run.freeze(self.case, value, self.key, False), data))
        self.assertEqual(peak[0], 1)
        lock = yaml.safe_load((self.case_dir / 'data_lock.yaml').read_text())
        self.assertEqual(hashlib.sha256((self.root / lock['public'][0]['path']).read_bytes()).hexdigest(),
                         lock['public'][0]['sha256'])

if __name__ == '__main__':
    unittest.main()
