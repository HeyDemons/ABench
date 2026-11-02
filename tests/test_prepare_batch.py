"""abench prepare: where sources are downloaded, how they are checked, and how a case stops for the operator."""
import hashlib
import http.server
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'prepare'))
import batch  # noqa: E402

REV = 'a' * 40


class DownloadPlanTest(unittest.TestCase):
    def test_sources_map_to_files_below_the_case_raw_directory(self):
        plan = batch.downloads
        self.assertEqual(plan({'name': 'Qwen2.5-0.5B/', 'url': f'https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/{REV}',
                               'files': ['config.json', 'model.safetensors']}),
                         [(f'https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/{REV}/config.json', 'Qwen2.5-0.5B/config.json'),
                          (f'https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/{REV}/model.safetensors', 'Qwen2.5-0.5B/model.safetensors')])
        self.assertEqual(plan({'name': 'gpt2 tokenizer', 'note': 'RAW/gpt2/', 'files': ['vocab.json'],
                               'url': f'https://huggingface.co/datasets/o/r/tree/{REV}'}),
                         [(f'https://huggingface.co/datasets/o/r/resolve/{REV}/vocab.json', 'gpt2/vocab.json')])
        self.assertEqual(plan({'name': 'ETTh1', 'note': 'RAW/ETT-small/ETTh1.csv',
                               'url': f'https://github.com/o/r/blob/{REV}/ETT-small/ETTh1.csv'}),
                         [(f'https://raw.githubusercontent.com/o/r/{REV}/ETT-small/ETTh1.csv', 'ETT-small/ETTh1.csv')])
        self.assertEqual(plan({'name': 'a.hdf5', 'path': '1D/Burgers/Train', 'url': 'https://x.org/a.hdf5'}),
                         [('https://x.org/a.hdf5', '1D/Burgers/Train/a.hdf5')])
        self.assertEqual(plan({'name': 'ProteinGym DMS (v1.3)', 'url': 'https://x.org/files/DMS.zip?download=1'}),
                         [('https://x.org/files/DMS.zip?download=1', 'DMS.zip')])
        self.assertEqual(plan({'name': '../escape', 'url': 'https://x.org/f.bin'}), [('https://x.org/f.bin', 'f.bin')])
        for nothing in ({'name': 'gym-pusht', 'url': 'https://github.com/huggingface/gym-pusht'},
                        {'name': 'MQAR', 'generator': 'cases/A-LIN-002/harness/mqar.py'},
                        {'name': 'repo/', 'url': f'https://huggingface.co/o/r/tree/{REV}'}):
            self.assertEqual(plan(nothing), [])

    def test_every_case_has_a_download_plan_with_unique_destinations(self):
        import csv
        with (ROOT / 'verify.csv').open(newline='') as f:
            cases = [r['case_id'] for r in csv.DictReader(f)]
        for case in cases:
            paths = [p for s in batch.sources(case) for _, p in batch.downloads(s)]
            self.assertEqual(len(paths), len(set(paths)), case)
            self.assertFalse([p for p in paths if p.startswith('/') or '..' in Path(p).parts], case)


class ServedFiles:
    """A local HTTP server: path -> (content type, body)."""
    def __init__(self, files):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                kind, body = files[self.path]
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class DownloadTest(unittest.TestCase):
    def test_files_are_fetched_checked_and_web_pages_left_to_the_operator(self):
        body = b'col\n1\n'
        with ServedFiles({'/a.csv': ('text/csv', body), '/page': ('text/html; charset=utf-8', b'<html>login</html>')}) as web, \
                tempfile.TemporaryDirectory() as raw, patch.dict('os.environ', {'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'}):
            good = {'name': 'a.csv', 'url': web.url + '/a.csv', 'sha256': hashlib.sha256(body).hexdigest(),
                    'md5': hashlib.md5(body).hexdigest(), 'size': str(len(body))}
            page = {'name': 'data.hdf5', 'url': web.url + '/page'}
            with patch.object(batch, 'sources', return_value=[good, page]):
                got = batch.download('X-1', Path(raw))
                self.assertEqual((got['files'], got['fetched']), (1, 1))
                self.assertIn('serves a web page', got['manual'][0])
                self.assertEqual((Path(raw) / 'a.csv').read_bytes(), body)
                self.assertEqual(batch.download('X-1', Path(raw))['fetched'], 0)          # already there
            with patch.object(batch, 'sources', return_value=[dict(good, sha256='0' * 64)]):
                with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                    batch.download('X-1', Path(raw))
                self.assertFalse((Path(raw) / 'a.csv').exists())                          # fetched again next time


class FreezeRetryTest(unittest.TestCase):
    def test_missing_raw_paths_are_read_from_errors_and_linked_to_downloaded_files(self):
        text = ("FileNotFoundError: [Errno 2] No such file or directory: '/raw/gowalla/train.txt'\n"
                "SystemExit: raw file missing: /raw/ETT-small/ETTh1.csv\n"
                "source verification failed for ['esm2/config.json', 'vocab.json']")
        self.assertEqual(batch.missing_paths(text), ['/raw/gowalla/train.txt', '/raw/ETT-small/ETTh1.csv',
                                                     'esm2/config.json', 'vocab.json'])
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            (raw / 'train.txt').write_text('0 1\n')
            (raw / 'models' / 'tok').mkdir(parents=True)
            self.assertTrue(batch._link_missing(raw, str(raw / 'gowalla' / 'train.txt')))
            self.assertEqual((raw / 'gowalla' / 'train.txt').read_text(), '0 1\n')
            self.assertTrue(batch._link_missing(raw, 'cache/tok'))                           # a directory
            self.assertTrue((raw / 'cache' / 'tok').is_dir())
            self.assertFalse(batch._link_missing(raw, '/elsewhere/train.txt'))               # outside raw
            self.assertFalse(batch._link_missing(raw, 'absent.bin'))                         # nothing to link
            (raw / 'other').mkdir()
            (raw / 'other' / 'x.csv').write_text('a')
            (raw / 'x.csv').write_text('b')
            self.assertFalse(batch._link_missing(raw, 'y/x.csv'))                            # ambiguous


class PrepareFlowTest(unittest.TestCase):
    def test_each_case_stops_at_its_first_operator_step_and_the_rest_continue(self):
        rows = [{'id': 'A-1', 'category': 'llm_opt'}, {'id': 'B-1', 'category': 'cv'}, {'id': 'C-1', 'category': 'pro'}]
        calls = []

        def freeze(case, *args, **kwargs):
            calls.append(('freeze', case))
            if case == 'B-1':
                raise batch.Manual('freeze failed: raw file missing')

        def calibrate(case, *args, **kwargs):
            calls.append(('calibrate', case))
            if case == 'C-1':
                raise RuntimeError('docker gone')
            return {'verdict': 'evaluated'}
        with patch.object(batch, 'download', return_value={'files': 1, 'fetched': 0, 'manual': []}), \
                patch.object(batch, 'frozen', return_value=False), patch.object(batch, 'freeze', side_effect=freeze), \
                patch.object(batch, 'active', side_effect=lambda case: False), \
                patch.object(batch, 'calibrate', side_effect=calibrate), patch.object(batch, 'activate', return_value='activated'):
            out = batch.prepare(rows, Path('/raw'), Path('/key'), log=lambda line: None)
        self.assertEqual([r['status'] for r in out], ['done', 'needs operator', 'failed'])
        self.assertEqual(out[0]['steps']['activate'], 'activated')
        self.assertNotIn(('calibrate', 'B-1'), calls)
        self.assertIn('docker gone', out[2]['reason'])

    def test_calibrations_share_the_listed_gpus_one_case_each(self):
        import threading, time
        rows = [{'id': f'G-{i}', 'gpu': True} for i in range(4)] + [{'id': 'C-1', 'gpu': False}]
        lock, busy, peak, seen = threading.Lock(), {}, {}, {}

        def calibrate(case, row, key, run_id, gpu, nonreference):
            with lock:
                busy[gpu] = busy.get(gpu, 0) + 1
                peak[gpu] = max(peak.get(gpu, 0), busy[gpu])
                seen[case] = gpu
            time.sleep(0.05)
            with lock:
                busy[gpu] -= 1
            return {'verdict': 'evaluated'}
        with patch.object(batch, 'download', return_value={'files': 0, 'fetched': 0, 'manual': []}), \
                patch.object(batch, 'frozen', return_value=True), patch.object(batch, 'active', return_value=False), \
                patch.object(batch, 'needs_gpu', side_effect=lambda case, row: row['gpu']), \
                patch.object(batch, 'calibrate', side_effect=calibrate), patch.object(batch, 'activate', return_value='activated'):
            out = batch.prepare(rows, Path('/raw'), Path('/key'), gpus='1,2', jobs=4, log=lambda line: None)
        self.assertEqual([r['case'] for r in out], [r['id'] for r in rows])          # order kept
        self.assertTrue(all(r['status'] == 'done' for r in out))
        self.assertEqual({seen[f'G-{i}'] for i in range(4)}, {'1', '2'})
        self.assertIsNone(seen['C-1'])                                            # CPU calibration takes no GPU
        self.assertEqual({g: n for g, n in peak.items() if g}, {'1': 1, '2': 1})   # never two cases on one GPU

    def test_frozen_and_active_cases_skip_freezing_and_calibration(self):
        with patch.object(batch, 'download', return_value={'files': 2, 'fetched': 0, 'manual': []}), \
                patch.object(batch, 'frozen', return_value=True), patch.object(batch, 'active', return_value=True), \
                patch.object(batch, 'freeze') as freeze, patch.object(batch, 'calibrate') as calibrate, \
                patch.object(batch, 'activate', return_value='already active'):
            out = batch.prepare([{'id': 'A-1'}], Path('/raw'), Path('/key'), log=lambda line: None)
        freeze.assert_not_called()
        calibrate.assert_not_called()
        self.assertEqual(out[0]['steps'], {'download': {'files': 2, 'fetched': 0, 'manual': []}, 'freeze': 'already frozen',
                                           'calibrate': 'already active', 'activate': 'already active'})

    def test_activation_waits_for_a_tuned_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'cases' / 'B-OPT-004' / 'case').mkdir(parents=True)
            lock = root / 'cases' / 'B-OPT-004' / 'case' / 'data_lock.yaml'
            lock.write_text('activation: awaiting_data\nsynthetic: false\n')
            (root / 'verify.csv').write_text('case_id,baseline_tuned\nB-OPT-004,pending\n')
            with patch.object(batch, 'ROOT', root):
                with self.assertRaisesRegex(batch.Manual, 'baseline not tuned'):
                    batch.activate('B-OPT-004')
                self.assertIn('awaiting_data', lock.read_text())
                lock.write_text('activation: active\n')
                self.assertEqual(batch.activate('B-OPT-004'), 'already active')


if __name__ == '__main__':
    unittest.main()
