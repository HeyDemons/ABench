"""Publication contracts: remove private material without changing authoritative artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import export_public
import redact_records as records


class _TemporaryPublication(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path


class RecordPublicationTests(_TemporaryPublication):
    def test_private_identity_and_credentials_are_removed_but_measurements_are_preserved(self):
        secret, gpu, image = 'dummy-provider-credential', 'GPU-11111111-2222-3333-4444-555555555555', 'sha256:' + 'a' * 64
        original = {
            'provenance': 'judge_real', 'scorable': True, 'metrics': {'accuracy': .875, 'tokens': 8192},
            'software': {'python': '3.11.13', 'distributions': {'fixture': '10.0.0.1'}},
            'hardware': {'cpu': 'Example CPU', 'gpus': [{'name': 'NVIDIA RTX A6000', 'uuid': gpu}]},
            'image_id': image, 'view': '/abench/view/task.md', 'path': '/abench-private/operator/state',
            'source': 'https://arxiv.org/abs/2401.00001?context=physics%20and%20math',
            'model_endpoint': {'base_url': 'https://gateway.example.invalid/v1'},
            'error': 'echo ' + secret + ' https://gateway.example.invalid/v1/request',
            'clientSecret': 'dummy-client-secret',
            'download': 'https://operator:dummy-password@example.org/file?download=1&token=dummy-url-token',
            'proxy': 'http://127.0.0.1:7890', 'loopback': '::1',
            'sandbox': {
                'raw': {'ID': 'private-daemon', 'Name': 'validation-node', 'NCPU': 64},
                'inspect': {'Id': 'c' * 64, 'Image': image, 'State': {},
                    'Config': {'Hostname': 'validation-node', 'Env': [
                        'OPENAI_API_KEY=' + secret, 'HOME=/home/operator',
                        'HTTP_PROXY=http://127.0.0.1/' + secret,
                        'DOCKER_HOST=tcp://operator-docker.example.invalid:2376']},
                    'NetworkSettings': {'IPAddress': '172.18.0.2', 'GlobalIPv6Address': 'fd00::123'}},
            },
        }
        source, target = self.write('evaluation.json', json.dumps(original)), self.root / 'public.json'
        before = source.read_bytes()
        records.publish(source, target, environ={})
        public = json.loads(target.read_text())
        self.assertEqual(source.read_bytes(), before)
        for key in ('metrics', 'software', 'image_id', 'view', 'source', 'proxy', 'loopback'):
            self.assertEqual(public[key], original[key])
        self.assertEqual(public['hardware']['gpus'][0]['name'], original['hardware']['gpus'][0]['name'])
        self.assertEqual(public['sandbox']['inspect']['Image'], image)
        self.assertEqual(public['provenance'], 'sanitized_public_copy')
        self.assertFalse(public['scorable'])
        self.assertFalse(public[records.MARKER]['scoring_attestation'])
        for private in (secret, gpu, 'private-daemon', 'validation-node', '/home/operator', '/abench-private',
                        'gateway.example.invalid', 'operator-docker.example.invalid', '172.18.0.2', 'fd00::123',
                        'c' * 64, 'dummy-client-secret', 'dummy-password', 'dummy-url-token'):
            self.assertNotIn(private, json.dumps(public))

    def test_publication_refuses_overwriting_originals_existing_files_or_symlinks(self):
        source = self.write('input.json', '{"metrics":{"accuracy":1}}')
        target = self.write('public.json', 'existing publication')
        alias = self.root / 'alias.json'
        alias.symlink_to(source)
        for input_path, output_path in ((source, source), (source, target), (source, alias), (alias, self.root / 'new.json')):
            with self.subTest(input=input_path.name, output=output_path.name), self.assertRaises(records.RedactionError):
                records.publish(input_path, output_path, environ={})
        self.assertEqual(source.read_text(), '{"metrics":{"accuracy":1}}')
        self.assertEqual(target.read_text(), 'existing publication')
        self.assertFalse((self.root / 'new.json').exists())

class SourceSnapshotTests(_TemporaryPublication):
    def setUp(self):
        super().setUp()
        self.repo = self.root / 'repo'
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True, capture_output=True)
        self.public = {'README.md': b'public docs', 'abench': b'#!/bin/sh\n',
                       'cases/TEST/case/agent/task.md': b'exact agent task'}
        private = {name: b'private configuration' for name in (
            '.env', 'runs/private.json', 'sandbox/hidden/private.enc', 'spec/env/images.local.lock.yaml',
            '.ssh/config', '.aws/credentials', '.kube/config', '.docker/config.json', '.config/config.yaml',
            '.netrc', '.npmrc', '.git-credentials')}
        for name in ('export_public.py', 'redact_records.py'):
            self.public['tools/' + name] = (Path(export_public.__file__).parent / name).read_bytes()
        for name, content in {**self.public, **private}.items():
            self.write('repo/' + name, content)
        (self.repo / 'abench').chmod(0o755)
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True, capture_output=True)

    def test_snapshot_preserves_source_and_manifest_but_excludes_private_state_and_history(self):
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.org',
                        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'private history fixture'], check=True, capture_output=True)
        target = self.root / 'public.tar.gz'
        export_public.export(self.repo, target)
        with tarfile.open(target) as archive:
            self.assertEqual(set(archive.getnames()), {*self.public, export_public.MANIFEST})
            manifest = json.load(archive.extractfile(export_public.MANIFEST))
            self.assertFalse(manifest['git_history_included'])
            for name, content in self.public.items():
                actual = archive.extractfile(name).read()
                self.assertEqual(actual, content)
                self.assertEqual(hashlib.sha256(actual).hexdigest(), manifest['files'][name]['sha256'])
            self.assertEqual(archive.getmember('abench').mode, 0o755)
            self.assertTrue(all(m.uid == 0 and m.gid == 0 and m.uname == '' and m.gname == '' and m.mtime == 0 for m in archive))
        self.assertTrue((self.repo / '.env').is_file())

    def test_sensitive_or_linked_sources_never_leave_a_partial_archive(self):
        secret = '\u79c1\u5bc6-credential-value'
        source, target = self.repo / 'README.md', self.root / 'public.tar.gz'
        for content in (b'x' * (1024**2 - 1) + secret.encode(),
                        b'https://operator:dummy-password@private-gateway.invalid/path'):
            with self.subTest(kind='credential' if len(content) > 1024**2 else 'URL'):
                source.write_bytes(content)
                with self.assertRaises(ValueError):
                    export_public.export(self.repo, target, secrets=[secret])
                self.assertFalse(target.exists())
        source.unlink()
        source.symlink_to(self.repo / '.env')
        with self.assertRaises(ValueError):
            export_public.export(self.repo, target)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
