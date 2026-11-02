"""Existing ciphertext remains readable and unauthenticated plaintext is never released."""
import hashlib
import hmac
import io
import contextlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import hidden_store as store


def legacy_blob(key, nonce, plain):
    enc, mac = store._keys(key)
    encrypted = bytearray(len(plain))
    for offset in range(0, len(plain), 32):
        stream = hashlib.sha256(enc + nonce + (offset // 32).to_bytes(8, 'big')).digest()
        chunk = plain[offset:offset + 32]
        encrypted[offset:offset + len(chunk)] = bytes(a ^ b for a, b in zip(chunk, stream))
    content = store.MAGIC + nonce + bytes(encrypted)
    return content + hmac.new(mac, content, hashlib.sha256).digest()


class HiddenStreamingTests(unittest.TestCase):
    def test_file_cli_is_atomic_and_private_and_keeps_legacy_ciphertext(self):
        key, nonce, plain = bytes(range(32)), bytes(range(16)), b'private labels' * 100
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            key_file, source, encrypted, restored = (root / name for name in ('key', 'plain', 'data.enc', 'restored'))
            key_file.write_text(key.hex())
            source.write_bytes(plain)
            with patch.object(store.secrets, 'token_bytes', return_value=nonce), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(store.main(['hidden_store', 'encrypt', str(key_file), str(source), str(encrypted)]), 0)
                self.assertEqual(encrypted.read_bytes(), legacy_blob(key, nonce, plain))
                self.assertEqual(store.main(['hidden_store', 'decrypt', str(key_file), str(encrypted), str(restored)]), 0)
            self.assertEqual(restored.read_bytes(), plain)
            self.assertEqual(restored.stat().st_mode & 0o777, 0o600)
            key_file.write_text((b'x' * 32).hex())
            with self.assertRaises(ValueError):
                store.main(['hidden_store', 'decrypt', str(key_file), str(encrypted), str(restored)])
            self.assertEqual(restored.read_bytes(), plain)

    def test_legacy_bytes_and_streams_match_across_partial_blocks(self):
        key, nonce = bytes(range(32)), bytes(range(16))
        for size in (0, 1, 31, 32, 33, 95, 4097):
            plain = bytes(i % 251 for i in range(size))
            expected = legacy_blob(key, nonce, plain)
            with self.subTest(size=size), patch.object(store.secrets, 'token_bytes', return_value=nonce):
                self.assertEqual(store.encrypt_bytes(key, plain), expected)
                encrypted = io.BytesIO()
                identity = store.encrypt_stream(key, io.BytesIO(plain), encrypted, chunk_size=57)
                self.assertEqual(encrypted.getvalue(), expected)
                self.assertEqual(identity, {'sha256_plain': hashlib.sha256(plain).hexdigest(), 'bytes': size})
                decrypted = io.BytesIO()
                self.assertEqual(store.decrypt_stream(key, io.BytesIO(expected), decrypted, chunk_size=43), identity)
                self.assertEqual(decrypted.getvalue(), plain)
                self.assertEqual(store.decrypt_bytes(key, expected), plain)

    def test_corruption_and_wrong_keys_do_not_write_plaintext(self):
        key = b'k' * 32
        original = store.encrypt_bytes(key, b'private labels')
        variants = [original[:-1] + bytes([original[-1] ^ 1]), original[:-8], b'wrong header',
                    original[:22] + bytes([original[22] ^ 1]) + original[23:]]
        for blob in variants:
            with self.subTest(length=len(blob)), self.assertRaises(ValueError):
                destination = io.BytesIO()
                try:
                    store.decrypt_stream(key, io.BytesIO(blob), destination, chunk_size=7)
                finally:
                    self.assertEqual(destination.getvalue(), b'')
        destination = io.BytesIO()
        with self.assertRaises(ValueError):
            store.decrypt_stream(b'x' * 32, io.BytesIO(original), destination)
        self.assertEqual(destination.getvalue(), b'')

if __name__ == '__main__':
    unittest.main()
