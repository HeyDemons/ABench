"""Static encrypted storage of hidden evaluation data (standard library only).

The hidden set exists on disk only as ciphertext; plaintext is decrypted only during evaluation into the judge's temporary directory and deleted afterwards.
The agent stage never gets the key (the runner neither puts the key file in the agent view nor writes it into the agent's environment variables).

Format: b"ABHID1" | nonce(16) | ciphertext | tag(32)
  keystream_i = SHA256(enc_key || nonce || i.to_bytes(8))      (SHA-256 counter mode)
  tag         = HMAC-SHA256(mac_key, b"ABHID1" || nonce || ciphertext)
  enc_key / mac_key are derived from the master key via HMAC.

This is a stopgap without third-party crypto libraries; once the container backend is ready it can switch to AES-GCM, with the header version bumped.

Usage:
  python3 tools/hidden_store.py keygen  KEYFILE
  python3 tools/hidden_store.py encrypt KEYFILE PLAIN  OUT.enc
  python3 tools/hidden_store.py decrypt KEYFILE IN.enc OUT
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import tempfile
from pathlib import Path

MAGIC = b"ABHID1"


def _keys(master: bytes) -> tuple[bytes, bytes]:
    return (hmac.new(master, b"abench-hidden-enc", hashlib.sha256).digest(),
            hmac.new(master, b"abench-hidden-mac", hashlib.sha256).digest())


def _xor_stream(enc_key: bytes, nonce: bytes, data: bytes, *, offset: int = 0) -> bytes:
    out = bytearray(len(data))
    prefix = enc_key + nonce
    position = 0
    while position < len(data):
        counter, skip = divmod(offset + position, 32)
        size = min(32 - skip, len(data) - position)
        ks = hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest()
        chunk = data[position:position + size]
        value = int.from_bytes(chunk, "little") ^ int.from_bytes(ks[skip:skip + size], "little")
        out[position:position + size] = value.to_bytes(size, "little")
        position += size
    return bytes(out)


def encrypt_stream(master, source, destination, *, chunk_size=1024 * 1024):
    """Write the existing ABHID1 format with bounded buffers; report its plaintext identity."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    enc_key, mac_key = _keys(master)
    nonce = secrets.token_bytes(16)
    header = MAGIC + nonce
    mac, digest = hmac.new(mac_key, header, hashlib.sha256), hashlib.sha256()
    destination.write(header)
    offset = 0
    for block in iter(lambda: source.read(chunk_size), b""):
        digest.update(block)
        encrypted = _xor_stream(enc_key, nonce, block, offset=offset)
        mac.update(encrypted)
        destination.write(encrypted)
        offset += len(block)
    destination.write(mac.digest())
    return {"sha256_plain": digest.hexdigest(), "bytes": offset}


def decrypt_stream(master, source, destination, *, chunk_size=1024 * 1024):
    """Authenticate a seekable ABHID1 source before writing any plaintext to a private parent stream."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    start = source.tell()
    header = source.read(len(MAGIC) + 16)
    source.seek(0, 2)
    end = source.tell()
    length = end - start - len(header) - 32
    if not header.startswith(MAGIC) or len(header) != len(MAGIC) + 16 or length < 0:
        raise ValueError("not an ABench hidden-store file")
    enc_key, mac_key = _keys(master)
    mac = hmac.new(mac_key, header, hashlib.sha256)
    source.seek(start + len(header))
    # Decode exactly the ciphertext authenticated here, even if the original file changes afterwards.
    with tempfile.TemporaryFile() as verified:
        remaining = length
        while remaining:
            block = source.read(min(chunk_size, remaining))
            if not block:
                raise ValueError("truncated hidden-store file")
            mac.update(block)
            verified.write(block)
            remaining -= len(block)
        if not hmac.compare_digest(mac.digest(), source.read(32)):
            raise ValueError("hidden-store integrity check failed (wrong key or modified file)")
        verified.seek(0)
        offset, digest = 0, hashlib.sha256()
        while offset < length:
            block = verified.read(min(chunk_size, length - offset))
            plain = _xor_stream(enc_key, header[len(MAGIC):], block, offset=offset)
            digest.update(plain)
            destination.write(plain)
            offset += len(block)
    return {"sha256_plain": digest.hexdigest(), "bytes": offset}


def read_key(path: Path) -> bytes:
    raw = Path(path).read_text(encoding="ascii").strip()
    key = bytes.fromhex(raw)
    if len(key) != 32:
        raise ValueError("hidden key must be 32 bytes hex")
    return key


def encrypt_bytes(master: bytes, plain: bytes) -> bytes:
    enc_key, mac_key = _keys(master)
    nonce = secrets.token_bytes(16)
    ct = _xor_stream(enc_key, nonce, plain)
    tag = hmac.new(mac_key, MAGIC + nonce + ct, hashlib.sha256).digest()
    return MAGIC + nonce + ct + tag


def decrypt_bytes(master: bytes, blob: bytes) -> bytes:
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 16 + 32:
        raise ValueError("not an ABench hidden-store file")
    enc_key, mac_key = _keys(master)
    nonce = blob[len(MAGIC):len(MAGIC) + 16]
    ct, tag = blob[len(MAGIC) + 16:-32], blob[-32:]
    if not hmac.compare_digest(tag, hmac.new(mac_key, MAGIC + nonce + ct, hashlib.sha256).digest()):
        raise ValueError("hidden-store integrity check failed (wrong key or modified file)")
    return _xor_stream(enc_key, nonce, ct)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 2
    cmd = argv[1]
    if cmd == "keygen" and len(argv) == 3:
        p = Path(argv[2]); p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
        print("wrote", p); return 0
    if cmd in ("encrypt", "decrypt") and len(argv) == 5:
        key = read_key(Path(argv[2]))
        target = Path(argv[4])
        temporary = None
        try:
            with Path(argv[3]).open('rb') as source, \
                    tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                transform = encrypt_stream if cmd == 'encrypt' else decrypt_stream
                transform(key, source, output)
                size = output.tell()
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print("wrote", argv[4], size, "bytes"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
