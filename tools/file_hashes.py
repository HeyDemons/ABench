"""Bounded-memory file hashing for operator and judge code, never copied into an agent view."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(path: Path, algorithm: str = "sha256") -> bytes:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def sha256_file(path: Path) -> str:
    return file_digest(path).hex()


def md5_file(path: Path) -> str:
    return file_digest(path, "md5").hex()


def git_blob_sha1(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    """Hash relative filenames and content using the existing frozen-data/submission format."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*")
                       if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file_digest(path))
    return digest.hexdigest()
