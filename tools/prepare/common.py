"""Shared helpers for freezing the data of the ABench standard cases (run by the operator, never inside a sandbox).

Each case module tools/prepare/<case_id_lower_underscored>.py defines
  SOURCES  list of {name, url, md5|sha256 (optional), license, note}  – what to download, and from where
  build(raw: Path) -> Frozen        real data from the operator's download directory
  synthetic(n: int) -> Frozen       learnable fake data with the same schema, for plumbing tests only
and tools/prepare/run.py does the rest: writes the public files, encrypts the hidden split, copies frozen model
files, and records every hash in cases/<ID>/case/data_lock.yaml, which tools/build_standard_cases.py merges into
case/oj.yaml. The judge refuses to run unless the files on disk match those hashes.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from file_hashes import md5_file, sha256_file, tree_hash

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Frozen:
    public: dict[str, dict[str, np.ndarray]]          # file name -> arrays (saved as .npz) under sandbox/data/<ID>/
    hidden: dict[str, np.ndarray]                     # one npz, encrypted, under sandbox/hidden/<ID>/hidden.npz.enc
    sources_sha256: dict[str, str] = field(default_factory=dict)
    models: dict[str, Path] = field(default_factory=dict)   # name -> source directory, copied to sandbox/models/<ID>/<name>
    judge_models: dict[str, Path] = field(default_factory=dict)  # evaluator-only models (never in a sandbox): sandbox/judge_models/<ID>/<name>
    raw_files: dict[str, Path] = field(default_factory=dict)  # extra public files copied verbatim (name -> source path)
    notes: dict = field(default_factory=dict)


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def check(p: Path, md5: str | None = None, sha256: str | None = None) -> str:
    """Verify a raw file against a published checksum; returns its sha256 (recorded in the lock file)."""
    if not p.is_file():
        raise SystemExit(f"raw file missing: {p}")
    if md5 and md5_file(p) != md5:
        raise SystemExit(f"{p.name}: md5 mismatch (expected {md5})")
    s = sha256_file(p)
    if sha256 and s != sha256:
        raise SystemExit(f"{p.name}: sha256 mismatch (expected {sha256})")
    return s


def fetch(url: str, dest: Path, md5: str | None = None, sha256: str | None = None) -> Path:
    """Download url to dest unless it is already there; then verify. For the operator's networked machine."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(dest)
    check(dest, md5, sha256)
    return dest


def dev_indices(n: int, frac: float, seed: int, y: np.ndarray | None = None) -> np.ndarray:
    """A fixed dev split of the public training data (stratified when labels are given)."""
    rng = np.random.default_rng(seed)
    if y is None:
        return np.sort(rng.choice(n, size=max(1, int(round(n * frac))), replace=False)).astype(np.int64)
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, int(round(len(idx) * frac)))
        out.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


def text_pack(xs) -> tuple[np.ndarray, np.ndarray]:
    """Many documents as (UTF-8 bytes uint8, offsets int64 of length N + 1); see cases/TC-FT-001/harness/textpack.py.
    Used instead of text_array when documents are long, where a fixed-width <U array would be mostly padding."""
    enc = [str(x).encode("utf-8") for x in xs]
    off = np.zeros(len(enc) + 1, np.int64)
    np.cumsum([len(b) for b in enc], out=off[1:])
    return np.frombuffer(b"".join(enc), np.uint8).copy(), off


def text_array(xs) -> np.ndarray:
    """Unicode string array (dtype <U...), storable in npz without pickle."""
    return np.asarray([str(x) for x in xs], dtype=np.str_)
