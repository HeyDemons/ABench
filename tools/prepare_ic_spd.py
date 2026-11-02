#!/usr/bin/env python3
"""IC-SPD-001 data preparation: public training set + encrypted hidden test set (run by the operator after downloading the raw files).

Raw files (downloaded by the operator on a networked machine into one directory):
  CIFAR-10   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz   (official, md5 c58f30108f718f92721af3b95e74349a)
             Mirror: https://huggingface.co/datasets/uoft-cs/cifar10
  CIFAR-10.1 https://github.com/modestyachts/CIFAR-10.1/raw/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/
             cifar10.1_v6_data.npy, cifar10.1_v6_labels.npy (2,000 images, class-balanced; Recht et al. 2018; MIT;
             pinned to commit d9982abb, sha256 checked below)

Output:
  sandbox/data/IC-SPD-001/train.npz          public: x uint8 (50000,32,32,3), y int64, dev_idx int64 (5000)
  sandbox/hidden/IC-SPD-001/test.npz.enc     hidden: x uint8 (12000,32,32,3), y, src (0=CIFAR-10 test, 1=CIFAR-10.1)
and write hashes and row counts back to cases/IC-SPD-001/case/oj.yaml.

Usage:
  python3 tools/prepare_ic_spd.py --raw-dir <download dir> --key-file ~/.abench/hidden.key
  ABENCH_ALLOW_SYNTHETIC=1 python3 tools/prepare_ic_spd.py --synthetic 3000 --key-file <key>
      Synthetic data is only for pipeline tests: oj.yaml is marked data.synthetic: true, and the judge refuses it unless ABENCH_ALLOW_SYNTHETIC=1 (never scorable).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hidden_store  # noqa: E402

CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
DEV_SEED, DEV_PER_CLASS = 20260701, 500


class _SafeUnpickler(pickle.Unpickler):
    """The python version of CIFAR-10 is a pickle; only numpy arrays and built-in containers may be reconstructed, and any other object is rejected."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("builtins", "dict"), ("builtins", "list")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _batch(tf: tarfile.TarFile, name: str) -> tuple[np.ndarray, np.ndarray]:
    member = tf.getmember(f"cifar-10-batches-py/{name}")
    d = _SafeUnpickler(tf.extractfile(member), encoding="bytes").load()
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(d[b"labels"], dtype=np.int64)


CIFAR101_SHA256 = {  # files at commit d9982abb0bfc4846b8d13a11e66b887d946205d0 of modestyachts/CIFAR-10.1, checked 2026-10-21
    "cifar10.1_v6_data.npy": "2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6",
    "cifar10.1_v6_labels.npy": "ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448",
}

def load_raw(raw: Path):
    tar = raw / "cifar-10-python.tar.gz"
    md5 = hashlib.md5(tar.read_bytes()).hexdigest()
    if md5 != CIFAR10_MD5:
        raise SystemExit(f"cifar-10-python.tar.gz md5 {md5} != {CIFAR10_MD5}")
    with tarfile.open(tar, "r:gz") as tf:
        parts = [_batch(tf, f"data_batch_{i}") for i in range(1, 6)]
        xte, yte = _batch(tf, "test_batch")
    xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    for f, want in CIFAR101_SHA256.items():
        got = hashlib.sha256((raw / f).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{f} sha256 {got} != {want} (CIFAR-10.1 commit d9982abb)")
    x101 = np.load(raw / "cifar10.1_v6_data.npy", allow_pickle=False).astype(np.uint8)
    y101 = np.load(raw / "cifar10.1_v6_labels.npy", allow_pickle=False).astype(np.int64)
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3) and x101.shape == (2000, 32, 32, 3)
    sources = {"cifar-10-python.tar.gz": hashlib.sha256(tar.read_bytes()).hexdigest(),
               "cifar10.1_v6_data.npy": hashlib.sha256((raw / "cifar10.1_v6_data.npy").read_bytes()).hexdigest(),
               "cifar10.1_v6_labels.npy": hashlib.sha256((raw / "cifar10.1_v6_labels.npy").read_bytes()).hexdigest()}
    return xtr, ytr, xte, yte, x101, y101, sources


def synthetic(n: int):
    """Learnable synthetic 32×32 images: one color/stripe pattern per class plus noise. For pipeline tests only."""
    rng = np.random.default_rng(0)
    proto = rng.integers(0, 256, size=(10, 32, 32, 3)).astype(np.float32)

    def make(m, shift=0.0):
        y = np.arange(m) % 10
        rng.shuffle(y)
        x = proto[y] * 0.55 + rng.normal(128 + shift, 60, size=(m, 32, 32, 3))
        return np.clip(x, 0, 255).astype(np.uint8), y.astype(np.int64)
    xtr, ytr = make(n)
    xte, yte = make(max(n // 5, 200))
    x101, y101 = make(max(n // 25, 100), shift=12.0)          # slight distribution shift
    return xtr, ytr, xte, yte, x101, y101, {"synthetic": f"seed0-n{n}"}


def dev_split(y: np.ndarray, per_class: int) -> np.ndarray:
    rng = np.random.default_rng(DEV_SEED)
    idx = [rng.choice(np.flatnonzero(y == c), size=min(per_class, int((y == c).sum()) // 5), replace=False)
           for c in range(10)]
    return np.sort(np.concatenate(idx)).astype(np.int64)


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--key-file", required=True)
    a = ap.parse_args()
    if a.synthetic:
        if os.environ.get("ABENCH_ALLOW_SYNTHETIC") != "1":
            raise SystemExit("synthetic data requires ABENCH_ALLOW_SYNTHETIC=1 (plumbing tests only)")
        xtr, ytr, xte, yte, x101, y101, sources = synthetic(a.synthetic)
    elif a.raw_dir:
        xtr, ytr, xte, yte, x101, y101, sources = load_raw(Path(a.raw_dir))
    else:
        raise SystemExit("pass --raw-dir (real data) or --synthetic N (plumbing test)")
    dev = dev_split(ytr, DEV_PER_CLASS)
    from prepare.artifacts import Publication, atomic_text, verify_frozen
    from file_hashes import sha256_file
    import tempfile
    case = "IC-SPD-001"
    oj_path = ROOT / "cases" / case / "case/oj.yaml"
    with Publication(ROOT, case) as publication:
        pub = publication.staging / "sandbox/data/IC-SPD-001/train.npz"
        hid = publication.staging / "sandbox/hidden/IC-SPD-001/test.npz.enc"
        pub.parent.mkdir(parents=True)
        hid.parent.mkdir(parents=True)
        with pub.open("wb") as output:
            np.savez(output, x=xtr, y=ytr, dev_idx=dev)
        xh = np.concatenate([xte, x101])
        with tempfile.TemporaryFile() as plain, os.fdopen(os.open(hid, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
            np.savez(plain, x=xh, y=np.concatenate([yte, y101]),
                     src=np.concatenate([np.zeros(len(yte), np.int64), np.ones(len(y101), np.int64)]))
            plain.seek(0)
            identity = hidden_store.encrypt_stream(hidden_store.read_key(Path(a.key_file)), plain, output)
        oj = yaml.safe_load(oj_path.read_text(encoding="utf-8"))
        oj["data"]["public"][0].update(path=str(pub.relative_to(publication.staging)), sha256=sha256_file(pub),
                                       rows=int(len(ytr)), dev_rows=int(len(dev)))
        oj["data"]["hidden"].update(path=str(hid.relative_to(publication.staging)), sha256_plain=identity["sha256_plain"],
                                    rows=int(len(xh)), rows_by_src={"cifar10": int(len(yte)), "cifar10_1": int(len(y101))})
        oj["data"]["sources_sha256"] = sources
        if a.synthetic:
            oj["data"]["synthetic"] = True
        else:
            oj["data"].pop("synthetic", None)
        lock = {"case_id": case, "activation": oj["judge"].get("activation"),
                "synthetic": bool(a.synthetic), **{key: oj["data"][key] for key in
                ("public_dir", "public", "hidden", "sources_sha256")}}
        verify_frozen(publication.staging, lock, Path(a.key_file))
        lock_path = oj_path.parent / "data_lock.yaml"
        staged_lock = publication.staging / "data_lock.yaml"
        atomic_text(staged_lock, yaml.safe_dump(lock, sort_keys=False), 0o644)
        def generate():
            import build_standard_cases
            build_standard_cases.build_handcrafted(ROOT / "spec/handcrafted/IC-SPD-001.yaml", check=False)
        publication.publish([(lock_path, staged_lock), (oj_path, None),
                             (ROOT / "sandbox/data" / case, pub.parent),
                             (ROOT / "sandbox/hidden" / case, hid.parent)],
                            before_commit=generate, last=(lock_path,))
        pub, hid = ROOT / pub.relative_to(publication.staging), ROOT / hid.relative_to(publication.staging)
    print(f"public {pub} {len(ytr)} rows (dev {len(dev)}); hidden {hid} {len(xh)} rows (encrypted)"
          + ("  [SYNTHETIC]" if a.synthetic else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
