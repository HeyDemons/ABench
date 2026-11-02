"""Imagenette (fastai, Apache-2.0): 10 easily classified ImageNet classes, the 160-pixel release (shortest side 160)
with the official 70/30 split of December 2019: 9,469 training and 3,925 validation images. Shared by IC-RES-002 and
V-BB-006. Images are centre-cropped to 160 x 160 and stored as uint8; the training images are public, the validation
images and their labels are the held-out set.

The archive is checked as fastai checks it (fastai/data/download_checks.py at commit 46231bdc, via fastdownload): its
size and the md5 of its first MiB.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SIZE, MD5_FIRST_MIB = 99003388, "c475f8f7617a200ba35b9facc48443c3"
SOURCES = [{"name": "imagenette2-160.tgz", "url": URL, "size": SIZE, "md5_first_mib": MD5_FIRST_MIB,
            "license": "Apache-2.0 (github.com/fastai/imagenette LICENSE); images from ImageNet"}]
CLASSES = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042", "n03425413",
           "n03445777", "n03888257"]
N_TRAIN, N_VAL, SIDE = 9469, 3925, 160


def check_archive(p: Path) -> str:
    if p.stat().st_size != SIZE:
        raise SystemExit(f"{p.name}: size {p.stat().st_size}, expected {SIZE}")
    with open(p, "rb") as f:
        if hashlib.md5(f.read(2 ** 20)).hexdigest() != MD5_FIRST_MIB:
            raise SystemExit(f"{p.name}: md5 of the first MiB does not match fastai's")
    return check(p)


def _square(img) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    if s != SIDE:
        img = img.resize((round(w * SIDE / s), round(h * SIDE / s)))
        w, h = img.size
    left, top = (w - SIDE) // 2, (h - SIDE) // 2
    return np.asarray(img.crop((left, top, left + SIDE, top + SIDE)), np.uint8)


def load(raw: Path) -> tuple[dict, dict, dict]:
    from PIL import Image                     # operator environment only
    p = raw / "imagenette2-160.tgz"
    sums = {p.name: check_archive(p)}
    data = {"train": ([], []), "val": ([], [])}
    with tarfile.open(p, "r:gz") as tf:
        for m in sorted(tf.getmembers(), key=lambda m: m.name):
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 4 or not parts[3].lower().endswith(".jpeg"):
                continue
            split, wnid = parts[1], parts[2]
            img = Image.open(io.BytesIO(tf.extractfile(m).read()))
            data[split][0].append(_square(img))
            data[split][1].append(CLASSES.index(wnid))
    xtr, ytr = np.stack(data["train"][0]), np.asarray(data["train"][1], np.int64)
    xva, yva = np.stack(data["val"][0]), np.asarray(data["val"][1], np.int64)
    assert len(ytr) == N_TRAIN and len(yva) == N_VAL, (len(ytr), len(yva))
    assert np.bincount(ytr, minlength=10).min() > 800
    return {"x": xtr, "y": ytr}, {"x_test": xva, "y_test": yva}, sums


def frozen(raw: Path) -> Frozen:
    pub, hid, sums = load(raw)
    return Frozen(public={"train.npz": pub}, hidden=hid, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    """n images per class with a class-dependent colour, so that a classifier can learn something."""
    rng = np.random.default_rng(0)
    def make(k):
        y = np.repeat(np.arange(10), k)
        base = rng.integers(0, 255, (10, 3))
        x = (base[y][:, None, None, :] + rng.normal(0, 40, (len(y), SIDE, SIDE, 3))).clip(0, 255).astype(np.uint8)
        return x, y.astype(np.int64)
    xtr, ytr = make(n)
    xva, yva = make(max(2, n // 2))
    return Frozen(public={"train.npz": {"x": xtr, "y": ytr}}, hidden={"x_test": xva, "y_test": yva},
                  sources_sha256={"synthetic": f"seed0-n{n}"})
