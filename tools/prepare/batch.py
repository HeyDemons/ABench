#!/usr/bin/env python3
"""Prepare many cases in one go (operator, on a networked machine): download -> freeze -> calibrate -> activate.

  ./abench prepare --category data_science --raw-root ~/abench-raw [--gpus 0 --allow-nonreference-gpu]
  ./abench prepare --cases DS-GBM-001,RS-SEQ-001 --raw-root ~/abench-raw --steps download,freeze

Per case, in order, each step skipped when already done:
  download   every source of the case module's SOURCES with a URL into RAW_ROOT/<case>/: plain files, Hugging Face
             files (resolve URLs, and tree URLs file by file), GitHub blob links as raw files. Checksums given in
             SOURCES (md5, sha256, git blob sha1, md5 of the first MiB, size) are checked; a URL that serves a web
             page instead of a file is reported as a manual download.
  freeze     tools/prepare/run.py with a fresh secret seed when the case has one (escrowed next to the key by
             run.py). When the case module asks for a raw path that is missing, a downloaded file of the same name
             is linked there and the freeze is retried.
  calibrate  ./abench calibrate with the reference submission; GPU cases need --gpus.
  activate   only after the calibration ends `evaluated`, and never while verify.csv says the case's baseline still
             has to be tuned (baseline_tuned: pending).
Cases that need something only the operator can do are listed with the reason; the others continue.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE.parent))

STEPS = ("download", "freeze", "calibrate", "activate")
HF_RESOLVE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
HF_TREE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/tree/([^/]+)/?$")
GITHUB_BLOB = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$")
MISSING = (re.compile(r"No such file or directory: '([^']+)'"), re.compile(r"raw file missing: (\S+)"),
           re.compile(r"source verification failed for (\[.*\])"))
# IC-SPD-001 is prepared by its own script (tools/prepare_ic_spd.py), which checks these files itself.
IC_SPD_SOURCES = [
    {"name": "cifar-10-python.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
     "md5": "c58f30108f718f92721af3b95e74349a"},
    *({"name": f, "url": "https://github.com/modestyachts/CIFAR-10.1/raw/d9982abb0bfc4846b8d13a11e66b887d946205d0/"
       f"datasets/{f}"} for f in ("cifar10.1_v6_data.npy", "cifar10.1_v6_labels.npy"))]


class Manual(Exception):
    """A step only the operator can finish (a file to fetch by hand, a baseline to tune, a GPU to give)."""


def module_for(case: str):
    from prepare.run import module_for as find
    return find(case)


def sources(case: str) -> list[dict]:
    return IC_SPD_SOURCES if case == "IC-SPD-001" else list(getattr(module_for(case), "SOURCES", []) or [])


def _relative(text: str) -> str | None:
    """A source name usable as a path below the case's raw directory, or None."""
    text = text.strip().rstrip("/")
    if not text or " " in text or text.startswith(("/", "~")) or ".." in Path(text).parts:
        return None
    return text


def downloads(source: dict) -> list[tuple[str, str]]:
    """(url, path below the case's raw directory) of every file one SOURCES entry names; [] when there is nothing
    to download (generators, installed packages, repositories without a file list)."""
    url = str(source.get("url") or "")
    if not url.startswith("http"):
        return []
    note = re.search(r"\bRAW/(\S+)", str(source.get("note") or ""))
    tree = HF_TREE.match(url)
    if tree:
        if not source.get("files"):
            return []
        base = (note.group(1).rstrip("/") if note else None) or _relative(str(source.get("name") or "")) or tree.group(2).split("/")[1]
        prefix = f"https://huggingface.co/{tree.group(1) or ''}{tree.group(2)}/resolve/{tree.group(3)}/"
        return [(prefix + f, f"{base}/{f}") for f in source["files"]]
    blob = GITHUB_BLOB.match(url)
    if blob:
        url = f"https://raw.githubusercontent.com/{blob.group(1)}/{blob.group(2)}/{blob.group(3)}"
    elif re.fullmatch(r"https://github\.com/[^/]+/[^/]+/?(\.git)?", url):
        return []                                  # a repository, e.g. a package installed in the images
    name = _relative(str(source.get("name") or ""))
    if note and not note.group(1).endswith("/"):
        path = note.group(1)
    elif note:
        path = note.group(1) + (name or Path(url.split("?")[0]).name)
    elif source.get("path") and name:
        path = f"{source['path'].strip('/')}/{name}"
    else:
        path = name or Path(url.split("?")[0]).name
    return [(url, path)]


def _md5(path: Path, limit: int | None = None) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(limit) if limit else b"")
        if limit is None:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def check(path: Path, source: dict, single: bool) -> None:
    """The checksums SOURCES gives for this file (only when the entry names a single file)."""
    from prepare.common import sha256_file
    if not single:
        return
    if source.get("md5") and _md5(path) != source["md5"]:
        raise ValueError(f"{path.name}: md5 mismatch")
    if source.get("sha256") and sha256_file(path) != source["sha256"]:
        raise ValueError(f"{path.name}: sha256 mismatch")
    if source.get("md5_first_mib") and _md5(path, 1 << 20) != source["md5_first_mib"]:
        raise ValueError(f"{path.name}: md5 of the first MiB mismatch")
    if str(source.get("size") or "").isdigit() and path.stat().st_size != int(source["size"]):
        raise ValueError(f"{path.name}: size {path.stat().st_size} != {source['size']}")
    if source.get("git_blob_sha1"):
        data = path.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != source["git_blob_sha1"]:
            raise ValueError(f"{path.name}: git blob sha1 mismatch")


def fetch(url: str, dest: Path) -> None:
    headers = {"User-Agent": "abench-prepare"}
    if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        if "text/html" in (r.headers.get("Content-Type") or "") and dest.suffix not in (".html", ".htm"):
            raise Manual(f"{url} serves a web page, not the file; download {dest.name} by hand into {dest.parent}")
        with open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)


def download(case: str, raw: Path) -> dict:
    """Fetch what is missing; returns counts. Files only the operator can fetch (web pages, repositories without a
    file list) are listed, not fatal: the freeze reports them if the case needs them."""
    done, fetched, manual = 0, 0, []
    for source in sources(case):
        files = downloads(source)
        if HF_TREE.match(str(source.get("url") or "")) and not source.get("files"):
            manual.append(f"{source.get('name')}: no file list; download the repository snapshot {source['url']} by hand")
        for url, rel in files:
            dest = raw / rel
            try:
                if not dest.is_file():
                    fetch(url, dest)
                    fetched += 1
                check(dest, source, len(files) == 1)
                done += 1
            except Manual as exc:
                manual.append(str(exc))
            except ValueError:
                dest.unlink(missing_ok=True)              # a corrupt or changed file is fetched again next time
                raise
    return {"files": done, "fetched": fetched, "manual": manual}


def frozen(case: str) -> bool:
    lock = ROOT / "cases" / case / "case" / "data_lock.yaml"
    if not lock.is_file():
        return False
    data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
    paths = [p.get("path") for p in data.get("public") or []] + [(data.get("hidden") or {}).get("path")]
    return not data.get("synthetic") and bool(data.get("hidden")) and all(p and (ROOT / p).is_file() for p in paths)


def _link_missing(raw: Path, wanted: str) -> bool:
    """Link a downloaded file (or directory) of the same name to the path the case module asked for."""
    target = Path(wanted)
    if not target.is_absolute():
        target = raw / wanted
    try:
        target.resolve().relative_to(raw.resolve())
    except ValueError:
        return False
    if target.exists():
        return False
    hits = [p for p in raw.rglob(target.name) if p != target and not p.name.endswith(".part")]
    if len(hits) != 1:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(hits[0].resolve(), target_is_directory=hits[0].is_dir())
    return True


def missing_paths(text: str) -> list[str]:
    """Raw paths a failed freeze asked for: open() errors, common.check's "raw file missing" and run.py's source
    verification list."""
    wanted = []
    for pattern in MISSING:
        for match in pattern.finditer(text):
            value = match.group(1)
            wanted += [str(v) for v in yaml.safe_load(value)] if value.startswith("[") else [value]
    return wanted


def freeze(case: str, raw: Path, key: Path, allow_unverified: bool, attempts: int = 20) -> None:
    env = dict(os.environ)
    if case == "IC-SPD-001":
        command = [sys.executable, str(ROOT / "tools/prepare_ic_spd.py"), "--raw-dir", str(raw), "--key-file", str(key)]
    else:
        command = [sys.executable, str(HERE / "run.py"), "--case", case, "--raw-dir", str(raw), "--key-file", str(key)]
        command += ["--allow-unverified"] if allow_unverified else []
        seed_env = getattr(module_for(case), "SECRET_SEED_ENV", None)
        if seed_env and not env.get(seed_env):
            env[seed_env] = str(secrets.randbelow(2**31 - 1) + 1)     # escrowed by run.py next to the key
    for _ in range(attempts):
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            return
        text = result.stdout + result.stderr
        if not any([_link_missing(raw, w) for w in missing_paths(text)]):
            raise Manual(f"freeze failed: {text.strip().splitlines()[-1][:400] if text.strip() else result.returncode}")
    raise Manual("freeze failed: too many missing raw files")


def calibration(case: str, run_id: str) -> dict:
    from agents import run_dir
    path = run_dir(ROOT / "runs", "oracle", None, case, run_id) / "evaluation.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def calibrate(case: str, row: dict, key: Path, run_id: str, gpus: str | None, nonreference: bool) -> dict:
    if calibration(case, run_id).get("verdict") == "evaluated":
        return calibration(case, run_id)
    config = (yaml.safe_load((ROOT / "sandbox/calibration/cases.yaml").read_text()) or {}).get(case) or {}
    device = config.get("device") or (row["oj"].get("judge") or {}).get("device", "cpu")
    command = [sys.executable, str(ROOT / "tools/abench.py"), "calibrate", case, "--hidden-key-file", str(key), "--run-id", run_id]
    if int(row["resources"].get("gpus") or 0) and device != "cpu":
        if not gpus:
            raise Manual("GPU case: pass --gpus to calibrate it")
        command += ["--gpus", gpus] + (["--allow-nonreference-gpu"] if nonreference else [])
    subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = calibration(case, run_id)
    if result.get("verdict") != "evaluated":
        raise Manual(f"calibration verdict {result.get('verdict') or 'missing'}: see runs/oracle/.../{case}/{run_id}")
    return result


def activate(case: str) -> str:
    lock_path = ROOT / "cases" / case / "case" / "data_lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if lock.get("activation") == "active":
        return "already active"
    with (ROOT / "verify.csv").open(newline="") as f:
        registry = {r["case_id"]: r for r in csv.DictReader(f)}
    if registry.get(case, {}).get("baseline_tuned") == "pending":
        raise Manual("baseline not tuned yet (verify.csv baseline_tuned: pending); see next_step there")
    if lock.get("synthetic"):
        raise Manual("data is synthetic")
    lock["activation"] = "active"
    header = "# written by tools/prepare/run.py; do not edit by hand\n" if case != "IC-SPD-001" else ""
    lock_path.write_text(header + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "tools/build_standard_cases.py"), "--case", case], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return "activated"


def prepare(rows: list[dict], raw_root: Path, key: Path, steps=STEPS, gpus: str | None = None,
            nonreference: bool = False, allow_unverified: bool = False, run_id: str = "ojcal-prepare",
            log=print) -> list[dict]:
    """Run the steps for each case; a case stops at its first step that fails or needs the operator."""
    results = []
    for row in rows:
        case, raw = row["id"], raw_root / row["id"]
        record = {"case": case, "category": row.get("category"), "steps": {}}
        try:
            if "download" in steps:
                got = download(case, raw)
                record["steps"]["download"] = got
                log(f"{case} download: {got['files']} files ({got['fetched']} new)"
                    + (f"; manual: {'; '.join(got['manual'])}" if got["manual"] else ""))
            if "freeze" in steps:
                if frozen(case):
                    record["steps"]["freeze"] = "already frozen"
                else:
                    freeze(case, raw, key, allow_unverified)
                    record["steps"]["freeze"] = "frozen"
                log(f"{case} freeze: {record['steps']['freeze']}")
            if "calibrate" in steps:
                result = calibrate(case, row, key, run_id, gpus, nonreference)
                record["steps"]["calibrate"] = {"verdict": result.get("verdict"), "run_id": run_id}
                log(f"{case} calibrate: {result.get('verdict')}")
            if "activate" in steps:
                record["steps"]["activate"] = activate(case)
                log(f"{case} activate: {record['steps']['activate']}")
            record["status"] = "done"
        except Manual as exc:
            record.update(status="needs operator", reason=str(exc))
            log(f"{case}: needs operator: {exc}")
        except Exception as exc:                          # one broken case must not stop the others
            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}"[:500])
            log(f"{case}: failed: {record['reason']}")
        results.append(record)
    return results
