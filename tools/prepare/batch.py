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
  calibrate  ./abench calibrate with the reference submission (not again for an active case); GPU cases need --gpus.
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
HF_TREE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/tree/([^/]+)/?$")
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


def downloads(source: dict) -> list[tuple[str, str]]:
    """(url, path below the case's raw directory) of every file one SOURCES entry names; [] when there is nothing
    to download (generators, installed packages, repositories without a file list)."""
    from prepare.sources import files
    return [(url, path) for url, path, _ in files(source)]


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
        from file_hashes import git_blob_sha1
        if git_blob_sha1(path) != source["git_blob_sha1"]:
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
    from prepare.artifacts import cached_download, immutable_source
    from prepare.sources import files as source_files
    done, fetched, manual = 0, 0, []
    for source in sources(case):
        files = source_files(source)
        if HF_TREE.match(str(source.get("url") or "")) and not source.get("files"):
            manual.append(f"{source.get('name')}: no file list; download the repository snapshot {source['url']} by hand")
        for url, rel, checks in files:
            dest = raw / rel
            try:
                if not dest.is_file():
                    if immutable_source(url, checks):
                        fetched += cached_download(url, dest, checks, fetch, lambda path: check(path, checks, True))
                    else:
                        fetch(url, dest)
                        fetched += 1
                check(dest, checks, True)
                done += 1
            except Manual as exc:
                manual.append(str(exc))
            except ValueError:
                dest.unlink(missing_ok=True)              # a corrupt or changed file is fetched again next time
                raise
    return {"files": done, "fetched": fetched, "manual": manual}


def frozen(case: str) -> bool:
    from prepare.artifacts import Publication
    from file_hashes import sha256_file, tree_hash
    with Publication(ROOT, case):
        lock = ROOT / "cases" / case / "case" / "data_lock.yaml"
        if not lock.is_file():
            return False
        data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
        hidden = ROOT / str((data.get('hidden') or {}).get('path') or '')
        if data.get('synthetic') or not hidden.is_file():
            return False
        try:
            return (all((ROOT / item['path']).is_file() and sha256_file(ROOT / item['path']) == item['sha256']
                        for item in data.get('public') or [])
                    and all((ROOT / item['path']).is_dir() and tree_hash(ROOT / item['path']) == item['tree_sha256']
                            for kind in ('models', 'judge_models') for item in data.get(kind) or []))
        except (KeyError, OSError):
            return False


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
    from prepare.sources import resolve_paths
    resolve_paths(raw, sources(case))
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


def active(case: str) -> bool:
    lock = ROOT / "cases" / case / "case" / "data_lock.yaml"
    return lock.is_file() and (yaml.safe_load(lock.read_text(encoding="utf-8")) or {}).get("activation") == "active"


def calibration(case: str, run_id: str) -> dict:
    from agents import run_dir
    path = run_dir(ROOT / "runs", "oracle", None, case, run_id) / "evaluation.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def needs_gpu(case: str, row: dict) -> bool:
    """Whether calibrating the case runs its judge on a GPU (sandbox/calibration/cases.yaml may validate on CPU)."""
    config = (yaml.safe_load((ROOT / "sandbox/calibration/cases.yaml").read_text()) or {}).get(case) or {}
    device = config.get("device") or (row["oj"].get("judge") or {}).get("device", "cpu")
    return bool(int(row["resources"].get("gpus") or 0)) and device != "cpu"


def calibrate(case: str, row: dict, key: Path, run_id: str, gpus: str | None, nonreference: bool) -> dict:
    if calibration(case, run_id).get("verdict") == "evaluated":
        return calibration(case, run_id)
    command = [sys.executable, str(ROOT / "tools/abench.py"), "calibrate", case, "--hidden-key-file", str(key), "--run-id", run_id]
    if needs_gpu(case, row):
        if not gpus:
            raise Manual("GPU case: pass --gpus to calibrate it")
        command += ["--gpus", gpus] + (["--allow-nonreference-gpu"] if nonreference else [])
    subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = calibration(case, run_id)
    if result.get("verdict") != "evaluated":
        raise Manual(f"calibration verdict {result.get('verdict') or 'missing'}: see runs/oracle/.../{case}/{run_id}")
    return result


def activate(case: str) -> str:
    from prepare.artifacts import Publication
    with Publication(ROOT, case) as publication:
        return _activate(case, publication)


def _activate(case, publication):
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
    from prepare.artifacts import atomic_text
    staged = publication.staging / 'data_lock.yaml'
    atomic_text(staged, header + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), 0o644)
    def generate():
        subprocess.run([sys.executable, str(ROOT / "tools/build_standard_cases.py"), "--case", case], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL)
    publication.publish([(lock_path, staged), (lock_path.parent / 'oj.yaml', None)],
                        before_commit=generate, last=(lock_path,))
    return "activated"


def prepare(rows: list[dict], raw_root: Path, key: Path, steps=STEPS, gpus: str | None = None,
            nonreference: bool = False, allow_unverified: bool = False, run_id: str = "ojcal-prepare",
            log=print, jobs: int = 1) -> list[dict]:
    """Run the steps for each case; a case stops at its first step that fails or needs the operator.

    Downloads and freezes run for `jobs` cases at a time. Calibrations then use one worker per GPU in `gpus`
    (comma-separated), alongside one CPU worker. Waiting GPU cases do not occupy the CPU worker. Results keep
    the order of rows."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    records = {row["id"]: {"case": row["id"], "category": row.get("category"), "steps": {}} for row in rows}

    def attempt(row, work):
        record = records[row["id"]]
        if record.get("status") in ("needs operator", "failed"):
            return
        try:
            result = work(row["id"], record)
            record["status"] = "done"
            return result
        except Manual as exc:
            record.update(status="needs operator", reason=str(exc))
            log(f"{row['id']}: needs operator: {exc}")
        except Exception as exc:                          # one broken case must not stop the others
            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}"[:500])
            log(f"{row['id']}: failed: {record['reason']}")

    def fetch_and_freeze(case, record):
        raw = raw_root / case
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

    devices = queue.Queue()
    names = list(dict.fromkeys(g for g in (gpus or "").split(",") if g))
    for name in names:
        devices.put(name)
    by_id = {row["id"]: row for row in rows}

    def judge_and_activate(case, record):
        row = by_id[case]
        if "calibrate" in steps:
            if active(case):                          # calibrated before it was activated
                record["steps"]["calibrate"] = "already active"
            elif names and needs_gpu(case, row):
                gpu = devices.get()                   # one GPU case per GPU at a time
                try:
                    result = calibrate(case, row, key, run_id, gpu, nonreference)
                finally:
                    devices.put(gpu)
                record["steps"]["calibrate"] = {"verdict": result.get("verdict"), "run_id": run_id, "gpu": gpu}
            else:
                result = calibrate(case, row, key, run_id, None, nonreference)
                record["steps"]["calibrate"] = {"verdict": result.get("verdict"), "run_id": run_id}
            log(f"{case} calibrate: {record['steps']['calibrate']}")
        if "activate" in steps:
            record["steps"]["activate"] = activate(case)
            log(f"{case} activate: {record['steps']['activate']}")

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(lambda row: attempt(row, fetch_and_freeze), rows))
    if {"calibrate", "activate"} & set(steps):
        # Each GPU worker can always take a device; none wait for a lease while occupying the CPU lane.
        # Keep the phase boundary above: overlapping freezing with timed calibration can distort its results.
        with ThreadPoolExecutor(max_workers=1) as cpu_pool, \
                ThreadPoolExecutor(max_workers=max(1, len(names))) as gpu_pool:
            def choose_pool(case, record):
                use_gpu = ("calibrate" in steps and names and not active(case)
                           and needs_gpu(case, by_id[case]))
                return gpu_pool if use_gpu else cpu_pool
            pending = []
            for row in rows:
                executor = attempt(row, choose_pool)
                if executor is not None:
                    pending.append(executor.submit(attempt, row, judge_and_activate))
            for future in pending:
                future.result()
    for record in records.values():
        record.setdefault("status", "done")
    return [records[row["id"]] for row in rows]
