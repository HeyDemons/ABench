#!/usr/bin/env python3
"""Freeze the data of one ABench standard case (operator only, on a networked machine with the raw files).

  python3 tools/prepare/run.py --case TC-FT-001 --list-sources              # what to download, from where, licenses
  python3 tools/prepare/run.py --case TC-FT-001 --raw-dir RAW --key-file ~/.abench/hidden.key
  ABENCH_ALLOW_SYNTHETIC=1 python3 tools/prepare/run.py --case TC-FT-001 --synthetic 2000 --key-file K
      synthetic data is for plumbing tests only; the lock file is marked synthetic and the case stays awaiting_data
  python3 tools/prepare/run.py --case TC-FT-001 --activate                  # after the baseline calibration run

Writes sandbox/data/<ID>/*.npz (public), sandbox/hidden/<ID>/hidden.npz.enc (encrypted), sandbox/models/<ID>/...
(frozen pretrained files, if any) and cases/<ID>/case/data_lock.yaml, then regenerates case/oj.yaml.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE.parent))
import hidden_store  # noqa: E402
from file_hashes import git_blob_sha1 as _git_blob_sha1  # noqa: E402
from prepare.common import Frozen, md5_file, sha256_file, tree_hash  # noqa: E402


HF_RESOLVE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)$")
HF_TREE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/tree/([0-9a-f]{40})/?$")


def _get_json(url: str):
    import json
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def _hf_expected(dataset: bool, repo: str, rev: str, path: str, *, directories: dict | None = None) -> dict:
    """Hashes Hugging Face records for one file at a pinned revision: LFS sha256, or the git blob sha1."""
    import posixpath
    d = posixpath.dirname(path)
    api = f"https://huggingface.co/api/{'datasets/' if dataset else 'models/'}{repo}/tree/{rev}" + (f"/{d}" if d else "")
    directories = directories if directories is not None else {}
    if api not in directories:
        entries = {}
        for entry in _get_json(api):
            entries.setdefault(entry.get("path"), entry)
        directories[api] = entries
    entry = directories[api].get(path)
    if entry is not None:
        return {"sha256": (entry.get("lfs") or {}).get("oid"), "git_sha1": entry.get("oid"), "size": entry.get("size")}
    raise SystemExit(f"{path} not found at {repo}@{rev}")


def _locate(raw: Path, name: str) -> Path | None:
    p = raw / name
    if p.is_file():
        return p
    hits = [x for x in raw.rglob(Path(name).name) if x.is_file()]
    return hits[0] if len(hits) == 1 else None


def verify_sources(mod, raw: Path, allow_unverified: bool) -> list[dict]:
    """Check every raw file against what its publisher records: Hugging Face hashes at the pinned revision, or a
    published checksum file (checksum_url). md5/sha256 given in SOURCES are checked by the case module itself."""
    import urllib.request
    out = []
    # A pinned directory listing serves all files in it. Keep it local to this verification so a later
    # invocation still contacts the publisher and independently verifies every local file's bytes.
    directories = {}
    for s in getattr(mod, "SOURCES", []):
        url, name = s.get("url") or "", s.get("name") or ""
        targets = []
        from prepare.sources import files as source_files
        for resolved_url, raw_path, _ in source_files(s):
            m = HF_RESOLVE.match(resolved_url)
            if m:
                targets.append((raw_path, bool(m.group(1)), m.group(2), m.group(3), m.group(4)))
        for local_name, ds, repo, rev, path in targets:
            p = _locate(raw, local_name)
            row = {"file": local_name, "source": f"{repo}@{rev[:12]}/{path}"}
            if p is None:
                row["status"] = "raw file not found"
            else:
                try:
                    exp = _hf_expected(ds, repo, rev, path, directories=directories)
                    if exp.get("sha256"):
                        ok = sha256_file(p) == exp["sha256"]
                        row.update(check="lfs sha256", ok=ok)
                    else:
                        ok = _git_blob_sha1(p) == exp["git_sha1"]
                        row.update(check="git blob sha1", ok=ok)
                    row["status"] = "verified" if ok else "MISMATCH"
                except (OSError, ValueError) as e:
                    row["status"] = f"unverified ({type(e).__name__}: {e})"
            out.append(row)
        if s.get("checksum_url"):
            p = _locate(raw, s.get('raw_path') or name)
            row = {"file": name, "source": s["checksum_url"]}
            try:
                with urllib.request.urlopen(s["checksum_url"], timeout=60) as r:
                    want = re.findall(r"\b[0-9a-f]{32}\b", r.read().decode())[0]
                got = md5_file(p) if p else None
                row.update(check="published md5", ok=got == want, status="verified" if got == want else "MISMATCH")
            except (OSError, IndexError, ValueError) as e:
                row["status"] = f"unverified ({type(e).__name__}: {e})"
            out.append(row)
    for r in out:
        print(f"  {r['status']:>10}  {r['file']}  <- {r['source']}")
    bad = [r for r in out if r["status"] == "MISMATCH" or r["status"] == "raw file not found"]
    if bad:
        raise SystemExit(f"source verification failed for {[r['file'] for r in bad]}")
    if not allow_unverified and any(r["status"].startswith("unverified") for r in out):
        raise SystemExit("some sources could not be verified (offline?); rerun with network, or pass --allow-unverified "
                         "to record them as unverified in the lock file")
    return out


def escrow_seed(case: str, env: str, key_file: Path, hidden_sha: str) -> str:
    """Store the secret generator seed next to the hidden-set key (0600), so the held-out set can be regenerated."""
    import json
    import time
    d = key_file.expanduser().resolve().parent / "seeds"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    p = d / f"{case}.json"
    body = json.dumps({"case": case, "env": env, "seed": os.environ[env], "hidden_sha256_plain": hidden_sha,
                       "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # never readable by others, even briefly
    with os.fdopen(fd, "w") as f:
        f.write(body)
    p.chmod(0o600)
    return str(p)


def module_for(case: str):
    return importlib.import_module("prepare." + case.lower().replace("-", "_"))


def _write_frozen(case: str, fz: Frozen, key_file: Path, synthetic: bool, storage: Path) -> dict:
    pub_dir = storage / "sandbox" / "data" / case
    hid_path = storage / "sandbox" / "hidden" / case / "hidden.npz.enc"
    mod_dir = storage / "sandbox" / "models" / case
    for d in (pub_dir, hid_path.parent):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    lock: dict = {"case_id": case, "synthetic": bool(synthetic), "activation": "awaiting_data",
                  "public_dir": str(pub_dir.relative_to(storage)), "public": []}
    for name, arrays in fz.public.items():
        p = pub_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('wb') as output:
            np.savez(output, **arrays)
        first = next(iter(arrays.values()))
        lock["public"].append({"path": str(p.relative_to(storage)), "sha256": sha256_file(p),
                               "keys": sorted(arrays), "rows": int(len(first))})
    for name, src in fz.raw_files.items():
        p = pub_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
        lock["public"].append({"path": str(p.relative_to(storage)), "sha256": sha256_file(p)})
    with tempfile.TemporaryFile() as plain, os.fdopen(os.open(hid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        np.savez(plain, **fz.hidden)
        plain.seek(0)
        identity = hidden_store.encrypt_stream(hidden_store.read_key(key_file), plain, output)
    first = next(iter(fz.hidden.values()))
    lock["hidden"] = {"path": str(hid_path.relative_to(storage)), "sha256_plain": identity['sha256_plain'],
                      "keys": sorted(fz.hidden), "rows": int(len(first))}
    if fz.models:
        if mod_dir.exists():
            shutil.rmtree(mod_dir)
        lock["models_dir"] = str(mod_dir.relative_to(storage))
        lock["models"] = []
        for name, src in fz.models.items():
            dst = mod_dir / name
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", ".cache", "*.lock", "*.metadata"))
            lock["models"].append({"name": name, "path": str(dst.relative_to(storage)), "tree_sha256": tree_hash(dst)})
    if fz.judge_models:                                  # evaluator-only models, e.g. a reward model
        jdir = storage / "sandbox" / "judge_models" / case
        if jdir.exists():
            shutil.rmtree(jdir)
        lock["judge_models_dir"] = str(jdir.relative_to(storage))
        lock["judge_models"] = []
        for name, src in fz.judge_models.items():
            dst = jdir / name
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", ".cache", "*.lock", "*.metadata"))
            lock["judge_models"].append({"name": name, "path": str(dst.relative_to(storage)), "tree_sha256": tree_hash(dst)})
    lock["sources_sha256"] = fz.sources_sha256
    if fz.notes:
        lock["notes"] = fz.notes
    return lock


def freeze(case: str, fz: Frozen, key_file: Path, synthetic: bool, *, metadata=None, regenerate=False,
           seed_env=None) -> dict:
    """Build privately, verify, then publish with rollback and a lock written after the data directories."""
    from prepare.artifacts import Publication, atomic_text, verify_frozen
    with Publication(ROOT, case) as publication, ExitStack() as cleanup:
        lock = _write_frozen(case, fz, key_file, synthetic, publication.staging)
        verify_frozen(publication.staging, lock, key_file)
        if metadata:
            lock.update(metadata)
        lock_path = ROOT / 'cases' / case / 'case/data_lock.yaml'
        staged_lock = publication.staging / 'data_lock.yaml'
        atomic_text(staged_lock, '# written by tools/prepare/run.py; do not edit by hand\n'
                    + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), 0o644)
        mappings = [(lock_path, staged_lock)]
        if regenerate:
            mappings.append((lock_path.parent / 'oj.yaml', None))
        if seed_env:
            import json
            import time
            import uuid
            target = key_file.expanduser().resolve().parent / 'seeds' / (case + '.json')
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            nonce = uuid.uuid4().hex
            staged_seed = target.with_name('.' + target.name + '-' + nonce + '.new')
            body = {'case': case, 'env': seed_env, 'seed': os.environ[seed_env],
                    'hidden_sha256_plain': lock['hidden']['sha256_plain'],
                    'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            atomic_text(staged_seed, json.dumps(body, indent=1), 0o600)
            cleanup.callback(staged_seed.unlink, missing_ok=True)
            mappings.append((target, staged_seed, target.with_name('.' + target.name + '-' + nonce + '.old')))
            lock['secret_seed'] = {'env': seed_env, 'escrowed': True,
                                   'note': 'stored next to the hidden-set key (seeds/<case>.json, 0600), not in the repository'}
            atomic_text(staged_lock, '# written by tools/prepare/run.py; do not edit by hand\n'
                        + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), 0o644)
        for kind in ('data', 'hidden', 'models', 'judge_models'):
            staged = publication.staging / 'sandbox' / kind / case
            if staged.exists():
                mappings.append((ROOT / 'sandbox' / kind / case, staged))
        generate = (lambda: subprocess.run([sys.executable, str(ROOT / 'tools/build_standard_cases.py'),
                                            '--case', case], check=True)) if regenerate else None
        publication.publish(mappings, before_commit=generate, last=(lock_path,))
        return lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--raw-dir")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--key-file")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="record sources that could not be checked against their publisher (no network) instead of failing")
    a = ap.parse_args()
    mod = module_for(a.case)
    lock_path = ROOT / "cases" / a.case / "case" / "data_lock.yaml"
    if a.list_sources:
        print(yaml.safe_dump(mod.SOURCES, allow_unicode=True, sort_keys=False, width=110))
        return 0
    if a.activate:
        from prepare.artifacts import Publication, atomic_text
        with Publication(ROOT, a.case) as publication:
            lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
            if lock.get("synthetic"):
                raise SystemExit("refusing to activate a case whose data is synthetic")
            lock["activation"] = "active"
            staged = publication.staging / 'data_lock.yaml'
            atomic_text(staged, yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), 0o644)
            publication.publish([(lock_path, staged), (lock_path.parent / 'oj.yaml', None)],
                before_commit=lambda: subprocess.run([sys.executable, str(ROOT / "tools/build_standard_cases.py"),
                                                     "--case", a.case], check=True), last=(lock_path,))
    else:
        if not a.key_file:
            raise SystemExit("--key-file is required")
        if a.synthetic:
            if os.environ.get("ABENCH_ALLOW_SYNTHETIC") != "1":
                raise SystemExit("synthetic data requires ABENCH_ALLOW_SYNTHETIC=1 (plumbing tests only)")
            fz = mod.synthetic(a.synthetic)
        elif a.raw_dir:
            seed_env = getattr(mod, "SECRET_SEED_ENV", None)
            if seed_env and not os.environ.get(seed_env):
                raise SystemExit(f"set {seed_env} to a secret integer before freezing {a.case}")
            from prepare.sources import resolve_paths
            resolve_paths(Path(a.raw_dir), mod.SOURCES)
            print(f"{a.case}: verifying raw files against their publishers")
            verified = verify_sources(mod, Path(a.raw_dir), a.allow_unverified)
            fz = mod.build(Path(a.raw_dir))
        else:
            raise SystemExit("pass --raw-dir (real data) or --synthetic N (plumbing test)")
        metadata = {"source_verification": verified} if a.raw_dir and not a.synthetic else None
        lock = freeze(a.case, fz, Path(a.key_file), bool(a.synthetic), metadata=metadata, regenerate=True,
                      seed_env=seed_env if a.raw_dir and not a.synthetic else None)
    print(f"{a.case}: lock {lock_path.relative_to(ROOT)} activation={lock['activation']}"
          + ("  [SYNTHETIC]" if lock.get("synthetic") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
