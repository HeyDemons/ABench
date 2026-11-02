"""Allowlisted agent view, environment and artifact collection. Agent-visible paths are stable."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import site
import stat
import time
from pathlib import Path
from typing import Any
from leak_check import leaky, leak_allow
from validate_submission import SELF_REPORT
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools'
PAPER_TOOL = TOOLS / 'paper_search/abench_papers.py'
DEFAULT_PAPER_INDEX = Path.home() / '.abench/papers/current.sqlite'
BASE_ENV = ('PATH', 'LANG', 'LC_ALL', 'TERM', 'USER', 'LOGNAME', 'SHELL')
NETWORK_TEXT = {
    "egress": "- Network: online through the sandbox proxy (already set in HTTP_PROXY / HTTPS_PROXY). Papers, code,\n"
              "  documentation and package sites are reachable at any point of your work: look things up whenever\n"
              "  your experiments raise a question. Dataset and model hubs and this case's own data sources are\n"
              "  blocked; data and weights come only from $ABENCH_DATA_DIR (and $ABENCH_MODELS_DIR). Every\n"
              "  connection is logged with the run.",
    "none": "- Network: none in this run; use the offline paper search below.",
    "open": "- Network: unrestricted (unscored run).",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_readonly(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif p.is_dir():
            p.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(stat.S_IRWXU)
    for p in root.rglob("*"):
        try:
            p.chmod(stat.S_IRWXU if p.is_dir() else stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


def validate_hidden_key(raw: str) -> Path:
    """Validate the operator key before an agent process is started.

    The key is intentionally outside the repository and must not be readable by
    group/other users.  This is not a substitute for a container or separate
    user, but it prevents the two easy accidental leaks (a checked-in key and a
    world-readable key) from making it into a run.
    """
    supplied = Path(raw).expanduser()
    if supplied.is_symlink():
        raise SystemExit("hidden key must not be a symlink; pass the real key file")
    key = supplied.resolve()
    if not key.is_file():
        raise SystemExit("hidden key file not found")
    try:
        key.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit("hidden key must live outside the ABench repository")
    if key.stat().st_mode & 0o077:
        raise SystemExit("hidden key permissions are too broad; use chmod 600")
    return key


def view_rel(rel: str) -> str:
    """Path of an interface file inside interface/: relative to harness/ (keeps baselines/ and baseline/ subfolders)."""
    p = Path(rel)
    return p.relative_to("harness").as_posix() if p.parts and p.parts[0] == "harness" else p.name


def build_agent_view(case: str, oj: dict[str, Any], view: Path, *, copy_data: bool = True) -> dict[str, Any]:
    """Copy only allowlisted content; return the view manifest (paths and hashes)."""
    case_dir = ROOT / "cases" / case
    vis = oj.get("agent_visible", {})
    manifest: dict[str, Any] = {"files": {}, "public_sources": []}
    view.mkdir(parents=True)
    task_src = case_dir / vis.get("task", "case/agent/task.md")
    if not task_src.is_file():
        raise SystemExit(f"{case}: agent task file missing: {task_src.relative_to(ROOT)}")
    shutil.copy2(task_src, view / "task.md")
    for rel in vis.get("interface", []):
        src = case_dir / rel
        if not src.is_file():
            raise SystemExit(f"{case}: interface file missing: {rel}")
        dst = view / "interface" / view_rel(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Recheck the exact files copied into the view at runtime.  Static lint is
    # useful before a run, but the runner must not trust a stale lint result if
    # an operator edits a task/interface after validation.
    for rel in ["task.md"] + [f"interface/{view_rel(rel)}" for rel in vis.get("interface", [])]:
        p = view / rel
        hits = leaky(p.read_text(encoding="utf-8", errors="replace"), leak_allow(oj))
        if hits:
            raise SystemExit(f"agent view leaks evaluator content in {rel}: {hits}")
    tdir = view / "templates"
    tdir.mkdir()
    prof = (oj.get("submission") or {}).get("report_profile") or ("judgment_report_v2" if "-T7-" in case else "design_report_v2")
    profile = prof.rsplit("_v2", 1)[0]
    shutil.copy2(ROOT / "sandbox" / "templates" / f"{profile}.html", tdir / "report_template.html")
    shutil.copy2(ROOT / "sandbox" / "templates" / "references.yaml", tdir / "references_template.yaml")
    (view / "tools").mkdir()
    shutil.copy2(PAPER_TOOL, view / "tools" / "abench_papers.py")   # agent side of the offline paper search
    pub = oj.get("data", {}).get("public") or []
    from runtime.cache import file_stamp
    for item in pub:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit(f"public data path must stay inside the repository: {item['path']}")
        src = ROOT / item["path"]
        name = Path(item["path"]).name
        rel = 'data/' + name
        if name in ('.', '..') or rel in manifest['files']:
            raise SystemExit(f"invalid or duplicate public data basename: {name}")
        before = file_stamp(src)
        actual = sha256_file(src)
        if actual != item["sha256"] or file_stamp(src) != before:
            raise SystemExit(f"public data hash mismatch: {item['path']}")
        manifest['public_sources'].append({'name': name, 'source': str(src), 'sha256': actual,
                                           'verified_stat': before})
        manifest['files'][rel] = actual
        dst = view / "data" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_data:
            shutil.copy2(src, dst)
            if file_stamp(src) != before:
                raise SystemExit(f"public data changed while copying: {item['path']}")
        else:
            dst.touch()             # Prompt data listing only; Docker mounts the validated original files here.
    # defense: judge-side files must never appear in the view
    banned = re.compile(r"(eval_protocol|scoring|metric_registry|reference_pool|anti_gaming|oj\.yaml|hidden|\.enc$|\.key$)", re.I)
    for p in view.rglob("*"):
        if p.is_file():
            rel = p.relative_to(view).as_posix()
            if banned.search(rel):
                raise SystemExit(f"agent view would expose evaluator-side file: {rel}")
            if rel not in manifest['files']:
                manifest["files"][rel] = sha256_file(p)
    make_readonly(view)
    return manifest


def agent_env(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, str], list[str]]:
    env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
    env.update({
        "HOME": str(paths["home"]), "TMPDIR": str(paths["tmp"]), "PYTHONDONTWRITEBYTECODE": "1",
        "ABENCH_CASE_ID": args.case, "ABENCH_RUN_ID": args.run_id,
        "ABENCH_VIEW_DIR": str(paths["view"]), "ABENCH_TASK": str(paths["view"] / "task.md"),
        "ABENCH_DATA_DIR": str(paths["view"] / "data"), "ABENCH_WORKSPACE": str(paths["workspace"]),
        "ABENCH_SUBMISSION_DIR": str(paths["submission"]), "ABENCH_MODEL_DIR": str(paths["model"]),
        "ABENCH_CHECKPOINT_DIR": str(paths["checkpoint"]),
        "ABENCH_DEVICE": args.device,
        "ABENCH_SMOKE": "1" if args.smoke else "0",
    })
    # macOS launchd injects compiler/toolchain variables even for an otherwise
    # minimal env.  Clear them so a smoke agent cannot accidentally inherit
    # host include/library search paths; the names remain visible only as empty
    # system compatibility variables.
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "MANPATH", "SDKROOT"):
        env[name] = ""
    passed = []
    for name in args.pass_env:
        if name in os.environ:
            env[name] = os.environ[name]
            passed.append(name)
    if args.device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # The runner gives the agent an ephemeral HOME.  On macOS the bundled
    # system Python may have numpy/torch in the operator's user site, so retain
    # that dependency path for host-soft smoke.  A container wrapper must use
    # its own image dependencies and therefore does not receive this host path.
    if getattr(args, "backend", "host") == "host" and not getattr(args, "agent_wrapper", None):
        user_site = site.getusersitepackages()
        if Path(user_site).is_dir():
            prior = os.environ.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(x for x in (prior, user_site) if x)
    return env, passed


def resource_limits(oj: dict, sandbox: dict, timeout_seconds: float) -> dict:
    """Resource information from the same policy and verified inspect used to create the sandbox."""
    from container_policy import resources_for, bytes_value
    requested = sandbox.get('resources') or resources_for(oj)
    actual = ((sandbox.get('inspect') or {}).get('resource_enforcement') or {}).get('actual') or {}
    enforced = sandbox.get('backend') == 'docker' and sandbox.get('resources_enforced') is True
    limits = {'cpu_cores': actual.get('cpus', 0) / 1e9 if enforced else None,
              'memory_bytes': actual.get('memory') if enforced else None,
              'pids': actual.get('pids') if enforced else None,
              'shm_bytes': bytes_value(requested['shm_size']) if enforced else None}
    current = (sandbox.get('probes') or {}).get('memory_current_bytes')
    return {'schema': 'abench.resource_limits/v1', 'enforced': enforced, 'limits': limits,
            'memory_current_bytes_at_start': current,
            'memory_headroom_bytes_at_start': (max(0, limits['memory_bytes'] - current)
                                               if limits['memory_bytes'] is not None and current is not None else None),
            'requested': requested, 'session_seconds': timeout_seconds,
            'development_budget': oj.get('agent_budget') or {},
            'gpu_count': int(requested.get('gpus', 0)),
            'shared_gpu': bool(sandbox.get('shared_gpu_validation')),
            'gpus': [{k: g.get(k) for k in ('index', 'name', 'memory_mb', 'free_memory_mb')}
                     for g in sandbox.get('gpu_inventory') or []]}


def agent_prompt(paths: dict[str, Path], network: str = "none") -> str:
    view = paths["view"]
    task = (view / "task.md").read_text(encoding="utf-8")
    files = sorted(p.relative_to(view).as_posix() for p in view.rglob("*")
                   if p.is_file() and not p.relative_to(view).as_posix().startswith("data/"))
    data = sorted(p.relative_to(view).as_posix() for p in (view / "data").rglob("*") if p.is_file()) \
        if (view / "data").is_dir() else []
    listing = "\n".join(f"  - {f}" for f in files)
    data_line = ", ".join(data) if data else "none (this case has no data files)"
    return task + f"""

---
Run environment
- Read-only view directory: {view} (also $ABENCH_VIEW_DIR). Files:
{listing}
- Public data in {view / 'data'} ($ABENCH_DATA_DIR): {data_line}.
- Before writing any code, read interface/interface.md and interface/sandbox_guide.md in full; they define the entry
  point, the training program and the self-check you must use.
{NETWORK_TEXT[network]}
- Offline paper search (arXiv titles and abstracts, fast; every call is logged with the run):
  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"   and   ... show <arXiv id>
- Your working directory is {paths['workspace']} ($ABENCH_WORKSPACE); write code, train and keep intermediate files there.
- Scratch checkpoints can go to {paths['checkpoint']}; it is deleted with the workspace. A checkpoint the task asks you
  to submit must be inside the submission directory.
- Write the final submission to {paths['submission']} ($ABENCH_SUBMISSION_DIR), containing only the files the task asks
  for; submission is one-shot.
- The session deadline (Unix seconds) is in the environment variable ABENCH_DEADLINE; processes are killed at that time,
  so write the submission early.
"""


def paper_index_info(arg: str | None) -> dict[str, Any] | None:
    """Path and provenance (meta table) of the paper index, or None if none is installed."""
    import sqlite3
    p = Path(arg).expanduser() if arg else Path(os.environ.get("ABENCH_PAPERS_INDEX") or DEFAULT_PAPER_INDEX)
    if not p.is_file():
        if arg:
            raise SystemExit(f"paper index not found: {p}")
        return None
    con = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM meta"))
    finally:
        con.close()
    return {"path": str(p.resolve()), **{k: meta.get(k) for k in ("snapshot_sha256", "cutoff", "categories", "papers",
                                                                     "builder")}}


def filtered_paper_index(src: str, exclude: list, dst: Path) -> dict[str, Any]:
    """A copy of the paper index without the given arXiv ids (judgment cases: the later papers that settle the
    verdicts). The list itself never reaches the sandbox; the copy is what the agent's tool opens."""
    from contextlib import closing
    import sqlite3
    dst.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(f"file:{src}?mode=ro", uri=True)) as a, closing(sqlite3.connect(dst)) as b:
        with a, b:
            a.backup(b)
    con = sqlite3.connect(dst)
    try:
        removed = 0
        for pid in exclude:
            row = con.execute("SELECT rowid, title, authors, abstract FROM papers WHERE id = ?", (str(pid),)).fetchone()
            if row is None:
                continue
            con.execute("INSERT INTO papers_fts(papers_fts, rowid, title, authors, abstract) VALUES('delete', ?, ?, ?, ?)", row)
            con.execute("DELETE FROM papers WHERE rowid = ?", (row[0],))
            removed += 1
        con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('case_filter', ?)", (f"{removed} papers withheld",))
        con.commit()
    finally:
        con.close()
    dst.chmod(0o444)
    return {"path": str(dst), "withheld": removed}


def collect_retrieval_log(src: Path, dst: Path) -> dict[str, Any]:
    """Copy the agent's retrieval log into the run record and check its hash chain."""
    if not src.is_file():
        return {"entries": 0, "chain_ok": True, "sha256": None}
    raw = src.read_bytes()
    dst.write_bytes(raw)
    lines = raw.splitlines()
    ok, prev, cmds = True, "0" * 64, {}
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            ok = False
            continue
        ok = ok and e.get("prev_sha256") == prev
        prev = hashlib.sha256(line).hexdigest()
        cmds[e.get("cmd", "?")] = cmds.get(e.get("cmd", "?"), 0) + 1
    return {"entries": len(lines), "by_command": cmds, "chain_ok": ok, "sha256": sha256_bytes(raw)}


def collect_submission(src: Path, dst: Path, oj: dict[str, Any]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    allowed = (oj.get("submission") or {}).get("allowed_top_level")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) and x for x in allowed):
        raise SystemExit("submission.allowed_top_level must be a non-empty string list")
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return kept, dropped
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(src).as_posix()
        if SELF_REPORT.search(rel) or "__pycache__" in rel or rel.split("/")[0] not in allowed:
            dropped.append(rel); continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
        kept.append(rel)
    return kept, dropped
