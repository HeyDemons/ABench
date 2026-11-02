#!/usr/bin/env python3
"""Check whether each case's agent task statement conforms to task statement standard v2.1 (spec/task_statement_standard.md).

Reads files only and runs no submissions. Prints each case's conformance and what is missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from leak_check import leak_allow, leaky  # noqa: E402
from container_policy import bytes_value, resources_for  # noqa: E402

SECTIONS = ["Task", "Data", "Interface", "Metrics", "Evaluation", "Sandbox Guide", "Deliverables", "Rules"]


def sections(md: str) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def allocation_issues(guide: str, oj: dict) -> list[str]:
    """The Sandbox Guide must state what the runner applies from oj.yaml: session length, development compute
    allowance, GPU count and, when stated, CPU cores and RAM. A GPU's VRAM ('24GB reference GPU') is not RAM."""
    session = re.search(r"session lasts at most ([0-9.]+) hours?", guide, re.I)
    compute = re.search(r"(?:Compute|GPU usage) during development is capped at ([0-9.]+) (CPU-core|GPU|H100)-hours?",
                        guide, re.I)
    if not session or not compute:
        return ["Sandbox Guide: cannot read the session length or the development compute allowance"]
    try:
        resources = resources_for(oj)
    except ValueError as exc:
        return [f"oj.yaml: {exc}"]
    # The hardware label is the first bullet up to its first ';' (build_standard_cases.HARDWARE).
    hardware = next((line for line in guide.splitlines() if line.startswith("- ")), "").split(";")[0]
    if m := re.search(r"\b(\d+)\s*[×x]\s*H100\b", hardware):
        gpus = int(m[1])
    elif re.search(r"\bno GPU\b", hardware, re.I):
        gpus = 0
    elif m := re.search(r"\b(one|\d+)\b[^;]*?\bGPUs?\b", hardware, re.I):
        gpus = 1 if m[1].lower() == "one" else int(m[1])
    else:
        return ["Sandbox Guide: cannot read the GPU allocation"]
    budget = oj.get("agent_budget") or {}
    key = {"cpu-core": "cpu_core_hours", "gpu": "gpu_hours", "h100": "h100_hours"}[compute[2].lower()]
    stated = {"session hours": (float(session[1]), resources["wall_clock_limit_h"]),
              f"agent_budget.{key}": (float(compute[1]), budget.get(key)), "GPUs": (gpus, resources.get("gpus"))}
    if "session_hours" in budget:
        stated["agent_budget.session_hours"] = (float(session[1]), budget["session_hours"])
    if m := re.search(r"\b(\d+) CPU cores\b", hardware, re.I):
        stated["CPU cores"] = (int(m[1]), resources.get("cpus"))
    if m := re.search(r"\b([0-9.]+)\s*([KMGT]i?B)\s+(?:of\s+)?(?:RAM|system memory)\b", hardware, re.I):
        stated["RAM bytes"] = (bytes_value(m[1] + m[2]), bytes_value(resources["memory"]))
    return [f"Sandbox Guide states {name} {said}, oj.yaml has {applied}"
            for name, (said, applied) in stated.items() if applied is None or float(said) != float(applied)]


def check_case(cid: str) -> dict:
    case = ROOT / "cases" / cid / "case"
    oj = yaml.safe_load((case / "oj.yaml").read_text(encoding="utf-8")) or {}
    vis = oj.get("agent_visible") or {}
    task_path = case.parent / vis.get("task", "case/agent/task.md")
    md = task_path.read_text(encoding="utf-8") if task_path.is_file() else ""
    sec = sections(md)
    issues: list[str] = []
    for s in SECTIONS:
        if s not in sec or len(sec[s].strip()) < 20:
            issues.append(f"missing or too-short section: {s}")
    metrics = sec.get("Metrics", "")
    if metrics:
        if not re.search(r"^\|.*\|.*\|", metrics, re.M):
            issues.append("Metrics: no metric table")
        if not re.search(r"primary metric|\(primary\)", metrics, re.I):
            issues.append("Metrics: primary metric not marked")
        if not re.search(r"threshold|no threshold", metrics, re.I):
            issues.append("Metrics: threshold not stated (or no-threshold not declared)")
        if not re.search(r"seed|mean|median|average|aggregat", metrics, re.I):
            issues.append("Metrics: aggregation not stated")
        if not re.search(r"valid|invalid", metrics, re.I):
            issues.append("Metrics: no validity rules")
    guide = sec.get("Sandbox Guide", "")
    if guide:
        for need, msg in ((r"ABENCH_DEADLINE", "session limit ABENCH_DEADLINE"), (r"ABENCH_SUBMISSION_DIR", "submission directory"),
                          (r"[Nn]etwork", "network restrictions"), (r"GPU|CPU", "hardware"), (r"selfcheck|self-check", "self-check command")):
            if not re.search(need, guide):
                issues.append(f"Sandbox Guide: missing {msg}")
        issues += allocation_issues(guide, oj)
    task =sec.get("Task", "")
    if task:
        # v2.1 working procedure: retrieval with the offline paper tool, related-work synthesis, a stated list of
        # changes relative to the anchor, and development with the self-check
        if oj.get("mode") == "judgment":    # judgment cases: evidence, calibrated judgments, a verification plan
            needs = ((r"### Workflow", "### Workflow subsection"), (r"abench_papers\.py", "paper search tool"),
                     (r"judgments\.yaml", "judgments file"), (r"verification plan", "verification plan"),
                     (r"selfcheck", "self-check in the workflow"))
        else:
            needs = ((r"### Workflow", "### Workflow subsection"), (r"abench_papers\.py", "paper search tool"),
                     (r"Related Work", "related-work synthesis (report Section 2)"),
                     (r'id="changes"', 'changes list <ol id="changes">'), (r"selfcheck", "self-check in the workflow"))
        for need, msg in needs:
            if not re.search(need, task):
                issues.append(f"Task: missing {msg}")
    if oj.get("mode") == "checkpoint" or (oj.get("submission") or {}).get("kind") == "code_plus_checkpoint":
        req = (oj.get("submission") or {}).get("required_files") or []
        if oj.get("standard") and "train.py" not in req:
            issues.append("checkpoint submission without required train.py (the program that produced the checkpoint)")
    iface = vis.get("interface") or []
    if not iface:
        issues.append("oj.yaml: no agent-visible interface files")
    for rel in iface:
        source = case.parent / rel
        if not source.is_file():
            issues.append(f"interface file does not exist: {rel}")
            continue
        hits = leaky(source.read_text(encoding="utf-8", errors="replace"), leak_allow(oj))
        if hits:
            issues.append(f"interface contains disallowed judge-side terms: {rel}: {hits}")
    if oj.get("task_statement") != "v2.1":
        issues.append("oj.yaml: task_statement: v2.1 not declared")
    judge = oj.get("judge") or {}
    if judge.get("kind") in (None, "pending"):
        issues.append("judge not configured (judge.kind: pending)")
    hits = leaky(md, leak_allow(oj))
    if hits:
        issues.append(f"statement contains disallowed judge-side terms: {hits}")
    state = "conform" if not issues else ("statement_only" if not [i for i in issues if not i.startswith("judge")] else "draft")
    return {"case": cid, "state": state, "judge": judge.get("kind", "pending"),
            "activation": judge.get("activation", "active" if judge.get("kind") not in (None, "pending") else None),
            "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case")
    ap.add_argument("--strict", action="store_true", help="return non-zero if any case does not conform")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ids = [a.case] if a.case else sorted(p.parent.parent.name for p in (ROOT / "cases").glob("*/case/oj.yaml"))
    res = [check_case(c) for c in ids]
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        counts: dict[str, int] = {}
        for r in res:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        print("task statement standard v2.1:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        for r in res:
            if r["state"] == "conform" or a.case:
                print(f"  {r['case']}: {r['state']}  judge={r['judge']} activation={r['activation']}")
                for i in r["issues"]:
                    print("    -", i)
    return 1 if a.strict and any(r["state"] != "conform" for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
