"""Export data for the demo site: site/data/abench-data.js (window.ABENCH = {...}).

All data comes from real files in the repository:
  index.yaml                                 all tracks and cases
  cases/B-OPT-001/outputs/*/score_report.json scoring results of the three submissions (training losses and judge outputs are simulated)
  cases/B-OPT-001/outputs/mock/runs_*.json    "budget–final loss" curves of the reference methods (simulated)
  cases/B-OPT-001/case/paper_snapshot/*.jsonl paper snapshot and external resolution cache
  cases/B-OPT-001/example_submission/*        report sources of the example submission; retrieval log and local discoveries optional

A JS file rather than JSON is used so that index.html also works when opened by double-clicking (file://).
"""
import json
from pathlib import Path

import yaml
from catalog import load_catalog, snapshot, distribution

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "B-OPT-001"
OUT = ROOT / "site" / "data" / "abench-data.js"
SUBS = ["example_submission", "clone_muon", "rankgate_linear_gate"]
SUB_LABEL = {"example_submission": "RankGate (example submission)",
             "clone_muon": "Renamed Muon clone (negative)",
             "rankgate_linear_gate": "Early linear-gate version (negative)"}


def jl(p):
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


SITE = ROOT / "site"
REPORT_COPIES = {  # report copies for previews inside the site (files outside the site root are unreachable from the local server)
    "reports/B-OPT-001-rankgate.html": ROOT / "cases/B-OPT-001/example_submission/report/index.html",
    "reports/design_template.html": ROOT / "sandbox/templates/design_report.html",
    "reports/judgment_template.html": ROOT / "sandbox/templates/judgment_report.html",
}


def oj_summary(cid: str) -> dict:
    """Summary of the agent view and OJ configuration. Reads only oj.yaml and task.md, never the content of judge-side files."""
    case = ROOT / "cases" / cid / "case"
    oj_path = case / "oj.yaml"
    if not oj_path.is_file():
        return {}
    oj = yaml.safe_load(oj_path.read_text(encoding="utf-8")) or {}
    sub, judge, data = oj.get("submission") or {}, oj.get("judge") or {}, oj.get("data") or {}
    vis = oj.get("agent_visible") or {}
    prof = sub.get("report_profile") or "design_report_v2"
    view = ["task.md"] + [f"interface/{Path(r).relative_to('harness').as_posix() if Path(r).parts[:1] == ('harness',) else Path(r).name}" for r in vis.get("interface") or []]
    view += ["templates/report_template.html", "templates/references_template.yaml"]
    view += [f"data/{Path(d['path']).name}" for d in data.get("public") or [] if d.get("path")]
    task = case / "agent" / "task.md"
    return {
        "kind": sub.get("kind"), "entry": sub.get("entry"), "checkpoint_dir": sub.get("checkpoint_dir"),
        "allowed": sub.get("allowed_top_level") or [], "report_profile": prof,
        "judge": judge.get("kind", "pending"),
        "judge_metrics": ([judge["metrics"]] if isinstance(judge.get("metrics"), str) else list(judge.get("metrics") or [])),
        "judge_planned": judge.get("planned"), "handcrafted": bool(oj.get("handcrafted")),
        "standard": bool(oj.get("standard")), "activation": judge.get("activation"),
        "task_statement": oj.get("task_statement"), "mode": oj.get("mode"), "template": oj.get("template"),
        "required_files": sub.get("required_files") or [], "agent_budget": oj.get("agent_budget") or {},
        "checkpoint_max_bytes": sub.get("checkpoint_max_bytes"),
        "public_rows": sum(int(d.get("rows") or 0) for d in data.get("public") or []),
        "hidden_rows": int((data.get("hidden") or {}).get("rows") or 0),
        "view": view,
        "task_md": task.read_text(encoding="utf-8") if task.is_file() else "",
    }


def _module_consts(path: Path, names: tuple) -> dict:
    """Literal module-level constants (PRIMARY, GATES, ...) read with ast, without importing the case code."""
    import ast
    out = {}
    if not path.is_file():
        return out
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(tgt, ast.Name) and tgt.id in names and getattr(node, "value", None) is not None:
            try:
                out[tgt.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _e2e_cases() -> set:
    """Cases whose whole pipeline runs end to end on synthetic data in tests/test_standard_cases.py."""
    consts = _module_consts(ROOT / "tests" / "test_standard_cases.py", ("NUMPY_CASES",))
    return set(consts.get("NUMPY_CASES") or {})


def standard_cases(index: dict) -> list:
    """One row per standard case: what is submitted, what the evaluator does, metrics, gates and verification state."""
    track_of = {c["id"]: t for t in index.get("tracks", []) for c in t.get("cases", []) or []}
    e2e = _e2e_cases()
    rows = []
    specs = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
             for p in sorted((ROOT / "spec" / "standard_cases").glob("*.yaml"))}
    ids = sorted(set(specs) | {"IC-SPD-001"})
    for cid in ids:
        s = specs.get(cid) or {}
        oj = yaml.safe_load((ROOT / "cases" / cid / "case" / "oj.yaml").read_text(encoding="utf-8")) or {}
        sub, judge, data = oj.get("submission") or {}, oj.get("judge") or {}, oj.get("data") or {}
        consts = _module_consts(ROOT / "cases" / cid / "harness" / "metrics.py", ("PRIMARY", "GATES"))
        t = track_of.get(cid) or {}
        anchor = s.get("anchor") or {}
        src = (data.get("sources") or [{}])[0]
        mode = oj.get("mode") or ("judge_trains" if cid == "IC-SPD-001" else None)
        rows.append({
            "id": cid, "title": s.get("title") or next((c.get("task") for c in t.get("cases", []) if c["id"] == cid), ""),
            "track": t.get("name"), "group": t.get("group"),
            "anchor": anchor.get("title") or next((c.get("anchor") for c in t.get("cases", []) if c["id"] == cid), ""),
            "arxiv": anchor.get("arxiv") or next((c.get("arxiv") for c in t.get("cases", []) if c["id"] == cid), None),
            "mode": mode, "kind": sub.get("kind"), "entry": sub.get("entry"),
            "required_files": sub.get("required_files") or [],
            "hardware": ((s.get("hardware") or {}).get("kind") or ("gpu" if cid == "IC-SPD-001" else None)),
            "runs": len(judge.get("seeds") or []) or None,
            "primary": consts.get("PRIMARY") or ({"metric": "train_seconds", "goal": "min"} if cid == "IC-SPD-001" else None),
            "gates": consts.get("GATES") if "GATES" in consts else (
                [{"metric": "cifar10_test_accuracy", "min": 0.94}, {"metric": "cifar10_1_accuracy", "min": 0.80}]
                if cid == "IC-SPD-001" else None),          # IC-SPD-001 has its own timing judge, gates from its task.md
            "baseline": s.get("baseline"),
            "data_source": src.get("name"), "license": src.get("license"),
            "activation": judge.get("activation"),
            "verified_e2e_synthetic": cid in e2e,
            "reference_verification": s.get("reference_verification"),
        })
    return rows


def copy_reports() -> list:
    out = []
    for rel, src in REPORT_COPIES.items():
        if src.is_file():
            dst = SITE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            out.append(rel)
    return out


def display_q(raw: dict) -> dict:
    """Keep historical unresolved score inputs out of the public source chart.

    The score reports are immutable historical fixtures and may contain an
    unresolved source identifier from an old simulated run.  It must remain
    available for audit, but the browser chart should not present it as a
    verified citation.  The aggregate Q value is retained unchanged.
    """
    q = dict(raw or {})
    rows = list(q.get("sources") or [])
    unresolved = [row for row in rows if row.get("status") == "unresolvable"]
    q["sources"] = [row for row in rows if row.get("status") != "unresolvable"]
    if unresolved:
        # Keep only the count in the public bundle.  The original rows remain
        # in the immutable score report for audit, while stale/fabricated
        # identifiers cannot be mistaken for citations in the browser data.
        q["unresolved_source_count"] = len(unresolved)
    return q


def registry() -> dict:
    """verify.csv rows by case id: the hand-maintained design and verification state shown in the case browser."""
    import csv
    p = ROOT / "verify.csv"
    if not p.is_file():
        return {}
    keep = ("area", "subarea", "design", "mode", "device", "data_source", "ref_run", "ref_result", "data_frozen", "activation",
            "review_tier", "review_note", "blockers", "next_step", "last_update")
    return {r["case_id"]: {k: r.get(k, "") for k in keep} for r in csv.DictReader(open(p, newline="", encoding="utf-8"))}


def main():
    index = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
    reg = registry()
    catalog = load_catalog(ROOT)
    catalog_by = {row["id"]: row for row in catalog}

    # Keep the browser view honest about scaffold readiness.  The index status
    # describes the repository artifact (all current entries are ``scaffold``),
    # while the card readiness describes how close a case is to real scoring.
    # Derive this once here so the static site cannot silently drop or invent a
    # status label when a new case is added.
    for track in index.get("tracks", []):
        for item in track.get("cases", []) or []:
            cid = item.get("id")
            card_path = ROOT / "cases" / str(cid) / "case" / "anchor_card.yaml"
            card = yaml.safe_load(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
            item["readiness"] = (card or {}).get("readiness", "contract_only")
            item["anchor_status"] = (card or {}).get("status")
            item["oj"] = oj_summary(str(cid))
            item["registry"] = reg.get(str(cid), {})
            item["catalog"] = snapshot(catalog_by[str(cid)])
            item["difficulty"] = catalog_by[str(cid)]["difficulty"]
            item["budget"] = catalog_by[str(cid)]["resource_tier"]

    # ---- paper snapshot (hero citation graph)
    snap = jl(CASE / "case/paper_snapshot/seed_neighborhood.jsonl") + jl(CASE / "case/paper_snapshot/resolver_cache.jsonl")
    rep0 = json.loads((CASE / "outputs/example_submission/score_report.json").read_text(encoding="utf-8"))
    weights = rep0["paper_weights"]
    papers = [{"id": p["id"], "title": p["title"], "arxiv": p.get("arxiv"), "year": int(p["first_public"][:4]),
               "date": p["first_public"], "venue": p["venue"], "tier": p["tier"], "role": p["role"],
               "field": p.get("field", "ml"), "w": weights.get(p["id"], {}).get("w")} for p in snap]

    # ---- scoring of the three submissions
    subs = {}
    for name in SUBS:
        r = json.loads((CASE / f"outputs/{name}/score_report.json").read_text(encoding="utf-8"))
        subs[name] = {
            "label": SUB_LABEL[name],
            "composite": r["composite"],
            "L0": r["L0"],
            "M": {k: r["M"][k] for k in ("M", "aggregate_speedup", "aggregate_speedup_lcb", "gate_pass", "gate_checks", "ood")}
                 | {"per_setting": {k: {kk: v[kk] for kk in ("speedup", "per_seed", "submission_loss_mean", "best_ref_at_1x", "time_ratio")}
                                    for k, v in r["M"]["per_setting"].items()}},
            "D": {"D": r["D"]["D"], "gate_pass": r["D"]["gate_pass"], "gate_checks": r["D"]["gate_checks"],
                  "threshold": r["D"]["behavioral_threshold"], "noise_floor": r["D"]["noise_floor"]["floor"],
                  "behavioral": {k: r["D"]["behavioral"][k] for k in ("D_beh", "D_global", "D_patch", "nearest_reference", "similarity_by_reference", "nearest_by_tensor")},
                  "code": r["D"]["code"]["D_code"], "code_nearest": r["D"]["code"]["nearest_reference"],
                  "mech": r["D"]["mechanism"]["D_mech"], "mech_levels": r["D"]["mechanism"]["levels"],
                  "nearest_prior_art": r["D"]["mechanism"]["nearest_prior_art"]},
            "Q": display_q(r["Q"]),
            "X": r["X"],
        }

    # ---- reference curves (L2_main = μ2)
    runs = json.loads((CASE / "outputs/mock/runs_example_submission.json").read_text(encoding="utf-8"))
    curves = {}
    for setting, s in runs["settings"].items():
        curves[setting] = {
            "token_budget": s["token_budget"],
            "time_ratios": s["time_ratios"],
            "references": {k: {"budgets": v["budgets"], "loss": [sum(x) / len(x) for x in v["loss"]]} for k, v in s["references"].items()},
        }
    for name in SUBS:
        rr = json.loads((CASE / f"outputs/mock/runs_{name}.json").read_text(encoding="utf-8"))
        for setting, s in rr["settings"].items():
            curves[setting].setdefault("submissions", {})[name] = {"loss": s["submission"]["loss"], "time_ratio": s["submission"]["time_ratio"]}

    # ---- process material of the example submission
    ex = CASE / "example_submission"
    # since v2 the retrieval log is recorded by the operator; the example's historical log moved to sim/legacy_example/
    retrieval = jl(ex / "retrieval_log.jsonl") or jl(CASE / "sim/legacy_example/retrieval_log.jsonl")
    # v2 reports register sources in references.yaml; retain a legacy fallback
    # so historical site snapshots can still be rebuilt from old submissions.
    ref_path = ex / "references.yaml"
    if ref_path.is_file():
        ref_doc = yaml.safe_load(ref_path.read_text(encoding="utf-8")) or {}
        sources = []
        for ref in ref_doc.get("references", []):
            if ref.get("arxiv"):
                sid = f"arXiv:{ref['arxiv']}"
            elif ref.get("doi"):
                sid = f"doi:{ref['doi']}"
            else:
                sid = f"url:{ref.get('url')}"
            sources.append({"id": sid, "borrowed": ref.get("used_for", ""),
                            "code": ref.get("code_location") or "unused"})
    else:
        sources = yaml.safe_load((ex / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    discoveries_path = ex / "discoveries.yaml"
    if discoveries_path.is_file():
        discoveries = (yaml.safe_load(discoveries_path.read_text(encoding="utf-8")) or {}).get("discoveries", [])
    elif (CASE / "sim/discovery_claims.yaml").is_file():
        # v2: local discoveries are extracted from the report by the operator; the example claims are kept in sim/
        discoveries = (yaml.safe_load((CASE / "sim/discovery_claims.yaml").read_text(encoding="utf-8")) or {}).get("discoveries", [])
    else:
        discoveries = []
    proto = yaml.safe_load((CASE / "case/eval_protocol.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((CASE / "case/scoring_config.yaml").read_text(encoding="utf-8"))
    refpool = yaml.safe_load((CASE / "case/reference_pool.yaml").read_text(encoding="utf-8"))["references"]

    data = {
        "_note": "Only the training losses and judge outputs in the historical demo of B-OPT-001 are simulated; no results were generated for the new cases, and formal scoring must use a real runner",
        "index": index,
        "catalog_filters": {key: sorted({str(row[key]) for row in catalog})
                            for key in ("difficulty", "resource_tier", "area", "group", "type", "subset")},
        "catalog_distribution": distribution(catalog),
        "standard": standard_cases(index),
        "areas": yaml.safe_load((ROOT / "spec" / "areas.yaml").read_text(encoding="utf-8"))["areas"],
        "launch": {"case": "TC-FT-001", "script": "./abench doctor; ./abench images pull; python3 tools/prepare/run.py --case TC-FT-001 --raw-dir RAW --key-file ~/.abench/hidden.key; ./abench calibrate TC-FT-001; ./abench run --case TC-FT-001 --agent mock",
                   "steps": ["preflight", "key", "data", "mock", "agent", "judge"],
                   "verified": "all six steps run in a cloud container with synthetic data and a stub claude CLI (2026-10-20); "
                               "the runner and judge path calibrated end to end on 2026-10-23 (see Calibration runs)",
                   "pending": "the real Yelp Review Full data (Hugging Face) and a run with the real claude CLI"},
        "calibration": {
            "date": "2026-10-23",
            "runs": [
                {"case": "TC-FT-001", "data": "synthetic (Yelp not reachable here)", "primary": "accuracy 1.000 [1.000, 1.000]",
                 "agent": "train.py 2 s"},
                {"case": "DS-PIPE-002", "data": "synthetic (OpenML not reachable here)",
                 "primary": "mean accuracy 0.904 [0.896, 0.911]", "agent": "train.py 6 s"},
                {"case": "FN-LTSF-005", "data": "real ETT + electricity",
                 "primary": "MSE ratio 0.999 [0.993, 1.007]", "agent": "train.py --final 4.7 min"},
            ],
            "note": ("OJ calibration (./abench calibrate CASE, on the data frozen in the checkout): the oracle adapter trains the reference submission with "
                     "its own train.py in a container and tools/oj_eval.py judges the held-out split in fresh stage containers. "
                     "The historical numbers above predate the Docker migration. The record gives the base score "
                     "of the case and its interval. Calibration runs are never scorable (synthetic data, or a case awaiting "
                     "its data freeze)."),
        },
        "reports": copy_reports(),
        "case": {
            "id": "B-OPT-001",
            "papers": papers,
            "submissions": subs,
            "curves": curves,
            "retrieval": retrieval,
            "sources": sources,
            "discoveries": discoveries,
            "settings_weights": proto["settings_weights"],
            "ladder": proto["ladder"],
            "budget": proto["budget"],
            "exponents": cfg["composite"]["exponents"],
            "discovery_bonus": cfg["composite"]["discovery_bonus"],
            "references": [{"id": r["id"], "name": r["name"], "arxiv": r.get("arxiv"), "venue": r["venue"]} for r in refpool],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.ABENCH = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
