#!/usr/bin/env python3
"""ABench v2 submission format check (OJ-style).

A submission contains only: model/method code, a checkpoint (when required), the design report report/index.html and references.yaml.
This check looks only at the format; it runs no model and computes no metric:

  - the layout matches the submission_kind in case/oj.yaml (code_only / code_plus_checkpoint / report_only);
  - no self-reported metric or result files, no symlinks, total size within the limit;
  - the report uses the arXiv HTML (LaTeXML) structure, has the required anchors, is offline and self-contained,
    and has no experiment, result, ablation, metric or evaluation sections;
  - references.yaml is complete and corresponds one to one with the report's bibliography.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "spec" / "report_profiles.yaml"
# Only result-like *data* files are reserved.  A broad prefix-only pattern
# used here previously classified ordinary source modules such as
# ``src/metrics.py``, ``evaluation.py`` and ``result_types.py`` as self-reported
# results.  Keep the reservation at the file boundary, but require either an
# extensionless reserved name or a known data/log extension.  This still
# catches common aliases such as ``metrics.jsonl`` and ``eval_results.parquet``
# while leaving implementation files (including ``scorer.py``) usable.
_SELF_REPORT_STEMS = r"(?:metrics|results?|scores?|evaluation|eval_results?)"
_SELF_REPORT_DATA_EXTENSIONS = (
    "json", "jsonl", "ndjson", "csv", "tsv", "txt", "yaml", "yml", "toml", "xml",
    "npy", "npz", "parquet", "pkl", "pickle", "pt", "pth", "safetensors", "bin",
    "arrow", "feather", "sqlite", "db", "dat", "log",
)
_SELF_REPORT_DATA_EXTENSIONS_RE = "|".join(_SELF_REPORT_DATA_EXTENSIONS)
SELF_REPORT = re.compile(
    rf"(^|/)(?:{_SELF_REPORT_STEMS}(?:[^/]*)?\.(?:{_SELF_REPORT_DATA_EXTENSIONS_RE})|{_SELF_REPORT_STEMS})$",
    re.I,
)
EXTERNAL = re.compile(r"^\s*(https?:)?//", re.I)


def load_profiles() -> Dict[str, Any]:
    return yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8"))


def profile_for(case_id: str, profiles: Dict[str, Any]) -> str:
    rules = profiles["assignment_rules"]
    explicit = rules.get("exceptions", {}) or {}
    if case_id in explicit:
        return explicit[case_id]
    for suffix in rules["judgment_report_v2"]["suffixes"]:
        if suffix in case_id:
            return "judgment_report_v2"
    return "design_report_v2"


def load_oj(case_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not case_id:
        return None
    p = ROOT / "cases" / case_id / "case" / "oj.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None


class _Report(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: list[str] = []
        self.classes: set[str] = set()
        self.bib_ids: list[str] = []
        self.headings: list[str] = []
        self.forbidden_tags: list[str] = []
        self.external: list[str] = []
        self.cite_targets: list[str] = []
        self.meta: dict[str, str] = {}
        self.changes: Optional[list[dict]] = None     # items of <ol id="changes">: {"text", "code"}
        self._ol_depth = 0                            # nesting depth inside <ol id="changes">
        self._li: Optional[dict] = None
        self._h: Optional[str] = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        if a.get("id"):
            if a["id"] in self.ids:
                self.duplicate_ids.append(a["id"])
            self.ids.add(a["id"])
        for c in (a.get("class") or "").split():
            self.classes.add(c)
        if tag == "li" and "ltx_bibitem" in (a.get("class") or "") and (a.get("id") or "").startswith("bib."):
            self.bib_ids.append(a["id"][4:])
        if tag in ("script", "link", "iframe", "object", "embed", "base", "meta") and not (
                tag == "meta" and ("charset" in a or a.get("http-equiv") or a.get("name") in
                                   ("viewport", "generator", "abench-schema", "abench-case-id"))):
            self.forbidden_tags.append(tag)
        if tag == "meta":
            key = (a.get("name") or a.get("property") or a.get("http-equiv") or "").lower()
            if key and a.get("content") is not None:
                self.meta[key] = a["content"]
        for key in ("src", "srcset", "data", "poster"):
            if a.get(key) and EXTERNAL.match(a[key]):
                self.external.append(f"{tag}[{key}]={a[key][:80]}")
        if tag == "a" and (a.get("href") or "").startswith("#bib."):
            self.cite_targets.append(a["href"][5:])
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._h, self._buf = tag, []
        if tag == "ol" and a.get("id") == "changes" and self.changes is None:
            self.changes, self._ol_depth = [], 1
        elif self._ol_depth:
            if tag in ("ol", "ul"):
                self._ol_depth += 1
            elif tag == "li" and self._ol_depth == 1:
                self._li = {"text": "", "code": False}
                self.changes.append(self._li)
            elif tag == "code" and self._li is not None:
                self._li["code"] = True

    def handle_endtag(self, tag: str) -> None:
        if self._h == tag:
            self.headings.append("".join(self._buf).strip())
            self._h = None
        if self._ol_depth and tag in ("ol", "ul"):
            self._ol_depth -= 1
            if not self._ol_depth:
                self._li = None

    def handle_data(self, data: str) -> None:
        if self._h:
            self._buf.append(data)
        if self._li is not None:
            self._li["text"] += data


def validate_report(sub: Path, case_id: Optional[str], profiles: Dict[str, Any], errors: List[str],
                    oj: Optional[Dict[str, Any]] = None) -> None:
    declared = ((oj or {}).get("submission") or {}).get("report_profile")
    expected = profile_for(case_id or "", profiles) if case_id else declared
    if case_id and declared and declared != expected:
        errors.append(f"report_profile {declared!r} disagrees with assignment rule {expected!r}")
    pname = declared or expected
    if pname not in profiles.get("profiles", {}):
        errors.append(f"unknown report_profile: {pname!r}")
        return
    prof = profiles["profiles"][pname]
    for f in prof["required_files"]:
        if not (sub / f).is_file():
            errors.append(f"report file missing: {f}")
    html_path = sub / "report" / "index.html"
    if not html_path.is_file():
        return
    p = _Report()
    p.feed(html_path.read_text(encoding="utf-8", errors="replace"))
    if case_id:
        declared_case = p.meta.get("abench-case-id")
        if declared_case and declared_case not in (case_id, "<CASE_ID>", "⟨CASE_ID⟩"):
            errors.append(f"report/index.html abench-case-id={declared_case!r} does not match {case_id!r}")
    # The bibliography counts by its ltx_bibitem entries; the section's own ltx_bibliography class is not required.
    for cls in ("ltx_document", "ltx_title_document", "ltx_abstract", "ltx_section", "ltx_bibitem"):
        if cls not in p.classes:
            errors.append(f"report/index.html lacks arXiv HTML structure: .{cls}")
    for anchor in prof["required_anchors"]:
        if anchor not in p.ids:
            errors.append(f"report/index.html lacks required anchor #{anchor}")
    for anchor in sorted(set(p.duplicate_ids)):
        errors.append(f"report/index.html has duplicate id: {anchor}")
    cl = prof.get("changes_list")
    if cl:
        items = [i for i in (p.changes or []) if i["text"].strip()]
        if p.changes is None:
            errors.append('report/index.html lacks the "Changes relative to the anchor" list <ol id="changes">')
        elif len(items) < int(cl.get("min_items", 1)):
            errors.append(f'report/index.html: <ol id="changes"> needs at least {cl.get("min_items", 1)} non-empty item(s)')
        elif cl.get("require_code_location"):
            for n, i in enumerate(items, 1):
                if not i["code"]:
                    errors.append(f'report/index.html: change {n} in <ol id="changes"> has no code location '
                                  '(a <code> element naming the file and function)')
    tokens = [t.lower() for t in profiles.get("forbidden_heading_tokens", [])]
    for h in p.headings:
        hit = [t for t in tokens if re.search(rf"\b{re.escape(t)}(?:s|al)?\b", h, re.I)]
        if hit:
            errors.append(f"report/index.html contains a forbidden section \"{h}\" (the report covers design rationale only, not {hit[0]})")
    for t in sorted(set(p.forbidden_tags)):
        errors.append(f"report/index.html may not use <{t}>; the report must be offline and self-contained")
    for e in p.external:
        errors.append(f"report/index.html references an external resource: {e}")
    for asset in (sub / "report").rglob("*"):
        if asset.is_file() and asset.suffix.lower() not in (".html", ".svg", ".png", ".jpg", ".jpeg", ".webp"):
            errors.append(f"file type not allowed under report/: {asset.relative_to(sub)}")

    ref_path = sub / "references.yaml"
    if not ref_path.is_file():
        return
    try:
        doc = yaml.safe_load(ref_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        errors.append(f"references.yaml is not valid YAML: {exc}"); return
    refs = doc.get("references") if isinstance(doc, dict) else None
    if not isinstance(refs, list) or not refs:
        errors.append("references.yaml must contain a non-empty references list"); return
    seen = set()
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            errors.append(f"references[{i}] must be a mapping"); continue
        rid = r.get("id")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", rid or ""):
            errors.append(f"references[{i}].id is missing or contains illegal characters"); continue
        if rid in seen:
            errors.append(f"duplicate id in references.yaml: {rid}")
        seen.add(rid)
        for k in ("title", "year", "used_for"):
            if not r.get(k):
                errors.append(f"references[{rid}] lacks {k}")
        if not any(r.get(k) for k in ("arxiv", "doi", "url")):
            errors.append(f"references[{rid}] must give one of arxiv, doi or url")
    bib = set(p.bib_ids)
    for rid in sorted(seen - bib):
        errors.append(f"{rid} in references.yaml does not appear in the report bibliography (needs <li id=\"bib.{rid}\">)")
    for rid in sorted(bib - seen):
        errors.append(f"report bibliography entry bib.{rid} is not registered in references.yaml")
    for rid in sorted(set(p.cite_targets) - bib):
        errors.append(f"citation #bib.{rid} in the body has no bibliography entry")


def validate_layout(sub: Path, oj: Optional[Dict[str, Any]], errors: List[str]) -> None:
    files = [p for p in sub.rglob("*")]
    for p in files:
        if p.is_symlink():
            errors.append(f"symlinks are not allowed: {p.relative_to(sub)}")
    rel = [p.relative_to(sub).as_posix() for p in files if p.is_file()]
    for r in rel:
        if SELF_REPORT.search(r):
            errors.append(f"the submission must not contain self-reported metric or result files: {r} (metrics are computed only by the judge)")
    if not oj:
        return
    s = oj.get("submission", {})
    kind = s.get("kind")
    total = sum((sub / r).stat().st_size for r in rel)
    if s.get("max_bytes") and total > s["max_bytes"]:
        errors.append(f"total submission size {total} bytes exceeds the limit {s['max_bytes']}")
    if not isinstance(s.get("allowed_top_level"), list) or not s.get("allowed_top_level") \
            or not all(isinstance(x, str) and x for x in s.get("allowed_top_level", [])):
        errors.append("oj.yaml lacks a valid submission.allowed_top_level allowlist")
    # optional: limit file types and size of the code area (so pretrained weights cannot be hidden in a code_only submission)
    suffixes = s.get("code_suffixes")
    if suffixes:
        allowed_sfx = {x.lower() for x in suffixes}
        ck_top = s.get("checkpoint_dir") if kind == "code_plus_checkpoint" else None
        code_files = [r for r in rel if r.split("/")[0] not in ("report", "references.yaml", "README.md", ck_top)]
        for r in code_files:
            if Path(r).suffix.lower() not in allowed_sfx:
                errors.append(f"file type not allowed in the code area: {r} (allowed: {', '.join(sorted(allowed_sfx))})")
        code_bytes = sum((sub / r).stat().st_size for r in code_files)
        if s.get("code_max_bytes") and code_bytes > s["code_max_bytes"]:
            errors.append(f"code and configuration total {code_bytes} bytes, over the limit {s['code_max_bytes']}")
    if kind in ("code_only", "code_plus_checkpoint"):
        entry = s.get("entry")
        if entry and not ((sub / entry).is_file() or (sub / entry).is_dir() and any((sub / entry).glob("*.py"))):
            errors.append(f"entry missing: {entry} (a file, or a directory of .py files for cases with one file per problem)")
        for f in s.get("required_files") or []:          # e.g. train.py, the program that produced the checkpoint
            if not (sub / f).is_file():
                errors.append(f"required file missing: {f}")
    if kind == "code_plus_checkpoint":
        ck = sub / s.get("checkpoint_dir", "checkpoint")
        if not ck.is_dir() or not any(x.is_file() for x in ck.rglob("*")):
            errors.append(f"checkpoint missing: {s.get('checkpoint_dir', 'checkpoint')}/ is empty or does not exist")
        elif s.get("checkpoint_max_bytes"):
            ck_bytes = sum(x.stat().st_size for x in ck.rglob("*") if x.is_file())
            if ck_bytes > s["checkpoint_max_bytes"]:
                errors.append(f"checkpoint is {ck_bytes} bytes, over the limit {s['checkpoint_max_bytes']}")
    allowed = s.get("allowed_top_level")
    if isinstance(allowed, list) and allowed:
        for r in rel:
            top = r.split("/")[0]
            if top not in allowed:
                errors.append(f"top-level file not allowed: {r} (allowed: {', '.join(allowed)})")


def validate_submission_dir(submission: Path, expected_case_id: Optional[str] = None, **_: Any) -> List[str]:
    errors: List[str] = []
    sub = Path(submission)
    if not sub.is_dir():
        return [f"submission directory does not exist: {sub}"]
    profiles = load_profiles()
    oj = load_oj(expected_case_id)
    validate_layout(sub, oj, errors)
    validate_report(sub, expected_case_id, profiles, errors, oj)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="ABench v2 submission format check")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--case-id")
    a = ap.parse_args()
    errs = validate_submission_dir(Path(a.submission), a.case_id)
    print(json.dumps({"valid": not errs, "errors": errs}, ensure_ascii=False, indent=2))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
