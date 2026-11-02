"""Independent judge: decrypt labels on the host, run each stage in a fresh offline Docker container.

Only unlabeled inputs cross into the stage. metrics.py, bootstrap intervals and gates run on the host.
The local transport remains available to selfchecks and internal regression/calibration tests; it never scores.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import hidden_store  # noqa: E402
from file_hashes import sha256_file, tree_hash  # noqa: E402
from validate_submission import validate_submission_dir  # noqa: E402

JUDGE_VERSION = "abench-oj-judge-v2.1"


def _dependency_attestation(image: str | None = None, locked: bool = False, lock: str | None = None) -> dict[str, Any]:
    """A locked image (published digest, or a host-local image ID lock) and the host tools tree are both required
    for scoreability."""
    expected = os.environ.get("ABENCH_JUDGE_TOOLS_SHA256", "").lower()
    actual = tree_hash(TOOLS)
    tools_attested = bool(expected) and expected == actual
    return {"source": "container_image" if image else "interpreter_default", "image": image, "paths": [],
            "trusted": bool(image and locked and tools_attested), "image_locked": bool(locked),
            "image_lock": lock if locked else None,
            "tools_sha256": actual, "tools_sha256_expected": expected or None, "tools_attested": tools_attested}


def _environment_check(case: str, image: str | None = None, locked: bool = False) -> dict[str, Any]:
    from images import env_lock_path
    oj = load_oj(case)
    environment = oj.get("environment") or {}
    profile = environment.get("profile") or oj.get("profile")
    lock = env_lock_path(profile, environment.get("version"))
    matches = bool(image and locked and lock.is_file())
    reason = None if matches else 'judge image digest or profile environment lock not verified'
    return {'profile':profile,'image':image,'lock':str(lock.relative_to(ROOT)),
            'matches':matches,'differences':[] if matches else [reason],
            **({'reason':reason} if reason else {})}



def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_oj(case: str) -> dict[str, Any]:
    p = ROOT / "cases" / case / "case" / "oj.yaml"
    if not p.is_file():
        raise SystemExit(f"{case}: case/oj.yaml not found")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _verify_public(case: str, oj: dict[str, Any]) -> Path | None:
    """Check the frozen public files (sha256 per file) and return the public data directory."""
    data = oj.get("data") or {}
    pdir = data.get("public_dir")
    if not pdir:
        return None
    root = (ROOT / pdir).resolve()
    for f in data.get("public") or []:
        p = ROOT / f["path"]
        if not p.is_file() or sha256_file(p) != f["sha256"]:
            raise SystemExit(f"public data hash mismatch: {f['path']}")
    return root


def _verify_models(oj: dict[str, Any], kind: str = "models") -> Path | None:
    """The frozen model directory (kind "models", also in the agent's sandbox) or the evaluator-only one (kind
    "judge_models", e.g. a reward model; never mounted in a sandbox), after checking every tree hash."""
    data = oj.get("data") or {}
    mdir = data.get(f"{kind}_dir")
    if not mdir:
        return None
    root = (ROOT / mdir).resolve()
    for m in data.get(kind) or []:
        p = ROOT / m["path"]
        if not p.is_dir() or tree_hash(p) != m["tree_sha256"]:
            raise SystemExit(f"frozen model tree hash mismatch: {m['path']}")
    return root


def _aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for m in per_seed for k in m})
    return {k: round(float(np.mean([m[k] for m in per_seed if k in m])), 6) for k in keys}


def _gates(gates: list[dict], m: dict[str, float]) -> dict[str, Any]:
    out = {}
    for g in gates:
        v = m.get(g["metric"])
        ok = v is not None and np.isfinite(v) and ("min" not in g or v >= g["min"]) and ("max" not in g or v <= g["max"])
        out[g["metric"]] = {"min": g.get("min"), "max": g.get("max"), "value": v, "pass": bool(ok)}
        if "min" in g and "max" not in g:
            out[g["metric"]]["threshold"] = g["min"]  # legacy IC-SPD record field

    return out


def _primary_ci(metrics, res: dict, reference: dict, params: dict) -> dict | None:
    """Bootstrap 95% interval of the primary metric when the case declares RESAMPLE (examples of the held-out set)."""
    groups = getattr(metrics, "RESAMPLE", None)
    primary = (getattr(metrics, "PRIMARY", None) or {}).get("metric")
    if not groups or not primary:
        return None
    import abench_metrics
    try:
        return abench_metrics.bootstrap_ci(metrics.score, res["outputs"], reference, params, res["seconds"], groups, primary)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def evaluate_harness(case: str, submission: Path, key_file: Path, oj: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="abench-judge-"))
    os.chmod(tmp, 0o700)
    launcher = rec.get("_launcher")
    try:
        return _evaluate_harness(case, submission, key_file, oj, rec, tmp)
    finally:
        if launcher is not None:
            launcher.close_inputs()                     # the read-only inputs shared by this evaluation's stages
        shutil.rmtree(tmp, ignore_errors=True)


def _evaluate_harness(case: str, submission: Path, key_file: Path, oj: dict[str, Any], rec: dict[str, Any],
                      tmp: Path) -> dict[str, Any]:
    """harness: run the case pipeline (harness/stages.py) through the shared stage runner, once per hidden seed.

    The held-out inputs are copied into a run directory only after the timed phase has ended; labels and reference
    values stay in this process and are scored by the case's harness/metrics.py. If the case declares a baseline
    submission, it is run first under identical conditions and relative metrics are added.
    """
    j = oj["judge"]
    launcher = rec.pop("_launcher", None)
    cdir = ROOT / "cases" / case
    kit = cdir / "harness" / "abench_stage.py"
    stages_path = cdir / j.get("stages", "harness/stages.py")
    metrics_path = cdir / j.get("metrics", "harness/metrics.py")
    for p in (kit, stages_path, metrics_path):
        if not p.resolve().is_file() or not str(p.resolve()).startswith(str(cdir.resolve()) + os.sep):
            raise SystemExit(f"judge file not found inside case: {p}")
    sys.path.insert(0, str(kit.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("abench_stage", kit)
    abench_stage = importlib.util.module_from_spec(spec)
    sys.modules["abench_stage"] = abench_stage
    spec.loader.exec_module(abench_stage)
    stages = abench_stage.load_module(stages_path, f"abench_stages_{case.replace('-', '_')}")
    metrics = abench_stage.load_module(metrics_path, f"abench_metrics_{case.replace('-', '_')}")
    rec["judge_harness_sha256"] = {p.name: sha256_bytes(p.read_bytes()) for p in
                                   (kit, stages_path, metrics_path, kit.parent / "abench_metrics.py")}
    public = _verify_public(case, oj)
    models = _verify_models(oj)
    judge_models = _verify_models(oj, "judge_models")
    hid = oj["data"]["hidden"]
    plain = hidden_store.decrypt_bytes(hidden_store.read_key(key_file), (ROOT / hid["path"]).read_bytes())
    if sha256_bytes(plain) != hid["sha256_plain"]:
        raise SystemExit("hidden split plaintext hash mismatch")
    with np.load(io.BytesIO(plain), allow_pickle=False) as z:
        hidden = {k: np.asarray(z[k]) for k in z.files}
    del plain
    in_keys = list(getattr(stages, "HELD_OUT_KEYS", []))
    missing = [k for k in in_keys if k not in hidden]
    if missing:
        raise SystemExit(f"hidden split lacks input keys {missing}")
    reference = dict(hidden)                                     # stays in this process only
    make_inputs = getattr(stages, "held_out_inputs", None)     # optional: build model inputs in this process
    held_inputs = make_inputs(hidden) if make_inputs else {k: hidden[k] for k in in_keys}
    pre_keys = list(getattr(stages, "HELD_OUT_PRE_KEYS", []))      # inputs a stage gets before its timed phase
    if [k for k in pre_keys if k not in hidden]:
        raise SystemExit(f"hidden split lacks pre-timed input keys {[k for k in pre_keys if k not in hidden]}")
    pre_inputs = {k: hidden[k] for k in pre_keys}
    device = os.environ.get("ABENCH_JUDGE_DEVICE") or j.get("device", "cuda")
    extra_env = None
    params = getattr(stages, "PARAMS", {})

    def run_all(sub: Path, tag: str) -> tuple[list[dict], list[dict], str | None]:
        sub_copy = tmp / f"{tag}_submission"
        shutil.copytree(sub, sub_copy, ignore=shutil.ignore_patterns("__pycache__"))
        held = tmp / "held_out_inputs.npz"
        if held_inputs and not held.exists():
            np.savez(held, **held_inputs)
            held.chmod(0o600)                  # only the parent copies it into post/ after the timed phase
        pre = tmp / "held_out_pre.npz"
        if pre_inputs and not pre.exists():
            np.savez(pre, **pre_inputs)
            pre.chmod(0o600)
        runs, per_seed = [], []
        for i, seed in enumerate(j["seeds"]):
            res = abench_stage.run_pipeline(stages, stages_path, sub_copy, tmp / f"{tag}_run{i}", public=public,
                                            models=models, held_out=held if held_inputs else None, seed=int(seed),
                                            device=device, extra_env=extra_env,
                                            held_pre=pre if pre_inputs else None, judge_models=judge_models,
                                            launcher=launcher)
            row = {"seed_index": i, "status": res["status"], "seconds": res["seconds"],
                   "stages": [{k: v for k, v in s.items() if k not in ("meta",)} | {"meta": s.get("meta")}
                              for s in res["stages"]]}
            if case == "IC-SPD-001" and row["status"] == "finish_time_exceeded":
                row["status"] = "predict_time_exceeded"
            if res["status"] == "ok":
                try:
                    m = metrics.score(res["outputs"], reference, params, res["seconds"])
                    per_seed.append(m)
                    row["metrics"] = m
                    if case == "IC-SPD-001":
                        # Preserve the pre-container speedrun record fields for existing consumers.
                        row.update(res["stages"][0])
                        row["train_seconds"] = res["seconds"]["train"]
                        row["predict_seconds"] = res["seconds"].get("train.finish", 0.0)
                        row["accuracy"] = {name: m[f"accuracy_{name}_mean"]
                                           for name in ("cifar10", "cifar10_1")}
                    ci = _primary_ci(metrics, res, reference, params)
                    if ci:
                        row["primary_ci95"] = ci
                except metrics.OutputError as e:
                    row["status"], row["detail"] = "wrong_output_format", str(e)
            runs.append(row)
            shutil.rmtree(tmp / f"{tag}_run{i}", ignore_errors=True)
            if row["status"] != "ok":
                return runs, per_seed, row["status"]
        return runs, per_seed, None

    try:
        base_agg = None
        if j.get("baseline"):
            bdir = (cdir / j["baseline"]).resolve()
            b_runs, b_seed, b_bad = run_all(bdir, "baseline")
            rec["baseline"] = {"path": j["baseline"], "tree_sha256": tree_hash(bdir), "runs": b_runs}
            if b_bad:
                raise SystemExit(f"baseline failed ({b_bad}); the judge environment is not valid for this case")
            base_agg = _aggregate(b_seed)
            rec["baseline"]["metrics"] = base_agg
        runs, per_seed, bad = run_all(submission, "submission")
        rec["runs"] = runs
        rec["n_hidden"] = ({name: int((hidden["src"] == val).sum())
                            for name, val in j.get("subsets", {"cifar10": 0, "cifar10_1": 1}).items()}
                           if case == "IC-SPD-001" else
                           {k: int(len(v)) for k, v in hidden.items() if hasattr(v, "__len__")})
        if bad:
            rec.update(verdict=bad, metrics=None)
            return rec
        m = metrics.aggregate(per_seed) if hasattr(metrics, "aggregate") else _aggregate(per_seed)
        pm = (getattr(metrics, "PRIMARY", None) or {}).get("metric")
        if pm and len(per_seed) > 1 and all(pm in r for r in per_seed):   # spread of the primary metric over runs
            m[f"{pm}_sd_over_runs"] = round(float(np.std([r[pm] for r in per_seed], ddof=1)), 6)
        if base_agg is not None and hasattr(metrics, "relative"):
            m.update({k: round(float(v), 6) for k, v in metrics.relative(m, base_agg).items()})
        gates = _gates(list(getattr(metrics, "GATES", [])), m)
        m["valid"] = all(g["pass"] for g in gates.values())
        cis = [r["primary_ci95"] for r in runs if r.get("primary_ci95")]
        ci = ({"low": round(float(np.mean([c["low"] for c in cis])), 6), "high": round(float(np.mean([c["high"] for c in cis])), 6),
               "level": 0.95, "method": "percentile bootstrap over held-out examples, mean over runs"} if cis else None)
        rec.update(verdict="evaluated", metrics=m, gates=gates, primary=getattr(metrics, "PRIMARY", None),
                   primary_ci95=ci, n_seeds=len(per_seed))
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(case: str, submission: Path, key_file: Path | None, run_mode: str = "benchmark",
             backend: str | None = None, gpus: str | None = None, image: str | None = None,
             run_id: str | None = None) -> dict[str, Any]:
    oj = load_oj(case)
    j = oj.get("judge", {})
    backend = backend or os.environ.get("ABENCH_JUDGE_BACKEND", "host")
    launcher = None
    locked = False
    lock = None
    if backend == "docker" and run_mode != "smoke":
        from judge_container import DockerStageLauncher
        launcher = DockerStageLauncher.for_case(oj, gpus=gpus, image=image, run_id=run_id)
        image, locked, lock = launcher.image, launcher.locked, getattr(launcher, "lock", None)
    elif backend not in ("host","docker"):
        raise SystemExit(f"unsupported judge backend {backend!r}")
    dependency = _dependency_attestation(image, locked, lock) if launcher else _dependency_attestation()
    rec: dict[str, Any] = {"schema_version": "2.0", "case_id": case, "judge_version": JUDGE_VERSION,
                           "provenance": "judge_real", "run_mode": run_mode,
                           # Format-only rejection, smoke and pending cases execute no submission stage.
                           "judge_isolation": "not_started", "judge_backend_requested": backend,
                           "scorable": run_mode == "benchmark",
                           "judge_dependency": dependency,
                           "submission_tree_sha256": tree_hash(submission),
                           "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if run_mode == "benchmark" and not dependency.get("trusted"):
        rec["scorable"] = False
        rec["scoreability_reason"] = ("judge requires a locked image and matching ABENCH_JUDGE_TOOLS_SHA256"
                                      if launcher else "host judge is a non-scoring calibration transport")
    if run_mode == "benchmark":
        rec["environment"] = _environment_check(case, image, locked) if launcher else _environment_check(case)
        if not rec["environment"].get("matches"):
            rec["scorable"] = False
            why = rec["environment"].get("reason") or "judge environment differs from the profile lock"
            rec["scoreability_reason"] = "; ".join(x for x in (rec.get("scoreability_reason"), why) if x)
    if (oj.get('resources') or {}).get('leaderboard') == 'none':
        rec['scorable'] = False
        rec['scoreability_reason'] = '; '.join(filter(None,[rec.get('scoreability_reason'),'pipeline calibration case']))
    errors = validate_submission_dir(submission, case)
    if errors:
        rec.update(verdict="submission_invalid", errors=errors, metrics=None)
        return rec
    if run_mode == "smoke":
        # A smoke run verifies the view/collection/report contract only.  Do not
        # decrypt hidden data or execute the judge: even a real-looking metric
        # from a smoke run must never be mistaken for a benchmark result.
        rec.update(verdict="smoke_unscored", metrics=None,
                   note="smoke runs are contract checks only and are never scored")
        return rec
    if run_mode != "benchmark":
        rec.update(verdict="submission_invalid", metrics=None,
                   errors=[f"unsupported run_mode={run_mode!r}; only benchmark or smoke is allowed"])
        return rec
    if j.get("kind") in (None, "pending"):
        rec.update(verdict="judge_pending", metrics=None,
                   note="this case's judge is not frozen yet; only the format check was done")
        return rec
    if j.get("activation") == "awaiting_data":
        data = oj.get("data") or {}
        hashed = bool(re.fullmatch(r"[0-9a-f]{64}", str((data.get("hidden") or {}).get("sha256_plain", ""))))
        frozen = (not data.get("synthetic")) and hashed
        # plumbing check of the whole launch path on synthetic data: explicit opt-in, recorded, never scorable
        synthetic_ok = bool(data.get("synthetic")) and hashed and os.environ.get("ABENCH_ALLOW_SYNTHETIC") == "1"
        if not (frozen or synthetic_ok):
            rec.update(verdict="judge_pending", metrics=None,
                       note="this case's data has not been frozen by the operator yet (judge.activation: awaiting_data); "
                            "only the format check was done")
            return rec
        # real data frozen but the case is not activated yet: a calibration run, never scorable
        rec["scorable"] = False
        rec["calibration"] = True
        rec["scoreability_reason"] = "calibration run: case not activated yet (judge.activation: awaiting_data)"
        if synthetic_ok:
            rec["synthetic_data"] = True
            rec["scoreability_reason"] = "plumbing run on SYNTHETIC data (ABENCH_ALLOW_SYNTHETIC=1); metrics are meaningless"
    # submission code runs from here on: decide where it runs before anything is decrypted
    rec["judge_isolation"] = "container" if launcher else "host_soft"
    if launcher:
        rec["judge_image"] = image
        rec["judge_image_digest"] = launcher.digest
        rec["judge_platform"] = launcher.platform
        rec['judge_resource_enforcement'] = launcher.resources_enforced
        if not launcher.resources_enforced:
            rec['scorable'] = False
            rec['scoreability_reason'] = '; '.join(filter(None,[rec.get('scoreability_reason'),'judge resource limits are not enforced']))
        rec["judge_isolation_detail"] = {"backend": "docker", "image": image, "image_locked": locked, "image_lock": lock,
                                         "containers": launcher.records, "inputs": launcher.input_record}
        if launcher.platform != "linux/amd64":
            rec["scorable"] = False
            rec["scoreability_reason"] = "; ".join(filter(None, [rec.get("scoreability_reason"), "non-reference platform"]))
        if getattr(launcher, "gpu_inventory", None):
            rec["judge_gpu_inventory"] = launcher.gpu_inventory
            rec["judge_gpu_class_matches"] = launcher.gpu_class_matches     # reported; does not affect scoring
        if not launcher.hardware_matches:
            rec["scorable"] = False
            reason = getattr(launcher, "hardware_reason", None) or "judge hardware does not match the case resources"
            rec["scoreability_reason"] = "; ".join(filter(None, [rec.get("scoreability_reason"), reason]))
    elif rec.get("scorable"):
        rec["scorable"] = False
        rec["scoreability_reason"] = "; ".join(filter(None, [rec.get("scoreability_reason"), "host_soft transport"]))
    dev = os.environ.get("ABENCH_JUDGE_DEVICE")
    if dev and dev != j.get("device", "cuda"):
        rec["judge_device_override"] = {"oj_yaml": j.get("device", "cuda"), "used": dev}
        rec["scorable"] = False
        rec["scoreability_reason"] = "; ".join(x for x in (rec.get("scoreability_reason"),
                                                            f"judge device overridden to {dev}") if x)
    if launcher:
        rec["_launcher"] = launcher
    if j["kind"] == "harness":
        if key_file is None:
            raise SystemExit("--hidden-key-file is required to evaluate on the hidden split")
        return evaluate_harness(case, submission, key_file, oj, rec)
    raise SystemExit(f"unsupported judge kind {j['kind']!r}; regenerate it as kind: harness")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--hidden-key-file")
    ap.add_argument("--run-mode", choices=["benchmark", "smoke"], default="benchmark")
    ap.add_argument("--out")
    ap.add_argument("--backend", choices=["docker", "host"], default="docker")
    ap.add_argument("--gpus")
    ap.add_argument("--image", help="explicit image for unscored deployment validation")
    a = ap.parse_args()
    if a.backend == "host" and a.run_mode != "smoke":
        ap.error("host backend requires --run-mode smoke; selfcheck is available for local dev evaluation")
    if a.out and Path(a.out).exists():
        raise SystemExit(f"evaluation record already exists and is immutable: {a.out}")
    from runtime.cleanup import sigterm_interrupts
    with sigterm_interrupts():          # a stopped judge still removes its containers and volumes
        rec = evaluate(a.case, Path(a.submission).resolve(), Path(a.hidden_key_file) if a.hidden_key_file else None, a.run_mode, backend=a.backend, gpus=a.gpus, image=a.image)
    text = json.dumps(rec, ensure_ascii=False, indent=2)
    if a.out:
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"evaluation record already exists and is immutable: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if rec.get("verdict") in ("evaluated", "judge_pending", "smoke_unscored") else 1


if __name__ == "__main__":
    sys.exit(main())
