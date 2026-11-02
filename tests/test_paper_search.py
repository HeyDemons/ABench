"""Offline paper search: index build with the cutoff filter, search/show through the agent CLI, and the hash-chained
retrieval log. Uses a synthetic snapshot written to a temp dir (no real paper metadata is stored in the repository)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "paper_search"
sys.path.insert(0, str(TOOLS))
import build_paper_index  # noqa: E402


def has_fts5() -> bool:
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(a)")
        return True
    except sqlite3.OperationalError:
        return False


def rec(pid, title, abstract, cats, created):
    return {"id": pid, "title": title, "authors": "A. Author and B. Author", "abstract": abstract, "categories": cats,
            "versions": [{"version": "v1", "created": created}, {"version": "v2", "created": "Mon, 1 Jan 2030 00:00:00 GMT"}]}


SNAPSHOT = [
    rec("2101.00001", "Gated recurrent state updates for linear attention",
        "We study decay gates for linear attention and associative recall.", "cs.LG cs.CL", "Mon, 4 Jan 2021 10:00:00 GMT"),
    rec("2201.00002", "Delta rule memories", "A delta-rule state update improves associative recall in linear attention.",
        "cs.LG", "Tue, 4 Jan 2022 10:00:00 GMT"),
    rec("2607.00003", "Linear attention after the cutoff", "Associative recall with linear attention, published late.",
        "cs.LG", "Wed, 1 Jul 2026 09:00:00 GMT"),
    rec("2102.00004", "Galaxy rotation curves", "Linear attention is not mentioned; astrophysics only.", "astro-ph.GA",
        "Mon, 1 Feb 2021 10:00:00 GMT"),
]


@unittest.skipUnless(has_fts5(), "SQLite without FTS5")
class PaperSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        snap = d / "snapshot.json"
        snap.write_text("\n".join(json.dumps(r) for r in SNAPSHOT) + "\n", encoding="utf-8")
        self.index = d / "papers.sqlite"
        self.meta = build_paper_index.build(snap, self.index, "2026-07-01", ["cs."])
        self.log = d / "ws" / ".abench" / "retrieval_log.jsonl"
        self.env = {**os.environ, "ABENCH_PAPERS_INDEX": str(self.index), "ABENCH_WORKSPACE": str(d / "ws")}
        self.env.pop("ABENCH_RETRIEVAL_LOG", None)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(TOOLS / "abench_papers.py"), *args], capture_output=True, text=True,
                              env=env or self.env)

    def test_index_applies_cutoff_and_categories(self):
        self.assertEqual(self.meta["papers"], "2")          # one paper is after the cutoff, one is not in cs.*
        self.assertEqual(self.meta["cutoff"], "2026-07-01")

    def test_search_show_and_log(self):
        r = self.cli("search", "associative recall linear attention", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        ids = [x["id"] for x in json.loads(r.stdout)]
        self.assertEqual(set(ids), {"2101.00001", "2201.00002"})
        r = self.cli("show", "arXiv:2201.00002v1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Delta rule memories", r.stdout)
        self.assertIn("2022-01-04", r.stdout)
        self.assertEqual(self.cli("show", "2607.00003").returncode, 1)     # after the cutoff: never shown
        lines = self.log.read_bytes().splitlines()
        self.assertEqual(len(lines), 3)
        entries = [json.loads(x) for x in lines]
        self.assertEqual([e["cmd"] for e in entries], ["search", "show", "show"])
        self.assertEqual(entries[0]["prev_sha256"], "0" * 64)
        for prev, e in zip(lines, entries[1:]):
            self.assertEqual(e["prev_sha256"], hashlib.sha256(prev).hexdigest())

    def test_env_cutoff_is_stricter(self):
        env = {**self.env, "ABENCH_CUTOFF": "2022-01-01"}
        r = self.cli("search", "linear attention", "--json", env=env)
        self.assertEqual([x["id"] for x in json.loads(r.stdout)], ["2101.00001"])

    def test_query_syntax_is_safe(self):
        r = self.cli("search", 'recall" OR (NEAR* -- ;DROP', "--json")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_runner_collects_log(self):
        """A smoke run whose 'agent' calls the tool twice: the runner exposes it in the view and keeps the log."""
        res = Path(self.tmp.name) / "runs"
        cmd = ('sh -c \'python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "associative recall" >/dev/null; '
               'python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" show 2101.00001 >/dev/null\'')
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "sandbox_runner.py"), "--case", "A-LIN-002",
                            "--harness", "command", "--command", cmd, "--backend", "host", "--smoke", "--papers-index", str(self.index),
                            "--result-root", str(res), "--run-id", "papers-test", "--timeout-seconds", "60"],
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                if not k.startswith("ABENCH_")})
        run = res / "command" / "llm_opt" / "A-LIN-002" / "papers-test"      # runs/<harness>[-<model>]/<category>/<case>/<run>
        rec = json.loads((run / "agent_run.json").read_text())
        self.assertEqual(rec["retrieval_log"]["entries"], 2, r.stderr[-2000:])
        self.assertTrue(rec["retrieval_log"]["chain_ok"])
        self.assertEqual(rec["paper_index"]["cutoff"], "2026-07-01")
        self.assertIn("tools/abench_papers.py", rec["agent_view"]["files"])
        log = (run / "retrieval_log.jsonl").read_text().splitlines()
        self.assertEqual(set(json.loads(log[0])["results"]), {"2101.00001", "2201.00002"})


if __name__ == "__main__":
    unittest.main()
