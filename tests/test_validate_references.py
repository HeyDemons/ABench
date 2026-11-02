#!/usr/bin/env python3
"""Static and mocked-network regression tests for reference integrity checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import validate_references  # noqa: E402


REPORT = """<article><section id='bib' class='ltx_bibliography'><ul>
<li id='bib.good' class='ltx_bibitem'>Good</li>
</ul></section><p><a href='#bib.good'>cite</a></p></article>"""


class ReferenceIntegrityTest(unittest.TestCase):
    def _write(self, ref_text: str) -> Path:
        root = Path(tempfile.mkdtemp())
        (root / "report").mkdir()
        (root / "report/index.html").write_text(REPORT, encoding="utf-8")
        path = root / "references.yaml"
        path.write_text(ref_text, encoding="utf-8")
        return path

    def test_static_valid_and_malformed_id(self) -> None:
        valid = self._write("""schema: abench.references/v2\nreferences:\n  - id: good\n    title: A paper\n    year: 2020\n    used_for: context\n    arxiv: "1234.56789"\n""")
        self.assertEqual(validate_references.check_file(valid, False, False, 1.0)["errors"], [])
        bad = self._write("""references:\n  - id: good\n    title: A paper\n    year: 2020\n    used_for: context\n    arxiv: "2202.05846-wrong"\n""")
        self.assertTrue(any("malformed arXiv" in x for x in validate_references.check_file(bad, False, False, 1.0)["errors"]))

    def test_title_mismatch_is_always_an_error(self) -> None:
        path = self._write("""references:\n  - id: good\n    title: Claimed optimizer title\n    year: 2020\n    used_for: context\n    arxiv: "1234.56789"\n""")
        mismatch = {"id": "good", "arxiv": "1234.56789", "registered_title": "Claimed optimizer title",
                    "canonical_title": "A completely different paper", "title_match": False}
        with patch.object(validate_references, "check_arxiv", return_value=mismatch):
            result = validate_references.check_file(path, True, False, 1.0)
        self.assertTrue(any("title mismatch" in x for x in result["errors"]))


class ReportStructureTest(unittest.TestCase):
    """The bibliography counts by its ltx_bibitem entries, whatever class its section carries."""
    def check(self, html: str) -> list[str]:
        import shutil
        import validate_submission
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        shutil.copytree(ROOT / "cases/RS-SEQ-001/example_submission", root / "sub")
        (root / "sub/report/index.html").write_text(html, encoding="utf-8")
        errors: list[str] = []
        oj = validate_submission.load_oj("RS-SEQ-001")
        validate_submission.validate_report(root / "sub", "RS-SEQ-001", validate_submission.load_profiles(), errors, oj)
        return [e for e in errors if "arXiv HTML structure" in e]

    def test_bibliography_in_an_ordinary_section_is_accepted_but_entries_are_required(self) -> None:
        html = (ROOT / "cases/RS-SEQ-001/example_submission/report/index.html").read_text(encoding="utf-8")
        self.assertIn('class="ltx_bibliography"', html)
        self.assertEqual(self.check(html), [])
        self.assertEqual(self.check(html.replace('class="ltx_bibliography"', 'class="ltx_section"')), [])
        self.assertEqual(self.check(html.replace("ltx_bibitem", "ltx_item")),
                         ["report/index.html lacks arXiv HTML structure: .ltx_bibitem"])


if __name__ == "__main__":
    unittest.main()
