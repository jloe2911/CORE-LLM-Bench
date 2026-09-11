from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release import validate_release  # noqa: E402


class ReleaseValidationTests(unittest.TestCase):
    def test_frozen_release(self) -> None:
        report = validate_release()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["overall_total"], 9032)
        self.assertEqual(report["task_totals"], {"BQA": 5999, "OEQA": 3033})
        self.assertEqual(report["taxonomy_size"], 20)
        self.assertEqual(len(report["instantiated_reasoning_counts"]), 8)


if __name__ == "__main__":
    unittest.main()
