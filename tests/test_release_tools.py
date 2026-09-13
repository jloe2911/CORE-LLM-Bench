from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release import assert_no_family_payload, validate_release  # noqa: E402


class ReleaseValidationTests(unittest.TestCase):
    def test_frozen_release(self) -> None:
        report = validate_release()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["overall_total"], 9032)
        self.assertEqual(report["task_totals"], {"BQA": 5999, "OEQA": 3033})
        self.assertEqual(report["taxonomy_size"], 20)
        self.assertEqual(len(report["instantiated_reasoning_counts"]), 8)

    def test_public_safe_release_selection(self) -> None:
        report = validate_release("public-safe")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["overall_total"], 5272)
        self.assertEqual(report["task_totals"], {"BQA": 3455, "OEQA": 1817})
        self.assertEqual(
            report["dataset_totals"],
            {"Pizza100": 984, "Pizza250": 1232, "OWL2Bench": 3056},
        )

    def test_public_safe_family_artifact_assertion(self) -> None:
        with self.assertRaisesRegex(ValueError, "Family artifact"):
            assert_no_family_payload(ROOT / "final_benchmark")


if __name__ == "__main__":
    unittest.main()
