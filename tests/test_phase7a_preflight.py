from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase7a_preflight as phase7a  # noqa: E402
import run_v1_1_experiments as runner  # noqa: E402
import validate_v1_1_experiment_completeness as completeness  # noqa: E402


STAGE = ROOT / "release" / "v1.1.0-staging"
PREFLIGHT = ROOT / "release" / "v1.1.0-preflight"


class Phase7APreflightTests(unittest.TestCase):
    def test_manifest_integrity_covers_every_experiment_cell_once(self) -> None:
        config = json.loads(
            (STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8")
        )
        report = runner.audit_manifest_integrity(config)
        self.assertEqual(report["pending_rows"], 9048 * 3 * 3)

    def test_duplicate_audit_finds_the_fatal_semantic_conflicts(self) -> None:
        report = json.loads((STAGE / "validation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(
            sum(value["incompatible_groups"] for value in report["duplicate_audit"].values()),
            0,
        )
        details = list(
            csv.DictReader(
                (STAGE / "input_equivalence_groups.csv").open(encoding="utf-8")
            )
        )
        self.assertEqual(len(details), 24)

    def test_response_parser_preserves_current_contract(self) -> None:
        parsed = phase7a.parse_response("ANSWER: TRUE\nCONFIDENCE: 0.9", "BQA")
        self.assertTrue(parsed["usable"])
        self.assertEqual(parsed["status"], "requested_schema_conformant")
        fallback = phase7a.parse_response("ANSWER: ClassA", "OEQA")
        self.assertEqual(fallback["confidence"], 0.5)
        self.assertEqual(
            fallback["status"], "accepted_by_current_parser_with_confidence_default"
        )
        self.assertFalse(phase7a.parse_response("[ERROR] timeout", "BQA")["usable"])

    def test_configuration_is_fail_closed(self) -> None:
        config = json.loads(
            (STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8")
        )
        self.assertFalse(config["ready_for_execution"])
        self.assertFalse(config["execution_authorized"])
        with self.assertRaises(RuntimeError):
            runner.validate_config(config, execute=True)
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            runner.validate_config(
                config, execute=True, selected_model_ids={"gpt-5-mini-2025-08-07"}
            )
        runner.validate_config(
            config,
            execute=True,
            selected_model_ids={"google/gemini-2.5-flash-lite"},
        )
        runner.validate_config(config, execute=False)

    def test_partial_execution_preflight_is_exact(self) -> None:
        config = json.loads(
            (STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8")
        )
        report = runner.validate_frozen_preflight(config)
        self.assertEqual(report["membership_rows"], 9048)
        self.assertEqual(report["representations"], ["NL", "FS", "AR"])
        self.assertEqual(set(report["pending_by_model_id"].values()), {27144})
        self.assertEqual(report["incompatible_input_gold_groups"], 0)

    def test_canary_rejects_an_empty_checkpoint(self) -> None:
        absent = PREFLIGHT / ".absent_canary.jsonl"
        absent.unlink(missing_ok=True)
        with self.assertRaisesRegex(ValueError, "Canary validation failed"):
            runner.validate_canary(absent, "google/gemini-2.5-flash-lite")

    def test_request_fingerprint_binds_input_config_and_model(self) -> None:
        row = {"input_hash": "a" * 64}
        model = {"model_id": "example", "temperature": 0}
        first = runner.request_fingerprint(row, model)
        self.assertEqual(first, runner.request_fingerprint(row, model))
        self.assertNotEqual(
            first,
            runner.request_fingerprint({"input_hash": "b" * 64}, model),
        )

    def test_checkpoint_resume_never_accepts_duplicate_terminal_rows(self) -> None:
        record = {
            "task_id": "1",
            "representation": "NL",
            "model": "GPT-5 mini",
            "status": "usable",
        }
        path = PREFLIGHT / ".test_observations.jsonl"
        path.unlink(missing_ok=True)
        try:
            runner.append_checkpoint(path, record)
            self.assertEqual(len(runner.load_terminal_keys(path)), 1)
            runner.append_checkpoint(path, record)
            with self.assertRaises(ValueError):
                runner.load_terminal_keys(path)
        finally:
            path.unlink(missing_ok=True)

    def test_completeness_validator_blocks_empty_new_run(self) -> None:
        absent = PREFLIGHT / ".absent_test_observations.jsonl"
        absent.unlink(missing_ok=True)
        report = completeness.validate(absent)
        self.assertEqual(report["reused_responses"], 0)
        self.assertEqual(report["newly_generated_responses"], 0)
        self.assertEqual(report["missing_observations"], 81432)
        self.assertFalse(report["final_metrics_allowed"])


if __name__ == "__main__":
    unittest.main()
