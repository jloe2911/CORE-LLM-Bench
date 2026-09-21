#!/usr/bin/env python3
"""Require exactly one usable observation for every frozen v1.1 experiment cell."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
sys.path.insert(0, str(ROOT / "scripts"))
from phase7a_preflight import read_csv, write_json  # noqa: E402


EXPECTED_MATRIX = 81432
MODELS = ("GPT-5 mini", "Gemini 2.5 Flash-Lite", "Qwen3-30B-A3B-Instruct")
ROUTING_CONTRACT = {
    "GPT-5 mini": ("OpenAI", "OPENAI_API_KEY"),
    "Gemini 2.5 Flash-Lite": ("OpenRouter", "OPENROUTER_API_KEY"),
    "Qwen3-30B-A3B-Instruct": ("OpenRouter", "OPENROUTER_API_KEY"),
}
REPRESENTATIONS = ("NL", "FS", "AR")
PHASE7D = ROOT / "release" / "v1.1.0-phase7d"


def key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["task_id"]), str(record["representation"]), str(record["model"])


def load_new(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(new_path: Path) -> dict[str, Any]:
    manifest = read_csv(STAGE / "primary_experiment_manifest.csv")
    attempts = load_new(new_path)
    expected = {key(row) for row in manifest}
    if len(manifest) != EXPECTED_MATRIX or len(expected) != EXPECTED_MATRIX:
        raise ValueError("Primary experiment manifest is not a unique 81,432-cell matrix")
    terminal_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    technical_failures: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in attempts:
        if (record.get("api_provider"), record.get("credential_source")) != ROUTING_CONTRACT[record["model"]]:
            raise ValueError(f"Result routing violation for {record['model']}")
        record_key = key(record)
        if record["status"] in {"usable", "malformed_response"}:
            terminal_by_key[record_key].append(record)
        elif record["status"] == "technical_failure":
            technical_failures[record_key] = record

    duplicate_new = {k: v for k, v in terminal_by_key.items() if len(v) > 1}
    usable_new = {
        k for k, values in terminal_by_key.items()
        if len(values) == 1 and values[0]["status"] == "usable"
    }
    malformed = {
        k for k, values in terminal_by_key.items()
        if len(values) == 1 and values[0]["status"] == "malformed_response"
    }
    usable = usable_new
    missing = expected - usable - malformed
    unresolved = set(technical_failures) & missing
    unexpected = set(terminal_by_key) - expected
    passed = not any(
        (
            duplicate_new, malformed, unresolved,
            missing, unexpected,
        )
    ) and len(usable) == EXPECTED_MATRIX
    return {
        "status": "complete" if passed else "incomplete",
        "expected_observations": EXPECTED_MATRIX,
        "usable_observations": len(usable),
        "reused_responses": 0,
        "newly_generated_responses": len(usable_new),
        "unresolved_technical_failures": len(unresolved),
        "malformed_responses": len(malformed),
        "duplicate_observations": len(duplicate_new),
        "missing_observations": len(missing),
        "unexpected_observations": len(unexpected),
        "final_metrics_allowed": passed,
        "sample_unresolved_technical_failures": sorted(unresolved)[:20],
        "sample_malformed_responses": sorted(malformed)[:20],
        "sample_duplicate_observations": sorted(
            set(duplicate_new)
        )[:20],
        "sample_missing_observations": sorted(missing)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-observations",
        type=Path,
        default=PHASE7D / "responses" / "new_observations.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PHASE7D / "experiment_completeness_report.json",
    )
    args = parser.parse_args()
    report = validate(args.new_observations)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
