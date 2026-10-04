#!/usr/bin/env python3
"""Freeze the validated minimum-rerun package with file SHA-256 hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
OUTPUT_JSON = AUDIT / "artifact_freeze_manifest.json"
OUTPUT_SUMS = AUDIT / "SHA256SUMS"
FILES = [
    "README.md", "JOURNAL_METHOD_QUALIFICATION.md", "model_revalidation_2026-09-28.json",
    "core_llm_bench_v1_1_corrected.parquet", "corrected_model_input_manifest.csv",
    "false_bqa_replacements.csv", "replacement_generation_report.json",
    "replacement_reasoner_input.tsv", "replacement_openllet_results.tsv",
    "replacement_structural_results.tsv", "question_semantic_audit.csv",
    "nl_category_b_reuse_audit.csv", "observation_differential_audit.csv",
    "rerun_manifest.csv", "model_identity_audit.csv", "semantic_correction_report.json",
    "minimum_rerun_validation_report.json",
]
SCRIPTS = [
    "scripts/prepare_v1_1_false_replacements.py", "scripts/finalize_v1_1_semantic_audit.py",
    "scripts/validate_v1_1_minimum_rerun.py", "scripts/run_v1_1_minimum_rerun.py",
    "scripts/freeze_v1_1_minimum_rerun.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [AUDIT / name for name in FILES]
    paths += sorted((AUDIT / "replacement_contexts").glob("*.ttl"))
    paths += [ROOT / name for name in SCRIPTS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze missing files: {missing}")
    records = []
    for path in paths:
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "status": "frozen-validated-no-api-execution",
        "freeze_date": date.today().isoformat(),
        "published_v1_1_modified": False,
        "paid_api_calls_made": 0,
        "corrected_questions": 9048,
        "replacement_false_bqa": 67,
        "executable_calls": 16443,
        "files": records,
    }
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    OUTPUT_SUMS.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"status": "PASS", "frozen_files": len(records),
                      "manifest": str(OUTPUT_JSON), "sha256s": str(OUTPUT_SUMS)}, indent=2))


if __name__ == "__main__":
    main()
