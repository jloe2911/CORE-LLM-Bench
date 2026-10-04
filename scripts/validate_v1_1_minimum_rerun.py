#!/usr/bin/env python3
"""Offline integrity gates for the corrected benchmark and 16,443-call plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
PUBLISHED = ROOT / "release" / "v1.1.0" / "benchmark" / "core_llm_bench_v1_1.parquet"
PUBLISHED_SHA256 = "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(sha256(PUBLISHED) == PUBLISHED_SHA256, "Archived v1.1.0 Parquet changed")
    original = pd.read_parquet(PUBLISHED)
    corrected = pd.read_parquet(AUDIT / "core_llm_bench_v1_1_corrected.parquet")
    replacements = pd.read_csv(AUDIT / "false_bqa_replacements.csv", dtype=str).fillna("")
    q_audit = pd.read_csv(AUDIT / "question_semantic_audit.csv", dtype=str).fillna("")
    inputs = pd.read_csv(AUDIT / "corrected_model_input_manifest.csv", dtype=str).fillna("")
    diff = pd.read_csv(AUDIT / "observation_differential_audit.csv", dtype=str).fillna("")
    rerun = pd.read_csv(AUDIT / "rerun_manifest.csv", dtype=str).fillna("")
    nl = pd.read_csv(AUDIT / "nl_category_b_reuse_audit.csv", dtype=str).fillna("")
    primary = pd.read_csv(AUDIT / "replacement_openllet_results.tsv", sep="\t", dtype=str).fillna("")
    independent = pd.read_csv(AUDIT / "replacement_structural_results.tsv", sep="\t", dtype=str).fillna("")

    require(len(corrected) == corrected.task_id.nunique() == 9048, "Corrected benchmark membership failed")
    require(((corrected.task_group == "BQA") & (corrected.gold_answer == "TRUE")).sum() == 3016, "TRUE balance failed")
    require(((corrected.task_group == "BQA") & (corrected.gold_answer == "FALSE")).sum() == 3016, "FALSE balance failed")
    require((corrected.task_group == "OEQA").sum() == 3016, "OEQA count failed")
    require(len(replacements) == replacements.task_id.nunique() == replacements.positive_task_id.nunique() == 67,
            "Replacement cardinality or positive preservation failed")
    require(replacements.pair_group_id.nunique() == 67, "Replacement pair groups are not one-to-one")
    for frame, name in ((primary, "Openllet"), (independent, "structural")):
        require(len(frame) == 67 and set(frame.status) == {"ok"}, f"{name} replacement coverage failed")
        require(set(frame.consistent.str.lower()) == {"true"}, f"{name} context consistency failed")
        require(set(frame.entailed.str.lower()) == {"false"}, f"{name} found an entailed replacement")

    original_by_id = original.set_index(original.task_id.astype(str), drop=False)
    corrected_by_id = corrected.set_index(corrected.task_id.astype(str), drop=False)
    protected = ["task_id", "semantic_key", "pair_group_id", "gold_answer", "source_provenance_key", "root_entity"]
    for positive_id in replacements.positive_task_id:
        left, right = original_by_id.loc[positive_id], corrected_by_id.loc[positive_id]
        require(all(str(left[field]) == str(right[field]) for field in protected),
                f"Protected positive member changed: {positive_id}")
        require(str(right.gold_answer) == "TRUE", f"Positive member no longer TRUE: {positive_id}")

    require(len(inputs) == 27144, "Corrected input manifest row count failed")
    require(inputs.groupby(["task_id", "representation"]).size().eq(1).all(), "Corrected input keys are not unique")
    require(set(inputs.task_id) == set(corrected.task_id.astype(str)), "Corrected input manifest misses questions")
    replacement_hashes = replacements.set_index("task_id")
    input_hashes = inputs.set_index(["task_id", "representation"]).input_hash
    for task_id, row in replacement_hashes.iterrows():
        for representation, column in (("NL", "nl_prompt_hash"), ("FS", "fs_prompt_hash"), ("AR", "ar_prompt_hash")):
            require(input_hashes[(task_id, representation)] == row[column], f"Replacement prompt hash drift: {task_id}/{representation}")

    require(len(q_audit) == 9048 and q_audit.corrected_openllet_semantically_valid.str.lower().eq("true").all(),
            "Not every corrected question passes the primary semantic gate")
    require(len(diff) == 81432 and diff.observation_input_hash_matches_original.str.lower().eq("true").all(),
            "Frozen observation identity failed")
    reused = diff[diff.reuse_frozen_observation.str.lower() == "true"]
    rerun_cells = diff[diff.rerun_required.str.lower() == "true"]
    require(len(reused) == 64974 and len(rerun_cells) == 16458, "Reuse/rerun partition failed")
    require(len(reused) + len(rerun_cells) == len(diff), "Observation partition is incomplete")
    qualified = reused[reused.reuse_basis == "qualified_non_exact_prompt_transfer"]
    require(len(qualified) == 15855 and qualified.exact_corrected_prompt_match.str.lower().eq("false").all(),
            "Category-B qualified reuse accounting failed")
    require(len(nl) == 5352 and (nl.category == "B").sum() == 5285 and (nl.category == "A").sum() == 67,
            "NL category audit failed")
    require(len(rerun) == 16458 and rerun.expected_api_call_count.astype(int).sum() == 16443,
            "Executable rerun call accounting failed")
    executable = rerun[rerun.deduplicated_request.str.lower() == "true"]
    require(len(executable) == executable.deduplication_group.nunique() == 16443,
            "Executable request IDs are not unique")
    require(executable.groupby("model").size().to_dict() == {
        "GPT-5 mini": 5481, "Gemini 2.5 Flash-Lite": 5481, "Qwen3-30B-A3B-Instruct": 5481,
    }, "Per-model request accounting failed")

    report = {
        "status": "PASS",
        "archived_v1_1_parquet_sha256": PUBLISHED_SHA256,
        "corrected_questions": 9048,
        "balanced_true_false_pairs": 3016,
        "replacement_false_questions": 67,
        "primary_and_independent_non_entailments": 67,
        "corrected_model_inputs": 27144,
        "frozen_observations_validated": 81432,
        "frozen_observations_reused": 64974,
        "category_b_nl_prompts": 5285,
        "rerun_observation_cells": 16458,
        "deduplicated_paid_calls": 16443,
        "paid_api_calls_made": 0,
    }
    (AUDIT / "minimum_rerun_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
