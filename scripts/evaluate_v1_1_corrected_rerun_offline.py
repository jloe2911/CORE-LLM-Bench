#!/usr/bin/env python3
"""Build and evaluate the unpublished corrected v1.1 matrix, strictly offline.

This program validates the completed minimum-rerun checkpoint and journals,
merges corrected reruns with scientifically reusable frozen observations, and
scores the resulting 81,432 cells.  It imports no API client and cannot issue
model requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:  # direct script execution
    import evaluate_v1_1_final_offline as base
    from oeqa_explanations import _construct_tags
except ModuleNotFoundError:  # package import in tests
    from scripts import evaluate_v1_1_final_offline as base
    from scripts.oeqa_explanations import _construct_tags


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/correction/v1.1.0-semantic-audit"
RERUN = ROOT / "data/output/v1.1.0-minimum-rerun"
BENCHMARK = AUDIT / "core_llm_bench_v1_1_corrected.parquet"
INPUTS = AUDIT / "corrected_model_input_manifest.csv"
DIFFERENTIAL = AUDIT / "observation_differential_audit.csv"
RERUN_MANIFEST = AUDIT / "rerun_manifest.csv"
CHECKPOINT = RERUN / "checkpoint.sqlite3"
RESPONSES = RERUN / "responses.jsonl"
ATTEMPTS = RERUN / "retry_attempts.jsonl"
DEFAULT_OUTPUT = ROOT / "results/v1.1.0-corrected-rerun-review"
EXPECTED_COST = 13.53325078
EXPECTED_REUSED = 64_974
EXPECTED_RERUN_CELLS = 16_458
EXPECTED_REQUESTS = 16_443


def rendered_axiom_tags(text: str) -> tuple[str, ...]:
    tags = list(_construct_tags(text))
    rendered_patterns = (
        ("H", (" SubPropertyOf:",)), ("S", ("Symmetric:",)),
        ("T", ("Transitive:",)), ("F", ("Functional:",)),
        ("J", ("DisjointClasses:", "DisjointProperties:")),
        ("V", ("Reflexive:",)), ("Y", ("Irreflexive:",)),
        ("A", ("Asymmetric:",)), ("Q", ("Equivalent:",)),
    )
    for tag, markers in rendered_patterns:
        if any(marker in text for marker in markers) and tag not in tags:
            tags.append(tag)
    return tuple(tags or ["D"])


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_record(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_strict(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                malformed += 1
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                malformed += 1
                raise RuntimeError(f"Non-object JSONL record at {path}:{line_number}")
            rows.append(value)
    return rows, malformed


def validate_rerun() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    plan = read_csv(RERUN_MANIFEST)
    executable = [row for row in plan if row["deduplicated_request"].lower() == "true"]
    if len(plan) != EXPECTED_RERUN_CELLS or len(executable) != EXPECTED_REQUESTS:
        raise RuntimeError("Rerun plan accounting drift")
    plan_by_request = {row["deduplication_group"]: row for row in executable}
    if len(plan_by_request) != EXPECTED_REQUESTS:
        raise RuntimeError("Duplicate executable rerun request IDs")

    responses, malformed_response_lines = load_jsonl_strict(RESPONSES)
    response_ids = [str(row.get("request_id", "")) for row in responses]
    duplicate_response_records = len(response_ids) - len(set(response_ids))
    if len(responses) != EXPECTED_REQUESTS or duplicate_response_records:
        raise RuntimeError("Response journal is incomplete or duplicated")
    response_by_request = dict(zip(response_ids, responses))

    db = sqlite3.connect(f"file:{CHECKPOINT.as_posix()}?mode=ro", uri=True)
    try:
        integrity = [row[0] for row in db.execute("PRAGMA integrity_check")]
        checkpoint_rows = db.execute(
            "SELECT request_id,model,task_id,representation,input_hash,status,"
            "reserved_usd,charged_usd,COALESCE(uncertain_charged_usd,0),"
            "response_sha256,error FROM requests"
        ).fetchall()
        attempt_statuses = dict(db.execute(
            "SELECT status,COUNT(*) FROM execution_attempts GROUP BY status"
        ).fetchall())
        unresolved_attempts = db.execute(
            "SELECT COUNT(*) FROM execution_attempts WHERE status IN "
            "('in_flight','unresolved','retry_exhausted')"
        ).fetchone()[0]
        reconciliation_events = db.execute(
            "SELECT COUNT(*) FROM reconciliation_events"
        ).fetchone()[0]
    finally:
        db.close()

    if integrity != ["ok"] or len(checkpoint_rows) != EXPECTED_REQUESTS:
        raise RuntimeError("Checkpoint integrity or row-count failure")
    checkpoint = {row[0]: row for row in checkpoint_rows}
    if len(checkpoint) != EXPECTED_REQUESTS:
        raise RuntimeError("Duplicate checkpoint request IDs")

    status_counts = Counter(row[5] for row in checkpoint_rows)
    pending = status_counts.get("pending", 0)
    failed = status_counts.get("failed", 0)
    in_flight = status_counts.get("in_flight", 0)
    unexpected_status = sum(v for k, v in status_counts.items() if k != "completed")
    reserved = sum(float(row[6]) for row in checkpoint_rows)
    charged = sum(float(row[7]) for row in checkpoint_rows)
    uncertain = sum(float(row[8]) for row in checkpoint_rows)

    missing_response = set(checkpoint) - set(response_by_request)
    extra_response = set(response_by_request) - set(checkpoint)
    hash_mismatches = 0
    plan_mismatches = 0
    malformed_observations = 0
    unusable_observations = 0
    route_mismatches = 0
    expected_routes = {
        "GPT-5 mini": ({"openai/gpt-5-mini", "openai/gpt-5-mini-2025-08-07"}, "OpenAI"),
        "Gemini 2.5 Flash-Lite": ({"google/gemini-2.5-flash-lite"}, "Google AI Studio"),
        "Qwen3-30B-A3B-Instruct": ({"qwen/qwen3-30b-a3b-instruct-2507"}, "Alibaba"),
    }
    for request_id, response in response_by_request.items():
        cp = checkpoint[request_id]
        plan_row = plan_by_request.get(request_id)
        if plan_row is None:
            plan_mismatches += 1
            continue
        if digest_record(response) != str(cp[9]):
            hash_mismatches += 1
        expected = (
            plan_row["model"], plan_row["task_id"], plan_row["representation"],
            plan_row["corrected_prompt_hash"],
        )
        observed_checkpoint = (str(cp[1]), str(cp[2]), str(cp[3]), str(cp[4]))
        observed_response = (
            str(response.get("model", "")), str(response.get("task_id", "")),
            str(response.get("representation", "")), str(response.get("input_hash", "")),
        )
        if observed_checkpoint != expected or observed_response != expected:
            plan_mismatches += 1
        models, provider = expected_routes[plan_row["model"]]
        if (str(response.get("returned_model", "")) not in models or
                str(response.get("returned_provider", "")) != provider):
            route_mismatches += 1
        if response.get("status") != "usable":
            malformed_observations += 1
        parsed = response.get("parsed_response") or {}
        if not parsed.get("usable"):
            unusable_observations += 1

    fatal_counts = {
        "pending": pending,
        "failed": failed,
        "in_flight": in_flight,
        "unexpected_checkpoint_status": unexpected_status,
        "malformed_jsonl_lines": malformed_response_lines,
        "duplicate_response_records": duplicate_response_records,
        "missing_response_records": len(missing_response),
        "extra_response_records": len(extra_response),
        "response_hash_mismatches": hash_mismatches,
        "checkpoint_manifest_or_response_mismatches": plan_mismatches,
        "provider_or_model_route_mismatches": route_mismatches,
        "unresolved_attempts": unresolved_attempts,
    }
    if any(fatal_counts.values()) or abs(reserved) > 1e-12 or abs(uncertain) > 1e-12:
        raise RuntimeError(f"Completed rerun validation failed: {fatal_counts}")
    if abs(charged - EXPECTED_COST) > 1e-10:
        raise RuntimeError(f"Checkpoint cost mismatch: {charged:.8f}")

    return response_by_request, {
        "status": "passed",
        "sqlite_integrity": integrity,
        "checkpoint_status_counts": dict(status_counts),
        "completed_requests": len(checkpoint_rows),
        "response_journal_records": len(responses),
        "charged_usd": round(charged, 8),
        "reserved_usd": reserved,
        "uncertain_charged_usd": uncertain,
        **fatal_counts,
        "malformed_accepted_observations": malformed_observations,
        "unusable_malformed_accepted_observations": unusable_observations,
        "historical_attempt_status_counts": attempt_statuses,
        "reconciliation_events": reconciliation_events,
        "response_journal_sha256": base.sha256_file(RESPONSES),
        "checkpoint_sha256": base.sha256_file(CHECKPOINT),
        "attempt_journal_sha256": base.sha256_file(ATTEMPTS),
        "rerun_manifest_sha256": base.sha256_file(RERUN_MANIFEST),
    }


def frozen_observations() -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    for model, path in base.SOURCES.items():
        rows, malformed = load_jsonl_strict(path)
        accepted = [row for row in rows if row.get("status") in base.ACCEPTED]
        source_counts[model] = len(accepted)
        for row in accepted:
            key = (str(row["task_id"]), str(row["representation"]), str(row["model"]))
            if key in result:
                raise RuntimeError(f"Duplicate frozen observation: {key}")
            result[key] = row
        if malformed:
            raise RuntimeError(f"Malformed frozen observation journal: {path}")
    if len(result) != base.EXPECTED_TOTAL:
        raise RuntimeError(f"Frozen observation matrix has {len(result)} cells")
    return result, {"accepted_by_source": source_counts}


def build_matrix(
    rerun_responses: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    differential = read_csv(DIFFERENTIAL)
    if len(differential) != base.EXPECTED_TOTAL:
        raise RuntimeError("Differential audit is not the complete observation matrix")
    rerun_plan = read_csv(RERUN_MANIFEST)
    rerun_by_cell = {
        (row["task_id"], row["representation"], row["model"]): row
        for row in rerun_plan
    }
    if len(rerun_by_cell) != EXPECTED_RERUN_CELLS:
        raise RuntimeError("Rerun manifest has duplicate cells")
    inputs = read_csv(INPUTS)
    input_by_cell = {(row["task_id"], row["representation"]): row for row in inputs}
    if len(input_by_cell) != base.EXPECTED_PER_MODEL:
        raise RuntimeError("Corrected input manifest is incomplete or duplicated")
    benchmark = pd.read_parquet(BENCHMARK)
    questions = {str(row["task_id"]): row for row in benchmark.to_dict(orient="records")}
    if len(questions) != base.EXPECTED_QUESTIONS:
        raise RuntimeError("Corrected benchmark question accounting failed")
    frozen, frozen_audit = frozen_observations()

    matrix: list[dict[str, Any]] = []
    reused = rerun = qualified = 0
    semantic_mismatches = hash_mismatches = missing = duplicates = 0
    seen: set[tuple[str, str, str]] = set()
    for decision in differential:
        key = (decision["task_id"], decision["representation"], decision["model"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        question = questions.get(decision["task_id"])
        canonical_input = input_by_cell.get((decision["task_id"], decision["representation"]))
        if question is None or canonical_input is None:
            missing += 1
            continue
        corrected_semantic_key = str(question["corrected_semantic_key"])
        if (decision["corrected_semantic_key"] != corrected_semantic_key or
                canonical_input["corrected_semantic_key"] != corrected_semantic_key):
            semantic_mismatches += 1
        if decision["corrected_prompt_hash"] != canonical_input["input_hash"]:
            hash_mismatches += 1

        if decision["reuse_frozen_observation"].lower() == "true":
            source = frozen.get(key)
            if source is None:
                missing += 1
                continue
            if str(source.get("input_hash", "")) != decision["original_prompt_hash"]:
                hash_mismatches += 1
            observation = dict(source)
            provenance = "reused_frozen"
            reused += 1
            if decision["reuse_basis"] != "exact_input_hash":
                qualified += 1
        else:
            plan = rerun_by_cell.get(key)
            if plan is None:
                missing += 1
                continue
            source = rerun_responses.get(plan["deduplication_group"])
            if source is None:
                missing += 1
                continue
            observation = dict(source)
            provenance = "corrected_rerun"
            rerun += 1
        observation.update({
            "task_id": decision["task_id"],
            "model": decision["model"],
            "representation": decision["representation"],
            "dataset": str(question["dataset"]),
            "hop": str(question["hop"]),
            "task": str(question["task_group"]),
            "semantic_key": corrected_semantic_key,
            "input_hash": decision["corrected_prompt_hash"],
            "observation_provenance": provenance,
            "reuse_basis": decision["reuse_basis"] if provenance == "reused_frozen" else "",
            "rerun_request_id": (
                rerun_by_cell[key]["deduplication_group"] if provenance == "corrected_rerun" else ""
            ),
        })
        matrix.append(observation)

    fatal = {
        "missing_observations": missing,
        "duplicate_final_cells": duplicates,
        "semantic_mismatches": semantic_mismatches,
        "prompt_hash_mismatches": hash_mismatches,
    }
    if (len(matrix) != base.EXPECTED_TOTAL or len(seen) != base.EXPECTED_TOTAL or
            reused != EXPECTED_REUSED or rerun != EXPECTED_RERUN_CELLS or any(fatal.values())):
        raise RuntimeError(f"Corrected matrix construction failed: {fatal}")
    return matrix, {
        "status": "passed",
        "final_observations": len(matrix),
        "unique_final_cells": len(seen),
        "reused_frozen_observations": reused,
        "corrected_rerun_observations": rerun,
        "qualified_nonexact_prompt_transfers": qualified,
        **fatal,
        **frozen_audit,
        "corrected_benchmark_sha256": base.sha256_file(BENCHMARK),
        "corrected_input_manifest_sha256": base.sha256_file(INPUTS),
        "differential_audit_sha256": base.sha256_file(DIFFERENTIAL),
    }


def aggregate_outputs(scored: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "overall_by_model": base.aggregate(scored, ["model"]),
        "results_by_task": base.aggregate(scored, ["model", "task"]),
        "results_by_representation": base.aggregate(scored, ["model", "representation"]),
        "results_by_dataset": base.aggregate(scored, ["model", "dataset"]),
        "results_by_hop": base.aggregate(scored, ["model", "hop"]),
        "results_full_factorial": base.aggregate(
            scored, ["model", "dataset", "task", "representation", "hop"]
        ),
        "explanation_complexity": base.aggregate(scored, ["model", "task", "complexity_bin"]),
        "explanation_complexity_detailed": base.aggregate(
            scored, ["model", "dataset", "task", "representation", "hop", "complexity_bin"]
        ),
    }


def compare_table(name: str, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_path = ROOT / "results/v1.1.0-final/csv" / f"{name}.csv"
    previous = read_csv(previous_path)
    metrics = [
        "answer_exact_match", "answer_f1", "confidence_correctness_alignment",
        "oeqa_hallucination_rate",
    ]
    dimensions = [key for key in current[0] if key not in {
        "n", "oeqa_n", "oeqa_empty_predictions", "oeqa_generated_answers",
        "oeqa_unsupported_answers", "malformed_accepted", *metrics,
    }]
    old_by_key = {tuple(row.get(dim, "") for dim in dimensions): row for row in previous}
    output: list[dict[str, Any]] = []
    for row in current:
        key = tuple(str(row.get(dim, "")) for dim in dimensions)
        old = old_by_key.get(key)
        if old is None:
            raise RuntimeError(f"Published comparison row missing for {name}: {key}")
        compared: dict[str, Any] = {dim: row.get(dim, "") for dim in dimensions}
        compared["n"] = row["n"]
        for metric in metrics:
            old_value = old.get(metric, "")
            new_value = row.get(metric, "")
            compared[f"published_{metric}"] = old_value
            compared[f"corrected_{metric}"] = new_value
            compared[f"delta_{metric}"] = (
                float(new_value) - float(old_value)
                if old_value not in (None, "") and new_value not in (None, "") else ""
            )
        output.append(compared)
    if len(output) != len(previous):
        raise RuntimeError(f"Published/current row-count mismatch for {name}")
    return output


def corrected_complexity_metadata(benchmark: pd.DataFrame) -> tuple[dict[str, tuple[int, str]], dict[str, Any]]:
    questions = {str(row["task_id"]): row for row in benchmark.to_dict(orient="records")}
    metadata: dict[str, tuple[int, str]] = {}
    mismatches: list[str] = []
    for task_id, question in questions.items():
        basis = question
        if question["task_group"] == "BQA" and str(question["gold_answer"]) == "FALSE":
            positive_id = question.get("positive_task_id")
            if pd.isna(positive_id) or str(int(positive_id)) not in questions:
                raise RuntimeError(f"Missing paired-positive proof metadata for {task_id}")
            basis = questions[str(int(positive_id))]
        answer_groups = basis.get("answer_explanations")
        if isinstance(answer_groups, str):
            answer_groups = json.loads(answer_groups)
        states: set[frozenset[tuple[str, tuple[str, ...]]]] = {frozenset()}
        for group in answer_groups or []:
            alternatives = group.get("alternatives", [])
            if not alternatives:
                raise RuntimeError(f"Missing corrected proof alternative for {task_id}")
            next_states: set[frozenset[tuple[str, tuple[str, ...]]]] = set()
            for state in states:
                current = dict(state)
                for alternative in alternatives:
                    axioms = alternative.get("axioms", [])
                    merged = dict(current)
                    for axiom in axioms:
                        identity = str(axiom.get("axiom", ""))
                        explicit = axiom.get("tag")
                        inferred = ((explicit,) if isinstance(explicit, str) and len(explicit) == 1
                                    else rendered_axiom_tags(identity))
                        primitive = tuple(tag for tag in inferred if tag != "M")
                        if identity in merged and merged[identity] != primitive:
                            raise RuntimeError(
                                f"Inconsistent reconstructed proof tag for {task_id}: "
                                f"{merged[identity]!r} vs {primitive!r} for {identity}"
                            )
                        merged[identity] = primitive
                    next_states.add(frozenset(merged.items()))
            states = next_states
        if not states:
            raise RuntimeError(f"No corrected complete proof states for {task_id}")
        count = min(sum(len(tags) for _, tags in state) for state in states)
        declared = int(question["raw_minimum_complete_primitive_tag_complexity"])
        if count != declared:
            mismatches.append(task_id)
        if question["task_group"] == "BQA":
            complexity_bin = "Low" if count == 1 else "Medium" if count == 2 else "High"
        else:
            complexity_bin = "Low" if count <= 3 else "Medium" if count <= 5 else "High"
        metadata[task_id] = (count, complexity_bin)
    return metadata, {
        "complexity_source": (
            "answer_explanations exact conjunctive/disjunctive reconstruction; "
            "shared axioms deduplicated; primitive tags reconstructed with the "
            "existing deterministic fallback; M excluded"
        ),
        "stale_declared_complexity_mismatches": len(mismatches),
        "stale_declared_complexity_mismatch_task_ids": mismatches,
    }


def score_matrix(
    matrix: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    benchmark = pd.read_parquet(BENCHMARK)
    questions = {str(row["task_id"]): row for row in benchmark.to_dict(orient="records")}
    complexity, complexity_audit = corrected_complexity_metadata(benchmark)
    benchmark = benchmark.copy()
    benchmark["complexity_bin"] = benchmark["task_id"].map(
        lambda value: complexity[str(value)][1]
    )
    scored: list[dict[str, Any]] = []
    for observation in matrix:
        question = questions[str(observation["task_id"])]
        parsed = base.parse_response(base.raw_text(observation), str(question["task_group"]))
        representation = str(observation["representation"])
        gold = base.gold_for_representation(question, representation)
        em, f1, precision, recall = base.answer_set_scores(parsed["answer"], gold)
        hall: float | str = ""
        empty_prediction: int | str = ""
        generated_count: int | str = ""
        unsupported_count: int | str = ""
        if question["task_group"] == "OEQA":
            unsupported_count, generated_count, hall_value = base.hallucination_counts(parsed["answer"], gold)
            empty_prediction = int(generated_count == 0)
            hall = "" if hall_value is None else hall_value
        scored.append({
            "task_id": str(observation["task_id"]),
            "semantic_key": str(question["corrected_semantic_key"]),
            "model": observation["model"],
            "dataset": question["dataset"],
            "task": question["task_group"],
            "representation": representation,
            "hop": question["hop"],
            "prompt_hash": observation["input_hash"],
            "observation_provenance": observation["observation_provenance"],
            "reuse_basis": observation["reuse_basis"],
            "rerun_request_id": observation["rerun_request_id"],
            "complexity": complexity[str(observation["task_id"])][0],
            "complexity_bin": complexity[str(observation["task_id"])][1],
            "primitive_reasoning_tags": "".join(sorted(base.tags(question["primitive_reasoning_tags"]))),
            "observation_status": observation["status"],
            "parser_status": parsed["status"],
            "empty_prediction": int(not parsed["usable"]),
            "confirmed_defective_ar_prompt": int(
                representation == "AR" and
                str(observation["task_id"]) in base.CONFIRMED_DEFECTIVE_AR_TASK_IDS
            ),
            "confidence": parsed["confidence"],
            "answer_exact_match": em,
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "confidence_correctness_alignment": base.confidence_correctness_alignment(parsed["confidence"], f1),
            "oeqa_empty_prediction": empty_prediction,
            "oeqa_generated_answer_count": generated_count,
            "oeqa_unsupported_answer_count": unsupported_count,
            "oeqa_hallucination_rate": hall,
        })
    return scored, benchmark, complexity_audit


def latex_outputs(output: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    specs = {
        "overall_by_model": ([("model", "Model")], "Corrected overall results"),
        "results_by_task": ([("model", "Model"), ("task", "Task")], "Corrected results by task"),
        "results_by_representation": ([("model", "Model"), ("representation", "Representation")], "Corrected results by representation"),
        "results_by_dataset": ([("model", "Model"), ("dataset", "Dataset")], "Corrected results by dataset"),
        "results_by_hop": ([("model", "Model"), ("hop", "Hop")], "Corrected results by hop"),
        "explanation_complexity": ([("model", "Model"), ("task", "Task"), ("complexity_bin", "Complexity")], "Corrected explanation-complexity results"),
    }
    metric_columns = [
        ("n", "N"), ("answer_exact_match", "EM"), ("answer_f1", "F1"),
        ("confidence_correctness_alignment", "Conf.-corr."),
        ("oeqa_hallucination_rate", "OEQA hall."),
    ]
    percent = {"answer_exact_match", "answer_f1", "confidence_correctness_alignment", "oeqa_hallucination_rate"}
    for name, (dimensions, caption) in specs.items():
        text = base.latex_table(
            tables[name], dimensions + metric_columns, caption,
            f"tab:core-v11-corrected-{name.replace('_', '-')}", percent,
        )
        path = output / "latex" / f"{name}.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    global BENCHMARK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    args = parser.parse_args()
    BENCHMARK = args.benchmark.resolve()
    output = args.output_dir.resolve()
    if output in {ROOT / "release/v1.1.0", ROOT / "results/v1.1.0-final"}:
        raise RuntimeError("Refusing to overwrite a published/frozen artifact tree")

    rerun_responses, rerun_audit = validate_rerun()
    matrix, matrix_audit = build_matrix(rerun_responses)
    scored, benchmark, complexity_audit = score_matrix(matrix)
    tables = aggregate_outputs(scored)

    base.write_csv(output / "csv/per_observation_scores.csv", scored)
    base.write_csv(
        output / "audit/malformed_accepted_observations.csv",
        [row for row in scored if row["observation_status"] == "malformed_response"],
    )
    for name, rows in tables.items():
        base.write_csv(output / "csv" / f"{name}.csv", rows)
        comparison = compare_table(name, rows)
        base.write_csv(output / "comparison" / f"{name}_vs_published.csv", comparison)

    stats_rows: list[dict[str, Any]] = []
    for dims in (["dataset", "hop", "task_group"], ["dataset", "hop"], ["task_group"], ["complexity_bin"]):
        counts = benchmark.groupby(dims, dropna=False).size().reset_index(name="questions")
        for row in counts.to_dict(orient="records"):
            stats_rows.append({
                "breakdown": "+".join(dims),
                **{key: row.get(key, "") for key in ("dataset", "hop", "task_group", "complexity_bin")},
                "questions": int(row["questions"]),
            })
    base.write_csv(output / "csv/final_dataset_statistics.csv", stats_rows,
                   ["breakdown", "dataset", "hop", "task_group", "complexity_bin", "questions"])
    latex_outputs(output, tables)

    audit = {
        "status": "passed",
        "offline_only": True,
        "rerun": rerun_audit,
        "matrix": matrix_audit,
        "corrected_proof_metadata": complexity_audit,
        "evaluator": {
            "sageqa_evaluator_sha256": base.SAGEQA_EVALUATOR_CANONICAL_SHA256,
            "sageqa_source_commit": base.SAGEQA_SOURCE_COMMIT,
            "nl_fs_gold_field": "gold_answer",
            "ar_gold_field": "ar_gold_answer",
            "oeqa_hallucination_definition": "unsupported generated normalized answers / generated normalized answers; empty predictions excluded from denominator",
        },
    }
    base.write_json(output / "audit/integrity_audit.json", audit)

    model_lines = [
        f"| {row['model']} | {row['n']:,} | {100*row['answer_exact_match']:.2f} | "
        f"{100*row['answer_f1']:.2f} | {100*row['confidence_correctness_alignment']:.2f} | "
        f"{100*float(row['oeqa_hallucination_rate']):.2f} |"
        for row in tables["overall_by_model"]
    ]
    report = f"""# Offline corrected v1.1 evaluation review

Status: **complete; unpublished review artifact**. No model/API calls are made by this pipeline.

## Integrity

- Checkpoint: {rerun_audit['completed_requests']:,}/{EXPECTED_REQUESTS:,} completed; zero pending, failed, in-flight, duplicate, hash-mismatched, route-mismatched, or unresolved requests.
- The journal contains {rerun_audit['malformed_accepted_observations']:,} schema-nonconformant completed outputs ({rerun_audit['unusable_malformed_accepted_observations']:,} blank/unusable). They are retained as malformed accepted observations in the denominator, consistent with the published evaluation protocol; therefore the requested zero-malformed gate is not satisfied.
- Response journal: {rerun_audit['response_journal_records']:,} unique validated records.
- Locally recorded cost: ${rerun_audit['charged_usd']:.8f}; zero reserved or uncertain charge.
- Final matrix: {matrix_audit['reused_frozen_observations']:,} reusable frozen + {matrix_audit['corrected_rerun_observations']:,} corrected rerun = {matrix_audit['final_observations']:,} unique observations.
- Qualified non-exact-prompt transfers among reused observations: {matrix_audit['qualified_nonexact_prompt_transfers']:,}.

## Overall corrected results (%)

| Model | N | EM | F1 | Confidence-correctness | OEQA hallucination |
|---|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

All primary, factorial, explanation-complexity, and published-comparison tables are under `csv/`, `comparison/`, and `latex/`. Full reasoning-tag results are generated separately from this validated per-observation score matrix.
"""
    (output / "EVALUATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    generated = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json")
    base.write_json(output / "REPRODUCIBILITY_MANIFEST.json", {
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": base.sha256_file(Path(__file__)),
        "offline_only": True,
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): base.sha256_file(path)
            for path in (BENCHMARK, INPUTS, DIFFERENTIAL, RERUN_MANIFEST, CHECKPOINT, RESPONSES, ATTEMPTS)
        },
        "outputs": {
            str(path.relative_to(output)).replace("\\", "/"): base.sha256_file(path)
            for path in generated
        },
    })
    print(json.dumps({
        "status": "complete", "offline_only": True,
        "observations": len(scored), "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
