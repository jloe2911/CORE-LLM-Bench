#!/usr/bin/env python3
"""Offline Phase 7A integrity, duplicate-input, and response-reuse preflight.

This module never imports or initializes an API client.  It treats the Phase 6
materialization as immutable input and writes only additional Phase 7A staging
artifacts beside it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
PREFLIGHT = ROOT / "release" / "v1.1.0-preflight"
LEGACY_RESULTS = ROOT / "data" / "output" / "final_benchmark_llm_results"
PHASE5 = ROOT / "data" / "output_v1_1_staging" / "phase5"
MODELS = (
    "GPT-5 mini",
    "Gemini 2.5 Flash-Lite",
    "Qwen3-30B-A3B-Instruct",
)
REPRESENTATIONS = ("NL", "FS", "AR")
EXPECTED_TASKS = 9048
EXPECTED_MATRIX = EXPECTED_TASKS * len(MODELS) * len(REPRESENTATIONS)
EXPECTED_PENDING = 68678
EXPECTED_REUSED = 12754
CONFIG_VERSION = "core-llm-bench-v1.1-experiment-config-1"
PROMPT_TEMPLATE_VERSION = "api_calls.create_context_specific_prompt@24c4520"

sys.path.insert(0, str(ROOT / "scripts"))
import phase5_freeze_membership as phase5  # noqa: E402
import phase6_materialize_release as phase6  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_gold(row: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    if row["task_group"] == "BQA":
        return "BQA", (row["gold_answer"].strip().upper(),)
    answers = tuple(
        sorted(part.strip() for part in row["gold_answer"].split(";") if part.strip())
    )
    return "OEQA", answers


def parse_response(response: str, task_group: str) -> dict[str, Any]:
    """Mirror the current evaluator without weakening its historical contract."""

    text = str(response or "").strip()
    if not text:
        return {"usable": False, "status": "empty", "answer": "", "confidence": None}
    upper = text.upper()
    error_markers = (
        "[ERROR]", "ERROR:", "INSUFFICIENT CREDIT", "INSUFFICIENT BALANCE",
        "PAYMENT REQUIRED", "QUOTA EXCEEDED", "RATE LIMIT EXCEEDED",
    )
    if upper.startswith(error_markers) or any(marker in upper for marker in error_markers[2:]):
        return {
            "usable": False,
            "status": "api_or_error_placeholder",
            "answer": "",
            "confidence": None,
        }
    answer_match = re.search(r"ANSWER:\s*([^\n\r]+)", text, re.IGNORECASE)
    answer = (
        answer_match.group(1).strip()
        if answer_match
        else (text.splitlines()[0].strip() if text.splitlines() else text)
    )
    if task_group == "BQA":
        lowered = answer.lower()
        if "true" in lowered:
            answer = "TRUE"
        elif "false" in lowered:
            answer = "FALSE"
    confidence_match = re.search(
        r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE
    )
    confidence = 0.5
    explicit_confidence = False
    if confidence_match:
        try:
            confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
            explicit_confidence = True
        except ValueError:
            pass
    status = (
        "requested_schema_conformant"
        if answer_match and explicit_confidence
        else "accepted_by_current_parser_with_confidence_default"
        if answer_match
        else "accepted_by_current_parser_first_line_fallback"
    )
    return {
        "usable": bool(answer),
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "explicit_answer_field": bool(answer_match),
        "explicit_confidence_field": explicit_confidence,
    }


def audit_manifest_integrity() -> dict[str, Any]:
    inputs = read_csv(STAGE / "model_input_manifest.csv")
    pending = read_csv(STAGE / "pending_model_runs.csv")
    task_rows = read_csv(STAGE / "task_id_mapping.csv")
    valid_ids = {str(value) for value in range(1, EXPECTED_TASKS + 1)}
    task_ids = {row["new_task_id"] for row in task_rows}
    if task_ids != valid_ids:
        raise ValueError("Task IDs are not exactly 1..9048")
    if len(inputs) != EXPECTED_TASKS * len(REPRESENTATIONS):
        raise ValueError(f"Expected 27,144 input rows, found {len(inputs)}")
    if len(pending) != EXPECTED_PENDING:
        raise ValueError(f"Expected {EXPECTED_PENDING} pending rows, found {len(pending)}")

    input_by_cell: dict[tuple[str, str], dict[str, str]] = {}
    assignments: dict[tuple[str, str, str], str] = {}
    reusable_cells: set[tuple[str, str, str]] = set()
    for row in inputs:
        task_id = row["task_id"]
        representation = row["representation"]
        if task_id not in valid_ids or representation not in REPRESENTATIONS:
            raise ValueError(f"Invalid input cell: {task_id}/{representation}")
        cell = (task_id, representation)
        if cell in input_by_cell:
            raise ValueError(f"Duplicate input manifest cell: {cell}")
        input_by_cell[cell] = row
        statuses = json.loads(row["reuse_status_by_model"])
        if set(statuses) != set(MODELS):
            raise ValueError(f"Invalid model status set for {cell}: {set(statuses)}")
        for model, status in statuses.items():
            experiment_cell = (task_id, representation, model)
            if status not in {"reusable", "rerun required"}:
                raise ValueError(f"Invalid assignment status {status}: {experiment_cell}")
            assignments[experiment_cell] = status
            if status == "reusable":
                reusable_cells.add(experiment_cell)

    pending_cells: set[tuple[str, str, str]] = set()
    for row in pending:
        task_id, representation, model = (
            row["task_id"], row["representation"], row["model"]
        )
        cell = (task_id, representation, model)
        if task_id not in valid_ids or representation not in REPRESENTATIONS or model not in MODELS:
            raise ValueError(f"Invalid pending cell: {cell}")
        if cell in pending_cells:
            raise ValueError(f"Duplicate pending cell: {cell}")
        pending_cells.add(cell)
        manifest = input_by_cell[(task_id, representation)]
        if row["input_hash"] != manifest["input_hash"]:
            raise ValueError(f"Pending/input hash mismatch: {cell}")
        if assignments[cell] != "rerun required":
            raise ValueError(f"Reusable cell appears in pending manifest: {cell}")

    expected_cells = {
        (str(task_id), representation, model)
        for task_id in range(1, EXPECTED_TASKS + 1)
        for representation in REPRESENTATIONS
        for model in MODELS
    }
    if set(assignments) != expected_cells:
        raise ValueError("Model-input assignments do not cover the complete matrix")
    if pending_cells | reusable_cells != expected_cells:
        raise ValueError("Pending and reusable cells do not cover the matrix")
    if pending_cells & reusable_cells:
        raise ValueError("Pending and reusable assignments overlap")
    if len(reusable_cells) != EXPECTED_REUSED:
        raise ValueError(f"Expected {EXPECTED_REUSED} reused cells, found {len(reusable_cells)}")
    return {
        "input_rows": len(inputs),
        "pending_rows": len(pending),
        "reusable_rows": len(reusable_cells),
        "expected_matrix_rows": len(expected_cells),
        "valid_task_ids": True,
        "valid_models_and_representations": True,
        "pending_hashes_match": True,
        "pending_reuse_overlap": 0,
        "missing_experiment_cells": 0,
        "multiply_assigned_experiment_cells": 0,
    }


def audit_duplicate_hashes() -> dict[str, Any]:
    inputs = read_csv(STAGE / "model_input_manifest.csv")
    task_by_id = {
        row["new_task_id"]: row for row in read_csv(STAGE / "task_id_mapping.csv")
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inputs:
        grouped[row["input_hash"]].append(row)

    detail_rows: list[dict[str, Any]] = []
    same_rep_groups = Counter()
    same_rep_rows = Counter()
    cross_rep_groups = 0
    cross_rep_rows: set[tuple[str, str]] = set()
    incompatible_groups = 0
    compatible_groups = 0
    for input_hash, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        representations = {row["representation"] for row in rows}
        cross_rep = len(representations) > 1
        if cross_rep:
            cross_rep_groups += 1
            cross_rep_rows.update((row["task_id"], row["representation"]) for row in rows)
        else:
            representation = rows[0]["representation"]
            same_rep_groups[representation] += 1
            same_rep_rows[representation] += len(rows)
        semantics = {normalized_gold(task_by_id[row["task_id"]]) for row in rows}
        compatible = len(semantics) == 1
        incompatible_groups += int(not compatible)
        compatible_groups += int(compatible)
        for row in rows:
            task = task_by_id[row["task_id"]]
            if not compatible:
                classification = "incompatible_gold_semantics_failure"
            elif row["representation"] == "AR":
                classification = "abstract_equivalence_candidate_requires_review"
            elif row["representation"] == "FS":
                classification = "formal_duplicate_candidate_requires_review"
            else:
                classification = "natural_language_duplicate_candidate_requires_review"
            detail_rows.append(
                {
                    "input_hash": input_hash,
                    "representation": row["representation"],
                    "task_id": row["task_id"],
                    "dataset": task["dataset"],
                    "hop": task["hop"],
                    "task_group": task["task_group"],
                    "gold_answer": task["gold_answer"],
                    "gold_semantics_compatible": str(compatible).lower(),
                    "classification": classification,
                    "duplicate_group_size": len(rows),
                }
            )
    write_csv(
        PREFLIGHT / "duplicate_input_hash_audit.csv",
        detail_rows,
        [
            "input_hash", "representation", "task_id", "dataset", "hop",
            "task_group", "gold_answer", "gold_semantics_compatible",
            "classification", "duplicate_group_size",
        ],
    )
    report = {
        "scope": "exact evaluated input hashes, audited independently within each model",
        "note": "The prompt hash is model-independent, so each count applies identically to all three models.",
        "same_representation": {
            representation: {
                "duplicate_hash_groups": same_rep_groups[representation],
                "rows_involved": same_rep_rows[representation],
            }
            for representation in REPRESENTATIONS
        },
        "across_representations": {
            "duplicate_hash_groups": cross_rep_groups,
            "rows_involved": len(cross_rep_rows),
        },
        "compatible_duplicate_hash_groups": compatible_groups,
        "incompatible_duplicate_hash_groups": incompatible_groups,
        "automatic_deduplication_performed": False,
        "scientific_review_required": bool(detail_rows),
        "fatal_incompatible_gold_semantics": incompatible_groups > 0,
    }
    write_json(PREFLIGHT / "duplicate_input_hash_report.json", report)
    return report


def historical_run_config_summary() -> dict[str, Any]:
    files = sorted(LEGACY_RESULTS.rglob("run_config.json"))
    field_values: dict[str, Counter[str]] = defaultdict(Counter)
    model_identifiers: set[tuple[str, str]] = set()
    for path in files:
        config = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "max_workers", "batch_size", "checkpoint_frequency",
            "max_api_calls", "limit_questions",
        ):
            field_values[field][json.dumps(config.get(field), sort_keys=True)] += 1
        for value in config.get("models", {}).values():
            model_identifiers.add((value["provider"], value["model_id"]))
    return {
        "run_config_files_examined": len(files),
        "recorded_field_values": {
            key: dict(sorted(values.items())) for key, values in sorted(field_values.items())
        },
        "recorded_provider_model_identifiers": [
            {"provider": provider, "model_id": model_id}
            for provider, model_id in sorted(model_identifiers)
        ],
        "source_root": LEGACY_RESULTS.relative_to(ROOT).as_posix(),
    }


def experiment_config(duplicate_report: dict[str, Any]) -> dict[str, Any]:
    prompt_contract = {
        "message_roles": ["user"],
        "system_prompt": None,
        "user_prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "structured_output": None,
        "requested_text_schema": {
            "BQA": ["ANSWER: TRUE|FALSE", "CONFIDENCE: float in [0,1]"],
            "OEQA": ["ANSWER: local names, semicolon-separated", "CONFIDENCE: float in [0,1]"],
            "explanation_requested": False,
        },
        "current_parser_contract": {
            "answer": "ANSWER field preferred; first-line fallback retained from v1.0 evaluator",
            "confidence": "CONFIDENCE field preferred; absent or invalid values default to 0.5",
            "extra_structured_metadata": False,
        },
    }
    common = {
        "api": "OpenAI-compatible chat.completions.create",
        "temperature": 0.0,
        "top_p": 0.9,
        "seed": None,
        "maximum_output_tokens": 1024,
        "maximum_output_tokens_parameter": "max_tokens",
        "timeout_seconds": 30,
        "retry_behavior_v1_0": "no automatic retry in process_single_model_request",
        "response_format": None,
        "tools": None,
    }
    models = {
        "GPT-5 mini": {
            "model_id": "gpt-5-mini-2025-08-07",
            "provider": "OpenAI",
            "runtime": "OpenAI Python client, Chat Completions API",
            **common,
            "temperature": None,
            "top_p": None,
            "maximum_output_tokens_parameter": "max_completion_tokens",
            "reasoning_effort": "low",
            "verbosity": "low",
            "provider_specific_parameters": {},
            "parameter_note": "temperature and top_p were omitted, not explicitly set",
        },
        "Gemini 2.5 Flash-Lite": {
            "model_id": "google/gemini-2.5-flash-lite",
            "provider": "OpenRouter",
            "runtime": "OpenAI Python client against https://openrouter.ai/api/v1",
            **common,
            "reasoning_or_thinking": "no parameter sent; provider/model default not recorded",
            "provider_specific_parameters": {
                "presence_penalty": 0.0,
                "frequency_penalty": 0.1,
            },
        },
        "Qwen3-30B-A3B-Instruct": {
            "model_id": "qwen/qwen3-30b-a3b-instruct-2507",
            "provider": "OpenRouter",
            "runtime": "OpenAI Python client against https://openrouter.ai/api/v1",
            **common,
            "reasoning_or_thinking": "no parameter sent; provider/model default not recorded",
            "provider_specific_parameters": {
                "presence_penalty": 0.0,
                "frequency_penalty": 0.1,
            },
        },
    }
    uncertainties = [
        {
            "id": "request-parameter-provenance",
            "blocking": True,
            "detail": "Historical run_config files record models and operational settings but not the per-request parameter dictionary; values above are reconstructed from the evaluation source used by the saved pipeline.",
        },
        {
            "id": "openrouter-backend-revision",
            "blocking": True,
            "detail": "OpenRouter model IDs are recorded, but the routed backend/provider revision and returned model identifier were not persisted in reusable checkpoint rows.",
        },
        {
            "id": "thinking-defaults",
            "blocking": True,
            "detail": "No Gemini or Qwen thinking/reasoning parameter was sent; the effective provider/model default was not recorded.",
        },
        {
            "id": "run-specific-concurrency",
            "blocking": True,
            "detail": "Historical run_config files contain more than one concurrency, batch, and checkpoint setting. Exact values are recoverable per source run, but there is no single v1.0-wide value.",
        },
    ]
    if duplicate_report["fatal_incompatible_gold_semantics"]:
        uncertainties.append(
            {
                "id": "incompatible-duplicate-inputs",
                "blocking": True,
                "detail": "Exact evaluated inputs with incompatible gold semantics exist; see duplicate_input_hash_audit.csv.",
            }
        )
    return {
        "config_version": CONFIG_VERSION,
        "status": "blocked_pending_scientific_resolution",
        "execution_authorized": False,
        "ready_for_execution": False,
        "phase6_commit": "7d1cf3b",
        "prompt_and_response_contract": prompt_contract,
        "models": models,
        "execution_design": {
            "sole_work_source": "release/v1.1.0-staging/pending_model_runs.csv",
            "recommended_max_workers": 1,
            "checkpoint_frequency": "after every completed provider response",
            "batching": "one manifest row per submitted request",
            "technical_retry_policy": {
                "maximum_retries_after_initial_attempt": 3,
                "backoff_seconds": [2, 4, 8],
                "retryable": [
                    "timeout", "rate_limit", "provider_transient_error", "empty_response"
                ],
                "terminal_without_resampling": [
                    "nonempty_malformed_structured_output", "valid_model_answer"
                ],
            },
            "request_fingerprint": "sha256(canonical JSON of input_hash, config_version, model configuration)",
        },
        "historical_evidence": historical_run_config_summary(),
        "configuration_uncertainties": uncertainties,
    }


def build_reuse_mapping() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = read_csv(STAGE / "model_input_manifest.csv")
    task_by_id = {
        row["new_task_id"]: row for row in read_csv(STAGE / "task_id_mapping.csv")
    }
    predictions = phase5.prediction_inventory()
    source_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    parse_statuses: Counter[str] = Counter()
    failures: list[str] = []
    for input_row in inputs:
        statuses = json.loads(input_row["reuse_status_by_model"])
        for model, status in statuses.items():
            if status != "reusable":
                continue
            task = task_by_id[input_row["task_id"]]
            dataset_key = "Family" if task["dataset"] == "FamilyOWL" else task["dataset"]
            saved = predictions.get(
                (
                    dataset_key, task["hop"], model, input_row["representation"],
                    task["semantic_key"],
                )
            )
            if saved is None:
                failures.append(f"missing source row: {input_row['task_id']}/{model}/{input_row['representation']}")
                continue
            if phase6.saved_prompt_hash(saved, input_row["representation"]) != input_row["input_hash"]:
                failures.append(f"input hash mismatch: {input_row['task_id']}/{model}/{input_row['representation']}")
                continue
            response_column = saved["_response_column"]
            response = saved.get(response_column, "")
            parsed = parse_response(response, task["task_group"])
            if not parsed["usable"]:
                failures.append(f"unusable response: {input_row['task_id']}/{model}/{input_row['representation']}")
                continue
            source_path = saved["_source_path"]
            if source_path not in source_hashes:
                source_hashes[source_path] = sha256_file(ROOT / source_path)
            counts[(model, input_row["representation"])] += 1
            parse_statuses[parsed["status"]] += 1
            rows.append(
                {
                    "task_id": input_row["task_id"],
                    "representation": input_row["representation"],
                    "model": model,
                    "input_hash": input_row["input_hash"],
                    "source_path": source_path,
                    "source_file_sha256": source_hashes[source_path],
                    "source_legacy_task_id": saved.get("Task ID", ""),
                    "source_response_column": response_column,
                    "response_sha256": sha256_bytes(response.encode("utf-8")),
                    "parsed_answer": parsed["answer"],
                    "parsed_confidence": parsed["confidence"],
                    "parse_status": parsed["status"],
                    "provenance_status": "exact model and exact evaluated input hash",
                }
            )
    rows.sort(key=lambda row: (int(row["task_id"]), REPRESENTATIONS.index(row["representation"]), row["model"]))
    write_csv(
        PREFLIGHT / "reused_response_mapping.csv",
        rows,
        [
            "task_id", "representation", "model", "input_hash", "source_path",
            "source_file_sha256", "source_legacy_task_id", "source_response_column",
            "response_sha256", "parsed_answer", "parsed_confidence", "parse_status",
            "provenance_status",
        ],
    )
    if failures or len(rows) != EXPECTED_REUSED:
        raise ValueError(
            f"Reuse validation failed: rows={len(rows)}, failures={failures[:5]}"
        )
    report = {
        "total": len(rows),
        "by_model_representation": {
            f"{model}/{representation}": counts[(model, representation)]
            for model in MODELS for representation in REPRESENTATIONS
        },
        "parse_statuses": dict(sorted(parse_statuses.items())),
        "empty_responses": 0,
        "api_or_error_placeholders": 0,
        "model_mismatches": 0,
        "input_hash_mismatches": 0,
        "original_response_files_modified": False,
    }
    write_json(PREFLIGHT / "reuse_validation_report.json", report)
    return rows, report


def main() -> int:
    integrity = audit_manifest_integrity()
    duplicate_report = audit_duplicate_hashes()
    config = experiment_config(duplicate_report)
    write_json(PREFLIGHT / "experiment_config_v1_1.json", config)
    _, reuse_report = build_reuse_mapping()
    report = {
        "phase": "7A",
        "status": "failed_preflight" if duplicate_report["fatal_incompatible_gold_semantics"] else "preflight_complete",
        "phase6_commit": "7d1cf3b",
        "manifest_integrity": integrity,
        "duplicate_input_hash_audit": duplicate_report,
        "reuse_validation": reuse_report,
        "configuration_status": config["status"],
        "configuration_uncertainties": config["configuration_uncertainties"],
        "response_schema": config["prompt_and_response_contract"]["current_parser_contract"],
        "retry_policy": config["execution_design"]["technical_retry_policy"],
        "runner": "scripts/run_v1_1_experiments.py",
        "completeness_validator": "scripts/validate_v1_1_experiment_completeness.py",
        "test_report": "release/v1.1.0-preflight/phase7a_test_report.json",
        "expected_actual_calls": EXPECTED_PENDING,
        "llm_or_api_calls_made": 0,
        "benchmark_membership_or_content_changed": False,
        "v1_0_files_modified": False,
    }
    write_json(PREFLIGHT / "phase7a_preflight_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if duplicate_report["fatal_incompatible_gold_semantics"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
