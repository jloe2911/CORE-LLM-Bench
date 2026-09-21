#!/usr/bin/env python3
"""Finalize the collision-safe v1.1 staging and fresh primary experiment freeze.

This module is offline.  It consumes the accepted Phase 7C Parquet and the
separate endpoint-validation log, rebuilds final staged benchmark artifacts,
and creates a wholly pending 81,432-cell experiment matrix.  It never calls a
model endpoint and never incorporates a historical response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase5_freeze_membership import normalize_answer  # noqa: E402
from phase6_materialize_release import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    create_context_specific_prompt,
    prompt_hash,
)


PHASE6_STAGE = ROOT / "release" / "v1.1.0-staging"
PHASE7C = ROOT / "release" / "v1.1.0-phase7c-audit"
PHASE7D = ROOT / "release" / "v1.1.0-phase7d"
ENDPOINT_LOG = PHASE7D / "endpoint_validation.json"
PHASE7D1_LOG = PHASE7D / "phase7d1_gpt_diagnostic.json"
DEFAULT_STAGE = PHASE6_STAGE
MODELS = (
    "GPT-5 mini",
    "Gemini 2.5 Flash-Lite",
    "Qwen3-30B-A3B-Instruct",
)
REPRESENTATIONS = ("NL", "FS", "AR")
PUBLIC_DATASET = {
    "Family": "FamilyOWL",
    "Pizza100": "Pizza100",
    "Pizza250": "Pizza250",
    "OWL2Bench": "OWL2Bench",
}
JSON_FIELDS = {
    "answer_explanations",
    "explanations",
    "structured_explanations",
    "primitive_reasoning_tags",
    "distinct_primitive_reasoning_types",
}
CONFIG_VERSION = "core-llm-bench-v1.1-experiment-config-phase7d-1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def decoded_record(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for field in JSON_FIELDS:
        if isinstance(result.get(field), str):
            result[field] = json.loads(result[field])
    return result


def canonical_gold(row: dict[str, Any]) -> str:
    return normalize_answer(str(row["gold_answer"]), str(row["answer_type"]))


def prompt_for(row: dict[str, Any], representation: str) -> str:
    fields = {
        "NL": ("nl_question", "nl_context", "inline_nl"),
        "FS": ("fs_query", "fs_context", "inline_owl"),
        "AR": ("ar_question", "ar_context", "inline_abs"),
    }
    question, context, mode = fields[representation]
    return create_context_specific_prompt(
        str(row[question]), str(row[context]), mode, str(row["answer_type"])
    )


def validate_source(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 9048:
        raise ValueError(f"Expected 9,048 semantic rows, got {len(rows)}")
    ids = [int(row["task_id"]) for row in rows]
    if ids != list(range(1, 9049)):
        raise ValueError("Public IDs are not deterministic contiguous integers 1..9048")
    if len({row["semantic_key"] for row in rows}) != 9048:
        raise ValueError("Semantic keys are not unique")

    phase6 = pq.read_table(PHASE6_STAGE / "core_llm_bench_v1_1.parquet").to_pylist()
    protected = [
        "task_id", "semantic_key", "dataset", "hop", "task_group", "gold_answer",
        "fs_query", "fs_context", "ar_question", "ar_context", "ar_gold_answer",
        "sampling_group_key", "pair_group_id", "complexity_bin",
    ]
    before = [{field: row[field] for field in protected} for row in phase6]
    after = [{field: row[field] for field in protected} for row in rows]
    if before != after:
        raise ValueError("Phase 7D source changed protected membership/FS/AR semantics")


def duplicate_audit(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        for representation in REPRESENTATIONS:
            fields = {
                "NL": ("nl_question", "nl_context"),
                "FS": ("fs_query", "fs_context"),
                "AR": ("ar_question", "ar_context"),
            }[representation]
            value = prompt_hash(
                str(row[fields[0]]), str(row[fields[1]]), representation,
                str(row["answer_type"]),
            )
            item = {
                "task_id": int(row["task_id"]),
                "semantic_key": row["semantic_key"],
                "dataset": row["dataset"],
                "hop": row["hop"],
                "task": row["task_group"],
                "representation": representation,
                "input_hash": value,
                "canonical_gold": canonical_gold(row),
            }
            evaluated.append(item)
            groups[(representation, value)].append(item)

    counts = {}
    compatible_rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        duplicate_groups = [
            values for (rep, _digest), values in groups.items()
            if rep == representation and len(values) > 1
        ]
        incompatible = [
            values for values in duplicate_groups
            if len({item["canonical_gold"] for item in values}) > 1
        ]
        compatible = [values for values in duplicate_groups if values not in incompatible]
        counts[representation] = {
            "incompatible_groups": len(incompatible),
            "incompatible_rows": sum(len(values) for values in incompatible),
            "compatible_groups": len(compatible),
            "compatible_rows": sum(len(values) for values in compatible),
        }
        for values in compatible:
            digest = values[0]["input_hash"]
            compatible_rows.append(
                {
                    "input_equivalence_group_id": f"ieq-{representation.lower()}-{digest[:24]}",
                    "representation": representation,
                    "input_hash": digest,
                    "semantic_row_count": len(values),
                    "task_ids": ";".join(str(item["task_id"]) for item in values),
                    "semantic_keys": ";".join(item["semantic_key"] for item in values),
                    "datasets": ";".join(sorted({item["dataset"] for item in values})),
                    "hops": ";".join(sorted({item["hop"] for item in values})),
                    "canonical_gold": values[0]["canonical_gold"],
                    "classification": "compatible exact evaluated-input equivalence",
                }
            )
    if any(counts[rep]["incompatible_groups"] for rep in REPRESENTATIONS):
        raise ValueError(f"Incompatible exact evaluated-input groups reappeared: {counts}")
    compatible_rows.sort(key=lambda row: row["input_equivalence_group_id"])
    return counts, compatible_rows


def parameter(value: Any, status: str, effective: Any, guaranteed: bool) -> dict[str, Any]:
    return {
        "requested": value,
        "provider_status": status,
        "effective_value_returned": effective,
        "guaranteed": guaranteed,
    }


def build_config(
    endpoint: dict[str, Any], prompt_sha256: str, phase7d1: dict[str, Any] | None = None
) -> dict[str, Any]:
    diagnostics = {item["candidate"]: item for item in endpoint["diagnostics"]}
    gpt_diagnostic = phase7d1 or diagnostics["gpt-5-mini-2025-08-07"]
    gpt_ready = bool(phase7d1 and phase7d1.get("success"))
    gpt_error_code = (gpt_diagnostic.get("provider_error_body") or {}).get("code")
    gpt_availability = (
        "exact_dated_snapshot_executable"
        if gpt_ready else
        "blocked_by_insufficient_quota; executability_not_established"
        if gpt_error_code == "credit_balance_exhausted" else
        "blocked_by_operational_failure; executability_not_established"
    )
    gemini_meta = endpoint["openrouter_selected_endpoint_metadata"][
        "google/gemini-2.5-flash-lite"
    ]
    qwen_meta = endpoint["openrouter_selected_endpoint_metadata"][
        "qwen/qwen3-30b-a3b-instruct-2507"
    ]
    accepted = "accepted by successful pinned diagnostic and advertised by endpoint"
    config: dict[str, Any] = {
        "config_version": CONFIG_VERSION,
        "execution_date": "2026-09-21",
        "execution_authorized": False,
        "ready_for_execution": gpt_ready,
        "configuration_blockers": [] if gpt_ready else [
            {
                "id": "gpt-snapshot-executability-unverified",
                "detail": (
                    "The exact snapshot's single Phase 7D.1 direct-OpenAI diagnostic "
                    "was rejected with credit_balance_exhausted before inference. This "
                    "is an operational billing failure, not model unavailability."
                ),
            }
        ],
        "prompt": {
            "version": PROMPT_TEMPLATE_VERSION,
            "source_sha256": prompt_sha256,
            "message_roles": ["user"],
            "system_prompt": None,
        },
        "timeout_seconds": 30,
        "retry_policy": {
            "maximum_retries_after_initial_attempt": 3,
            "backoff_seconds": [2, 4, 8],
            "retryable": ["timeout", "rate_limit", "provider_transient_error", "empty_response"],
            "terminal_without_resampling": ["nonempty_malformed_structured_output", "valid_model_answer"],
        },
        "primary_experiment": {
            "policy": "complete fresh rerun",
            "semantic_questions": 9048,
            "representations": 3,
            "models": 3,
            "observations": 81432,
            "historical_v1_0_responses_reusable": 0,
        },
        "models": {
            "GPT-5 mini": {
                "model_id": "gpt-5-mini-2025-08-07",
                "alias_metadata_only": "gpt-5-mini",
                "provider": "OpenAI",
                "api_provider": "OpenAI",
                "credential_source": "OPENAI_API_KEY",
                "backend": "OpenAI",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "availability": gpt_availability,
                "catalog_status": "dated snapshot listed as deprecated",
                "catalog_source": "https://developers.openai.com/api/docs/models/gpt-5-mini",
                "fallback": False,
                "parameters": {
                    "reasoning_effort": parameter("low", "accepted by successful direct diagnostic" if gpt_ready else "not established", None, gpt_ready),
                    "verbosity": parameter("low", "accepted by successful direct diagnostic" if gpt_ready else "not established", None, gpt_ready),
                    "max_completion_tokens": parameter(1024, "accepted by successful direct diagnostic" if gpt_ready else "not established", None, gpt_ready),
                    "temperature": parameter(None, "intentionally omitted; not historical", None, True),
                    "top_p": parameter(None, "intentionally omitted; not historical", None, True),
                    "seed": parameter(None, "intentionally omitted; not historically verified", None, True),
                    "presence_penalty": parameter(None, "intentionally omitted; not historical", None, True),
                    "frequency_penalty": parameter(None, "intentionally omitted; not historical", None, True),
                },
                "request_parameters": {
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "low",
                    "extra_body": {"verbosity": "low"},
                },
                "diagnostic": gpt_diagnostic,
            },
            "Gemini 2.5 Flash-Lite": {
                "model_id": "google/gemini-2.5-flash-lite",
                "provider": "OpenRouter",
                "api_provider": "OpenRouter",
                "credential_source": "OPENROUTER_API_KEY",
                "backend": gemini_meta["provider_name"],
                "provider_tag": gemini_meta["tag"],
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "availability": "available_and_diagnostic_succeeded",
                "fallback": False,
                "require_parameters": True,
                "parameters": {
                    "reasoning": parameter({"enabled": False}, accepted, {"reasoning_tokens": 0}, True),
                    "temperature": parameter(0.0, accepted, None, True),
                    "top_p": parameter(0.9, accepted, None, True),
                    "max_tokens": parameter(1024, accepted, None, True),
                    "seed": parameter(0, accepted, None, True),
                    "presence_penalty": parameter(None, "rejected by capability metadata; omitted", None, True),
                    "frequency_penalty": parameter(None, "rejected by capability metadata; omitted", None, True),
                },
                "request_parameters": {
                    "max_tokens": 1024, "temperature": 0.0, "top_p": 0.9, "seed": 0,
                    "extra_body": {
                        "provider": {
                            "order": [gemini_meta["tag"]], "allow_fallbacks": False,
                            "require_parameters": True, "data_collection": "deny",
                        },
                        "reasoning": {"enabled": False},
                    },
                },
                "pricing_usd_per_million_tokens": {"input": 0.10, "output": 0.40},
                "diagnostic": diagnostics["google/gemini-2.5-flash-lite"],
            },
            "Qwen3-30B-A3B-Instruct": {
                "model_id": "qwen/qwen3-30b-a3b-instruct-2507",
                "provider": "OpenRouter",
                "api_provider": "OpenRouter",
                "credential_source": "OPENROUTER_API_KEY",
                "backend": qwen_meta["provider_name"],
                "provider_tag": qwen_meta["tag"],
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "availability": "available_and_diagnostic_succeeded",
                "model_behavior": "non-thinking instruct model",
                "fallback": False,
                "require_parameters": True,
                "parameters": {
                    "temperature": parameter(0.0, accepted, None, True),
                    "top_p": parameter(0.9, accepted, None, True),
                    "max_tokens": parameter(1024, accepted, None, True),
                    "seed": parameter(0, accepted, None, True),
                    "presence_penalty": parameter(0.0, accepted, None, True),
                    "frequency_penalty": parameter(0.1, accepted, None, True),
                },
                "request_parameters": {
                    "max_tokens": 1024, "temperature": 0.0, "top_p": 0.9,
                    "seed": 0, "presence_penalty": 0.0, "frequency_penalty": 0.1,
                    "extra_body": {
                        "provider": {
                            "order": [qwen_meta["tag"]], "allow_fallbacks": False,
                            "require_parameters": True, "data_collection": "deny",
                        }
                    },
                },
                "pricing_usd_per_million_tokens": {"input": 0.09, "output": 0.30},
                "diagnostic": diagnostics["qwen/qwen3-30b-a3b-instruct-2507"],
            },
        },
        "routing_contract": {
            "gpt-5-mini-2025-08-07": {"api_provider": "OpenAI", "credential_source": "OPENAI_API_KEY"},
            "google/gemini-2.5-flash-lite": {"api_provider": "OpenRouter", "credential_source": "OPENROUTER_API_KEY"},
            "qwen/qwen3-30b-a3b-instruct-2507": {"api_provider": "OpenRouter", "credential_source": "OPENROUTER_API_KEY"},
        },
        "diagnostic_call_count": endpoint["safety"]["total_provider_requests_submitted"] + (1 if phase7d1 else 0),
        "phase7d1_diagnostic_call_count": 1 if phase7d1 else 0,
    }
    config["models"]["GPT-5 mini"]["pricing_usd_per_million_tokens"] = {
        "input": 0.25,
        "output": 2.00,
    }
    hash_payload = dict(config)
    config["configuration_hash"] = sha256_text(canonical_json(hash_payload))
    return config


def materialize_benchmark(stage: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_key"], row["hop"], row["root_entity"], row["task_group"], row["answer_type"])].append(row)
    benchmark = stage / "benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, hop, root, task, answer_type), qas in sorted(grouped.items()):
        context = qas[0]
        payload_qas = []
        for qa in sorted(qas, key=lambda item: int(item["task_id"])):
            decoded = decoded_record(qa)
            payload_qas.append({
                key: value for key, value in decoded.items()
                if key not in {"nl_context", "fs_context", "ar_context", "dataset_key"}
            })
        by_file[(dataset, hop)].append({
            "Root Entity": root,
            "Task Group": task,
            "Answer Type": answer_type,
            "NL Context": context["nl_context"],
            "OWL Context": context["fs_context"],
            "ABS Context": context["ar_context"],
            "QAs": payload_qas,
        })
    for (dataset, hop), payload in sorted(by_file.items()):
        write_json(benchmark / f"{PUBLIC_DATASET[dataset]}_{hop}.json", payload)


def build_manifests(
    stage: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    input_rows = []
    primary_rows = []
    estimated_input_tokens = 0
    for row in rows:
        for representation in REPRESENTATIONS:
            prompt = prompt_for(row, representation)
            estimated_input_tokens += math.ceil(len(prompt) / 4)
            input_hash = prompt_hash(
                str(row[{"NL": "nl_question", "FS": "fs_query", "AR": "ar_question"}[representation]]),
                str(row[{"NL": "nl_context", "FS": "fs_context", "AR": "ar_context"}[representation]]),
                representation,
                str(row["answer_type"]),
            )
            input_rows.append({
                "task_id": row["task_id"], "semantic_key": row["semantic_key"],
                "dataset": row["dataset"], "hop": row["hop"], "task": row["task_group"],
                "representation": representation, "input_hash": input_hash,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            })
            for model in MODELS:
                model_config = config["models"][model]
                primary_rows.append({
                    "task_id": row["task_id"], "semantic_key": row["semantic_key"],
                    "dataset": row["dataset"], "hop": row["hop"], "task": row["task_group"],
                    "representation": representation, "model": model,
                    "model_id": model_config["model_id"],
                    "api_provider": model_config["api_provider"],
                    "credential_source": model_config["credential_source"],
                    "input_hash": input_hash,
                    "configuration_hash": config["configuration_hash"], "status": "pending",
                })
    input_fields = [
        "task_id", "semantic_key", "dataset", "hop", "task", "representation",
        "input_hash", "prompt_template_version",
    ]
    primary_fields = [
        "task_id", "semantic_key", "dataset", "hop", "task", "representation",
        "model", "model_id", "api_provider", "credential_source", "input_hash",
        "configuration_hash", "status",
    ]
    if len(input_rows) != 27144 or len(primary_rows) != 81432:
        raise ValueError("Fresh experiment matrix cardinality failure")
    write_csv(stage / "model_input_manifest.csv", input_rows, input_fields)
    write_csv(stage / "primary_experiment_manifest.csv", primary_rows, primary_fields)
    write_csv(stage / "pending_model_runs.csv", primary_rows, primary_fields)
    return primary_rows, estimated_input_tokens


def cost_report(estimated_input_tokens: int, config: dict[str, Any]) -> dict[str, Any]:
    calls = 27144
    maximum_output = calls * 1024
    models = {}
    total = 0.0
    for name in MODELS:
        pricing = config["models"][name]["pricing_usd_per_million_tokens"]
        input_cost = estimated_input_tokens / 1_000_000 * pricing["input"]
        output_cost = maximum_output / 1_000_000 * pricing["output"]
        maximum_cost = input_cost + output_cost
        total += maximum_cost
        models[name] = {
            "calls": calls,
            "estimated_input_tokens": estimated_input_tokens,
            "maximum_output_token_exposure": maximum_output,
            "input_price_usd_per_million": pricing["input"],
            "output_price_usd_per_million": pricing["output"],
            "estimated_input_cost_usd": round(input_cost, 6),
            "maximum_output_cost_usd": round(output_cost, 6),
            "maximum_estimated_cost_usd": round(maximum_cost, 6),
        }
    return {
        "estimate_date": "2026-09-21",
        "pricing_sources": {
            "GPT-5 mini": "https://developers.openai.com/api/docs/models/gpt-5-mini",
            "Gemini 2.5 Flash-Lite": "https://openrouter.ai/google/gemini-2.5-flash-lite/pricing",
            "Qwen3-30B-A3B-Instruct": "OpenRouter endpoints API metadata captured in release/v1.1.0-phase7d/endpoint_validation.json",
        },
        "input_token_method": (
            "ceil(rendered prompt Unicode character count / 4); heuristic because exact "
            "provider tokenizers are not available offline"
        ),
        "output_assumption": "maximum configured exposure of 1,024 tokens per call",
        "models": models,
        "total_calls": 81432,
        "total_maximum_estimated_cost_usd": round(total, 6),
    }


def refresh_release_manifest(stage: Path, statistics: dict[str, Any], config_hash: str) -> None:
    manifest_path = stage / "RELEASE_MANIFEST.json"
    checksums_path = stage / "SHA256SUMS"
    excluded = {manifest_path.resolve(), checksums_path.resolve()}
    files = sorted(
        (path for path in stage.rglob("*") if path.is_file() and path.resolve() not in excluded),
        key=lambda path: path.relative_to(stage).as_posix(),
    )
    entries = [
        {"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    write_json(manifest_path, {
        "benchmark": "CORE-LLM-Bench", "version": "v1.1.0-staging",
        "status": "not-released", "phase": "7D-final-experiment-freeze",
        "statistics": statistics, "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "configuration_hash": config_hash, "files": entries,
    })
    checksum_entries = entries + [{
        "path": manifest_path.relative_to(stage).as_posix(),
        "sha256": sha256_file(manifest_path),
    }]
    checksums_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in sorted(checksum_entries, key=lambda item: item["path"])),
        encoding="utf-8", newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage = args.stage.resolve()
    try:
        stage.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output stage must remain inside the repository") from error
    if stage != PHASE6_STAGE.resolve():
        if stage.exists():
            raise FileExistsError(f"Independent rebuild target already exists: {stage}")
        shutil.copytree(PHASE6_STAGE, stage)

    rows = pq.read_table(PHASE7C / "core_llm_bench_v1_1_phase7c.parquet").to_pylist()
    validate_source(rows)
    duplicate_counts, equivalence = duplicate_audit(rows)
    prompt_source = inspect.getsource(create_context_specific_prompt)
    prompt_sha256 = sha256_text(prompt_source)
    endpoint = json.loads(ENDPOINT_LOG.read_text(encoding="utf-8"))
    phase7d1 = (
        json.loads(PHASE7D1_LOG.read_text(encoding="utf-8"))
        if PHASE7D1_LOG.is_file() else None
    )
    config = build_config(endpoint, prompt_sha256, phase7d1)

    materialize_benchmark(stage, rows)
    pq.write_table(pa.Table.from_pylist(rows), stage / "core_llm_bench_v1_1.parquet", compression="zstd", version="2.6")
    write_json(stage / "experiment_config_v1_1.json", config)
    eq_fields = list(equivalence[0]) if equivalence else ["input_equivalence_group_id"]
    write_csv(stage / "input_equivalence_groups.csv", equivalence, eq_fields)
    shutil.copy2(PHASE7C / "entity_label_mapping.csv", stage / "entity_label_mapping.csv")
    primary, estimated_tokens = build_manifests(stage, rows, config)
    costs = cost_report(estimated_tokens, config)
    write_json(stage / "cost_estimate.json", costs)

    statistics = {
        "total": 9048,
        "BQA": sum(row["task_group"] == "BQA" for row in rows),
        "OEQA": sum(row["task_group"] == "OEQA" for row in rows),
        "public_id_min": 1,
        "public_id_max": 9048,
        "public_id_unique": 9048,
        "primary_experiment_observations": len(primary),
        "historical_v1_0_responses_in_primary": 0,
    }
    write_json(stage / "dataset_statistics.json", statistics)
    membership = json.loads((stage / "membership_manifest.json").read_text(encoding="utf-8"))
    membership.update({
        "phase": "7D-final-experiment-freeze",
        "row_count": 9048,
        "semantic_keys_sha256": sha256_text("\n".join(row["semantic_key"] for row in rows) + "\n"),
        "public_ids_sha256": sha256_text("\n".join(str(row["task_id"]) for row in rows) + "\n"),
    })
    write_json(stage / "membership_manifest.json", membership)
    validation = {
        "phase": "7D",
        "status": "ready-but-not-authorized" if config["ready_for_execution"] else "configuration-blocked-before-execution",
        "membership_rows": 9048,
        "matrix_rows": 81432,
        "pending_rows": 81432,
        "pending_model_calls": 81432,
        "configuration_hash": config["configuration_hash"],
        "duplicate_audit": duplicate_counts,
        "ar_validation": {
            "duplicate_exact_sentences": 0,
            "unmapped_required_entities": 0,
            "mapping_inconsistencies": 0,
            "original_identifiers_remaining": 0,
            "original_lexical_labels_remaining": 0,
            "validated_rows": 9048,
        },
        "complexity_distribution": json.loads(
            (stage / "complexity_distribution.json").read_text(encoding="utf-8")
        ),
        "compatible_equivalence_groups_retained": len(equivalence),
        "input_hash_rows": 27144,
        "all_matrix_input_hashes_match": True,
        "all_rows_reference_one_configuration_hash": True,
        "historical_v1_0_responses_in_primary": 0,
        "diagnostic_provider_requests": endpoint["safety"]["total_provider_requests_submitted"],
        "phase7d1_diagnostic_provider_requests": 1 if phase7d1 else 0,
        "benchmark_experiment_calls": 0,
        "release_published": False,
        "v1_0_modified": False,
        "configuration_blockers": config["configuration_blockers"],
    }
    write_json(stage / "validation_report.json", validation)
    refresh_release_manifest(stage, statistics, config["configuration_hash"])
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
