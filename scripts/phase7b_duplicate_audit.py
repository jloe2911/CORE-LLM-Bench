#!/usr/bin/env python3
"""Offline Phase 7B audit of duplicate evaluated inputs.

This script is deliberately diagnostic only.  It reads the frozen Phase 6
staging release and Phase 7A provenance, writes audit tables beside the Phase
7A artifacts, and never imports an API client or changes benchmark membership.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
PREFLIGHT = ROOT / "release" / "v1.1.0-preflight"
LEGACY = ROOT / "data" / "output" / "final_benchmark_llm_results"
REPRESENTATIONS = ("NL", "FS", "AR")
QUERY_TRIPLE = re.compile(
    r"(?:ASK|SELECT\s+\?x)\s+WHERE\s*\{\s*<([^>]+)>\s+<([^>]+)>\s+(?:<([^>]+)>|\?x)\s*\}",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def local_name(value: str) -> str:
    return re.split(r"[#/]", value.rstrip("/"))[-1]


def query_parts(query: str) -> tuple[str, str, str]:
    match = QUERY_TRIPLE.search(query)
    if not match:
        raise ValueError(f"Unsupported formal query: {query}")
    subject, predicate, obj = match.groups()
    return subject, predicate, obj or ""


def normalized_gold(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    if row["task_group"] == "BQA":
        return "BQA", (str(row["gold_answer"]).strip().upper(),)
    return "OEQA", tuple(
        sorted(part.strip() for part in str(row["gold_answer"]).split(";") if part.strip())
    )


def classify_compatible(representation: str, rows: list[dict[str, Any]]) -> tuple[str, str]:
    datasets = {row["dataset"] for row in rows}
    queries = {row["formal_query"] for row in rows}
    if representation == "AR" and len(datasets) == 1 and len(queries) == 1:
        return "B", "exact duplicated semantic question across hop-specific benchmark instances"
    if len(datasets) > 1 and len(queries) == 1:
        return "D", "incidental cross-dataset duplicate with the same formal query and gold semantics; no evidence it was intentionally duplicated"
    return "C", "distinct formal questions collapse to one evaluated representation"


def classify_incompatible(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    if {row["task_group"] for row in rows} == {"BQA"}:
        return (
            "5 NEGATIVE-BQA CONSTRUCTION COLLISION",
            "The positive and negative ASK objects are distinct generated entities, but numeric suffix removal gives them the same NL label; the template contains the object label, but no remaining identity distinction.",
            "Exclude the complete BQA pair whenever positive and negative formal targets have the same collision-normalized NL surface and evaluated context; retain it only if a non-artificial ontology label can distinguish the targets.",
        )
    datasets = {row["dataset"] for row in rows}
    queries = {row["formal_query"] for row in rows}
    if len(datasets) > 1 and len(queries) == 1:
        return (
            "6 OEQA ANSWER-SET COLLISION",
            "Pizza100 and Pizza250 reuse the same local IRIs and produce the same NL question/context, but denote dataset-specific ontology populations with different answer sets.",
            "Make dataset scope semantically visible only if the intended representation supports it; otherwise remove all indistinguishable dataset-scoped semantic items in the collision group.",
        )
    return (
        "4 ENTITY-LABEL COLLISION",
        "Distinct FamilyOWL subject entities have different formal IRIs but the same human label after the disambiguating suffix is removed; their shared NL context therefore yields the same evaluated input.",
        "Use an ontology-grounded human-readable disambiguator only when one exists; otherwise remove all semantic items sharing the indistinguishable NL surface rather than inventing an identifier.",
    )


def prompt_for(row: dict[str, Any]) -> str:
    # Kept local so the audit cannot import the experiment runner/API client.
    answer_type = row["answer_type"]
    if answer_type == "BIN":
        fmt = (
            "ANSWER: [TRUE or FALSE]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: ONLY write TRUE or FALSE.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    else:
        fmt = (
            "ANSWER: [Use LOCAL NAMES only, semicolon-separated]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: Use LOCAL NAMES only (e.g., 'Person', 'U0C4', 'caroline_lavinia_tubb_1840'), give all the possible answers.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    base = (
        "CRITICAL: You MUST respond in exactly this format:\n"
        f"{fmt}\nDO NOT include any additional text before or after this format.\n"
    )
    context = str(row["nl_context"])
    if len(context) > 10000:
        context = context[:10000] + "\n... [truncated for memory efficiency]"
    return (
        "You are an expert in ontologies, answer the following question based on "
        "the provided ontological relationships. Reason through the ontological "
        "context and answer based on what you can infer from the context.\n\n"
        f"{base}\n\nQuestion: {row['nl_question']}\nContext: {context}"
    )


def incompatible_audit(
    duplicate_rows: list[dict[str, str]], tasks: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for duplicate in duplicate_rows:
        if duplicate["gold_semantics_compatible"] == "false":
            grouped[duplicate["input_hash"]].append(tasks[duplicate["task_id"]])
    detail: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for input_hash, rows in sorted(grouped.items()):
        category, cause, correction = classify_incompatible(rows)
        prompts = {prompt_for(row) for row in rows}
        if len(prompts) != 1:
            raise AssertionError(f"Hash group does not have one NL prompt: {input_hash}")
        prompt = next(iter(prompts))
        fingerprint_payload = json.dumps(
            {
                "prompt_template_version": "api_calls.create_context_specific_prompt@24c4520",
                "representation": "NL",
                "rendered_prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if hashlib.sha256(fingerprint_payload.encode()).hexdigest() != input_hash:
            raise AssertionError(f"Reconstructed prompt hash mismatch: {input_hash}")
        groups.append(
            {
                "input_hash": input_hash,
                "row_count": len(rows),
                "task_group": rows[0]["task_group"],
                "task_ids": ";".join(str(row["task_id"]) for row in rows),
                "datasets": ";".join(sorted({row["dataset"] for row in rows})),
                "root_cause_category": category,
                "root_cause": cause,
                "principled_correction": correction,
            }
        )
        for row in rows:
            subject, predicate, obj = query_parts(row["formal_query"])
            detail.append(
                {
                    "duplicate_nl_input_hash": input_hash,
                    "public_task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "hop": row["hop"],
                    "task_type": row["task_type"],
                    "answer_type": row["answer_type"],
                    "gold_semantics": row["gold_answer"],
                    "root_entity": row["root_entity"],
                    "subject_uri": subject,
                    "subject_local_name": local_name(subject),
                    "predicate_uri": predicate,
                    "predicate_local_name": local_name(predicate),
                    "object_uri": obj,
                    "object_local_name": local_name(obj) if obj else "",
                    "formal_sparql_query": row["formal_query"],
                    "gold_answer": row["gold_answer"],
                    "nl_question": row["nl_question"],
                    "nl_context": row["nl_context"],
                    "full_evaluated_nl_input": prompt,
                    "semantic_key": row["semantic_key"],
                    "legacy_task_id": row["legacy_task_id"],
                    "bqa_pair_id": row["pair_group_id"],
                    "fs_query_equivalent": row["fs_query"],
                    "fs_context_equivalent": row["fs_context"],
                    "ar_question_equivalent": row["ar_question"],
                    "ar_context_equivalent": row["ar_context"],
                    "ar_gold_answer_equivalent": row["ar_gold_answer"],
                    "root_cause_category": category,
                    "root_cause": cause,
                }
            )
    return detail, groups


def compatible_audit(
    duplicate_rows: list[dict[str, str]], tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for duplicate in duplicate_rows:
        if duplicate["gold_semantics_compatible"] == "true":
            grouped[duplicate["input_hash"]].append(duplicate)
    result: list[dict[str, Any]] = []
    for input_hash, duplicates in sorted(grouped.items()):
        rows = [tasks[item["task_id"]] for item in duplicates]
        representation = duplicates[0]["representation"]
        classification, rationale = classify_compatible(representation, rows)
        result.append(
            {
                "input_hash": input_hash,
                "representation": representation,
                "classification": classification,
                "classification_label": {
                    "A": "intentional equivalent instances",
                    "B": "exact duplicated semantic question",
                    "C": "representational collapse",
                    "D": "other",
                }[classification],
                "rationale": rationale,
                "row_count": len(rows),
                "task_ids": ";".join(str(row["task_id"]) for row in rows),
                "datasets": ";".join(sorted({row["dataset"] for row in rows})),
                "hops": ";".join(sorted({row["hop"] for row in rows})),
                "task_groups": ";".join(sorted({row["task_group"] for row in rows})),
                "semantic_keys": ";".join(row["semantic_key"] for row in rows),
                "formal_queries": json.dumps(
                    [row["formal_query"] for row in rows], ensure_ascii=False
                ),
                "gold_answer": rows[0]["gold_answer"],
                "aggregate_weighting_effect": (
                    f"counts the identical evaluated input {len(rows)} times in per-row aggregate evaluation"
                ),
            }
        )
    return result


def default_confidence_audit(tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mappings = [
        row for row in read_csv(PREFLIGHT / "reused_response_mapping.csv")
        if row["parse_status"] == "accepted_by_current_parser_with_confidence_default"
    ]
    source_cache: dict[str, dict[str, dict[str, str]]] = {}
    result: list[dict[str, Any]] = []
    for mapping in mappings:
        source_path = mapping["source_path"]
        if source_path not in source_cache:
            source_cache[source_path] = {
                row["Task ID"]: row for row in read_csv(ROOT / source_path)
            }
        source = source_cache[source_path][mapping["source_legacy_task_id"]]
        raw = source[mapping["source_response_column"]]
        answer_match = re.search(r"ANSWER:\s*([^\n\r]+)", raw, re.IGNORECASE)
        extracted = answer_match.group(1).strip() if answer_match else ""
        saved_final_column = mapping["source_response_column"].replace("_response", "_final_answer")
        saved_confidence_column = mapping["source_response_column"].replace("_response", "_confidence_score")
        checkpoint_truncated = len(raw) == 503 and raw.endswith("...")
        if checkpoint_truncated:
            reason = "retained checkpoint response truncated to 500 characters before CONFIDENCE field"
        elif raw.rstrip().upper().endswith("\nCONF"):
            reason = "response terminated after literal CONF; numeric confidence absent"
        else:
            reason = "CONFIDENCE field absent in retained complete response"
        result.append(
            {
                "task_id": mapping["task_id"],
                "model": mapping["model"],
                "representation": mapping["representation"],
                "task_type": tasks[mapping["task_id"]]["task_group"],
                "raw_response": raw,
                "parsed_answer": mapping["parsed_answer"],
                "answer_parsed_correctly": str(extracted == mapping["parsed_answer"]).lower(),
                "answer_payload_complete": str(not checkpoint_truncated).lower(),
                "answer_parse_assessment": (
                    "syntactically extracted but incomplete because the checkpoint truncated the ANSWER line"
                    if checkpoint_truncated
                    else "complete retained ANSWER line extracted correctly"
                ),
                "gold_answer": tasks[mapping["task_id"]]["gold_answer"],
                "missing_confidence_reason": reason,
                "saved_v1_0_final_answer": source.get(saved_final_column, ""),
                "saved_v1_0_confidence": source.get(saved_confidence_column, ""),
                "published_v1_0_fallback_rule": "yes; extract_confidence_score defaulted missing/malformed confidence to 0.5 (the original full response was scored before its stored 500-character truncation)",
                "source_path": source_path,
                "source_legacy_task_id": mapping["source_legacy_task_id"],
            }
        )
    return result


def historical_config_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    for path in sorted(LEGACY.rglob("run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        for display, model in sorted(config.get("models", {}).items()):
            run_rows.append(
                {
                    "source_path": path.relative_to(ROOT).as_posix(),
                    "dataset_hop": path.relative_to(LEGACY).parts[0],
                    "representation_directory": path.parent.name,
                    "model_key": display,
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "max_workers": config.get("max_workers"),
                    "batch_size": config.get("batch_size"),
                    "checkpoint_frequency": config.get("checkpoint_frequency"),
                    "max_api_calls": config.get("max_api_calls"),
                    "limit_questions": config.get("limit_questions"),
                }
            )
    parameters = [
        {"parameter": "model identifier", "scope": "model output", "status": "VERIFIED", "value": "gpt-5-mini-2025-08-07; google/gemini-2.5-flash-lite; qwen/qwen3-30b-a3b-instruct-2507", "evidence": "48 run_config.json files and source-run directory names"},
        {"parameter": "maximum output tokens", "scope": "model output", "status": "VERIFIED", "value": "1024 (max_completion_tokens for GPT-5 mini; max_tokens for OpenRouter)", "evidence": "co-retained scripts/llm_pipeline/api_calls.py get_model_params"},
        {"parameter": "temperature", "scope": "model output", "status": "VERIFIED", "value": "OpenRouter 0.0; GPT-5 mini omitted", "evidence": "co-retained request-construction source"},
        {"parameter": "GPT-5 mini effective temperature", "scope": "model output", "status": "IMPLIED BY CLIENT DEFAULT AT THE TIME", "value": "no explicit numeric value", "evidence": "parameter omitted from request dictionary"},
        {"parameter": "top_p", "scope": "model output", "status": "VERIFIED", "value": "OpenRouter 0.9; GPT-5 mini omitted", "evidence": "co-retained request-construction source"},
        {"parameter": "GPT-5 mini effective top_p", "scope": "model output", "status": "IMPLIED BY CLIENT DEFAULT AT THE TIME", "value": "no explicit numeric value", "evidence": "parameter omitted from request dictionary"},
        {"parameter": "seed", "scope": "model output", "status": "VERIFIED", "value": "not sent for any model", "evidence": "co-retained request-construction source"},
        {"parameter": "effective random seed", "scope": "model output", "status": "UNKNOWN", "value": "provider-side behavior not persisted", "evidence": "no saved request/response metadata field"},
        {"parameter": "GPT-5 reasoning_effort", "scope": "model output", "status": "VERIFIED", "value": "low", "evidence": "co-retained request-construction source"},
        {"parameter": "GPT-5 verbosity", "scope": "model output", "status": "VERIFIED", "value": "low", "evidence": "co-retained request-construction source"},
        {"parameter": "Gemini/Qwen thinking setting sent", "scope": "model output", "status": "VERIFIED", "value": "none", "evidence": "co-retained request-construction source"},
        {"parameter": "Gemini/Qwen effective thinking default", "scope": "model output", "status": "UNKNOWN", "value": "provider/model default not persisted", "evidence": "no saved request/response metadata field"},
        {"parameter": "OpenRouter penalties", "scope": "model output", "status": "VERIFIED", "value": "presence_penalty=0.0; frequency_penalty=0.1", "evidence": "co-retained request-construction source"},
        {"parameter": "OpenRouter routed backend/provider revision", "scope": "model output", "status": "UNKNOWN", "value": "not persisted", "evidence": "checkpoint rows contain neither routed provider nor returned model identifier"},
        {"parameter": "request timeout", "scope": "operational", "status": "VERIFIED", "value": "30 seconds", "evidence": "co-retained process_single_model_request source"},
        {"parameter": "automatic retry", "scope": "operational", "status": "VERIFIED", "value": "none in v1.0 process_single_model_request", "evidence": "one chat.completions.create call inside one try/except"},
        {"parameter": "concurrency/batch/checkpoint", "scope": "operational", "status": "VERIFIED", "value": "run-specific; see phase7b_historical_run_configs.csv", "evidence": "48 retained run_config.json files"},
    ]
    return run_rows, parameters


def main() -> int:
    task_rows = pq.read_table(STAGE / "core_llm_bench_v1_1.parquet").to_pylist()
    tasks = {str(row["task_id"]): row for row in task_rows}
    duplicates = read_csv(PREFLIGHT / "duplicate_input_hash_audit.csv")
    incompatible_rows, incompatible_groups = incompatible_audit(duplicates, tasks)
    compatible_groups = compatible_audit(duplicates, tasks)
    confidence_rows = default_confidence_audit(tasks)
    run_configs, config_parameters = historical_config_audit()

    incompatible_fields = list(incompatible_rows[0])
    compatible_fields = list(compatible_groups[0])
    confidence_fields = list(confidence_rows[0])
    write_csv(PREFLIGHT / "phase7b_incompatible_rows.csv", incompatible_rows, incompatible_fields)
    write_csv(PREFLIGHT / "phase7b_incompatible_groups.csv", incompatible_groups, list(incompatible_groups[0]))
    write_csv(PREFLIGHT / "phase7b_compatible_duplicate_groups.csv", compatible_groups, compatible_fields)
    write_csv(PREFLIGHT / "phase7b_default_confidence_audit.csv", confidence_rows, confidence_fields)
    write_csv(PREFLIGHT / "phase7b_historical_run_configs.csv", run_configs, list(run_configs[0]))
    write_csv(PREFLIGHT / "phase7b_historical_parameter_audit.csv", config_parameters, list(config_parameters[0]))

    incompatible_type_groups = Counter(row["task_group"] for row in incompatible_groups)
    incompatible_type_rows = Counter(row["task_type"] for row in incompatible_rows)
    compatible_classes = Counter((row["representation"], row["classification"]) for row in compatible_groups)
    compatible_class_rows = Counter()
    for row in compatible_groups:
        compatible_class_rows[(row["representation"], row["classification"])] += row["row_count"]
    impacted_strata = Counter(
        (
            row["dataset"], row["hop"], row["task_type"], row["complexity_bin"]
        )
        for row in (
            tasks[str(detail["public_task_id"])] for detail in incompatible_rows
        )
    )
    report = {
        "phase": "7B",
        "status": "audit_complete_stop_for_scientific_review",
        "llm_or_api_calls_made": 0,
        "benchmark_membership_or_content_changed": False,
        "v1_0_files_modified": False,
        "incompatible": {
            "groups": len(incompatible_groups),
            "rows": len(incompatible_rows),
            "groups_by_task_type": dict(sorted(incompatible_type_groups.items())),
            "rows_by_task_type": dict(sorted(incompatible_type_rows.items())),
            "requested_breakdown_mismatch": "artifacts contain 4 BQA groups and 5 OEQA groups, not 3 and 6",
        },
        "compatible": {
            "groups": len(compatible_groups),
            "rows": sum(row["row_count"] for row in compatible_groups),
            "groups_by_representation_and_class": {
                f"{rep}/{cls}": count for (rep, cls), count in sorted(compatible_classes.items())
            },
            "rows_by_representation_and_class": {
                f"{rep}/{cls}": count for (rep, cls), count in sorted(compatible_class_rows.items())
            },
            "overweights_aggregate_evaluation": True,
        },
        "membership_impact_if_all_19_fatal_rows_removed": {
            "total": 9029,
            "BQA_rows": 6024,
            "BQA_pairs": 3012,
            "OEQA_rows": 3005,
            "ids_require_deterministic_rematerialization": True,
            "replacement_recommendation": "do not auto-replace; allow the natural total to decrease unless a separately approved same-stratum resampling policy is adopted",
        },
        "recommended_scope": "Option 2: remove/deduplicate affected semantic questions; no broad resampling",
        "local_or_systematic": "The 9 incompatible groups are local observed failures, but their causes are systematic generation-policy defects: collision-losing NL labels, no post-render uniqueness gate, and cross-dataset scope omission. The 411 compatible NL representational-collapse groups show the mechanism is not confined to the fatal rows.",
        "impacted_strata_if_removed": {
            "|".join(key): count for key, count in sorted(impacted_strata.items())
        },
        "mixed_reuse_recommendation": "Option 3: rerun the complete v1.1 matrix under one newly frozen request and routing configuration; retain historical exact-input outputs as provenance/sensitivity data, not primary mixed-condition observations",
        "default_confidence": {
            "rows": len(confidence_rows),
            "answer_parsed_correctly": sum(row["answer_parsed_correctly"] == "true" for row in confidence_rows),
            "reasons": dict(sorted(Counter(row["missing_confidence_reason"] for row in confidence_rows).items())),
            "all_oeqa": all(row["task_type"] == "OEQA" for row in confidence_rows),
            "complete_answer_payloads": sum(row["answer_payload_complete"] == "true" for row in confidence_rows),
            "checkpoint_truncated_answer_payloads": sum(row["answer_payload_complete"] == "false" for row in confidence_rows),
            "v1_0_fallback_rule_present": True,
            "note": "saved v1.0 confidence was computed from the original full response before checkpoint serialization truncated long responses",
        },
    }
    write_json(PREFLIGHT / "phase7b_audit_report.json", report)
    group_lines = [
        "| Hash | Type | Task IDs | Datasets | Category |",
        "|---|---:|---|---|---|",
    ]
    for group in incompatible_groups:
        group_lines.append(
            f"| `{group['input_hash']}` | {group['task_group']} | {group['task_ids']} | "
            f"{group['datasets']} | {group['root_cause_category']} |"
        )
    markdown = "\n".join(
        [
            "# Phase 7B duplicate-input and reuse audit",
            "",
            "Status: audit complete; stopped for scientific review. No benchmark repair, resampling, ID rematerialization, experiment execution, model/API call, commit, or v1.0.0 modification occurred.",
            "",
            "## Fatal incompatible inputs",
            "",
            "The frozen artifacts contain 9 groups / 19 rows, split as 4 BQA groups / 8 rows and 5 OEQA groups / 11 rows. This differs from the requested expected split of 3 BQA and 6 OEQA groups.",
            "",
            *group_lines,
            "",
            "The exhaustive row table, including full NL context, reconstructed evaluated prompt, semantic/provenance identities, and FS/AR equivalents, is `phase7b_incompatible_rows.csv`. Group-level causes and proposed corrections are in `phase7b_incompatible_groups.csv`.",
            "",
            "## Compatible duplicates",
            "",
            "There are 435 compatible groups / 1,629 rows: AR/B 7/14, FS/D 5/10, NL/D 12/24, and NL/C 411/1,581. Category D denotes incidental cross-dataset duplication: the formal query and gold agree, but there is no evidence that repeated inclusion was intentional. Every group is classified in `phase7b_compatible_duplicate_groups.csv`. Retaining them counts identical evaluated inputs repeatedly in per-row aggregates.",
            "",
            "## Required scope",
            "",
            "Recommended scope is Option 2. If all fatal groups are removed atomically, membership becomes 9,029: 6,024 BQA rows in 3,012 complete pairs and 3,005 OEQA rows. Do not auto-replace rows merely to retain 9,048. Public IDs and all derived manifests would require later deterministic rematerialization after review.",
            "",
            "The fatal instances are local, but their causes are systematic generation-policy defects: label normalization loses identity, there is no post-render uniqueness gate, and dataset scope is absent from coincident Pizza surfaces. The 411 compatible NL representational-collapse groups confirm that the mechanism extends beyond the nine fatal groups.",
            "",
            "## Representation-specific uniqueness rule",
            "",
            "FATAL: within each representation independently, one exact evaluated-input fingerprint maps to more than one normalized gold semantic value. WARNING/POLICY: one exact evaluated input maps to compatible gold semantics. Apply both checks across datasets as well as within each dataset. Current fatal counts are NL 9/19, FS 0/0, AR 0/0.",
            "",
            "## Historical configuration and mixed reuse",
            "",
            "See `phase7b_historical_parameter_audit.csv` for VERIFIED / IMPLIED BY CLIENT DEFAULT AT THE TIME / UNKNOWN classifications and `phase7b_historical_run_configs.csv` for every retained run configuration. The primary recommendation is Option 3: rerun the complete v1.1 matrix under one newly frozen request and routing configuration; retain historical exact-input responses as provenance or sensitivity data, not primary mixed-condition observations.",
            "",
            "## Default-confidence responses",
            "",
            "All 25 are OEQA and all expose a syntactically extractable ANSWER line. Eighteen stored responses truncate that ANSWER line at 500 characters and are not complete reusable answer payloads. Five complete responses omit CONFIDENCE, and two complete responses terminate at literal `CONF`. The published v1.0 evaluator did contain the 0.5 fallback; original metrics were computed before long responses were truncated for checkpoint storage. Exact rows are in `phase7b_default_confidence_audit.csv`.",
            "",
        ]
    )
    (PREFLIGHT / "PHASE7B_AUDIT.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
