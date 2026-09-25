#!/usr/bin/env python3
"""Reproducible, fail-closed, offline evaluation of CORE-LLM-Bench v1.1.

The script reads the frozen 9,048-question benchmark and the three completed
primary observation files.  It never imports an API client and never changes
benchmark or observation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
DEFAULT_OUTPUT = ROOT / "results" / "v1.1.0-evaluation-corrected-v1"
PREVIOUS_OUTPUT = ROOT / "results" / "v1.1.0-final"
EXPECTED_BENCHMARK_SHA256 = "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc"
EXPECTED_INPUT_MANIFEST_SHA256 = "191c1c0dc221a5829dfa361f86fd1ebf8bbfd54e8f89eeaf0621810001715d18"
EXPECTED_QUESTIONS = 9048
EXPECTED_PER_MODEL = 27144
EXPECTED_TOTAL = 81432
ACCEPTED = {"usable", "malformed_response"}
CONFIRMED_DEFECTIVE_AR_TASK_IDS = {"1274", "4094", "5165"}
TAXONOMY = ("D", "H", "T", "S", "A", "J", "N", "E", "∩", "¬", "I", "F", "V", "Y", "Q", "R", "C", "L", "U", "M")

sys.path.insert(0, str(ROOT / "scripts" / "llm_pipeline"))
from sageqa_answer_metrics import (  # noqa: E402
    SAGEQA_EVALUATOR_CANONICAL_SHA256,
    SAGEQA_SOURCE_COMMIT,
    answer_set_scores,
    confidence_correctness_alignment,
    load_sageqa_evaluator,
)


SOURCES = {
    "GPT-5 mini": ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
    "Gemini 2.5 Flash-Lite": ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
    "Qwen3-30B-A3B-Instruct": ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def raw_text(row: dict[str, Any]) -> str:
    value = row.get("raw_response_text")
    if value is None:
        value = row.get("raw_response")
    if value is None:
        value = ((row.get("raw_provider_response") or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
    return str(value or "")


def parse_response(text: str, task: str) -> dict[str, Any]:
    """Parse a saved response without allowing ANSWER whitespace across lines."""
    text = str(text or "").strip()
    answer_match = re.search(r"(?im)^ANSWER:[ \t]*([^\n\r]*)", text)
    answer = answer_match.group(1).strip() if answer_match else (text.splitlines()[0].strip() if text.splitlines() else text)
    if task == "BQA":
        lowered = answer.lower()
        if "true" in lowered:
            answer = "TRUE"
        elif "false" in lowered:
            answer = "FALSE"
    confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    confidence = 0.5
    explicit_confidence = False
    if confidence_match:
        try:
            confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
            explicit_confidence = True
        except ValueError:
            pass
    return {
        "answer": answer,
        "confidence": confidence,
        "explicit_answer_field": bool(answer_match),
        "explicit_confidence_field": explicit_confidence,
        "status": (
            "explicit_blank_answer" if answer_match and not answer
            else "requested_schema_conformant" if answer_match and explicit_confidence
            else "accepted_by_current_parser_with_confidence_default" if answer_match
            else "accepted_by_current_parser_first_line_fallback"
        ),
        "usable": bool(answer),
    }


def hallucination_counts(prediction: Any, gold: Any) -> tuple[int, int, float | None]:
    """Return unsupported/generated counts against the complete normalized gold set.

    An empty prediction has a zero generated-answer denominator and therefore no
    hallucination rate; callers report it separately instead of assigning a rate.
    """
    split_items = load_sageqa_evaluator().split_answer_items
    actual = set(split_items(str(prediction or "")))
    expected = set(split_items(str(gold or "")))
    generated = len(actual)
    unsupported = len(actual - expected)
    return unsupported, generated, (unsupported / generated if generated else None)


def gold_for_representation(question: dict[str, Any], representation: str) -> Any:
    return question["ar_gold_answer"] if representation == "AR" else question["gold_answer"]


def independent_ar_items(value: Any) -> set[str]:
    """Small audit-only normalizer, independent of the scoring adapter."""
    parts = re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+", str(value or ""), flags=re.IGNORECASE)
    result = set()
    for part in parts:
        item = part.strip()
        if item.startswith("http"):
            item = item.rstrip("/>").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        item = re.sub(r"_\d{3,4}$", "", item)
        item = item.replace("_", " ")
        item = re.sub(r"([a-z])([A-Z])", r"\1 \2", item)
        item = re.sub(r"[^\w\s]", "", item).lower()
        item = re.sub(r"\b(?:a|an|the)\b", " ", item)
        item = " ".join(item.split())
        if item:
            result.add(item)
    return result


def independent_set_scores(prediction: Any, gold: Any) -> tuple[float, float]:
    predicted = independent_ar_items(prediction)
    expected = independent_ar_items(gold)
    if not predicted or not expected:
        return float(predicted == expected), 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(predicted == expected), f1


def tags(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    text = str(value or "").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {char for char in TAXONOMY if char in text}


def audit_inputs(benchmark: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_hash = sha256_file(STAGE / "core_llm_bench_v1_1.parquet")
    input_hash = sha256_file(STAGE / "model_input_manifest.csv")
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256 or input_hash != EXPECTED_INPUT_MANIFEST_SHA256:
        raise RuntimeError(f"Frozen input hash mismatch: benchmark={benchmark_hash}, input_manifest={input_hash}")
    if len(benchmark) != EXPECTED_QUESTIONS or benchmark.task_id.nunique() != EXPECTED_QUESTIONS:
        raise RuntimeError("Benchmark is not the frozen 9,048-question set")

    bin_mismatches = 0
    for row in benchmark.to_dict(orient="records"):
        value = int(row["raw_minimum_complete_primitive_tag_complexity"])
        if row["task_group"] == "BQA":
            expected_bin = "Low" if value == 1 else "Medium" if value == 2 else "High"
        else:
            expected_bin = "Low" if value <= 3 else "Medium" if value <= 5 else "High"
        bin_mismatches += str(row["complexity_bin"]) != expected_bin
    if bin_mismatches:
        raise RuntimeError(f"Final task-specific complexity-bin mismatches: {bin_mismatches}")

    canonical_inputs = read_csv(STAGE / "model_input_manifest.csv")
    input_cells = {(row["task_id"], row["representation"]): row for row in canonical_inputs}
    if len(canonical_inputs) != EXPECTED_PER_MODEL or len(input_cells) != EXPECTED_PER_MODEL:
        raise RuntimeError("Frozen model-input manifest is incomplete or duplicated")

    manifest = read_csv(STAGE / "primary_experiment_manifest.csv")
    if len(manifest) != EXPECTED_TOTAL:
        raise RuntimeError(f"Primary manifest has {len(manifest)} rows, expected {EXPECTED_TOTAL}")
    expected = {(row["task_id"], row["representation"], row["model"]): row for row in manifest}
    if len(expected) != EXPECTED_TOTAL:
        raise RuntimeError("Canonical primary manifest contains duplicate cells")
    primary_input_mismatches = sum(
        row["input_hash"] != input_cells[(row["task_id"], row["representation"])]["input_hash"]
        for row in manifest
    )
    if primary_input_mismatches:
        raise RuntimeError(f"Primary/canonical input-manifest hash mismatches: {primary_input_mismatches}")

    source_audits: list[dict[str, Any]] = []
    all_accepted: list[dict[str, Any]] = []
    for model, path in SOURCES.items():
        raw = load_jsonl(path)
        accepted = [row for row in raw if row.get("status") in ACCEPTED]
        rejected = [row for row in raw if row.get("status") not in ACCEPTED]
        keys = [(str(row.get("task_id")), str(row.get("representation")), str(row.get("model"))) for row in accepted]
        duplicates = len(keys) - len(set(keys))
        expected_keys = {key for key in expected if key[2] == model}
        actual_keys = set(keys)
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        mismatches = Counter()
        parse_mismatches = 0
        blank_answer_corrections = 0
        unexpected_parse_mismatches = 0
        providers = Counter()
        returned_models = Counter()
        for row, key in zip(accepted, keys):
            canonical = expected.get(key)
            if canonical is None:
                continue
            for field in ("semantic_key", "dataset", "hop", "task", "input_hash"):
                if str(row.get(field, "")) != str(canonical.get(field, "")):
                    mismatches[field] += 1
            reparsed = parse_response(raw_text(row), str(row["task"]))
            recorded = row.get("parsed_response") or {}
            if reparsed.get("answer") != recorded.get("answer") or float(reparsed.get("confidence", .5)) != float(recorded.get("confidence", .5)):
                parse_mismatches += 1
                if reparsed["status"] == "explicit_blank_answer" and not reparsed["answer"]:
                    blank_answer_corrections += 1
                else:
                    unexpected_parse_mismatches += 1
            provider = str((row.get("raw_provider_response") or {}).get("provider") or row.get("observed_provider_backend") or "")
            returned = str(row.get("returned_model_identifier") or row.get("returned_model") or (row.get("raw_provider_response") or {}).get("model") or "")
            providers[provider] += 1
            returned_models[returned] += 1

        expected_route = {
            "Gemini 2.5 Flash-Lite": ({"Google AI Studio"}, {"google/gemini-2.5-flash-lite"}),
            "Qwen3-30B-A3B-Instruct": ({"Alibaba"}, {"qwen/qwen3-30b-a3b-instruct-2507"}),
            "GPT-5 mini": ({"OpenAI"}, {"openai/gpt-5-mini"}),
        }[model]
        provider_mismatches = sum(count for value, count in providers.items() if value not in expected_route[0])
        model_mismatches = sum(count for value, count in returned_models.items() if value not in expected_route[1])
        audit = {
            "model": model,
            "source": str(path.relative_to(ROOT)),
            "raw_rows": len(raw),
            "accepted_observations": len(accepted),
            "usable": sum(row.get("status") == "usable" for row in accepted),
            "malformed_accepted": sum(row.get("status") == "malformed_response" for row in accepted),
            "diagnostic_nonaccepted_rows": len(rejected),
            "duplicate_accepted_cells": duplicates,
            "missing_cells": len(missing),
            "extra_cells": len(extra),
            "semantic_or_input_mismatches": sum(mismatches.values()),
            "mismatch_fields": dict(mismatches),
            "parser_replay_mismatches": parse_mismatches,
            "intentional_blank_answer_parser_corrections": blank_answer_corrections,
            "unexpected_parser_mismatches": unexpected_parse_mismatches,
            "provider_mismatches": provider_mismatches,
            "returned_model_mismatches": model_mismatches,
            "observed_providers": ";".join(sorted(providers)),
            "returned_model_identifiers": ";".join(sorted(returned_models)),
        }
        fatal = (len(accepted) != EXPECTED_PER_MODEL or duplicates or missing or extra or mismatches or unexpected_parse_mismatches or provider_mismatches or model_mismatches)
        if fatal:
            raise RuntimeError(f"Observation integrity failure for {model}: {audit}")
        source_audits.append(audit)
        all_accepted.extend(accepted)

    probe_path = ROOT / "data/output/v1.1.0-gpt-openrouter-primary/gpt_identity_probe.json"
    metadata_path = ROOT / "data/output/v1.1.0-gpt-openrouter-primary/gpt_execution_metadata.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact = "gpt-5-mini-2025-08-07"
    if metadata.get("non_secret_effective_config", {}).get("scientific_target") != exact:
        raise RuntimeError("GPT execution metadata does not record the dated scientific target")
    if not probe.get("exact_august_7_snapshot_identity_established") or probe.get("successful_requested_model") != "openai/gpt-5-mini-2025-08-07":
        raise RuntimeError("GPT dated-snapshot identity probe did not pass")

    return all_accepted, {
        "status": "passed",
        "benchmark_questions": len(benchmark),
        "canonical_manifest_rows": len(manifest),
        "canonical_input_cells": len(input_cells),
        "primary_to_canonical_input_hash_mismatches": primary_input_mismatches,
        "complexity_bin_mismatches": bin_mismatches,
        "accepted_observations": len(all_accepted),
        "benchmark_sha256": benchmark_hash,
        "model_input_manifest_sha256": input_hash,
        "gpt_exact_scientific_identifier": exact,
        "gpt_explicit_dated_request_identifier": probe["successful_requested_model"],
        "gpt_response_returned_identifier": probe["returned_model"],
        "gpt_identity_basis": probe["identity_basis"],
        "gpt_identity_note": "The response model field is the OpenRouter alias; dated identity is established by the successful explicit dated request and recorded scientific target, not by the alias alone.",
        "sources": source_audits,
    }


def aggregate(rows: list[dict[str, Any]], dimensions: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[d] for d in dimensions)].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        oeqa_items = [item for item in items if item["task"] == "OEQA"]
        generated = sum(int(item["oeqa_generated_answer_count"]) for item in oeqa_items)
        unsupported = sum(int(item["oeqa_unsupported_answer_count"]) for item in oeqa_items)
        output.append({
            **dict(zip(dimensions, key)),
            "n": len(items),
            "answer_exact_match": sum(float(item["answer_exact_match"]) for item in items) / len(items),
            "answer_f1": sum(float(item["answer_f1"]) for item in items) / len(items),
            "confidence_correctness_alignment": sum(float(item["confidence_correctness_alignment"]) for item in items) / len(items),
            "oeqa_n": len(oeqa_items),
            "oeqa_empty_predictions": sum(int(item["oeqa_empty_prediction"]) for item in oeqa_items),
            "oeqa_generated_answers": generated,
            "oeqa_unsupported_answers": unsupported,
            "oeqa_hallucination_rate": unsupported / generated if generated else "",
            "malformed_accepted": sum(item["observation_status"] == "malformed_response" for item in items),
        })
    return output


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def latex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], caption: str, label: str, percent: set[str] | None = None) -> str:
    percent = percent or set()
    align = "l" * len(columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(title for _, title in columns) + r" \\", r"\midrule"]
    for row in rows:
        cells = []
        for field, _ in columns:
            value = row.get(field, "")
            if field in percent and value != "":
                value = f"{100 * float(value):.2f}"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    benchmark = pd.read_parquet(STAGE / "core_llm_bench_v1_1.parquet")
    observations, integrity = audit_inputs(benchmark)
    bench = {str(row["task_id"]): row for row in benchmark.to_dict(orient="records")}

    scored: list[dict[str, Any]] = []
    for observation in observations:
        question = bench[str(observation["task_id"])]
        parsed = parse_response(raw_text(observation), str(observation["task"]))
        representation = str(observation["representation"])
        gold = gold_for_representation(question, representation)
        em, f1, precision, recall = answer_set_scores(parsed["answer"], gold)
        hall: float | str = ""
        empty_prediction: int | str = ""
        generated_count: int | str = ""
        unsupported_count: int | str = ""
        if question["task_group"] == "OEQA":
            unsupported_count, generated_count, hall_value = hallucination_counts(parsed["answer"], gold)
            empty_prediction = int(generated_count == 0)
            hall = "" if hall_value is None else hall_value
        scored.append({
            "task_id": str(observation["task_id"]),
            "semantic_key": question["semantic_key"],
            "model": observation["model"],
            "dataset": question["dataset"],
            "task": question["task_group"],
            "representation": representation,
            "hop": question["hop"],
            "complexity": int(question["raw_minimum_complete_primitive_tag_complexity"]),
            "complexity_bin": question["complexity_bin"],
            "primitive_reasoning_tags": "".join(sorted(tags(question["primitive_reasoning_tags"]))),
            "observation_status": observation["status"],
            "parser_status": parsed["status"],
            "empty_prediction": int(not parsed["usable"]),
            "confirmed_defective_ar_prompt": int(representation == "AR" and str(observation["task_id"]) in CONFIRMED_DEFECTIVE_AR_TASK_IDS),
            "confidence": parsed["confidence"],
            "answer_exact_match": em,
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "confidence_correctness_alignment": confidence_correctness_alignment(parsed["confidence"], f1),
            "oeqa_empty_prediction": empty_prediction,
            "oeqa_generated_answer_count": generated_count,
            "oeqa_unsupported_answer_count": unsupported_count,
            "oeqa_hallucination_rate": hall,
        })

    per_fields = list(scored[0])
    write_csv(output / "csv/per_observation_scores.csv", scored, per_fields)
    overall = aggregate(scored, ["model"])
    full = aggregate(scored, ["model", "dataset", "task", "representation", "hop"])
    by_dataset = aggregate(scored, ["model", "dataset"])
    by_task = aggregate(scored, ["model", "task"])
    by_representation = aggregate(scored, ["model", "representation"])
    by_hop = aggregate(scored, ["model", "hop"])
    complexity = aggregate(scored, ["model", "task", "complexity_bin"])
    complexity_detailed = aggregate(scored, ["model", "dataset", "task", "representation", "hop", "complexity_bin"])
    for name, rows in (
        ("overall_by_model", overall), ("results_full_factorial", full), ("results_by_dataset", by_dataset),
        ("results_by_task", by_task), ("results_by_representation", by_representation), ("results_by_hop", by_hop),
        ("explanation_complexity", complexity), ("explanation_complexity_detailed", complexity_detailed),
    ):
        write_csv(output / f"csv/{name}.csv", rows)

    sensitivity_scored = [row for row in scored if not row["confirmed_defective_ar_prompt"]]
    sensitivity_outputs = {
        "overall_by_model": aggregate(sensitivity_scored, ["model"]),
        "results_by_representation": aggregate(sensitivity_scored, ["model", "representation"]),
        "results_full_factorial": aggregate(sensitivity_scored, ["model", "dataset", "task", "representation", "hop"]),
    }
    for name, rows in sensitivity_outputs.items():
        write_csv(output / f"csv/without_confirmed_defective_ar/{name}.csv", rows)
    defective_rows = [
        {
            "task_id": task_id,
            "status": "confirmed_defective_ar_prompt",
            "basis": {
                "1274": "gold class and supporting type statement absent from AR context",
                "4094": "displayed AR type contradicts the AR gold class set",
                "5165": "required inverse-property identity absent from AR context",
            }[task_id],
        }
        for task_id in sorted(CONFIRMED_DEFECTIVE_AR_TASK_IDS, key=int)
    ]
    write_csv(output / "audit/confirmed_defective_ar_questions.csv", defective_rows)

    scored_by_key = {(row["task_id"], row["representation"], row["model"]): row for row in scored}
    sample_rows = []
    for task_id in ("1274", "4094", "5165", "6967"):
        question = bench[task_id]
        for model in SOURCES:
            observation = next(
                row for row in observations
                if str(row["task_id"]) == task_id and row["representation"] == "AR" and row["model"] == model
            )
            parsed = parse_response(raw_text(observation), str(observation["task"]))
            gold = question["ar_gold_answer"]
            independent_em, independent_f1 = independent_set_scores(parsed["answer"], gold)
            evaluator_row = scored_by_key[(task_id, "AR", model)]
            verified = (
                abs(independent_em - float(evaluator_row["answer_exact_match"])) < 1e-12
                and abs(independent_f1 - float(evaluator_row["answer_f1"])) < 1e-12
            )
            if not verified:
                raise RuntimeError(f"Independent AR score mismatch for task {task_id}, {model}")
            sample_rows.append({
                "task_id": task_id,
                "model": model,
                "confirmed_defective_ar_prompt": int(task_id in CONFIRMED_DEFECTIVE_AR_TASK_IDS),
                "raw_response": raw_text(observation),
                "corrected_parsed_prediction": parsed["answer"],
                "ar_gold_answer": gold,
                "independent_exact_match": independent_em,
                "independent_f1": independent_f1,
                "evaluator_exact_match": evaluator_row["answer_exact_match"],
                "evaluator_f1": evaluator_row["answer_f1"],
                "verified": verified,
            })
    write_csv(output / "audit/independent_ar_sample_verification.csv", sample_rows)

    previous_by_representation = {
        (row["model"], row["representation"]): row
        for row in read_csv(PREVIOUS_OUTPUT / "csv/results_by_representation.csv")
    }
    comparison_rows = []
    for current in by_representation:
        previous = previous_by_representation[(current["model"], current["representation"])]
        comparison_rows.append({
            "model": current["model"],
            "representation": current["representation"],
            "n": current["n"],
            "previous_answer_exact_match": previous["answer_exact_match"],
            "corrected_answer_exact_match": current["answer_exact_match"],
            "answer_exact_match_delta": float(current["answer_exact_match"]) - float(previous["answer_exact_match"]),
            "previous_answer_f1": previous["answer_f1"],
            "corrected_answer_f1": current["answer_f1"],
            "answer_f1_delta": float(current["answer_f1"]) - float(previous["answer_f1"]),
            "previous_oeqa_hallucination_rate": previous["oeqa_hallucination_rate"],
            "corrected_oeqa_hallucination_rate": current["oeqa_hallucination_rate"],
            "hallucination_rate_delta": float(current["oeqa_hallucination_rate"]) - float(previous["oeqa_hallucination_rate"]),
            "corrected_oeqa_empty_predictions": current["oeqa_empty_predictions"],
            "corrected_oeqa_generated_answers": current["oeqa_generated_answers"],
        })
    write_csv(output / "csv/comparison_with_previous_by_representation.csv", comparison_rows)

    coverage_rows = []
    tag_performance = []
    benchmark_rows = benchmark.to_dict(orient="records")
    for tag in TAXONOMY:
        for task in ("ALL", "BQA", "OEQA"):
            population = [row for row in benchmark_rows if task == "ALL" or row["task_group"] == task]
            covered_ids = {str(row["task_id"]) for row in population if tag in tags(row["primitive_reasoning_tags"]) or (tag == "M" and str(row["m_status"]) != "never")}
            coverage_rows.append({"tag": tag, "task": task, "covered_questions": len(covered_ids), "total_questions": len(population), "coverage_rate": len(covered_ids) / len(population) if population else 0.0})
            selected = [row for row in scored if str(row["task_id"]) in covered_ids]
            if selected:
                perf = aggregate(selected, ["model"])
                for item in perf:
                    tag_performance.append({"tag": tag, "task": task, **item})
    write_csv(output / "csv/reasoning_tag_coverage.csv", coverage_rows)
    write_csv(output / "csv/reasoning_tag_performance.csv", tag_performance)

    stats_rows = []
    for dims in (["dataset", "hop", "task_group"], ["dataset", "hop"], ["task_group"], ["complexity_bin"]):
        counts = benchmark.groupby(dims, dropna=False).size().reset_index(name="questions")
        for row in counts.to_dict(orient="records"):
            stats_rows.append({"breakdown": "+".join(dims), **{key: row.get(key, "") for key in ("dataset", "hop", "task_group", "complexity_bin")}, "questions": int(row["questions"])})
    write_csv(output / "csv/final_dataset_statistics.csv", stats_rows, ["breakdown", "dataset", "hop", "task_group", "complexity_bin", "questions"])
    write_csv(output / "csv/observation_integrity.csv", integrity["sources"])
    write_json(output / "audit/integrity_audit.json", integrity)

    metric_cols = [("model", "Model"), ("n", "N"), ("answer_exact_match", "EM"), ("answer_f1", "F1"), ("confidence_correctness_alignment", "Conf.-corr."), ("oeqa_empty_predictions", "OEQA empty"), ("oeqa_generated_answers", "Generated"), ("oeqa_hallucination_rate", "OEQA hall.")]
    percent = {"answer_exact_match", "answer_f1", "confidence_correctness_alignment", "oeqa_hallucination_rate"}
    (output / "latex").mkdir(parents=True, exist_ok=True)
    (output / "latex/overall_results.tex").write_text(latex_table(overall, metric_cols, "Final CORE-LLM-Bench v1.1 results (percent).", "tab:core-v11-overall", percent), encoding="utf-8", newline="\n")
    (output / "latex/overall_results_without_confirmed_defective_ar.tex").write_text(
        latex_table(sensitivity_outputs["overall_by_model"], metric_cols, "Corrected v1.1 sensitivity results excluding three confirmed defective AR prompts (percent).", "tab:core-v11-overall-ar-sensitivity", percent),
        encoding="utf-8", newline="\n"
    )
    for name, rows, dim, title in (
        ("results_by_dataset", by_dataset, "dataset", "Dataset"), ("results_by_task", by_task, "task", "Task"),
        ("results_by_representation", by_representation, "representation", "Representation"), ("results_by_hop", by_hop, "hop", "Hop"),
    ):
        cols = [("model", "Model"), (dim, title), ("n", "N"), ("answer_exact_match", "EM"), ("answer_f1", "F1"), ("confidence_correctness_alignment", "Conf.-corr."), ("oeqa_empty_predictions", "OEQA empty"), ("oeqa_generated_answers", "Generated"), ("oeqa_hallucination_rate", "OEQA hall.")]
        (output / f"latex/{name}.tex").write_text(latex_table(rows, cols, f"Final v1.1 results by {title.lower()} (percent).", f"tab:core-v11-{dim}", percent), encoding="utf-8", newline="\n")
    comp_cols = [("model", "Model"), ("task", "Task"), ("complexity_bin", "Complexity"), ("n", "N"), ("answer_exact_match", "EM"), ("answer_f1", "F1")]
    (output / "latex/explanation_complexity.tex").write_text(latex_table(complexity, comp_cols, "Answer quality by final task-specific explanation-complexity bin (percent).", "tab:core-v11-complexity", percent), encoding="utf-8", newline="\n")
    coverage_all = [row for row in coverage_rows if row["task"] == "ALL"]
    (output / "latex/reasoning_tag_coverage.tex").write_text(latex_table(coverage_all, [("tag", "Tag"), ("covered_questions", "Questions"), ("coverage_rate", "Coverage")], "Final benchmark reasoning-tag coverage (percent).", "tab:core-v11-tags", {"coverage_rate"}), encoding="utf-8", newline="\n")

    model_lines = []
    for row in overall:
        model_lines.append(f"| {row['model']} | {row['n']:,} | {100*row['answer_exact_match']:.2f} | {100*row['answer_f1']:.2f} | {100*row['confidence_correctness_alignment']:.2f} | {row['oeqa_empty_predictions']:,} | {row['oeqa_generated_answers']:,} | {100*float(row['oeqa_hallucination_rate']):.2f} | {row['malformed_accepted']} |")
    sensitivity_lines = []
    for row in sensitivity_outputs["overall_by_model"]:
        sensitivity_lines.append(f"| {row['model']} | {row['n']:,} | {100*row['answer_exact_match']:.2f} | {100*row['answer_f1']:.2f} | {100*row['confidence_correctness_alignment']:.2f} | {row['oeqa_empty_predictions']:,} | {row['oeqa_generated_answers']:,} | {100*float(row['oeqa_hallucination_rate']):.2f} |")
    nl_fs_answer_changed = any(
        abs(float(row["answer_exact_match_delta"])) > 1e-12 or abs(float(row["answer_f1_delta"])) > 1e-12
        for row in comparison_rows if row["representation"] in {"NL", "FS"}
    )
    report = f"""# CORE-LLM-Bench v1.1 final offline evaluation

Status: **complete**. The evaluation used {EXPECTED_QUESTIONS:,} frozen questions and {EXPECTED_TOTAL:,} accepted observations. No model/API request is made by the evaluation script.

## Integrity and identity

- All three models contribute exactly {EXPECTED_PER_MODEL:,} unique accepted observations; missing, duplicate, extra, semantic, input-hash, provider, returned-model, and unexpected parser-replay mismatches are all zero.
- The corrected parser classifies explicit blank `ANSWER:` fields as empty in every representation. These intentional replay differences are recorded in `observation_integrity.csv`.
- Malformed-but-accepted outputs remain in every relevant denominator (Gemini 5; Qwen 58; GPT 7).
- Qwen's nine non-accepted transient diagnostic rows are not observations and are reported separately in the integrity audit.
- GPT exact scientific identifier: `{integrity['gpt_exact_scientific_identifier']}`. The explicit dated OpenRouter request was `{integrity['gpt_explicit_dated_request_identifier']}` and succeeded through provider `OpenAI`; response records returned alias `{integrity['gpt_response_returned_identifier']}`. The alias alone is not used as dated-snapshot evidence.

## Overall results (%)

| Model | N | Answer EM | Answer F1 | Confidence-correctness | OEQA empty | Generated answers | OEQA hallucination | Malformed accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

## Sensitivity excluding confirmed defective AR prompts (%)

| Model | N | Answer EM | Answer F1 | Confidence-correctness | OEQA empty | Generated answers | OEQA hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sensitivity_lines)}

Exactly {len(CONFIRMED_DEFECTIVE_AR_TASK_IDS)} AR questions are explicitly flagged as confirmed defective (task IDs {', '.join(sorted(CONFIRMED_DEFECTIVE_AR_TASK_IDS, key=int))}). The prevalence of additional semantic-support defects remains unresolved; this sensitivity analysis must not be read as proving that all other AR prompts are semantically supported.

## Comparison with the previous evaluation

AR now uses `ar_gold_answer`. NL/FS answer EM and F1 {'changed' if nl_fs_answer_changed else 'did not change'}; their hallucination values changed because the corrected definition uses the complete representation-specific gold set and excludes empty predictions from the generated-answer denominator. The detailed representation-level deltas are in `csv/comparison_with_previous_by_representation.csv`.

## Methods

Answer EM/F1 use the hash-pinned SAGE-QA ontology answer-set evaluator (`{SAGEQA_EVALUATOR_CANONICAL_SHA256}`, source commit `{SAGEQA_SOURCE_COMMIT}`). AR uses `ar_gold_answer`; NL/FS use `gold_answer`. Confidence-correctness alignment is `1 - abs(confidence - per-question Answer F1)` with the documented 0.5 fallback. OEQA hallucination is the micro-average `unsupported generated answers / generated answers` against the complete representation-specific normalized gold set. Empty predictions are counted separately and have an undefined per-observation rate; an aggregate with zero generated answers is emitted blank rather than assigned a value. BQA rows are excluded. Complexity is the frozen minimum complete primitive-tag count with BQA bins 1/2/3+ and OEQA bins 1-3/4-5/6+. `M` is metadata and is excluded from primitive complexity.

Detailed factorial, complexity, reasoning-tag, dataset-statistics, and per-observation outputs are in `csv/`; manuscript tables are in `latex/`; the machine-readable provenance gate is in `audit/integrity_audit.json`.
"""
    (output / "FINAL_RESULTS_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    generated = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json")
    reproducibility = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__)),
        "offline_only": True,
        "benchmark_sha256": integrity["benchmark_sha256"],
        "model_input_manifest_sha256": integrity["model_input_manifest_sha256"],
        "source_observation_sha256": {model: sha256_file(path) for model, path in SOURCES.items()},
        "generated_files": {str(path.relative_to(output)): sha256_file(path) for path in generated},
    }
    write_json(output / "REPRODUCIBILITY_MANIFEST.json", reproducibility)
    print(json.dumps({"status": "complete", "output": str(output), "observations": len(scored)}, indent=2))


if __name__ == "__main__":
    main()
