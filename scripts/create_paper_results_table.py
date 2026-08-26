#!/usr/bin/env python3
"""Regenerate Chapter 4 ontology tables from frozen saved predictions only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.llm_pipeline.sageqa_answer_metrics import (
        BenchmarkMismatchError,
        SAGEQA_EVALUATOR_SHA256,
        benchmark_csv_path,
        detect_models,
        read_csv_rows,
        sageqa_evaluator_path,
        score_checkpoint_rows,
        validate_checkpoint_rows,
    )
except ImportError:
    from llm_pipeline.sageqa_answer_metrics import (
        BenchmarkMismatchError,
        SAGEQA_EVALUATOR_SHA256,
        benchmark_csv_path,
        detect_models,
        read_csv_rows,
        sageqa_evaluator_path,
        score_checkpoint_rows,
        validate_checkpoint_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("data/output/final_benchmark_llm_results")
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "chapter4_sageqa"
SETTINGS = {"nl": "NL", "sparql": "FS", "abs": "AR"}
DATASETS = ("FamilyOWL", "pizza_100", "pizza_250", "OWL2Bench")
CHAPTER4_DATASETS = set(DATASETS)
DATASET_LABELS = {
    "FamilyOWL": "Family",
    "pizza_100": "Pizza 100",
    "pizza_250": "Pizza 250",
    "OWL2Bench": "OWL2Bench",
}
DATASET_SLUGS = {
    "FamilyOWL": "family",
    "pizza_100": "pizza_100",
    "pizza_250": "pizza_250",
    "OWL2Bench": "owl2bench",
}
MODEL_LABELS = {
    "openai_gpt_5_mini_2025_08_07": "GPT-5 mini",
    "openrouter_google_gemini_2_5_flash_lite": "Gemini 2.5 Flash-Lite",
    "openrouter_qwen_qwen3_30b_a3b_instruct_2507": "Qwen3-30B-A3B-Instruct",
}
MODEL_ORDER = {name: index for index, name in enumerate(MODEL_LABELS)}

IDENTITY_COLUMNS = ["dataset", "model_id", "model", "setting"]
METRIC_COLUMNS = [
    "binary_answer_em_1-hop",
    "binary_answer_em_2-hop",
    "binary_confidence_correctness_alignment_1-hop",
    "binary_confidence_correctness_alignment_2-hop",
    "open_ended_answer_em_1-hop",
    "open_ended_answer_em_2-hop",
    "open_ended_answer_f1_1-hop",
    "open_ended_answer_f1_2-hop",
    "open_ended_confidence_correctness_alignment_1-hop",
    "open_ended_confidence_correctness_alignment_2-hop",
    "open_ended_hallucination_1-hop",
    "open_ended_hallucination_2-hop",
]
COUNT_COLUMNS = [
    "binary_n_1-hop",
    "binary_n_2-hop",
    "open_ended_n_1-hop",
    "open_ended_n_2-hop",
]
LOWER_IS_BETTER = {
    "open_ended_hallucination_1-hop",
    "open_ended_hallucination_2-hop",
}

max_csv_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_csv_field_size)
        break
    except OverflowError:
        max_csv_field_size //= 10


class DuplicateModelSettingSourceError(ValueError):
    """More than one benchmark-valid artifact claims the same result cell."""


@dataclass(frozen=True)
class ValidatedSource:
    dataset: str
    hop: str
    setting: str
    model_name: str
    checkpoint_path: Path
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Chapter 4 SAGE-QA Answer EM/F1 CSV and LaTeX tables."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decimals", type=int, default=1)
    return parser.parse_args()


def parse_dataset_hop_dir(path: Path) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+)_(1hop|2hop)", path.name)
    return (match.group(1), match.group(2)) if match else None


def iter_checkpoint_sources(base_dir: Path):
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {base_dir}")
    for dataset_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        parsed = parse_dataset_hop_dir(dataset_dir)
        if parsed is None:
            continue
        dataset, hop = parsed
        if dataset not in CHAPTER4_DATASETS:
            continue
        for model_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for setting in SETTINGS:
                run_dir = model_dir / setting
                checkpoint_path = run_dir / "LATEST_checkpoint.csv"
                if checkpoint_path.is_file():
                    yield dataset, hop, setting, run_dir, checkpoint_path


def discover_validated_sources(
    base_dir: Path, project_root: Path = PROJECT_ROOT
) -> tuple[dict[tuple[str, str, str, str], ValidatedSource], list[str]]:
    """Validate benchmark identity/content before selecting any result cell."""

    sources: dict[tuple[str, str, str, str], ValidatedSource] = {}
    rejected: list[str] = []
    benchmark_cache: dict[tuple[str, str, str], list[dict[str, str]]] = {}

    for dataset, hop, setting, run_dir, checkpoint_path in iter_checkpoint_sources(
        base_dir
    ):
        benchmark_path = benchmark_csv_path(project_root, dataset, hop, setting)
        if not benchmark_path.is_file():
            continue
        cache_key = (dataset, hop, setting)
        benchmark_rows = benchmark_cache.setdefault(
            cache_key, read_csv_rows(benchmark_path)
        )
        checkpoint_rows = read_csv_rows(checkpoint_path)
        try:
            validate_checkpoint_rows(checkpoint_rows, benchmark_rows, checkpoint_path)
        except BenchmarkMismatchError as exc:
            rejected.append(str(exc))
            continue

        for model_name in detect_models(checkpoint_rows):
            key = (dataset, hop, setting, model_name)
            source = ValidatedSource(
                dataset, hop, setting, model_name, checkpoint_path, run_dir
            )
            if key in sources:
                raise DuplicateModelSettingSourceError(
                    "Duplicate benchmark-valid model-setting source for "
                    f"{key}: {sources[key].checkpoint_path} and {checkpoint_path}"
                )
            sources[key] = source

    return sources, rejected


def display_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", "-"))


def pct_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text.removesuffix("%"))
    except ValueError:
        return None


def read_hallucination(source: ValidatedSource) -> float | None:
    """Read the persisted OEQA hallucination mean.

    The Chapter 4 hallucination metric returns ``None`` for binary questions,
    so its persisted overall mean is the open-ended-only mean. This value is
    not an Answer EM/F1 metric and is intentionally not redefined here.
    """

    summary_path = source.run_dir / "metrics" / f"{source.setting}_key_findings_summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open(encoding="utf-8") as handle:
        summaries = json.load(handle).get("key_findings_summary", {})
    return pct_to_float(
        summaries.get(source.model_name, {})
        .get("overall_metrics", {})
        .get("hallucination_score")
    )


def ontology_hallucination_score(
    prediction: Any, gold: Any, context: list[str]
) -> float:
    """Apply the existing deterministic Chapter 4 OEQA hallucination formula."""

    def parse_answers(value: Any) -> list[str]:
        text = str(value or "").lower().strip()
        if ";" in text:
            answers = text.split(";")
        elif "," in text:
            answers = text.split(",")
        elif "\n" in text:
            answers = text.split("\n")
        else:
            answers = [text]
        cleaned = []
        for answer in answers:
            answer = re.sub(r"[^\w\s]", "", answer.strip())
            answer = re.sub(r"\s+", " ", answer).strip()
            if answer:
                cleaned.append(answer)
        return cleaned

    def normalize(value: Any) -> str:
        text = str(value).lower().strip().replace("_", " ")
        text = re.sub(r"\s*\d{4}$", "", text)
        return re.sub(r"\s+", " ", text).strip()

    actual_answers = parse_answers(prediction)
    if not actual_answers:
        return 1.0
    expected_answers = parse_answers(gold)
    context_text = " ".join(context)
    valid_entities = set(expected_answers)
    valid_entities.update(re.findall(r"\b[a-z]+(?:_[a-z]+)*_\d{4}\b", context_text.lower()))
    valid_entities.update(
        value.lower() for value in re.findall(r"\b[A-Z][a-zA-Z]+\b", context_text)
    )
    valid_entities.update(
        value.lower() for value in re.findall(r"<[^>]+#([^>]+)>", context_text)
    )
    normalized_valid = [normalize(value) for value in valid_entities]
    hallucinated = 0
    for answer in actual_answers:
        normalized = normalize(answer)
        if not any(
            normalized == valid
            or normalized in valid
            or valid in normalized
            or normalized.replace(" ", "") == valid.replace(" ", "")
            for valid in normalized_valid
        ):
            hallucinated += 1
    return hallucinated / len(actual_answers)


def recompute_hallucination(
    checkpoint_rows: list[dict[str, str]], model_name: str
) -> float:
    answer_column = f"{model_name}_final_answer"
    scores = []
    for row in checkpoint_rows:
        if str(row.get("Answer Type", "BIN")).strip().upper() == "BIN":
            continue
        context = [
            f"Task Type: {row.get('Task Type', '')}",
            f"Answer Type: {row.get('Answer Type', 'BIN')}",
            f"Ontology: {row.get('Root Entity', '')}",
            f"SPARQL Query: {row.get('SPARQL Query', '')}",
        ]
        scores.append(
            ontology_hallucination_score(
                row.get(answer_column, "") or "", row.get("Answer", ""), context
            )
        )
    if not scores:
        raise ValueError(f"No open-ended rows for hallucination metric: {model_name}")
    return 100.0 * sum(scores) / len(scores)


def _expected_source_keys() -> set[tuple[str, str, str, str]]:
    return {
        (dataset, hop, setting, model)
        for dataset in DATASETS
        for hop in ("1hop", "2hop")
        for setting in SETTINGS
        for model in MODEL_LABELS
    }


def collect_rows(
    base_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources, rejected = discover_validated_sources(base_dir)
    missing = sorted(_expected_source_keys() - set(sources))
    extra = sorted(set(sources) - _expected_source_keys())
    if missing or extra:
        raise BenchmarkMismatchError(
            f"Incomplete Chapter 4 result matrix: missing={missing}, extra={extra}"
        )

    table: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows_cache: dict[Path, list[dict[str, str]]] = {}
    scored_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    binary_em_f1_mismatches: list[str] = []
    empty_or_error_predictions = 0

    for key in sorted(sources):
        source = sources[key]
        hop_label = "1-hop" if source.hop == "1hop" else "2-hop"
        setting_label = SETTINGS[source.setting]
        table_key = (source.dataset, source.model_name, setting_label)
        row = table.setdefault(
            table_key,
            {
                "dataset": source.dataset,
                "model_id": source.model_name,
                "model": display_model_name(source.model_name),
                "setting": setting_label,
            },
        )
        checkpoint_rows = rows_cache.setdefault(
            source.checkpoint_path, read_csv_rows(source.checkpoint_path)
        )
        scored = scored_cache.setdefault(
            (source.checkpoint_path, source.model_name),
            score_checkpoint_rows(checkpoint_rows, source.model_name),
        )
        answer_column = f"{source.model_name}_final_answer"
        empty_or_error_predictions += sum(
            not str(item.get(answer_column, "") or "").strip()
            or str(item.get(answer_column, "") or "").strip().upper().startswith(
                ("[ERROR]", "ERROR")
            )
            for item in checkpoint_rows
        )
        for item in scored["per_question"]:
            if item["answer_type"] == "BIN" and item["answer_em"] != item["answer_f1"]:
                binary_em_f1_mismatches.append("::".join((*key, item["task_id"])))

        for answer_bucket, column_prefix in (("binary", "binary"), ("open", "open_ended")):
            metrics = scored["aggregates"][answer_bucket]
            row[f"{column_prefix}_n_{hop_label}"] = metrics["n"]
            row[f"{column_prefix}_answer_em_{hop_label}"] = 100.0 * metrics["answer_em"]
            if answer_bucket == "open":
                row[f"{column_prefix}_answer_f1_{hop_label}"] = 100.0 * metrics["answer_f1"]
            row[f"{column_prefix}_confidence_correctness_alignment_{hop_label}"] = (
                100.0 * metrics["confidence_correctness_alignment"]
            )
        row[f"open_ended_hallucination_{hop_label}"] = recompute_hallucination(
            checkpoint_rows, source.model_name
        )

    if binary_em_f1_mismatches:
        raise AssertionError(
            "Binary Answer EM/F1 differ for "
            f"{len(binary_em_f1_mismatches)} rows: {binary_em_f1_mismatches[:5]}"
        )

    rows = list(table.values())
    rows.sort(
        key=lambda item: (
            DATASETS.index(str(item["dataset"])),
            MODEL_ORDER.get(str(item["model_id"]), 99),
            list(SETTINGS.values()).index(str(item["setting"])),
        )
    )

    unique_paths = sorted({source.checkpoint_path for source in sources.values()})
    manifest = {
        "scorer": {
            "implementation": "SAGE-QA answer_set_scores()/evaluate()",
            "path": str(sageqa_evaluator_path().resolve()),
            "sha256": SAGEQA_EVALUATOR_SHA256,
        },
        "source_policy": "LATEST_checkpoint.csv after exact Task ID, Answer Type, and Answer validation against the setting-specific benchmark CSV",
        "hallucination_policy": "Existing deterministic Chapter 4 OEQA entity hallucination mean recomputed over every saved open-ended row; empty/error predictions score 1.0",
        "validated_result_cells": len(sources),
        "expected_result_cells": len(_expected_source_keys()),
        "unique_checkpoint_files": len(unique_paths),
        "scored_question_model_observations": sum(
            int(row["binary_n_1-hop"])
            + int(row["binary_n_2-hop"])
            + int(row["open_ended_n_1-hop"])
            + int(row["open_ended_n_2-hop"])
            for row in rows
        ),
        "empty_or_error_predictions_preserved": empty_or_error_predictions,
        "binary_answer_em_f1_mismatch_count": len(binary_em_f1_mismatches),
        "rejected_checkpoint_count": len(rejected),
        "rejected_checkpoints": rejected,
        "checkpoint_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in unique_paths
        },
    }
    return rows, manifest


def format_cell(value: float | None, decimals: int) -> str:
    if value is None or str(value).strip() == "":
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{decimals}f}"


def write_csv(rows: Iterable[dict[str, Any]], output_csv: Path, decimals: int) -> None:
    rows = list(rows)
    columns = IDENTITY_COLUMNS + COUNT_COLUMNS + METRIC_COLUMNS
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    if column in IDENTITY_COLUMNS + COUNT_COLUMNS
                    else format_cell(row.get(column), decimals)
                    for column in columns
                }
            )


def latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
    )


def best_values(
    rows: list[dict[str, Any]], decimals: int
) -> dict[tuple[str, str], float]:
    """Select the best displayed setting within each model and metric column."""

    best: dict[tuple[str, str], float] = {}
    for model_id in MODEL_LABELS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        for column in METRIC_COLUMNS:
            values = [
                round(float(row[column]), decimals)
                for row in model_rows
                if row.get(column) is not None
            ]
            if values:
                key = (model_id, column)
                best[key] = (
                    min(values) if column in LOWER_IS_BETTER else max(values)
                )
    return best


def latex_metric_cell(
    row: dict[str, Any],
    column: str,
    best: dict[tuple[str, str], float],
    decimals: int,
) -> str:
    value = row.get(column)
    text = format_cell(value, decimals) or "--"
    key = (str(row["model_id"]), column)
    if value is not None and key in best and math.isclose(
        round(float(value), decimals), best[key], rel_tol=0.0, abs_tol=1e-12
    ):
        return rf"\textbf{{{text}}}"
    return text


def write_latex_dataset(
    rows: list[dict[str, Any]], output_tex: Path, decimals: int
) -> None:
    if not rows:
        raise ValueError(f"No dataset rows for {output_tex}")
    dataset = str(rows[0]["dataset"])
    best = best_values(rows, decimals)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{Chapter 4 results for {latex_escape(DATASET_LABELS[dataset])}. Answer EM and Answer F1 use the hash-pinned SAGE-QA ontology evaluator. All values are percentages. Bold values indicate the best setting within each model.}}",
        rf"\label{{tab:chapter4_{DATASET_SLUGS[dataset]}_sageqa}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llcccccccccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Binary Questions}} & \multicolumn{8}{c}{\textbf{Open-Ended Questions}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-14}",
        r"\textbf{Model} & \textbf{Setting} & \multicolumn{2}{c}{\textbf{A-EM}} & \multicolumn{2}{c}{\textbf{Align.}} & \multicolumn{2}{c}{\textbf{A-EM}} & \multicolumn{2}{c}{\textbf{A-F1}} & \multicolumn{2}{c}{\textbf{Align.}} & \multicolumn{2}{c}{\textbf{Hall.}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}",
        r"& & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop \\",
        r"\midrule",
    ]
    for model_index, model_id in enumerate(MODEL_LABELS):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        for row_index, row in enumerate(model_rows):
            cells = [
                rf"\multirow{{3}}{{*}}{{{latex_escape(row['model'])}}}"
                if row_index == 0
                else "",
                latex_escape(row["setting"]),
            ] + [
                latex_metric_cell(row, column, best, decimals)
                for column in METRIC_COLUMNS
            ]
            lines.append(" & ".join(cells) + r" \\")
        if model_index != len(MODEL_LABELS) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    rows: list[dict[str, Any]], manifest: dict[str, Any], output_dir: Path, decimals: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "chapter4_all_results.csv", decimals)
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        slug = DATASET_SLUGS[dataset]
        write_csv(dataset_rows, output_dir / f"{slug}_results.csv", decimals)
        write_latex_dataset(dataset_rows, output_dir / f"{slug}_table.tex", decimals)
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows, manifest = collect_rows(args.base_dir)
    write_outputs(rows, manifest, args.output_dir, args.decimals)
    print(f"Validated result cells: {manifest['validated_result_cells']}")
    print(
        "Binary Answer EM/F1 mismatches: "
        f"{manifest['binary_answer_em_f1_mismatch_count']}"
    )
    print(
        "Empty/error predictions preserved: "
        f"{manifest['empty_or_error_predictions_preserved']}"
    )
    print(f"Wrote Chapter 4 artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
