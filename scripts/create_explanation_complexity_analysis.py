#!/usr/bin/env python3
"""Recompute Chapter 4 explanation-complexity deltas with SAGE-QA Answer F1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.create_paper_results_table import (
        DATASETS,
        DATASET_LABELS,
        DEFAULT_BASE_DIR,
        DEFAULT_OUTPUT_DIR,
        PROJECT_ROOT,
        SETTINGS,
        discover_validated_sources,
    )
    from scripts.llm_pipeline.sageqa_answer_metrics import (
        SAGEQA_EVALUATOR_SHA256,
        benchmark_csv_path,
        read_csv_rows,
        sageqa_evaluator_path,
        score_checkpoint_rows,
    )
except ImportError:
    from create_paper_results_table import (
        DATASETS,
        DATASET_LABELS,
        DEFAULT_BASE_DIR,
        DEFAULT_OUTPUT_DIR,
        PROJECT_ROOT,
        SETTINGS,
        discover_validated_sources,
    )
    from llm_pipeline.sageqa_answer_metrics import (
        SAGEQA_EVALUATOR_SHA256,
        benchmark_csv_path,
        read_csv_rows,
        sageqa_evaluator_path,
        score_checkpoint_rows,
    )


DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "explanation_complexity_answer_f1.csv"
DEFAULT_OUTPUT_TEX = DEFAULT_OUTPUT_DIR / "explanation_complexity_answer_f1.tex"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "explanation_complexity_pooled.json"

max_csv_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_csv_field_size)
        break
    except OverflowError:
        max_csv_field_size //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def complexity_bin(value: int) -> str:
    if value == 1:
        return "Low"
    if value in (2, 3):
        return "Medium"
    return "High"


def read_complexity_by_task(path: Path) -> dict[str, int]:
    rows = read_csv_rows(path)
    values: dict[str, int] = {}
    for row in rows:
        task_id = str(row.get("Task ID", ""))
        raw = str(row.get("Max Tag Length", "")).strip()
        if not task_id or not raw:
            raise ValueError(f"Missing Task ID/Max Tag Length in {path}")
        value = int(float(raw))
        if task_id in values and values[task_id] != value:
            raise ValueError(f"Conflicting complexity for {task_id} in {path}")
        values[task_id] = value
    return values


def collect_observations(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources, rejected = discover_validated_sources(results_dir)
    if len(sources) != 72:
        raise ValueError(f"Expected 72 validated Chapter 4 cells, found {len(sources)}")

    checkpoint_cache: dict[Path, list[dict[str, str]]] = {}
    score_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    complexity_cache: dict[tuple[str, str, str], dict[str, int]] = {}
    observations: list[dict[str, Any]] = []
    empty_or_error = 0

    for key in sorted(sources):
        source = sources[key]
        checkpoint_rows = checkpoint_cache.setdefault(
            source.checkpoint_path, read_csv_rows(source.checkpoint_path)
        )
        scored = score_cache.setdefault(
            (source.checkpoint_path, source.model_name),
            score_checkpoint_rows(checkpoint_rows, source.model_name),
        )
        complexity_key = (source.dataset, source.hop, source.setting)
        complexity_path = benchmark_csv_path(
            PROJECT_ROOT, source.dataset, source.hop, source.setting
        )
        complexity_by_task = complexity_cache.setdefault(
            complexity_key, read_complexity_by_task(complexity_path)
        )
        answer_column = f"{source.model_name}_final_answer"
        row_by_task = {str(row["Task ID"]): row for row in checkpoint_rows}

        for item in scored["per_question"]:
            task_id = item["task_id"]
            if task_id not in complexity_by_task:
                raise ValueError(
                    f"Validated prediction lacks complexity join: {key}::{task_id}"
                )
            raw_prediction = str(row_by_task[task_id].get(answer_column, "") or "")
            if not raw_prediction.strip() or raw_prediction.strip().upper().startswith(
                ("[ERROR]", "ERROR")
            ):
                empty_or_error += 1
            complexity = complexity_by_task[task_id]
            observations.append(
                {
                    "dataset": source.dataset,
                    "hop": source.hop,
                    "setting": SETTINGS[source.setting],
                    "model_id": source.model_name,
                    "task_id": task_id,
                    "answer_group": "BQA"
                    if item["answer_type"] == "BIN"
                    else "OEQA",
                    "max_tag_length": complexity,
                    "complexity_bin": complexity_bin(complexity),
                    "answer_f1": float(item["answer_f1"]),
                }
            )

    manifest = {
        "scorer": {
            "implementation": "SAGE-QA answer_set_scores()/evaluate() per-question Answer F1",
            "path": str(sageqa_evaluator_path().resolve()),
            "sha256": SAGEQA_EVALUATOR_SHA256,
        },
        "complexity_variable": "Max Tag Length",
        "bins": {"Low": "1", "Medium": "2-3", "High": "4+"},
        "pooling": "Unweighted mean of already-computed per-question Answer F1 values across models, settings, and hops",
        "validated_result_cells": len(sources),
        "included_observations": len(observations),
        "empty_or_error_predictions_included": empty_or_error,
        "excluded_observations": 0,
        "rejected_checkpoint_count": len(rejected),
    }
    return observations, manifest


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty complexity bucket")
    return sum(values) / len(values)


def select_scores(
    observations: list[dict[str, Any]],
    answer_group: str,
    bin_name: str,
    dataset: str | None = None,
) -> list[float]:
    return [
        float(item["answer_f1"])
        for item in observations
        if item["answer_group"] == answer_group
        and item["complexity_bin"] == bin_name
        and (dataset is None or item["dataset"] == dataset)
    ]


def aggregate_datasets(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        bqa_low = select_scores(observations, "BQA", "Low", dataset)
        bqa_high = select_scores(observations, "BQA", "High", dataset)
        oeqa_low = select_scores(observations, "OEQA", "Low", dataset)
        oeqa_high = select_scores(observations, "OEQA", "High", dataset)
        row = {
            "Dataset": DATASET_LABELS[dataset],
            "dataset_id": dataset,
            "BQA Low": 100.0 * mean(bqa_low),
            "BQA High": 100.0 * mean(bqa_high),
            "BQA Delta": 100.0 * (mean(bqa_high) - mean(bqa_low)),
            "OEQA Low": 100.0 * mean(oeqa_low),
            "OEQA High": 100.0 * mean(oeqa_high),
            "OEQA Delta": 100.0 * (mean(oeqa_high) - mean(oeqa_low)),
            "counts": {
                "BQA Low": len(bqa_low),
                "BQA High": len(bqa_high),
                "OEQA Low": len(oeqa_low),
                "OEQA High": len(oeqa_high),
            },
        }
        rows.append(row)
    return rows


def pooled_summary(
    observations: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for answer_group in ("BQA", "OEQA"):
        low = select_scores(observations, answer_group, "Low")
        high = select_scores(observations, answer_group, "High")
        low_mean = 100.0 * mean(low)
        high_mean = 100.0 * mean(high)
        summary[answer_group] = {
            "low_complexity_mean_f1": low_mean,
            "high_complexity_mean_f1": high_mean,
            "difference_high_minus_low": high_mean - low_mean,
            "decrease_low_minus_high": low_mean - high_mean,
            "low_n": len(low),
            "high_n": len(high),
        }

        delta_key = f"{answer_group} Delta"
        decreases = [
            {
                "dataset": row["Dataset"],
                "decrease": -float(row[delta_key]),
                "delta_high_minus_low": float(row[delta_key]),
            }
            for row in dataset_rows
        ]
        summary[answer_group]["smallest_dataset_level_decrease"] = min(
            decreases, key=lambda item: item["decrease"]
        )
        summary[answer_group]["largest_dataset_level_decrease"] = max(
            decreases, key=lambda item: item["decrease"]
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "Dataset",
        "BQA Low",
        "BQA High",
        "BQA Delta",
        "OEQA Low",
        "OEQA High",
        "OEQA Delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row[column]
                    if column == "Dataset"
                    else f"{float(row[column]):.1f}"
                    for column in columns
                }
            )


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mean per-question Answer F1 by Pellet explanation complexity, pooled over all three models, NL/FS/AR settings, and 1-hop/2-hop conditions. Low complexity is a maximum tag length of 1; high complexity is 4 or more. $\Delta=\mathrm{High}-\mathrm{Low}$. Values are percentages.}",
        r"\label{tab:chapter4_explanation_complexity_sageqa}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{BQA}} & \multicolumn{3}{c}{\textbf{OEQA}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"\textbf{Dataset} & \textbf{Low} & \textbf{High} & \boldmath$\Delta$ & \textbf{Low} & \textbf{High} & \boldmath$\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Dataset']} & {row['BQA Low']:.1f} & {row['BQA High']:.1f} & "
            f"{row['BQA Delta']:+.1f} & {row['OEQA Low']:.1f} & "
            f"{row['OEQA High']:.1f} & {row['OEQA Delta']:+.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    observations, manifest = collect_observations(args.results_dir)
    dataset_rows = aggregate_datasets(observations)
    summary = pooled_summary(observations, dataset_rows)
    write_csv(args.output_csv, dataset_rows)
    write_latex(args.output_tex, dataset_rows)
    payload = {
        **manifest,
        "dataset_bucket_counts": {
            row["Dataset"]: row["counts"] for row in dataset_rows
        },
        "pooled_across_datasets": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Included observations: {manifest['included_observations']}")
    print(
        "Empty/error predictions included: "
        f"{manifest['empty_or_error_predictions_included']}"
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_tex}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
