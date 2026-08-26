#!/usr/bin/env python3
"""Regenerate Chapter 4 ontology tables from saved predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.llm_pipeline.sageqa_answer_metrics import (
        BenchmarkMismatchError,
        benchmark_csv_path,
        detect_models,
        read_csv_rows,
        score_checkpoint_rows,
        validate_checkpoint_rows,
    )
except ImportError:
    from llm_pipeline.sageqa_answer_metrics import (
        BenchmarkMismatchError,
        benchmark_csv_path,
        detect_models,
        read_csv_rows,
        score_checkpoint_rows,
        validate_checkpoint_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("data/output/final_benchmark_llm_results")
DEFAULT_OUTPUT_CSV = DEFAULT_BASE_DIR / "combined_1hop_2hop_results_table.csv"
DEFAULT_OUTPUT_TEX = DEFAULT_BASE_DIR / "combined_1hop_2hop_results_table.tex"
SETTINGS = {"nl": "NL", "sparql": "FS", "abs": "AR"}
CHAPTER4_DATASETS = {"FamilyOWL", "OWL2Bench"}

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
        description="Create Chapter 4 Answer EM/F1 CSV and LaTeX tables."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--decimals", type=int, default=1)
    parser.add_argument("--no-tex", action="store_true")
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
    """Validate before keying, then reject duplicates instead of overwriting."""

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
        benchmark_rows = benchmark_cache.setdefault(cache_key, read_csv_rows(benchmark_path))
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
    replacements = {
        "openai_gpt_4_1_mini": "GPT-4.1-mini",
        "openai_gpt_5_mini": "GPT-5-mini",
        "deepseek_v3": "DeepSeek-V3",
        "llama_4_maverick": "LLaMA-4-Maverick",
    }
    if model_name in replacements:
        return replacements[model_name]
    name = model_name
    for prefix in ("openai_", "anthropic_", "google_", "meta_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", "-")


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


def collect_rows(base_dir: Path) -> list[dict[str, Any]]:
    sources, rejected = discover_validated_sources(base_dir)
    if not sources:
        raise FileNotFoundError(f"No benchmark-valid checkpoint sources under {base_dir}")

    table: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows_cache: dict[Path, list[dict[str, str]]] = {}
    for source in sources.values():
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
        scored = score_checkpoint_rows(checkpoint_rows, source.model_name)["aggregates"]
        for answer_bucket, column_prefix in (("binary", "binary"), ("open", "open_ended")):
            metrics = scored[answer_bucket]
            row[f"{column_prefix}_answer_em_{hop_label}"] = 100.0 * metrics["answer_em"]
            row[f"{column_prefix}_answer_f1_{hop_label}"] = 100.0 * metrics["answer_f1"]
            row[f"{column_prefix}_confidence_correctness_alignment_{hop_label}"] = (
                100.0 * metrics["confidence_correctness_alignment"]
            )
        row[f"open_ended_hallucination_{hop_label}"] = read_hallucination(source)

    for message in rejected:
        print(f"Rejected mismatched checkpoint: {message}")

    setting_order = {label: index for index, label in enumerate(SETTINGS.values())}
    rows = list(table.values())
    rows.sort(
        key=lambda row: (
            str(row["dataset"]).lower(),
            str(row["model"]).lower(),
            setting_order.get(str(row["setting"]), 99),
        )
    )
    return rows


IDENTITY_COLUMNS = ["dataset", "model_id", "model", "setting"]
METRIC_COLUMNS = [
    f"{answer_type}_{metric}_{hop}"
    for answer_type in ("binary", "open_ended")
    for metric in ("answer_em", "answer_f1", "confidence_correctness_alignment")
    for hop in ("1-hop", "2-hop")
] + [f"open_ended_hallucination_{hop}" for hop in ("1-hop", "2-hop")]


def format_cell(value: float | None, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def write_csv(rows: list[dict[str, Any]], output_csv: Path, decimals: int) -> None:
    columns = IDENTITY_COLUMNS + METRIC_COLUMNS
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    if column in IDENTITY_COLUMNS
                    else format_cell(row.get(column), decimals)
                    for column in columns
                }
            )


def latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
    )


def row_value(row: dict[str, Any], key: str, decimals: int) -> str:
    return format_cell(row.get(key), decimals) or "--"


def write_latex(rows: list[dict[str, Any]], output_tex: Path, decimals: int) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)

    lines = [
        r"\begin{tabular}{@{}lllcccccccccccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{6}{c}{\textbf{Binary Questions}} & \multicolumn{8}{c}{\textbf{Open-Ended Questions}} \\",
        r"\cmidrule(lr){4-9} \cmidrule(lr){10-17}",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Setting}",
        r"& \multicolumn{2}{c}{\textbf{Ans. EM}} & \multicolumn{2}{c}{\textbf{Ans. F1}} & \multicolumn{2}{c}{\textbf{Conf.--Corr.}}",
        r"& \multicolumn{2}{c}{\textbf{Ans. EM}} & \multicolumn{2}{c}{\textbf{Ans. F1}} & \multicolumn{2}{c}{\textbf{Conf.--Corr.}} & \multicolumn{2}{c}{\textbf{Hall.}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15} \cmidrule(lr){16-17}",
        r"& & & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop \\",
        r"\midrule",
    ]
    keys = sorted(grouped, key=lambda item: (item[0].lower(), item[1].lower()))
    for group_index, (dataset, model) in enumerate(keys):
        model_rows = sorted(
            grouped[(dataset, model)],
            key=lambda row: list(SETTINGS.values()).index(row["setting"]),
        )
        for row_index, row in enumerate(model_rows):
            cells = [
                rf"\multirow{{{len(model_rows)}}}{{*}}{{{latex_escape(dataset)}}}"
                if row_index == 0
                else "",
                rf"\multirow{{{len(model_rows)}}}{{*}}{{{latex_escape(model)}}}"
                if row_index == 0
                else "",
                latex_escape(row["setting"]),
            ] + [row_value(row, column, decimals) for column in METRIC_COLUMNS]
            lines.append(" & ".join(cells) + r" \\")
        if group_index != len(keys) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.base_dir)
    write_csv(rows, args.output_csv, args.decimals)
    print(f"Wrote CSV: {args.output_csv}")
    if not args.no_tex:
        write_latex(rows, args.output_tex, args.decimals)
        print(f"Wrote LaTeX: {args.output_tex}")


if __name__ == "__main__":
    main()
