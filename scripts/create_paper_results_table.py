#!/usr/bin/env python3
"""Create a combined benchmark results table.

The script reads the final benchmark result CSVs plus their
metrics/*_key_findings_summary.json files and writes:

* a wide CSV with the same metric layout as the paper table
* a LaTeX tabular fragment using grouped model rows

By default it discovers all available runs under:
data/output/final_benchmark_llm_results/{dataset}_{1hop,2hop}/{model}/{nl,abs,sparql}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE_DIR = Path("data/output/final_benchmark_llm_results")
DEFAULT_OUTPUT_CSV = Path(
    "data/output/final_benchmark_llm_results/combined_1hop_2hop_results_table.csv"
)
DEFAULT_OUTPUT_TEX = Path(
    "data/output/final_benchmark_llm_results/combined_1hop_2hop_results_table.tex"
)
SETTINGS = {
    "nl": "NL",
    "sparql": "FS",
    "abs": "AR",
}
ANSWER_TYPE_LABELS = {
    "binary": "binary",
    "mc": "open",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined CSV and LaTeX table for available benchmark results."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--decimals", type=int, default=1)
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Only write the CSV output.",
    )
    return parser.parse_args()


def pct_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def clean_answer_items(text: Any) -> set[str]:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    if ";" in text:
        items = text.split(";")
    elif "," in text:
        items = text.split(",")
    else:
        items = [text]

    cleaned_items: set[str] = set()
    for item in items:
        item = item.strip()
        item = re.sub(r'[_*#@$%^&()+=\[\]{}|\\:";\'<>?/~`]', "", item)
        item = re.sub(r"\s+", " ", item).strip()
        item = re.sub(r"\s+\d{4}$", "", item)
        item = re.sub(r"\s+\d{4}\s+", " ", item)
        item = re.sub(r"^\d+$", "", item)
        item = re.sub(r"[^a-z0-9\s]", "", item)
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            cleaned_items.add(item)
    return cleaned_items


def jaccard_accuracy(expected: Any, actual: Any, answer_type: str) -> float:
    expected_set = clean_answer_items(expected)
    actual_set = clean_answer_items(actual)

    if not expected_set and not actual_set:
        score = 1.0
    elif not expected_set or not actual_set:
        score = 0.0
    else:
        intersection = len(expected_set.intersection(actual_set))
        union = len(expected_set.union(actual_set))
        score = intersection / union if union else 0.0

    return (
        1.0
        if answer_type == "BIN" and score == 1.0
        else (0.0 if answer_type == "BIN" else score)
    )


def confidence_calibration(confidence: Any, accuracy: float) -> float:
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.5
    confidence_value = max(0.0, min(1.0, confidence_value))
    return 1.0 - abs(confidence_value - accuracy)


def detect_models_from_csv(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    suffix = "_final_answer"
    return sorted(col[: -len(suffix)] for col in header if col.endswith(suffix))


def is_valid_answer(value: Any) -> bool:
    if value is None:
        return False
    text = str(value)
    return (
        bool(text.strip())
        and not text.startswith("[ERROR]")
        and not text.startswith("ERROR")
    )


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def confidence_by_type(csv_path: Path, model_name: str) -> dict[str, float | None]:
    final_answer_col = f"{model_name}_final_answer"
    confidence_col = f"{model_name}_confidence_score"
    values: dict[str, list[float]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            final_answer = row.get(final_answer_col)
            if not is_valid_answer(final_answer):
                continue

            answer_type = row.get("Answer Type", "BIN").strip().upper()
            bucket = "binary" if answer_type == "BIN" else "mc"
            accuracy = jaccard_accuracy(
                row.get("Answer", ""), final_answer, answer_type
            )
            values[bucket].append(
                confidence_calibration(row.get(confidence_col, 0.5), accuracy) * 100.0
            )

    return {bucket: mean(scores) for bucket, scores in values.items()}


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


def read_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open(encoding="utf-8") as handle:
        return json.load(handle).get("key_findings_summary", {})


def get_metric_paths(run_dir: Path, setting: str) -> tuple[Path, Path]:
    csv_path = run_dir / f"{setting}_final_benchmark_results_FINAL.csv"
    summary_path = run_dir / "metrics" / f"{setting}_key_findings_summary.json"
    return csv_path, summary_path


def parse_dataset_hop_dir(path: Path) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+)_(1hop|2hop)", path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def iter_available_runs(base_dir: Path):
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {base_dir}")

    dataset_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    for dataset_dir in dataset_dirs:
        parsed = parse_dataset_hop_dir(dataset_dir)
        if parsed is None:
            continue

        dataset, hop = parsed
        for model_dir in sorted(
            path for path in dataset_dir.iterdir() if path.is_dir()
        ):
            for setting in SETTINGS:
                run_dir = model_dir / setting
                if run_dir.exists():
                    yield dataset, hop, setting, run_dir


def format_cell(value: float | None, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def collect_rows(base_dir: Path) -> list[dict[str, str]]:
    table: dict[tuple[str, str, str], dict[str, float | None | str]] = {}
    skipped_runs: list[str] = []

    for dataset, hop, setting, run_dir in iter_available_runs(base_dir):
        hop_label = "1-hop" if hop == "1hop" else "2-hop"
        setting_label = SETTINGS[setting]
        csv_path, summary_path = get_metric_paths(run_dir, setting)
        if not csv_path.exists() or not summary_path.exists():
            skipped_runs.append(str(run_dir))
            continue

        summary = read_summary(summary_path)
        models = sorted(set(summary.keys()) | set(detect_models_from_csv(csv_path)))

        for model_name in models:
            key = (dataset, model_name, setting_label)
            table.setdefault(
                key,
                {
                    "dataset": dataset,
                    "model_id": model_name,
                    "model": display_model_name(model_name),
                    "setting": setting_label,
                },
            )
            model_summary = summary.get(model_name, {})
            by_type = model_summary.get("performance_by_answer_type", {})
            overall = model_summary.get("overall_metrics", {})
            conf = confidence_by_type(csv_path, model_name)

            table[key][f"binary_jaccard_accuracy_{hop_label}"] = pct_to_float(
                by_type.get("binary", {}).get("average_accuracy")
            )
            table[key][f"binary_confidence_{hop_label}"] = conf.get("binary")
            table[key][f"open_ended_jaccard_accuracy_{hop_label}"] = pct_to_float(
                by_type.get("mc", {}).get("average_accuracy")
            )
            table[key][f"open_ended_confidence_{hop_label}"] = conf.get("mc")
            table[key][f"open_ended_hallucination_{hop_label}"] = pct_to_float(
                overall.get("hallucination_score")
            )

    if not table:
        raise FileNotFoundError(
            f"No complete result runs found under {base_dir}. Expected "
            "{dataset}_{1hop,2hop}/{model}/{nl,abs,sparql}/"
            "{setting}_final_benchmark_results_FINAL.csv and metrics JSON files."
        )

    if skipped_runs:
        print(
            f"Skipped {len(skipped_runs)} incomplete run directories missing CSV or metrics JSON."
        )

    rows = list(table.values())
    setting_order = {label: index for index, label in enumerate(SETTINGS.values())}
    rows.sort(
        key=lambda row: (
            str(row["dataset"]).lower(),
            str(row["model"]).lower(),
            setting_order.get(str(row["setting"]), 99),
        )
    )
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path, decimals: int) -> None:
    columns = [
        "dataset",
        "model_id",
        "model",
        "setting",
        "binary_jaccard_accuracy_1-hop",
        "binary_jaccard_accuracy_2-hop",
        "binary_confidence_1-hop",
        "binary_confidence_2-hop",
        "open_ended_jaccard_accuracy_1-hop",
        "open_ended_jaccard_accuracy_2-hop",
        "open_ended_confidence_1-hop",
        "open_ended_confidence_2-hop",
        "open_ended_hallucination_1-hop",
        "open_ended_hallucination_2-hop",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row[column]
                    if column in ("dataset", "model_id", "model", "setting")
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
        r"\begin{tabular}{@{}lllcccccccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{\textbf{Binary Questions}} & \multicolumn{6}{c}{\textbf{Open-Ended Questions}} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-13}",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Setting}",
        r"& \multicolumn{2}{c}{\textbf{Jac. Acc.}} & \multicolumn{2}{c}{\textbf{Conf.}}",
        r"& \multicolumn{2}{c}{\textbf{Jac. Acc.}} & \multicolumn{2}{c}{\textbf{Conf.}} & \multicolumn{2}{c}{\textbf{Hall.}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"& & & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop & 1-hop & 2-hop \\",
        r"\midrule",
    ]

    group_keys = sorted(
        grouped.keys(), key=lambda item: (item[0].lower(), item[1].lower())
    )
    for group_index, (dataset, model) in enumerate(group_keys):
        model_rows = grouped[(dataset, model)]
        model_rows.sort(key=lambda row: list(SETTINGS.values()).index(row["setting"]))
        for row_index, row in enumerate(model_rows):
            dataset_cell = (
                rf"\multirow{{{len(model_rows)}}}{{*}}{{{latex_escape(dataset)}}}"
                if row_index == 0
                else ""
            )
            model_cell = (
                rf"\multirow{{{len(model_rows)}}}{{*}}{{{latex_escape(model)}}}"
                if row_index == 0
                else ""
            )
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        model_cell,
                        latex_escape(row["setting"]),
                        row_value(row, "binary_jaccard_accuracy_1-hop", decimals),
                        row_value(row, "binary_jaccard_accuracy_2-hop", decimals),
                        row_value(row, "binary_confidence_1-hop", decimals),
                        row_value(row, "binary_confidence_2-hop", decimals),
                        row_value(row, "open_ended_jaccard_accuracy_1-hop", decimals),
                        row_value(row, "open_ended_jaccard_accuracy_2-hop", decimals),
                        row_value(row, "open_ended_confidence_1-hop", decimals),
                        row_value(row, "open_ended_confidence_2-hop", decimals),
                        row_value(row, "open_ended_hallucination_1-hop", decimals),
                        row_value(row, "open_ended_hallucination_2-hop", decimals),
                    ]
                )
                + r" \\"
            )
        if group_index != len(group_keys) - 1:
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
