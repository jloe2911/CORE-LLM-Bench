#!/usr/bin/env python3
"""Summarize performance by Pellet-derived explanation complexity.

Complexity is the ``Max Tag Length`` used during stratified benchmark
sampling. It is joined to existing model predictions by Task ID and SPARQL
query. Saved answers are scored with the same lexical normalization used by
the benchmark tables. The analysis reports low/medium/high bins and Spearman
correlations against the original numeric complexity value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import spearmanr

try:
    from scripts.llm_pipeline.answer_normalization import (
        ANSWER_NORMALIZATION_VERSION,
        normalized_jaccard_accuracy,
    )
except ImportError:
    # Support direct execution:
    # python scripts/create_explanation_complexity_analysis.py
    from llm_pipeline.answer_normalization import (
        ANSWER_NORMALIZATION_VERSION,
        normalized_jaccard_accuracy,
    )


DEFAULT_RESULTS_DIR = Path("data/output/final_benchmark_llm_results")
DEFAULT_DATA_DIR = Path("data/output")
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "explanation_complexity_results.csv"
DEFAULT_CORRELATION_CSV = (
    DEFAULT_RESULTS_DIR / "explanation_complexity_correlations.csv"
)
DEFAULT_OUTPUT_TEX = Path("paper_material/explanation_complexity_table.tex")
DEFAULT_OUTPUT_TEXT = Path("paper_material/performance_by_explanation_complexity.tex")
DEFAULT_DATASETS = ("FamilyOWL", "OWL2Bench", "pizza_100", "pizza_250")
DATASET_LABELS = {
    "FamilyOWL": "FamilyOWL",
    "OWL2Bench": "OWL2Bench",
    "pizza_100": "Pizza 100",
    "pizza_250": "Pizza 250",
}

MODEL_LABELS = {
    "openai_gpt_5_mini_2025_08_07": "GPT-5-mini",
    "openrouter_google_gemini_2_5_flash_lite": "Gemini-2.5-Flash-Lite",
    "openrouter_qwen_qwen3_30b_a3b_instruct_2507": "Qwen3-30B-A3B-Instruct",
}
MODEL_ORDER = {
    "GPT-5-mini": 0,
    "Gemini-2.5-Flash-Lite": 1,
    "Qwen3-30B-A3B-Instruct": 2,
}
SETTING_LABELS = {"nl": "NL", "sparql": "FS", "abs": "AR"}
SETTING_ORDER = {"NL": 0, "FS": 1, "AR": 2}
BIN_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--correlation-csv", type=Path, default=DEFAULT_CORRELATION_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--output-text", type=Path, default=DEFAULT_OUTPUT_TEXT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=(
            "Datasets to include. Defaults to the four substantive benchmark "
            "datasets and excludes toy_example."
        ),
    )
    return parser.parse_args()


def set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_query(value: Any) -> str:
    return " ".join(str(value or "").split())


def jaccard_accuracy(expected: Any, actual: Any, answer_type: str) -> float:
    return normalized_jaccard_accuracy(expected, actual, answer_type)


def valid_answer(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and not text.upper().startswith(("[ERROR]", "ERROR"))


def complexity_bin(value: int) -> str:
    if value == 1:
        return "Low"
    if value in (2, 3):
        return "Medium"
    return "High"


def discover_result_files(
    results_dir: Path,
    included_datasets: set[str],
) -> Iterable[tuple[str, str, str, Path, tuple[str, ...]]]:
    candidates: dict[tuple[str, str, str, str], Path] = {}

    for dataset_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        match = re.fullmatch(r"(.+)_(1hop|2hop)", dataset_dir.name)
        if not match:
            continue
        dataset, hop = match.groups()
        if dataset not in included_datasets:
            continue

        for setting_dir in sorted(
            path
            for path in dataset_dir.glob("*/*")
            if path.is_dir() and path.name in SETTING_LABELS
        ):
            setting = setting_dir.name
            final_path = setting_dir / f"{setting}_final_benchmark_results_FINAL.csv"
            checkpoint_path = setting_dir / "LATEST_checkpoint.csv"
            available = [
                path for path in (final_path, checkpoint_path) if path.exists()
            ]
            if not available:
                continue

            # A few interrupted runs contain a tiny file named FINAL followed
            # by a newer, complete checkpoint. Prefer FINAL only when it is
            # plausibly as complete as the checkpoint.
            if final_path.exists() and (
                not checkpoint_path.exists()
                or final_path.stat().st_size >= 0.5 * checkpoint_path.stat().st_size
            ):
                result_path = final_path
            else:
                result_path = max(
                    available,
                    key=lambda path: (path.stat().st_mtime, path.stat().st_size),
                )

            with result_path.open(newline="", encoding="utf-8-sig") as handle:
                header = next(csv.reader(handle))
            for model in model_ids(header):
                key = (dataset, hop, setting, model)
                previous = candidates.get(key)
                if previous is None or (
                    result_path.stat().st_mtime,
                    result_path.stat().st_size,
                ) > (
                    previous.stat().st_mtime,
                    previous.stat().st_size,
                ):
                    candidates[key] = result_path

    grouped: dict[tuple[str, str, str, Path], list[str]] = defaultdict(list)
    for (dataset, hop, setting, model), path in candidates.items():
        grouped[(dataset, hop, setting, path)].append(model)

    for (dataset, hop, setting, path), models in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0].lower(),
            item[0][1],
            item[0][2],
            str(item[0][3]).lower(),
        ),
    ):
        yield dataset, hop, setting, path, tuple(sorted(models))


def model_ids(fieldnames: list[str]) -> list[str]:
    suffix = "_final_answer"
    return sorted(name[: -len(suffix)] for name in fieldnames if name.endswith(suffix))


def read_complexity_lookup(
    path: Path,
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    exact: dict[tuple[str, str], int] = {}
    by_task_values: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get("Max Tag Length", "")).strip()
            if not raw:
                continue
            value = int(float(raw))
            task_id = str(row.get("Task ID", ""))
            query = normalize_query(row.get("SPARQL Query"))
            exact[(task_id, query)] = value
            by_task_values[task_id].add(value)
    unique = {
        task_id: next(iter(values))
        for task_id, values in by_task_values.items()
        if len(values) == 1
    }
    return exact, unique


def collect_observations(
    results_dir: Path,
    data_dir: Path,
    included_datasets: Iterable[str] = DEFAULT_DATASETS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    observations: list[dict[str, Any]] = []
    exclusions: dict[str, int] = defaultdict(int)
    lookup_cache: dict[
        tuple[str, str], tuple[dict[tuple[str, str], int], dict[str, int]]
    ] = {}

    for dataset, hop, setting, result_path, selected_models in discover_result_files(
        results_dir,
        set(included_datasets),
    ):
        source_path = data_dir / dataset / hop / "SPARQL_questions_sampling.csv"
        if not source_path.exists():
            exclusions["runs_without_sampling_csv"] += 1
            continue
        cache_key = (dataset, hop)
        if cache_key not in lookup_cache:
            lookup_cache[cache_key] = read_complexity_lookup(source_path)
        exact, unique = lookup_cache[cache_key]

        with result_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            available_models = set(model_ids(reader.fieldnames or []))
            models = [model for model in selected_models if model in available_models]
            for row in reader:
                task_id = str(row.get("Task ID", ""))
                query = normalize_query(row.get("SPARQL Query"))
                complexity = exact.get((task_id, query), unique.get(task_id))
                if complexity is None:
                    exclusions["predictions_without_complexity_match"] += len(models)
                    continue
                answer_type = str(row.get("Answer Type", "BIN")).upper()
                for model in models:
                    actual = row.get(f"{model}_final_answer")
                    if not valid_answer(actual):
                        exclusions["invalid_or_missing_model_answers"] += 1
                        continue
                    observations.append(
                        {
                            "dataset": dataset,
                            "hop": hop,
                            "setting": SETTING_LABELS[setting],
                            "model": MODEL_LABELS.get(model, model),
                            "task_id": task_id,
                            "task_type": str(row.get("Task Type", "Unknown")),
                            "answer_type": answer_type,
                            "max_tag_length": complexity,
                            "complexity_bin": complexity_bin(complexity),
                            "accuracy": jaccard_accuracy(
                                row.get("Answer", ""), actual, answer_type
                            ),
                        }
                    )
    return observations, dict(exclusions)


def aggregate_bins(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for item in observations:
        key = (
            item["model"],
            item["setting"],
            item["answer_type"],
            item["complexity_bin"],
        )
        buckets[key].append(item["accuracy"])

    rows = []
    for (model, setting, answer_type, bin_name), scores in buckets.items():
        rows.append(
            {
                "model": model,
                "setting": setting,
                "answer_type": answer_type,
                "complexity_bin": bin_name,
                "n": len(scores),
                "mean_score": sum(scores) / len(scores),
            }
        )
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row["model"]), 99),
            SETTING_ORDER.get(str(row["setting"]), 99),
            str(row["answer_type"]),
            BIN_ORDER[str(row["complexity_bin"])],
        )
    )
    return rows


def aggregate_datasets(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for item in observations:
        buckets[
            item["dataset"],
            item["answer_type"],
            item["complexity_bin"],
        ].append(item["accuracy"])

    rows = []
    for dataset in sorted(
        {item["dataset"] for item in observations},
        key=lambda value: (
            DEFAULT_DATASETS.index(value) if value in DEFAULT_DATASETS else 99
        ),
    ):
        row: dict[str, Any] = {"dataset": dataset}
        for answer_type in ("BIN", "MC"):
            for bin_name in ("Low", "High"):
                values = buckets[(dataset, answer_type, bin_name)]
                row[f"{answer_type}_{bin_name}"] = sum(values) / len(values)
            row[f"{answer_type}_delta"] = (
                row[f"{answer_type}_High"] - row[f"{answer_type}_Low"]
            )
        rows.append(row)
    return rows


def correlations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for item in observations:
        buckets[(item["model"], item["setting"], item["answer_type"])].append(
            (item["max_tag_length"], item["accuracy"])
        )

    rows = []
    for (model, setting, answer_type), values in buckets.items():
        complexities = [value[0] for value in values]
        scores = [value[1] for value in values]
        if len(set(complexities)) < 2 or len(set(scores)) < 2:
            rho, p_value = math.nan, math.nan
        else:
            result = spearmanr(complexities, scores)
            rho, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "model": model,
                "setting": setting,
                "answer_type": answer_type,
                "n": len(values),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row["model"]), 99),
            SETTING_ORDER.get(str(row["setting"]), 99),
            str(row["answer_type"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def score_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "--"
    return f"{100 * float(row['mean_score']):.1f} ({row['n']})"


def p_value_cell(value: float) -> str:
    if math.isnan(value):
        return "--"
    if value < 0.001:
        return "$<.001$"
    return f"{value:.3f}".lstrip("0")


def rho_cell(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:+.3f}"


def write_latex_tables(
    path: Path,
    bin_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    dataset_names: list[str],
) -> None:
    bin_lookup = {
        (row["model"], row["setting"], row["answer_type"], row["complexity_bin"]): row
        for row in bin_rows
    }
    corr_lookup = {
        (row["model"], row["setting"], row["answer_type"]): row
        for row in correlation_rows
    }
    pairs = sorted(
        {(row["model"], row["setting"]) for row in bin_rows},
        key=lambda value: (
            MODEL_ORDER.get(value[0], 99),
            SETTING_ORDER.get(value[1], 99),
        ),
    )
    dataset_text = ", ".join(DATASET_LABELS.get(name, name) for name in dataset_names)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Performance by explanation complexity and dataset, pooled "
        r"over models, settings, and extraction radii. Values are normalized "
        r"mean scores in percent. A negative $\Delta$ means lower performance "
        r"on high-complexity questions.}",
        r"\label{4:tab_complexity_dataset}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{BQA} & \multicolumn{3}{c}{OEQA} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Dataset & Low & High & $\Delta$ & Low & High & $\Delta$ \\",
        r"\midrule",
    ]
    for row in dataset_rows:
        lines.append(
            f"{latex_escape(DATASET_LABELS.get(row['dataset'], row['dataset']))} & "
            f"{100 * row['BIN_Low']:.1f} & {100 * row['BIN_High']:.1f} & "
            f"{100 * row['BIN_delta']:+.1f} & "
            f"{100 * row['MC_Low']:.1f} & {100 * row['MC_High']:.1f} & "
            f"{100 * row['MC_delta']:+.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    for answer_type, label, metric in (
        ("BIN", "BQA", "accuracy"),
        ("MC", "OEQA", "Jaccard score"),
    ):
        lines.extend(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                rf"\caption{{{label} performance by Pellet-derived "
                rf"explanation complexity. Values are mean {metric} "
                r"in percent after lexical answer normalization, pooled over "
                rf"{dataset_text} at both extraction radii within "
                r"each row; response counts are in parentheses.}",
                rf"\label{{4:tab_complexity_{label.lower()}}}",
                r"\begin{tabular}{llrrrr}",
                r"\toprule",
                r"Model & Setting & Low (1) & Medium (2--3) & High ($\geq4$) & "
                r"$\Delta_{\mathrm{High-Low}}$ \\",
                r"\midrule",
            ]
        )
        previous_model = None
        for model, setting in pairs:
            low = bin_lookup.get((model, setting, answer_type, "Low"))
            medium = bin_lookup.get((model, setting, answer_type, "Medium"))
            high = bin_lookup.get((model, setting, answer_type, "High"))
            if previous_model is not None and previous_model != model:
                lines.append(r"\midrule")
            delta = (
                f"{100 * (float(high['mean_score']) - float(low['mean_score'])):+.1f}"
                if low and high
                else "--"
            )
            lines.append(
                f"{latex_escape(model)} & {setting} & {score_cell(low)} & "
                f"{score_cell(medium)} & {score_cell(high)} & {delta} \\\\"
            )
            previous_model = model
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )

    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Spearman correlation between explanation complexity "
            r"and response score. Negative $\rho$ means that performance "
            r"decreases as complexity increases.}",
            r"\label{4:tab_complexity_spearman}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Model & Setting & $\rho_{\mathrm{BQA}}$ & $p_{\mathrm{BQA}}$ "
            r"& $\rho_{\mathrm{OEQA}}$ & $p_{\mathrm{OEQA}}$ \\",
            r"\midrule",
        ]
    )
    previous_model = None
    for model, setting in pairs:
        bqa = corr_lookup.get((model, setting, "BIN"))
        oeqa = corr_lookup.get((model, setting, "MC"))
        if previous_model is not None and previous_model != model:
            lines.append(r"\midrule")
        bqa_rho = rho_cell(float(bqa["spearman_rho"])) if bqa else "--"
        oeqa_rho = rho_cell(float(oeqa["spearman_rho"])) if oeqa else "--"
        lines.append(
            f"{latex_escape(model)} & {setting} & {bqa_rho} & "
            f"{p_value_cell(float(bqa['p_value'])) if bqa else '--'} & "
            f"{oeqa_rho} & "
            f"{p_value_cell(float(oeqa['p_value'])) if oeqa else '--'} \\\\"
        )
        previous_model = model
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def pooled_bin_summary(
    observations: list[dict[str, Any]], answer_type: str, bin_name: str
) -> tuple[float, int]:
    values = [
        item["accuracy"]
        for item in observations
        if item["answer_type"] == answer_type and item["complexity_bin"] == bin_name
    ]
    return sum(values) / len(values), len(values)


def pooled_correlation(
    observations: list[dict[str, Any]], answer_type: str
) -> tuple[float, float, int]:
    selected = [item for item in observations if item["answer_type"] == answer_type]
    result = spearmanr(
        [item["max_tag_length"] for item in selected],
        [item["accuracy"] for item in selected],
    )
    return float(result.statistic), float(result.pvalue), len(selected)


def inline_significance(value: float) -> str:
    if value < 0.001:
        return r"$p<.001$"
    return rf"$p={value:.3f}$"


def write_subsection(path: Path, observations: list[dict[str, Any]]) -> None:
    bqa_low, bqa_low_n = pooled_bin_summary(observations, "BIN", "Low")
    bqa_high, bqa_high_n = pooled_bin_summary(observations, "BIN", "High")
    oeqa_low, oeqa_low_n = pooled_bin_summary(observations, "MC", "Low")
    oeqa_high, oeqa_high_n = pooled_bin_summary(observations, "MC", "High")
    bqa_rho, bqa_p, bqa_n = pooled_correlation(observations, "BIN")
    oeqa_rho, oeqa_p, oeqa_n = pooled_correlation(observations, "MC")
    bin_rows = aggregate_bins(observations)
    bin_lookup = {
        (row["model"], row["setting"], row["answer_type"], row["complexity_bin"]): row
        for row in bin_rows
    }
    pairs = sorted(
        {(row["model"], row["setting"]) for row in bin_rows},
        key=lambda value: (
            MODEL_ORDER.get(value[0], 99),
            SETTING_ORDER.get(value[1], 99),
        ),
    )

    def high_low_deltas(answer_type: str) -> list[float]:
        values = []
        for model, setting in pairs:
            low = bin_lookup[(model, setting, answer_type, "Low")]
            high = bin_lookup[(model, setting, answer_type, "High")]
            values.append(100 * (float(high["mean_score"]) - float(low["mean_score"])))
        return values

    bqa_deltas = high_low_deltas("BIN")
    oeqa_deltas = high_low_deltas("MC")
    condition_count = len(pairs)
    dataset_rows = aggregate_datasets(observations)
    bqa_dataset_deltas = [100 * row["BIN_delta"] for row in dataset_rows]
    oeqa_dataset_deltas = [100 * row["MC_delta"] for row in dataset_rows]
    unique_oeqa_questions = {
        (
            item["dataset"],
            item["hop"],
            item["task_id"],
            item["complexity_bin"],
            item["task_type"],
        )
        for item in observations
        if item["answer_type"] == "MC"
    }
    oeqa_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _, _, _, bin_name, task_type in unique_oeqa_questions:
        oeqa_task_counts[bin_name][task_type] += 1
    high_oeqa_total = sum(oeqa_task_counts["High"].values())
    high_membership = oeqa_task_counts["High"]["Membership"]
    high_property = oeqa_task_counts["High"]["Property Assertion"]
    low_oeqa_total = sum(oeqa_task_counts["Low"].values())
    low_property = oeqa_task_counts["Low"]["Property Assertion"]
    low_composition = (
        rf"every low-complexity OEQA question ($n={low_oeqa_total}$) is a "
        r"property assertion"
        if low_property == low_oeqa_total
        else (
            f"{low_property} of the {low_oeqa_total} low-complexity OEQA "
            "questions are property assertions"
        )
    )
    dataset_labels = [
        DATASET_LABELS.get(value, value)
        for value in sorted({item["dataset"] for item in observations})
    ]
    datasets = (
        dataset_labels[0]
        if len(dataset_labels) == 1
        else (
            " and ".join(dataset_labels)
            if len(dataset_labels) == 2
            else ", ".join(dataset_labels[:-1]) + f", and {dataset_labels[-1]}"
        )
    )
    radii = " and ".join(
        value.replace("hop", "-hop")
        for value in sorted({item["hop"] for item in observations})
    )
    text = rf"""\subsection{{Performance by Explanation Complexity}}
\label{{sec:performance-explanation-complexity}}

We use the length of the Pellet proof tags as a simple question-difficulty
indicator: low complexity is a tag length of 1, medium is 2--3, and high is 4
or more.  This is a property of the benchmark proof, not the number of steps
written by the model.  Answers are scored with the same lexical normalization
used in the main results, so differences such as \texttt{{FishTopping}} versus
\texttt{{Fish Topping}} do not create artificial errors.

The analysis covers {datasets} at the {radii}
extraction radii.
Tables~\ref{{4:tab_complexity_bqa}} and
\ref{{4:tab_complexity_oeqa}} show the model--setting details.
Table~\ref{{4:tab_complexity_dataset}} provides the simpler dataset-level
overview.  All tables report the mean score in each complexity group.
The final column is the high-complexity score minus the low-complexity score;
a negative value therefore means that performance worsened on more complex
proofs.

\paragraph{{Observed pattern.}}
Scores decrease as explanation complexity increases.  Pooled across
models and settings, BQA accuracy falls from {100 * bqa_low:.1f}\%
($n={bqa_low_n}$) for low-complexity questions to {100 * bqa_high:.1f}\%
($n={bqa_high_n}$) for high-complexity questions, a decrease of
{100 * (bqa_low - bqa_high):.1f} percentage points.  OEQA Jaccard falls more
sharply, from {100 * oeqa_low:.1f}\% ($n={oeqa_low_n}$) to
{100 * oeqa_high:.1f}\% ($n={oeqa_high_n}$), a decrease of
{100 * (oeqa_low - oeqa_high):.1f} points.  All {condition_count}
model--setting combinations show a negative high--low difference.  The BQA
differences range from {min(bqa_deltas):.1f} to {max(bqa_deltas):.1f} points;
the OEQA differences range from {min(oeqa_deltas):.1f} to
{max(oeqa_deltas):.1f} points.

The decline also appears separately in all four datasets.  At the dataset
level, the BQA high--low differences range from
{min(bqa_dataset_deltas):.1f} to {max(bqa_dataset_deltas):.1f} points, and the
OEQA differences range from {min(oeqa_dataset_deltas):.1f} to
{max(oeqa_dataset_deltas):.1f} points.

Table~\ref{{4:tab_complexity_spearman}} confirms the same pattern without
grouping complexity into bins.  The pooled correlations are
$\rho={bqa_rho:+.3f}$ for BQA ($n={bqa_n}$,
{inline_significance(bqa_p)}), indicating a weak negative association, and
$\rho={oeqa_rho:+.3f}$ for OEQA ($n={oeqa_n}$,
{inline_significance(oeqa_p)}), indicating a stronger negative association.

\paragraph{{Interpretation.}}
The BQA results provide broad evidence that longer benchmark proofs are
associated with lower accuracy.  OEQA requires more caution because
complexity and task type are strongly confounded.  Of the
{high_oeqa_total} distinct high-complexity OEQA questions,
{high_membership} are membership (\texttt{{rdf:type}}) questions and
{high_property} are property assertions.  By contrast, {low_composition}.
For membership questions, the reference contains the full set of inferred
types, while a model often returns only the immediate type.  The near-zero
high-complexity OEQA scores therefore reflect both longer proofs and this
stricter answer requirement; they do not isolate a pure complexity effect.

Overall, the analysis shows a consistent negative pattern, particularly for
OEQA, but it does not show that proof length alone causes the decline.  The
same questions are also evaluated under several models and settings, so we
emphasize effect sizes and their consistency rather than treating the pooled
$p$-values as independent confirmatory tests.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_metadata(
    path: Path,
    observations: list[dict[str, Any]],
    exclusions: dict[str, int],
) -> None:
    datasets = sorted({item["dataset"] for item in observations})
    hops = sorted({item["hop"] for item in observations})
    unique_questions = {
        (
            item["dataset"],
            item["hop"],
            item["task_id"],
            item["answer_type"],
            item["complexity_bin"],
            item["task_type"],
        )
        for item in observations
    }
    oeqa_task_composition: dict[str, Counter[str]] = defaultdict(Counter)
    for _, _, _, answer_type, bin_name, task_type in unique_questions:
        if answer_type == "MC":
            oeqa_task_composition[bin_name][task_type] += 1
    metadata = {
        "complexity_variable": "Max Tag Length",
        "answer_normalization": ANSWER_NORMALIZATION_VERSION,
        "bins": {"Low": "1", "Medium": "2-3", "High": "4+"},
        "included_observations": len(observations),
        "datasets": datasets,
        "hops": hops,
        "oeqa_distinct_question_composition": {
            bin_name: dict(sorted(counts.items()))
            for bin_name, counts in sorted(
                oeqa_task_composition.items(),
                key=lambda item: BIN_ORDER[item[0]],
            )
        },
        "exclusions": exclusions,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_csv_field_limit()
    observations, exclusions = collect_observations(
        args.results_dir,
        args.data_dir,
        args.datasets,
    )
    if not observations:
        raise FileNotFoundError(
            "No predictions could be joined to sampling-time Max Tag Length."
        )
    bin_rows = aggregate_bins(observations)
    dataset_rows = aggregate_datasets(observations)
    correlation_rows = correlations(observations)
    write_csv(args.output_csv, bin_rows)
    write_csv(args.correlation_csv, correlation_rows)
    write_latex_tables(
        args.output_tex,
        bin_rows,
        correlation_rows,
        dataset_rows,
        sorted({item["dataset"] for item in observations}),
    )
    write_subsection(args.output_text, observations)
    write_metadata(
        args.output_text.with_name("explanation_complexity_metadata.json"),
        observations,
        exclusions,
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.correlation_csv}")
    print(f"Wrote {args.output_tex}")
    print(f"Wrote {args.output_text}")
    print(f"Observations included: {len(observations)}")
    for name, count in sorted(exclusions.items()):
        print(f"Excluded {name}: {count}")


if __name__ == "__main__":
    main()
