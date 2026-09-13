#!/usr/bin/env python3
"""Create a read-only old-Jaccard versus new-SAGE-QA Chapter 4 audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.create_explanation_complexity_analysis import (
        complexity_bin,
        read_complexity_by_task,
    )
    from scripts.create_paper_results_table import (
        DATASETS,
        DATASET_LABELS,
        DEFAULT_BASE_DIR,
        DEFAULT_OUTPUT_DIR,
        MODEL_LABELS,
        PROJECT_ROOT,
        SETTINGS,
        benchmark_csv_path,
        discover_validated_sources,
        pct_to_float,
        read_csv_rows,
    )
    from scripts.llm_pipeline.answer_normalization import normalized_jaccard_accuracy
except ImportError:
    from create_explanation_complexity_analysis import (
        complexity_bin,
        read_complexity_by_task,
    )
    from create_paper_results_table import (
        DATASETS,
        DATASET_LABELS,
        DEFAULT_BASE_DIR,
        DEFAULT_OUTPUT_DIR,
        MODEL_LABELS,
        PROJECT_ROOT,
        SETTINGS,
        benchmark_csv_path,
        discover_validated_sources,
        pct_to_float,
        read_csv_rows,
    )
    from llm_pipeline.answer_normalization import normalized_jaccard_accuracy


NEW_RESULTS = DEFAULT_OUTPUT_DIR / "chapter4_all_results.csv"
OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "old_vs_new_condition_metrics.csv"
OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "old_vs_new_interpretation_data.json"


def read_new_rows() -> list[dict[str, str]]:
    with NEW_RESULTS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_old_summary(source) -> dict[str, Any]:
    path = source.run_dir / "metrics" / f"{source.setting}_key_findings_summary.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)["key_findings_summary"][source.model_name]
    return data


def old_answer_metrics(source) -> dict[str, dict[str, float]]:
    rows = read_csv_rows(source.checkpoint_path)
    answer_column = f"{source.model_name}_final_answer"
    confidence_column = f"{source.model_name}_confidence_score"
    scores: dict[str, list[float]] = defaultdict(list)
    alignments: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prediction = str(row.get(answer_column, "") or "")
        if not prediction.strip() or prediction.startswith(("[ERROR]", "ERROR")):
            continue
        group = "BQA" if row.get("Answer Type") == "BIN" else "OEQA"
        score = normalized_jaccard_accuracy(
            row.get("Answer", ""), prediction, str(row.get("Answer Type", ""))
        )
        try:
            confidence = float(row.get(confidence_column, 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        scores[group].append(100.0 * score)
        alignments[group].append(100.0 * (1.0 - abs(confidence - score)))
    return {
        group: {
            "jaccard": mean(scores[group]),
            "alignment": mean(alignments[group]),
        }
        for group in ("BQA", "OEQA")
    }


def condition_rows() -> list[dict[str, Any]]:
    new_lookup = {
        (row["dataset"], row["model_id"], row["setting"]): row
        for row in read_new_rows()
    }
    sources, _ = discover_validated_sources(DEFAULT_BASE_DIR)
    rows: list[dict[str, Any]] = []
    for key in sorted(sources):
        source = sources[key]
        hop = "1-hop" if source.hop == "1hop" else "2-hop"
        setting = SETTINGS[source.setting]
        new = new_lookup[(source.dataset, source.model_name, setting)]
        old_summary = read_old_summary(source)
        old = old_answer_metrics(source)
        hall = pct_to_float(old_summary["overall_metrics"]["hallucination_score"])
        for group, new_prefix in (("BQA", "binary"), ("OEQA", "open_ended")):
            rows.append(
                {
                    "dataset": DATASET_LABELS[source.dataset],
                    "dataset_id": source.dataset,
                    "model": MODEL_LABELS[source.model_name],
                    "model_id": source.model_name,
                    "setting": setting,
                    "hop": hop,
                    "answer_group": group,
                    "old_jaccard": old[group]["jaccard"],
                    "old_alignment": old[group]["alignment"],
                    "new_answer_em": float(new[f"{new_prefix}_answer_em_{hop}"]),
                    "new_answer_f1": float(
                        new[f"{new_prefix}_answer_em_{hop}"]
                        if group == "BQA"
                        else new[f"{new_prefix}_answer_f1_{hop}"]
                    ),
                    "new_alignment": float(
                        new[
                            f"{new_prefix}_confidence_correctness_alignment_{hop}"
                        ]
                    ),
                    "old_hallucination": hall if group == "OEQA" else None,
                    "new_hallucination": float(
                        new[f"open_ended_hallucination_{hop}"]
                    )
                    if group == "OEQA"
                    else None,
                }
            )
    return rows


def write_condition_csv(rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def ranked_means(
    rows: list[dict[str, Any]], group_key: str, metric: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row[group_key])].append(float(row[metric]))
    return sorted(
        ({"name": key, "mean": mean(values)} for key, values in buckets.items()),
        key=lambda item: (-item["mean"], item["name"]),
    )


def dataset_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset_id in DATASETS:
        selected = [row for row in rows if row["dataset_id"] == dataset_id]
        entry: dict[str, Any] = {}
        for answer_group in ("BQA", "OEQA"):
            group_rows = [row for row in selected if row["answer_group"] == answer_group]
            entry[answer_group] = {
                "old_setting_order": ranked_means(
                    group_rows, "setting", "old_jaccard"
                ),
                "new_setting_order": ranked_means(
                    group_rows, "setting", "new_answer_f1"
                ),
                "old_hop_order": ranked_means(group_rows, "hop", "old_jaccard"),
                "new_hop_order": ranked_means(group_rows, "hop", "new_answer_f1"),
                "old_model_order": ranked_means(
                    group_rows, "model", "old_jaccard"
                ),
                "new_model_order": ranked_means(
                    group_rows, "model", "new_answer_f1"
                ),
                "old_overall_mean": mean(
                    [float(row["old_jaccard"]) for row in group_rows]
                ),
                "new_overall_mean_f1": mean(
                    [float(row["new_answer_f1"]) for row in group_rows]
                ),
            }
            old_setting = {
                item["name"]: item["mean"]
                for item in entry[answer_group]["old_setting_order"]
            }
            new_setting = {
                item["name"]: item["mean"]
                for item in entry[answer_group]["new_setting_order"]
            }
            entry[answer_group]["old_abstraction_minus_nl"] = (
                old_setting["AR"] - old_setting["NL"]
            )
            entry[answer_group]["new_abstraction_minus_nl"] = (
                new_setting["AR"] - new_setting["NL"]
            )
        result[DATASET_LABELS[dataset_id]] = entry
    return result


def old_complexity_rows() -> list[dict[str, Any]]:
    sources, _ = discover_validated_sources(DEFAULT_BASE_DIR)
    checkpoint_cache: dict[Path, list[dict[str, str]]] = {}
    complexity_cache: dict[tuple[str, str, str], dict[str, int]] = {}
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key in sorted(sources):
        source = sources[key]
        rows = checkpoint_cache.setdefault(
            source.checkpoint_path, read_csv_rows(source.checkpoint_path)
        )
        complexity_key = (source.dataset, source.hop, source.setting)
        complexity = complexity_cache.setdefault(
            complexity_key,
            read_complexity_by_task(
                benchmark_csv_path(
                    PROJECT_ROOT, source.dataset, source.hop, source.setting
                )
            ),
        )
        answer_column = f"{source.model_name}_final_answer"
        for row in rows:
            prediction = str(row.get(answer_column, "") or "")
            if not prediction.strip() or prediction.strip().upper().startswith(
                ("[ERROR]", "ERROR")
            ):
                continue
            answer_group = "BQA" if row.get("Answer Type") == "BIN" else "OEQA"
            task_id = str(row["Task ID"])
            bin_name = complexity_bin(complexity[task_id])
            if bin_name not in ("Low", "High"):
                continue
            buckets[(source.dataset, answer_group, bin_name)].append(
                normalized_jaccard_accuracy(
                    row.get("Answer", ""), prediction, str(row.get("Answer Type", ""))
                )
            )
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        row: dict[str, Any] = {"dataset": DATASET_LABELS[dataset]}
        for answer_group in ("BQA", "OEQA"):
            low = mean(buckets[(dataset, answer_group, "Low")]) * 100.0
            high = mean(buckets[(dataset, answer_group, "High")]) * 100.0
            row[f"{answer_group}_low"] = low
            row[f"{answer_group}_high"] = high
            row[f"{answer_group}_delta"] = high - low
        output.append(row)
    return output


def main() -> None:
    rows = condition_rows()
    write_condition_csv(rows)
    payload = {
        "dataset_aggregates": dataset_aggregates(rows),
        "old_jaccard_complexity_by_dataset": old_complexity_rows(),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
