#!/usr/bin/env python3
"""Compare corrected reasoning-tag aggregates with the published-v1.1 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = {
    "tag_frequencies": ["tag_basis", "task", "tag"],
    "tag_cooccurrence": ["tag_basis", "task", "tag_a", "tag_b"],
    "descriptive_by_task_tag_representation": ["task", "representation", "scope", "tag_basis", "tag"],
    "descriptive_by_model": ["task", "representation", "model", "scope", "tag_basis", "tag"],
    "descriptive_by_dataset_hop": ["task", "representation", "dataset", "hop", "scope", "tag_basis", "tag"],
    "bqa_label_supplement": ["task", "representation", "bqa_label", "scope", "tag_basis", "tag"],
    "oeqa_gold_set_size": ["task", "representation", "gold_answer_set_size", "scope", "tag_basis", "tag"],
    "raw_tag_contrasts": ["scope", "tag_basis", "task", "representation", "tag", "outcome"],
    "adjusted_associations": ["scope", "tag_basis", "task", "outcome", "term"],
    "tied_minimum_sensitivity": ["task", "representation", "scope", "tag"],
    "defective_ar_sensitivity": ["task", "representation", "scope_all", "tag_basis_all", "tag"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=Path("results/v1.1.0-reasoning-tag-analysis/csv"))
    parser.add_argument("--corrected", type=Path, default=Path("results/v1.1.0-corrected-rerun-review/reasoning-tag-analysis/csv"))
    parser.add_argument("--output", type=Path, default=Path("results/v1.1.0-corrected-rerun-review/comparison/reasoning_tags"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for name, keys in KEYS.items():
        old = pd.read_csv(args.published / f"{name}.csv")
        new = pd.read_csv(args.corrected / f"{name}.csv")
        merged = new.merge(old, on=keys, how="outer", suffixes=("_corrected", "_published"), indicator=True)
        merged["comparison_status"] = merged.pop("_merge").map({
            "both": "matched", "left_only": "added_corrected", "right_only": "published_only"
        })
        numeric_common = [
            column for column in new.columns
            if column not in keys and column in old.columns
            and pd.api.types.is_numeric_dtype(new[column])
            and pd.api.types.is_numeric_dtype(old[column])
        ]
        for column in numeric_common:
            merged[f"delta_{column}"] = (
                merged[f"{column}_corrected"] - merged[f"{column}_published"]
            )
        merged.to_csv(args.output / f"{name}_vs_published.csv", index=False, lineterminator="\n")
        matched = merged[merged["comparison_status"] == "matched"]
        summary[name] = {
            "published_rows": len(old), "corrected_rows": len(new),
            "matched_rows": len(matched),
            "added_corrected_rows": int((merged["comparison_status"] == "added_corrected").sum()),
            "published_only_rows": int((merged["comparison_status"] == "published_only").sum()),
            "maximum_absolute_numeric_delta": {
                column: float(matched[f"delta_{column}"].abs().max())
                for column in numeric_common if len(matched) and matched[f"delta_{column}"].notna().any()
            },
        }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "tables": len(KEYS), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
