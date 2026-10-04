#!/usr/bin/env python3
"""Repair stale proof-derived metadata in the corrected v1.1 benchmark.

The answer-group alternatives are the canonical proof source.  This script is
strictly offline: it imports no API client and performs no model inference.
Only rows whose cached minimum primitive complexity disagrees with an exact
answer-group reconstruction are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evaluate_v1_1_corrected_rerun_offline import rendered_axiom_tags
except ModuleNotFoundError:
    from scripts.evaluate_v1_1_corrected_rerun_offline import rendered_axiom_tags


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data/correction/v1.1.0-semantic-audit/"
    "core_llm_bench_v1_1_corrected.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "data/correction/v1.1.1/"
    "core_llm_bench_v1_1_1_corrected.parquet"
)
DEFAULT_REPORT = ROOT / "data/correction/v1.1.1/proof_metadata_repair_report.json"
EXPECTED_STALE_ROWS = 885


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decoded(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def axiom_tag(axiom: dict[str, Any]) -> str:
    explicit = axiom.get("tag")
    tags = (
        (explicit,)
        if isinstance(explicit, str) and len(explicit) == 1
        else rendered_axiom_tags(str(axiom.get("axiom", "")))
    )
    primitive = tuple(tag for tag in tags if tag != "M")
    if len(primitive) != 1:
        raise ValueError(f"Expected one non-M primitive tag, found {primitive!r}")
    return primitive[0]


def reconstruct(answer_groups: list[dict[str, Any]]) -> dict[str, Any]:
    # A state is an immutable identity -> primitive-tag mapping.  M is never
    # admitted, and repeated/shared axioms collapse by semantic identity.
    states: set[frozenset[tuple[str, str]]] = {frozenset()}
    combination_count = 1
    for group in answer_groups:
        alternatives = group.get("alternatives", [])
        if not alternatives:
            raise ValueError("Answer group has no proof alternative")
        combination_count *= len(alternatives)
        next_states: set[frozenset[tuple[str, str]]] = set()
        for state in states:
            current = dict(state)
            for alternative in alternatives:
                merged = dict(current)
                for axiom in alternative.get("axioms", []):
                    identity = str(axiom.get("axiom", ""))
                    if not identity:
                        raise ValueError("Proof axiom has no semantic identity")
                    tag = axiom_tag(axiom)
                    if identity in merged and merged[identity] != tag:
                        raise ValueError(
                            f"Inconsistent primitive tag for shared axiom {identity!r}"
                        )
                    merged[identity] = tag
                next_states.add(frozenset(merged.items()))
        states = next_states
    if not states:
        raise ValueError("No complete proof states reconstructed")

    complexities = {state: len(state) for state in states}
    minimum = min(complexities.values())
    maximum = max(complexities.values())
    minima = sorted(
        (state for state, count in complexities.items() if count == minimum),
        key=lambda state: tuple(sorted(state)),
    )
    minimum_explanations = []
    for state in minima:
        ordered = sorted(state)
        minimum_explanations.append(
            {
                "axioms": [
                    {"axiom": identity, "tag": tag} for identity, tag in ordered
                ],
                "tag_sequence": "".join(tag for _, tag in ordered),
            }
        )

    minimum_tag_sets = [{tag for _, tag in state} for state in minima]
    primitive_tags = sorted(set().union(*minimum_tag_sets))
    return {
        "complete_explanation": {
            "combination_count": combination_count,
            "max_axiom_count": maximum,
            "min_axiom_count": minimum,
            "minimum_explanations": minimum_explanations,
            "correction_note": (
                "Exact answer-group reconstruction; conjunctive across answers, "
                "disjunctive within an answer; shared axioms deduplicated by "
                "semantic identity; M excluded."
            ),
        },
        "raw_minimum_complete_primitive_tag_complexity": minimum,
        "raw_maximum_complete_primitive_tag_complexity": maximum,
        "primitive_reasoning_tags": primitive_tags,
        "distinct_primitive_reasoning_types": primitive_tags,
        "minimum_distinct_primitive_type_count": min(map(len, minimum_tag_sets)),
        "maximum_distinct_primitive_type_count": max(map(len, minimum_tag_sets)),
        "m_status": "never",
    }


def complexity_bin(task_group: str, minimum: int) -> str:
    if task_group == "BQA":
        return "Low" if minimum == 1 else "Medium" if minimum == 2 else "High"
    return "Low" if minimum <= 3 else "Medium" if minimum <= 5 else "High"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if source == output:
        raise RuntimeError("Refusing to overwrite the source corrected Parquet")
    frame = pd.read_parquet(source)
    if len(frame) != 9048 or frame.task_id.nunique() != 9048:
        raise RuntimeError("Corrected benchmark membership is not 9,048 unique tasks")

    reconstructed: dict[int, dict[str, Any]] = {}
    stale_ids: list[int] = []
    for row in frame.to_dict(orient="records"):
        task_id = int(row["task_id"])
        summary = reconstruct(decoded(row["answer_explanations"]))
        reconstructed[task_id] = summary
        if (
            int(row["raw_minimum_complete_primitive_tag_complexity"])
            != summary["raw_minimum_complete_primitive_tag_complexity"]
        ):
            stale_ids.append(task_id)
    if len(stale_ids) != EXPECTED_STALE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_STALE_ROWS} stale rows, found {len(stale_ids)}"
        )

    repaired = frame.copy()
    index_by_task = {int(value): index for index, value in repaired.task_id.items()}
    derived_fields = (
        "complete_explanation",
        "raw_minimum_complete_primitive_tag_complexity",
        "raw_maximum_complete_primitive_tag_complexity",
        "primitive_reasoning_tags",
        "distinct_primitive_reasoning_types",
        "minimum_distinct_primitive_type_count",
        "maximum_distinct_primitive_type_count",
        "m_status",
        "complexity_bin",
    )
    for task_id in stale_ids:
        index = index_by_task[task_id]
        summary = reconstructed[task_id]
        minimum = summary["raw_minimum_complete_primitive_tag_complexity"]
        values = {
            **summary,
            "complexity_bin": complexity_bin(str(repaired.at[index, "task_group"]), minimum),
        }
        for field in derived_fields:
            value = values[field]
            if field in {
                "complete_explanation",
                "primitive_reasoning_tags",
                "distinct_primitive_reasoning_types",
            }:
                value = canonical(value)
            repaired.at[index, field] = value

    # Prediction/prompt/gold/proof sources are immutable in this repair.
    allowed = set(derived_fields)
    changed_fields = {
        field for field in repaired.columns if not frame[field].equals(repaired[field])
    }
    unauthorized = changed_fields - allowed
    if unauthorized:
        raise RuntimeError(f"Unauthorized fields changed: {sorted(unauthorized)}")
    changed_ids = {
        int(frame.at[index, "task_id"])
        for field in changed_fields
        for index in frame.index
        if frame.at[index, field] != repaired.at[index, field]
    }
    if changed_ids != set(stale_ids):
        raise RuntimeError("Changed task set does not exactly equal the stale task set")

    output.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(output, index=False)
    roundtrip = pd.read_parquet(output)
    if len(roundtrip) != 9048 or roundtrip.task_id.nunique() != 9048:
        raise RuntimeError("Repaired Parquet round-trip membership failed")
    residual = 0
    for row in roundtrip.to_dict(orient="records"):
        summary = reconstruct(decoded(row["answer_explanations"]))
        residual += int(
            int(row["raw_minimum_complete_primitive_tag_complexity"])
            != summary["raw_minimum_complete_primitive_tag_complexity"]
        )
    if residual:
        raise RuntimeError(f"Repaired Parquet retains {residual} stale summaries")

    report = {
        "status": "PASS",
        "offline_only": True,
        "source_path": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "output_path": output.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(output),
        "questions": 9048,
        "stale_rows_detected": len(stale_ids),
        "rows_repaired": len(changed_ids),
        "residual_stale_rows": residual,
        "changed_fields": sorted(changed_fields),
        "repaired_task_ids": stale_ids,
        "prediction_fields_modified": False,
        "answer_group_proofs_modified": False,
        "method": (
            "exact answer-group proof reconstruction with shared-axiom "
            "deduplication and M exclusion"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: report[k] for k in (
        "status", "questions", "stale_rows_detected", "rows_repaired",
        "residual_stale_rows", "output_sha256"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
