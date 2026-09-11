"""Thin Chapter 4 adapter for the authoritative SAGE-QA answer evaluator.

This module deliberately contains no Answer EM or Answer F1 formula. It loads
the vendored, manuscript-frozen SAGE-QA ``answer_set_scores``/``evaluate``
implementation, verifies its canonical source hash, validates Chapter 4
checkpoint rows against the frozen benchmark sampling CSV, and only adapts
field names.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


# The historical working-tree file used CRLF line endings. The canonical hash
# below is over the same source normalized to LF, so validation is portable.
SAGEQA_EVALUATOR_SHA256 = (
    "1610a67d64c48d46e0530dc71ffe073ca6ddae98d0a9b3c83116b0009a384713"
)
SAGEQA_EVALUATOR_CANONICAL_SHA256 = (
    "86b9bedfb3784145f3d20e2b9b8b6082ee4252e57918807bf02529b3904eefd6"
)
SAGEQA_SOURCE_COMMIT = "dbdbb50708bdc6c686ef82518ec71c1d1bf55985"
SAGEQA_EVALUATOR_RELATIVE_PATH = Path(
    "vendor/sageqa_evaluate_owl_qa_predictions.py"
)
MODEL_ANSWER_SUFFIX = "_final_answer"


class BenchmarkMismatchError(ValueError):
    """A saved prediction CSV is not aligned to its declared benchmark."""


def sageqa_evaluator_path() -> Path:
    return Path(__file__).resolve().parent / SAGEQA_EVALUATOR_RELATIVE_PATH


@lru_cache(maxsize=1)
def load_sageqa_evaluator() -> ModuleType:
    """Load the exact vendored SAGE-QA evaluator, failing on source drift."""

    path = sageqa_evaluator_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Vendored SAGE-QA evaluator not found at {path}."
        )

    canonical_source = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(canonical_source).hexdigest()
    if digest != SAGEQA_EVALUATOR_CANONICAL_SHA256:
        raise RuntimeError(
            "Vendored SAGE-QA evaluator hash mismatch: "
            f"expected {SAGEQA_EVALUATOR_CANONICAL_SHA256}, got {digest} ({path})"
        )

    spec = importlib.util.spec_from_file_location(
        "sageqa_authoritative_owl_evaluator", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SAGE-QA evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def answer_set_scores(prediction: Any, gold: Any) -> tuple[float, float, float, float]:
    """Delegate Answer EM/F1/precision/recall to SAGE-QA unchanged."""

    return load_sageqa_evaluator().answer_set_scores(
        "" if prediction is None else str(prediction),
        "" if gold is None else str(gold),
    )


def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Delegate the normal ontology evaluation path to SAGE-QA unchanged."""

    return load_sageqa_evaluator().evaluate(*args, **kwargs)


def benchmark_csv_path(project_root: Path, dataset: str, hop: str, setting: str) -> Path:
    suffix = "_abs" if setting == "abs" else ""
    return (
        project_root
        / "data"
        / "output"
        / dataset
        / hop
        / f"SPARQL_questions_sampling{suffix}.csv"
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def detect_models(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return sorted(
        name[: -len(MODEL_ANSWER_SUFFIX)]
        for name in rows[0]
        if name.endswith(MODEL_ANSWER_SUFFIX)
    )


def validate_checkpoint_rows(
    checkpoint_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, str]],
    checkpoint_path: Path | str = "checkpoint",
) -> None:
    """Require a one-to-one, content-preserving join on Chapter 4 Task ID."""

    label = str(checkpoint_path)
    for name, rows in (("checkpoint", checkpoint_rows), ("benchmark", benchmark_rows)):
        ids = [str(row.get("Task ID", "")) for row in rows]
        missing_id_rows = sum(not value for value in ids)
        duplicates = sorted(
            value for value, count in Counter(ids).items() if value and count > 1
        )
        if missing_id_rows or duplicates:
            raise BenchmarkMismatchError(
                f"{label}: invalid {name} Task IDs "
                f"(blank={missing_id_rows}, duplicates={duplicates[:5]})"
            )

    checkpoint_by_id = {str(row["Task ID"]): row for row in checkpoint_rows}
    benchmark_by_id = {str(row["Task ID"]): row for row in benchmark_rows}
    missing = sorted(set(benchmark_by_id) - set(checkpoint_by_id))
    extra = sorted(set(checkpoint_by_id) - set(benchmark_by_id))
    if missing or extra:
        raise BenchmarkMismatchError(
            f"{label}: benchmark Task ID mismatch "
            f"(checkpoint={len(checkpoint_rows)}, benchmark={len(benchmark_rows)}, "
            f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]})"
        )

    content_mismatches: list[str] = []
    for task_id, benchmark_row in benchmark_by_id.items():
        checkpoint_row = checkpoint_by_id[task_id]
        for field in ("Answer Type", "Answer"):
            if str(checkpoint_row.get(field, "")) != str(benchmark_row.get(field, "")):
                content_mismatches.append(f"{task_id}:{field}")
                break
    if content_mismatches:
        raise BenchmarkMismatchError(
            f"{label}: benchmark semantic-content mismatch for "
            f"{len(content_mismatches)} rows ({content_mismatches[:5]})"
        )


def confidence_correctness_alignment(confidence: Any, answer_f1: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.5
    value = max(0.0, min(1.0, value))
    return 1.0 - abs(value - answer_f1)


def score_checkpoint_rows(
    rows: list[dict[str, str]], model_name: str
) -> dict[str, Any]:
    """Score every row, including empty/error predictions, via SAGE-QA."""

    answer_col = f"{model_name}{MODEL_ANSWER_SUFFIX}"
    confidence_col = f"{model_name}_confidence_score"
    if not rows or answer_col not in rows[0]:
        raise ValueError(f"Missing required model answer column: {answer_col}")

    per_question: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prediction = row.get(answer_col, "") or ""
        gold = row.get("Answer", "") or ""
        em, f1, precision, recall = answer_set_scores(prediction, gold)
        answer_type = str(row.get("Answer Type", "BIN")).strip().upper()
        bucket = "binary" if answer_type == "BIN" else "open"
        scored = {
            "task_id": str(row.get("Task ID", "")),
            "answer_type": answer_type,
            "prediction": prediction,
            "gold": gold,
            "answer_em": em,
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "confidence_correctness_alignment": confidence_correctness_alignment(
                row.get(confidence_col, 0.5), f1
            ),
        }
        per_question.append(scored)
        grouped[bucket].append(scored)
        grouped["all"].append(scored)

    aggregates: dict[str, dict[str, float | int]] = {}
    for bucket in ("binary", "open", "all"):
        items = grouped.get(bucket, [])
        count = len(items)
        aggregates[bucket] = {
            "n": count,
            "answer_em": sum(item["answer_em"] for item in items) / count
            if count
            else 0.0,
            "answer_f1": sum(item["answer_f1"] for item in items) / count
            if count
            else 0.0,
            "confidence_correctness_alignment": sum(
                item["confidence_correctness_alignment"] for item in items
            )
            / count
            if count
            else 0.0,
        }
    return {"aggregates": aggregates, "per_question": per_question}
