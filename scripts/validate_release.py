#!/usr/bin/env python3
"""Offline integrity validation for the CORE-LLM-Bench v1.0 release."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "final_benchmark"
VERSION = "1.0.0"

ARTIFACTS = (
    ("Family", "1hop", "FamilyOWL.zip", "FamilyOWL_1hop.json", 1880),
    ("Family", "2hop", "FamilyOWL.zip", "FamilyOWL_2hop.json", 1880),
    ("Pizza100", "1hop", "pizza_100.zip", "pizza_100_1hop.json", 492),
    ("Pizza100", "2hop", "pizza_100.zip", "pizza_100_2hop.json", 492),
    ("Pizza250", "1hop", "pizza_250.zip", "pizza_250_1hop.json", 616),
    ("Pizza250", "2hop", "pizza_250.zip", "pizza_250_2hop.json", 616),
    ("OWL2Bench", "1hop", "OWL2Bench.zip", "OWL2Bench_1hop.json", 1466),
    ("OWL2Bench", "2hop", "OWL2Bench.zip", "OWL2Bench_2hop.json", 1590),
)
EXPECTED_DATASET_TOTALS = {
    "Family": 3760,
    "Pizza100": 984,
    "Pizza250": 1232,
    "OWL2Bench": 3056,
}
EXPECTED_TASK_TOTALS = {"BQA": 5999, "OEQA": 3033}
EXPECTED_REASONING_COUNTS = {
    "D": 9032,
    "H": 3278,
    "I": 2324,
    "R": 1660,
    "M": 1020,
    "N": 168,
    "S": 55,
    "T": 2,
}
REASONING_TAXONOMY = {
    "D", "H", "T", "S", "A", "J", "N", "E", "∩", "¬",
    "I", "F", "V", "Y", "Q", "R", "C", "L", "U", "M",
}
GROUP_FIELDS = {
    "Task Type", "Answer Type", "Root Entity", "OWL Context", "NL Context",
    "ABS Context", "QAs",
}
QA_FIELDS = {
    "Task ID", "SPARQL Query", "NL Question", "ABS Question", "ABS Answer",
    "Answer", "Minimum Explanation", "Explanations", "Explanation Count",
    "Explanation Min", "Explanation Max",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_member(zip_path: Path, member: str) -> list[dict[str, Any]]:
    if not zip_path.is_file():
        raise ValueError(f"Missing benchmark package: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure in {zip_path}: {bad}")
            if member not in archive.namelist():
                raise ValueError(f"Missing {member} in {zip_path}")
            value = json.loads(archive.read(member).decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable artifact {zip_path}::{member}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty JSON list in {zip_path}::{member}")
    return value


def iter_flat_rows(
    groups: list[dict[str, Any]], dataset: str, hop: str
) -> Iterator[dict[str, Any]]:
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"{dataset}/{hop} group {group_index} is not an object")
        missing = GROUP_FIELDS - set(group)
        if missing:
            raise ValueError(f"{dataset}/{hop} group {group_index} missing {sorted(missing)}")
        if not isinstance(group["QAs"], list) or not group["QAs"]:
            raise ValueError(f"{dataset}/{hop} group {group_index} has no QAs")
        for qa_index, qa in enumerate(group["QAs"]):
            if not isinstance(qa, dict):
                raise ValueError(f"{dataset}/{hop} QA {qa_index} is not an object")
            missing = QA_FIELDS - set(qa)
            if missing:
                raise ValueError(f"{dataset}/{hop} QA {qa_index} missing {sorted(missing)}")
            yield {"dataset": dataset, "hop": hop, "group": group, "qa": qa}


def nonblank(value: Any) -> bool:
    return value is not None and bool(str(value).strip()) and str(value) != "None"


def parse_explanations(value: Any, source: str) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    elif nonblank(value):
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid explanation serialization in {source}") from exc
    else:
        return []
    if not isinstance(parsed, list):
        raise ValueError(f"Explanations are not a list in {source}")
    return parsed


def load_reasoning_metadata(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing reasoning metadata index: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dataset", "hop", "task_id", "task_group", "min_tag_length",
        "max_tag_length", "reasoning_tags", "is_negative_bqa",
        "linked_positive_task_id", "benchmark_version",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"Reasoning metadata schema is incomplete: {path}")
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["hop"], row["task_id"])
        if key in result:
            raise ValueError(f"Duplicate reasoning metadata key: {key}")
        result[key] = row
    return result


def validate_manifest() -> None:
    path = BENCHMARK_DIR / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing release manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_version") != VERSION:
        raise ValueError("Release manifest benchmark_version mismatch")
    for package in manifest.get("packages", []):
        package_path = ROOT / package["path"]
        if package_path.stat().st_size != package["bytes"]:
            raise ValueError(f"Size mismatch for {package_path}")
        if sha256_file(package_path) != package["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {package_path}")
        members = package.get("members", [])
        if members:
            with zipfile.ZipFile(package_path) as archive:
                for member in members:
                    content = archive.read(member["path"])
                    if len(content) != member["bytes"]:
                        raise ValueError(
                            f"Size mismatch for {package_path}::{member['path']}"
                        )
                    if hashlib.sha256(content).hexdigest() != member["sha256"]:
                        raise ValueError(
                            f"SHA-256 mismatch for {package_path}::{member['path']}"
                        )


def validate_release() -> dict[str, Any]:
    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != VERSION:
        raise ValueError("VERSION does not contain 1.0.0")
    validate_manifest()
    metadata = load_reasoning_metadata(BENCHMARK_DIR / "reasoning_metadata.csv")

    seen: set[tuple[str, str, str]] = set()
    dataset_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    artifact_counts: dict[str, int] = {}
    reasoning_counts: Counter[str] = Counter()

    for dataset, hop, zip_name, member, expected_rows in ARTIFACTS:
        groups = load_member(BENCHMARK_DIR / zip_name, member)
        row_count = 0
        for item in iter_flat_rows(groups, dataset, hop):
            row_count += 1
            group, qa = item["group"], item["qa"]
            task_id = str(qa["Task ID"]).strip()
            key = (dataset, hop, task_id)
            if not task_id or key in seen:
                raise ValueError(f"Blank or duplicate benchmark identity: {key}")
            if not task_id.startswith(f"{hop}-"):
                raise ValueError(f"Task ID has the wrong hop prefix: {task_id}")
            seen.add(key)
            dataset_counts[dataset] += 1

            if group["Task Type"] not in {"Membership", "Property Assertion"}:
                raise ValueError(f"Invalid reasoning task for {key}: {group['Task Type']}")

            for field in ("Root Entity", "OWL Context", "NL Context", "ABS Context"):
                if not nonblank(group[field]):
                    raise ValueError(f"Blank {field} for {key}")
            for field in ("SPARQL Query", "NL Question", "ABS Question", "Answer"):
                if not nonblank(qa[field]):
                    raise ValueError(f"Blank {field} for {key}")

            answer_type = str(group["Answer Type"]).upper()
            answer = str(qa["Answer"]).upper()
            if answer_type == "BIN":
                task_group = "BQA"
                if answer not in {"TRUE", "FALSE"}:
                    raise ValueError(f"Invalid BQA label for {key}: {answer}")
                label_counts[answer] += 1
            elif answer_type == "MC":
                task_group = "OEQA"
                if not nonblank(qa["ABS Answer"]):
                    raise ValueError(f"Blank abstract gold answer for {key}")
            else:
                raise ValueError(f"Invalid Answer Type for {key}: {answer_type}")
            task_counts[task_group] += 1

            meta = metadata.get(key)
            if meta is None or meta["task_group"] != task_group:
                raise ValueError(f"Missing or conflicting reasoning metadata for {key}")
            if meta["benchmark_version"] != VERSION:
                raise ValueError(f"Metadata version mismatch for {key}")
            try:
                min_length = int(meta["min_tag_length"])
                max_length = int(meta["max_tag_length"])
            except ValueError as exc:
                raise ValueError(f"Invalid complexity metadata for {key}") from exc
            if min_length < 1 or max_length < min_length:
                raise ValueError(f"Invalid complexity range for {key}")
            tags = set(meta["reasoning_tags"])
            if not tags or not tags <= REASONING_TAXONOMY:
                raise ValueError(f"Invalid reasoning tags for {key}: {tags}")
            reasoning_counts.update(tags)

            negative = task_group == "BQA" and answer == "FALSE"
            if negative:
                if meta["is_negative_bqa"] != "true" or not meta["linked_positive_task_id"]:
                    raise ValueError(f"Negative BQA proof link is missing for {key}")
            else:
                explanations = parse_explanations(qa["Explanations"], str(key))
                if not explanations or not all(
                    nonblank(qa[field])
                    for field in ("Minimum Explanation", "Explanation Count", "Explanation Min", "Explanation Max")
                ):
                    raise ValueError(f"Explanation metadata is incomplete for {key}")

        if row_count != expected_rows:
            raise ValueError(
                f"Row count mismatch for {member}: {row_count} != {expected_rows}"
            )
        artifact_counts[member] = row_count

    if len(metadata) != len(seen) or set(metadata) != seen:
        raise ValueError("Reasoning metadata and benchmark identity sets differ")
    if dict(dataset_counts) != EXPECTED_DATASET_TOTALS:
        raise ValueError(f"Dataset totals mismatch: {dict(dataset_counts)}")
    if dict(task_counts) != EXPECTED_TASK_TOTALS:
        raise ValueError(f"Task totals mismatch: {dict(task_counts)}")
    if len(seen) != 9032:
        raise ValueError(f"Overall total mismatch: {len(seen)}")
    if set(reasoning_counts) != set(EXPECTED_REASONING_COUNTS):
        raise ValueError(f"Instantiated reasoning tags mismatch: {dict(reasoning_counts)}")
    if dict(reasoning_counts) != EXPECTED_REASONING_COUNTS:
        raise ValueError(f"Reasoning coverage counts mismatch: {dict(reasoning_counts)}")

    return {
        "benchmark_version": VERSION,
        "status": "PASS",
        "artifacts": artifact_counts,
        "dataset_totals": dict(dataset_counts),
        "task_totals": dict(task_counts),
        "binary_labels": dict(label_counts),
        "overall_total": len(seen),
        "taxonomy_size": len(REASONING_TAXONOMY),
        "instantiated_reasoning_counts": dict(reasoning_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args()
    try:
        report = validate_release()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"RELEASE VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("CORE-LLM-Bench release validation: PASS")
        print(f"Version: {report['benchmark_version']}")
        print(f"Unique question-hop instances: {report['overall_total']:,}")
        print(f"BQA / OEQA: {report['task_totals']['BQA']:,} / {report['task_totals']['OEQA']:,}")
        print(f"Reasoning taxonomy / instantiated: {report['taxonomy_size']} / {len(report['instantiated_reasoning_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
