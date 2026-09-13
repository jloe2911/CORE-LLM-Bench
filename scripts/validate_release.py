#!/usr/bin/env python3
"""Offline integrity validation for the CORE-LLM-Bench v1.0.0 release."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from release_profiles import (
    PROFILES,
    VERSION,
    artifacts_for,
    get_profile,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "final_benchmark"
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
PUBLIC_SAFE_MARKERS = (
    b"FamilyOWL",
    b'"dataset": "Family"',
    b"http://www.co-ode.org/roberts/family-tree.owl",
    b"http://www.semanticweb.org/family",
    b"fhkb",
)
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
FAMILY_REFERENCE_ALLOWLIST = {
    "docs/FAMILY_RECONSTRUCTION.md",
    "scripts/validate_family_reconstruction.py",
    "release/RELEASE_NOTES_v1.0.0_PUBLIC_SAFE.md",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}", re.I),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:Users|home)\\", re.I),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)
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


def validate_release(profile_name: str = "full") -> dict[str, Any]:
    profile = get_profile(profile_name)
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

    for artifact in artifacts_for(profile):
        dataset = artifact.dataset
        hop = artifact.hop
        zip_name = artifact.zip_name
        member = artifact.member
        expected_rows = artifact.rows
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

    selected_metadata = {key for key in metadata if key[0] in profile.datasets}
    if selected_metadata != seen:
        raise ValueError("Reasoning metadata and benchmark identity sets differ")
    if dict(dataset_counts) != profile.dataset_totals:
        raise ValueError(f"Dataset totals mismatch: {dict(dataset_counts)}")
    if dict(task_counts) != profile.task_totals:
        raise ValueError(f"Task totals mismatch: {dict(task_counts)}")
    if len(seen) != profile.overall_total:
        raise ValueError(f"Overall total mismatch: {len(seen)}")
    if profile.name == "full":
        if set(reasoning_counts) != set(EXPECTED_REASONING_COUNTS):
            raise ValueError(
                f"Instantiated reasoning tags mismatch: {dict(reasoning_counts)}"
            )
        if dict(reasoning_counts) != EXPECTED_REASONING_COUNTS:
            raise ValueError(
                f"Reasoning coverage counts mismatch: {dict(reasoning_counts)}"
            )

    return {
        "benchmark_version": VERSION,
        "profile": profile.name,
        "status": "PASS",
        "artifacts": artifact_counts,
        "dataset_totals": dict(dataset_counts),
        "task_totals": dict(task_counts),
        "binary_labels": dict(label_counts),
        "overall_total": len(seen),
        "taxonomy_size": len(REASONING_TAXONOMY),
        "instantiated_reasoning_counts": dict(reasoning_counts),
    }


def _validate_text_safety(path: Path, text: str) -> None:
    if "v1.0," + ".0" in text:
        raise ValueError(f"Malformed version string in {path}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Possible secret in {path}")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Absolute local path in {path}")


def assert_no_family_payload(package_dir: Path) -> None:
    """Reject Family artifacts or records while allowing explanatory documentation."""
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir).as_posix()
        if "family" in relative.lower() and relative not in FAMILY_REFERENCE_ALLOWLIST:
            raise ValueError(f"Family artifact found in public-safe package: {relative}")
        if path.suffix.lower() in {".zip", ".parquet", ".csv", ".json"}:
            if path.suffix.lower() == ".parquet":
                try:
                    import pyarrow.parquet as pq
                except ImportError as exc:
                    raise ValueError("pyarrow is required to inspect Parquet payloads") from exc
                table = pq.read_table(path)
                if "dataset" in table.column_names:
                    datasets = set(table.column("dataset").to_pylist())
                    if "Family" in datasets or "FamilyOWL" in datasets:
                        raise ValueError(f"Family records found in {relative}")
                continue
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if "family" in member.lower():
                            raise ValueError(f"Family member found in {relative}: {member}")
                        content = archive.read(member).lower()
                        if any(marker.lower() in content for marker in PUBLIC_SAFE_MARKERS):
                            raise ValueError(f"Family content found in {relative}: {member}")
                continue
            if relative == "zenodo_metadata.json":
                continue
            content = path.read_bytes().lower()
            if any(marker.lower() in content for marker in PUBLIC_SAFE_MARKERS):
                raise ValueError(f"Family content found in public-safe payload: {relative}")


def _read_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    return checksums


def validate_prepared_package(
    package_dir: Path, profile_name: str, package_kind: str
) -> dict[str, Any]:
    profile = get_profile(profile_name)
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "RELEASE_MANIFEST.json"
    checksum_path = package_dir / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ValueError(f"Missing release manifest or checksums in {package_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_version") != VERSION or manifest.get("profile") != profile.name:
        raise ValueError("Prepared package version/profile mismatch")
    if manifest.get("statistics", {}).get("overall") != profile.overall_total:
        raise ValueError("Prepared package statistics mismatch")

    listed = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.name not in {"RELEASE_MANIFEST.json", "SHA256SUMS"}
    }
    if set(listed) != actual:
        raise ValueError(
            f"Manifest file set mismatch: missing={sorted(actual - set(listed))}, "
            f"extra={sorted(set(listed) - actual)}"
        )
    checksums = _read_checksum_file(checksum_path)
    checksum_expected = actual | {"RELEASE_MANIFEST.json"}
    if set(checksums) != checksum_expected:
        raise ValueError("SHA256SUMS file set mismatch")
    for relative in sorted(actual):
        path = package_dir / relative
        if path.stat().st_size != listed[relative]["bytes"]:
            raise ValueError(f"Manifest size mismatch: {relative}")
        digest = sha256_file(path)
        if digest != listed[relative]["sha256"] or checksums[relative] != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    if checksums["RELEASE_MANIFEST.json"] != sha256_file(manifest_path):
        raise ValueError("Release manifest checksum mismatch")

    for path in package_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            _validate_text_safety(path, path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"CRC failure: {path}")
                for member in archive.namelist():
                    if member.endswith(".json"):
                        member_text = archive.read(member).decode("utf-8")
                        _validate_text_safety(Path(f"{path}--{member}"), member_text)
                        json.loads(member_text)

    if package_kind == "huggingface":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("pyarrow is required to validate Hugging Face export") from exc
        parquet_files = list(package_dir.glob("*.parquet"))
        if len(parquet_files) != 1:
            raise ValueError("Hugging Face package must contain exactly one Parquet file")
        table = pq.read_table(parquet_files[0])
        if table.num_rows != profile.overall_total:
            raise ValueError("Hugging Face row count mismatch")
        if set(table.column("dataset").to_pylist()) != set(profile.datasets):
            raise ValueError("Hugging Face dataset set mismatch")
        if set(table.column("benchmark_version").to_pylist()) != {VERSION}:
            raise ValueError("Hugging Face benchmark version mismatch")
        required_columns = {
            "task_id", "dataset", "hop", "task_type", "reasoning_task",
            "answer_type", "binary_label", "gold_answer", "ar_gold_answer",
            "root_entity", "nl_question", "nl_context", "fs_query", "fs_context",
            "ar_question", "ar_context", "minimum_explanation", "explanations",
            "explanation_count", "min_tag_length", "max_tag_length",
            "reasoning_tags", "linked_positive_task_id", "benchmark_version",
            "source_package", "source_member",
        }
        if set(table.column_names) != required_columns:
            raise ValueError("Hugging Face schema mismatch")
        identities = set(
            zip(
                table.column("dataset").to_pylist(),
                table.column("hop").to_pylist(),
                table.column("task_id").to_pylist(),
                strict=True,
            )
        )
        if len(identities) != profile.overall_total:
            raise ValueError("Duplicate Hugging Face question-hop identity")
        for field in table.schema:
            if str(field.type) not in {"string", "large_string"}:
                continue
            for value in table.column(field.name).to_pylist():
                if value is not None:
                    _validate_text_safety(
                        Path(f"{parquet_files[0]}--{field.name}"), value
                    )
    elif package_kind == "zenodo":
        benchmark_dir = package_dir / "benchmark"
        expected_zips = {
            artifact.zip_name for artifact in artifacts_for(profile)
        }
        actual_zips = {path.name for path in benchmark_dir.glob("*.zip")}
        if actual_zips != expected_zips:
            raise ValueError("Zenodo benchmark artifact set mismatch")
        seen: set[tuple[str, str, str]] = set()
        for artifact in artifacts_for(profile):
            groups = load_member(benchmark_dir / artifact.zip_name, artifact.member)
            rows = list(iter_flat_rows(groups, artifact.dataset, artifact.hop))
            if len(rows) != artifact.rows:
                raise ValueError(f"Zenodo row count mismatch: {artifact.member}")
            for item in rows:
                key = (
                    artifact.dataset,
                    artifact.hop,
                    str(item["qa"]["Task ID"]),
                )
                if key in seen:
                    raise ValueError(f"Duplicate Zenodo benchmark identity: {key}")
                seen.add(key)
        metadata = load_reasoning_metadata(benchmark_dir / "reasoning_metadata.csv")
        if set(metadata) != seen or len(seen) != profile.overall_total:
            raise ValueError("Zenodo benchmark/metadata identity mismatch")

    if not profile.includes_family:
        assert_no_family_payload(package_dir)
    return {
        "benchmark_version": VERSION,
        "status": "PASS",
        "profile": profile.name,
        "package_kind": package_kind,
        "overall_total": profile.overall_total,
        "files": len(actual) + 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument(
        "--package-kind", choices=("huggingface", "zenodo"), default="zenodo"
    )
    args = parser.parse_args()
    try:
        if args.package_dir:
            report = validate_prepared_package(
                args.package_dir, args.profile, args.package_kind
            )
        else:
            report = validate_release(args.profile)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"RELEASE VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"CORE-LLM-Bench {args.profile} release validation: PASS")
        print(f"Version: {report['benchmark_version']}")
        print(f"Unique question-hop instances: {report['overall_total']:,}")
        if args.package_dir:
            print(f"Package kind / files: {report['package_kind']} / {report['files']}")
        else:
            print(f"BQA / OEQA: {report['task_totals']['BQA']:,} / {report['task_totals']['OEQA']:,}")
            print(f"Reasoning taxonomy / instantiated: {report['taxonomy_size']} / {len(report['instantiated_reasoning_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
