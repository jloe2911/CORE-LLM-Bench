#!/usr/bin/env python3
"""Measure Pellet reasoning-type coverage in the final sampled benchmark.

The unit of analysis is a unique (dataset, hop, Task ID) question-hop instance.
Tags are unioned over every valid reasoner explanation associated with a question.
Negative binary questions inherit the explanation of the positive entailment from
which the generator constructed the paired false ASK query; that link is resolved
and validated against the complete generated question CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "reasoning_coverage"

DATASETS = {
    "Family": "FamilyOWL",
    "Pizza100": "pizza_100",
    "Pizza250": "pizza_250",
    "OWL2Bench": "OWL2Bench",
}
HOPS = ("1hop", "2hop")
EXPECTED_DATASET_TOTALS = {
    "Family": 3760,
    "Pizza100": 984,
    "Pizza250": 1232,
    "OWL2Bench": 3056,
}
EXPECTED_TOTAL = 9032

REASONING_TYPES = (
    ("D", "Direct assertions"),
    ("H", "Hierarchy"),
    ("T", "Transitivity"),
    ("S", "Symmetry"),
    ("A", "Asymmetric properties"),
    ("J", "Disjointness constraints"),
    ("N", "Property chains"),
    ("E", "Existential restrictions"),
    ("∩", "Intersection"),
    ("¬", "Complement"),
    ("I", "Inverse properties"),
    ("F", "Functional properties"),
    ("V", "Reflexive properties"),
    ("Y", "Irreflexive properties"),
    ("Q", "Equivalence"),
    ("R", "Domain/range restrictions"),
    ("C", "Cardinality restrictions"),
    ("L", "Universal restrictions"),
    ("U", "Union"),
    ("M", "Multiple TBox axiom types"),
)
KNOWN_TAGS = {tag for tag, _ in REASONING_TYPES}

ASK_RE = re.compile(
    r"ASK\s+(?:WHERE\s+)?\{\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\}",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplanationRecord:
    root_entity: str
    subject: str
    predicate: str
    object_: str
    tags: frozenset[str]
    tag_lengths: tuple[int, ...]
    explanation_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-final-json-validation",
        action="store_true",
        help="Skip the slower Task-ID comparison against final_benchmark/*.json.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def local_name(uri: str) -> str:
    if uri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
        return "rdf:type"
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def parse_ask(query: str, *, source: str) -> tuple[str, str, str]:
    match = ASK_RE.fullmatch(" ".join(str(query).split()))
    if match is None:
        raise ValueError(f"Malformed ASK query in {source}: {query!r}")
    return tuple(local_name(value) for value in match.groups())  # type: ignore[return-value]


def tags_from_explanations(
    explanations: Any, *, source: str
) -> tuple[frozenset[str], tuple[int, ...], int]:
    if not isinstance(explanations, list) or not explanations:
        raise ValueError(f"Missing explanations in {source}")

    tags: set[str] = set()
    tag_lengths: list[int] = []
    for explanation_index, explanation in enumerate(explanations):
        if not isinstance(explanation, list):
            raise ValueError(
                f"Explanation {explanation_index} is not a list in {source}"
            )
        tag_fields = [
            item[4:]
            for item in explanation
            if isinstance(item, str) and item.startswith("TAG:")
        ]
        if len(tag_fields) != 1 or not tag_fields[0]:
            raise ValueError(
                f"Expected one non-empty TAG field in explanation "
                f"{explanation_index} of {source}; found {tag_fields!r}"
            )
        unknown = set(tag_fields[0]) - KNOWN_TAGS
        if unknown:
            raise ValueError(f"Unknown reasoning tags {sorted(unknown)} in {source}")
        # Set update deliberately counts repeated characters (e.g. HH) once.
        tags.update(tag_fields[0])
        tag_lengths.append(len(tag_fields[0]))

    return frozenset(tags), tuple(tag_lengths), len(explanations)


def load_explanation_indexes(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str, str], ExplanationRecord],
    dict[tuple[str, str, str], list[ExplanationRecord]],
    dict[str, int],
]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in {path}")

    by_triple: dict[tuple[str, str, str, str], ExplanationRecord] = {}
    by_subject_predicate: dict[
        tuple[str, str, str], list[ExplanationRecord]
    ] = defaultdict(list)
    explanation_paths = 0

    for explanation_key, value in raw.items():
        source = f"{path}::{explanation_key}"
        if not isinstance(value, dict) or not isinstance(value.get("inferred"), dict):
            raise ValueError(f"Malformed explanation record in {source}")
        root_entity = str(explanation_key).split("||", 1)[0]
        inferred = value["inferred"]
        subject = str(inferred.get("subject", ""))
        predicate = str(inferred.get("predicate", ""))
        object_ = str(inferred.get("object", ""))
        if not all((root_entity, subject, predicate, object_)):
            raise ValueError(f"Incomplete inferred triple in {source}")

        tags, tag_lengths, actual_count = tags_from_explanations(
            value.get("explanations"), source=source
        )
        declared_count = int(value.get("explanationCount", -1))
        if declared_count != actual_count:
            raise ValueError(
                f"Explanation-count mismatch in {source}: "
                f"declared {declared_count}, found {actual_count}"
            )
        record = ExplanationRecord(
            root_entity,
            subject,
            predicate,
            object_,
            tags,
            tag_lengths,
            actual_count,
        )
        triple_key = (root_entity, subject, predicate, object_)
        if triple_key in by_triple:
            raise ValueError(f"Duplicate inferred triple in {path}: {triple_key}")
        by_triple[triple_key] = record
        by_subject_predicate[(root_entity, subject, predicate)].append(record)
        explanation_paths += actual_count

    return by_triple, dict(by_subject_predicate), {
        "records": len(by_triple),
        "explanation_paths": explanation_paths,
    }


def positive_binary_links(
    rows: Iterable[dict[str, str]], *, source: Path
) -> dict[tuple[str, str, str], tuple[str, str, str, str]]:
    links: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
    for row in rows:
        if row.get("Answer Type", "").upper() != "BIN":
            continue
        if row.get("Answer", "").upper() != "TRUE":
            continue
        subject, predicate, object_ = parse_ask(
            row.get("SPARQL Query", ""), source=str(source)
        )
        root = row.get("Root Entity", "")
        group_key = (root, subject, predicate)
        triple_key = (root, subject, predicate, object_)
        previous = links.get(group_key)
        if previous is not None and previous != triple_key:
            raise ValueError(
                f"Ambiguous positive BIN source for {group_key} in {source}: "
                f"{previous} versus {triple_key}"
            )
        links[group_key] = triple_key
    return links


def require_columns(rows: list[dict[str, str]], path: Path) -> None:
    required = {
        "Task ID",
        "Root Entity",
        "Task Type",
        "Answer Type",
        "SPARQL Query",
        "Answer",
        "Min Tag Length",
        "Max Tag Length",
    }
    if not rows:
        raise ValueError(f"No question rows in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def union_record_tags(records: Iterable[ExplanationRecord]) -> frozenset[str]:
    return frozenset(tag for record in records for tag in record.tags)


def validate_representation_copy(
    canonical_rows: list[dict[str, str]], alternate_path: Path
) -> dict[str, Any]:
    alternate = read_csv(alternate_path)
    canonical_by_id = {row["Task ID"]: row for row in canonical_rows}
    alternate_by_id = {row["Task ID"]: row for row in alternate}
    if len(alternate_by_id) != len(alternate):
        raise ValueError(f"Duplicate Task IDs in {alternate_path}")
    if set(canonical_by_id) != set(alternate_by_id):
        raise ValueError(f"Task-ID mismatch in representation copy {alternate_path}")
    # MC answers are entity-abstracted in the AR copy by design. Identity,
    # task grouping, and explanation-complexity metadata must remain invariant.
    compared = ("Answer Type", "Min Tag Length", "Max Tag Length")
    conflicts = [
        task_id
        for task_id in canonical_by_id
        if any(
            canonical_by_id[task_id].get(field) != alternate_by_id[task_id].get(field)
            for field in compared
        )
    ]
    if conflicts:
        raise ValueError(
            f"Metadata conflicts in {alternate_path}; first Task ID: {conflicts[0]}"
        )
    return {"path": str(alternate_path.resolve()), "rows": len(alternate)}


def collect_final_json_task_ids(value: Any, task_ids: list[str]) -> None:
    if isinstance(value, dict):
        if "Task ID" in value:
            task_ids.append(str(value["Task ID"]))
        for nested in value.values():
            collect_final_json_task_ids(nested, task_ids)
    elif isinstance(value, list):
        for nested in value:
            collect_final_json_task_ids(nested, task_ids)


def validate_final_json(path: Path, sampled_ids: set[str]) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        source = str(path.resolve())
    else:
        zip_path = path.parent / f"{path.stem.rsplit('_', 1)[0]}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Neither {path} nor its packaged ZIP {zip_path} exists"
            )
        with zipfile.ZipFile(zip_path) as archive:
            value = json.loads(archive.read(path.name).decode("utf-8"))
        source = f"{zip_path.resolve()}::{path.name}"
    task_ids: list[str] = []
    collect_final_json_task_ids(value, task_ids)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate Task IDs in {path}")
    if set(task_ids) != sampled_ids:
        raise ValueError(f"Task-ID mismatch between sampled CSV and {path}")
    return {"path": source, "task_ids": len(task_ids)}


def analyze(skip_final_json_validation: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    seen_question_keys: set[tuple[str, str, str]] = set()
    negative_links = 0
    tag_length_mismatches: list[dict[str, str]] = []

    for dataset_label, dataset_dir in DATASETS.items():
        for hop in HOPS:
            source_dir = PROJECT_ROOT / "data" / "output" / dataset_dir / hop
            sampled_path = source_dir / "SPARQL_questions_sampling.csv"
            full_path = source_dir / "SPARQL_questions.csv"
            explanations_path = source_dir / "Explanations.json"
            sampled_rows = read_csv(sampled_path)
            full_rows = read_csv(full_path)
            require_columns(sampled_rows, sampled_path)
            require_columns(full_rows, full_path)

            sampled_ids = [row["Task ID"] for row in sampled_rows]
            if any(not task_id.strip() for task_id in sampled_ids):
                raise ValueError(f"Blank Task ID in {sampled_path}")
            if len(sampled_ids) != len(set(sampled_ids)):
                raise ValueError(f"Duplicate Task IDs in {sampled_path}")

            by_triple, by_group, explanation_stats = load_explanation_indexes(
                explanations_path
            )
            positive_links = positive_binary_links(full_rows, source=full_path)

            nl_copy = validate_representation_copy(
                sampled_rows, source_dir / "SPARQL_questions_sampling_nl.csv"
            )
            abs_copy = validate_representation_copy(
                sampled_rows, source_dir / "SPARQL_questions_sampling_abs.csv"
            )
            final_json_info = None
            if not skip_final_json_validation:
                final_json_info = validate_final_json(
                    PROJECT_ROOT / "final_benchmark" / f"{dataset_dir}_{hop}.json",
                    set(sampled_ids),
                )

            file_negative_links = 0
            for row in sampled_rows:
                task_id = row["Task ID"]
                question_key = (dataset_label, hop, task_id)
                if question_key in seen_question_keys:
                    raise ValueError(f"Duplicate question-hop key: {question_key}")
                seen_question_keys.add(question_key)

                answer_type = row["Answer Type"].upper()
                answer = row["Answer"].upper()
                task_group = "BQA" if answer_type == "BIN" else "OEQA"
                root = row["Root Entity"]
                linked_positive_task_id = None

                if task_group == "BQA":
                    subject, predicate, object_ = parse_ask(
                        row["SPARQL Query"], source=f"{sampled_path}::{task_id}"
                    )
                    if answer == "TRUE":
                        explanation_key = (root, subject, predicate, object_)
                    elif answer == "FALSE":
                        group_key = (root, subject, predicate)
                        explanation_key = positive_links.get(group_key)
                        if explanation_key is None:
                            raise ValueError(
                                f"No linked positive entailment for negative {task_id}"
                            )
                        linked_positive_task_id = next(
                            full_row["Task ID"]
                            for full_row in full_rows
                            if full_row["Answer Type"].upper() == "BIN"
                            and full_row["Answer"].upper() == "TRUE"
                            and full_row["Root Entity"] == root
                            and parse_ask(
                                full_row["SPARQL Query"], source=str(full_path)
                            )
                            == explanation_key[1:]
                        )
                        negative_links += 1
                        file_negative_links += 1
                    else:
                        raise ValueError(f"Unexpected binary answer for {task_id}: {answer}")

                    record = by_triple.get(explanation_key)
                    if record is None:
                        raise ValueError(
                            f"No explanation record for {task_id}: {explanation_key}"
                        )
                    tags = record.tags
                else:
                    normalized = " ".join(row["SPARQL Query"].split())
                    select_match = re.fullmatch(
                        r"SELECT\s+\?\w+\s+(?:WHERE\s+)?\{\s*<([^>]+)>\s+"
                        r"<([^>]+)>\s+\?\w+\s*\}",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                    if select_match is None:
                        raise ValueError(f"Malformed OEQA SELECT query for {task_id}")
                    subject, predicate = (
                        local_name(value) for value in select_match.groups()
                    )
                    records = by_group.get((root, subject, predicate), [])
                    if not records:
                        raise ValueError(f"No valid explanations for OEQA {task_id}")
                    tags = union_record_tags(records)

                if not tags:
                    raise ValueError(f"No reasoning tags resolved for {task_id}")

                # The CSV stores tag-string complexity, not the number of unique types.
                # Only compare exact BQA source paths where the two quantities share a
                # source record; repeated letters remain relevant to stored complexity.
                if task_group == "BQA":
                    source_record = by_triple[explanation_key]
                    declared = (
                        int(float(row["Min Tag Length"])),
                        int(float(row["Max Tag Length"])),
                    )
                    observed = (
                        min(source_record.tag_lengths),
                        max(source_record.tag_lengths),
                    )
                    if declared != observed:
                        tag_length_mismatches.append(
                            {
                                "dataset": dataset_label,
                                "hop": hop,
                                "task_id": task_id,
                                "declared": str(declared),
                                "observed": str(observed),
                            }
                        )

                questions.append(
                    {
                        "dataset": dataset_label,
                        "hop": hop,
                        "task_id": task_id,
                        "task_group": task_group,
                        "tags": tags,
                        "min_tag_length": int(float(row["Min Tag Length"])),
                        "max_tag_length": int(float(row["Max Tag Length"])),
                        "is_negative_bqa": task_group == "BQA" and answer == "FALSE",
                        "linked_positive_task_id": linked_positive_task_id,
                    }
                )

            source_audit.append(
                {
                    "dataset": dataset_label,
                    "dataset_directory": dataset_dir,
                    "hop": hop,
                    "sampled_questions": str(sampled_path.resolve()),
                    "complete_questions_for_negative_links": str(full_path.resolve()),
                    "explanations": str(explanations_path.resolve()),
                    "sampled_rows": len(sampled_rows),
                    "full_rows": len(full_rows),
                    "negative_bqa_links": file_negative_links,
                    "explanation_records": explanation_stats["records"],
                    "explanation_paths": explanation_stats["explanation_paths"],
                    "validated_representation_copies": [nl_copy, abs_copy],
                    "validated_final_json": final_json_info,
                }
            )

    audit = {
        "unit_of_analysis": "unique (dataset, hop, Task ID)",
        "task_mapping": "Answer Type == BIN -> BQA; all observed non-BIN rows (MC) -> OEQA",
        "tag_aggregation": (
            "Set union across all TAG strings in all valid explanations; repeated "
            "letters within an explanation count once"
        ),
        "negative_bqa_handling": (
            "Resolve the unique positive BIN entailment with the same Root Entity, "
            "ASK subject, and ASK predicate in SPARQL_questions.csv, then use that "
            "positive inferred triple's complete explanation set"
        ),
        "columns_used": [
            "Task ID",
            "Root Entity",
            "Task Type",
            "Answer Type",
            "SPARQL Query",
            "Answer",
            "Min Tag Length (validation only)",
            "Max Tag Length (validation only)",
        ],
        "explanation_fields_used": [
            "inferred.subject",
            "inferred.predicate",
            "inferred.object",
            "explanations[*] TAG:<string>",
            "explanationCount (validation only)",
        ],
        "source_files": source_audit,
        "negative_bqa_questions_linked": negative_links,
        "missing_explanation_metadata": 0,
        "invalid_explanation_metadata": 0,
        "bqa_tag_length_mismatches": tag_length_mismatches,
    }
    return questions, audit


def denominator_counts(questions: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for question in questions:
        counts[(question["dataset"], question["task_group"])] += 1
        counts[(question["dataset"], "Total")] += 1
        counts[("Overall", question["task_group"])] += 1
        counts[("Overall", "Total")] += 1
    return dict(counts)


def output_columns() -> list[str]:
    columns = ["reasoning_tag", "reasoning_type"]
    for dataset in DATASETS:
        columns.extend(f"{dataset}_{group}" for group in ("BQA", "OEQA", "Total"))
    columns.extend(f"Overall_{group}" for group in ("BQA", "OEQA", "Total"))
    return columns


def make_count_rows(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag, reasoning_type in REASONING_TYPES:
        row: dict[str, Any] = {"reasoning_tag": tag, "reasoning_type": reasoning_type}
        for dataset in DATASETS:
            for group in ("BQA", "OEQA"):
                row[f"{dataset}_{group}"] = sum(
                    question["dataset"] == dataset
                    and question["task_group"] == group
                    and tag in question["tags"]
                    for question in questions
                )
            row[f"{dataset}_Total"] = (
                row[f"{dataset}_BQA"] + row[f"{dataset}_OEQA"]
            )
        for group in ("BQA", "OEQA"):
            row[f"Overall_{group}"] = sum(
                question["task_group"] == group and tag in question["tags"]
                for question in questions
            )
        row["Overall_Total"] = row["Overall_BQA"] + row["Overall_OEQA"]
        rows.append(row)
    return rows


def make_percentage_rows(
    count_rows: list[dict[str, Any]], denominators: dict[tuple[str, str], int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count_row in count_rows:
        row: dict[str, Any] = {
            "reasoning_tag": count_row["reasoning_tag"],
            "reasoning_type": count_row["reasoning_type"],
        }
        for dataset in (*DATASETS.keys(), "Overall"):
            for group in ("BQA", "OEQA", "Total"):
                column = f"{dataset}_{group}"
                row[column] = round(
                    100.0 * count_row[column] / denominators[(dataset, group)], 6
                )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns())
        writer.writeheader()
        writer.writerows(rows)


def write_reasoning_metadata(path: Path, questions: list[dict[str, Any]]) -> None:
    """Write the release-facing, one-row-per-question-hop reasoning index."""
    tag_order = {tag: index for index, (tag, _) in enumerate(REASONING_TYPES)}
    fields = [
        "dataset",
        "hop",
        "task_id",
        "task_group",
        "min_tag_length",
        "max_tag_length",
        "reasoning_tags",
        "is_negative_bqa",
        "linked_positive_task_id",
        "benchmark_version",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for question in questions:
            writer.writerow(
                {
                    **{field: question.get(field, "") for field in fields},
                    "reasoning_tags": "".join(
                        sorted(question["tags"], key=tag_order.__getitem__)
                    ),
                    "is_negative_bqa": str(question["is_negative_bqa"]).lower(),
                    "benchmark_version": "1.0.0",
                }
            )


def latex_tag(tag: str) -> str:
    return {"∩": r"$\cap$", "¬": r"$\neg$"}.get(tag, tag)


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%")


def write_combined_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pellet reasoning-type coverage in the final sampled benchmark. Counts indicate unique question--hop instances whose reasoner-generated explanations contain the corresponding reasoning type. One question may contain multiple reasoning types, so counts are not mutually exclusive.}",
        r"\label{tab:reasoning-type-coverage}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Reasoning type & Tag & Family & Pizza 100 & Pizza 250 & OWL2Bench & Total "
        + r"\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["reasoning_type"]),
                latex_tag(row["reasoning_tag"]),
                row["Family_Total"],
                row["Pizza100_Total"],
                row["Pizza250_Total"],
                row["OWL2Bench_Total"],
                row["Overall_Total"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Pellet reasoning-type coverage split by benchmark task. Counts indicate unique question--hop instances whose reasoner-generated explanations contain the corresponding reasoning type. One question may contain multiple reasoning types, so counts are not mutually exclusive.}",
        r"\label{tab:reasoning-type-coverage-by-task}",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Family} & \multicolumn{2}{c}{Pizza 100} & \multicolumn{2}{c}{Pizza 250} & \multicolumn{2}{c}{OWL2Bench} & \multicolumn{2}{c}{Total} "
        + r"\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        "Reasoning type & Tag & BQA & OEQA & BQA & OEQA & BQA & OEQA & BQA & OEQA & BQA & OEQA "
        + r"\\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            latex_escape(row["reasoning_type"]),
            latex_tag(row["reasoning_tag"]),
        ]
        for dataset in (*DATASETS.keys(), "Overall"):
            values.extend([str(row[f"{dataset}_BQA"]), str(row[f"{dataset}_OEQA"])])
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(
    count_rows: list[dict[str, Any]], denominators: dict[tuple[str, str], int]
) -> None:
    nonzero = [row for row in count_rows if row["Overall_Total"] > 0]
    zero = [row for row in count_rows if row["Overall_Total"] == 0]
    most_common = sorted(
        nonzero, key=lambda row: (-row["Overall_Total"], row["reasoning_tag"])
    )[:5]
    least_common = sorted(
        nonzero, key=lambda row: (row["Overall_Total"], row["reasoning_tag"])
    )[:5]
    print("Reasoning-type coverage summary")
    print(f"Total unique sampled question-hop instances: {denominators[('Overall', 'Total')]:,}")
    print(
        f"BQA/OEQA totals: {denominators[('Overall', 'BQA')]:,} / "
        f"{denominators[('Overall', 'OEQA')]:,}"
    )
    print(f"Reasoning types represented: {len(nonzero)} / {len(REASONING_TYPES)}")
    print(
        "Zero coverage: "
        + (", ".join(f"{row['reasoning_tag']} ({row['reasoning_type']})" for row in zero) or "none")
    )
    print(
        "Five most common: "
        + ", ".join(
            f"{row['reasoning_tag']} ({row['reasoning_type']}): {row['Overall_Total']:,}"
            for row in most_common
        )
    )
    print(
        "Five least common non-zero: "
        + ", ".join(
            f"{row['reasoning_tag']} ({row['reasoning_type']}): {row['Overall_Total']:,}"
            for row in least_common
        )
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions, audit = analyze(args.skip_final_json_validation)
    denominators = denominator_counts(questions)
    count_rows = make_count_rows(questions)
    percentage_rows = make_percentage_rows(count_rows, denominators)

    actual_dataset_totals = {
        dataset: denominators[(dataset, "Total")] for dataset in DATASETS
    }
    totals_match = (
        actual_dataset_totals == EXPECTED_DATASET_TOTALS
        and denominators[("Overall", "Total")] == EXPECTED_TOTAL
    )
    if not totals_match:
        raise ValueError(
            f"Sampled totals do not match the frozen expected totals: "
            f"{actual_dataset_totals}, overall={denominators[('Overall', 'Total')]}"
        )
    if audit["bqa_tag_length_mismatches"]:
        raise ValueError(
            f"Found {len(audit['bqa_tag_length_mismatches'])} BQA explanation-link "
            "tag-length mismatches"
        )

    audit.update(
        {
            "denominators": {
                f"{dataset}_{group}": denominators[(dataset, group)]
                for dataset in (*DATASETS.keys(), "Overall")
                for group in ("BQA", "OEQA", "Total")
            },
            "expected_dataset_totals": EXPECTED_DATASET_TOTALS,
            "expected_overall_total": EXPECTED_TOTAL,
            "expected_totals_match": totals_match,
            "reasoning_types_represented": sum(
                row["Overall_Total"] > 0 for row in count_rows
            ),
            "zero_coverage_tags": [
                row["reasoning_tag"]
                for row in count_rows
                if row["Overall_Total"] == 0
            ],
        }
    )

    write_csv(output_dir / "reasoning_coverage_counts.csv", count_rows)
    write_csv(output_dir / "reasoning_coverage_percentages.csv", percentage_rows)
    write_reasoning_metadata(
        PROJECT_ROOT / "final_benchmark" / "reasoning_metadata.csv", questions
    )
    write_combined_latex(output_dir / "reasoning_coverage_combined.tex", count_rows)
    write_split_latex(output_dir / "reasoning_coverage_by_task.tex", count_rows)
    (output_dir / "reasoning_coverage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_summary(count_rows, denominators)
    print(f"Expected benchmark totals match: yes ({EXPECTED_TOTAL:,})")
    print(f"Negative BQA questions linked to positive explanations: {audit['negative_bqa_questions_linked']:,}")
    print("Missing/invalid explanation metadata: 0 / 0")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
