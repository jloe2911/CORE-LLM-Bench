"""Structural BQA pairing and pre-sampling benchmark corrections.

The functions in this module deliberately do not derive BQA semantics from
``Task ID``.  A binary sampling unit is identified by dataset, hop, root
entity, and the subject/predicate parsed from its ASK query.  OEQA keeps the
legacy Task-ID grouping until the planned sequential-ID migration.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

import pandas as pd


ASK_TRIPLE_RE = re.compile(
    r"^\s*ASK\s+(?:WHERE\s*)?\{\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>"
    r"\s*\.?\s*\}\s*$",
    re.IGNORECASE,
)
SELECT_TRIPLE_RE = re.compile(
    r"^\s*SELECT\s+\?\w+\s+WHERE\s*\{\s*<([^>]+)>\s+<([^>]+)>\s+\?\w+"
    r"\s*\.?\s*\}\s*$",
    re.IGNORECASE,
)
LEGACY_TASK_GROUP_RE = re.compile(r"(-BIN-.+-NEG-BIN|-(BIN|MC))$")
ELIGIBLE_TASK_TYPES = ("Membership", "Property Assertion")


@dataclass(frozen=True)
class DomainConceptFilterReport:
    dataset: str
    hop: str
    domainconcept_bqa_targets_removed: int
    domainconcept_positives_replaced: int
    domainconcept_only_groups_omitted: int
    oeqa_domainconcept_answers_retained: int


@dataclass(frozen=True)
class PairingAuditRow:
    dataset: str
    hop: str
    task_type: str
    true_bqa_count: int
    false_bqa_count: int
    paired_groups: int
    unpaired_positives: int
    unpaired_negatives: int
    missing_source_pair_positives: int
    missing_source_pair_negatives: int
    broken_sample_positives: int
    broken_sample_negatives: int


def canonical_dataset_name(dataset: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(dataset).lower())
    return {
        "pizza100": "Pizza100",
        "pizza250": "Pizza250",
        "owl2bench": "OWL2Bench",
        "family": "Family",
        "familyowl": "Family",
    }.get(normalized, str(dataset))


def parse_ask_triple(query: str) -> tuple[str, str, str]:
    """Return the subject, predicate, and object from a single-triple ASK."""

    match = ASK_TRIPLE_RE.fullmatch(str(query))
    if match is None:
        raise ValueError(f"Unsupported BQA ASK query: {query!r}")
    return match.group(1), match.group(2), match.group(3)


def parse_select_subject_predicate(query: str) -> tuple[str, str]:
    """Return the subject and predicate from a generated OEQA SELECT."""

    match = SELECT_TRIPLE_RE.fullmatch(str(query))
    if match is None:
        raise ValueError(f"Unsupported OEQA SELECT query: {query!r}")
    return match.group(1), match.group(2)


def uri_local_name(uri: str) -> str:
    return re.split(r"[#/:]", str(uri))[-1]


def legacy_oeqa_task_group(task_id: str) -> str:
    """Preserve the pre-v1.1 Task-ID grouping behavior for OEQA only."""

    return LEGACY_TASK_GROUP_RE.sub("", str(task_id))


def sampling_group_key(
    row: pd.Series | dict[str, object], dataset: str, hop: str
) -> tuple[Hashable, ...]:
    if str(row["Answer Type"]).upper() == "BIN":
        subject, predicate, _ = parse_ask_triple(str(row["SPARQL Query"]))
        return (
            "BQA",
            canonical_dataset_name(dataset),
            str(hop),
            str(row["Root Entity"]),
            subject,
            predicate,
        )
    return ("OEQA", legacy_oeqa_task_group(str(row["Task ID"])))


def _answer_set_contains_domainconcept(answer: object) -> bool:
    return any(
        uri_local_name(value.strip()) == "DomainConcept"
        for value in str(answer).split(";")
        if value.strip()
    )


def _bqa_object(row: pd.Series | dict[str, object]) -> str:
    return parse_ask_triple(str(row["SPARQL Query"]))[2]


def _representative_order(object_uri: str) -> tuple[int, str]:
    """Mirror the generator's rdf:type representative ordering."""

    local_name = uri_local_name(object_uri)
    generic_priority = int(
        local_name in {"Thing", "NamedIndividual", "DomainEntity", "Person"}
    )
    return generic_priority, object_uri


def _structural_group_key(
    row: pd.Series | dict[str, object], dataset: str, hop: str
) -> tuple[Hashable, ...]:
    if str(row["Answer Type"]).upper() == "BIN":
        subject, predicate, _ = parse_ask_triple(str(row["SPARQL Query"]))
    else:
        subject, predicate = parse_select_subject_predicate(
            str(row["SPARQL Query"])
        )
    return (
        "BQA",
        canonical_dataset_name(dataset),
        str(hop),
        str(row["Root Entity"]),
        subject,
        predicate,
    )


def prepare_eligible_rows(
    df: pd.DataFrame, dataset: str, hop: str
) -> tuple[pd.DataFrame, DomainConceptFilterReport]:
    """Apply task eligibility and the narrow Pizza DomainConcept BQA rule."""

    eligible = df[df["Task Type"].isin(ELIGIBLE_TASK_TYPES)].copy()
    is_pizza = canonical_dataset_name(dataset) in {"Pizza100", "Pizza250"}
    excluded_indexes: set[object] = set()
    replacement_groups: set[tuple[Hashable, ...]] = set()
    omitted_groups: set[tuple[Hashable, ...]] = set()

    if is_pizza:
        candidates = eligible[
            (eligible["Answer Type"].astype(str).str.upper() == "BIN")
            & (eligible["Task Type"] == "Membership")
        ].copy()
        candidates["_group_key"] = [
            sampling_group_key(row, dataset, hop)
            for _, row in candidates.iterrows()
        ]

        for group_key, group in candidates.groupby("_group_key", sort=False):
            true_rows = group[
                group["Answer"].astype(str).str.upper() == "TRUE"
            ]
            domain_true = true_rows[
                true_rows.apply(
                    lambda row: uri_local_name(_bqa_object(row))
                    == "DomainConcept",
                    axis=1,
                )
            ]
            non_domain_true = true_rows.drop(index=domain_true.index)

            domain_targets = group[
                group.apply(
                    lambda row: uri_local_name(_bqa_object(row))
                    == "DomainConcept",
                    axis=1,
                )
            ]
            excluded_indexes.update(domain_targets.index)

            if domain_true.empty:
                continue

            if non_domain_true.empty:
                # Remove the entire atomic BQA group, including the negative
                # originally paired with the excluded DomainConcept positive.
                excluded_indexes.update(group.index)
                omitted_groups.add(group_key)
                continue

            selected_index = min(
                non_domain_true.index,
                key=lambda index: _representative_order(
                    _bqa_object(non_domain_true.loc[index])
                ),
            )
            excluded_indexes.update(
                index for index in non_domain_true.index if index != selected_index
            )
            original_index = min(
                true_rows.index,
                key=lambda index: _representative_order(
                    _bqa_object(true_rows.loc[index])
                ),
            )
            if original_index in domain_true.index:
                replacement_groups.add(group_key)

        # OEQA contains the complete entailed answer set and therefore keeps
        # the policy audit informative even after the generator has already
        # replaced or omitted the corresponding BQA representative.
        oeqa_membership = eligible[
            (eligible["Answer Type"].astype(str).str.upper() != "BIN")
            & (eligible["Task Type"] == "Membership")
        ]
        for _, row in oeqa_membership.iterrows():
            answers = [
                value.strip()
                for value in str(row["Answer"]).split(";")
                if value.strip()
            ]
            if not any(uri_local_name(value) == "DomainConcept" for value in answers):
                continue
            group_key = _structural_group_key(row, dataset, hop)
            non_domain_answers = [
                value
                for value in answers
                if uri_local_name(value) != "DomainConcept"
            ]
            if not non_domain_answers:
                omitted_groups.add(group_key)
                replacement_groups.discard(group_key)
                continue
            original = min(answers, key=_representative_order)
            if uri_local_name(original) == "DomainConcept":
                replacement_groups.add(group_key)

    oeqa_retained = sum(
        str(row["Answer Type"]).upper() != "BIN"
        and _answer_set_contains_domainconcept(row["Answer"])
        for _, row in eligible.iterrows()
    )
    if excluded_indexes:
        eligible = eligible.drop(index=excluded_indexes)

    report = DomainConceptFilterReport(
        dataset=canonical_dataset_name(dataset),
        hop=str(hop),
        domainconcept_bqa_targets_removed=sum(
            uri_local_name(_bqa_object(row)) == "DomainConcept"
            for _, row in candidates.iterrows()
            if row.name in excluded_indexes
        )
        if is_pizza
        else 0,
        domainconcept_positives_replaced=len(replacement_groups),
        domainconcept_only_groups_omitted=len(omitted_groups),
        oeqa_domainconcept_answers_retained=oeqa_retained,
    )
    return eligible, report


def add_group_keys(df: pd.DataFrame, dataset: str, hop: str) -> pd.DataFrame:
    keyed = df.copy()
    keyed["Sampling Group Key"] = [
        sampling_group_key(row, dataset, hop) for _, row in keyed.iterrows()
    ]
    return keyed


def _binary_group_answers(
    df: pd.DataFrame, dataset: str, hop: str, task_type: str
) -> dict[tuple[Hashable, ...], set[str]]:
    rows = df[
        (df["Answer Type"].astype(str).str.upper() == "BIN")
        & (df["Task Type"] == task_type)
    ]
    answers: dict[tuple[Hashable, ...], set[str]] = {}
    for _, row in rows.iterrows():
        answer = str(row["Answer"]).upper()
        if answer not in {"TRUE", "FALSE"}:
            raise ValueError(f"Unexpected BQA answer {answer!r}")
        key = sampling_group_key(row, dataset, hop)
        answers.setdefault(key, set()).add(answer)
    return answers


def audit_bqa_pairing(
    eligible_source: pd.DataFrame,
    sampled: pd.DataFrame,
    dataset: str,
    hop: str,
) -> list[PairingAuditRow]:
    """Audit sampled BQA pairs and classify source absence versus breakage."""

    report: list[PairingAuditRow] = []
    task_types = sorted(
        set(eligible_source.get("Task Type", pd.Series(dtype=str)).astype(str))
        | set(sampled.get("Task Type", pd.Series(dtype=str)).astype(str))
    )
    for task_type in task_types:
        source_groups = _binary_group_answers(
            eligible_source, dataset, hop, task_type
        )
        sample_groups = _binary_group_answers(sampled, dataset, hop, task_type)
        true_count = int(
            (
                (sampled["Answer Type"].astype(str).str.upper() == "BIN")
                & (sampled["Task Type"] == task_type)
                & (sampled["Answer"].astype(str).str.upper() == "TRUE")
            ).sum()
        )
        false_count = int(
            (
                (sampled["Answer Type"].astype(str).str.upper() == "BIN")
                & (sampled["Task Type"] == task_type)
                & (sampled["Answer"].astype(str).str.upper() == "FALSE")
            ).sum()
        )
        paired = sum(values >= {"TRUE", "FALSE"} for values in sample_groups.values())
        positive_only = {
            key for key, values in sample_groups.items() if values == {"TRUE"}
        }
        negative_only = {
            key for key, values in sample_groups.items() if values == {"FALSE"}
        }
        broken_positive = sum(
            source_groups.get(key, set()) >= {"TRUE", "FALSE"}
            for key in positive_only
        )
        broken_negative = sum(
            source_groups.get(key, set()) >= {"TRUE", "FALSE"}
            for key in negative_only
        )
        report.append(
            PairingAuditRow(
                dataset=canonical_dataset_name(dataset),
                hop=str(hop),
                task_type=task_type,
                true_bqa_count=true_count,
                false_bqa_count=false_count,
                paired_groups=paired,
                unpaired_positives=len(positive_only),
                unpaired_negatives=len(negative_only),
                missing_source_pair_positives=len(positive_only) - broken_positive,
                missing_source_pair_negatives=len(negative_only) - broken_negative,
                broken_sample_positives=broken_positive,
                broken_sample_negatives=broken_negative,
            )
        )
    return report


def assert_no_broken_source_pairs(report: Iterable[PairingAuditRow]) -> None:
    broken = [
        row
        for row in report
        if row.broken_sample_positives or row.broken_sample_negatives
    ]
    if broken:
        details = "; ".join(
            f"{row.dataset}/{row.hop}/{row.task_type}: "
            f"positive-only={row.broken_sample_positives}, "
            f"negative-only={row.broken_sample_negatives}"
            for row in broken
        )
        raise ValueError(f"Sampling broke valid source BQA pairs: {details}")


def assert_owl2bench_membership_pairs(report: Iterable[PairingAuditRow]) -> None:
    membership = next(
        (
            row
            for row in report
            if row.dataset == "OWL2Bench" and row.task_type == "Membership"
        ),
        None,
    )
    if membership is None:
        raise ValueError("OWL2Bench sampling audit has no Membership row")
    if membership.false_bqa_count == 0 or membership.paired_groups == 0:
        raise ValueError(
            "Corrected sampled OWL2Bench Membership must contain FALSE questions "
            "and paired TRUE/FALSE groups"
        )


def print_domainconcept_report(report: DomainConceptFilterReport) -> None:
    print(
        "DomainConcept filter: "
        f"dataset={report.dataset}, hop={report.hop}, "
        "DomainConcept BQA targets removed="
        f"{report.domainconcept_bqa_targets_removed}, "
        "DomainConcept positives replaced="
        f"{report.domainconcept_positives_replaced}, "
        "DomainConcept-only groups omitted="
        f"{report.domainconcept_only_groups_omitted}, "
        "OEQA DomainConcept answer rows retained="
        f"{report.oeqa_domainconcept_answers_retained}"
    )


def print_pairing_report(report: Iterable[PairingAuditRow]) -> None:
    print("BQA pairing audit:")
    for row in report:
        print(
            f"  {row.dataset}/{row.hop}/{row.task_type}: "
            f"TRUE={row.true_bqa_count}, FALSE={row.false_bqa_count}, "
            f"paired={row.paired_groups}, "
            f"unpaired_positive={row.unpaired_positives}, "
            f"unpaired_negative={row.unpaired_negatives}, "
            "missing_source_pair_positive="
            f"{row.missing_source_pair_positives}, "
            "missing_source_pair_negative="
            f"{row.missing_source_pair_negatives}, "
            f"broken_sample_positive={row.broken_sample_positives}, "
            f"broken_sample_negative={row.broken_sample_negatives}"
        )


def validate_files(
    source_file: str | Path,
    sampled_file: str | Path,
    dataset: str,
    hop: str,
    require_owl2bench_membership: bool = False,
) -> list[PairingAuditRow]:
    source = pd.read_csv(source_file)
    sampled = pd.read_csv(sampled_file)
    eligible_source, filter_report = prepare_eligible_rows(source, dataset, hop)
    sampled_eligible, sampled_filter_report = prepare_eligible_rows(
        sampled, dataset, hop
    )
    if sampled_filter_report.domainconcept_bqa_targets_removed:
        raise ValueError("Sampled benchmark contains excluded DomainConcept BQA rows")
    print_domainconcept_report(filter_report)
    report = audit_bqa_pairing(eligible_source, sampled_eligible, dataset, hop)
    print_pairing_report(report)
    assert_no_broken_source_pairs(report)
    if require_owl2bench_membership or canonical_dataset_name(dataset) == "OWL2Bench":
        assert_owl2bench_membership_pairs(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit structural BQA pairing in a sampled benchmark CSV."
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--sampled-file", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hop", required=True, choices=("1hop", "2hop"))
    parser.add_argument("--require-owl2bench-membership", action="store_true")
    args = parser.parse_args()
    validate_files(
        args.source_file,
        args.sampled_file,
        args.dataset,
        args.hop,
        args.require_owl2bench_membership,
    )


if __name__ == "__main__":
    main()
