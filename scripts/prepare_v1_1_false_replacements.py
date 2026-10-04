#!/usr/bin/env python3
"""Prepare deterministic replacement FALSE BQA members without API calls.

The published v1.1.0 release is read-only.  This script identifies the 67
published FALSE questions that the primary reasoner found entailed, preserves
their paired TRUE members, and emits replacement queries/prompts plus a small
reasoner batch for independent semantic validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "release" / "v1.1.0" / "benchmark" / "core_llm_bench_v1_1.parquet"
DEFAULT_AUDIT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
GENEALOGY = "http://www.example.com/genealogy.owl#"
OWL2BENCH = "https://kracr.iiitd.edu.in/OWL2Bench#"
URI_PATTERN = re.compile(r"<([^>]+)>")

sys.path.insert(0, str(ROOT / "scripts"))
from finalize_v1_1_semantic_audit import load_mapping, stable_hash  # noqa: E402
from phase6_materialize_release import abstract_question, prompt_hash  # noqa: E402
from prepare_v1_1_semantic_audit import local_name, render_query  # noqa: E402


def scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def row_hash(row: pd.Series) -> str:
    payload = {str(key): scalar(value) for key, value in row.items()}
    return stable_hash(payload)


def replace_label(question: str, old_iri: str, new_iri: str) -> str:
    old = re.sub(r"(?<!^)(?=[A-Z])", " ", local_name(old_iri)).replace("_", " ")
    new = re.sub(r"(?<!^)(?=[A-Z])", " ", local_name(new_iri)).replace("_", " ")
    corrected = str(question).replace(" (Genealogy)", "")
    replaced, count = re.subn(re.escape(old), new, corrected, flags=re.IGNORECASE)
    if count != 1:
        raise RuntimeError(f"Expected one label replacement {old!r} in {question!r}, got {count}")
    return replaced


def replacement_object(dataset: str, positive_object: str) -> tuple[str, str]:
    if dataset == "Family":
        if positive_object == GENEALOGY + "Man":
            return GENEALOGY + "Woman", "opposite disjoint peer class"
        if positive_object == GENEALOGY + "Woman":
            return GENEALOGY + "Man", "opposite disjoint peer class"
        raise RuntimeError(f"Unexpected Family positive target: {positive_object}")
    if dataset == "OWL2Bench" and positive_object == OWL2BENCH + "ComedyMovie":
        return OWL2BENCH + "Movie", "same-domain movie entity present in the exact context"
    raise RuntimeError(f"No approved deterministic replacement for {dataset}/{positive_object}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    audit = args.audit.resolve()
    benchmark = pd.read_parquet(PUBLISHED)
    by_id = benchmark.set_index(benchmark.task_id.astype(str), drop=False)
    identity = pd.read_csv(audit / "query_identity_audit.csv", dtype=str).fillna("").set_index("task_id")
    semantic = pd.read_csv(audit / "question_semantic_audit.csv", dtype=str).fillna("")
    invalid = semantic[
        (semantic.task_type == "BQA")
        & (semantic.original_gold.str.upper() == "FALSE")
        & (semantic.corrected_gold.str.upper() == "TRUE")
    ].sort_values("task_id", key=lambda values: values.astype(int))
    if len(invalid) != 67:
        raise RuntimeError(f"Expected 67 invalid FALSE BQAs, got {len(invalid)}")

    contexts = audit / "replacement_contexts"
    contexts.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reasoner_rows: list[dict[str, str]] = []
    positive_ids: set[str] = set()
    pair_ids: set[str] = set()
    for task_id in invalid.task_id:
        false_row = by_id.loc[task_id]
        positive_id = str(int(false_row.positive_task_id))
        positive = by_id.loc[positive_id]
        if positive.gold_answer != "TRUE" or false_row.pair_group_id != positive.pair_group_id:
            raise RuntimeError(f"Broken source pair for task {task_id}")
        if positive_id in positive_ids or str(false_row.pair_group_id) in pair_ids:
            raise RuntimeError(f"Duplicate replacement pair for task {task_id}")
        positive_ids.add(positive_id)
        pair_ids.add(str(false_row.pair_group_id))

        positive_terms = URI_PATTERN.findall(str(identity.loc[positive_id, "corrected_query"]))
        if len(positive_terms) != 3:
            raise RuntimeError(f"Positive task {positive_id} is not a three-IRI ASK")
        subject, predicate, positive_object = positive_terms
        target, rationale = replacement_object(str(false_row.dataset_key), positive_object)
        query = render_query("BIN", (subject, predicate, target))
        nl = replace_label(str(positive.nl_question), positive_object, target)
        uri_map, text_map = load_mapping(str(false_row.dataset_key), str(false_row.hop))
        ar = abstract_question(query, uri_map, text_map)
        if not ar:
            raise RuntimeError(f"Empty AR replacement for task {task_id}")

        context = str(false_row.fs_context)
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        context_path = contexts / f"{context_hash}.ttl"
        if not context_path.exists():
            context_path.write_text(context, encoding="utf-8", newline="\n")
        elif context_path.read_text(encoding="utf-8") != context:
            raise RuntimeError(f"Context hash collision: {context_hash}")

        rows.append({
            "task_id": task_id,
            "pair_group_id": false_row.pair_group_id,
            "positive_task_id": positive_id,
            "positive_semantic_key": positive.semantic_key,
            "positive_row_sha256": row_hash(positive),
            "dataset": false_row.dataset_key,
            "hop": false_row.hop,
            "root_entity": false_row.root_entity,
            "subject_iri": subject,
            "predicate_iri": predicate,
            "positive_object_iri": positive_object,
            "invalid_false_object_iri": identity.loc[task_id, "object_iri"],
            "replacement_object_iri": target,
            "original_false_query": false_row.formal_query,
            "replacement_query": query,
            "original_false_nl_question": false_row.nl_question,
            "replacement_nl_question": nl,
            "original_false_ar_question": false_row.ar_question,
            "replacement_ar_question": ar,
            "nl_prompt_hash": prompt_hash(nl, str(false_row.nl_context), "NL", "BIN"),
            "fs_prompt_hash": prompt_hash(query, context, "FS", "BIN"),
            "ar_prompt_hash": prompt_hash(ar, str(false_row.ar_context), "AR", "BIN"),
            "context_sha256": context_hash,
            "candidate_policy": rationale,
            "expected_gold": "FALSE",
        })
        reasoner_rows.append({
            "task_id": task_id,
            "context_path": str(context_path.resolve()),
            "answer_type": "BIN",
            "subject_iri": subject,
            "predicate_iri": predicate,
            "object_iri": target,
        })

    replacements = pd.DataFrame(rows)
    if replacements.positive_row_sha256.nunique() != 67:
        raise RuntimeError("Positive-member preservation hashes are not unique")
    replacements.to_csv(audit / "false_bqa_replacements.csv", index=False, lineterminator="\n")
    pd.DataFrame(reasoner_rows).sort_values(["context_path", "task_id"], kind="stable").to_csv(
        audit / "replacement_reasoner_input.tsv", sep="\t", index=False, lineterminator="\n"
    )
    report = {
        "status": "prepared-no-api-calls",
        "replacement_false_bqa": len(replacements),
        "preserved_distinct_positive_members": replacements.positive_task_id.nunique(),
        "preserved_distinct_pair_groups": replacements.pair_group_id.nunique(),
        "family_replacements": int((replacements.dataset == "Family").sum()),
        "owl2bench_replacements": int((replacements.dataset == "OWL2Bench").sum()),
        "replacement_prompt_hashes": 67 * 3,
        "next_step": "Run Openllet and the OWLAPI structural reasoner over replacement_reasoner_input.tsv.",
    }
    (audit / "replacement_generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
