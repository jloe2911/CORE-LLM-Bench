#!/usr/bin/env python3
"""Build the isolated v1.1 symbolic source-pool audit from regenerated proofs.

This script is deliberately offline.  It reads frozen subgraphs, regenerated
symbolic outputs, legacy source-pool CSVs, and source ontologies.  It never
samples the final benchmark and never reads or writes model predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable

import pandas as pd
from rdflib import Graph, OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.llm_pipeline.benchmark_corrections import (  # noqa: E402
    canonical_dataset_name,
    parse_ask_triple,
    prepare_eligible_rows,
    sampling_group_key,
)
from scripts.oeqa_explanations import (  # noqa: E402
    _construct_tags,
    explanation_axioms,
    explanation_tag,
)


STAGE = ROOT / "data" / "output_v1_1_symbolic"
TAGS = ("D", "H", "T", "S", "A", "J", "N", "E", "∩", "¬", "I", "F", "V", "Y", "Q", "R", "C", "L", "U", "M")
PRIMITIVE_TAGS = TAGS[:-1]
DATASETS = {
    "Family": ("FamilyOWL", "family.owl", "FamilyOWL_{hop}"),
    "Pizza100": ("pizza_100", "pizza_100.owl", "pizza_100_{hop}"),
    "Pizza250": ("pizza_250", "pizza_250.owl", "pizza_250_{hop}"),
    "OWL2Bench": (
        "OWL2Bench",
        "OWL2DL-1.owl",
        "OWL2Bench_{hop}_paired_sampled_500_seed_13",
    ),
}
NESTED_TAGS = {"E", "∩", "¬", "C", "L", "U"}


def normalize_query(value: object) -> str:
    return " ".join(str(value).split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def root_timing_audit(log_text: str) -> list[dict[str, Any]]:
    events: list[tuple[datetime, str]] = []
    pattern = re.compile(
        r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+[^ ]*) .*Processing file \d+/\d+: (.+)$"
    )
    completed = re.compile(
        r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+[^ ]*) .*Sequential processing completed"
    )
    end_time: datetime | None = None
    for line in log_text.splitlines():
        match = pattern.match(line)
        if match:
            events.append((datetime.fromisoformat(match.group(1)), match.group(2)))
        match = completed.match(line)
        if match:
            end_time = datetime.fromisoformat(match.group(1))
    stalls = []
    for index, (started, root) in enumerate(events):
        ended = events[index + 1][0] if index + 1 < len(events) else end_time
        if ended is None:
            continue
        seconds = (ended - started).total_seconds()
        if seconds >= 60:
            stalls.append({"root_file": root, "seconds": round(seconds, 3)})
    return stalls


def read_explanations(path: Path) -> dict[str, Any]:
    from scripts.explanations_fix import fix_explanations_json

    return json.loads(fix_explanations_json(path.read_text(encoding="utf-8")))


def proof_from_structured(raw: dict[str, Any]) -> dict[str, Any]:
    axioms: dict[str, tuple[str, ...]] = {}
    for item in raw.get("semanticAxioms", []):
        identity = str(item["identity"])
        tags = tuple(tag for tag in item.get("primitiveTags", []) if tag != "M")
        if identity in axioms and axioms[identity] != tags:
            raise ValueError(f"Conflicting tags for semantic axiom {identity}")
        axioms[identity] = tags
    primitive = tuple(tag for tags in axioms.values() for tag in tags)
    distinct = tuple(tag for tag in PRIMITIVE_TAGS if tag in primitive)
    non_direct = set(primitive) - {"D"}
    expected_m = len(non_direct) >= 2
    if bool(raw.get("m")) != expected_m:
        raise ValueError("Structured M metadata disagrees with heterogeneity definition")
    if int(raw.get("axiomCount", -1)) != len(axioms):
        raise ValueError("Structured axiom count disagrees with semantic identities")
    if int(raw.get("primitiveTagCount", -1)) != len(primitive):
        raise ValueError("Structured primitive count disagrees with per-axiom tags")
    if Counter(raw.get("primitiveTags", [])) != Counter(primitive):
        raise ValueError("Structured primitive tags disagree with per-axiom tags")
    if set(raw.get("distinctPrimitiveTags", [])) != set(distinct):
        raise ValueError("Structured distinct primitive tags disagree with per-axiom tags")
    return {
        "axioms": axioms,
        "axiom_count": len(axioms),
        "primitive_tags": primitive,
        "primitive_tag_count": len(primitive),
        "distinct_primitive_tags": distinct,
        "m": expected_m,
    }


def record_proofs(record: dict[str, Any]) -> list[dict[str, Any]]:
    structured = record.get("structuredExplanations")
    if structured is None:
        raise ValueError("Regenerated explanation lacks structuredExplanations")
    return [proof_from_structured(item) for item in structured]


def summarize_proofs(proofs: list[dict[str, Any]]) -> dict[str, Any]:
    if not proofs:
        raise ValueError("Eligible inference has no proof alternatives")
    primitive_sets = [set(proof["distinct_primitive_tags"]) for proof in proofs]
    m_values = [proof["m"] for proof in proofs]
    return {
        "minimum_axiom_count": min(p["axiom_count"] for p in proofs),
        "maximum_axiom_count": max(p["axiom_count"] for p in proofs),
        "minimum_primitive_tag_count": min(p["primitive_tag_count"] for p in proofs),
        "maximum_primitive_tag_count": max(p["primitive_tag_count"] for p in proofs),
        "distinct_primitive_reasoning_types": [
            tag for tag in PRIMITIVE_TAGS if any(tag in tags for tags in primitive_sets)
        ],
        "m_status": "always" if all(m_values) else "never" if not any(m_values) else "selection-dependent",
        "proof_count": len(proofs),
    }


def complete_oeqa_metrics(answer_groups: list[dict[str, Any]]) -> dict[str, Any]:
    # Fold equivalent semantic unions after each answer to avoid materializing
    # a potentially large raw Cartesian product.
    states: dict[frozenset[str], dict[str, tuple[str, ...]]] = {frozenset(): {}}
    combination_count = 1
    for group in answer_groups:
        alternatives = group["proofs"]
        combination_count *= len(alternatives)
        next_states: dict[frozenset[str], dict[str, tuple[str, ...]]] = {}
        for current in states.values():
            for proof in alternatives:
                union = dict(current)
                for identity, tags in proof["axioms"].items():
                    if identity in union and union[identity] != tags:
                        raise ValueError(f"Conflicting union provenance for {identity}")
                    union[identity] = tags
                next_states[frozenset(union)] = union
        states = next_states
    metrics = []
    for axioms in states.values():
        primitive = tuple(tag for tags in axioms.values() for tag in tags)
        distinct = set(primitive)
        metrics.append(
            {
                "axioms": len(axioms),
                "primitive": len(primitive),
                "distinct": len(distinct),
                "m": len(distinct - {"D"}) >= 2,
                "tags": distinct,
            }
        )
    m_values = [metric["m"] for metric in metrics]
    return {
        "complete_combination_count": combination_count,
        "unique_semantic_union_count": len(states),
        "complete_min_axiom_count": min(m["axioms"] for m in metrics),
        "complete_max_axiom_count": max(m["axioms"] for m in metrics),
        "complete_min_primitive_tag_count": min(m["primitive"] for m in metrics),
        "complete_max_primitive_tag_count": max(m["primitive"] for m in metrics),
        "complete_min_distinct_primitive_type_count": min(m["distinct"] for m in metrics),
        "complete_max_distinct_primitive_type_count": max(m["distinct"] for m in metrics),
        "m_status": "always" if all(m_values) else "never" if not any(m_values) else "selection-dependent",
        "distinct_primitive_reasoning_types": [
            tag for tag in PRIMITIVE_TAGS if any(tag in m["tags"] for m in metrics)
        ],
    }


def explanation_indexes(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    asks: dict[str, Any] = {}
    selects: dict[str, list[Any]] = defaultdict(list)
    select_seen: set[tuple[str, str]] = set()

    def add_select(query: str, wrapped: dict[str, Any]) -> None:
        identity = (query, wrapped["source_key"])
        if identity not in select_seen:
            select_seen.add(identity)
            selects[query].append(wrapped)

    for key, record in data.items():
        wrapped = {
            "source_key": key,
            "source_answer": str(record.get("inferred", {}).get("object", "")),
            "proofs": record_proofs(record),
        }
        for query in record.get("sparqlQueries", []):
            normalized = normalize_query(query)
            if normalized.upper().startswith("ASK"):
                asks[normalized] = wrapped
                subject, predicate, _ = parse_ask_triple(normalized)
                add_select(
                    normalize_query(
                        f"SELECT ?x WHERE {{ <{subject}> <{predicate}> ?x }}"
                    ),
                    wrapped,
                )
            elif normalized.upper().startswith("SELECT"):
                add_select(normalized, wrapped)
    return asks, selects


def old_complexity_index(path: Path, dataset: str, hop: str) -> tuple[dict[str, int], dict[tuple[Any, ...], int]]:
    old = pd.read_csv(path)
    exact: dict[str, int] = {}
    structural: dict[tuple[Any, ...], int] = {}
    for _, row in old.iterrows():
        value = int(row["Max Tag Length"])
        exact[normalize_query(row["SPARQL Query"])] = value
        if str(row["Answer Type"]).upper() == "BIN":
            structural[(sampling_group_key(row, dataset, hop), str(row["Answer"]).upper())] = value
    return exact, structural


def explanation_old_tags(path: Path) -> dict[str, dict[str, Counter[str]]]:
    result: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"tags": Counter(), "duplicate_surplus": Counter()}
    )
    for record in read_explanations(path).values():
        query_keys: set[str] = set()
        for query in record.get("sparqlQueries", []):
            query_key = normalize_query(query)
            query_keys.add(query_key)
            if query_key.upper().startswith("ASK"):
                subject, predicate, _ = parse_ask_triple(query_key)
                query_keys.add(
                    normalize_query(
                        f"SELECT ?x WHERE {{ <{subject}> <{predicate}> ?x }}"
                    )
                )
        for query_key in query_keys:
            for proof in record.get("explanations", []):
                stored = Counter(explanation_tag(proof))
                expected: Counter[str] = Counter()
                for axiom in explanation_axioms(proof):
                    tags = _construct_tags(axiom) or ("D",)
                    expected.update(tags)
                if len(set(expected) - {"D"}) >= 2:
                    expected["M"] += 1
                result[query_key]["tags"] |= stored
                result[query_key]["duplicate_surplus"].update(
                    {tag: count for tag, count in (stored - expected).items() if count > 0}
                )
    return dict(result)


def build_rows(dataset: str, hop: str, stage_dir: Path, legacy_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    staged_questions = pd.read_csv(stage_dir / "SPARQL_questions.csv")
    eligible, filter_report = prepare_eligible_rows(staged_questions, dataset, hop)
    explanations = read_explanations(stage_dir / "Explanations.json")
    asks, selects = explanation_indexes(explanations)
    old_exact, old_structural = old_complexity_index(
        legacy_dir / "SPARQL_questions.csv", dataset, hop
    )
    old_tags = explanation_old_tags(legacy_dir / "Explanations.json")
    positives: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for _, source in eligible.iterrows():
        if str(source["Answer Type"]).upper() != "BIN" or str(source["Answer"]).upper() != "TRUE":
            continue
        query = normalize_query(source["SPARQL Query"])
        record = asks.get(query)
        if record is None:
            raise ValueError(f"No exact positive explanation for {dataset}/{hop}: {query}")
        positives[sampling_group_key(source, dataset, hop)] = {
            "record": record,
            "summary": summarize_proofs(record["proofs"]),
            "task_id": str(source["Task ID"]),
            "query": query,
            "stable_old_complexity": old_exact.get(query),
        }

    failures: list[dict[str, str]] = []
    for _, source in eligible.iterrows():
        task = "BQA" if str(source["Answer Type"]).upper() == "BIN" else "OEQA"
        query = normalize_query(source["SPARQL Query"])
        answer = str(source["Answer"]).upper()
        record: dict[str, Any]
        provenance: dict[str, Any]
        if task == "BQA":
            group_key = sampling_group_key(source, dataset, hop)
            positive = positives.get(group_key)
            if positive is None:
                failures.append({"task_id": str(source["Task ID"]), "error": "no linked positive entailment"})
                continue
            record = dict(positive["summary"])
            provenance = {
                "positive_task_id": positive["task_id"],
                "positive_query": positive["query"],
                "source_explanation_key": positive["record"]["source_key"],
                "inherited_by_negative": answer == "FALSE",
            }
            primary = record["minimum_primitive_tag_count"]
            # A BQA pair is mapped through its exact positive entailment.  A
            # replaced Pizza DomainConcept positive is therefore intentionally
            # unmapped even if its structural group is unchanged.
            stable_old = positive["stable_old_complexity"]
            legacy_evidence = old_tags.get(
                positive["query"], {"tags": Counter(), "duplicate_surplus": Counter()}
            )
        else:
            source_records = selects.get(query, [])
            by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in source_records:
                by_answer[item["source_answer"]].extend(item["proofs"])
            gold = [value.strip() for value in str(source["Answer"]).split(";") if value.strip()]
            missing = [value for value in gold if not by_answer.get(value)]
            if missing:
                failures.append({"task_id": str(source["Task ID"]), "error": f"missing exact answers: {missing}"})
                continue
            answer_groups = [{"answer": value, "proofs": by_answer[value]} for value in gold]
            record = complete_oeqa_metrics(answer_groups)
            record["answer_count"] = len(gold)
            provenance = {
                "answers": [
                    {
                        "answer": group["answer"],
                        "source": "inferred.object",
                        "proof_count": len(group["proofs"]),
                    }
                    for group in answer_groups
                ]
            }
            primary = record["complete_min_primitive_tag_count"]
            stable_old = old_exact.get(query)
            legacy_evidence = old_tags.get(
                query, {"tags": Counter(), "duplicate_surplus": Counter()}
            )

        corrected_tags = set(record["distinct_primitive_reasoning_types"])
        legacy_tag_counter = legacy_evidence["tags"]
        rows.append(
            {
                "dataset": canonical_dataset_name(dataset),
                "dataset_dir": DATASETS[canonical_dataset_name(dataset)][0],
                "hop": hop,
                "task_type": task,
                "task_id": str(source["Task ID"]),
                "root_entity": str(source["Root Entity"]),
                "sparql_query": query,
                "answer": answer if task == "BQA" else str(source["Answer"]),
                "primary_complexity": primary,
                "old_v1_complexity": stable_old,
                "old_vs_corrected": None if stable_old is None else ("decreased" if primary < stable_old else "increased" if primary > stable_old else "unchanged"),
                "attribution_evidence": {
                    "legacy_m_present": legacy_tag_counter["M"] > 0,
                    "legacy_duplicate_semantic_tag_surplus": bool(legacy_evidence["duplicate_surplus"]),
                    "new_nested_tag": bool(
                        (corrected_tags & NESTED_TAGS) - set(legacy_tag_counter)
                    ),
                    "complete_answer_conjunction": task == "OEQA" and record.get("answer_count", 0) > 1,
                },
                "complexity": record,
                "provenance": provenance,
            }
        )
    bqa_groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"answers": set(), "complexities": set()}
    )
    bqa_key_by_task: dict[str, tuple[Any, ...]] = {}
    for _, source in eligible.iterrows():
        if str(source["Answer Type"]).upper() != "BIN":
            continue
        key = sampling_group_key(source, dataset, hop)
        bqa_key_by_task[str(source["Task ID"])] = key
        bqa_groups[key]["answers"].add(str(source["Answer"]).upper())
    for row in rows:
        if row["task_type"] != "BQA":
            continue
        key = bqa_key_by_task[row["task_id"]]
        bqa_groups[key]["complexities"].add(row["primary_complexity"])
    shared_metadata_violations = sum(
        len(group["complexities"]) > 1 for group in bqa_groups.values()
    )
    return rows, {
        "eligible_rows": len(eligible),
        "completed_rows": len(rows),
        "failures": failures,
        "domainconcept_filter": filter_report.__dict__,
        "bqa_pairing": {
            "groups": len(bqa_groups),
            "paired_true_false": sum(group["answers"] >= {"TRUE", "FALSE"} for group in bqa_groups.values()),
            "positive_only": sum(group["answers"] == {"TRUE"} for group in bqa_groups.values()),
            "negative_only": sum(group["answers"] == {"FALSE"} for group in bqa_groups.values()),
            "shared_complexity_metadata_violations": shared_metadata_violations,
        },
    }


def percentile(values: list[int], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[int]) -> dict[str, Any]:
    data = list(values)
    frequencies = Counter(data)
    return {
        "count": len(data),
        "minimum": min(data, default=0),
        "maximum": max(data, default=0),
        "mean": round(statistics.fmean(data), 6) if data else 0,
        "median": statistics.median(data) if data else 0,
        "standard_deviation": round(statistics.pstdev(data), 6) if data else 0,
        "p25": percentile(data, 25),
        "p50": percentile(data, 50),
        "p75": percentile(data, 75),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "frequency": {str(key): frequencies[key] for key in sorted(frequencies)},
    }


def group_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dimensions = (
        ("overall", lambda r: "overall"),
        ("dataset_task", lambda r: f"{r['dataset']}/{r['task_type']}"),
        ("hop", lambda r: r["hop"]),
        ("dataset_task_hop", lambda r: f"{r['dataset']}/{r['task_type']}/{r['hop']}"),
    )
    for name, key_fn in dimensions:
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row["primary_complexity"])
        result[name] = {key: distribution(values) for key, values in sorted(grouped.items())}
    return result


def classify_fixed(value: int, low_max: int, medium_max: int) -> str:
    return "Low" if value <= low_max else "Medium" if value <= medium_max else "High"


def bin_report(rows: list[dict[str, Any]], classifier) -> dict[str, Any]:
    splits = {
        "overall": lambda r: "overall",
        "dataset": lambda r: r["dataset"],
        "task_type": lambda r: r["task_type"],
        "hop": lambda r: r["hop"],
        "dataset_task_hop": lambda r: f"{r['dataset']}/{r['task_type']}/{r['hop']}",
    }
    report: dict[str, Any] = {}
    for split, key_fn in splits.items():
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            grouped[key_fn(row)][classifier(row)] += 1
        report[split] = {}
        for key, counts in sorted(grouped.items()):
            total = sum(counts.values())
            values = {
                label: {
                    "count": counts[label],
                    "percentage": round(100 * counts[label] / total, 4),
                }
                for label in ("Low", "Medium", "High")
            }
            values["extremely_sparse_bins"] = [
                label for label in ("Low", "Medium", "High")
                if counts[label] == 0 or counts[label] / total < 0.05
            ]
            report[split][key] = values
    return report


def candidate_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["primary_complexity"] for row in rows]
    p33 = percentile(values, 100 / 3)
    p67 = percentile(values, 200 / 3)
    low_cut = max(1, math.floor(p33))
    medium_cut = max(low_cut + 1, math.floor(p67))
    sorted_rows = sorted(rows, key=lambda r: (r["primary_complexity"], r["dataset"], r["hop"], r["task_id"]))
    quantile_labels: dict[tuple[str, str, str], str] = {}
    for index, row in enumerate(sorted_rows):
        label = "Low" if index < len(rows) / 3 else "Medium" if index < 2 * len(rows) / 3 else "High"
        quantile_labels[(row["dataset"], row["hop"], row["task_id"])] = label
    schemes = {
        "legacy": {
            "definition": "Low=1; Medium=2-3; High=4+",
            "comparability": "Directly comparable with published v1 bins, but uses corrected primary complexity.",
            "counts": bin_report(rows, lambda r: classify_fixed(r["primary_complexity"], 1, 3)),
        },
        "distribution_aware_fixed": {
            "definition": f"Low<={low_cut}; Medium={low_cut + 1}-{medium_cut}; High={medium_cut + 1}+",
            "derivation": {"pooled_p33": p33, "pooled_p67": p67},
            "comparability": "Integer thresholds are data-derived and are not directly comparable with published v1 bins.",
            "counts": bin_report(rows, lambda r: classify_fixed(r["primary_complexity"], low_cut, medium_cut)),
        },
        "quantile_rank": {
            "definition": "Three equal-count pooled rank bins with deterministic identity tie-breaking.",
            "comparability": "Balanced pooled sizes, but equal complexity values may cross bins and v1 comparability is weak.",
            "counts": bin_report(rows, lambda r: quantile_labels[(r["dataset"], r["hop"], r["task_id"])]),
        },
    }
    return schemes


def tbox_census(path: Path) -> dict[str, int]:
    graph = Graph()
    graph.parse(path)
    counts = Counter()
    counts["D"] = sum(1 for _ in graph.triples((None, RDF.type, None)))
    counts["H"] = sum(1 for _ in graph.triples((None, RDFS.subClassOf, None))) + sum(1 for _ in graph.triples((None, RDFS.subPropertyOf, None)))
    counts["J"] = sum(1 for _ in graph.triples((None, OWL.disjointWith, None))) + sum(1 for _ in graph.triples((None, RDF.type, OWL.AllDisjointClasses)))
    counts["N"] = sum(1 for _ in graph.triples((None, OWL.propertyChainAxiom, None)))
    counts["E"] = sum(1 for _ in graph.triples((None, OWL.someValuesFrom, None)))
    counts["∩"] = sum(1 for _ in graph.triples((None, OWL.intersectionOf, None)))
    counts["¬"] = sum(1 for _ in graph.triples((None, OWL.complementOf, None)))
    counts["I"] = sum(1 for _ in graph.triples((None, OWL.inverseOf, None)))
    counts["Q"] = sum(1 for _ in graph.triples((None, OWL.equivalentClass, None))) + sum(1 for _ in graph.triples((None, OWL.equivalentProperty, None)))
    counts["R"] = sum(1 for _ in graph.triples((None, RDFS.domain, None))) + sum(1 for _ in graph.triples((None, RDFS.range, None)))
    counts["C"] = sum(sum(1 for _ in graph.triples((None, predicate, None))) for predicate in (OWL.cardinality, OWL.minCardinality, OWL.maxCardinality, OWL.qualifiedCardinality, OWL.minQualifiedCardinality, OWL.maxQualifiedCardinality))
    counts["L"] = sum(1 for _ in graph.triples((None, OWL.allValuesFrom, None)))
    counts["U"] = sum(1 for _ in graph.triples((None, OWL.unionOf, None)))
    for tag, owl_type in {"T": OWL.TransitiveProperty, "S": OWL.SymmetricProperty, "A": OWL.AsymmetricProperty, "F": OWL.FunctionalProperty, "V": OWL.ReflexiveProperty, "Y": OWL.IrreflexiveProperty}.items():
        counts[tag] = sum(1 for _ in graph.triples((None, RDF.type, owl_type)))
    counts["M"] = 0
    return {tag: counts[tag] for tag in TAGS}


def proof_coverage(stage_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in read_explanations(stage_path).values():
        for proof in record_proofs(record):
            counts.update(proof["primitive_tags"])
            if proof["m"]:
                counts["M"] += 1
    return counts


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "hop", "task_type", "task_id", "root_entity", "sparql_query", "answer", "primary_complexity", "old_v1_complexity", "old_vs_corrected", "complexity_json", "provenance_json", "attribution_evidence_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{field: row.get(field) for field in fields},
                "complexity_json": json.dumps(row["complexity"], ensure_ascii=False, sort_keys=True),
                "provenance_json": json.dumps(row["provenance"], ensure_ascii=False, sort_keys=True),
                "attribution_evidence_json": json.dumps(row["attribution_evidence"], ensure_ascii=False, sort_keys=True),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    all_rows: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    regenerated_coverage: Counter[str] = Counter()
    coverage_by_dataset: dict[str, Counter[str]] = {}
    coverage_by_dataset_hop: dict[str, Counter[str]] = {}
    eligible_coverage: Counter[str] = Counter()

    for dataset, (dataset_dir, ontology, resources_pattern) in DATASETS.items():
        dataset_coverage: Counter[str] = Counter()
        for hop in ("1hop", "2hop"):
            stage_dir = stage / dataset_dir / hop
            legacy_dir = ROOT / "data" / "output" / dataset_dir / hop
            resources = ROOT / "data" / "resources" / resources_pattern.format(hop=hop)
            if not (stage_dir / "Explanations.json").exists():
                raise FileNotFoundError(f"Missing regenerated output: {stage_dir}")
            rows, run = build_rows(dataset, hop, stage_dir, legacy_dir)
            resource_roots = {path.stem for path in resources.glob("*.ttl")}
            staged_questions = pd.read_csv(stage_dir / "SPARQL_questions.csv")
            output_roots = set(staged_questions["Root Entity"].astype(str))
            log_path = stage / "_logs" / f"{dataset_dir}_{hop}.stdout.log"
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            run.update(
                {
                    "resource_root_count": len(resource_roots),
                    "roots_with_generated_questions": len(output_roots),
                    "roots_with_no_generated_question": sorted(resource_roots - output_roots),
                    "generator_success_marker": "Success: true" in log_text,
                    "generator_failure_markers": [
                        line for line in log_text.splitlines()
                        if "Processing failed" in line or "Failed to process ontology" in line
                    ],
                    "roots_over_60_seconds": root_timing_audit(log_text),
                }
            )
            all_rows.extend(rows)
            runs[f"{dataset}/{hop}"] = run
            proof_counts = proof_coverage(stage_dir / "Explanations.json")
            regenerated_coverage.update(proof_counts)
            dataset_coverage.update(proof_counts)
            coverage_by_dataset_hop[f"{dataset}/{hop}"] = proof_counts
            for row in rows:
                eligible_coverage.update(set(row["complexity"]["distinct_primitive_reasoning_types"]))
                if row["complexity"]["m_status"] != "never":
                    eligible_coverage["M"] += 1
            for source in sorted(resources.glob("*.ttl")):
                inputs.append(file_record(source))
            if (resources / "sample_manifest.json").exists():
                inputs.append(file_record(resources / "sample_manifest.json"))
            inputs.extend(file_record(legacy_dir / name) for name in ("SPARQL_questions.csv", "Explanations.json"))
        coverage_by_dataset[dataset] = dataset_coverage
        inputs.append(file_record(ROOT / "data" / "input" / ontology))

    tbox_by_dataset = {dataset: tbox_census(ROOT / "data" / "input" / spec[1]) for dataset, spec in DATASETS.items()}
    tbox_total = {tag: sum(table[tag] for table in tbox_by_dataset.values()) for tag in TAGS}
    coverage = {
        tag: {
            "ontology_tbox_occurrence": tbox_total[tag],
            "ontology_tbox_present": bool(tbox_total[tag]),
            "regenerated_explanation_occurrence": regenerated_coverage[tag],
            "eligible_question_occurrence": eligible_coverage[tag],
            "by_dataset_regenerated": {dataset: coverage_by_dataset[dataset][tag] for dataset in DATASETS},
            "by_dataset_tbox": {dataset: tbox_by_dataset[dataset][tag] for dataset in DATASETS},
        }
        for tag in TAGS
    }
    deltas = Counter(row["old_vs_corrected"] for row in all_rows if row["old_vs_corrected"])
    deltas_by_task = {
        task: dict(Counter(row["old_vs_corrected"] for row in all_rows if row["task_type"] == task and row["old_vs_corrected"]))
        for task in ("BQA", "OEQA")
    }
    attribution = {
        flag: sum(bool(row["attribution_evidence"][flag]) for row in all_rows)
        for flag in ("legacy_m_present", "legacy_duplicate_semantic_tag_surplus", "new_nested_tag", "complete_answer_conjunction")
    }
    eligible_tag_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        tags = set(row["complexity"]["distinct_primitive_reasoning_types"])
        if row["complexity"]["m_status"] != "never":
            tags.add("M")
        for key in (
            "overall",
            f"{row['dataset']}",
            f"{row['dataset']}/{row['hop']}",
            f"{row['dataset']}/{row['task_type']}",
            f"{row['dataset']}/{row['hop']}/{row['task_type']}",
        ):
            eligible_tag_groups[key].update(tags)
    deterministic_files = {}
    for name in ("SPARQL_questions.csv", "Explanations.json"):
        first = stage / "_determinism" / "run_e" / name
        second = stage / "_determinism" / "run_f" / name
        deterministic_files[name] = {
            "run_e_sha256": sha256(first),
            "run_f_sha256": sha256(second),
            "byte_identical": first.read_bytes() == second.read_bytes(),
        }
    staged_files = [
        file_record(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and "audit" not in path.relative_to(stage).parts
    ]
    report = {
        "phase": 4,
        "status": "scientific-review-staging",
        "primary_complexity": "minimum complete primitive-tag count",
        "taxonomy": list(TAGS),
        "m_definition": "metadata true iff a proof union contains at least two distinct non-direct TBox tag types; never counted as primitive",
        "runs": runs,
        "eligible_source_pool_count": len(all_rows),
        "eligible_source_pool_by_dataset_task_hop": dict(Counter(f"{r['dataset']}/{r['task_type']}/{r['hop']}" for r in all_rows)),
        "old_vs_corrected": {"overall": dict(deltas), "by_task_type": deltas_by_task, "attribution_evidence_not_mutually_exclusive": attribution},
        "distributions": group_distributions(all_rows),
        "candidate_bins": candidate_bins(all_rows),
        "reasoning_coverage": coverage,
        "tag_counts": {
            "counting_rule": "regenerated counts are primitive occurrences across proof alternatives; eligible counts are per-question tag presence, with M presence recorded separately",
            "regenerated_proof_occurrences_by_dataset_hop": {
                key: {tag: counts[tag] for tag in TAGS}
                for key, counts in sorted(coverage_by_dataset_hop.items())
            },
            "eligible_question_presence": {
                key: {tag: counts[tag] for tag in TAGS}
                for key, counts in sorted(eligible_tag_groups.items())
            },
        },
        "previously_zero_now_nonzero": [tag for tag in ("A", "J", "E", "∩", "¬", "F", "V", "Y", "Q", "C", "L", "U") if regenerated_coverage[tag]],
        "reuse_assessment": {
            "nl_prompt_rerun_due_solely_to_symbolic_correction": False,
            "fs_prompt_rerun_due_solely_to_symbolic_correction": False,
            "gold_answer_labels_change_due_solely_to_symbolic_correction": False,
            "changes": "symbolic metadata and eventual sampling membership/analysis; AR requirements remain governed by Phase 3",
            "prediction_reuse_key": "dataset, hop, task type, normalized SPARQL query, answer, and context identity; never legacy Task ID alone",
        },
        "expected_sampling_implications": {
            "final_sampling_blocked_pending_review": True,
            "reason": "complexity is a stratification variable and no final Low/Medium/High scheme has been selected",
            "other_known_membership_change": "Phase 1 excludes Pizza DomainConcept Membership BQA targets while retaining OEQA answers",
            "expected_effect": "final v1.1 membership can change even when logical questions remain reusable because corrected complexity strata and Phase 1 eligibility are applied before sampling",
        },
        "expected_manuscript_revisions": [
            "benchmark construction and stratified-sampling complexity definition",
            "BQA positive/negative provenance description",
            "OEQA complete-answer conjunctive explanation semantics",
            "reasoning-type coverage table and claims based on regenerated proof evidence",
            "complexity distribution and Low/Medium/High bin counts",
            "any result analysis joined to v1 Max Tag Length or legacy Task ID",
            "limitations/reuse statement distinguishing unchanged NL/FS prompts from changed sampling metadata",
        ],
        "known_implementation_or_documentation_consumers_to_update_only_after_bin_review": [
            "README.md complexity and eight-tag v1 coverage statements",
            "scripts/llm_pipeline/stratified_sampling.py Max Tag Length stratification",
            "scripts/create_explanation_complexity_analysis.py v1 Max Tag Length join and 1/2-3/4+ bins",
            "scripts/create_chapter4_interpretation_audit.py legacy complexity analysis",
            "scripts/validate_release.py and release metadata fixed v1 coverage counts",
        ],
        "deterministic_regeneration_replay": {
            "scope": "the complete 13-root frozen toy_example_1hop input, regenerated twice in independent output directories",
            "files": deterministic_files,
        },
        "prohibitions_confirmed": {
            "llm_or_api_calls": False,
            "final_sampling_run": False,
            "sequential_public_ids_assigned": False,
            "nl_verbalizations_regenerated": False,
            "ar_prompts_regenerated": False,
            "final_benchmark_created": False,
        },
        "v1_artifact_guard": {
            "published_archives": [
                file_record(path)
                for path in sorted((ROOT / "final_benchmark").glob("*.zip"))
            ],
            "write_root_used": stage.relative_to(ROOT).as_posix(),
            "v1_output_paths_written": False,
        },
        "input_manifest": inputs,
        "output_manifest_before_audit_files": staged_files,
    }
    audit_dir = stage / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows_path = audit_dir / "eligible_source_pool_complexity.csv"
    write_rows(all_rows, rows_path)
    report_path = audit_dir / "phase4_report.json"
    report["audit_artifacts"] = [
        file_record(rows_path),
        {
            "path": report_path.relative_to(ROOT).as_posix(),
            "note": "self-describing report; a self-hash is intentionally omitted",
        },
    ]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(all_rows), "failures": sum(len(run["failures"]) for run in runs.values())}, indent=2))


if __name__ == "__main__":
    main()
