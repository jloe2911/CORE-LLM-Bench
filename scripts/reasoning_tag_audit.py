#!/usr/bin/env python3
"""Read-only v1.0 explanation and ontology-construct audit.

This preserves the published 20-tag taxonomy and distinguishes constructs
defined in source TBoxes from constructs exercised by stored explanations.
No benchmark or explanation files are written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import zipfile
import re

from rdflib import Graph, OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.oeqa_explanations import (  # noqa: E402
    _construct_tags,
    build_answer_explanations,
    explanation_axioms,
    explanation_tag,
    split_gold_answers,
)

TAGS = ("D", "H", "T", "S", "A", "J", "N", "E", "∩", "¬", "I", "F", "V", "Y", "Q", "R", "C", "L", "U", "M")
DATASETS = {
    "Family": ("FamilyOWL.zip", "FamilyOWL", "family.owl"),
    "Pizza100": ("pizza_100.zip", "pizza_100", "pizza_100.owl"),
    "Pizza250": ("pizza_250.zip", "pizza_250", "pizza_250.owl"),
    "OWL2Bench": ("OWL2Bench.zip", "OWL2Bench", "OWL2DL-1.owl"),
}


def iter_qas(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".json"):
                continue
            hop = "2hop" if "2hop" in member else "1hop"
            for group in json.loads(archive.read(member)):
                for qa in group.get("QAs", []):
                    yield hop, group, qa


def iter_explanations(qa):
    explanations = qa.get("Explanations") or []
    if explanations:
        yield from explanations
    elif qa.get("Minimum Explanation"):
        yield qa["Minimum Explanation"]


def expected_tags(explanation):
    tags = []
    for axiom in explanation_axioms(explanation):
        found = list(_construct_tags(axiom))
        tags.extend(found or ["D"])
    if len(set(tags) - {"D"}) >= 2:
        tags.append("M")
    return Counter(tags)


def explanation_audit(zip_path: Path):
    occurrences = Counter()
    explanations = 0
    artificial = 0
    artificial_types = Counter()
    mismatches = 0
    for _, _, qa in iter_qas(zip_path):
        for explanation in iter_explanations(qa):
            explanations += 1
            stored = Counter(explanation_tag(explanation))
            expected = expected_tags(explanation)
            occurrences.update(stored)
            if stored != expected:
                mismatches += 1
            affected = False
            for tag in "HINST":
                surplus = stored[tag] - expected[tag]
                if surplus > 0:
                    affected = True
                    artificial_types[tag] += surplus
            if affected:
                artificial += 1
    return {
        "explanations": explanations,
        "stored_tag_occurrences": dict(occurrences),
        "independent_reconstruction_mismatches": mismatches,
        "explanations_with_artificial_duplicate_occurrences": artificial,
        "artificial_duplicate_occurrences_by_tag": dict(artificial_types),
    }


def tbox_census(path: Path):
    graph = Graph()
    graph.parse(path)
    type_map = {
        "T": OWL.TransitiveProperty, "S": OWL.SymmetricProperty,
        "A": OWL.AsymmetricProperty, "F": OWL.FunctionalProperty,
        "V": OWL.ReflexiveProperty, "Y": OWL.IrreflexiveProperty,
    }
    counts = Counter()
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
    for tag, owl_type in type_map.items():
        counts[tag] = sum(1 for _ in graph.triples((None, RDF.type, owl_type)))
    return {tag: counts[tag] for tag in TAGS if tag not in {"D", "M"}}


def normalize_query(query):
    return " ".join(str(query).split())


def source_select_records(dataset_dir: str, hop: str):
    path = ROOT / "data" / "output" / dataset_dir / hop / "Explanations.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for key, value in data.items():
        record = {
            "Source Answer": value.get("inferred", {}).get("object"),
            "Explanations": value.get("explanations") or [],
            "Source Key": key,
        }
        for query in value.get("sparqlQueries", []):
            query_text = str(query)
            if query_text.lstrip().upper().startswith("SELECT"):
                select_query = normalize_query(query_text)
            else:
                match = re.search(
                    r"ASK\s+(?:WHERE\s+)?\{\s*<([^>]+)>\s+<([^>]+)>\s+<[^>]+>\s*\}",
                    query_text,
                    flags=re.IGNORECASE,
                )
                if not match:
                    continue
                subject, predicate = match.groups()
                select_query = normalize_query(
                    f"SELECT ?x WHERE {{ <{subject}> <{predicate}> ?x }}"
                )
            bucket = lookup.setdefault(select_query, [])
            if record not in bucket:
                bucket.append(record)
    return lookup


def oeqa_audit(zip_path: Path, dataset_dir: str):
    lookups = {hop: source_select_records(dataset_dir, hop) for hop in ("1hop", "2hop")}
    total = multi_answer = validated = 0
    validation_errors = []
    legacy_minima = []
    complete_minima = []
    complete_maxima = []
    primitive_minima = []
    primitive_maxima = []
    for hop, group, qa in iter_qas(zip_path):
        if str(group.get("Answer Type", "")).upper() != "MC":
            continue
        total += 1
        if len(split_gold_answers(qa.get("Answer"))) > 1:
            multi_answer += 1
        try:
            _, metrics = build_answer_explanations(
                qa.get("Answer"), lookups[hop].get(normalize_query(qa.get("SPARQL Query")), [])
            )
        except ValueError as exc:
            if len(validation_errors) < 10:
                validation_errors.append({"task_id": qa.get("Task ID"), "error": str(exc)})
            continue
        validated += 1
        old = qa.get("Minimum Explanation") or []
        legacy_minima.append(len(explanation_axioms(old)))
        complete_minima.append(metrics["Complete Explanation Min Axiom Count"])
        complete_maxima.append(metrics["Complete Explanation Max Axiom Count"])
        primitive_minima.append(metrics["Complete Explanation Min Primitive Tag Count"])
        primitive_maxima.append(metrics["Complete Explanation Max Primitive Tag Count"])
    def summary(values):
        return {"min": min(values, default=0), "max": max(values, default=0), "mean": round(statistics.fmean(values), 4) if values else 0}
    return {
        "oeqas": total,
        "oeqas_affected_by_flat_multi_answer_semantics": multi_answer,
        "exact_source_provenance_validated": validated,
        "validation_errors": validation_errors,
        "candidate_complexity_summaries": {
            "legacy_single_minimum_axioms": summary(legacy_minima),
            "complete_minimum_union_axioms": summary(complete_minima),
            "complete_maximum_union_axioms": summary(complete_maxima),
            "complete_minimum_primitive_tags": summary(primitive_minima),
            "complete_maximum_primitive_tags": summary(primitive_maxima),
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "final_benchmark")
    args = parser.parse_args()
    report = {"taxonomy": list(TAGS), "datasets": {}}
    for name, (zip_name, dataset_dir, ontology_name) in DATASETS.items():
        explanation = explanation_audit(args.benchmark_dir / zip_name)
        tbox = tbox_census(ROOT / "data" / "input" / ontology_name)
        exercised = {tag: explanation["stored_tag_occurrences"].get(tag, 0) for tag in TAGS}
        report["datasets"][name] = {
            "oeqa": oeqa_audit(args.benchmark_dir / zip_name, dataset_dir),
            "explanations": explanation,
            "tbox_constructs": tbox,
            "exercised_constructs": exercised,
            "present_but_unexercised": [tag for tag, count in tbox.items() if count and not exercised[tag]],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
