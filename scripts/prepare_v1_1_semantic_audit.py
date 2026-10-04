#!/usr/bin/env python3
"""Prepare an offline, identity-preserving audit of published v1.1.0.

This script never calls an API and never writes under release/v1.1.0.  It
extracts the frozen OWL contexts, repairs query IRIs from dataset-specific
ontology identity, and emits the batch input consumed by two Java reasoners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from rdflib import Graph, URIRef


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "release" / "v1.1.0" / "benchmark"
DEFAULT_OUTPUT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
GENEALOGY = "http://www.example.com/genealogy.owl#"
OWL2BENCH = "https://kracr.iiitd.edu.in/OWL2Bench#"
PIZZA_HASH = "http://www.co-ode.org/ontologies/pizza/pizza.owl#"
PIZZA_SLASH = "http://www.co-ode.org/ontologies/pizza/"
URI_PATTERN = re.compile(r"<([^>]+)>")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_name(iri: str) -> str:
    return unquote(str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1])


def parse_query(value: str) -> tuple[str, str, str | None]:
    iris = URI_PATTERN.findall(str(value))
    if str(value).lstrip().upper().startswith("ASK") and len(iris) == 3:
        return iris[0], iris[1], iris[2]
    if str(value).lstrip().upper().startswith("SELECT") and len(iris) == 2:
        return iris[0], iris[1], None
    raise ValueError(f"Unsupported query: {value!r}")


def corrected_terms(row: dict[str, object]) -> tuple[str, str, str | None]:
    old_subject, old_predicate, old_object = parse_query(str(row["formal_query"]))
    subject, predicate = local_name(old_subject), local_name(old_predicate)
    object_name = None if old_object is None else local_name(old_object)
    dataset = str(row["dataset_key"])
    if dataset == "Family":
        return GENEALOGY + subject, (RDF_TYPE if predicate == "type" else GENEALOGY + predicate), (
            None if object_name is None else GENEALOGY + object_name
        )
    if dataset == "OWL2Bench":
        return OWL2BENCH + subject, (RDF_TYPE if predicate == "type" else OWL2BENCH + predicate), (
            None if object_name is None else OWL2BENCH + object_name
        )
    if dataset in {"Pizza100", "Pizza250"}:
        return PIZZA_SLASH + subject, (RDF_TYPE if predicate == "type" else PIZZA_HASH + predicate), (
            None
            if object_name is None
            else (PIZZA_HASH + object_name if predicate == "type" else PIZZA_SLASH + object_name)
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def render_query(answer_type: str, terms: tuple[str, str, str | None]) -> str:
    subject, predicate, object_iri = terms
    if answer_type == "BIN":
        if object_iri is None:
            raise ValueError("BIN query lacks object")
        return f"ASK WHERE {{ <{subject}> <{predicate}> <{object_iri}> }}"
    return f"SELECT ?x WHERE {{ <{subject}> <{predicate}> ?x }}"


def validate_signature(graph: Graph, terms: tuple[str, str, str | None]) -> list[str]:
    all_iris = {str(value) for triple in graph for value in triple if isinstance(value, URIRef)}
    return [iri for iri in terms if iri is not None and iri not in all_iris and iri != RDF_TYPE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    contexts = output / "contexts"
    contexts.mkdir(parents=True, exist_ok=True)

    benchmark_path = PUBLISHED / "core_llm_bench_v1_1.parquet"
    benchmark = pd.read_parquet(benchmark_path)
    rows: list[dict[str, object]] = []
    context_paths: dict[str, Path] = {}
    context_signatures: dict[str, set[str]] = {}
    signature_failures = 0
    for record in benchmark.to_dict(orient="records"):
        context = str(record["fs_context"])
        context_hash = sha256_text(context)
        path = context_paths.get(context_hash)
        if path is None:
            graph = Graph()
            graph.parse(data=context, format="turtle")
            path = contexts / f"{context_hash}.ttl"
            path.write_text(context, encoding="utf-8", newline="\n")
            context_paths[context_hash] = path
            context_signatures[context_hash] = {
                str(value) for triple in graph for value in triple if isinstance(value, URIRef)
            }
        terms = corrected_terms(record)
        signature = context_signatures[context_hash]
        missing = [
            iri for iri in terms if iri is not None and iri not in signature and iri != RDF_TYPE
        ]
        signature_failures += bool(missing)
        corrected = render_query(str(record["answer_type"]), terms)
        rows.append(
            {
                "task_id": str(record["task_id"]),
                "context_hash": context_hash,
                "context_path": str(path),
                "answer_type": str(record["answer_type"]),
                "subject_iri": terms[0],
                "predicate_iri": terms[1],
                "object_iri": terms[2] or "",
                "original_query": str(record["formal_query"]),
                "corrected_query": corrected,
                "query_changed": corrected != str(record["formal_query"]),
                "signature_missing_iris": ";".join(missing),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["context_hash", "task_id"], kind="stable")
    frame.to_csv(output / "query_identity_audit.csv", index=False, lineterminator="\n")
    columns = ["task_id", "context_path", "answer_type", "subject_iri", "predicate_iri", "object_iri"]
    frame[columns].to_csv(output / "reasoner_input.tsv", sep="\t", index=False, lineterminator="\n")
    report = {
        "status": "prepared-no-llm-calls",
        "published_benchmark": str(benchmark_path),
        "published_benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "questions": len(frame),
        "unique_contexts": len(context_paths),
        "changed_queries": int(frame.query_changed.sum()),
        "unchanged_queries": int((~frame.query_changed).sum()),
        "signature_failures": signature_failures,
        "next_step": "Run ReasonerBatchAudit with Openllet, then StructuralReasoner where feasible.",
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
