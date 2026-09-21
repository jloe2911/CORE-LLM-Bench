from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.phase4_symbolic_audit import (
    build_rows,
    complete_oeqa_metrics,
    proof_from_structured,
)


def structured(*axioms, m=False):
    primitive = [tag for _, tags in axioms for tag in tags]
    return {
        "semanticAxioms": [
            {"identity": identity, "rendering": identity, "primitiveTags": tags}
            for identity, tags in axioms
        ],
        "primitiveTags": primitive,
        "distinctPrimitiveTags": sorted(set(primitive)),
        "axiomCount": len(dict(axioms)),
        "primitiveTagCount": len(primitive),
        "m": m,
    }


def test_duplicate_semantic_axiom_is_counted_once():
    raw = structured(("ClassAssertion(A x)", ["D"]), ("ClassAssertion(A x)", ["D"]))
    raw["axiomCount"] = 1
    raw["primitiveTagCount"] = 1
    raw["primitiveTags"] = ["D"]
    proof = proof_from_structured(raw)
    assert proof["axiom_count"] == 1
    assert proof["primitive_tag_count"] == 1


def test_nested_existential_intersection_and_cardinality_are_retained():
    proof = proof_from_structured(
        structured(
            (
                "SubClassOf(A ObjectSomeValuesFrom(p ObjectIntersectionOf(B ObjectMinCardinality(2 q C))))",
                ["H", "E", "∩", "C"],
            ),
            m=True,
        )
    )
    assert proof["primitive_tags"] == ("H", "E", "∩", "C")
    assert proof["m"] is True


def test_complete_oeqa_is_conjunctive_and_deduplicates_shared_axioms():
    shared = ("ObjectPropertyRange(p C)", ["R"])
    a = proof_from_structured(structured(("ClassAssertion(A x)", ["D"]), shared))
    b = proof_from_structured(structured(("ClassAssertion(B y)", ["D"]), shared))
    result = complete_oeqa_metrics(
        [{"answer": "A", "proofs": [a]}, {"answer": "B", "proofs": [b]}]
    )
    assert result["complete_combination_count"] == 1
    assert result["complete_min_axiom_count"] == 3
    assert result["complete_min_primitive_tag_count"] == 3
    assert result["m_status"] == "never"


def test_bqa_negative_inherits_exact_positive_complexity(tmp_path: Path):
    stage = tmp_path / "stage"
    legacy = tmp_path / "legacy"
    stage.mkdir()
    legacy.mkdir()
    positive = "ASK WHERE { <urn:s> <urn:p> <urn:o> }"
    negative = "ASK WHERE { <urn:s> <urn:p> <urn:n> }"
    columns = [
        "Task ID", "Root Entity", "Size of ontology TBox", "Size of ontology ABox",
        "Task Type", "Answer Type", "SPARQL Query", "Predicate", "Answer",
        "Min Tag Length", "Max Tag Length",
    ]
    rows = [
        ["positive", "root", 1, 1, "Property Assertion", "BIN", positive, "p", "TRUE", 1, 1],
        ["negative", "root", 1, 1, "Property Assertion", "BIN", negative, "p", "FALSE", 1, 1],
    ]
    for directory in (stage, legacy):
        pd.DataFrame(rows, columns=columns).to_csv(directory / "SPARQL_questions.csv", index=False)
    explanation = {
        "root||s|p|o": {
            "inferred": {"subject": "s", "predicate": "p", "object": "o"},
            "explanations": [["ClassAssertion(A x)", "TAG:D"]],
            "structuredExplanations": [structured(("ClassAssertion(A x)", ["D"]))],
            "size": {"min": 1, "max": 1},
            "explanationCount": 1,
            "taskIds": ["positive"],
            "sparqlQueries": [positive],
        }
    }
    for directory in (stage, legacy):
        (directory / "Explanations.json").write_text(json.dumps(explanation), encoding="utf-8")
    result, audit = build_rows("Family", "1hop", stage, legacy)
    assert audit["failures"] == []
    by_answer = {row["answer"]: row for row in result}
    assert by_answer["TRUE"]["primary_complexity"] == by_answer["FALSE"]["primary_complexity"]
    assert by_answer["FALSE"]["provenance"]["positive_task_id"] == "positive"
    assert by_answer["FALSE"]["provenance"]["inherited_by_negative"] is True
