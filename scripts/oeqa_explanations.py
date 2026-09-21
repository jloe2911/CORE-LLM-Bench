"""Exact answer-grouped explanation semantics for open-ended QA.

An OEQA proof is conjunctive across answers and disjunctive within one
answer.  Source records must therefore retain the exact ``inferred.object``
provenance produced by the reasoner; no text matching is used here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


TAG_PREFIX = "TAG:"


def split_gold_answers(value: Any) -> list[str]:
    """Return the ordered, unique gold answers from the benchmark cell."""
    answers: list[str] = []
    for answer in str(value or "").split(";"):
        answer = answer.strip()
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def explanation_axioms(explanation: list[str]) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in explanation
        if not str(item).startswith(TAG_PREFIX)
    )


def explanation_tag(explanation: list[str]) -> str:
    for item in explanation:
        text = str(item)
        if text.startswith(TAG_PREFIX):
            return text[len(TAG_PREFIX) :]
    return ""


def primitive_tag_count(explanation: list[str]) -> int:
    """Count primitive tag occurrences; M is categorical metadata only."""
    return sum(1 for tag in explanation_tag(explanation) if tag != "M")


def _construct_tags(axiom: str) -> tuple[str, ...]:
    """Independently reconstruct primitive constructs in one rendered axiom.

    This fallback is used only for complete-union tag counts in the Python
    assembly stage.  Generation-time tagging remains OWLAPI-structural.
    """
    patterns = (
        ("D", r"\b(?:rdf:type|ClassAssertion|ObjectPropertyAssertion|DataPropertyAssertion)\b"),
        ("H", r"\b(?:SubClassOf|SubObjectPropertyOf|SubDataPropertyOf|rdfs:subClassOf|rdfs:subPropertyOf)\b"),
        ("Q", r"\b(?:EquivalentClasses|EquivalentObjectProperties|EquivalentDataProperties|equivalentClass|equivalentProperty)\b"),
        ("T", r"\bTransitiveObjectProperty\b"),
        ("S", r"\bSymmetricObjectProperty\b"),
        ("A", r"\bAsymmetricObjectProperty\b"),
        ("I", r"\b(?:InverseObjectProperties|inverseOf)\b"),
        ("F", r"\b(?:Inverse)?Functional(?:Object|Data)?Property\b"),
        ("N", r"\b(?:SubPropertyChainOf|ObjectPropertyChain|propertyChainAxiom)\b"),
        ("E", r"\b(?:Object|Data)?SomeValuesFrom\b|someValuesFrom"),
        ("L", r"\b(?:Object|Data)?AllValuesFrom\b|allValuesFrom"),
        ("C", r"\b(?:Object|Data)?(?:Min|Max|Exact)?Cardinality\b|cardinality"),
        ("∩", r"\bObjectIntersectionOf\b|intersectionOf"),
        ("U", r"\bObjectUnionOf\b|unionOf"),
        ("¬", r"\bObjectComplementOf\b|complementOf"),
        ("R", r"\b(?:Object|Data)Property(?:Domain|Range)\b|rdfs:(?:domain|range)|\b(?:Domain|Range)\b"),
        ("J", r"\bDisjoint(?:Classes|ObjectProperties|DataProperties)\b|disjointWith"),
        ("V", r"\bReflexiveObjectProperty\b"),
        ("Y", r"\bIrreflexiveObjectProperty\b"),
    )
    tags: list[str] = []
    for tag, pattern in patterns:
        tags.extend(tag for _ in re.finditer(pattern, axiom, flags=re.IGNORECASE))
    return tuple(tags)


def _alternative_metrics(explanation: list[str]) -> dict[str, Any]:
    axioms = explanation_axioms(explanation)
    return {
        "Axiom Count": len(axioms),
        "Primitive Tag Count": primitive_tag_count(explanation),
    }


def _complete_union_metrics(answer_groups: list[dict[str, Any]]) -> dict[str, int]:
    """Enumerate exact conjunctive selections with deterministic state folding."""
    states: set[tuple[frozenset[str], frozenset[tuple[str, str, int]]]] = {
        (frozenset(), frozenset())
    }
    for group in answer_groups:
        next_states: set[tuple[frozenset[str], frozenset[tuple[str, str, int]]]] = set()
        for axiom_union, tag_union in states:
            for explanation in group["Alternatives"]:
                axioms = frozenset(explanation_axioms(explanation))
                provenance = set()
                for axiom in axioms:
                    tags = _construct_tags(axiom) or ("D",)
                    provenance.update(
                        (axiom, tag, occurrence)
                        for occurrence, tag in enumerate(tags)
                    )
                next_states.add((axiom_union | axioms, tag_union | provenance))
        states = next_states

    axiom_counts = [len(axioms) for axioms, _ in states]
    tag_counts = [len(tags) for _, tags in states]
    distinct_non_direct_counts = [
        len({tag for _, tag, _ in tags if tag != "D"}) for _, tags in states
    ]
    min_types = min(distinct_non_direct_counts, default=0)
    max_types = max(distinct_non_direct_counts, default=0)
    if min_types >= 2:
        m_status = "always"
    elif max_types < 2:
        m_status = "never"
    else:
        m_status = "selection-dependent"
    return {
        "Complete Explanation Combination Count": _product(
            len(group["Alternatives"]) for group in answer_groups
        ),
        "Complete Explanation Min Axiom Count": min(axiom_counts, default=0),
        "Complete Explanation Max Axiom Count": max(axiom_counts, default=0),
        "Complete Explanation Min Primitive Tag Count": min(tag_counts, default=0),
        "Complete Explanation Max Primitive Tag Count": max(tag_counts, default=0),
        "Complete Explanation Min Distinct Non-Direct Tag Type Count": min_types,
        "Complete Explanation Max Distinct Non-Direct Tag Type Count": max_types,
        "Complete Explanation M Status": m_status,
    }


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def build_answer_explanations(
    gold_answer_value: Any, source_records: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build and validate exact answer groups from source provenance."""
    gold_answers = split_gold_answers(gold_answer_value)
    gold_set = set(gold_answers)
    grouped: dict[str, list[list[str]]] = {answer: [] for answer in gold_answers}

    for record in source_records:
        answer = record.get("Source Answer")
        if answer not in gold_set:
            raise ValueError(
                f"Explanation source answer {answer!r} is outside gold set {gold_answers!r}"
            )
        for explanation in record.get("Explanations") or []:
            key = json.dumps(explanation, ensure_ascii=False, sort_keys=True)
            if all(
                json.dumps(existing, ensure_ascii=False, sort_keys=True) != key
                for existing in grouped[answer]
            ):
                grouped[answer].append(explanation)

    missing = [answer for answer, alternatives in grouped.items() if not alternatives]
    if missing:
        raise ValueError(f"Gold answers lack exact-provenance explanations: {missing}")

    answer_groups: list[dict[str, Any]] = []
    for answer in gold_answers:
        alternatives = grouped[answer]
        metrics = [_alternative_metrics(explanation) for explanation in alternatives]
        answer_groups.append(
            {
                "Answer": answer,
                "Source Provenance": {"inferred.object": answer},
                "Alternatives": alternatives,
                "Alternative Count": len(alternatives),
                "Minimum Proof Axiom Count": min(m["Axiom Count"] for m in metrics),
                "Maximum Proof Axiom Count": max(m["Axiom Count"] for m in metrics),
                "Minimum Primitive Tag Count": min(
                    m["Primitive Tag Count"] for m in metrics
                ),
                "Maximum Primitive Tag Count": max(
                    m["Primitive Tag Count"] for m in metrics
                ),
            }
        )

    if len(answer_groups) != len(gold_answers):
        raise ValueError("Gold answers must be represented exactly once")
    return answer_groups, _complete_union_metrics(answer_groups)
