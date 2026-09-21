"""Reusable validation helpers for abstract-representation benchmark rows."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable

from rdflib import Graph, Literal, URIRef

from scripts.ontology_tools.abstraction.OntologyAbstractor import (
    LEXICAL_ANNOTATIONS,
    STANDARD_PREFIXES,
    abstract_graph,
    local_name,
)


GENERIC_VOCABULARY = {
    "class",
    "classes",
    "instance",
    "instances",
    "property",
    "properties",
    "relationship",
    "relationships",
    "individual",
    "individuals",
    "ontology",
    "entity",
    "entities",
    "type",
    "types",
    "true",
    "false",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
URI_IN_QUERY = re.compile(r"<([^>]+)>")


@dataclass
class ARValidationResult:
    duplicate_exact_sentences: list[str] = field(default_factory=list)
    unmapped_required_entities: list[str] = field(default_factory=list)
    mapping_inconsistencies: list[str] = field(default_factory=list)
    original_identifiers_remaining: list[str] = field(default_factory=list)
    original_lexical_labels_remaining: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(asdict(self).values())

    def counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in asdict(self).items()}


def replace_text(text, text_mappings: dict[str, str]) -> str:
    if text is None:
        return ""
    result = str(text)
    for original in sorted(text_mappings, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(original)}(?!\w)", re.IGNORECASE)
        result = pattern.sub(text_mappings[original], result)
    return result


def abstract_answer(answer, answer_type: str, text_mappings: dict[str, str]) -> str:
    """BQA truth values stay fixed; OEQA entity surfaces use the shared map."""
    if str(answer_type).upper() in {"BIN", "BQA", "BINARY"}:
        return "" if answer is None else str(answer)
    return replace_text(answer, text_mappings)


def source_terms(
    graphs: Iterable[Graph], uri_mappings: dict[URIRef, URIRef]
) -> tuple[set[str], set[str]]:
    identifiers = {local_name(uri) for uri in uri_mappings}
    labels: set[str] = set()
    for graph in graphs:
        for uri in uri_mappings:
            for predicate in LEXICAL_ANNOTATIONS:
                for value in graph.objects(uri, predicate):
                    if isinstance(value, Literal) and str(value).strip():
                        labels.add(str(value).strip())
    return identifiers, labels


def required_query_entities(sparql_query: str) -> set[str]:
    return {
        uri
        for uri in URI_IN_QUERY.findall(str(sparql_query or ""))
        if not uri.startswith(STANDARD_PREFIXES)
    }


def query_specific_text_mappings(
    sparql_query: str,
    graph: Graph,
    uri_mappings: dict[URIRef, URIRef],
) -> dict[str, str]:
    """Resolve even globally ambiguous labels by formal-query URI identity."""
    required_locals = {local_name(uri) for uri in required_query_entities(sparql_query)}
    candidates: dict[str, set[str]] = {}
    for original, abstract in uri_mappings.items():
        if local_name(original) not in required_locals:
            continue
        target = local_name(abstract)
        original_local = local_name(original)
        candidates.setdefault(original_local, set()).add(target)
        display_bases = {
            original_local,
            re.sub(r"_dynamic_\d+$", "", original_local, flags=re.IGNORECASE),
        }
        for display_base in display_bases:
            cleaned = re.sub(
                r"_\d{4}$|_\d+$|_v\d+$|_\w{2,3}$", "", display_base
            )
            cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
            cleaned = cleaned.replace("_", " ").replace("-", " ")
            cleaned = " ".join(word.capitalize() for word in cleaned.split())
            if cleaned:
                candidates.setdefault(cleaned, set()).add(target)
        for predicate in LEXICAL_ANNOTATIONS:
            for value in graph.objects(original, predicate):
                if isinstance(value, Literal) and str(value).strip():
                    candidates.setdefault(str(value).strip(), set()).add(target)
    return {
        source: next(iter(targets))
        for source, targets in candidates.items()
        if len(targets) == 1
    }


@lru_cache(maxsize=1024)
def _leak_pattern(terms: tuple[str, ...]):
    canonical = {}
    for term in terms:
        if term.casefold() not in GENERIC_VOCABULARY and term.strip():
            canonical.setdefault(term.casefold(), term)
    if not canonical:
        return None, canonical
    alternatives = "|".join(
        re.escape(term) for term in sorted(canonical.values(), key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE), canonical


def _find_leaks(text: str, terms: Iterable[str]) -> list[str]:
    normalized = tuple(sorted(set(terms), key=lambda value: (value.casefold(), value)))
    pattern, canonical = _leak_pattern(normalized)
    if pattern is None:
        return []
    found = {canonical[match.group(0).casefold()] for match in pattern.finditer(text)}
    return sorted(found, key=lambda value: (value.casefold(), value))


def validate_ar_row(
    *,
    original_question: str,
    original_answer: str,
    answer_type: str,
    sparql_query: str,
    abs_question: str,
    abs_context: str,
    abs_answer: str,
    uri_mappings: dict[URIRef, URIRef],
    text_mappings: dict[str, str],
    identifiers: Iterable[str],
    lexical_labels: Iterable[str],
    expected_abs_question: str | None = None,
) -> ARValidationResult:
    result = ARValidationResult()
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(abs_context or "") if part.strip()]
    counts = Counter(sentences)
    result.duplicate_exact_sentences = sorted(
        sentence for sentence, count in counts.items() if count > 1
    )

    required = required_query_entities(sparql_query)
    mapped_uri_strings = {str(uri) for uri in uri_mappings}
    result.unmapped_required_entities = sorted(
        uri
        for uri in required
        if uri not in mapped_uri_strings and local_name(uri) not in text_mappings
    )

    expected_question = (
        replace_text(original_question, text_mappings)
        if expected_abs_question is None
        else expected_abs_question
    )
    expected_answer = abstract_answer(original_answer, answer_type, text_mappings)
    if str(abs_question) != expected_question:
        result.mapping_inconsistencies.append("ABS Question differs from shared mapping")
    if str(abs_answer) != expected_answer:
        result.mapping_inconsistencies.append("ABS Answer differs from task-semantic mapping")

    combined = "\n".join((str(abs_question), str(abs_context), str(abs_answer)))
    result.original_identifiers_remaining = _find_leaks(combined, identifiers)
    result.original_lexical_labels_remaining = _find_leaks(combined, lexical_labels)
    return result


def render_abstract_context(graph: Graph, uri_mappings: dict[URIRef, URIRef]) -> str:
    """Render only the ABox context used by the final benchmark, in memory."""
    from scripts.llm_pipeline.verbalize_ontologies import (
        describe_individual_with_domain_independence,
        get_all_individuals,
    )
    from scripts.llm_pipeline.verbalize_abstract import deduplicate_sentences

    abstracted = abstract_graph(graph, uri_mappings)
    from rdflib.namespace import OWL, RDF

    classes = {
        subject
        for subject in abstracted.subjects(RDF.type, OWL.Class)
        if isinstance(subject, URIRef)
    }
    object_properties = {
        subject
        for subject in abstracted.subjects(RDF.type, OWL.ObjectProperty)
        if isinstance(subject, URIRef)
    }
    individuals = get_all_individuals(abstracted, classes, object_properties)
    descriptions = []
    for individual in sorted(individuals, key=str):
        description = describe_individual_with_domain_independence(
            abstracted, individual, classes, object_properties, individuals
        )
        if description:
            descriptions.append(description)
    return "\n".join(deduplicate_sentences(descriptions)) + "\n"
