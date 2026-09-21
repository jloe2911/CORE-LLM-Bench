"""Graph-semantic ontology abstraction for the AR benchmark condition.

The v1.0 implementation selected one dominant namespace and classified the
remaining local names lexically. That missed, among other things, Pizza
individuals whose slash namespace differed from the class hash namespace.
This module instead assigns roles from RDF/OWL declarations and graph use and
keeps URI identity until the final display layer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD


ABSTRACT_NAMESPACE = "http://www.example.com/abstracted.owl#"
CATEGORIES = ("classes", "object_properties", "data_properties", "individuals")
CATEGORY_PREFIX = {
    "classes": "Class",
    "object_properties": "Property",
    "data_properties": "DataProperty",
    "individuals": "Individual",
}

LEXICAL_ANNOTATIONS = {
    RDFS.label,
    URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
    URIRef("http://purl.org/dc/elements/1.1/title"),
    URIRef("http://xmlns.com/foaf/0.1/name"),
    URIRef("http://schema.org/name"),
    URIRef("https://schema.org/name"),
}

STANDARD_PREFIXES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    "http://www.w3.org/XML/1998/namespace",
    "http://www.w3.org/2004/02/skos/core#",
    "http://purl.org/dc/",
    "http://xmlns.com/foaf/",
    "http://schema.org/",
    "https://schema.org/",
)

CLASS_RELATIONS = {
    RDFS.subClassOf,
    OWL.equivalentClass,
    OWL.disjointWith,
    OWL.complementOf,
    OWL.someValuesFrom,
    OWL.allValuesFrom,
    OWL.onClass,
}
PROPERTY_RELATIONS = {
    RDFS.subPropertyOf,
    OWL.equivalentProperty,
    OWL.propertyDisjointWith,
    OWL.inverseOf,
    OWL.onProperty,
}
PROPERTY_TYPES = {
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.ReflexiveProperty,
    OWL.IrreflexiveProperty,
}


class RoleConflictError(ValueError):
    """Raised when a URI has incompatible abstraction roles."""


def local_name(uri: URIRef | str) -> str:
    value = str(uri).rstrip("/#")
    if "#" in value:
        value = value.rsplit("#", 1)[1]
    else:
        value = value.rsplit("/", 1)[-1]
    return unquote(value)


def _is_custom_uri(value) -> bool:
    return isinstance(value, URIRef) and not str(value).startswith(STANDARD_PREFIXES)


def _add_uri(target: set[URIRef], value) -> None:
    if _is_custom_uri(value):
        target.add(value)


def discover_entity_roles(graph: Graph) -> dict[str, set[URIRef]]:
    """Discover semantic roles from declarations and structural RDF use."""
    roles = {category: set() for category in CATEGORIES}

    declarations = {
        OWL.Class: "classes",
        RDFS.Class: "classes",
        OWL.ObjectProperty: "object_properties",
        OWL.DatatypeProperty: "data_properties",
        OWL.NamedIndividual: "individuals",
    }
    for rdf_type, category in declarations.items():
        for subject in graph.subjects(RDF.type, rdf_type):
            _add_uri(roles[category], subject)

    for predicate in CLASS_RELATIONS:
        for subject, obj in graph.subject_objects(predicate):
            if predicate in {RDFS.subClassOf, OWL.equivalentClass, OWL.disjointWith}:
                _add_uri(roles["classes"], subject)
            _add_uri(roles["classes"], obj)
    for cls in graph.objects(None, RDF.type):
        if cls not in PROPERTY_TYPES and cls not in {OWL.NamedIndividual, OWL.Ontology}:
            _add_uri(roles["classes"], cls)

    undecided_predicates: dict[URIRef, set[str]] = defaultdict(set)
    for predicate in PROPERTY_RELATIONS:
        for subject, obj in graph.subject_objects(predicate):
            if subject not in roles["data_properties"]:
                _add_uri(roles["object_properties"], subject)
            if obj not in roles["data_properties"]:
                _add_uri(roles["object_properties"], obj)
    # A domain constrains the subject class but does not distinguish object
    # from data properties. Declarations, range, and predicate use do.
    for prop, range_value in graph.subject_objects(RDFS.range):
        if _is_custom_uri(prop):
            if range_value == RDFS.Literal or str(range_value).startswith(str(XSD)):
                roles["data_properties"].add(prop)
            elif prop not in roles["data_properties"]:
                roles["object_properties"].add(prop)

    structural_predicates = {
        RDF.type,
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
        *LEXICAL_ANNOTATIONS,
    }
    for _subject, predicate, obj in graph:
        if predicate in structural_predicates or not _is_custom_uri(predicate):
            continue
        if isinstance(obj, Literal):
            undecided_predicates[predicate].add("data_properties")
        elif isinstance(obj, (URIRef, BNode)):
            undecided_predicates[predicate].add("object_properties")
    for predicate, categories in undecided_predicates.items():
        for category in categories:
            roles[category].add(predicate)

    for subject, cls in graph.subject_objects(RDF.type):
        if cls in roles["classes"] or cls == OWL.NamedIndividual:
            _add_uri(roles["individuals"], subject)
    for prop in roles["object_properties"]:
        for subject, obj in graph.subject_objects(prop):
            _add_uri(roles["individuals"], subject)
            _add_uri(roles["individuals"], obj)

    schema_entities = (
        roles["classes"] | roles["object_properties"] | roles["data_properties"]
    )
    roles["individuals"].difference_update(schema_entities)
    validate_role_conflicts(roles)
    return roles


def validate_role_conflicts(roles: dict[str, set[URIRef]]) -> None:
    memberships: dict[URIRef, list[str]] = defaultdict(list)
    for category, entities in roles.items():
        for entity in entities:
            memberships[entity].append(category)
    conflicts = {uri: cats for uri, cats in memberships.items() if len(cats) > 1}
    if conflicts:
        detail = "; ".join(
            f"{uri} -> {','.join(sorted(cats))}"
            for uri, cats in sorted(conflicts.items(), key=lambda item: str(item[0]))
        )
        raise RoleConflictError(f"Incompatible abstraction roles: {detail}")


def merge_entity_roles(role_sets: Iterable[dict[str, set[URIRef]]]):
    merged = {category: set() for category in CATEGORIES}
    for roles in role_sets:
        for category in CATEGORIES:
            merged[category].update(roles[category])
    validate_role_conflicts(merged)
    return merged


def create_abstraction_mappings(
    roles: dict[str, set[URIRef]],
) -> dict[str, dict[URIRef, URIRef]]:
    validate_role_conflicts(roles)
    mappings: dict[str, dict[URIRef, URIRef]] = {category: {} for category in CATEGORIES}
    for category in CATEGORIES:
        for index, entity in enumerate(sorted(roles[category], key=str), 1):
            mappings[category][entity] = URIRef(
                f"{ABSTRACT_NAMESPACE}{CATEGORY_PREFIX[category]}{index}"
            )
    return mappings


def flatten_mappings(mappings: dict[str, dict[URIRef, URIRef]]):
    flat: dict[URIRef, URIRef] = {}
    for category in CATEGORIES:
        for original, abstract in mappings[category].items():
            if original in flat and flat[original] != abstract:
                raise RoleConflictError(f"Multiple mappings for {original}")
            flat[original] = abstract
    return flat


def abstract_graph(graph: Graph, mappings: dict[URIRef, URIRef]) -> Graph:
    """Return a clean abstract graph without source lexical annotations."""
    result = Graph()
    for prefix, namespace in graph.namespaces():
        result.bind(prefix, namespace)
    result.bind("abs", URIRef(ABSTRACT_NAMESPACE))
    for subject, predicate, obj in graph:
        if subject in mappings and predicate in LEXICAL_ANNOTATIONS:
            continue
        new_subject = mappings.get(subject, subject)
        new_predicate = mappings.get(predicate, predicate)
        new_obj = mappings.get(obj, obj) if isinstance(obj, URIRef) else obj
        result.add((new_subject, new_predicate, new_obj))
    return result


def build_text_mappings(
    graphs: Iterable[Graph], mappings: dict[URIRef, URIRef]
) -> tuple[dict[str, str], set[str]]:
    """Build unambiguous local-name/label aliases for question rendering."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for original, abstract in mappings.items():
        candidates[local_name(original)].add(local_name(abstract))
    for graph in graphs:
        for predicate in LEXICAL_ANNOTATIONS:
            for original, label in graph.subject_objects(predicate):
                if original in mappings and isinstance(label, Literal) and str(label).strip():
                    candidates[str(label).strip()].add(local_name(mappings[original]))
    aliases = {
        source: next(iter(targets))
        for source, targets in candidates.items()
        if len(targets) == 1
    }
    ambiguous = {source for source, targets in candidates.items() if len(targets) > 1}
    return dict(sorted(aliases.items())), ambiguous


def write_mapping_file(
    path: Path,
    mappings: dict[str, dict[URIRef, URIRef]],
    aliases: dict[str, str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("=== ONTOLOGY ABSTRACTION MAPPINGS ===\n\n")
        for category in CATEGORIES:
            handle.write(f"=== {category.upper().replace('_', ' ')} ===\n")
            for original, abstract in sorted(mappings[category].items(), key=lambda x: str(x[0])):
                handle.write(f"<{original}> -> <{abstract}>\n")
            handle.write("\n")
        if aliases:
            handle.write("=== TEXT ALIASES ===\n")
            for original, abstract in sorted(aliases.items()):
                handle.write(f"@alias {original} -> <{ABSTRACT_NAMESPACE}{abstract}>\n")
            handle.write("\n")


class OntologyAbstractor:
    """Compatibility facade exposing the graph-semantic implementation."""

    def extract_ontology_elements(self, ontology_text: str):
        graph = Graph().parse(data=ontology_text, format="turtle")
        return discover_entity_roles(graph)

    def create_abstraction_mappings(self, elements):
        normalized = {
            category: {URIRef(value) for value in elements[category]}
            for category in CATEGORIES
        }
        return create_abstraction_mappings(normalized)

    def abstract_ontology_text(self, ontology_text: str, mappings):
        graph = Graph().parse(data=ontology_text, format="turtle")
        normalized = {
            URIRef(str(key).strip("<>")): URIRef(str(value).strip("<>"))
            for key, value in mappings.items()
        }
        return abstract_graph(graph, normalized).serialize(format="turtle")


def analyze_ontology_directory(ontology_dir: str | Path):
    ttl_files = sorted(Path(ontology_dir).glob("**/*.ttl"), key=lambda p: str(p))
    role_sets = []
    for ttl_file in ttl_files:
        graph = Graph().parse(ttl_file, format="turtle")
        role_sets.append(discover_entity_roles(graph))
    merged = merge_entity_roles(role_sets)
    mappings = create_abstraction_mappings(merged)
    return ttl_files, merged, mappings


def process_ontology_abstraction(ontology_dir: str, output_dir: str):
    """Write corrected abstract ontologies and one deterministic mapping file."""
    output_root = Path(output_dir)
    ontology_output = output_root / "abstracted_ontologies"
    ontology_output.mkdir(parents=True, exist_ok=True)
    ttl_files, roles, mappings = analyze_ontology_directory(ontology_dir)
    flat = flatten_mappings(mappings)

    mapping_file = output_root / "abstraction_mappings.txt"
    graphs = [Graph().parse(source, format="turtle") for source in ttl_files]
    aliases, ambiguous_aliases = build_text_mappings(graphs, flat)
    write_mapping_file(mapping_file, mappings, aliases)
    for source in ttl_files:
        graph = Graph().parse(source, format="turtle")
        abstracted = abstract_graph(graph, flat)
        (ontology_output / source.name).write_text(
            abstracted.serialize(format="turtle"), encoding="utf-8", newline="\n"
        )

    print(f"Processed {len(ttl_files)} ontologies")
    for category in CATEGORIES:
        print(f"{category}: {len(roles[category])}")
    if ambiguous_aliases:
        print(f"ambiguous text aliases excluded: {len(ambiguous_aliases)}")
    return {"ontologies_dir": ontology_output, "mappings_file": mapping_file}


def main():
    parser = argparse.ArgumentParser(description="Graph-semantic ontology abstraction")
    parser.add_argument("--ontology_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    process_ontology_abstraction(args.ontology_dir, args.output_dir)


if __name__ == "__main__":
    main()
