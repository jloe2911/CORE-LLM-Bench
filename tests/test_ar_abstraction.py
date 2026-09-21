from __future__ import annotations

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS

from scripts.ar_validation import (
    abstract_answer,
    query_specific_text_mappings,
    render_abstract_context,
    source_terms,
    validate_ar_row,
)
from scripts.llm_pipeline.verbalize_ontologies import get_nice_label
from scripts.llm_pipeline.verbalize_abstract import (
    deduplicate_sentences,
    parse_mapping_lines,
    parse_uri_mapping_lines,
    query_text_mappings,
)
from scripts.ontology_tools.abstraction.OntologyAbstractor import (
    RoleConflictError,
    abstract_graph,
    build_text_mappings,
    create_abstraction_mappings,
    discover_entity_roles,
    flatten_mappings,
    local_name,
    merge_entity_roles,
)


EX = Namespace("http://example.test/pizza#")
INST = Namespace("http://example.test/pizza/")


def professor_graph() -> Graph:
    """Minimal trace fixture for the reported Pizza100 regression.

    The old dominant-namespace extractor saw ``EX`` classes/properties but not
    slash-namespace ``INST`` individuals. Their retained shared labels then
    caused label-keyed relation aggregation to emit the same sentence eight
    times.
    """
    graph = Graph()
    graph.add((EX.AsparagusTopping, RDF.type, OWL.Class))
    graph.add((EX.AsparagusTopping, RDFS.label, Literal("Asparagus Topping")))
    graph.add((EX.hasTopping, RDF.type, OWL.ObjectProperty))
    graph.add((EX.hasTopping, RDFS.label, Literal("has topping")))
    graph.add((INST.AsparagusTopping_dynamic_1, RDF.type, OWL.NamedIndividual))
    graph.add((INST.AsparagusTopping_dynamic_1, RDF.type, EX.AsparagusTopping))
    for suffix in (0, 11, 18, 64, 71, 76, 77, 87):
        individual = INST[f"Parmense_{suffix}"]
        graph.add((individual, RDF.type, OWL.NamedIndividual))
        graph.add((individual, RDFS.label, Literal("Parmense")))
        graph.add((individual, EX.hasTopping, INST.AsparagusTopping_dynamic_1))
    return graph


def corrected(graph: Graph):
    roles = discover_entity_roles(graph)
    categorized = create_abstraction_mappings(roles)
    flat = flatten_mappings(categorized)
    aliases, ambiguous = build_text_mappings([graph], flat)
    return roles, categorized, flat, aliases, ambiguous


def test_professor_asparagus_example_has_no_repeated_source_sentence():
    graph = professor_graph()
    _roles, _categorized, mappings, _aliases, _ambiguous = corrected(graph)
    context = render_abstract_context(graph, mappings)

    assert "Parmense" not in context
    assert "Asparagus Topping" not in context
    assert context.count(" Property1 Individual1.") == 8
    assert len(context.splitlines()) == len(set(context.splitlines()))


def test_pizza_slash_namespace_individuals_receive_individual_mappings():
    graph = professor_graph()
    roles, categorized, _flat, _aliases, _ambiguous = corrected(graph)
    assert INST.Parmense_0 in roles["individuals"]
    assert local_name(categorized["individuals"][INST.Parmense_0]).startswith("Individual")


def test_distinct_entities_with_same_label_do_not_collapse():
    graph = professor_graph()
    _roles, categorized, mappings, _aliases, ambiguous = corrected(graph)
    assert "Parmense" in ambiguous
    assert categorized["individuals"][INST.Parmense_0] != categorized["individuals"][INST.Parmense_11]
    context = render_abstract_context(graph, mappings)
    assert len([line for line in context.splitlines() if "Property1" in line]) == 8

    query = (
        f"ASK WHERE {{ <{INST.Parmense_0}> <{RDF.type}> "
        f"<{EX.AsparagusTopping}> }}"
    )
    # Query-specific identity can safely resolve a globally ambiguous label.
    assert query_specific_text_mappings(query, graph, mappings)["Parmense"] == local_name(
        mappings[INST.Parmense_0]
    )


def test_abstract_rendering_ignores_source_labels():
    graph = professor_graph()
    _roles, _categorized, mappings, _aliases, _ambiguous = corrected(graph)
    abstracted = abstract_graph(graph, mappings)
    for source, target in mappings.items():
        assert get_nice_label(abstracted, target) == local_name(target)
        assert not list(abstracted.objects(target, RDFS.label))


def test_duplicate_relation_surfaces_remain_distinct_by_uri():
    graph = Graph()
    graph.add((EX.Person, RDF.type, OWL.Class))
    for prop in (EX.likes, EX.admires):
        graph.add((prop, RDF.type, OWL.ObjectProperty))
        graph.add((prop, RDFS.label, Literal("same label")))
    for individual in (INST.a, INST.b):
        graph.add((individual, RDF.type, OWL.NamedIndividual))
        graph.add((individual, RDF.type, EX.Person))
    graph.add((INST.a, EX.likes, INST.b))
    graph.add((INST.a, EX.admires, INST.b))
    _roles, _categorized, mappings, _aliases, ambiguous = corrected(graph)
    context = render_abstract_context(graph, mappings)
    assert "same label" in ambiguous
    assert "Property1" in context and "Property2" in context


def test_final_sentence_deduplication_preserves_first_order():
    assert deduplicate_sentences(
        ["A relates to B. C relates to D.", "A relates to B. E relates to F."]
    ) == ["A relates to B.", "C relates to D.", "E relates to F."]


def test_query_mapping_resolves_dynamic_display_form():
    uri_map = {
        str(INST.MushroomTopping_dynamic_1): "Individual9",
        str(EX.isToppingOf): "Property8",
    }
    query = (
        f"ASK WHERE {{ <{INST.MushroomTopping_dynamic_1}> "
        f"<{EX.isToppingOf}> <{INST.AnchoviesTopping_dynamic_1}> }}"
    )
    aliases = query_text_mappings(query, uri_map)
    assert aliases["Mushroom Topping"] == "Individual9"
    assert aliases["Is Topping Of"] == "Property8"


def test_mapping_parser_separates_uri_identity_from_safe_aliases():
    lines = (
        "=== ONTOLOGY ABSTRACTION MAPPINGS ===\n"
        "<http://one.test#Same> -> <http://www.example.com/abstracted.owl#Class1>\n"
        "<http://two.test#Same> -> <http://www.example.com/abstracted.owl#Class2>\n"
        "=== TEXT ALIASES ===\n"
        "@alias Unique -> <http://www.example.com/abstracted.owl#Class3>\n"
    ).splitlines()
    assert parse_mapping_lines(lines) == {"Unique": "Class3"}
    assert parse_uri_mapping_lines(lines) == {
        "http://one.test#Same": "Class1",
        "http://two.test#Same": "Class2",
    }


def test_question_context_answer_share_one_mapping():
    graph = professor_graph()
    _roles, _categorized, mappings, aliases, _ambiguous = corrected(graph)
    identifiers, labels = source_terms([graph], mappings)
    question = "Which type contains AsparagusTopping_dynamic_1?"
    answer = "AsparagusTopping"
    abs_question = "Which type contains Individual1?"
    abs_answer = aliases[answer]
    result = validate_ar_row(
        original_question=question,
        original_answer=answer,
        answer_type="MC",
        sparql_query=f"SELECT ?x WHERE {{ <{INST.AsparagusTopping_dynamic_1}> a ?x }}",
        abs_question=abs_question,
        abs_context=render_abstract_context(graph, mappings),
        abs_answer=abs_answer,
        uri_mappings=mappings,
        text_mappings=aliases,
        identifiers=identifiers,
        lexical_labels=labels,
    )
    assert result.valid, result


def test_oeqa_abstract_answer_transforms_all_entities():
    aliases = {"AsparagusTopping": "Class1", "Parmense_0": "Individual2"}
    assert abstract_answer("AsparagusTopping; Parmense_0", "MC", aliases) == (
        "Class1; Individual2"
    )


def test_bqa_answer_is_not_transformed():
    assert abstract_answer("TRUE", "BIN", {"TRUE": "Class9"}) == "TRUE"


def test_normal_nl_label_behavior_is_unchanged():
    graph = Graph()
    graph.add((EX.Person, RDFS.label, Literal("Human Being")))
    assert get_nice_label(graph, EX.Person) == "Human Being"


def test_mapping_is_deterministic_across_graph_and_merge_order():
    first = professor_graph()
    second = Graph()
    second.add((EX.Other, RDF.type, OWL.Class))
    mappings_a = create_abstraction_mappings(
        merge_entity_roles([discover_entity_roles(first), discover_entity_roles(second)])
    )
    mappings_b = create_abstraction_mappings(
        merge_entity_roles([discover_entity_roles(second), discover_entity_roles(first)])
    )
    assert mappings_a == mappings_b


def test_incompatible_property_roles_fail_validation():
    graph = Graph()
    graph.add((EX.mixed, RDF.type, OWL.ObjectProperty))
    graph.add((INST.a, EX.mixed, INST.b))
    graph.add((INST.a, EX.mixed, Literal("literal")))
    try:
        discover_entity_roles(graph)
    except RoleConflictError as exc:
        assert "mixed" in str(exc)
    else:
        raise AssertionError("Expected explicit object/data role conflict")
