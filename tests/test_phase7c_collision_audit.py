import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from scripts.phase7c_collision_audit import (
    build_dataset_inventory,
    collision_safe_context,
    collision_safe_mapping,
    collision_safe_question,
    evaluated_hash,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "release" / "v1.1.0-staging" / "core_llm_bench_v1_1.parquet"
PHASE7C = ROOT / "release" / "v1.1.0-phase7c-audit"


def test_collision_mapping_preserves_unique_and_restores_numeric_identity():
    candidates = {
        "http://example.org/Employee_10": "Employee",
        "http://example.org/Employee_47": "Employee",
        "http://example.org/Manager": "Manager",
    }
    mapping = collision_safe_mapping(candidates)
    assert mapping["http://example.org/Employee_10"] == "Employee 10"
    assert mapping["http://example.org/Employee_47"] == "Employee 47"
    assert mapping["http://example.org/Manager"] == "Manager"
    assert len(set(mapping.values())) == len(mapping)
    assert mapping == collision_safe_mapping(dict(reversed(list(candidates.items()))))


def test_identical_local_names_use_compact_namespace_identity():
    candidates = {
        "http://www.example.com/genealogy.owl#teachesCourse": "Teaches Course",
        "https://kracr.iiitd.edu.in/OWL2Bench#teachesCourse": "Teaches Course",
        "http://www.example.com/genealogy.owl#Employee_10": "Employee",
        "https://kracr.iiitd.edu.in/OWL2Bench#Employee_10": "Employee",
    }
    mapping = collision_safe_mapping(candidates)
    assert mapping["http://www.example.com/genealogy.owl#teachesCourse"] == "Teaches Course (Genealogy)"
    assert mapping["https://kracr.iiitd.edu.in/OWL2Bench#teachesCourse"] == "Teaches Course (Owl2bench)"
    assert mapping["http://www.example.com/genealogy.owl#Employee_10"] == "Employee 10 (Genealogy)"
    assert mapping["https://kracr.iiitd.edu.in/OWL2Bench#Employee_10"] == "Employee 10 (Owl2bench)"
    assert all("http" not in label for label in mapping.values())


def test_question_and_context_keep_formal_entities_distinct_without_gold_leakage():
    graph = Graph()
    teaches = URIRef("http://example.org/teaches")
    course = URIRef("http://example.org/Course_1")
    employee10 = URIRef("http://example.org/Employee_10")
    employee47 = URIRef("http://example.org/Employee_47")
    graph.add((teaches, RDF.type, OWL.ObjectProperty))
    graph.add((course, teaches, employee10))
    graph.add((employee47, teaches, course))
    candidates = {
        str(teaches): "Teaches",
        str(course): "Course",
        str(employee10): "Employee",
        str(employee47): "Employee",
    }
    mapping = collision_safe_mapping(candidates)
    positive = collision_safe_question(
        "Is Course related to Employee through Teaches?",
        "ASK WHERE { <http://example.org/Course_1> <http://example.org/teaches> <http://example.org/Employee_10> }",
        candidates,
        mapping,
    )
    negative = collision_safe_question(
        "Is Course related to Employee through Teaches?",
        "ASK WHERE { <http://example.org/Course_1> <http://example.org/teaches> <http://example.org/Employee_47> }",
        candidates,
        mapping,
    )
    context = collision_safe_context(graph, mapping)
    assert positive != negative
    assert "Employee 10" in positive and "Employee 47" in negative
    assert "Employee 10" in context and "Employee 47" in context
    assert "TRUE" not in positive + negative + context
    assert "FALSE" not in positive + negative + context


def test_question_rendering_does_not_replace_object_surface_inside_predicate():
    candidates = {
        "http://example.org/Person_58": "Person",
        "http://example.org/teachesCourse": "Teaches Course",
        "http://example.org/Course_14": "Course",
        "http://example.org/Course_19": "Course",
    }
    mapping = collision_safe_mapping(candidates)
    rendered = collision_safe_question(
        "Is Person related to Course through Teaches Course?",
        "ASK WHERE { <http://example.org/Person_58> <http://example.org/teachesCourse> <http://example.org/Course_14> }",
        candidates,
        mapping,
    )
    assert rendered == "Is Person related to Course 14 through Teaches Course?"


def test_same_rendered_nl_with_different_gold_has_same_hash_for_audit_gate():
    base = {
        "task_group": "BQA",
        "answer_type": "BIN",
        "nl_question": "Is A related to B?",
        "nl_context": "A is related to B.",
        "fs_query": "ASK WHERE { <a> <p> <b> }",
        "fs_context": "<a> <p> <b> .",
        "ar_question": "Is Individual0 related to Individual1?",
        "ar_context": "Individual0 Property0 Individual1.",
    }
    assert evaluated_hash({**base, "gold_answer": "TRUE"}, "NL") == evaluated_hash(
        {**base, "gold_answer": "FALSE"}, "NL"
    )


def test_query_only_negative_target_participates_in_dataset_collision_inventory(monkeypatch):
    graph = Graph().parse(
        data="<http://example.org/Employee_10> a <http://www.w3.org/2002/07/owl#NamedIndividual> .",
        format="turtle",
    )
    monkeypatch.setattr(Graph, "parse", lambda self, *_args, **_kwargs: graph)
    mapping, _rows = build_dataset_inventory(
        "Toy",
        [object()],
        {"http://example.org/Employee_10", "http://example.org/Employee_47"},
    )
    assert mapping["http://example.org/Employee_10"] == "Employee 10"
    assert mapping["http://example.org/Employee_47"] == "Employee 47"


def test_phase7c_exact_fatal_fixtures_all_resolve():
    report = json.loads((PHASE7C / "phase7c_audit_report.json").read_text(encoding="utf-8"))
    outcomes = report["phase7b_conflict_outcomes"]
    assert len(outcomes) == 9
    assert all(row["outcome"] == "resolved" for row in outcomes)
    assert sum(row["task_group"] == "BQA" for row in outcomes) == 4
    assert sum(row["task_group"] == "OEQA" and row["datasets"] == "FamilyOWL" for row in outcomes) == 3
    assert sum(row["task_group"] == "OEQA" and row["datasets"] == "Pizza100;Pizza250" for row in outcomes) == 2
    assert report["post_fix"]["NL"]["incompatible_groups"] == 0
    assert report["post_fix"]["FS"]["incompatible_groups"] == 0
    assert report["post_fix"]["AR"]["incompatible_groups"] == 0


def test_phase7c_preserves_gold_fs_ar_and_semantic_membership():
    protected = [
        "task_id", "semantic_key", "gold_answer", "fs_query", "fs_context",
        "ar_question", "ar_context", "ar_gold_answer", "sampling_group_key",
    ]
    before = pq.read_table(PHASE6, columns=protected)
    phase7c_parquet = PHASE7C / "core_llm_bench_v1_1_phase7c.parquet"
    if not phase7c_parquet.exists():
        pointer = (
            ROOT / "release" / "v1.1.0-staging" / "LARGE_ARTIFACTS.md"
        ).read_text(encoding="utf-8")
        assert "core_llm_bench_v1_1_phase7c.parquet" in pointer
        assert "be75f246fcd2ea54ce5fb82411da9a61393cdc3c4ffba0ac12e38b8a084d7431" in pointer
        pytest.skip("Phase 7C parquet is intentionally external to the review branch")
    after = pq.read_table(phase7c_parquet, columns=protected)
    assert before.equals(after)
    assert before.num_rows == 9048


def test_phase7c_equivalence_group_ids_are_stable_and_compatible():
    with (PHASE7C / "input_equivalence_groups.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    ids = [row["input_equivalence_group_id"] for row in groups]
    assert len(ids) == len(set(ids))
    assert all(group_id.startswith("ieq-") for group_id in ids)
    assert all(row["classification_category"] != "1" for row in groups)
    assert all(row["identical_gold_semantics"] for row in groups)


def test_phase7c_mapping_excludes_ontology_headers_and_full_uri_surfaces():
    with (PHASE7C / "entity_label_mapping.csv").open(encoding="utf-8", newline="") as handle:
        mappings = list(csv.DictReader(handle))
    assert not any("Owl2bench Individual Subgraph" in row["candidate_normalized_label"] for row in mappings)
    assert not any("http://" in row["final_nl_label"] or "https://" in row["final_nl_label"] for row in mappings)
