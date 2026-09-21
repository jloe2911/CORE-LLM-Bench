from scripts.phase7b_duplicate_audit import (
    classify_compatible,
    classify_incompatible,
    query_parts,
)


def test_query_parts_handles_ask_and_select():
    assert query_parts("ASK WHERE { <s> <p> <o> }") == ("s", "p", "o")
    assert query_parts("SELECT ?x WHERE { <s> <p> ?x }") == ("s", "p", "")


def test_bqa_collision_is_pair_level_negative_construction_collision():
    rows = [{"task_group": "BQA"}, {"task_group": "BQA"}]
    category, _, correction = classify_incompatible(rows)
    assert category.startswith("5 ")
    assert "complete BQA pair" in correction


def test_compatible_classification_distinguishes_semantics_and_representation():
    exact = [
        {"dataset": "OWL2Bench", "formal_query": "ASK WHERE { <s> <p> <o> }"},
        {"dataset": "OWL2Bench", "formal_query": "ASK WHERE { <s> <p> <o> }"},
    ]
    equivalent = [
        {"dataset": "Pizza100", "formal_query": "SELECT ?x WHERE { <s> <p> ?x }"},
        {"dataset": "Pizza250", "formal_query": "SELECT ?x WHERE { <s> <p> ?x }"},
    ]
    collapsed = [
        {"dataset": "FamilyOWL", "formal_query": "SELECT ?x WHERE { <s1> <p> ?x }"},
        {"dataset": "FamilyOWL", "formal_query": "SELECT ?x WHERE { <s2> <p> ?x }"},
    ]
    assert classify_compatible("AR", exact)[0] == "B"
    assert classify_compatible("FS", equivalent)[0] == "D"
    assert classify_compatible("NL", collapsed)[0] == "C"
