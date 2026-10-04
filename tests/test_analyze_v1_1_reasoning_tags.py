import math

import pandas as pd
import pytest

from scripts.analyze_v1_1_reasoning_tags import (
    _canonical_tags,
    _clustered_mean_ci,
    _proof_tags,
    _reconstruct_complete_minima,
)


def test_canonical_tags_are_primitive_and_stably_ordered() -> None:
    assert _canonical_tags(["R", "D", "H", "D"]) == "DHR"
    with pytest.raises(ValueError, match="Invalid primitive"):
        _canonical_tags(["D", "M"])


def test_proof_tags_use_axiom_level_final_schema() -> None:
    proof = {
        "axioms": [
            {"axiom": "a", "tag": "D"},
            {"axiom": "b", "tag": "I"},
            {"axiom": "c", "tag": "D"},
        ]
    }
    assert _proof_tags(proof) == {"D", "I"}


def test_clustered_interval_counts_pair_clusters_not_observations() -> None:
    frame = pd.DataFrame(
        {
            "answer_f1": [0.0, 0.0, 1.0, 1.0],
            "cluster_id": ["pair-a", "pair-a", "pair-b", "pair-b"],
        }
    )
    mean, se, low, high, clusters = _clustered_mean_ci(frame, "answer_f1")
    assert mean == 0.5
    assert clusters == 2
    assert se > 0
    assert low == 0.0
    assert high == 1.0


def test_single_cluster_interval_is_explicitly_undefined() -> None:
    frame = pd.DataFrame({"answer_f1": [0.25, 0.75], "cluster_id": ["one", "one"]})
    mean, se, low, high, clusters = _clustered_mean_ci(frame, "answer_f1")
    assert mean == 0.5
    assert clusters == 1
    assert math.isnan(se) and math.isnan(low) and math.isnan(high)


def test_corrected_answer_groups_reconstruct_and_deduplicate_complete_minima() -> None:
    groups = [
        {"alternatives": [{"axioms": [
            {"axiom": "a <p> b", "tag": "D"},
            {"axiom": "<p> SubPropertyOf: <q>"},
        ]}]},
        {"alternatives": [
            {"axioms": [
                {"axiom": "<p> SubPropertyOf: <q>"},
                {"axiom": "<q> Range <C>"},
            ]},
            {"axioms": [
                {"axiom": "<p> SubPropertyOf: <q>"},
                {"axiom": "<q> Range <C>"},
                {"axiom": "Symmetric: <q>"},
            ]},
        ]},
    ]

    minima, count = _reconstruct_complete_minima(groups)

    assert count == 3
    assert minima == [{"D", "H", "R"}]
