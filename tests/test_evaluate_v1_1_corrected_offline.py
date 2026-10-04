from __future__ import annotations

import math

import pytest

from scripts.evaluate_v1_1_final_offline import (
    gold_for_representation,
    hallucination_counts,
    parse_response,
)


def test_ar_selects_representation_specific_gold() -> None:
    question = {"gold_answer": "OriginalEntity", "ar_gold_answer": "Individual42"}

    assert gold_for_representation(question, "AR") == "Individual42"
    assert gold_for_representation(question, "NL") == "OriginalEntity"
    assert gold_for_representation(question, "FS") == "OriginalEntity"


@pytest.mark.parametrize("representation", ["NL", "FS", "AR"])
def test_explicit_blank_answer_does_not_capture_confidence(representation: str) -> None:
    parsed = parse_response("ANSWER:   \nCONFIDENCE: 1.0", "OEQA")

    assert representation in {"NL", "FS", "AR"}  # documents all affected paths
    assert parsed["answer"] == ""
    assert parsed["status"] == "explicit_blank_answer"
    assert parsed["usable"] is False
    assert parsed["confidence"] == 1.0


def test_nonblank_answer_still_parses_on_its_own_line() -> None:
    parsed = parse_response("ANSWER:\tIndividual1; Individual2\r\nCONFIDENCE: 0.8", "OEQA")

    assert parsed["answer"] == "Individual1; Individual2"
    assert parsed["status"] == "requested_schema_conformant"


def test_hallucination_uses_complete_gold_set_without_substrings() -> None:
    unsupported, generated, rate = hallucination_counts(
        "Individual1; Individual2; Individual20",
        "Individual1; Individual2; Individual3",
    )

    assert unsupported == 1
    assert generated == 3
    assert math.isclose(rate or 0.0, 1 / 3)


def test_empty_prediction_has_explicit_zero_denominator_and_no_rate() -> None:
    unsupported, generated, rate = hallucination_counts("", "Individual1; Individual2")

    assert unsupported == 0
    assert generated == 0
    assert rate is None
