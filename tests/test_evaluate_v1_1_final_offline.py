from scripts.evaluate_v1_1_final_offline import hallucination_counts, parse_response, tags


def test_frozen_parser_retains_malformed_with_default_confidence():
    parsed = parse_response("ANSWER: FALSE\nCONFIDENCE:", "BQA")
    assert parsed["answer"] == "FALSE"
    assert parsed["confidence"] == 0.5
    assert parsed["status"] == "accepted_by_current_parser_with_confidence_default"


def test_empty_oeqa_prediction_has_no_generated_answer_denominator():
    assert hallucination_counts("", "Alice") == (0, 0, None)


def test_reasoning_tags_parse_json():
    assert tags('["D","I"]') == {"D", "I"}
