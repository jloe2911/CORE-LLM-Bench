"""Deterministic normalization for ontology benchmark answers.

This module is deliberately independent of the LLM pipeline.  It operates only
on saved expected and actual answer strings, so metrics can be recalculated
offline without making model API calls.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import unquote


ANSWER_NORMALIZATION_VERSION = "lexical-v1"

_CAMEL_CASE_BOUNDARIES = (
    re.compile(r"(?<=[a-z0-9])(?=[A-Z])"),
    re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])"),
)
_DYNAMIC_SUFFIX = re.compile(
    r"(?:[\s_-]+dynamic)(?:[\s_-]*\d+)?$",
    flags=re.IGNORECASE,
)


def canonicalize_answer_item(value: Any) -> str:
    """Return a comparison key for one ontology answer item.

    The normalization is lexical rather than semantic: it makes common surface
    forms equivalent but does not infer class hierarchy relationships.
    """

    text = unquote(str(value).strip())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    # Only treat the value as a URI when it has a URI scheme.  This avoids
    # changing ordinary answers that happen to contain a slash.
    if "://" in text:
        text = text.rstrip(">").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        text = text.lstrip("<")

    for boundary in _CAMEL_CASE_BOUNDARIES:
        text = boundary.sub(" ", text)

    # Generated Pizza individuals use suffixes such as `_dynamic_1`.  The
    # suffix identifies a generated copy, not a distinct ontology label.
    text = _DYNAMIC_SUFFIX.sub("", text)

    text = text.casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_answer_items(value: Any) -> set[str]:
    """Split and normalize a semicolon- or comma-delimited answer."""

    normalized: set[str] = set()
    for item in re.split(r"[;,]", str(value)):
        canonical = canonicalize_answer_item(item)
        if canonical:
            normalized.add(canonical)
    return normalized


def normalized_jaccard_accuracy(
    expected: Any,
    actual: Any,
    answer_type: str,
) -> float:
    """Calculate normalized Jaccard accuracy using the benchmark BIN policy."""

    expected_set = normalize_answer_items(expected)
    actual_set = normalize_answer_items(actual)

    if not expected_set and not actual_set:
        score = 1.0
    elif not expected_set or not actual_set:
        score = 0.0
    else:
        intersection = len(expected_set.intersection(actual_set))
        union = len(expected_set.union(actual_set))
        score = intersection / union if union else 0.0

    if str(answer_type).strip().upper() == "BIN":
        return 1.0 if score == 1.0 else 0.0
    return score
