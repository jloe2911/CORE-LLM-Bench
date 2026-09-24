"""Canonical public v1.1 explanation schema built from OWLAPI semantics.

Semantic axiom identities are private construction data. Public axioms are
deterministic OWLAPI Manchester renderings, and ``tag_sequence`` records the
primitive tag corresponding to each axiom in stored axiom order. It is not a
causal, derivation, or reasoner firing order.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable


PUBLIC_TAGS = frozenset("DHIRSTN")


def _proof(
    raw: dict[str, Any], render_axiom: Callable[[str], str]
) -> dict[str, Any]:
    seen: set[str] = set()
    private_axioms: list[tuple[str, str, str]] = []
    for item in raw.get("semanticAxioms", []):
        identity = str(item["identity"])
        tags = [str(tag) for tag in item.get("primitiveTags", [])]
        if identity in seen:
            continue
        if len(tags) != 1 or tags[0] not in PUBLIC_TAGS:
            raise ValueError(
                f"Unsupported public primitive tag assignment {tags!r} for {identity}"
            )
        seen.add(identity)
        private_axioms.append((identity, render_axiom(identity), tags[0]))
    if not private_axioms:
        raise ValueError("Explanation alternative has no semantic OWLAPI axioms")
    return _public_proof(private_axioms, include_count=True)


def _public_proof(
    private_axioms: list[tuple[str, str, str]], *, include_count: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "axioms": [
            {"axiom": rendering, "tag": tag}
            for _identity, rendering, tag in private_axioms
        ],
        "tag_sequence": "".join(tag for _identity, _rendering, tag in private_axioms),
        "_semantic_identities": [identity for identity, _rendering, _tag in private_axioms],
    }
    if include_count:
        result["axiom_count"] = len(private_axioms)
    return result


def _deduplicate_proofs(proofs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for proof in proofs:
        key = tuple(sorted(proof["_semantic_identities"]))
        unique.setdefault(key, proof)
    return list(unique.values())


def answer_group(
    answer: str,
    structured_proofs: Iterable[dict[str, Any]],
    render_axiom: Callable[[str], str],
    **metadata: Any,
) -> dict[str, Any]:
    alternatives = _deduplicate_proofs(
        _proof(raw, render_axiom) for raw in structured_proofs
    )
    if not alternatives:
        raise ValueError(f"Answer {answer!r} has no explanation alternatives")
    counts = [alternative["axiom_count"] for alternative in alternatives]
    return {
        "answer": answer,
        **metadata,
        "alternatives": alternatives,
        "alternative_count": len(alternatives),
        "min_proof_axiom_count": min(counts),
        "max_proof_axiom_count": max(counts),
    }


def complete_explanation(answer_groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not answer_groups:
        return {
            "combination_count": 0,
            "min_axiom_count": None,
            "max_axiom_count": None,
            "minimum_explanations": [],
        }
    selections = product(*(group["alternatives"] for group in answer_groups))
    union_by_key: dict[tuple[str, ...], list[tuple[str, str, str]]] = {}
    combination_count = 1
    for group in answer_groups:
        combination_count *= len(group["alternatives"])
    all_counts: list[int] = []
    for selection in selections:
        ordered: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for proof in selection:
            for identity, item in zip(proof["_semantic_identities"], proof["axioms"]):
                if identity not in seen:
                    seen.add(identity)
                    ordered.append((identity, item["axiom"], item["tag"]))
        key = tuple(sorted(seen))
        all_counts.append(len(key))
        union_by_key.setdefault(key, ordered)
    minimum = min(all_counts)
    minima = [
        _public_proof(union_by_key[key], include_count=False)
        for key in sorted(union_by_key)
        if len(key) == minimum
    ]
    return {
        "combination_count": combination_count,
        "min_axiom_count": minimum,
        "max_axiom_count": max(all_counts),
        "minimum_explanations": minima,
    }


def strip_private_semantic_identities(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_private_semantic_identities(item)
            for key, item in value.items()
            if key != "_semantic_identities"
        }
    if isinstance(value, list):
        return [strip_private_semantic_identities(item) for item in value]
    return value
