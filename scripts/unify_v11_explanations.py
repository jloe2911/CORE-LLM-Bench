#!/usr/bin/env python3
"""Transform v1.1-candidate benchmark records into the unified explanation
schema proposed after auditing review/v1.1-benchmark-candidate.

Collapses `explanations` / `structured_explanations` / `legacy_explanation_fields`
into `answer_explanations` -- the OEQA source field's own name, now used
identically for BQA too, so there is exactly one explanation field instead of
three -- renders each axiom as a single Manchester-Syntax string
(replacing the `identity`/`rendering` split), preserves true derivation
order in `tag_sequence` (the source `TAG:` suffix is canonically re-sorted
and is not used here), and drops `M` entirely.

Reuses the axiom-classification approach from verify_tags.py (adapted:
v1.1's own tags are not doubled, so each construct costs exactly 1 unit)
and the subsumption/dedup approach from minimum_required_explanations.py
to compute `complete_explanation`.

Does not touch the v1.1 branch's own Java pipeline or benchmark files --
this reads the extracted JSON and writes a new, separate transformed copy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

V11_FILES = [
    "FamilyOWL_1hop.json",
    "OWL2Bench_1hop.json",
    "OWL2Bench_2hop.json",
    "Pizza100_1hop.json",
    "Pizza100_2hop.json",
    "Pizza250_1hop.json",
    "Pizza250_2hop.json",
]

# --- axiom classification (v1.1's own tags are not doubled: 1 unit/use) ---
LINE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("H", re.compile(r"SubClassOf|rdfs:subPropertyOf")),
    ("I", re.compile(r"owl:inverseOf")),
    ("N", re.compile(r"PropertyChain|ObjectPropertyChain|SubObjectPropertyOf")),
    ("S", re.compile(r"SymmetricObjectProperty")),
    ("T", re.compile(r"TransitiveObjectProperty")),
    ("R", re.compile(r"\bDomain\b|\bRange\b")),
]


def classify_line(line: str) -> str:
    for letter, pattern in LINE_PATTERNS:
        if pattern.search(line):
            return letter
    return "D"


# --- best-effort Manchester Syntax rendering (see proposal doc's open
# question: this needs a pinned renderer for byte-exact output; this is a
# deterministic approximation from the existing rendered text) ---
CHAIN_RE = re.compile(
    r"(?:PropertyChain\(|ObjectPropertyChain[(<]).*?"
)


def to_manchester(line: str) -> str:
    if " rdf:type " in line:
        subj, obj = line.split(" rdf:type ")
        return f"{subj} Type: {obj}"
    if " SubClassOf " in line:
        subj, obj = line.split(" SubClassOf ")
        return f"{subj} SubClassOf: {obj}"
    if " rdfs:subPropertyOf " in line:
        subj, obj = line.split(" rdfs:subPropertyOf ")
        return f"{subj} SubPropertyOf: {obj}"
    if " owl:inverseOf " in line:
        subj, obj = line.split(" owl:inverseOf ")
        return f"{subj} InverseOf: {obj}"
    if " Domain " in line:
        subj, obj = line.split(" Domain ")
        return f"{subj} Domain: {obj}"
    if " Range " in line:
        subj, obj = line.split(" Range ")
        return f"{subj} Range: {obj}"
    m = re.match(r"SymmetricObjectProperty\((.+)\)", line)
    if m:
        return f"{m.group(1)} Characteristics: Symmetric"
    m = re.match(r"TransitiveObjectProperty\((.+)\)", line)
    if m:
        return f"{m.group(1)} Characteristics: Transitive"
    m = re.search(
        r"ObjectPropertyChain\(<[^#]*#([^>]+)>\s*<[^#]*#([^>]+)>\)\s*<[^#]*#([^>]+)>",
        line,
    )
    if m:
        p1, p2, sup = m.groups()
        return f"{p1} o {p2} SubPropertyOf: {sup}"
    if line.startswith("PropertyChain("):
        # The source data's PropertyChain(...) rendering has mojibake
        # composition/subsumption symbols in some files ("∘"/"∘" and
        # "⊑" show up re-encoded as "âˆ˘"/"âŠ‘") --
        # match either the correct codepoint or the corrupted byte sequence.
        m2 = re.match(
            r"PropertyChain\((.+?)\s*(?:∘|âˆ˜)\s*(.+?)\)\s*(?:⊑|âŠ‘)\s*(.+)",
            line,
        )
        if m2:
            p1, p2, sup = m2.groups()
            return f"{p1} o {p2} SubPropertyOf: {sup}"
    # already a plain fact assertion ("X property Y") -- Manchester-compatible as-is
    return line


def strip_tag(exp: list[str]) -> list[str]:
    return exp[:-1] if exp and exp[-1].startswith("TAG:") else exp


def to_axioms(content: list[str]) -> tuple[list[dict[str, str]], str]:
    axioms = [{"axiom": to_manchester(line), "tag": classify_line(line)} for line in content]
    tag_sequence = "".join(a["tag"] for a in axioms)
    return axioms, tag_sequence


def is_prefix(shorter: list[str], longer: list[str]) -> bool:
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def merge_grouped(blocks: list[list[str]]) -> list[str]:
    """Merge multiple independent proof blocks (one per answer group) by
    grouping same-tag axioms together, in order of each tag's first
    appearance across the merged sequence -- e.g. two independent DIR
    proofs merge to DDIIRR, not the arbitrary DIRDIR from concatenation.

    A single block's own internal order is a real causal chain and is
    preserved: this only reorders *across* blocks, and with one block it is
    a no-op (each line's bucket is visited in the order the block already
    has it, since a block never revisits a tag out of sequence)."""
    letter_order: list[str] = []
    for block in blocks:
        for line in block:
            letter = classify_line(line)
            if letter not in letter_order:
                letter_order.append(letter)

    buckets: dict[str, list[str]] = {letter: [] for letter in letter_order}
    for block in blocks:
        for line in block:
            buckets[classify_line(line)].append(line)

    return [line for letter in letter_order for line in buckets[letter]]


def build_answer_explanations(raw_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """raw_groups: list of {"Answer": str, "Alternatives": [[...], ...]}"""
    out = []
    for g in raw_groups:
        alt_contents = [strip_tag(alt) for alt in g["Alternatives"]]
        alternatives = []
        for content in alt_contents:
            axioms, tag_seq = to_axioms(content)
            alternatives.append({"axioms": axioms, "tag_sequence": tag_seq, "axiom_count": len(axioms)})
        counts = [a["axiom_count"] for a in alternatives]
        out.append(
            {
                "answer": g["Answer"],
                "alternatives": alternatives,
                "alternative_count": len(alternatives),
                "min_proof_axiom_count": min(counts),
                "max_proof_axiom_count": max(counts),
            }
        )
    return out


def complete_explanation(answer_explanations: list[dict[str, Any]], raw_alt_contents_by_group: list[list[list[str]]]) -> dict[str, Any]:
    if not answer_explanations:
        return {
            "combination_count": 0,
            "min_axiom_count": None,
            "max_axiom_count": None,
            "minimum_explanations": [],
        }

    combination_count = 1
    for g in answer_explanations:
        combination_count *= g["alternative_count"]

    # greedy pick: reuse subsumption logic -- an alternative "for free" covers
    # any other alternative (in any group) that is a prefix of it.
    all_contents = [c for group in raw_alt_contents_by_group for c in group]
    group_of = []
    for gi, group in enumerate(raw_alt_contents_by_group):
        group_of.extend([gi] * len(group))

    def greedy_pick(prefer_shortest: bool) -> list[tuple[int, set[int]]]:
        """Greedily pick one alternative per answer group (crediting prefix
        subsumption "for free"). Returns (alt_index, covered_group_indices)
        pairs -- covered includes the alt's own group plus any other group
        it subsumes for free."""
        remaining = set(range(len(answer_explanations)))
        picks: list[tuple[int, set[int]]] = []
        while remaining:
            best_idx, best_covered, best_cost = None, set(), None
            for idx, content in enumerate(all_contents):
                covered = {group_of[idx]}
                for j, other in enumerate(all_contents):
                    if group_of[j] in remaining and is_prefix(other, content):
                        covered.add(group_of[j])
                covered &= remaining
                if not covered:
                    continue
                cost = len(content) / len(covered)
                if best_idx is None or (cost < best_cost if prefer_shortest else cost > best_cost):
                    best_idx, best_covered, best_cost = idx, covered, cost
            if best_idx is None:
                break
            picks.append((best_idx, best_covered))
            remaining -= best_covered
        return picks

    min_picks = greedy_pick(prefer_shortest=True)

    # deduplicated total axiom count for the minimum (a plain count, invariant
    # across which tied alternative is chosen for any given contribution).
    seen: set[str] = set()
    blocks: list[list[str]] = []
    for idx, _ in min_picks:
        block = [line for line in all_contents[idx] if line not in seen]
        seen.update(block)
        if block:
            blocks.append(block)
    min_axiom_count = sum(len(b) for b in blocks)

    # max_axiom_count as a plain count only -- no detailed breakdown, since a
    # single "the" maximum explanation has the same arbitrary-tie problem the
    # minimum one did, without the compensating value of showing what's cheap.
    max_picks = greedy_pick(prefer_shortest=False)
    seen_max: set[str] = set()
    max_axiom_count = 0
    for idx, _ in max_picks:
        for line in all_contents[idx]:
            if line not in seen_max:
                seen_max.add(line)
                max_axiom_count += 1

    # Every group actually picked "for its own sake" (not merely covered for
    # free by another group's choice) may have several alternatives tied at
    # its own minimal cost -- e.g. task 7041's "Person" answer has 6 same-
    # cost routes, "Student" has 3, none of them the "true" single minimum.
    # Rather than silently keeping whichever the greedy loop lands on (see
    # the tag-distribution audit: 100% of ~93 similar tasks always favored
    # "Type: X" over "hasMajor ...", "isAdvisedBy ...", purely from
    # iteration order) or nesting per-group ties in a new shape, enumerate
    # every combination across the directly-picked groups and emit each as
    # its own minimum-explanation entry, in the original flat
    # {axioms, tag_sequence} shape -- just as a list instead of a single one.
    tied_per_pick: list[list[list[str]]] = []
    for idx, _covered in min_picks:
        own_group_idx = group_of[idx]
        own_length = len(all_contents[idx])
        own_group_alts = raw_alt_contents_by_group[own_group_idx]
        tied_per_pick.append([alt for alt in own_group_alts if len(alt) == own_length])

    minimum_explanations = []
    seen_keys: set[tuple[str, ...]] = set()
    for combo in itertools.product(*tied_per_pick):
        seen_lines: set[str] = set()
        combo_blocks: list[list[str]] = []
        for alt in combo:
            block = [line for line in alt if line not in seen_lines]
            seen_lines.update(block)
            if block:
                combo_blocks.append(block)
        axioms, tag_seq = to_axioms(merge_grouped(combo_blocks))
        key = tuple(a["axiom"] for a in axioms)
        if key not in seen_keys:
            seen_keys.add(key)
            minimum_explanations.append({"axioms": axioms, "tag_sequence": tag_seq})

    return {
        "combination_count": combination_count,
        "min_axiom_count": min_axiom_count,
        "max_axiom_count": max_axiom_count,
        "minimum_explanations": minimum_explanations,
    }


def transform_qa(qa: dict[str, Any]) -> dict[str, Any]:
    if qa.get("answer_type") == "BIN":
        if qa.get("gold_answer") == "TRUE":
            raw_groups = [{"Answer": "TRUE", "Alternatives": qa.get("explanations") or []}]
        else:
            raw_groups = []
    else:
        raw_groups = [
            {"Answer": g["Answer"], "Alternatives": g["Alternatives"]}
            for g in (qa.get("answer_explanations") or [])
        ]

    answer_explanations = build_answer_explanations(raw_groups)
    raw_contents = [[strip_tag(alt) for alt in g["Alternatives"]] for g in raw_groups]

    return {
        "task_id": qa["task_id"],
        "task_type": qa["task_type"],
        "nl_question": qa["nl_question"],
        "formal_query": qa["formal_query"],
        "gold_answer": qa["gold_answer"],
        "answer_explanations": answer_explanations,
        "complete_explanation": complete_explanation(answer_explanations, raw_contents),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v11-dir",
        type=Path,
        required=True,
        help="Directory containing the extracted v1.1 benchmark JSON files "
        "(FamilyOWL_1hop.json, OWL2Bench_1hop.json, ...)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_benchmark" / "v1.1_unified_schema")
    parser.add_argument("--task-id", type=int, help="Transform and print just this one task, don't write files")
    args = parser.parse_args()

    if args.task_id is not None:
        for fn in V11_FILES:
            path = args.v11_dir / fn
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            for item in data:
                for qa in item["QAs"]:
                    if qa["task_id"] == args.task_id:
                        print(json.dumps(transform_qa(qa), indent=2))
                        return 0
        print(f"task_id {args.task_id} not found")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fn in V11_FILES:
        path = args.v11_dir / fn
        if not path.is_file():
            print(f"skip (not found): {fn}")
            continue
        data = json.loads(path.read_text())
        transformed = []
        for item in data:
            for qa in item["QAs"]:
                transformed.append(transform_qa(qa))
        out_path = args.output_dir / fn
        out_path.write_text(json.dumps(transformed, indent=2), encoding="utf-8")
        print(f"{fn}: {len(transformed)} questions -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
