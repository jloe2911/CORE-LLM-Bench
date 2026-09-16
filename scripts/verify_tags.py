#!/usr/bin/env python3
"""Verify that every explanation's stored TAG correctly summarizes its own
axiom content, under the consolidated letter mapping (see
tbox_axiom_census.py and minimum_required_explanations.py's RELABEL).

Reconstructs an "expected" tag from the literal axiom lines in each
explanation (classifying each line by which DL construct it invokes) and
compares it against the dataset's own stored tag. This checks internal
consistency of the tagging, not whether the *reasoning itself* is correct.

Key finding this encodes: tags are NOT a literal trace of derivation order.
They group letters by a fixed canonical priority (D* < H < I < P < S < T < R,
then M appended once if 2+ distinct non-D types combine), regardless of the
order axioms were actually applied in the chain. See PRIORITY below.

Also provides simplify_tag() (enabled via --simplified), which corrects two
issues found in that raw scheme: H/I/P/S/T's redundant 2-units-per-use
doubling, and M's misuse as a "multiple simple axiom types" flag rather than
the nested-class-expression construct its taxonomy name implies.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimum_required_explanations import (  # noqa: E402
    DATASET_FILES,
    get_tag,
    load_member,
    strip_tag,
)

ROOT = Path(__file__).resolve().parents[1]

# Canonical grouping order for non-D letters. D always leads (one per direct
# fact); M is appended last, once, iff 2+ distinct non-D types are present.
PRIORITY = {"H": 1, "I": 2, "P": 3, "S": 4, "T": 5, "R": 6}

LINE_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("H", re.compile(r"SubClassOf"), 2),
    ("H", re.compile(r"rdfs:subPropertyOf"), 2),
    ("I", re.compile(r"owl:inverseOf"), 2),
    ("P", re.compile(r"PropertyChain|ObjectPropertyChain|SubObjectPropertyOf"), 2),
    ("S", re.compile(r"SymmetricObjectProperty"), 2),
    ("T", re.compile(r"TransitiveObjectProperty"), 2),
    ("R", re.compile(r"\bDomain\b|\bRange\b"), 1),
]


def classify_line(line: str) -> tuple[str, int]:
    for letter, pattern, weight in LINE_PATTERNS:
        if pattern.search(line):
            return letter, weight
    return "D", 1


def predict_tag(content: list[str]) -> str:
    d_count = 0
    groups: dict[str, str] = {}
    for line in content:
        letter, weight = classify_line(line)
        if letter == "D":
            d_count += 1
        else:
            groups[letter] = groups.get(letter, "") + letter * weight
    tag = "D" * d_count
    for letter in sorted(groups, key=lambda l: PRIORITY[l]):
        tag += groups[letter]
    if len(groups) >= 2:
        tag += "M"
    return tag


# Two fixes bundled into simplify_tag():
#
# 1. Doubling: H, I, P, S, T were each costing 2 tag-units per application
#    (verified with zero exceptions across 4,455+ explanations, scaling
#    exactly 2x per hop), while D and R cost 1. This was pure redundant
#    information — collapse every maximal run of a doubling letter to half
#    its length, bringing H/I/P/S/T in line with D/R's 1-unit convention.
#
# 2. M: per analysis/reasoning_coverage.py, M means "Multiple TBox axiom
#    types" — properly read, this is a complex/nested class expression
#    (multiple constructs, e.g. intersectionOf+someValuesFrom, composed
#    into ONE class definition), not "this chain happens to combine 2+
#    simple axiom types across separate steps." Checked: every one of the
#    1,351 M-tagged explanations in the dataset is the latter (a flat
#    chain of atomic H/I/P/R steps) — zero involve an actual nested class
#    expression (confirmed: zero intersectionOf/unionOf/complementOf/
#    someValuesFrom/cardinality anywhere in any explanation). So M is
#    dropped outright rather than relabeled: the composite information it
#    was carrying is already visible from the tag's letter diversity
#    itself (e.g. "DHR" already shows H and R both fired), so no
#    replacement letter is introduced. Under this fix M is reserved,
#    correctly, for a construct that never actually occurs in this
#    benchmark — consistent with E/C/¬/∩/O.
DOUBLING_LETTERS = set("HIPST")


def simplify_tag(tag: str) -> str:
    if tag.endswith("M"):
        tag = tag[:-1]
    out = []
    i = 0
    while i < len(tag):
        ch = tag[i]
        j = i
        while j < len(tag) and tag[j] == ch:
            j += 1
        run_len = j - i
        out.append(ch * (run_len // 2 if ch in DOUBLING_LETTERS else run_len))
        i = j
    return "".join(out)


def iter_all_explanations(item: dict[str, Any], qa: dict[str, Any]):
    """Yield every explanation for this QA (OEQA's list, or BIN's single one)."""
    exps = qa.get("Explanations")
    if exps:
        yield from exps
        return
    me = qa.get("Minimum Explanation")
    if me:
        yield me


def verify_dataset(
    zip_path: Path,
    members: list[str],
    answer_type: str | None,
    positive_only: bool,
    simplified: bool = False,
) -> tuple[int, int, list[tuple[str, str, list[str], str, str]]]:
    total = matches = 0
    mismatches: list[tuple[str, str, list[str], str, str]] = []
    for member in members:
        groups = load_member(zip_path, member)
        for item in groups:
            if answer_type and item.get("Answer Type") != answer_type:
                continue
            for qa in item["QAs"]:
                if positive_only and qa.get("Answer") != "TRUE":
                    continue
                for exp in iter_all_explanations(item, qa):
                    content = strip_tag(exp)
                    actual = get_tag(exp)
                    predicted = predict_tag(content)
                    if simplified:
                        actual = simplify_tag(actual)
                        predicted = simplify_tag(predicted)
                    total += 1
                    if predicted == actual:
                        matches += 1
                    else:
                        mismatches.append((member, qa["Task ID"], content, actual, predicted))
    return total, matches, mismatches


def sample_examples(
    zip_path: Path,
    members: list[str],
    answer_type: str | None,
    positive_only: bool,
    n: int,
    rng: random.Random,
    simplified: bool = False,
) -> list[tuple[str, list[str], str, str]]:
    pool: list[tuple[str, list[str], str, str]] = []
    for member in members:
        groups = load_member(zip_path, member)
        for item in groups:
            if answer_type and item.get("Answer Type") != answer_type:
                continue
            for qa in item["QAs"]:
                if positive_only and qa.get("Answer") != "TRUE":
                    continue
                exps = list(iter_all_explanations(item, qa))
                if not exps:
                    continue
                exp = exps[0]
                content = strip_tag(exp)
                actual, predicted = get_tag(exp), predict_tag(content)
                if simplified:
                    actual, predicted = simplify_tag(actual), simplify_tag(predicted)
                pool.append((qa["Task ID"], content, actual, predicted))
    return rng.sample(pool, min(n, len(pool)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "final_benchmark")
    parser.add_argument("--answer-type", choices=["BIN", "MC"], help="Restrict to BIN or MC questions")
    parser.add_argument("--positive-only", action="store_true", help="Restrict to Answer == TRUE")
    parser.add_argument("--sample", type=int, default=0, help="Print N random examples per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Apply simplify_tag(): collapse H/I/P/S/T runs to 1 unit per "
        "application (D and R already cost 1), and drop the M flag (it "
        "never marks a genuine nested class expression in this dataset — "
        "see simplify_tag's docstring).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grand_total = grand_match = 0
    for dataset, (zip_name, members) in DATASET_FILES.items():
        zip_path = args.benchmark_dir / zip_name

        if args.sample:
            print(f"========== {dataset} ==========")
            for tid, content, actual, predicted in sample_examples(
                zip_path, members, args.answer_type, args.positive_only, args.sample, rng, args.simplified
            ):
                status = "OK" if predicted == actual else "MISMATCH"
                print(f"[{status}] {tid}")
                for line in content:
                    print(f"    {line}")
                print(f"    actual={actual}  predicted={predicted}\n")

        total, matches, mismatches = verify_dataset(
            zip_path, members, args.answer_type, args.positive_only, args.simplified
        )
        grand_total += total
        grand_match += matches
        pct = 100 * matches / total if total else 0
        print(f"{dataset:10s} explanations={total:6d}  matches={matches:6d}  ({pct:.2f}%)")
        for member, tid, content, actual, predicted in mismatches[:5]:
            print(f"  MISMATCH  {member} {tid}")
            for line in content:
                print(f"    {line}")
            print(f"    actual={actual}  predicted={predicted}")

    pct = 100 * grand_match / grand_total if grand_total else 0
    print(f"\n{'TOTAL':10s} explanations={grand_total:6d}  matches={grand_match:6d}  ({pct:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
