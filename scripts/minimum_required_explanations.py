#!/usr/bin/env python3
"""Compute the minimum required (conjunctive, de-duplicated) explanation set
for open-ended (OEQA / MC) questions.

An OEQA question's `Explanations` list conflates two different relationships:
  - disjunctive alternatives: multiple proofs of the SAME answer (redundant;
    keep only the cheapest, or a longer chain that subsumes it "for free")
  - conjunctive requirements: proofs of DIFFERENT answers (all mandatory,
    since a complete answer needs every distinct value justified)

This script groups explanations by the answer each one proves, detects
prefix-subsumption between class-hierarchy chains (a longer chain "for free"
proves every class it passes through), greedily selects a minimum-cost
covering set, and de-duplicates identical trailing axioms (e.g. a repeated
`rdfs:subPropertyOf` schema line) shared across the selected explanations.

This is a heuristic, not a guaranteed-optimal solver: it captures the two
redundancy patterns actually observed in this benchmark (prefix-subsumption
chains, shared schema axioms) rather than performing general DL entailment.

Tags reported here use the consolidated letter mapping from
tbox_axiom_census.py (propertyChainAxiom = "P", not the dataset's own raw
"N", which is reserved for owl:complementOf under that mapping). See
RELABEL below for the one-character translation this requires.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def strip_tag(explanation: list[str]) -> list[str]:
    return explanation[:-1] if explanation and explanation[-1].startswith("TAG:") else explanation


def match_answers(content: list[str], answers: list[str]) -> set[str]:
    """Which of the question's distinct answers does this explanation prove?

    Uses exact token matching (not substring containment): answer names like
    `TomatoTopping_dynamic_1` are literal substrings of unrelated longer
    names like `SlicedTomatoTopping_dynamic_1`, so naive `in` checks produce
    false positives.
    """
    tokens = {tok for line in content for tok in line.replace(",", " ").split()}
    return {a for a in answers if a in tokens}


def is_prefix(shorter: list[str], longer: list[str]) -> bool:
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


# The dataset's own raw tags use "N" for owl:propertyChainAxiom. Under the
# consolidated mapping (see tbox_axiom_census.py), propertyChainAxiom moved
# to "P" so that "N" is free for owl:complementOf. D/H/I/R/M/S/T are
# unchanged between the two schemes, so only "N" needs relabeling here.
RELABEL = str.maketrans({"N": "P"})


def get_tag(explanation: list[str]) -> str:
    raw = explanation[-1][4:] if explanation and explanation[-1].startswith("TAG:") else ""
    return raw.translate(RELABEL)


def refined_tag(chosen_with_tags: list[tuple[list[str], str]]) -> str:
    """Collapse chosen explanations into `D_N·<suffix>` clusters.

    A cluster of N chosen explanations that share the same tag AND the exact
    same trailing schema suffix (everything after the first, answer-specific
    line) collapses to `D_N·<suffix-tag>`: N independent direct facts sharing
    one reusable schema axiom. Clusters that don't share a suffix, or that
    only contain one explanation, keep their original tag unchanged.
    Distinct clusters are joined with `∧` (conjunctively required together).
    """
    if not chosen_with_tags:
        return ""
    if len(chosen_with_tags) == 1:
        return chosen_with_tags[0][1]

    clusters: dict[tuple[str, tuple[str, ...]], int] = {}
    order: list[tuple[str, tuple[str, ...]]] = []
    for content, tag in chosen_with_tags:
        key = (tag, tuple(content[1:]))
        if key not in clusters:
            order.append(key)
        clusters[key] = clusters.get(key, 0) + 1

    parts = []
    for tag, suffix in order:
        n = clusters[(tag, suffix)]
        if n == 1:
            parts.append(tag)
        else:
            suffix_tag = tag[1:] if tag.startswith("D") else tag
            parts.append(f"D_{n}·{suffix_tag}")
    return " ∧ ".join(parts)


def minimum_required(qa: dict[str, Any]) -> tuple[list[list[str]], int, int, str]:
    """Returns (chosen_explanations, naive_triple_total, minimum_triple_total, refined_tag)."""
    explanations = qa.get("Explanations") or []
    answers = [a.strip() for a in (qa.get("Answer") or "").split(";") if a.strip()]
    if not explanations or not answers:
        return [], 0, 0, ""

    contents = [strip_tag(e) for e in explanations]
    naive_total = sum(len(c) for c in contents)

    groups: dict[str, list[int]] = {a: [] for a in answers}
    for idx, content in enumerate(contents):
        for a in match_answers(content, answers):
            groups.setdefault(a, []).append(idx)

    # answers no explanation could be matched to (matching heuristic miss):
    # fall back to treating every explanation as covering every unmatched
    # answer's group so the algorithm still terminates conservatively.
    unmatched = [a for a, idxs in groups.items() if not idxs]
    for a in unmatched:
        groups[a] = list(range(len(contents)))

    remaining = set(groups.keys())
    chosen: list[int] = []
    while remaining:
        best_idx, best_covered, best_cost = None, set(), None
        for idx, content in enumerate(contents):
            covered = set()
            for a, idxs in groups.items():
                if a not in remaining:
                    continue
                if idx in idxs or any(is_prefix(contents[j], content) for j in idxs):
                    covered.add(a)
            if not covered:
                continue
            cost = len(content) / len(covered)
            if best_idx is None or cost < best_cost:
                best_idx, best_covered, best_cost = idx, covered, cost
        if best_idx is None:
            break
        chosen.append(best_idx)
        remaining -= best_covered

    chosen_contents = [contents[i] for i in chosen]
    chosen_tags = [get_tag(explanations[i]) for i in chosen]
    unique_triples: list[str] = []
    seen = set()
    for content in chosen_contents:
        for line in content:
            if line not in seen:
                seen.add(line)
                unique_triples.append(line)

    tag = refined_tag(list(zip(chosen_contents, chosen_tags)))
    return chosen_contents, naive_total, len(unique_triples), tag


def iter_mc_qas(groups: list[dict[str, Any]]):
    for group in groups:
        if group.get("Answer Type") != "MC":
            continue
        for qa in group["QAs"]:
            yield qa


def load_member(zip_path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read(member).decode("utf-8"))


DATASET_FILES = {
    "Family": ("FamilyOWL.zip", ["FamilyOWL_1hop.json", "FamilyOWL_2hop.json"]),
    "Pizza100": ("pizza_100.zip", ["pizza_100_1hop.json", "pizza_100_2hop.json"]),
    "Pizza250": ("pizza_250.zip", ["pizza_250_1hop.json", "pizza_250_2hop.json"]),
    "OWL2Bench": ("OWL2Bench.zip", ["OWL2Bench_1hop.json", "OWL2Bench_2hop.json"]),
}


def find_qa(benchmark_dir: Path, task_id: str) -> dict[str, Any] | None:
    for _dataset, (zip_name, members) in DATASET_FILES.items():
        for member in members:
            for group in load_member(benchmark_dir / zip_name, member):
                for qa in group.get("QAs", []):
                    if qa.get("Task ID") == task_id:
                        return qa
    return None


def print_examples(benchmark_dir: Path, task_ids: list[str]) -> None:
    for task_id in task_ids:
        qa = find_qa(benchmark_dir, task_id)
        if qa is None:
            print(f"{task_id}: NOT FOUND")
            continue
        chosen, naive, minimum, tag = minimum_required(qa)
        print(f"{task_id}")
        print(f"  naive={naive} min={minimum} explanations_kept={len(chosen)}  refined_tag={tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "final_benchmark")
    parser.add_argument(
        "--examples",
        nargs="+",
        metavar="TASK_ID",
        help="Print naive/min/refined-tag for specific Task IDs instead of the full aggregate report.",
    )
    args = parser.parse_args()

    if args.examples:
        print_examples(args.benchmark_dir, args.examples)
        return 0

    grand_naive = grand_min = grand_qas = 0
    for dataset, (zip_name, members) in DATASET_FILES.items():
        zip_path = args.benchmark_dir / zip_name
        naive_total = min_total = n_qas = n_reduced = 0
        for member in members:
            groups = load_member(zip_path, member)
            for qa in iter_mc_qas(groups):
                _, naive, minimum, _tag = minimum_required(qa)
                if naive == 0:
                    continue
                naive_total += naive
                min_total += minimum
                n_qas += 1
                if minimum < naive:
                    n_reduced += 1
        pct = 100 * (1 - min_total / naive_total) if naive_total else 0
        print(
            f"{dataset:10s} OEQA tasks={n_qas:5d}  naive_triples={naive_total:6d}  "
            f"min_triples={min_total:6d}  reduction={pct:5.1f}%  "
            f"tasks_with_redundancy={n_reduced}"
        )
        grand_naive += naive_total
        grand_min += min_total
        grand_qas += n_qas

    pct = 100 * (1 - grand_min / grand_naive) if grand_naive else 0
    print(
        f"\n{'TOTAL':10s} OEQA tasks={grand_qas:5d}  naive_triples={grand_naive:6d}  "
        f"min_triples={grand_min:6d}  reduction={pct:5.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
