#!/usr/bin/env python3
"""Census which DL constructs each ontology's TBox actually uses, tagged
against the 20-type reasoning taxonomy in analysis/reasoning_coverage.py.

The `OWL Context` field on each benchmark item embeds the full ontology
Turtle serialization (TBox + a locally relevant ABox slice), so no single
item is guaranteed to carry every TBox declaration verbatim in one place.
This script unions class/property definition blocks across many items per
dataset to reconstruct the complete TBox, then classifies each block by
which taxonomy construct(s) it uses.

This complements minimum_required_explanations.py: that script measures
what reasoning the generated *explanations* actually exercise; this one
measures what reasoning the *ontology itself* defines. The gap between the
two is the real finding.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimum_required_explanations import DATASET_FILES, load_member  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TBOX_TYPES = ("owl:Class", "owl:ObjectProperty", "owl:DatatypeProperty", "owl:AnnotationProperty")

# (taxonomy letter, description, detection pattern)
#
# Consolidated mapping (merges several of the 20 taxonomy letters down so
# every checkable construct has a home among fewer, reused symbols):
#   S: SymmetricProperty ∪ AsymmetricProperty (both "how a property relates
#      its own inverse to itself" — merged)
#   V: ReflexiveProperty ∪ IrreflexiveProperty (both "self-relation" — merged)
#   A: owl:intersectionOf (conjunction)
#   R: rdfs:domain/range restrictions
#   O: owl:unionOf (disjunction, its own letter)
#   P: owl:propertyChainAxiom (moved off "N" to free it up)
#   N: owl:complementOf (negation; reuses the letter propertyChainAxiom vacated)
CONSTRUCTS: list[tuple[str, str, re.Pattern[str]]] = [
    ("H", "subClassOf/subPropertyOf a named class/property", re.compile(r"rdfs:sub(ClassOf|PropertyOf)\s+ns1:\w")),
    ("H*", "subClassOf a restriction (nested construct)", re.compile(r"rdfs:subClassOf\s+\[")),
    ("I", "owl:inverseOf", re.compile(r"owl:inverseOf")),
    ("R", "rdfs:domain / rdfs:range", re.compile(r"rdfs:(domain|range)")),
    ("O", "owl:unionOf", re.compile(r"owl:unionOf")),
    ("P", "owl:propertyChainAxiom", re.compile(r"owl:propertyChainAxiom")),
    ("T", "owl:TransitiveProperty", re.compile(r"owl:TransitiveProperty")),
    ("S", "Symmetric OR Asymmetric Property (merged)", re.compile(r"owl:(Symmetric|Asymmetric)Property")),
    ("F", "owl:FunctionalProperty / InverseFunctionalProperty", re.compile(r"owl:(Inverse)?FunctionalProperty")),
    ("V", "Reflexive OR Irreflexive Property (merged)", re.compile(r"owl:(Reflexive|Irreflexive)Property")),
    ("J", "owl:disjointWith / propertyDisjointWith", re.compile(r"owl:(propertyD|d)isjointWith")),
    ("Q", "owl:equivalentProperty", re.compile(r"owl:equivalentProperty")),
    ("E", "owl:someValuesFrom", re.compile(r"owl:someValuesFrom")),
    ("L", "owl:allValuesFrom", re.compile(r"owl:allValuesFrom")),
    ("C", "any cardinality restriction", re.compile(r"owl:(min|max)?[Qq]ualifiedCardinality|owl:(min|max)?[Cc]ardinality")),
    ("A", "owl:intersectionOf", re.compile(r"owl:intersectionOf")),
    ("N", "owl:complementOf", re.compile(r"owl:complementOf")),
]

# Originally {D,H,I,R,M,N,S,T} in the dataset's own explanation tagging.
# propertyChainAxiom is relabeled N -> P (see minimum_required_explanations
# .RELABEL), so the exercised set is {D,H,I,R,M,P,S,T} here — none of these
# 8 letters correspond to a CONSTRUCTS entry above, since D/M aren't TBox
# constructs this census can detect (D is an ABox fact; M, per verify_tags
# .simplify_tag's finding, is a mislabeled "multiple simple axiom types"
# flag, not a genuine nested-class-expression construct, so it's excluded
# here too — none of the 1,351 M-tagged explanations involve one).
INSTANTIATED_IN_EXPLANATIONS = {"D", "H", "I", "R", "P", "S", "T"}


def extract_tbox_blocks(owl_text: str) -> dict[str, str]:
    """Return {entity_name: block_text} for every owl:Class/*Property block."""
    blocks: dict[str, str] = {}
    for chunk in re.split(r"\n\n+", owl_text):
        chunk = chunk.strip()
        m = re.match(r"ns1:(\w+) a (" + "|".join(re.escape(t) for t in TBOX_TYPES) + r")\b", chunk)
        if m:
            blocks[m.group(1)] = chunk
    return blocks


def census_dataset(zip_path: Path, members: list[str], sample_size: int) -> dict[str, str]:
    """Union TBox blocks across `sample_size` items from each hop file."""
    all_blocks: dict[str, str] = {}
    for member in members:
        groups = load_member(zip_path, member)
        for item in groups[:sample_size]:
            all_blocks.update(extract_tbox_blocks(item["OWL Context"]))
    return all_blocks


def classify(blocks: dict[str, str]) -> tuple[Counter[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for name, block in blocks.items():
        for tag, _desc, pattern in CONSTRUCTS:
            if pattern.search(block):
                counts[tag] += 1
                if len(examples[tag]) < 3:
                    examples[tag].append(name)
    return counts, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "final_benchmark")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60,
        help="Items to union per hop file when reconstructing the TBox (default: 60)",
    )
    parser.add_argument("--dataset", choices=list(DATASET_FILES), help="Only run one dataset")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_FILES)
    for dataset in datasets:
        zip_name, members = DATASET_FILES[dataset]
        blocks = census_dataset(args.benchmark_dir / zip_name, members, args.sample_size)
        counts, examples = classify(blocks)
        print(f"=== {dataset} ({len(blocks)} distinct TBox class/property definitions unioned) ===")
        for tag, desc, _pattern in CONSTRUCTS:
            n = counts.get(tag, 0)
            status = "EXERCISED" if tag in INSTANTIATED_IN_EXPLANATIONS else ("unused" if n else "n/a")
            ex = ", ".join(examples.get(tag, [])[:3])
            print(f"  {tag:2s} {desc:45s} defined={n:4d}  [{status:9s}]  e.g. {ex}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
