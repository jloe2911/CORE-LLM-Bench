#!/usr/bin/env python3
"""Remove Pizza BIN tasks that trivially ask for DomainConcept membership.

DomainConcept is the root of the Pizza ontology's class hierarchy, so a BIN
(ASK) question such as "Is Hot Green Pepper Topping a DomainConcept?" is
answerable from ontology structure alone and carries no reasoning content.
MC (OEQA) questions that merely include DomainConcept as one of several valid
answers are left untouched, since the underlying question is still meaningful.

Writes a cleaned copy of each hop file alongside the frozen v1.0 release;
final_benchmark/pizza_250.zip itself is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def is_domainconcept_bin(qa: dict[str, Any], answer_type: str) -> bool:
    return answer_type == "BIN" and "DomainConcept" in qa.get("SPARQL Query", "")


def clean_groups(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Return (cleaned_groups, qas_removed, groups_dropped)."""
    cleaned: list[dict[str, Any]] = []
    qas_removed = 0
    for group in groups:
        answer_type = group.get("Answer Type")
        kept_qas = [
            qa for qa in group["QAs"] if not is_domainconcept_bin(qa, answer_type)
        ]
        qas_removed += len(group["QAs"]) - len(kept_qas)
        if kept_qas:
            cleaned.append({**group, "QAs": kept_qas})
    groups_dropped = len(groups) - len(cleaned)
    return cleaned, qas_removed, groups_dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=ROOT / "final_benchmark" / "pizza_250.zip",
        help="Source benchmark package to clean (default: pizza_250.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write cleaned *_1hop.json/*_2hop.json into "
        "(default: <zip-stem>_no_domainconcept/ next to the source zip)",
    )
    args = parser.parse_args()

    zip_path: Path = args.zip
    if not zip_path.is_file():
        print(f"Missing benchmark package: {zip_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or zip_path.with_name(
        f"{zip_path.stem}_no_domainconcept"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    total_dropped = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(archive.namelist())
        for member in members:
            if not member.endswith(".json"):
                continue
            groups = json.loads(archive.read(member).decode("utf-8"))
            orig_qas = sum(len(g["QAs"]) for g in groups)
            cleaned, qas_removed, groups_dropped = clean_groups(groups)
            total_removed += qas_removed
            total_dropped += groups_dropped
            out_path = output_dir / member
            out_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
            print(
                f"{member}: groups {len(groups)} -> {len(cleaned)} "
                f"(dropped {groups_dropped}), QAs {orig_qas} -> "
                f"{orig_qas - qas_removed} (removed {qas_removed})"
            )

    print(f"\nTotal DomainConcept BIN tasks removed: {total_removed}")
    print(f"Total groups dropped (no QAs left): {total_dropped}")
    print(f"Cleaned files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
