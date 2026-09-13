#!/usr/bin/env python3
"""Validate locally reconstructed Family JSON without redistributing it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from validate_release import ROOT, iter_flat_rows


EXPECTED = {"1hop": 1880, "2hop": 1880}
EXPECTED_TASKS = {"BQA": 2544, "OEQA": 1216}


def validate(directory: Path) -> dict[str, object]:
    identities: set[tuple[str, str]] = set()
    task_counts: Counter[str] = Counter()
    hop_counts: dict[str, int] = {}
    for hop, expected in EXPECTED.items():
        path = directory / f"FamilyOWL_{hop}.json"
        groups = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in iter_flat_rows(groups, "Family", hop):
            count += 1
            group = item["group"]
            task_id = str(item["qa"]["Task ID"])
            identity = (hop, task_id)
            if identity in identities:
                raise ValueError(f"Duplicate reconstructed identity: {identity}")
            identities.add(identity)
            answer_type = str(group["Answer Type"]).upper()
            if answer_type == "BIN":
                task_counts["BQA"] += 1
            elif answer_type == "MC":
                task_counts["OEQA"] += 1
            else:
                raise ValueError(f"Unexpected Answer Type: {answer_type}")
        if count != expected:
            raise ValueError(f"Family {hop} count mismatch: {count} != {expected}")
        hop_counts[hop] = count
    if dict(task_counts) != EXPECTED_TASKS:
        raise ValueError(f"Family task totals mismatch: {dict(task_counts)}")
    return {
        "status": "PASS",
        "overall": len(identities),
        "hops": hop_counts,
        "tasks": dict(task_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "final_benchmark"
    )
    args = parser.parse_args()
    try:
        report = validate(args.input_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"FAMILY RECONSTRUCTION VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
