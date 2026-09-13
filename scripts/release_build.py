"""Shared deterministic helpers for local release package construction."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from release_profiles import ReleaseProfile, VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_metadata(
    source: Path, destination: Path, profile: ReleaseProfile
) -> int:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["dataset"] in profile.datasets]
    if fieldnames is None:
        raise ValueError(f"Missing CSV header: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != profile.overall_total:
        raise ValueError(f"Filtered metadata has {len(rows)} rows")
    return len(rows)


def statistics(
    profile: ReleaseProfile,
    artifact_counts: dict[str, int],
    reasoning_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "benchmark_version": VERSION,
        "profile": profile.name,
        "unit": "unique question-hop instance",
        "overall": profile.overall_total,
        "tasks": profile.task_totals,
        "datasets": profile.dataset_totals,
        "dataset_hops": artifact_counts,
        "reasoning_type_counts": reasoning_counts,
    }


def reasoning_counts(metadata_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts.update(set(row["reasoning_tags"]))
    return dict(counts)


def dataset_card(profile: ReleaseProfile) -> str:
    if profile.includes_family:
        scope = (
            "This full candidate contains Family, Pizza 100, Pizza 250, and "
            "OWL2Bench. It is the intended canonical v1.0.0 distribution. "
            "Family/FHKB-derived material is subject to CC BY-SA 3.0."
        )
        limitation = (
            "Family/FHKB content is adapted from the Manchester tutorial resources; "
            "preserve its attribution, change notice, and CC BY-SA 3.0 terms."
        )
    else:
        scope = (
            "The paper evaluates four ontology datasets. This fallback distribution "
            "contains Pizza 100, Pizza 250, and OWL2Bench only; Family/FHKB is omitted "
            "as an optional reduced profile. See `docs/FAMILY_RECONSTRUCTION.md` for "
            "the provenance comparison and local reconstruction path."
        )
        limitation = (
            "Results in the paper include Family, but no Family source or derived "
            "benchmark payload is present in this public-safe package."
        )
    datasets = ", ".join(profile.datasets)
    return f"""---
pretty_name: CORE-LLM-Bench
version: {VERSION}
license: other
task_categories:
  - question-answering
language:
  - en
tags:
  - neurosymbolic-ai
  - ontology-reasoning
  - owl
---

# CORE-LLM-Bench v{VERSION} ({profile.name})

CORE-LLM-Bench evaluates verifiable ontology reasoning under aligned natural-
language (NL), formal-symbolic (FS), and abstract-representation (AR) inputs.
{scope}

## Profile statistics

- Datasets: {datasets}
- Unique question-hop instances: {profile.overall_total:,}
- BQA: {profile.task_totals['BQA']:,}
- OEQA: {profile.task_totals['OEQA']:,}
- Context depths: 1-hop and 2-hop
- Benchmark version: {VERSION}

Each Parquet row is one unique `(dataset, hop, task_id)` instance. NL, FS, and
AR are columns in that row, not duplicated model-response records. Fields cover
identity, task/label data, the three representations, reasoner-derived
explanations and complexity, reasoning tags, version, and source provenance.

## Validation and use

From the source repository, run:

```console
python scripts/validate_release.py --profile {profile.name} --package-kind huggingface --package-dir release/huggingface/{profile.name}
```

Keep all representations for an identity in the same split or analysis unit.
No model responses are included.

## Limitations, licensing, and citation

{limitation} Source materials have mixed provenance: Pizza is CC BY 3.0,
OWL2Bench is Apache-2.0, Family/FHKB-derived material is CC BY-SA 3.0,
repository software is MIT, and separable original author-created benchmark
material is CC BY 4.0. These do not form one blanket license. Consult
`NOTICE.md` and `CITATION.cff`. Release date, GitHub Release URL, Zenodo DOI,
Hugging Face URL, and unconfirmed ORCIDs remain deliberately unresolved.
"""


def write_manifest_and_checksums(
    package_dir: Path, profile: ReleaseProfile, stats: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = package_dir / "RELEASE_MANIFEST.json"
    checksum_path = package_dir / "SHA256SUMS"
    excluded = {manifest_path, checksum_path}
    files = sorted(
        (path for path in package_dir.rglob("*") if path.is_file() and path not in excluded),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(package_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "benchmark_name": "CORE-LLM-Bench",
        "benchmark_version": VERSION,
        "profile": profile.name,
        "statistics": stats,
        "files": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum_entries = entries + [
        {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        }
    ]
    checksum_path.write_text(
        "".join(
            f"{entry['sha256']}  {entry['path']}\n"
            for entry in sorted(checksum_entries, key=lambda item: item["path"])
        ),
        encoding="utf-8",
    )
    return manifest
