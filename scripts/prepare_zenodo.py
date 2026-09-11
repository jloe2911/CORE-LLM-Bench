#!/usr/bin/env python3
"""Prepare a local Zenodo deposit directory without uploading anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from validate_release import ROOT, VERSION, validate_release


FILES = (
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "RELEASE_AUDIT.md",
    "VERSION",
    "final_benchmark/manifest.json",
    "final_benchmark/reasoning_metadata.csv",
    "final_benchmark/FamilyOWL.zip",
    "final_benchmark/pizza_100.zip",
    "final_benchmark/pizza_250.zip",
    "final_benchmark/OWL2Bench.zip",
    "results/reasoning_coverage/reasoning_coverage_audit.json",
    "results/reasoning_coverage/reasoning_coverage_counts.csv",
    "results/reasoning_coverage/reasoning_coverage_percentages.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output_dir: Path) -> dict[str, object]:
    validate_release()
    package_dir = output_dir / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in FILES:
        source = ROOT / relative
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    metadata_source = output_dir / "zenodo_metadata.json"
    shutil.copy2(metadata_source, package_dir / metadata_source.name)
    entries.append(
        {
            "path": metadata_source.name,
            "bytes": (package_dir / metadata_source.name).stat().st_size,
            "sha256": sha256(package_dir / metadata_source.name),
        }
    )
    manifest = {"benchmark_version": VERSION, "files": entries}
    (package_dir / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "release" / "zenodo"
    )
    args = parser.parse_args()
    try:
        manifest = prepare(args.output_dir.resolve())
    except (OSError, ValueError) as exc:
        print(f"ZENODO PACKAGE PREPARATION FAILED: {exc}")
        return 1
    print(f"Zenodo package ready: {len(manifest['files'])} files")
    print(f"Directory: {args.output_dir.resolve() / 'package'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
