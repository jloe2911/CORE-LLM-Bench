#!/usr/bin/env python3
"""Prepare deterministic local Zenodo staging packages without uploading."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from release_build import (
    dataset_card,
    reasoning_counts,
    statistics,
    write_manifest_and_checksums,
    write_profile_metadata,
)
from release_profiles import PROFILES, VERSION, get_profile, zip_names_for
from validate_release import ROOT, validate_prepared_package, validate_release


COMMON_FILES = (
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "VERSION",
    "release/RELEASE_NOTES_v1.0.0.md",
    "docs/FAMILY_RECONSTRUCTION.md",
    "scripts/release_profiles.py",
    "scripts/release_build.py",
    "scripts/validate_release.py",
    "scripts/validate_family_reconstruction.py",
    "scripts/export_huggingface.py",
    "scripts/prepare_zenodo.py",
    "scripts/run_final_benchmark_pipeline.py",
    "final_benchmark/create_final_bench.py",
)


def _copy(relative: str, package_dir: Path, destination: str | None = None) -> None:
    source = ROOT / relative
    target = package_dir / (destination or relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _metadata(profile_name: str) -> dict[str, object]:
    profile = get_profile(profile_name)
    if profile.includes_family:
        scope = "Family, Pizza 100, Pizza 250, and OWL2Bench"
        note = (
            "This is the intended canonical full candidate. Family/FHKB-derived "
            "material is distributed under CC BY-SA 3.0; see NOTICE.md."
        )
    else:
        scope = "Pizza 100, Pizza 250, and OWL2Bench"
        note = (
            "The paper evaluates four ontology datasets, but this fallback distribution "
            "omits all Family/FHKB payload as an optional reduced profile."
        )
    affiliation = (
        "Department of Advanced Computing Sciences, Maastricht University, Netherlands"
    )
    return {
        "title": "CORE-LLM-Bench: A Controlled Neurosymbolic Benchmark for Ontology-Grounded Reasoning in Large Language Models",
        "upload_type": "dataset",
        "description": (
            f"CORE-LLM-Bench v{VERSION} ({profile.name}) contains "
            f"{profile.overall_total:,} unique question-hop instances across {scope}: "
            f"{profile.task_totals['BQA']:,} BQA and {profile.task_totals['OEQA']:,} "
            "OEQA instances with aligned NL, FS, and AR representations."
        ),
        "creators": [
            {"name": "Loesch, Julie", "affiliation": affiliation},
            {"name": "Falahatkar, Sara", "affiliation": affiliation},
            {
                "name": "Kilk, Nicole",
                "affiliation": "University College Maastricht, Maastricht University, Zwingelput 4, Maastricht, 6211 KH, Netherlands",
            },
            {"name": "Jakhar, Rishabh", "affiliation": affiliation},
            {
                "name": "Mutharaju, Raghava",
                "affiliation": "Mehta Family School of Data Science and AI, IIT Palakkad, Kerala, India",
            },
            {"name": "Dumontier, Michel", "affiliation": affiliation},
            {"name": "Celebi, Remzi", "affiliation": affiliation},
        ],
        "keywords": [
            "neurosymbolic AI",
            "large language models",
            "ontology reasoning",
            "benchmark",
            "symbolic reasoning",
            "description logic",
            "OWL",
            "question answering",
            "knowledge representation",
        ],
        "version": VERSION,
        "publication_date": "2026-09-13",
        "doi": "10.5281/zenodo.22742977",
        "related_identifiers": [
            {
                "identifier": "https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.0.0",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
        "notes": (
            f"{note} CORE-LLM-Bench v1.0.0 is archived at "
            "https://zenodo.org/records/22742977 under DOI "
            "10.5281/zenodo.22742977. The canonical tabular 9,032-row dataset "
            "view is available at "
            "https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench. "
            "Insert the complete verified conference "
            "citation only from a confirmed record; include ORCIDs only after "
            "mappings are confirmed. "
            "Licensing is component-specific: software MIT; separable original CORE "
            "material CC BY 4.0; Pizza CC BY 3.0; OWL2Bench Apache-2.0; and "
            "Family/FHKB-derived material CC BY-SA 3.0."
        ),
    }


def _write_deterministic_zip(package_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def prepare(output_dir: Path, profile_name: str = "full") -> dict[str, object]:
    profile = get_profile(profile_name)
    validation = validate_release(profile.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative in COMMON_FILES:
        _copy(relative, output_dir)
    if not profile.includes_family:
        _copy("release/RELEASE_NOTES_v1.0.0_PUBLIC_SAFE.md", output_dir)
    for zip_name in zip_names_for(profile):
        _copy(f"final_benchmark/{zip_name}", output_dir, f"benchmark/{zip_name}")

    metadata_path = output_dir / "benchmark" / "reasoning_metadata.csv"
    write_profile_metadata(
        ROOT / "final_benchmark" / "reasoning_metadata.csv", metadata_path, profile
    )
    stats = statistics(
        profile,
        validation["artifacts"],
        reasoning_counts(metadata_path),
    )
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(dataset_card(profile), encoding="utf-8")
    (output_dir / "zenodo_metadata.json").write_text(
        json.dumps(_metadata(profile.name), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest_and_checksums(output_dir, profile, stats)
    validate_prepared_package(output_dir, profile.name, "zenodo")

    archive_path = output_dir.parent / f"CORE-LLM-Bench-v{VERSION}-{profile.name}.zip"
    _write_deterministic_zip(output_dir, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"CRC failure in {archive_path}")
        archived = set(archive.namelist())
        packaged = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        if archived != packaged:
            raise ValueError("Archive and staging directory file sets differ")
        for relative in archived:
            if archive.read(relative) != (output_dir / relative).read_bytes():
                raise ValueError(f"Archive content mismatch: {relative}")
    manifest["package_bytes"] = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    manifest["archive"] = str(archive_path)
    manifest["archive_bytes"] = archive_path.stat().st_size
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Default: release/zenodo/<profile>"
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "release" / "zenodo" / args.profile
    )
    try:
        manifest = prepare(output_dir, args.profile)
    except (OSError, ValueError) as exc:
        print(f"ZENODO PACKAGE PREPARATION FAILED: {exc}")
        return 1
    print(
        f"Zenodo {args.profile} package ready: "
        f"{len(manifest['files'])} payload files"
    )
    print(f"Directory: {output_dir}")
    print(f"Archive: {manifest['archive']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
