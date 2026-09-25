#!/usr/bin/env python3
"""Build and validate the unpublished canonical CORE-LLM-Bench v1.1.0 tree.

This is an offline packaging operation. It copies already-finalized benchmark
and derived-analysis artifacts; it never regenerates questions, changes saved
model responses, or imports an API client.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "v1.1.0"
BENCHMARK_SOURCE = ROOT / "release" / "v1.1.0-schema-finalized"
EVALUATION_SOURCE = ROOT / "results" / "v1.1.0-evaluation-corrected-v1"
TAG_ANALYSIS_SOURCE = ROOT / "results" / "v1.1.0-reasoning-tag-analysis"
EXPECTED_HF_V1_COMMIT = "14ab8bcdcbcbd464b4768175c85adacd917393fc"
EXPECTED_BENCHMARK_SHA256 = "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc"
EXPECTED_INPUT_MANIFEST_SHA256 = "191c1c0dc221a5829dfa361f86fd1ebf8bbfd54e8f89eeaf0621810001715d18"
ZENODO_DOI = "10.5281/zenodo.22959957"
ZENODO_RECORD_ID = 22959957
ZENODO_RECORD_URL = "https://zenodo.org/records/22959957"
BENCHMARK_FILES = (
    "FamilyOWL_1hop.json", "FamilyOWL_2hop.json",
    "Pizza100_1hop.json", "Pizza100_2hop.json",
    "Pizza250_1hop.json", "Pizza250_2hop.json",
    "OWL2Bench_1hop.json", "OWL2Bench_2hop.json",
)
BENCHMARK_AUX_FILES = (
    "core_llm_bench_v1_1.parquet", "bqa_pair_mapping.csv", "complexity_distribution.json",
    "dataset_statistics.json", "entity_label_mapping.csv", "reasoning_coverage.json",
    "reasoning_metadata.csv", "task_id_mapping.csv", "validation_report.json",
)
PROVENANCE_FILES = (
    "model_input_manifest.csv", "experiment_config_v1_1.json", "input_equivalence_groups.csv",
    "primary_experiment_manifest.csv", "pending_model_runs.csv", "cost_estimate.json",
)
SCHEMA_FINALIZATION_FILES = ("RELEASE_MANIFEST.json", "SHA256SUMS", "membership_manifest.json")
CONTROL_FILES = {"RELEASE_MANIFEST.json", "SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=shutil.copy2)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_generated_manifest(root: Path, manifest_path: Path, key: str) -> int:
    manifest = load_json(manifest_path)
    entries = manifest[key]
    for entry_path, expected in (
        entries.items() if isinstance(entries, dict)
        else ((entry["path"], entry["sha256"]) for entry in entries)
    ):
        path = root / Path(str(entry_path).replace("\\", "/"))
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Manifest mismatch: {path}")
    return len(entries)


def verify_tree_parity(source: Path, target: Path) -> int:
    source_files = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*") if path.is_file()
    }
    target_files = {
        path.relative_to(target).as_posix(): path
        for path in target.rglob("*") if path.is_file()
    }
    if set(source_files) != set(target_files):
        raise ValueError(f"Copied tree membership drift: {target}")
    for relative, source_path in source_files.items():
        if sha256(source_path) != sha256(target_files[relative]):
            raise ValueError(f"Copied artifact drift: {target / relative}")
    return len(source_files)


def validate_complete_benchmark(package: Path) -> dict[str, Any]:
    source_manifest = load_json(BENCHMARK_SOURCE / "RELEASE_MANIFEST.json")
    expected = {entry["path"]: entry for entry in source_manifest["files"]}
    all_ids: set[int] = set()
    semantic_keys: set[str] = set()
    counts: Counter[str] = Counter()
    json_rows = 0
    for filename in BENCHMARK_FILES:
        relative = f"benchmark/{filename}"
        path = package / relative
        entry = expected[relative]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Finalized benchmark payload mismatch: {relative}")
        dataset, hop_ext = filename.rsplit("_", 1)
        hop = hop_ext.removesuffix(".json")
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Benchmark file is not a JSON list: {relative}")
        for context in payload:
            for row in context.get("QAs", []):
                task_id = int(row["task_id"])
                if task_id in all_ids or row["semantic_key"] in semantic_keys:
                    raise ValueError(f"Duplicate benchmark identity at task {task_id}")
                if row["dataset"] != dataset or row["hop"] != hop:
                    raise ValueError(f"Dataset/hop mismatch at task {task_id}")
                if not row.get("answer_explanations") or not row.get("complete_explanation"):
                    raise ValueError(f"Missing finalized explanation schema at task {task_id}")
                all_ids.add(task_id)
                semantic_keys.add(row["semantic_key"])
                counts[row["task_group"]] += 1
                counts[f"{dataset}/{hop}"] += 1
                json_rows += 1
        del payload

    if all_ids != set(range(1, 9049)) or semantic_keys.__len__() != 9048:
        raise ValueError("JSON benchmark identities are not exactly the frozen 1..9048 set")
    if counts["BQA"] != 6032 or counts["OEQA"] != 3016:
        raise ValueError(f"Unexpected benchmark task counts: {dict(counts)}")

    for filename in BENCHMARK_AUX_FILES:
        if sha256(BENCHMARK_SOURCE / filename) != sha256(package / "benchmark" / filename):
            raise ValueError(f"Finalized benchmark auxiliary artifact drift: {filename}")

    parquet = package / "benchmark/core_llm_bench_v1_1.parquet"
    if sha256(parquet) != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("Canonical Parquet hash mismatch")
    table = pq.read_table(parquet, columns=["task_id", "semantic_key", "task_group"])
    parquet_ids = {int(value.as_py()) for value in table.column("task_id")}
    parquet_semantic = {str(value.as_py()) for value in table.column("semantic_key")}
    if table.num_rows != 9048 or parquet_ids != all_ids or parquet_semantic != semantic_keys:
        raise ValueError("JSON/Parquet identity parity failed")
    return {
        "benchmark_json_files": 8,
        "benchmark_json_rows": json_rows,
        "benchmark_parquet_rows": table.num_rows,
        "public_ids": "1..9048 unique",
        "semantic_keys_unique": len(semantic_keys),
        "task_counts": {"BQA": counts["BQA"], "OEQA": counts["OEQA"]},
        "benchmark_sha256": sha256(parquet),
    }


def validate_source_provenance(package: Path) -> dict[str, Any]:
    if sha256(package / "provenance/model_input_manifest.csv") != EXPECTED_INPUT_MANIFEST_SHA256:
        raise ValueError("Frozen model-input manifest drift")
    evaluation = load_json(package / "evaluation/REPRODUCIBILITY_MANIFEST.json")
    if evaluation["benchmark_sha256"] != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("Evaluation is not bound to the canonical benchmark")
    source_hashes = evaluation["source_observation_sha256"]
    current_sources = {
        "GPT-5 mini": ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
        "Gemini 2.5 Flash-Lite": ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
        "Qwen3-30B-A3B-Instruct": ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
    }
    for model, path in current_sources.items():
        if sha256(path) != source_hashes[model]:
            raise ValueError(f"Original observation provenance drift: {model}")
    for filename in PROVENANCE_FILES:
        if sha256(BENCHMARK_SOURCE / filename) != sha256(package / "provenance" / filename):
            raise ValueError(f"Experiment provenance copy drift: {filename}")
    for filename in SCHEMA_FINALIZATION_FILES:
        source = BENCHMARK_SOURCE / filename
        target = package / "provenance/source_schema_finalization" / filename
        if sha256(source) != sha256(target):
            raise ValueError(f"Schema-finalization provenance copy drift: {filename}")
    evaluation_tree_files = verify_tree_parity(EVALUATION_SOURCE, package / "evaluation")
    tag_analysis_tree_files = verify_tree_parity(TAG_ANALYSIS_SOURCE, package / "reasoning_tag_analysis")
    eval_files = verify_generated_manifest(package / "evaluation", package / "evaluation/REPRODUCIBILITY_MANIFEST.json", "generated_files")
    tag_files = verify_generated_manifest(package / "reasoning_tag_analysis", package / "reasoning_tag_analysis/REPRODUCIBILITY_MANIFEST.json", "outputs")
    tag_validation = load_json(package / "reasoning_tag_analysis/audit/validation.json")
    if tag_validation["question_count"] != 9048 or tag_validation["observation_count"] != 81432:
        raise ValueError("Reasoning-tag analysis coverage mismatch")
    if tag_validation["canonical_inputs_modified"]:
        raise ValueError("Reasoning-tag analysis reports modified canonical inputs")
    return {
        "model_input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "accepted_observations": 81432,
        "source_observation_hashes_verified": source_hashes,
        "experiment_provenance_files_byte_preserved": len(PROVENANCE_FILES),
        "schema_finalization_files_byte_preserved": len(SCHEMA_FINALIZATION_FILES),
        "evaluation_tree_files_byte_preserved": evaluation_tree_files,
        "reasoning_tag_tree_files_byte_preserved": tag_analysis_tree_files,
        "evaluation_outputs_verified": eval_files,
        "reasoning_tag_outputs_verified": tag_files,
    }


def validate_v1_preservation(package: Path) -> dict[str, Any]:
    preservation = load_json(package / "PRESERVATION.json")
    hf = preservation["v1.0.0"]["hugging_face"]
    if hf["tag"] != "v1.0.0" or hf["commit"] != EXPECTED_HF_V1_COMMIT or not hf["independently_verified"]:
        raise ValueError("Historical Hugging Face tag verification is incomplete")
    archive = ROOT / "release/zenodo/CORE-LLM-Bench-v1.0.0-full.zip"
    zenodo = preservation["v1.0.0"]["zenodo"]
    if archive.stat().st_size != zenodo["archive_bytes"] or md5(archive) != zenodo["archive_md5"]:
        raise ValueError("Local preserved Zenodo v1.0.0 archive does not match the public record")
    if sha256(archive) != zenodo["archive_sha256"]:
        raise ValueError("Local preserved Zenodo v1.0.0 archive SHA-256 mismatch")
    return {
        "hugging_face_tag_commit": hf["commit"],
        "github_tag_commit": preservation["v1.0.0"]["github"]["commit"],
        "zenodo_doi": zenodo["doi"],
        "zenodo_archive_md5_verified": True,
    }


def validate_release_metadata(package: Path) -> dict[str, Any]:
    metadata = load_json(package / "RELEASE_METADATA.json")
    zenodo = metadata.get("zenodo", {})
    if zenodo.get("doi") != ZENODO_DOI or zenodo.get("record_id") != ZENODO_RECORD_ID:
        raise ValueError("Zenodo v1.1.0 DOI/record identity mismatch")
    if zenodo.get("record_url") != ZENODO_RECORD_URL or not zenodo.get("reserved"):
        raise ValueError("Zenodo v1.1.0 draft reservation metadata mismatch")
    if zenodo.get("published") is not False or metadata.get("publication_date") is not None:
        raise ValueError("Unpublished v1.1.0 must retain a pending publication date")
    citation = (package / "CITATION.cff").read_text(encoding="utf-8")
    if f'doi: "{ZENODO_DOI}"' not in citation or f'url: "{ZENODO_RECORD_URL}"' not in citation:
        raise ValueError("CITATION.cff does not contain the reserved v1.1.0 identifier")
    if "date-released:" in citation:
        raise ValueError("CITATION.cff date-released must remain absent before publication")
    return {
        "zenodo_doi": ZENODO_DOI,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "zenodo_record_url": ZENODO_RECORD_URL,
        "zenodo_published": False,
        "publication_date": None,
    }


def package_files(package: Path) -> list[Path]:
    return sorted(
        (path for path in package.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(package).as_posix(),
    )


def write_release_controls(package: Path, validation: dict[str, Any]) -> None:
    write_json(package / "VALIDATION_REPORT.json", validation)
    files = [path for path in package_files(package) if path.name not in CONTROL_FILES]
    manifest = {
        "benchmark": "CORE-LLM-Bench",
        "version": "1.1.0",
        "release_tag": "v1.1.0",
        "status": "prepared-not-published",
        "canonical_release_directory": "release/v1.1.0",
        "benchmark_questions": 9048,
        "accepted_observations": 81432,
        "zenodo": {
            "doi": ZENODO_DOI,
            "record_id": ZENODO_RECORD_ID,
            "record_url": ZENODO_RECORD_URL,
            "reserved": True,
            "published": False,
        },
        "publication_date": None,
        "files": [
            {"path": path.relative_to(package).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
    }
    write_json(package / "RELEASE_MANIFEST.json", manifest)
    checksum_paths = [path for path in package_files(package) if path.name != "SHA256SUMS"]
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(package).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8", newline="\n",
    )


def validate_controls(package: Path) -> dict[str, Any]:
    manifest = load_json(package / "RELEASE_MANIFEST.json")
    if manifest["version"] != "1.1.0" or manifest["status"] != "prepared-not-published":
        raise ValueError("Canonical release identity/status mismatch")
    zenodo = manifest.get("zenodo", {})
    if zenodo.get("doi") != ZENODO_DOI or zenodo.get("record_id") != ZENODO_RECORD_ID:
        raise ValueError("Release manifest Zenodo identity mismatch")
    if zenodo.get("published") is not False or manifest.get("publication_date") is not None:
        raise ValueError("Release manifest publication state mismatch")
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    actual_payload = {
        path.relative_to(package).as_posix()
        for path in package_files(package)
        if path.name not in CONTROL_FILES
    }
    if manifest_paths != actual_payload:
        raise ValueError("Release manifest membership is incomplete")
    for entry in manifest["files"]:
        path = package / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Release manifest mismatch: {entry['path']}")
    checksum_rows = {}
    with (package / "SHA256SUMS").open(encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip("\n").split("  ", 1)
            checksum_rows[relative] = digest
    expected_paths = {
        path.relative_to(package).as_posix()
        for path in package_files(package)
        if path.name != "SHA256SUMS"
    }
    if set(checksum_rows) != expected_paths:
        raise ValueError("SHA256SUMS membership is incomplete")
    for relative, digest in checksum_rows.items():
        if sha256(package / relative) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    active_text = "\n".join(
        (package / name).read_text(encoding="utf-8")
        for name in ("README.md", "RELEASE_NOTES.md", "RELEASE_METADATA.json", "RELEASE_MANIFEST.json")
    ).lower()
    if "v1.1.0-staging" in active_text or "not-released" in active_text:
        raise ValueError("Unresolved staging identity in canonical release metadata")
    return {"manifest_entries": len(manifest_paths), "checksum_entries": len(checksum_rows)}


def validate(package: Path) -> dict[str, Any]:
    return {
        "status": "passed",
        "offline_only": True,
        "benchmark": validate_complete_benchmark(package),
        "provenance": validate_source_provenance(package),
        "release_metadata": validate_release_metadata(package),
        "v1_preservation": validate_v1_preservation(package),
    }


def build(package: Path) -> None:
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite existing release tree: {package}")
    if package.parent.resolve() != (ROOT / "release").resolve():
        raise ValueError("The canonical release must be a direct child of release/")
    package.mkdir(parents=True)

    for filename in BENCHMARK_FILES:
        copy_file(BENCHMARK_SOURCE / "benchmark" / filename, package / "benchmark" / filename)
    for filename in BENCHMARK_AUX_FILES:
        copy_file(BENCHMARK_SOURCE / filename, package / "benchmark" / filename)
    for filename in PROVENANCE_FILES:
        copy_file(BENCHMARK_SOURCE / filename, package / "provenance" / filename)
    for filename in SCHEMA_FINALIZATION_FILES:
        copy_file(BENCHMARK_SOURCE / filename, package / "provenance/source_schema_finalization" / filename)
    copy_tree(EVALUATION_SOURCE, package / "evaluation")
    copy_tree(TAG_ANALYSIS_SOURCE, package / "reasoning_tag_analysis")
    for filename in ("LICENSE", "NOTICE.md"):
        copy_file(ROOT / filename, package / filename)
    (package / "VERSION").write_text("1.1.0\n", encoding="utf-8", newline="\n")
    (package / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: \"If you use CORE-LLM-Bench v1.1.0, cite the published conference paper and this dataset version.\"\n"
        "title: \"CORE-LLM-Bench: A Controlled Neurosymbolic Benchmark for Ontology-Grounded Reasoning in Large Language Models\"\n"
        "type: dataset\n"
        "version: 1.1.0\n"
        f"doi: \"{ZENODO_DOI}\"\n"
        f"url: \"{ZENODO_RECORD_URL}\"\n"
        "authors:\n"
        "  - family-names: \"Loesch\"\n    given-names: \"Julie\"\n    affiliation: \"Department of Advanced Computing Sciences, Maastricht University, Netherlands\"\n"
        "  - family-names: \"Falahatkar\"\n    given-names: \"Sara\"\n    affiliation: \"Department of Advanced Computing Sciences, Maastricht University, Netherlands\"\n"
        "  - family-names: \"Kilk\"\n    given-names: \"Nicole\"\n    affiliation: \"University College Maastricht, Maastricht University, Netherlands\"\n"
        "  - family-names: \"Jakhar\"\n    given-names: \"Rishabh\"\n    affiliation: \"Department of Advanced Computing Sciences, Maastricht University, Netherlands\"\n"
        "  - family-names: \"Mutharaju\"\n    given-names: \"Raghava\"\n    affiliation: \"Mehta Family School of Data Science and AI, IIT Palakkad, Kerala, India\"\n"
        "  - family-names: \"Dumontier\"\n    given-names: \"Michel\"\n    affiliation: \"Department of Advanced Computing Sciences, Maastricht University, Netherlands\"\n"
        "  - family-names: \"Celebi\"\n    given-names: \"Remzi\"\n    affiliation: \"Department of Advanced Computing Sciences, Maastricht University, Netherlands\"\n"
        "repository-code: \"https://github.com/jloe2911/CORE-LLM-Bench\"\n"
        "keywords:\n  - \"neurosymbolic AI\"\n  - \"ontology reasoning\"\n  - \"benchmark\"\n  - \"question answering\"\n",
        encoding="utf-8", newline="\n",
    )

    release_metadata = {
        "title": "CORE-LLM-Bench: A Controlled Neurosymbolic Benchmark for Ontology-Grounded Reasoning in Large Language Models",
        "version": "1.1.0",
        "release_tag": "v1.1.0",
        "status": "prepared-not-published",
        "prepared_date": "2026-09-25",
        "canonical_release_directory": "release/v1.1.0",
        "github": {"repository": "jloe2911/CORE-LLM-Bench", "tag": "v1.1.0", "published": False},
        "hugging_face": {"dataset": "jloe2911/CORE-LLM-Bench", "revision": "v1.1.0", "published": False},
        "zenodo": {
            "doi": ZENODO_DOI, "record_id": ZENODO_RECORD_ID,
            "record_url": ZENODO_RECORD_URL, "reserved": True, "published": False,
        },
        "publication_date": None,
        "scientific_inputs_regenerated": False,
        "model_responses_modified": False,
        "llm_api_calls": 0,
    }
    write_json(package / "RELEASE_METADATA.json", release_metadata)
    preservation = {
        "v1.0.0": {
            "github": {
                "tag": "v1.0.0", "tag_object": "6873eb5b6f656c744183e5851e5e2920a3b89c46",
                "commit": "ee766caa48bb23905956a14b1ca5836bcfe19e6d", "independently_verified": True,
                "verification_method": "git ls-remote --tags with peeled annotated tag",
            },
            "hugging_face": {
                "dataset": "jloe2911/CORE-LLM-Bench", "tag": "v1.0.0",
                "commit": EXPECTED_HF_V1_COMMIT, "independently_verified": True,
                "verification_method": "git ls-remote --tags against the public dataset repository",
            },
            "zenodo": {
                "record_id": 22742977, "concept_record_id": 22742976,
                "doi": "10.5281/zenodo.22742977", "status": "published",
                "archive": "CORE-LLM-Bench-v1.0.0-full.zip", "archive_bytes": 78761233,
                "archive_md5": "571dbc6447272352dfb6d38e391158fc",
                "archive_sha256": "4d89a978c8e86f2a846cb9b1b59b2378111b57df9a730a390ecf0df4bb47596d",
                "independently_verified": True,
                "verification_method": "Zenodo public record API plus local archive digest comparison",
            },
            "preservation_policy": "Immutable historical release; v1.1.0 is a new version and must not replace, retag, or overwrite these objects.",
        }
    }
    write_json(package / "PRESERVATION.json", preservation)
    (package / "README.md").write_text(
        "# CORE-LLM-Bench v1.1.0 release candidate\n\n"
        "Status: prepared, validated, and not published. This is the single canonical local v1.1.0 release tree.\n\n"
        "`benchmark/` contains the complete schema-finalized 9,048-question payload. `evaluation/` contains the corrected offline evaluation, and `reasoning_tag_analysis/` contains the finalized reasoning-tag analysis. Historical experiment manifests remain byte-preserved under `provenance/`; their staging-era labels are provenance, not the identity of this release.\n\n"
        f"No benchmark questions or model responses were regenerated or modified, and no LLM/API call was made while creating this tree. Zenodo DOI `{ZENODO_DOI}` and record ID `{ZENODO_RECORD_ID}` are reserved for this version; the record is not published and the publication date remains pending.\n",
        encoding="utf-8", newline="\n",
    )
    (package / "RELEASE_NOTES.md").write_text(
        "# CORE-LLM-Bench v1.1.0\n\n"
        "This release candidate provides the finalized explanation schema, collision-safe natural-language inputs, corrected offline answer and hallucination evaluation, and finalized reasoning-tag association analysis. It contains 9,048 questions (6,032 BQA in 3,016 TRUE/FALSE pairs and 3,016 OEQA) and evaluation of 81,432 accepted observations.\n\n"
        "The v1.0.0 GitHub tag/release, Zenodo record 22742977, and Hugging Face tag v1.0.0 remain immutable historical artifacts. See `PRESERVATION.json` for verified identities.\n\n"
        f"Zenodo DOI `{ZENODO_DOI}` and record ID `{ZENODO_RECORD_ID}` are reserved specifically for v1.1.0. The draft is not yet published, so the publication date remains pending.\n\n"
        "Known scientific limitation: three AR prompts (task IDs 1274, 4094, and 5165) are confirmed defective; the primary frozen-observation analysis retains them and provides a separately identified exclusion sensitivity. No response was rerun.\n",
        encoding="utf-8", newline="\n",
    )
    preliminary = validate(package)
    preliminary["canonical_release_directory"] = "release/v1.1.0"
    preliminary["staging_identifiers_resolved"] = True
    preliminary["publication_actions_performed"] = []
    write_release_controls(package, preliminary)
    controls = validate_controls(package)
    print(json.dumps({**preliminary, "controls": controls}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-controls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = args.output_dir.resolve()
    if args.validate_only and args.refresh_controls:
        raise ValueError("Choose either --validate-only or --refresh-controls")
    if args.validate_only:
        report = validate(package)
        report["controls"] = validate_controls(package)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.refresh_controls:
        report = validate(package)
        report["canonical_release_directory"] = "release/v1.1.0"
        report["staging_identifiers_resolved"] = True
        report["publication_actions_performed"] = []
        write_release_controls(package, report)
        report["controls"] = validate_controls(package)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        build(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
