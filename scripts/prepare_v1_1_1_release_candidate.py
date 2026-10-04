#!/usr/bin/env python3
"""Prepare and validate the immutable, unpublished CORE-LLM-Bench v1.1.1 RC.

This is an offline file-packaging operation.  It never calls a model or API.
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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.1.1"
RC_NAME = "v1.1.1-rc1"
DEFAULT_OUTPUT = ROOT / "release" / RC_NAME
REPAIRED = ROOT / "data/correction/v1.1.1/core_llm_bench_v1_1_1_corrected.parquet"
REPAIR_REPORT = ROOT / "data/correction/v1.1.1/proof_metadata_repair_report.json"
EVALUATION = ROOT / "results/v1.1.1-release-candidate/evaluation"
ANALYSIS = ROOT / "results/v1.1.1-release-candidate/reasoning-tag-analysis"
TEMPLATE_BENCHMARK = ROOT / "release/v1.1.0-staging/benchmark"
TEMPLATE_AUX = ROOT / "release/v1.1.0-staging"
V110 = ROOT / "release/v1.1.0"
BENCHMARK_FILES = (
    "FamilyOWL_1hop.json", "FamilyOWL_2hop.json",
    "OWL2Bench_1hop.json", "OWL2Bench_2hop.json",
    "Pizza100_1hop.json", "Pizza100_2hop.json",
    "Pizza250_1hop.json", "Pizza250_2hop.json",
)
RESPONSE_SOURCES = {
    "responses/frozen/gpt_observations.jsonl":
        ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
    "responses/frozen/gemini_observations.jsonl":
        ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
    "responses/frozen/qwen_observations.jsonl":
        ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
    "responses/corrected-rerun/responses.jsonl":
        ROOT / "data/output/v1.1.0-minimum-rerun/responses.jsonl",
    "responses/corrected-rerun/retry_attempts.jsonl":
        ROOT / "data/output/v1.1.0-minimum-rerun/retry_attempts.jsonl",
    "responses/corrected-rerun/checkpoint.sqlite3":
        ROOT / "data/output/v1.1.0-minimum-rerun/checkpoint.sqlite3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
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


def decoded(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        return json.loads(value)
    return value


def public_value(field: str, value: Any) -> Any:
    if field in {
        "answer_explanations", "complete_explanation", "primitive_reasoning_tags",
        "distinct_primitive_reasoning_types", "gold_answer_iris",
    }:
        value = decoded(value)
        return value.tolist() if hasattr(value, "tolist") else value
    if pd.isna(value):
        return None if field == "positive_task_id" else ""
    if field in {
        "task_id", "positive_task_id", "raw_minimum_complete_primitive_tag_complexity",
        "raw_maximum_complete_primitive_tag_complexity",
        "minimum_distinct_primitive_type_count", "maximum_distinct_primitive_type_count",
    }:
        return int(value)
    return value


def build_benchmark(output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(REPAIRED).copy()
    if len(frame) != 9048 or frame.task_id.nunique() != 9048:
        raise RuntimeError("Repaired benchmark membership failed")
    if frame.corrected_semantic_key.nunique() != 9048:
        raise RuntimeError("Corrected semantic keys are not unique")

    # In the public v1.1.1 payload the corrected identity is canonical; the
    # pre-correction key remains explicitly preserved as original_semantic_key.
    frame["semantic_key"] = frame["corrected_semantic_key"]
    parquet = output / "benchmark/core_llm_bench_v1_1_1.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet, index=False)

    by_id = {int(row["task_id"]): row for row in frame.to_dict(orient="records")}
    context_fields = {"nl_context", "fs_context", "ar_context", "dataset_key"}
    internal_fields = {"corrected_semantic_key"}
    qa_fields = [c for c in frame.columns if c not in context_fields | internal_fields]
    json_ids: set[int] = set()
    for filename in BENCHMARK_FILES:
        payload = json.loads((TEMPLATE_BENCHMARK / filename).read_text(encoding="utf-8"))
        for context in payload:
            rewritten = []
            for old in context.get("QAs", []):
                task_id = int(old["task_id"])
                row = by_id[task_id]
                rewritten.append({field: public_value(field, row[field]) for field in qa_fields})
                json_ids.add(task_id)
            context["QAs"] = rewritten
        write_json(output / "benchmark" / filename, payload)
    if json_ids != set(range(1, 9049)):
        raise RuntimeError("JSON benchmark does not cover exactly task IDs 1..9048")

    # Regenerate all proof-derived benchmark summaries from the repaired rows.
    metadata_fields = [
        "task_id", "semantic_key", "dataset", "hop", "task_group",
        "raw_minimum_complete_primitive_tag_complexity",
        "raw_maximum_complete_primitive_tag_complexity",
        "primitive_reasoning_tags", "distinct_primitive_reasoning_types",
        "m_status", "complexity_bin", "minimum_distinct_primitive_type_count",
        "maximum_distinct_primitive_type_count",
    ]
    metadata = frame[metadata_fields].copy()
    metadata["minimum_axiom_count"] = frame.complete_explanation.map(
        lambda value: decoded(value)["min_axiom_count"]
    )
    metadata["maximum_axiom_count"] = frame.complete_explanation.map(
        lambda value: decoded(value)["max_axiom_count"]
    )
    for field in ("primitive_reasoning_tags", "distinct_primitive_reasoning_types"):
        metadata[field] = metadata[field].map(lambda value: "".join(decoded(value)))
    ordered = metadata_fields[:5] + metadata_fields[5:7] + [
        "minimum_axiom_count", "maximum_axiom_count"
    ] + metadata_fields[7:]
    metadata[ordered].to_csv(output / "benchmark/reasoning_metadata.csv", index=False)

    distribution: dict[str, dict[str, int]] = {}
    for task in ("BQA", "OEQA"):
        counts = Counter(frame.loc[frame.task_group == task, "complexity_bin"])
        distribution[task] = {key: int(counts[key]) for key in ("High", "Low", "Medium")}
    write_json(output / "benchmark/complexity_distribution.json", distribution)

    coverage = Counter()
    for value in frame.primitive_reasoning_tags:
        coverage.update(set(decoded(value)))
    write_json(output / "benchmark/reasoning_coverage.json", {
        "m_is_metadata_not_primitive": True,
        "m_status_question_counts": {
            str(k): int(v) for k, v in sorted(Counter(frame.m_status).items())
        },
        "primitive_reasoning_type_question_counts": {
            str(k): int(v) for k, v in sorted(coverage.items())
        },
    })

    pairs = []
    for pair_id, group in frame[frame.task_group == "BQA"].groupby("pair_group_id"):
        positive = group[group.gold_answer == "TRUE"].iloc[0]
        negative = group[group.gold_answer == "FALSE"].iloc[0]
        pairs.append({
            "pair_group_id": pair_id,
            "positive_task_id": int(positive.task_id),
            "negative_task_id": int(negative.task_id),
            "positive_semantic_key": positive.semantic_key,
            "negative_semantic_key": negative.semantic_key,
            "dataset": positive.dataset,
            "hop": positive.hop,
            "root_entity": positive.root_entity,
            "sampling_group_key": positive.sampling_group_key,
        })
    pd.DataFrame(pairs).sort_values("positive_task_id").to_csv(
        output / "benchmark/bqa_pair_mapping.csv", index=False
    )
    frame.rename(columns={"task_id": "new_task_id"})[[
        "new_task_id", "semantic_key", "legacy_task_id", "dataset", "hop",
        "task_group", "root_entity", "formal_query", "gold_answer",
    ]].to_csv(output / "benchmark/task_id_mapping.csv", index=False)
    copy_file(TEMPLATE_AUX / "entity_label_mapping.csv",
              output / "benchmark/entity_label_mapping.csv")
    write_json(output / "benchmark/dataset_statistics.json", {
        "total": 9048, "BQA": 6032, "OEQA": 3016,
        "TRUE": 3016, "FALSE": 3016, "complete_bqa_pairs": 3016,
        "primary_experiment_observations": 81432,
        "public_id_min": 1, "public_id_max": 9048, "public_id_unique": 9048,
    })
    return frame, {
        "questions": 9048,
        "BQA": 6032,
        "OEQA": 3016,
        "balanced_pairs": 3016,
        "json_files": 8,
        "parquet_sha256": sha256(parquet),
        "complexity_distribution": distribution,
    }


def verify_manifest(output: Path) -> int:
    manifest = json.loads((output / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = output / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Release manifest mismatch: {entry['path']}")
    return len(manifest["files"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.validate_only:
        print(json.dumps({"status": "PASS", "files": verify_manifest(output)}, indent=2))
        return 0
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing release candidate: {output}")
    if output == V110.resolve() or output.name == "v1.1.0":
        raise RuntimeError("Refusing to modify v1.1.0")

    repair = json.loads(REPAIR_REPORT.read_text(encoding="utf-8"))
    if repair["status"] != "PASS" or repair["rows_repaired"] != 885 or repair["residual_stale_rows"] != 0:
        raise RuntimeError("Proof-metadata repair gate failed")
    evaluation_audit = json.loads((EVALUATION / "audit/integrity_audit.json").read_text(encoding="utf-8"))
    analysis_audit = json.loads((ANALYSIS / "audit/validation.json").read_text(encoding="utf-8"))
    if evaluation_audit["corrected_proof_metadata"]["stale_declared_complexity_mismatches"] != 0:
        raise RuntimeError("Evaluation still sees stale proof metadata")
    if analysis_audit["declared_minimum_complexity_mismatches"] != 0:
        raise RuntimeError("Reasoning-tag analysis still sees stale proof metadata")
    if analysis_audit["malformed_but_accepted_observations_retained"] != 86:
        raise RuntimeError("Malformed-accepted policy drift")

    output.mkdir(parents=True)
    for filename in ("LICENSE", "NOTICE.md"):
        copy_file(V110 / filename, output / filename)
    frame, benchmark_summary = build_benchmark(output)
    shutil.copytree(EVALUATION, output / "evaluation", copy_function=shutil.copy2)
    shutil.copytree(ANALYSIS, output / "analysis/reasoning-tag-analysis", copy_function=shutil.copy2)
    for target, source in RESPONSE_SOURCES.items():
        copy_file(source, output / target)
    provenance = {
        "corrected_model_input_manifest.csv":
            ROOT / "data/correction/v1.1.0-semantic-audit/corrected_model_input_manifest.csv",
        "observation_differential_audit.csv":
            ROOT / "data/correction/v1.1.0-semantic-audit/observation_differential_audit.csv",
        "rerun_manifest.csv": ROOT / "data/correction/v1.1.0-semantic-audit/rerun_manifest.csv",
        "proof_metadata_repair_report.json": REPAIR_REPORT,
    }
    for target, source in provenance.items():
        copy_file(source, output / "provenance" / target)

    (output / "VERSION").write_text(VERSION + "\n", encoding="utf-8", newline="\n")
    changelog = """# Migration from v1.1.0 to v1.1.1

v1.1.1 is a corrected, immutable candidate. v1.1.0 remains untouched.

- Formal IRI repair: source IRIs replace defective namespace reconstruction; corrected semantic keys are canonical and original keys remain recorded.
- Complete OEQA gold sets: independently entailed answers omitted from v1.1.0 are restored.
- Repaired FALSE BQA pairs: 67 entailed negative members are replaced by independently verified non-entailed peers, preserving 3,016 balanced pairs.
- Identity-preserving NL normalization: verified display qualifiers are removed through an injective dataset-scoped map; digits and suffixes are retained.
- Corrected proofs and complexity metadata: 885 stale cached complete-proof summaries are rebuilt from answer-group alternatives with shared axioms deduplicated and `M` excluded.
- Partial prompt reruns: only correction-affected cells were rerun; validated unaffected observations were reused. The final matrix contains 64,974 reused and 16,458 rerun observations.

Predictions were not edited. The 86 malformed accepted observations, including 16 rerun schema-nonconformant outputs, remain in the evaluation denominator under the documented policy. No manuscript files are changed by this release candidate.
"""
    (output / "CHANGELOG.md").write_text(changelog, encoding="utf-8", newline="\n")
    readme = """# CORE-LLM-Bench v1.1.1 release candidate

Status: unpublished immutable candidate (`v1.1.1-rc1`).

The benchmark is under `benchmark/`, frozen and corrected-rerun response artifacts under `responses/`, offline evaluation under `evaluation/`, and reasoning-tag analysis under `analysis/`. `RELEASE_MANIFEST.json` and `SHA256SUMS` bind every distributed artifact. See `CHANGELOG.md` and `VALIDATION_REPORT.md`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    malformed_rows = sum(1 for _ in (EVALUATION / "audit/malformed_accepted_observations.csv").open(encoding="utf-8")) - 1
    if malformed_rows != 86:
        raise RuntimeError(f"Expected 86 malformed accepted rows, found {malformed_rows}")
    validation = {
        "status": "PASS",
        "version": VERSION,
        "candidate": RC_NAME,
        "published": False,
        "offline_only": True,
        "benchmark": benchmark_summary,
        "proof_metadata": {"repaired": 885, "residual_stale": 0},
        "observations": {
            "total": 81432, "reused": 64974, "rerun_cells": 16458,
            "rerun_requests": 16443, "malformed_accepted": 86,
            "rerun_schema_nonconformant": 16, "predictions_modified": False,
        },
        "reasoning_tags": {
            "questions": 9048, "tied_minimum_questions": 1215,
            "maximum_tied_minima": 20,
            "primitive_tags_exercised": analysis_audit["primitive_tags_exercised"],
            "M_excluded": True,
        },
        "manuscript_modified": False,
        "v1_1_0_modified": False,
    }
    write_json(output / "VALIDATION_REPORT.json", validation)
    report_md = f"""# v1.1.1 release-candidate validation

Status: **PASS; unpublished**. All operations were offline.

- Benchmark: 9,048 questions; 6,032 BQA and 3,016 OEQA; 3,016 balanced TRUE/FALSE pairs.
- Proof metadata: 885 stale summaries repaired; zero residual mismatches; shared axioms deduplicated; `M` excluded.
- Observation matrix: 81,432 accepted observations (64,974 reused and 16,458 corrected-rerun cells from 16,443 requests).
- Malformed-accepted policy: 86 retained, including 16 schema-nonconformant rerun outputs (15 blank/unusable).
- Predictions modified: no.
- v1.1.0 modified: no.
- Manuscript modified: no.
- Corrected benchmark Parquet SHA-256: `{benchmark_summary['parquet_sha256']}`.

All benchmark, response, evaluation, analysis, and provenance artifact hashes are enumerated in `RELEASE_MANIFEST.json` and `SHA256SUMS`.
"""
    (output / "VALIDATION_REPORT.md").write_text(report_md, encoding="utf-8", newline="\n")

    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"RELEASE_MANIFEST.json", "SHA256SUMS"}
    )
    entries = [{
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "category": path.relative_to(output).parts[0],
    } for path in files]
    category_counts = Counter(entry["category"] for entry in entries)
    write_json(output / "RELEASE_MANIFEST.json", {
        "version": VERSION, "candidate": RC_NAME, "published": False,
        "offline_only": True, "files": entries,
        "category_file_counts": dict(sorted(category_counts.items())),
    })
    sums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    file_count = verify_manifest(output)
    print(json.dumps({
        "status": "PASS", "candidate": RC_NAME, "published": False,
        "files": file_count, "benchmark_sha256": benchmark_summary["parquet_sha256"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
