#!/usr/bin/env python3
"""Export CORE-LLM-Bench as one Parquet row per question-hop instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from release_build import dataset_card, statistics, write_manifest_and_checksums
from release_profiles import PROFILES, VERSION, artifacts_for, get_profile
from validate_release import (
    BENCHMARK_DIR,
    ROOT,
    iter_flat_rows,
    load_member,
    load_reasoning_metadata,
    validate_prepared_package,
    validate_release,
)


TAG_ORDER = "DHTSAJNE∩¬IFVYQRCLUM"


def export(output_dir: Path, profile_name: str = "full") -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face export requires pyarrow; install requirements-release.txt"
        ) from exc

    profile = get_profile(profile_name)
    validation = validate_release(profile.name)
    metadata = load_reasoning_metadata(BENCHMARK_DIR / "reasoning_metadata.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "core_llm_bench_v1_0.parquet"
    writer: pq.ParquetWriter | None = None
    dataset_counts: dict[str, int] = {}
    row_total = 0

    try:
        for artifact in artifacts_for(profile):
            dataset = artifact.dataset
            hop = artifact.hop
            zip_name = artifact.zip_name
            member = artifact.member
            batch: list[dict[str, Any]] = []
            groups = load_member(BENCHMARK_DIR / zip_name, member)
            for item in iter_flat_rows(groups, dataset, hop):
                group, qa = item["group"], item["qa"]
                task_id = str(qa["Task ID"])
                meta = metadata[(dataset, hop, task_id)]
                task_type = "BQA" if str(group["Answer Type"]).upper() == "BIN" else "OEQA"
                answer = str(qa["Answer"])
                batch.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "hop": hop,
                        "task_type": task_type,
                        "reasoning_task": str(group["Task Type"]),
                        "answer_type": str(group["Answer Type"]),
                        "binary_label": answer.upper() if task_type == "BQA" else None,
                        "gold_answer": answer,
                        "ar_gold_answer": str(qa["ABS Answer"]),
                        "root_entity": str(group["Root Entity"]),
                        "nl_question": str(qa["NL Question"]),
                        "nl_context": str(group["NL Context"]),
                        "fs_query": str(qa["SPARQL Query"]),
                        "fs_context": str(group["OWL Context"]),
                        "ar_question": str(qa["ABS Question"]),
                        "ar_context": str(group["ABS Context"]),
                        "minimum_explanation": str(qa["Minimum Explanation"]),
                        "explanations": str(qa["Explanations"]),
                        "explanation_count": str(qa["Explanation Count"]),
                        "min_tag_length": int(meta["min_tag_length"]),
                        "max_tag_length": int(meta["max_tag_length"]),
                        "reasoning_tags": [tag for tag in TAG_ORDER if tag in meta["reasoning_tags"]],
                        "linked_positive_task_id": meta["linked_positive_task_id"] or None,
                        "benchmark_version": VERSION,
                        "source_package": zip_name,
                        "source_member": member,
                    }
                )
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table)
            dataset_counts[f"{dataset}_{hop}"] = len(batch)
            row_total += len(batch)
    finally:
        if writer is not None:
            writer.close()

    report = {
        "benchmark_version": VERSION,
        "profile": profile.name,
        "rows": row_total,
        "dataset_hop_counts": dataset_counts,
        "parquet": parquet_path.name,
        "bytes": parquet_path.stat().st_size,
        "columns": table.schema.names,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(dataset_card(profile), encoding="utf-8")
    stats = statistics(
        profile,
        validation["artifacts"],
        validation["instantiated_reasoning_counts"],
    )
    write_manifest_and_checksums(output_dir, profile, stats)
    validate_prepared_package(output_dir, profile.name, "huggingface")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: release/huggingface/<profile>",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "release" / "huggingface" / args.profile
    )
    try:
        report = export(output_dir, args.profile)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"HUGGING FACE EXPORT FAILED: {exc}")
        return 1
    print(f"Hugging Face export ready: {report['rows']:,} rows")
    print(f"Parquet: {output_dir / report['parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
