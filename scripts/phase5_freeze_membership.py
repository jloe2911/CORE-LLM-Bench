#!/usr/bin/env python3
"""Freeze corrected semantic v1.1 membership without assigning public IDs.

This phase is deliberately offline.  It consumes the accepted Phase 4
symbolic staging area, preserves BQA pairs, records every eligible sampling
group decision, compares semantic membership with v1.0, and inventories saved
prediction reuse.  It never calls a model and never writes v1.0 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "llm_pipeline"))

from benchmark_corrections import (  # noqa: E402
    add_group_keys,
    audit_bqa_pairing,
    canonical_dataset_name,
    parse_ask_triple,
    prepare_eligible_rows,
    uri_local_name,
)
from stratified_sampling import (  # noqa: E402
    COMPLEXITY_BIN_COLUMN,
    CORRECTED_COMPLEXITY_COLUMN,
    SAMPLING_MODE_COLUMN,
    TASK_CLASS_COLUMN,
    select_corrected_groups,
)


PHASE4 = ROOT / "data" / "output_v1_1_symbolic"
DEFAULT_STAGE = ROOT / "data" / "output_v1_1_staging" / "phase5"
LEGACY_OUTPUT = ROOT / "data" / "output"
PREDICTION_ROOT = LEGACY_OUTPUT / "final_benchmark_llm_results"
DATASETS = {
    "Family": "FamilyOWL",
    "OWL2Bench": "OWL2Bench",
    "Pizza100": "pizza_100",
    "Pizza250": "pizza_250",
}
MODEL_NAMES = {
    "openai_gpt_5_mini_2025_08_07": "GPT-5 mini",
    "openrouter_google_gemini_2_5_flash_lite": "Gemini 2.5 Flash-Lite",
    "openrouter_qwen_qwen3_30b_a3b_instruct_2507":
        "Qwen3-30B-A3B-Instruct",
}
SETTING_DIRS = {"nl": "NL", "sparql": "FS", "abs": "AR"}
MEMBERSHIP_FIELDS = [
    "semantic_key",
    "source_task_id",
    "sampling_group_key",
    "dataset",
    "hop",
    "root",
    "normalized_query",
    "gold_answer",
    "task_type",
    "answer_type",
    "raw_complexity",
    "complexity_bin",
    "sampling_mode",
    "abox_size",
    "abox_bin",
    "source_provenance_key",
    "corrected_explanation_metadata",
    "corrected_provenance_metadata",
    "attribution_evidence",
    "v1_membership",
    "nl_question_sha256",
    "nl_context_sha256",
    "fs_query_sha256",
    "fs_context_sha256",
    "gold_answer_sha256",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str | None) -> str:
    return "" if value is None else sha256_bytes(value.encode("utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def normalize_query(value: object) -> str:
    return " ".join(str(value).split())


def normalize_answer(value: object, answer_type: object) -> str:
    text = str(value).strip()
    if str(answer_type).upper() == "BIN":
        return text.upper()
    return ";".join(sorted(part.strip() for part in text.split(";") if part.strip()))


def semantic_payload(
    dataset: str,
    hop: str,
    row: pd.Series | dict[str, Any],
) -> dict[str, str]:
    answer_type = str(row["Answer Type"])
    return {
        "dataset": canonical_dataset_name(dataset),
        "hop": str(hop),
        "task": "BQA" if answer_type.upper() == "BIN" else "OEQA",
        "root": str(row["Root Entity"]),
        "query": normalize_query(row["SPARQL Query"]),
        "answer": normalize_answer(row["Answer"], answer_type),
    }


def semantic_key(
    dataset: str,
    hop: str,
    row: pd.Series | dict[str, Any],
) -> str:
    payload = json.dumps(
        semantic_payload(dataset, hop, row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sem-v1.1-" + sha256_text(payload)


def group_key_text(value: object) -> str:
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _unique_by_task_id(frame: pd.DataFrame, label: str) -> dict[str, dict[str, str]]:
    if frame["Task ID"].duplicated().any():
        duplicates = frame.loc[frame["Task ID"].duplicated(), "Task ID"].head()
        raise ValueError(f"Duplicate Task IDs in {label}: {duplicates.tolist()}")
    return {
        str(row["Task ID"]): {column: str(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    }


def load_phase4_rows() -> pd.DataFrame:
    path = PHASE4 / "audit" / "eligible_source_pool_complexity.csv"
    frame = read_csv_text(path)
    frame["primary_complexity"] = frame["primary_complexity"].astype(int)
    return frame


def source_provenance_key(metadata: str, fallback: str) -> str:
    parsed = json.loads(metadata)
    key = parsed.get("source_explanation_key")
    if key:
        return str(key)
    answers = parsed.get("answers", [])
    if answers:
        payload = json.dumps(answers, ensure_ascii=False, sort_keys=True)
        return "oeqa-source-" + sha256_text(payload)
    return fallback


def prepare_partition(
    dataset: str,
    hop: str,
    phase4: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = DATASETS[dataset]
    source_path = PHASE4 / dataset_dir / hop / "SPARQL_questions.csv"
    source = read_csv_text(source_path).drop_duplicates()
    eligible, _ = prepare_eligible_rows(source, dataset, hop)
    eligible = eligible.reset_index(drop=True)
    metadata = phase4[
        (phase4["dataset"] == dataset) & (phase4["hop"] == hop)
    ].copy()
    metadata_by_id = _unique_by_task_id(
        metadata.rename(columns={"task_id": "Task ID"}),
        f"Phase 4 metadata {dataset}/{hop}",
    )
    source_ids = set(eligible["Task ID"])
    metadata_ids = set(metadata_by_id)
    if source_ids != metadata_ids:
        raise ValueError(
            f"Eligible source/Phase 4 mismatch for {dataset}/{hop}: "
            f"source-only={len(source_ids - metadata_ids)}, "
            f"metadata-only={len(metadata_ids - source_ids)}"
        )

    eligible[CORRECTED_COMPLEXITY_COLUMN] = [
        int(metadata_by_id[task_id]["primary_complexity"])
        for task_id in eligible["Task ID"]
    ]
    eligible = add_group_keys(eligible, dataset, hop)
    selected_groups, decisions = select_corrected_groups(
        eligible, benchmark_fraction=0.25, random_state=42
    )
    decision_by_group = {
        row["Sampling Group Key"]: row for _, row in decisions.iterrows()
    }
    eligible["Selected"] = eligible["Sampling Group Key"].isin(selected_groups)
    eligible[SAMPLING_MODE_COLUMN] = [
        decision_by_group[key][SAMPLING_MODE_COLUMN]
        for key in eligible["Sampling Group Key"]
    ]
    eligible[COMPLEXITY_BIN_COLUMN] = [
        decision_by_group[key][COMPLEXITY_BIN_COLUMN]
        for key in eligible["Sampling Group Key"]
    ]
    eligible["ABox Bin"] = [
        int(decision_by_group[key]["ABox Bin"])
        for key in eligible["Sampling Group Key"]
    ]
    eligible["_phase4"] = [metadata_by_id[value] for value in eligible["Task ID"]]
    return eligible, decisions


def load_v1_partition(dataset: str, hop: str) -> dict[str, dict[str, str]]:
    dataset_dir = DATASETS[dataset]
    base = LEGACY_OUTPUT / dataset_dir / hop
    raw = read_csv_text(base / "SPARQL_questions_sampling.csv")
    nl = read_csv_text(base / "SPARQL_questions_sampling_nl.csv")
    nl_by_id = _unique_by_task_id(nl, f"v1 NL {dataset}/{hop}")
    result: dict[str, dict[str, str]] = {}
    for _, row in raw.iterrows():
        key = semantic_key(dataset, hop, row)
        if key in result:
            raise ValueError(f"Duplicate v1 semantic identity: {key}")
        item = {column: str(row[column]) for column in raw.columns}
        item["NL Question"] = nl_by_id[item["Task ID"]].get("Question", "")
        result[key] = item
    return result


def prediction_inventory() -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    """Index saved model rows by dataset/hop/model/setting/semantic key."""

    inventory: dict[
        tuple[str, str, str, str, str], dict[str, str]
    ] = {}
    reverse_dirs = {value: key for key, value in DATASETS.items()}
    for path in sorted(PREDICTION_ROOT.rglob("LATEST_checkpoint.csv")):
        relative = path.relative_to(PREDICTION_ROOT)
        if len(relative.parts) < 4:
            continue
        dataset_hop = relative.parts[0]
        setting = SETTING_DIRS.get(relative.parts[-2])
        if setting is None:
            continue
        dataset = None
        hop = None
        for dataset_dir, canonical in reverse_dirs.items():
            for candidate_hop in ("1hop", "2hop"):
                if dataset_hop == f"{dataset_dir}_{candidate_hop}":
                    dataset, hop = canonical, candidate_hop
                    break
            if dataset is not None:
                break
        if dataset is None or dataset not in DATASETS:
            continue
        frame = read_csv_text(path)
        prefixes = [
            prefix
            for prefix in MODEL_NAMES
            if f"{prefix}_final_answer" in frame.columns
        ]
        for _, row in frame.iterrows():
            key = semantic_key(dataset, hop, row)
            row_dict = {column: str(row[column]) for column in frame.columns}
            for prefix in prefixes:
                inventory[(dataset, hop, MODEL_NAMES[prefix], setting, key)] = {
                    **row_dict,
                    "_response_column": f"{prefix}_response",
                    "_source_path": path.relative_to(ROOT).as_posix(),
                }
    return inventory


def frozen_v1_inputs() -> dict[tuple[str, str, str], dict[str, str]]:
    """Read canonical inputs from the published, immutable v1.0 ZIPs."""

    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for dataset, dataset_dir in DATASETS.items():
        zip_path = ROOT / "final_benchmark" / f"{dataset_dir}.zip"
        with zipfile.ZipFile(zip_path) as archive:
            for hop in ("1hop", "2hop"):
                member = f"{dataset_dir}_{hop}.json"
                with archive.open(member) as handle:
                    groups = json.load(handle)
                for group in groups:
                    for qa in group["QAs"]:
                        row = {
                            "Root Entity": group["Root Entity"],
                            "Answer Type": group["Answer Type"],
                            "SPARQL Query": qa["SPARQL Query"],
                            "Answer": qa["Answer"],
                        }
                        key = semantic_key(dataset, hop, row)
                        frozen = {
                            "NL Question": str(qa["NL Question"]),
                            "NL Context SHA256": sha256_text(str(group["NL Context"])),
                            "FS Query": str(qa["SPARQL Query"]),
                            "FS Context SHA256": sha256_text(str(group["OWL Context"])),
                            "Answer": str(qa["Answer"]),
                        }
                        previous = result.setdefault((dataset, hop, key), frozen)
                        if previous != frozen:
                            raise ValueError(
                                f"Conflicting frozen v1 input for {dataset}/{hop}/{key}"
                            )
                del groups
                gc.collect()
    return result


def write_csv(rows: Iterable[dict[str, Any]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 6) if total else 0.0


def grouped_bin_report(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        grouped[key][str(row["complexity_bin"])] += 1
    report: dict[str, Any] = {}
    for key, counts in sorted(grouped.items()):
        total = sum(counts.values())
        report["/".join(key)] = {
            label: {
                "count": counts[label],
                "percentage": percentage(counts[label], total),
            }
            for label in ("Low", "Medium", "High")
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if ROOT not in stage.parents:
        raise ValueError("Phase 5 staging must remain inside this repository")

    phase4 = load_phase4_rows()
    predictions = prediction_inventory()
    frozen_inputs = frozen_v1_inputs()
    v1_by_partition: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    all_eligible: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for hop in ("1hop", "2hop"):
            eligible, decisions = prepare_partition(dataset, hop, phase4)
            all_eligible.append(eligible)
            decisions = decisions.copy()
            decisions.insert(0, "dataset", dataset)
            decisions.insert(1, "hop", hop)
            all_decisions.append(decisions)
            v1 = load_v1_partition(dataset, hop)
            v1_by_partition[(dataset, hop)] = v1
            for _, source in eligible[eligible["Selected"]].iterrows():
                key = semantic_key(dataset, hop, source)
                metadata = source["_phase4"]
                decision = decisions[
                    decisions["Sampling Group Key"].map(group_key_text)
                    == group_key_text(source["Sampling Group Key"])
                ].iloc[0]
                old = v1.get(key)
                frozen = frozen_inputs.get((dataset, hop, key)) if old else None
                if old and frozen is None:
                    raise ValueError(
                        f"v1 semantic row absent from published ZIP: {dataset}/{hop}/{key}"
                    )
                nl_question = frozen.get("NL Question") if frozen else None
                nl_context_sha = frozen.get("NL Context SHA256") if frozen else ""
                fs_context_sha = frozen.get("FS Context SHA256") if frozen else ""
                gold = str(source["Answer"])
                explanation = str(metadata["complexity_json"])
                provenance = str(metadata["provenance_json"])
                selected_rows.append(
                    {
                        "semantic_key": key,
                        "source_task_id": str(source["Task ID"]),
                        "sampling_group_key": group_key_text(
                            source["Sampling Group Key"]
                        ),
                        "dataset": dataset,
                        "hop": hop,
                        "root": str(source["Root Entity"]),
                        "normalized_query": normalize_query(source["SPARQL Query"]),
                        "gold_answer": gold,
                        "task_type": "BQA"
                        if str(source["Answer Type"]).upper() == "BIN"
                        else "OEQA",
                        "answer_type": str(source["Answer Type"]),
                        "raw_complexity": int(source[CORRECTED_COMPLEXITY_COLUMN]),
                        "complexity_bin": str(source[COMPLEXITY_BIN_COLUMN]),
                        "sampling_mode": str(source[SAMPLING_MODE_COLUMN]),
                        "abox_size": int(float(source["Size of ontology ABox"])),
                        "abox_bin": int(decision["ABox Bin"]),
                        "source_provenance_key": source_provenance_key(
                            provenance, key
                        ),
                        "corrected_explanation_metadata": explanation,
                        "corrected_provenance_metadata": provenance,
                        "attribution_evidence": str(
                            metadata["attribution_evidence_json"]
                        ),
                        "v1_membership": "retained" if old else "new",
                        "nl_question_sha256": sha256_text(nl_question),
                        "nl_context_sha256": nl_context_sha,
                        "fs_query_sha256": sha256_text(
                            str(source["SPARQL Query"])
                        ),
                        "fs_context_sha256": fs_context_sha,
                        "gold_answer_sha256": sha256_text(gold),
                        "_nl_question": nl_question,
                        "_nl_context_sha256": nl_context_sha,
                        "_fs_context_sha256": fs_context_sha,
                        "_frozen_gold_answer": frozen.get("Answer") if frozen else None,
                        "_frozen_fs_query": normalize_query(frozen.get("FS Query"))
                        if frozen else None,
                    }
                )

    selected_rows.sort(key=lambda row: row["semantic_key"])
    if len({row["semantic_key"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("Selected semantic keys are not globally unique")

    membership_rows = [
        {field: row[field] for field in MEMBERSHIP_FIELDS}
        for row in selected_rows
    ]
    membership_path = stage / "semantic_membership.csv"
    write_csv(membership_rows, MEMBERSHIP_FIELDS, membership_path)
    jsonl_path = stage / "semantic_membership.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in membership_rows
        ),
        encoding="utf-8",
        newline="\n",
    )

    decision_rows: list[dict[str, Any]] = []
    for frame in all_decisions:
        for _, row in frame.iterrows():
            decision_rows.append(
                {
                    "dataset": row["dataset"],
                    "hop": row["hop"],
                    "task_type": row[TASK_CLASS_COLUMN],
                    "sampling_group_key": group_key_text(row["Sampling Group Key"]),
                    "raw_complexity": int(row[CORRECTED_COMPLEXITY_COLUMN]),
                    "complexity_bin": row[COMPLEXITY_BIN_COLUMN],
                    "abox_size": int(float(row["Size of ontology ABox"])),
                    "abox_bin": int(row["ABox Bin"]),
                    "sampling_mode": row[SAMPLING_MODE_COLUMN],
                    "selected": bool(row["Selected"]),
                }
            )
    decision_rows.sort(
        key=lambda row: (row["dataset"], row["hop"], row["task_type"], row["sampling_group_key"])
    )
    decision_fields = [
        "dataset", "hop", "task_type", "sampling_group_key",
        "raw_complexity", "complexity_bin", "abox_size", "abox_bin",
        "sampling_mode", "selected",
    ]
    decisions_path = stage / "eligible_group_decisions.csv"
    write_csv(decision_rows, decision_fields, decisions_path)

    selected_by_key = {row["semantic_key"]: row for row in selected_rows}
    comparison_rows: list[dict[str, Any]] = []
    comparison_counts: Counter[tuple[str, str, str, str]] = Counter()
    identity_counts: Counter[tuple[str, str]] = Counter()
    for dataset in DATASETS:
        for hop in ("1hop", "2hop"):
            old = v1_by_partition[(dataset, hop)]
            old_keys = set(old)
            new_keys = {
                key for key, row in selected_by_key.items()
                if row["dataset"] == dataset and row["hop"] == hop
            }
            for key in sorted(old_keys | new_keys):
                if key in old_keys and key in new_keys:
                    classification = "retained"
                elif key in old_keys:
                    classification = "removed"
                else:
                    classification = "new"
                source = selected_by_key.get(key)
                legacy = old.get(key)
                task = (
                    source["task_type"] if source else
                    semantic_payload(dataset, hop, legacy)["task"]
                )
                comparison_counts[(dataset, hop, task, classification)] += 1
                retained = classification == "retained"
                field_identity = {
                    "nl_question_byte_identical": retained
                    and source["_nl_question"] == legacy.get("NL Question"),
                    "nl_context_byte_identical": retained
                    and bool(source["_nl_context_sha256"]),
                    "fs_query_byte_identical": retained
                    and str(source["_frozen_fs_query"])
                    == str(source["normalized_query"]),
                    "fs_context_byte_identical": retained
                    and bool(source["_fs_context_sha256"]),
                    "gold_answer_byte_identical": retained
                    and source["gold_answer"] == source["_frozen_gold_answer"],
                }
                if retained:
                    for field, identical in field_identity.items():
                        identity_counts[(field, "identical" if identical else "changed_or_missing")] += 1
                comparison_rows.append(
                    {
                        "semantic_key": key,
                        "dataset": dataset,
                        "hop": hop,
                        "task_type": task,
                        "classification": classification,
                        **field_identity,
                    }
                )
    comparison_fields = [
        "semantic_key", "dataset", "hop", "task_type", "classification",
        "nl_question_byte_identical", "nl_context_byte_identical",
        "fs_query_byte_identical", "fs_context_byte_identical",
        "gold_answer_byte_identical",
    ]
    comparison_path = stage / "v1_vs_v1_1_membership.csv"
    write_csv(comparison_rows, comparison_fields, comparison_path)

    reuse_rows: list[dict[str, Any]] = []
    reuse_counts: Counter[tuple[str, str, str]] = Counter()
    for source in selected_rows:
        for model in MODEL_NAMES.values():
            for setting in ("NL", "FS", "AR"):
                reusable = False
                reason = "AR rerun required by Phase 3"
                saved = predictions.get(
                    (source["dataset"], source["hop"], model, setting, source["semantic_key"])
                )
                if setting != "AR":
                    if source["v1_membership"] != "retained":
                        reason = "new v1.1 membership"
                    elif saved is None:
                        reason = "no saved prediction row"
                    else:
                        if setting == "NL":
                            input_identical = (
                                saved.get("NL Question", "") == source["_nl_question"]
                                and sha256_text(saved.get("NL Context", ""))
                                == source["_nl_context_sha256"]
                                and saved.get("Answer", "")
                                == source["_frozen_gold_answer"]
                            )
                        else:
                            input_identical = (
                                normalize_query(saved.get("SPARQL Query", ""))
                                == source["normalized_query"]
                                and sha256_text(saved.get("OWL Context", ""))
                                == source["_fs_context_sha256"]
                                and saved.get("Answer", "")
                                == source["_frozen_gold_answer"]
                            )
                        response = saved.get(saved.get("_response_column", ""), "")
                        reusable = input_identical and bool(response.strip()) and not response.lstrip().upper().startswith("[ERROR]")
                        reason = (
                            "exact input identity and saved prediction"
                            if reusable else
                            "input differs, is missing, or saved response is unavailable"
                        )
                status = "reusable prediction available" if reusable else "rerun required"
                reuse_counts[(model, setting, status)] += 1
                reuse_rows.append(
                    {
                        "semantic_key": source["semantic_key"],
                        "dataset": source["dataset"],
                        "hop": source["hop"],
                        "model": model,
                        "setting": setting,
                        "status": status,
                        "reason": reason,
                    }
                )
    reuse_fields = [
        "semantic_key", "dataset", "hop", "model", "setting", "status", "reason"
    ]
    reuse_path = stage / "prediction_reuse_plan.csv"
    write_csv(reuse_rows, reuse_fields, reuse_path)

    selected_frame = pd.DataFrame(selected_rows)
    per_dataset_hop: dict[str, Any] = {}
    for (dataset, hop), frame in selected_frame.groupby(["dataset", "hop"], sort=True):
        bqa = frame[frame["task_type"] == "BQA"]
        oeqa = frame[frame["task_type"] == "OEQA"]
        per_dataset_hop[f"{dataset}/{hop}"] = {
            "BQA_rows": len(bqa),
            "BQA_TRUE": sum(str(value).upper() == "TRUE" for value in bqa["gold_answer"]),
            "BQA_FALSE": sum(str(value).upper() == "FALSE" for value in bqa["gold_answer"]),
            "BQA_groups": bqa["sampling_group_key"].nunique(),
            "OEQA_rows": len(oeqa),
            "total_rows": len(frame),
        }

    pairing_totals = Counter()
    domainconcept_targets = 0
    eligible_frame = pd.concat(all_eligible, ignore_index=True)
    for (dataset, hop), selected in eligible_frame[eligible_frame["Selected"]].groupby(
        [eligible_frame["_phase4"].map(lambda value: value["dataset"]),
         eligible_frame["_phase4"].map(lambda value: value["hop"])],
        sort=True,
    ):
        source = eligible_frame[
            (eligible_frame["_phase4"].map(lambda value: value["dataset"]) == dataset)
            & (eligible_frame["_phase4"].map(lambda value: value["hop"]) == hop)
        ]
        audit = audit_bqa_pairing(source, selected, dataset, hop)
        for item in audit:
            pairing_totals.update(
                {
                    "paired_groups": item.paired_groups,
                    "unpaired_positives": item.broken_sample_positives,
                    "unpaired_negatives": item.broken_sample_negatives,
                    "source_unpaired_positives": item.missing_source_pair_positives,
                    "source_unpaired_negatives": item.missing_source_pair_negatives,
                }
            )
        if dataset in {"Pizza100", "Pizza250"}:
            for _, row in selected.iterrows():
                if str(row["Answer Type"]).upper() != "BIN":
                    continue
                domainconcept_targets += int(
                    uri_local_name(parse_ask_triple(row["SPARQL Query"])[2])
                    == "DomainConcept"
                )
    if pairing_totals["unpaired_positives"] or pairing_totals["unpaired_negatives"]:
        raise ValueError("A valid BQA source pair was split")
    if domainconcept_targets:
        raise ValueError("Pizza DomainConcept BQA target survived Phase 5")

    mode_counts = Counter(row["sampling_mode"] for row in decision_rows)
    selected_mode_counts = Counter(
        row["sampling_mode"] for row in decision_rows if row["selected"]
    )
    collapsed_abox_complexity: Counter[tuple[str, str, int, str]] = Counter(
        (
            row["dataset"], row["hop"], row["abox_bin"],
            row["complexity_bin"],
        )
        for row in decision_rows
    )
    accepted_sparse_breakdown = Counter(
        f"{dataset}/{hop}"
        for (dataset, hop, _abox, _complexity), count
        in collapsed_abox_complexity.items()
        if count == 1
    )
    expected_sparse_breakdown = {
        "Family/1hop": 0,
        "Family/2hop": 0,
        "OWL2Bench/1hop": 6,
        "OWL2Bench/2hop": 3,
        "Pizza100/1hop": 7,
        "Pizza100/2hop": 7,
        "Pizza250/1hop": 9,
        "Pizza250/2hop": 8,
    }
    observed_sparse_breakdown = {
        key: accepted_sparse_breakdown[key]
        for key in expected_sparse_breakdown
    }
    if observed_sparse_breakdown != expected_sparse_breakdown:
        raise ValueError(
            "Accepted 40-stratum audit could not be reproduced: "
            f"{observed_sparse_breakdown}"
        )
    comparison_report: dict[str, Any] = {}
    for (dataset, hop, task, classification), count in sorted(comparison_counts.items()):
        comparison_report.setdefault(f"{dataset}/{hop}/{task}", {})[
            classification
        ] = count
    reuse_report: dict[str, Any] = {}
    for (model, setting, status), count in sorted(reuse_counts.items()):
        reuse_report.setdefault(model, {}).setdefault(setting, {})[status] = count

    report = {
        "phase": 5,
        "status": "staged-for-scientific-review",
        "primary_complexity": "minimum complete primitive-tag count",
        "final_complexity_bins": {
            "BQA": {"Low": "1", "Medium": "2", "High": "3+"},
            "OEQA": {"Low": "1-3", "Medium": "4-5", "High": "6+"},
        },
        "sampling": {
            "benchmark_fraction": 0.25,
            "random_state": 42,
            "fallback_algorithm": (
                "Within each dataset/hop/task partition, ABox-bin x complexity-bin "
                "strata of size >=2 are stratified normally. Groups in singleton "
                "combined strata are pooled only by the same task and complexity "
                "bin. Fallback pools of size >=2 are stratified; remaining singleton "
                "pools are ranked by SHA-256('42|' + canonical group key) and assigned "
                "to the benchmark only as needed to minimize absolute deviation from "
                "round-half-up(0.25 * partition groups). Exact half-count ties round "
                "toward the benchmark. Complexity bins never change."
            ),
            "eligible_group_mode_counts": dict(sorted(mode_counts.items())),
            "selected_group_mode_counts": dict(sorted(selected_mode_counts.items())),
            "accepted_audit_sparse_abox_complexity_pairs": {
                "count": sum(observed_sparse_breakdown.values()),
                "by_dataset_hop": observed_sparse_breakdown,
                "note": (
                    "This reproduces the accepted task-collapsed ABox-bin x "
                    "complexity-bin audit. Required fallback status is assigned "
                    "after adding task to the full stratum, so its eligible-group "
                    "count is larger."
                ),
            },
        },
        "membership": {
            "total_rows": len(selected_rows),
            "per_dataset_hop": per_dataset_hop,
            "complexity_by_task": grouped_bin_report(selected_rows, ["task_type"]),
            "complexity_by_dataset_hop_task": grouped_bin_report(
                selected_rows, ["dataset", "hop", "task_type"]
            ),
        },
        "bqa_pair_audit": {
            **dict(pairing_totals),
            "pizza_domainconcept_target_count": domainconcept_targets,
        },
        "v1_comparison": comparison_report,
        "retained_byte_identity": {
            field: {
                status: count
                for (candidate, status), count in identity_counts.items()
                if candidate == field
            }
            for field in sorted({field for field, _ in identity_counts})
        },
        "prediction_reuse": reuse_report,
        "estimated_remaining_model_calls": sum(
            count
            for (model, setting, status), count in reuse_counts.items()
            if status == "rerun required"
        ),
        "scientific_ambiguities": [],
        "prohibitions_confirmed": {
            "sequential_public_ids_assigned": False,
            "llm_or_api_calls": False,
            "final_release_created": False,
            "v1_0_artifacts_written": False,
            "phase5_committed": False,
        },
    }
    report_path = stage / "phase5_report.json"
    write_json(report, report_path)

    artifacts = [
        membership_path,
        jsonl_path,
        decisions_path,
        comparison_path,
        reuse_path,
        report_path,
    ]
    inputs = [
        PHASE4 / "audit" / "eligible_source_pool_complexity.csv",
        *[
            PHASE4 / dataset_dir / hop / "SPARQL_questions.csv"
            for dataset_dir in DATASETS.values()
            for hop in ("1hop", "2hop")
        ],
    ]
    manifest = {
        "phase": 5,
        "semantic_key_definition": (
            "SHA-256 of canonical JSON(dataset, hop, task, root, normalized query, normalized answer)"
        ),
        "public_ids_assigned": False,
        "parameters": {"benchmark_fraction": 0.25, "random_state": 42},
        "inputs": [file_record(path) for path in sorted(inputs)],
        "artifacts": [file_record(path) for path in artifacts],
    }
    write_json(manifest, stage / "membership_manifest.json")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
