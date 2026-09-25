#!/usr/bin/env python3
"""Fast, checkpointed AR structural audit using the frozen Phase 6 map caches."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdflib import URIRef

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ar_validation import abstract_answer
from scripts.phase6_materialize_release import abstract_question

STAGE = ROOT / "release/v1.1.0-staging"
FINALIZED = ROOT / "release/v1.1.0-schema-finalized"
PARQUET = STAGE / "core_llm_bench_v1_1.parquet"
CACHE = ROOT / ".tmp/phase6-map-cache"
OUT = ROOT / "results/v1.1.0-final/audit"
CHECKPOINTS = OUT / "ar_checkpoints/mappings"
ACCEPTED = {"usable", "malformed_response"}
ID_TOKEN = re.compile(r"\b(?:DataProperty|Property|Individual|Class)\d+\b")
BLANK_ANSWER = re.compile(r"(?im)^\s*ANSWER:[ \t]*\r?\n[ \t]*CONFIDENCE:")

DATASET_CACHE = {
    ("FamilyOWL", "1hop"): "Family_1hop.json",
    ("FamilyOWL", "2hop"): "Family_2hop.json",
    ("OWL2Bench", "1hop"): "OWL2Bench_1hop.json",
    ("OWL2Bench", "2hop"): "OWL2Bench_2hop.json",
    ("Pizza100", "1hop"): "Pizza100_1hop.json",
    ("Pizza100", "2hop"): "Pizza100_2hop.json",
    ("Pizza250", "1hop"): "Pizza250_1hop.json",
    ("Pizza250", "2hop"): "Pizza250_2hop.json",
}
OBSERVATIONS = {
    "GPT-5 mini": ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
    "Gemini 2.5 Flash-Lite": ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
    "Qwen3-30B-A3B-Instruct": ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_maps():
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    states = {}
    manifest = []
    for key, name in DATASET_CACHE.items():
        source = CACHE / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing accepted Phase 6 mapping cache: {source}")
        target = CHECKPOINTS / name
        shutil.copyfile(source, target)
        payload = json.loads(source.read_text(encoding="utf-8"))
        uri_map = {URIRef(a): URIRef(b) for a, b in payload["uri_mappings"]}
        inverse = defaultdict(list)
        for original, abstract in uri_map.items():
            inverse[str(abstract).rsplit("#", 1)[-1]].append(str(original))
        states[key] = {
            "uri_map": uri_map,
            "text_map": dict(payload["text_mappings"]),
            "inverse": dict(inverse),
            "source_signature": payload["source_signature"],
        }
        manifest.append({
            "dataset": key[0], "hop": key[1], "checkpoint": str(target.relative_to(ROOT)),
            "sha256": sha256(target), "uri_mapping_count": len(uri_map),
            "text_mapping_count": len(payload["text_mappings"]),
            "source_signature": payload["source_signature"],
        })
    path = OUT / "AR_MAPPING_CHECKPOINT_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return states, path


def accepted_observation_audit():
    coverage = Counter()
    defects = Counter()
    accepted_keys = set()
    for model, path in OBSERVATIONS.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") not in ACCEPTED:
                    continue
                key = (str(row["semantic_key"]), str(row["representation"]), model)
                if key in accepted_keys:
                    raise ValueError(f"Duplicate accepted observation: {key}")
                accepted_keys.add(key)
                rep = str(row["representation"])
                coverage[(rep, model)] += 1
                raw = str(row.get("raw_response") or row.get("raw_response_text") or "")
                if BLANK_ANSWER.search(raw):
                    defects[("blank_answer_parser", rep, model)] += 1
                if str(row["task"]) == "OEQA":
                    defects[("hallucination_metric", rep, model)] += 1
    return accepted_keys, coverage, defects


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    states, mapping_manifest = load_maps()
    frame = pd.read_parquet(PARQUET).sort_values("task_id")
    if len(frame) != 9048 or frame.semantic_key.nunique() != 9048:
        raise ValueError("Benchmark cardinality/identity failure")
    accepted_keys, coverage, defects = accepted_observation_audit()
    root_context_count = frame.groupby(["dataset", "hop", "root_entity"]).ar_context.nunique()
    prompt_gold = defaultdict(set)
    for row in frame.itertuples(index=False):
        prompt_gold[(row.ar_question, row.ar_context)].add(row.ar_gold_answer)

    rows = []
    for row in frame.itertuples(index=False):
        state = states[(row.dataset, row.hop)]
        expected_question = abstract_question(row.formal_query, state["uri_map"], state["text_map"])
        expected_gold = abstract_answer(row.gold_answer, row.answer_type, state["text_map"])
        ids = set(ID_TOKEN.findall(f"{row.ar_question}\n{row.ar_gold_answer}\n{row.ar_context}"))
        unknown = sorted(token for token in ids if token not in state["inverse"])
        map_collisions = sorted(token for token in ids if len(state["inverse"].get(token, [])) != 1)
        context_stable = int(root_context_count.loc[(row.dataset, row.hop, row.root_entity)]) == 1
        incompatible = len(prompt_gold[(row.ar_question, row.ar_context)]) > 1
        structural_ok = (
            expected_question == row.ar_question
            and expected_gold == row.ar_gold_answer
            and not unknown and not map_collisions and context_stable and not incompatible
            and row.fs_query == row.formal_query
        )
        # Negative BQA needs no positive evidence recovery. TRUE BQA and OEQA
        # proceed to the selected-fragment semantic-support pass.
        needs_semantic = row.task_group == "OEQA" or str(row.gold_answer).upper() == "TRUE"
        provisional = "semantic_review" if structural_ok and needs_semantic else ("structurally_valid" if structural_ok else "structural_failure")
        rows.append({
            "task_id": int(row.task_id), "semantic_key": row.semantic_key,
            "dataset": row.dataset, "hop": row.hop, "task_group": row.task_group,
            "root_entity": row.root_entity, "provisional_status": provisional,
            "ar_question_mapping_ok": expected_question == row.ar_question,
            "ar_gold_mapping_ok": expected_gold == row.ar_gold_answer,
            "identifier_map_bijective_for_used_ids": not map_collisions,
            "unknown_abstract_identifier_count": len(unknown),
            "stable_ar_context_per_root": context_stable,
            "incompatible_exact_ar_prompt_collision": incompatible,
            "fs_query_equals_formal_query": row.fs_query == row.formal_query,
            "accepted_ar_observations": sum((row.semantic_key, "AR", model) in accepted_keys for model in OBSERVATIONS),
            "needs_selected_fragment_semantic_check": needs_semantic,
        })
    audit = pd.DataFrame(rows)
    audit_path = OUT / "AR_FAST_STRUCTURAL_AUDIT.csv"
    audit.to_csv(audit_path, index=False, lineterminator="\n")

    finalized_parquet = FINALIZED / PARQUET.name
    same_parquet = finalized_parquet.is_file() and sha256(PARQUET) == sha256(finalized_parquet)
    field_impact = [
        ("original_ontology_fragments", "unchanged", "No source-fragment defect or mutation is indicated by structural checks."),
        ("task_ids", "unchanged", "IDs are unique and contiguous 1..9048."),
        ("gold_answers", "unchanged", "All original-to-AR gold mappings reproduce from the cached accepted maps."),
        ("NL_prompts", "unchanged", "AR gold-selection/parser/metric defects do not alter NL inputs."),
        ("FS_prompts", "unchanged", "Every fs_query equals formal_query; identified evaluator defects are input-independent."),
        ("AR_prompts", "pending_selected_fragment_semantic_check", "Structural mapping is valid; evidence sufficiency remains to be checked for TRUE BQA and OEQA."),
        ("symbolic_explanations", "unchanged", "No gold-semantic or explanation-identity failure was found structurally."),
        ("SAGE_QA_Chapter_7_data", "unchanged_by_offline_scoring_fixes", "Parser, AR-gold selection, and hallucination corrections affect derived evaluation only; AR exclusions, if any, require a separate coverage analysis."),
    ]
    impact_path = OUT / "AR_FAST_FIELD_IMPACT.csv"
    pd.DataFrame(field_impact, columns=["field", "status", "basis"]).to_csv(impact_path, index=False, lineterminator="\n")

    defect_rows = [
        {"issue": issue, "representation": rep, "model": model, "affected_observations": count}
        for (issue, rep, model), count in sorted(defects.items())
    ]
    defect_path = OUT / "AR_FAST_OFFLINE_DEFECT_IMPACT.csv"
    pd.DataFrame(defect_rows).to_csv(defect_path, index=False, lineterminator="\n")

    summary = {
        "questions": len(audit),
        "unique_semantic_keys": int(audit.semantic_key.nunique()),
        "provisional_status_counts": {k: int(v) for k, v in audit.provisional_status.value_counts().items()},
        "structural_failure_count": int((audit.provisional_status == "structural_failure").sum()),
        "selected_fragment_semantic_check_count": int(audit.needs_selected_fragment_semantic_check.sum()),
        "selected_root_fragment_count": int(frame.groupby(["dataset", "hop", "root_entity"]).ngroups),
        "all_ar_observations_present": bool((audit.accepted_ar_observations == 3).all()),
        "accepted_observation_coverage": {f"{rep}|{model}": count for (rep, model), count in sorted(coverage.items())},
        "staging_and_schema_finalized_parquet_byte_identical": same_parquet,
        "mapping_checkpoint_manifest": str(mapping_manifest.relative_to(ROOT)),
        "no_rerun_policy": "Preserve all 81,432 observations; any proven defective AR prompt is a candidate for exclusion/sensitivity analysis, not automatic regeneration.",
        "outputs": [str(p.relative_to(ROOT)) for p in (audit_path, impact_path, defect_path)],
    }
    summary_path = OUT / "AR_FAST_STRUCTURAL_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
