#!/usr/bin/env python3
"""Finalize the no-API v1.1.0 semantic correction and minimal rerun plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import URIRef


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "release" / "v1.1.0"
DEFAULT_AUDIT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
sys.path.insert(0, str(ROOT / "scripts"))
from phase6_materialize_release import abstract_question, prompt_hash  # noqa: E402
from prepare_v1_1_semantic_audit import local_name  # noqa: E402


MODELS = {
    "GPT-5 mini": {
        "model_id": "openai/gpt-5-mini",
        "scientific_target": "gpt-5-mini-2025-08-07",
        "api_provider": "OpenRouter",
        "provider_backend": "OpenAI",
        "configuration_hash": "4c7b1f65090715fbc99756b7d3b3c869c467e921d93976bce19cb85f91c81475",
        "observations": ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
        "temporal_caveat": "Dated scientific target routed through OpenRouter/OpenAI; returned identifier is an alias, and reruns occur later than frozen observations.",
    },
    "Gemini 2.5 Flash-Lite": {
        "model_id": "google/gemini-2.5-flash-lite",
        "scientific_target": "google/gemini-2.5-flash-lite",
        "api_provider": "OpenRouter",
        "provider_backend": "Google AI Studio",
        "configuration_hash": "c1562554bd9e252bf97356ef098edde997f813536dd8e547f35f5981a1023df3",
        "observations": ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
        "temporal_caveat": "Provider-pinned alias has no dated snapshot in the record; provider-side model drift between original and rerun is unavoidable.",
    },
    "Qwen3-30B-A3B-Instruct": {
        "model_id": "qwen/qwen3-30b-a3b-instruct-2507",
        "scientific_target": "qwen/qwen3-30b-a3b-instruct-2507",
        "api_provider": "OpenRouter",
        "provider_backend": "Alibaba",
        "configuration_hash": "68ba5917f3197234f41316cf765e9581b3421981041ccb899218968346d3b031",
        "observations": ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
        "temporal_caveat": "Dated model ID and Alibaba route can be repeated, but provider infrastructure and execution time differ.",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def split_answer(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def proof_alternative(record: dict[str, Any]) -> dict[str, Any]:
    proof = record["proofs"][0]
    return {
        "axiom_count": len(proof["functional_axiom_identities"]),
        "functional_axiom_identities": proof["functional_axiom_identities"],
        "axioms": [{"axiom": value} for value in proof["manchester_axioms"]],
        "tag_sequence": proof["reasoning_tag"],
        "proof_source": "Openllet-2.6.5-minimal-entailment-justification",
    }


def corrected_explanations(
    row: dict[str, Any], answer_iris: list[str], proof_index: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_groups = json.loads(row["answer_explanations"])
    if row["answer_type"] == "BIN":
        record = proof_index[(str(row["task_id"]), answer_iris[0])]
        alternative = proof_alternative(record)
        groups = [{
            "answer": "TRUE", "answer_iri": answer_iris[0], "alternative_count": 1,
            "alternatives": [alternative], "min_proof_axiom_count": alternative["axiom_count"],
            "max_proof_axiom_count": alternative["axiom_count"],
            "explanation_role": "queried_entailment_provenance", "proves_label": "TRUE",
        }]
    else:
        by_answer = {str(group["answer"]): group for group in original_groups}
        groups = []
        for iri in answer_iris:
            answer = local_name(iri)
            if answer in by_answer:
                group = dict(by_answer[answer])
                group["answer_iri"] = iri
                groups.append(group)
            else:
                alternative = proof_alternative(proof_index[(str(row["task_id"]), iri)])
                groups.append({
                    "answer": answer, "answer_iri": iri, "alternative_count": 1,
                    "alternatives": [alternative], "min_proof_axiom_count": alternative["axiom_count"],
                    "max_proof_axiom_count": alternative["axiom_count"],
                    "source_provenance": {"inferred.object_iri": iri},
                })
    new_alternatives = [
        group["alternatives"][0] for group in groups
        if group["alternatives"][0].get("proof_source")
    ]
    if row["answer_type"] == "BIN":
        bases = [{"axioms": [], "tag_sequence": ""}]
    else:
        original_complete = json.loads(row["complete_explanation"])
        bases = original_complete.get("minimum_explanations") or [{"axioms": [], "tag_sequence": ""}]
    completed = []
    for base in bases:
        union = {str(axiom.get("axiom", "")): axiom for axiom in base.get("axioms", [])}
        tags = [str(base.get("tag_sequence", ""))]
        for alternative in new_alternatives:
            tags.append(str(alternative.get("tag_sequence", "")))
            for axiom in alternative.get("axioms", []):
                union.setdefault(str(axiom.get("axiom", "")), axiom)
        completed.append({"axioms": list(union.values()), "tag_sequence": "".join(tags)})
    counts = [len(value["axioms"]) for value in completed]
    complete = {
        "combination_count": len(completed),
        "min_axiom_count": min(counts),
        "max_axiom_count": max(counts),
        "minimum_explanations": completed,
        "correction_note": "One deterministic minimal proof per required answer; shared axioms deduplicated by semantic identity.",
    }
    return groups, complete


def load_reasoner(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if len(frame) != 9048 or frame.task_id.nunique() != 9048:
        raise RuntimeError(f"Incomplete reasoner result {path}: {len(frame)} rows")
    if set(frame.status) != {"ok"} or set(frame.consistent.str.lower()) != {"true"}:
        raise RuntimeError(f"Reasoner failures in {path}: {frame.status.value_counts().to_dict()}")
    return frame.set_index("task_id", drop=False)


def load_mapping(dataset: str, hop: str) -> tuple[dict[URIRef, URIRef], dict[str, str]]:
    value = json.loads((ROOT / ".tmp/phase6-map-cache" / f"{dataset}_{hop}.json").read_text(encoding="utf-8"))
    return (
        {URIRef(source): URIRef(target) for source, target in value["uri_mappings"]},
        {str(key): str(target) for key, target in value["text_mappings"].items()},
    )


def audit_observation_identity() -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    reports = []
    for model, expected in MODELS.items():
        rows = read_jsonl(expected["observations"])
        accepted_rows = [row for row in rows if row.get("status") in {"usable", "malformed_response"}]
        for row in accepted_rows:
            key = (str(row["task_id"]), str(row["representation"]), model)
            if key in index:
                raise RuntimeError(f"Duplicate accepted observation {key}")
            index[key] = row
        returned_models = sorted({str(row.get("returned_model_identifier") or row.get("returned_model") or "") for row in accepted_rows})
        returned_providers = sorted({str(row.get("returned_provider") or row.get("observed_provider_backend") or "") for row in accepted_rows})
        configuration_hashes = sorted({str(row.get("configuration_hash") or "") for row in accepted_rows})
        reports.append({
            "model": model,
            "raw_observation_records": len(rows),
            "accepted_observations": len(accepted_rows),
            "superseded_technical_failures": len(rows) - len(accepted_rows),
            "requested_model": expected["model_id"],
            "scientific_target": expected["scientific_target"],
            "api_provider": expected["api_provider"],
            "expected_provider_backend": expected["provider_backend"],
            "returned_models": returned_models,
            "returned_providers": returned_providers,
            "configuration_hashes": configuration_hashes,
            "expected_configuration_hash": expected["configuration_hash"],
            "configuration_match": configuration_hashes == [expected["configuration_hash"]],
            "temporal_caveat": expected["temporal_caveat"],
        })
    if len(index) != 81432:
        raise RuntimeError(f"Expected 81,432 observations, got {len(index)}")
    return index, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    audit = args.audit.resolve()
    benchmark = pd.read_parquet(PUBLISHED / "benchmark/core_llm_bench_v1_1.parquet")
    identity = pd.read_csv(audit / "query_identity_audit.csv", dtype=str).fillna("").set_index("task_id")
    openllet = load_reasoner(audit / "openllet_results.tsv")
    structural = load_reasoner(audit / "structural_results.tsv")
    replacements = pd.read_csv(audit / "false_bqa_replacements.csv", dtype=str).fillna("").set_index("task_id")
    replacement_openllet = pd.read_csv(
        audit / "replacement_openllet_results.tsv", sep="\t", dtype=str
    ).fillna("").set_index("task_id")
    replacement_structural = pd.read_csv(
        audit / "replacement_structural_results.tsv", sep="\t", dtype=str
    ).fillna("").set_index("task_id")
    if len(replacements) != 67 or set(replacements.index) != set(replacement_openllet.index):
        raise RuntimeError("Replacement FALSE BQA audit is incomplete")
    for name, frame in (("Openllet", replacement_openllet), ("structural", replacement_structural)):
        if (
            set(frame.status) != {"ok"}
            or set(frame.consistent.str.lower()) != {"true"}
            or set(frame.entailed.str.lower()) != {"false"}
        ):
            raise RuntimeError(f"{name} did not validate all 67 replacement FALSE BQAs")
    original_inputs = pd.read_csv(PUBLISHED / "provenance/model_input_manifest.csv", dtype=str).fillna("")
    original_hashes = original_inputs.set_index(["task_id", "representation"]).input_hash.to_dict()
    observations, model_identity = audit_observation_identity()
    proof_rows = [json.loads(line) for line in (audit / "missing_entailment_explanations.jsonl").read_text(encoding="utf-8").splitlines() if line]
    proof_index = {(str(row["task_id"]), row["object_iri"]): row for row in proof_rows}

    mappings = {(dataset, hop): load_mapping(dataset, hop) for dataset in benchmark.dataset_key.unique() for hop in ("1hop", "2hop")}
    corrected_rows = []
    question_audit = []
    corrected_hashes: dict[tuple[str, str], str] = {}
    prompt_fields: dict[tuple[str, str], tuple[str, str]] = {}
    corrected_input_rows = []
    for row in benchmark.to_dict(orient="records"):
        task_id = str(row["task_id"])
        is_replacement = task_id in replacements.index
        query = (
            replacements.loc[task_id, "replacement_query"]
            if is_replacement else identity.loc[task_id, "corrected_query"]
        )
        primary = openllet.loc[task_id]
        entailed_iris = split_answer(primary.answer_iris)
        if is_replacement:
            corrected_gold = "FALSE"
            answer_iris = []
        elif row["answer_type"] == "BIN":
            corrected_gold = "TRUE" if primary.entailed.lower() == "true" else "FALSE"
            answer_iris = [identity.loc[task_id, "object_iri"]] if corrected_gold == "TRUE" else []
        else:
            answer_iris = entailed_iris
            corrected_gold = "; ".join(sorted(local_name(value) for value in answer_iris))
        nl_question = (
            replacements.loc[task_id, "replacement_nl_question"]
            if is_replacement else str(row["nl_question"]).replace(" (Genealogy)", "")
        )
        uri_map, text_map = mappings[(row["dataset_key"], row["hop"])]
        corrected_ar_question = abstract_question(query, uri_map, text_map)
        if is_replacement and corrected_ar_question != replacements.loc[task_id, "replacement_ar_question"]:
            raise RuntimeError(f"Replacement AR prompt drift for task {task_id}")
        ar_items = []
        ar_mapping_missing = []
        if row["answer_type"] == "BIN":
            corrected_ar_gold = corrected_gold
        else:
            for iri in answer_iris:
                target = uri_map.get(URIRef(iri))
                if target is None:
                    ar_mapping_missing.append(iri)
                else:
                    ar_items.append(local_name(target))
            corrected_ar_gold = "; ".join(sorted(ar_items))

        fixed = dict(row)
        fixed.update({
            "formal_query": query,
            "fs_query": query,
            "gold_answer": corrected_gold,
            "gold_answer_iris": answer_iris,
            "nl_question": nl_question,
            "ar_question": corrected_ar_question,
            "ar_gold_answer": corrected_ar_gold,
            "original_task_id": task_id,
            "original_semantic_key": row["semantic_key"],
        })
        if corrected_gold != str(row["gold_answer"]):
            groups, complete = corrected_explanations(row, answer_iris, proof_index)
            fixed["answer_explanations"] = json.dumps(groups, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fixed["complete_explanation"] = json.dumps(complete, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fixed["explanation_correction_source"] = "Openllet-2.6.5-minimal-entailment-justification"
            tag_sequences = [
                str(value.get("tag_sequence", "")).replace("M", "")
                for value in complete["minimum_explanations"]
            ]
            complexities = [len(value) for value in tag_sequences]
            distinct_tags = sorted(set("".join(tag_sequences)))
            fixed["raw_minimum_complete_primitive_tag_complexity"] = min(complexities)
            fixed["raw_maximum_complete_primitive_tag_complexity"] = max(complexities)
            fixed["primitive_reasoning_tags"] = json.dumps(distinct_tags, ensure_ascii=False, separators=(",", ":"))
            fixed["distinct_primitive_reasoning_types"] = json.dumps(distinct_tags, ensure_ascii=False, separators=(",", ":"))
            fixed["minimum_distinct_primitive_type_count"] = min(len(set(value)) for value in tag_sequences)
            fixed["maximum_distinct_primitive_type_count"] = max(len(set(value)) for value in tag_sequences)
            fixed["m_status"] = "always" if all("M" in str(value.get("tag_sequence", "")) for value in complete["minimum_explanations"]) else "sometimes" if any("M" in str(value.get("tag_sequence", "")) for value in complete["minimum_explanations"]) else "never"
            minimum_complexity = fixed["raw_minimum_complete_primitive_tag_complexity"]
            fixed["complexity_bin"] = (
                ("Low" if minimum_complexity == 1 else "Medium" if minimum_complexity == 2 else "High")
                if row["task_group"] == "BQA"
                else ("Low" if minimum_complexity <= 3 else "Medium" if minimum_complexity <= 5 else "High")
            )
        else:
            fixed["explanation_correction_source"] = "published-exact-metadata-retained"
        fixed["corrected_semantic_key"] = "sem-corrected-v1-" + stable_hash({
            "dataset": row["dataset_key"], "hop": row["hop"], "root": row["root_entity"],
            "query": query, "gold_answer_iris": answer_iris, "answer": corrected_gold,
        })
        corrected_rows.append(fixed)

        representation_values = {
            "NL": (nl_question, str(row["nl_context"])),
            "FS": (query, str(row["fs_context"])),
            "AR": (corrected_ar_question, str(row["ar_context"])),
        }
        for representation, (question, context) in representation_values.items():
            corrected_hashes[(task_id, representation)] = prompt_hash(question, context, representation, row["answer_type"])
            prompt_fields[(task_id, representation)] = (question, context)
            corrected_input_rows.append({
                "task_id": task_id,
                "corrected_semantic_key": fixed["corrected_semantic_key"],
                "dataset": row["dataset_key"],
                "hop": row["hop"],
                "task_type": row["task_group"],
                "answer_type": row["answer_type"],
                "representation": representation,
                "input_hash": corrected_hashes[(task_id, representation)],
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "prompt_template_version": "api_calls.create_context_specific_prompt@24c4520",
            })

        structural_row = replacement_structural.loc[task_id] if is_replacement else structural.loc[task_id]
        gold_changed = corrected_gold != str(row["gold_answer"])
        expected_true = str(row["gold_answer"]).upper() == "TRUE"
        bqa_valid = True if is_replacement else (
            row["answer_type"] != "BIN" or (primary.entailed.lower() == str(expected_true).lower())
        )
        structural_agrees = (
            structural_row.entailed.lower() == "false" if is_replacement else
            structural_row.entailed == primary.entailed
            if row["answer_type"] == "BIN"
            else set(split_answer(structural_row.answer_iris)) == set(answer_iris)
        )
        question_audit.append({
            "task_id": task_id,
            "dataset": row["dataset_key"],
            "hop": row["hop"],
            "task_type": row["task_group"],
            "original_query": row["formal_query"],
            "corrected_query": query,
            "query_changed": query != row["formal_query"],
            "original_gold": row["gold_answer"],
            "corrected_gold": corrected_gold,
            "gold_changed": gold_changed,
            "openllet_semantically_valid": bqa_valid if row["answer_type"] == "BIN" else not gold_changed,
            "original_openllet_semantically_valid": bqa_valid if row["answer_type"] == "BIN" else not gold_changed,
            "corrected_openllet_semantically_valid": True,
            "structural_reasoner_agrees": structural_agrees,
            "nl_question_changed": nl_question != row["nl_question"],
            "ar_question_changed": corrected_ar_question != row["ar_question"],
            "ar_mapping_missing": ";".join(ar_mapping_missing),
            "original_answer_explanations_retained": not gold_changed,
            "explanation_action": "retain_exact_metadata" if not gold_changed else "regenerate_for_corrected_complete_gold",
            "replacement_false_bqa": is_replacement,
        })

    corrected = pd.DataFrame(corrected_rows)
    questions = pd.DataFrame(question_audit)
    corrected.to_parquet(audit / "core_llm_bench_v1_1_corrected.parquet", index=False)
    questions.to_csv(audit / "question_semantic_audit.csv", index=False, lineterminator="\n")
    pd.DataFrame(corrected_input_rows).to_csv(
        audit / "corrected_model_input_manifest.csv", index=False, lineterminator="\n"
    )

    differential = []
    nl_reuse_audit = []
    for row in corrected_rows:
        task_id = str(row["task_id"])
        qa = questions[questions.task_id == task_id].iloc[0]
        is_replacement = bool(qa.replacement_false_bqa)
        for representation in ("NL", "FS", "AR"):
            old_hash = original_hashes[(task_id, representation)]
            new_hash = corrected_hashes[(task_id, representation)]
            if is_replacement:
                reason = "invalid_false_bqa_replaced"
                rerun = True
                reuse_basis = "none"
            elif representation == "FS" and qa.query_changed:
                reason = "formal_query_iri_corrected"
                rerun = True
                reuse_basis = "none"
            elif representation == "NL" and qa.nl_question_changed:
                reason = "presentation_only_genealogy_qualifier_removed"
                rerun = False
                reuse_basis = "qualified_non_exact_prompt_transfer"
            elif representation == "AR" and (qa.ar_question_changed or qa.ar_mapping_missing):
                reason = "abstract_identity_or_mapping_changed"
                rerun = True
                reuse_basis = "none"
            else:
                reason = "prompt_unchanged_reuse_frozen_response"
                rerun = False
                reuse_basis = "exact_input_hash"
            if representation == "NL" and qa.nl_question_changed:
                original_question = str(benchmark.loc[benchmark.task_id.astype(str) == task_id, "nl_question"].iloc[0])
                corrected_question = prompt_fields[(task_id, "NL")][0]
                presentation_only = (
                    not is_replacement
                    and original_question.replace(" (Genealogy)", "") == corrected_question
                )
                nl_reuse_audit.append({
                    "task_id": task_id,
                    "dataset": row["dataset_key"],
                    "task_type": row["task_group"],
                    "category": "B" if presentation_only else "A",
                    "original_nl_question": original_question,
                    "corrected_nl_question": corrected_question,
                    "original_prompt_hash": old_hash,
                    "corrected_prompt_hash": new_hash,
                    "exact_prompt_match": old_hash == new_hash,
                    "semantic_proposition_changed": not presentation_only,
                    "reuse_decision": "reuse_with_methodological_qualification" if presentation_only else "rerun",
                    "audit_basis": (
                        "only the erroneous parenthetical namespace qualifier was deleted; question and context otherwise exact"
                        if presentation_only else "replacement FALSE question changes the queried object"
                    ),
                })
            for model, config in MODELS.items():
                observation = observations[(task_id, representation, model)]
                exact_original = str(observation.get("input_hash")) == old_hash
                usage = (observation.get("raw_provider_response") or {}).get("usage") or {}
                differential.append({
                    "dataset": row["dataset"],
                    "task_id": task_id,
                    "corrected_semantic_key": row["corrected_semantic_key"],
                    "task_type": row["task_group"],
                    "answer_type": row["answer_type"],
                    "representation": representation,
                    "model": model,
                    "model_id": config["model_id"],
                    "provider": config["api_provider"],
                    "provider_backend": config["provider_backend"],
                    "configuration_hash": config["configuration_hash"],
                    "original_prompt_hash": old_hash,
                    "corrected_prompt_hash": new_hash,
                    "observation_input_hash_matches_original": exact_original,
                    "exact_corrected_prompt_match": old_hash == new_hash,
                    "reuse_frozen_observation": not rerun,
                    "reuse_basis": reuse_basis,
                    "gold_only_change": bool(qa.gold_changed and not rerun),
                    "reason": reason,
                    "rerun_required": rerun,
                    "expected_api_call_count": int(rerun),
                    "historical_input_tokens": int(usage.get("prompt_tokens") or 0),
                    "historical_output_tokens": int(usage.get("completion_tokens") or 0),
                    "temporal_provider_caveat": config["temporal_caveat"],
                })
    diff = pd.DataFrame(differential)
    diff.to_csv(audit / "observation_differential_audit.csv", index=False, lineterminator="\n")
    nl_audit = pd.DataFrame(nl_reuse_audit)
    if len(nl_audit) != 5352 or (nl_audit.category == "B").sum() != 5285 or (nl_audit.category == "A").sum() != 67:
        raise RuntimeError(f"Expected 5,285 category-B NL prompts, got {nl_audit.category.value_counts().to_dict()}")
    nl_audit.to_csv(audit / "nl_category_b_reuse_audit.csv", index=False, lineterminator="\n")
    reruns = diff[diff.rerun_required].copy()
    dedup_keys = ["model", "configuration_hash", "corrected_prompt_hash"]
    reruns["deduplication_group"] = reruns.groupby(dedup_keys, sort=True).ngroup().map(lambda value: f"request-{value + 1:06d}")
    reruns["deduplicated_request"] = ~reruns.duplicated(dedup_keys)
    reruns["expected_api_call_count"] = reruns.deduplicated_request.astype(int)
    manifest_columns = [
        "dataset", "task_id", "corrected_semantic_key", "task_type", "answer_type",
        "representation", "model", "model_id",
        "provider", "provider_backend", "configuration_hash", "original_prompt_hash",
        "corrected_prompt_hash", "reason", "deduplication_group", "deduplicated_request",
        "expected_api_call_count", "historical_input_tokens", "historical_output_tokens",
        "temporal_provider_caveat",
    ]
    reruns[manifest_columns].to_csv(audit / "rerun_manifest.csv", index=False, lineterminator="\n")
    pd.DataFrame(model_identity).to_csv(audit / "model_identity_audit.csv", index=False, lineterminator="\n")

    summary = {
        "status": "complete-plan-only-no-llm-calls",
        "questions": len(questions),
        "positive_bqa": int(((corrected.task_group == "BQA") & (corrected.gold_answer == "TRUE")).sum()),
        "negative_bqa": int(((corrected.task_group == "BQA") & (corrected.gold_answer == "FALSE")).sum()),
        "oeqa": int((corrected.task_group == "OEQA").sum()),
        "query_iri_changes": int(questions.query_changed.sum()),
        "gold_answer_changes": int(questions.gold_changed.sum()),
        "nl_prompt_changes": int(questions.nl_question_changed.sum()),
        "fs_prompt_changes": int(questions.query_changed.sum()),
        "ar_prompt_changes": int(questions.ar_question_changed.sum()),
        "ar_mapping_failures": int(questions.ar_mapping_missing.astype(bool).sum()),
        "replacement_false_bqa": int(questions.replacement_false_bqa.sum()),
        "category_b_nl_prompts_reused_with_qualification": int((nl_audit.category == "B").sum()),
        "original_openllet_valid_questions": int(questions.openllet_semantically_valid.sum()),
        "corrected_openllet_valid_questions": len(questions),
        "symbolic_explanation_rows_corrected": int(questions.gold_changed.sum()),
        "generated_entailment_proofs_audited": len(proof_rows),
        "new_entailment_proofs_applied": len(proof_rows) - len(replacements),
        "structural_reasoner_exact_agreements": int(questions.structural_reasoner_agrees.sum()),
        "structural_reasoner_nonagreements": int((~questions.structural_reasoner_agrees).sum()),
        "existing_observations_audited": len(diff),
        "frozen_observations_reused": int((~diff.rerun_required).sum()),
        "rerun_observation_cells": len(reruns),
        "deduplicated_expected_api_calls": int(reruns.expected_api_call_count.sum()),
        "rerun_counts_by_model_representation": reruns.groupby(["model", "representation"]).size().to_dict(),
        "published_v1_1_modified": False,
        "llm_api_calls_made": 0,
    }
    required = {
        "questions": 9048,
        "positive_bqa": 3016,
        "negative_bqa": 3016,
        "oeqa": 3016,
        "replacement_false_bqa": 67,
        "category_b_nl_prompts_reused_with_qualification": 5285,
        "existing_observations_audited": 81432,
        "frozen_observations_reused": 64974,
        "rerun_observation_cells": 16458,
        "deduplicated_expected_api_calls": 16443,
    }
    failures = {key: (summary.get(key), value) for key, value in required.items() if summary.get(key) != value}
    if failures:
        raise RuntimeError(f"Minimum-rerun accounting failed: {failures}")
    serializable = dict(summary)
    serializable["rerun_counts_by_model_representation"] = {
        f"{model}|{representation}": count
        for (model, representation), count in summary["rerun_counts_by_model_representation"].items()
    }
    (audit / "semantic_correction_report.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
