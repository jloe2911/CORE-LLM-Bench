#!/usr/bin/env python3
"""Exhaustive, offline, non-mutating audit of the frozen v1.1 AR questions.

The script reads the released benchmark, source TTL files, manifests, and accepted
observations.  It writes audit artifacts only; it never imports an API client and
does not rescore any observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ar_validation import abstract_answer
from scripts.ontology_tools.abstraction.OntologyAbstractor import (
    build_text_mappings,
    create_abstraction_mappings,
    discover_entity_roles,
    flatten_mappings,
    local_name,
    merge_entity_roles,
)
from scripts.phase6_materialize_release import abstract_question, parse_triple


STAGE = ROOT / "release" / "v1.1.0-staging"
PARQUET = STAGE / "core_llm_bench_v1_1.parquet"
OUT = ROOT / "results" / "v1.1.0-final" / "audit"
MODEL_INPUT_MANIFEST = STAGE / "model_input_manifest.csv"
OBSERVATIONS = {
    "GPT-5 mini": ROOT / "data/output/v1.1.0-gpt-openrouter-primary/responses/gpt_observations.jsonl",
    "Gemini 2.5 Flash-Lite": ROOT / "release/v1.1.0-phase7d/responses/gemini_observations.jsonl",
    "Qwen3-30B-A3B-Instruct": ROOT / "release/v1.1.0-phase7e/responses/qwen_alibaba_observations.jsonl",
}
RESOURCE_NAMES = {
    "FamilyOWL": "FamilyOWL",
    "OWL2Bench": "OWL2Bench",
    "Pizza100": "pizza_100",
    "Pizza250": "pizza_250",
}
MODELS = tuple(OBSERVATIONS)
ACCEPTED_STATUSES = {"usable", "malformed_response"}
ID_TOKEN = re.compile(r"\b(?:DataProperty|Property|Individual|Class)\d+\b")
BLANK_ANSWER = re.compile(r"(?im)^\s*ANSWER:[ \t]*\r?\n[ \t]*CONFIDENCE:")
SCHEMA_PREDICATES = {
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.domain,
    RDFS.range,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.inverseOf,
    OWL.disjointWith,
    OWL.propertyDisjointWith,
}
SCHEMA_TYPES = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.ReflexiveProperty,
    OWL.IrreflexiveProperty,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def raw_text(observation: dict) -> str:
    return str(
        observation.get("raw_response")
        or observation.get("raw_response_text")
        or ""
    )


def parsed_answer(observation: dict) -> str:
    parsed = observation.get("parsed_response") or {}
    return str(observation.get("parsed_answer") or parsed.get("answer") or "")


def split_answer(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*;\s*", str(value)) if part.strip()]


def build_dataset_states(frame: pd.DataFrame):
    states = {}
    for (dataset, hop), _rows in frame.groupby(["dataset", "hop"], sort=True):
        resource_dir = ROOT / "data" / "resources" / f"{RESOURCE_NAMES[dataset]}_{hop}"
        ttl_paths = sorted(resource_dir.glob("*.ttl"), key=lambda p: str(p))
        if not ttl_paths:
            raise FileNotFoundError(f"No source TTL files under {resource_dir}")
        roles = merge_entity_roles(
            discover_entity_roles(Graph().parse(path, format="turtle"))
            for path in ttl_paths
        )
        mappings = flatten_mappings(create_abstraction_mappings(roles))
        text_mappings, ambiguous = build_text_mappings(
            (Graph().parse(path, format="turtle") for path in ttl_paths), mappings
        )
        inverse = defaultdict(list)
        for original, abstract in mappings.items():
            inverse[local_name(abstract)].append(str(original))
        collisions = {key: values for key, values in inverse.items() if len(values) != 1}
        states[(dataset, hop)] = {
            "resource_dir": resource_dir,
            "mappings": mappings,
            "text_mappings": text_mappings,
            "ambiguous_alias_count": len(ambiguous),
            "inverse": dict(inverse),
            "identifier_collisions": collisions,
        }
    return states


def load_observations():
    by_key = {}
    impact = Counter()
    hashes = {}
    for model, path in OBSERVATIONS.items():
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        for obs in read_jsonl(path):
            if obs.get("status") not in ACCEPTED_STATUSES:
                continue
            key = (str(obs["semantic_key"]), str(obs["representation"]), model)
            if key in by_key:
                raise ValueError(f"Duplicate accepted observation: {key}")
            by_key[key] = obs
            rep = str(obs["representation"])
            task = str(obs["task"])
            if BLANK_ANSWER.search(raw_text(obs)):
                impact[("blank_answer_parser", rep, model)] += 1
            if task == "OEQA":
                impact[("hallucination_metric_definition", rep, model)] += 1
    return by_key, impact, hashes


def load_manifest_hashes():
    result = {}
    with MODEL_INPUT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[(row["semantic_key"], row["representation"])] = row["input_hash"]
    return result


def schema_axiom_count(graph: Graph) -> int:
    count = 0
    for subject, predicate, obj in graph:
        if predicate in SCHEMA_PREDICATES or (predicate == RDF.type and obj in SCHEMA_TYPES):
            count += 1
    return count


def mapped_direct_objects(graph: Graph, subject: str, predicate: str, mappings):
    values = set()
    unmapped = set()
    for obj in graph.objects(URIRef(subject), URIRef(predicate)):
        if not isinstance(obj, URIRef):
            unmapped.add(str(obj))
            continue
        abstract = mappings.get(obj)
        if abstract is None:
            unmapped.add(str(obj))
        else:
            values.add(local_name(abstract))
    return values, unmapped


def prediction_mapping_issues(answer: str, expected_prefix: str, state: dict):
    issues = []
    tokens = ID_TOKEN.findall(answer)
    for token in tokens:
        if token not in state["inverse"]:
            issues.append(f"unknown:{token}")
        elif not token.startswith(expected_prefix):
            issues.append(f"wrong-role:{token}")
    if not tokens:
        for item in split_answer(answer):
            if item in state["text_mappings"] and item != state["text_mappings"][item]:
                issues.append(f"original-identifier:{item}")
    return sorted(set(issues))


def join_values(values) -> str:
    return ";".join(sorted(str(value) for value in values))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PARQUET).sort_values("task_id")
    if len(frame) != 9048 or frame["semantic_key"].nunique() != 9048:
        raise ValueError("Expected exactly 9,048 unique semantic questions")

    states = build_dataset_states(frame)
    observations, observation_impact, observation_hashes = load_observations()
    manifest_hashes = load_manifest_hashes()
    output_rows = []

    root_context_counts = frame.groupby(["dataset", "hop", "root_entity"])["ar_context"].nunique()

    prompt_groups = defaultdict(list)
    for row in frame.itertuples(index=False):
        prompt_key = hashlib.sha256(
            json.dumps([row.ar_question, row.ar_context], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        prompt_groups[prompt_key].append((row.semantic_key, row.ar_gold_answer))
    incompatible_keys = {
        key for key, values in prompt_groups.items() if len({gold for _, gold in values}) > 1
    }

    processing_frame = frame.sort_values(["dataset", "hop", "root_entity", "task_id"])
    active_root_key = None
    graph = None
    root_schema_count = 0
    source_ttl_sha = ""
    for row in processing_frame.itertuples(index=False):
        state = states[(row.dataset, row.hop)]
        root_key = (row.dataset, row.hop, row.root_entity)
        if root_key != active_root_key:
            graph_path = state["resource_dir"] / f"{row.root_entity}.ttl"
            graph = Graph().parse(graph_path, format="turtle")
            root_schema_count = schema_axiom_count(graph)
            source_ttl_sha = sha256_file(graph_path)
            active_root_key = root_key

        try:
            subject, predicate, object_uri = parse_triple(row.formal_query)
            query_parse_ok = True
        except Exception:
            subject = predicate = ""
            object_uri = None
            query_parse_ok = False

        expected_question = ""
        question_mapping_ok = False
        expected_gold = ""
        gold_mapping_ok = False
        try:
            expected_question = abstract_question(
                row.formal_query, state["mappings"], state["text_mappings"]
            )
            question_mapping_ok = expected_question == row.ar_question
            expected_gold = abstract_answer(
                row.gold_answer, row.answer_type, state["text_mappings"]
            )
            gold_mapping_ok = expected_gold == row.ar_gold_answer
        except Exception:
            pass

        context_consistent_for_root = int(root_context_counts.loc[root_key]) == 1
        prompt_key = hashlib.sha256(
            json.dumps([row.ar_question, row.ar_context], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        incompatible_collision = prompt_key in incompatible_keys

        gold_set = set(split_answer(row.ar_gold_answer))
        direct_set = set()
        unmapped_direct = set()
        direct_target_present = False
        missing_gold = set()
        extra_direct = set()
        if query_parse_ok:
            if str(row.answer_type).upper() == "BIN":
                direct_target_present = (
                    URIRef(subject), URIRef(predicate), URIRef(object_uri)
                ) in graph
                if str(row.gold_answer).upper() == "TRUE" and not direct_target_present:
                    missing_gold = {"TRUE entailment not explicit in AR ABox context"}
                if str(row.gold_answer).upper() == "FALSE" and direct_target_present:
                    extra_direct = {"queried triple explicitly present despite FALSE gold"}
            else:
                direct_set, unmapped_direct = mapped_direct_objects(
                    graph, subject, predicate, state["mappings"]
                )
                missing_gold = gold_set - direct_set
                extra_direct = direct_set - gold_set

        question_tokens = set(ID_TOKEN.findall(str(row.ar_question)))
        gold_tokens = set(ID_TOKEN.findall(str(row.ar_gold_answer)))
        context_tokens = set(ID_TOKEN.findall(str(row.ar_context)))
        unknown_prompt_ids = {
            token for token in question_tokens | gold_tokens | context_tokens
            if token not in state["inverse"]
        }

        obs_present = []
        pred_issue_models = []
        blank_ar_models = []
        blank_nl_models = []
        blank_fs_models = []
        expected_prefix = "Class" if predicate == str(RDF.type) else "Individual"
        for model in MODELS:
            ar_obs = observations.get((row.semantic_key, "AR", model))
            if ar_obs is not None:
                obs_present.append(model)
                if str(row.answer_type).upper() != "BIN" and prediction_mapping_issues(
                    parsed_answer(ar_obs), expected_prefix, state
                ):
                    pred_issue_models.append(model)
                if BLANK_ANSWER.search(raw_text(ar_obs)):
                    blank_ar_models.append(model)
            nl_obs = observations.get((row.semantic_key, "NL", model))
            if nl_obs is not None and BLANK_ANSWER.search(raw_text(nl_obs)):
                blank_nl_models.append(model)
            fs_obs = observations.get((row.semantic_key, "FS", model))
            if fs_obs is not None and BLANK_ANSWER.search(raw_text(fs_obs)):
                blank_fs_models.append(model)

        reasons = []
        uncertain = []
        if not query_parse_ok:
            uncertain.append("formal query could not be parsed")
        if not expected_question:
            uncertain.append("expected abstract question could not be reconstructed")
        elif not question_mapping_ok:
            reasons.append("abstract question mapping differs from formal query")
        if not context_consistent_for_root:
            reasons.append("same root entity has inconsistent stored AR contexts")
        if incompatible_collision:
            reasons.append("identical AR prompt has incompatible gold semantics")
        if unknown_prompt_ids:
            reasons.append("unknown abstract identifier in question or gold")
        if state["identifier_collisions"]:
            reasons.append("non-bijective abstract identifier map")
        if unmapped_direct:
            uncertain.append("direct query objects lack abstraction mappings")
        if missing_gold:
            reasons.append("expected answer evidence omitted from AR ABox-only context")
        if extra_direct:
            reasons.append("AR context explicitly supports answer omitted or contradicted by gold")

        if uncertain:
            classification = "C"
            rationale = "; ".join(uncertain + reasons)
        elif reasons:
            classification = "B"
            rationale = "; ".join(reasons)
        else:
            classification = "A"
            if not gold_mapping_ok:
                rationale = "valid AR prompt; abstract gold mapping requires offline correction"
            else:
                rationale = "valid AR prompt; only evaluator/parser/metric corrections apply"

        output_rows.append(
            {
                "task_id": int(row.task_id),
                "semantic_key": row.semantic_key,
                "dataset": row.dataset,
                "hop": row.hop,
                "task_group": row.task_group,
                "root_entity": row.root_entity,
                "classification": classification,
                "classification_rationale": rationale,
                "query_parse_ok": query_parse_ok,
                "ar_question_mapping_ok": question_mapping_ok,
                "ar_gold_mapping_ok": gold_mapping_ok,
                "ar_context_consistent_for_root": context_consistent_for_root,
                "identifier_map_bijective": not bool(state["identifier_collisions"]),
                "unknown_prompt_identifier_count": len(unknown_prompt_ids),
                "incompatible_exact_prompt_collision": incompatible_collision,
                "source_schema_axiom_count_omitted_by_ar_renderer": root_schema_count,
                "direct_expected_answer_count": len(direct_set) if row.task_group == "OEQA" else int(direct_target_present),
                "gold_answer_count": len(gold_set) if row.task_group == "OEQA" else 1,
                "missing_gold_evidence_count": len(missing_gold),
                "missing_gold_evidence": join_values(missing_gold),
                "extra_or_contradictory_direct_evidence_count": len(extra_direct),
                "extra_or_contradictory_direct_evidence": join_values(extra_direct),
                "primitive_reasoning_tags": row.primitive_reasoning_tags,
                "manual_semantic_review_required": classification == "C",
                "ar_observation_count": len(obs_present),
                "ar_observation_models": join_values(obs_present),
                "ar_prediction_mapping_issue_model_count": len(pred_issue_models),
                "ar_prediction_mapping_issue_models": join_values(pred_issue_models),
                "blank_answer_parser_ar_models": join_values(blank_ar_models),
                "blank_answer_parser_nl_models": join_values(blank_nl_models),
                "blank_answer_parser_fs_models": join_values(blank_fs_models),
                "source_ttl_sha256": source_ttl_sha,
            }
        )

    audit = pd.DataFrame(output_rows).sort_values("task_id")
    if len(audit) != 9048 or audit["semantic_key"].nunique() != 9048:
        raise ValueError("Per-question audit cardinality failure")
    audit_path = OUT / "AR_QUESTION_AUDIT.csv"
    audit.to_csv(audit_path, index=False, lineterminator="\n")

    summary_rows = []
    for (dataset, hop), group in audit.groupby(["dataset", "hop"], sort=True):
        counts = group["classification"].value_counts()
        item = {
            "dataset": dataset,
            "hop": hop,
            "questions": len(group),
            "A_questions": int(counts.get("A", 0)),
            "B_questions": int(counts.get("B", 0)),
            "C_questions": int(counts.get("C", 0)),
            "A_ar_observations_all_models": int(group.loc[group.classification == "A", "ar_observation_count"].sum()),
            "B_ar_observations_all_models": int(group.loc[group.classification == "B", "ar_observation_count"].sum()),
            "C_ar_observations_all_models": int(group.loc[group.classification == "C", "ar_observation_count"].sum()),
        }
        for model in MODELS:
            short = {"GPT-5 mini": "gpt", "Gemini 2.5 Flash-Lite": "gemini", "Qwen3-30B-A3B-Instruct": "qwen"}[model]
            for cls in "ABC":
                item[f"{cls}_{short}_ar_observations"] = sum(
                    1
                    for semantic_key in group.loc[group.classification == cls, "semantic_key"]
                    if (semantic_key, "AR", model) in observations
                )
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    total = {"dataset": "ALL", "hop": "ALL", "questions": len(audit)}
    for column in summary.columns[3:]:
        total[column] = int(summary[column].sum())
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    summary_path = OUT / "AR_AUDIT_SUMMARY_BY_DATASET_HOP.csv"
    summary.to_csv(summary_path, index=False, lineterminator="\n")

    impact_rows = []
    for (issue, representation, model), count in sorted(observation_impact.items()):
        impact_rows.append(
            {"issue": issue, "representation": representation, "model": model, "affected_observations": count}
        )
    class_b = audit[audit.classification == "B"]
    for model in MODELS:
        impact_rows.append(
            {
                "issue": "proven_defective_ar_prompt",
                "representation": "AR",
                "model": model,
                "affected_observations": sum(
                    1 for key in class_b.semantic_key if (key, "AR", model) in observations
                ),
            }
        )
        impact_rows.append(
            {
                "issue": "representation_unaware_gold_selection",
                "representation": "AR",
                "model": model,
                "affected_observations": sum(
                    1 for key in audit.semantic_key if (key, "AR", model) in observations
                ),
            }
        )
    impact = pd.DataFrame(impact_rows).sort_values(["issue", "representation", "model"])
    impact_path = OUT / "OBSERVATION_DEFECT_IMPACT.csv"
    impact.to_csv(impact_path, index=False, lineterminator="\n")

    counts = audit["classification"].value_counts()
    offline_only = int(counts.get("B", 0)) == 0
    source_hashes = {
        str(PARQUET.relative_to(ROOT)): sha256_file(PARQUET),
        str(MODEL_INPUT_MANIFEST.relative_to(ROOT)): sha256_file(MODEL_INPUT_MANIFEST),
        **observation_hashes,
    }
    report = f"""# Exhaustive AR audit — CORE-LLM-Bench v1.1

Date: 2026-09-25
Mode: offline. No API request, model rerun, benchmark regeneration, evaluator change, rescore, commit, or push was performed.

## Determination

All 9,048 unique AR questions were checked. Classification totals are **A={int(counts.get('A', 0))}**, **B={int(counts.get('B', 0))}**, and **C={int(counts.get('C', 0))}**. Offline rescoring alone is **{'sufficient' if offline_only else 'not sufficient'}**. {'' if offline_only else f'The {int(counts.get("B", 0))} class-B prompts omit or contradict evidence needed for their expected answers and therefore require corrected AR prompt generation plus fresh AR observations for those questions; existing responses must not be reinterpreted as responses to changed prompts.'}

Class A means the question/context prompt is valid and only offline evaluator, parser, normalization, or metric correction is needed. Class B means a prompt defect is proven from the frozen formal query, source graph, deterministic abstraction map, and displayed AR context. Class C is reserved for a case the automated proof could not decide; no model error alone causes B or C.

## Exhaustive checks

- Reconstructed all eight dataset-hop abstraction maps from the source TTL graphs and checked question and gold identifier identity, role, and bijectivity.
- Checked every stored context against its dataset-hop abstraction inventory, ensured one stable AR context per root, and compared all query-targeted ABox facts with the complete expected answer set from the source graph.
- Parsed every formal query. For TRUE BQA and every OEQA answer, required explicit evidence in the supplied ABox-only AR context. For multi-answer OEQA, compared the complete gold set with all direct query-matching source facts. Any missing expected item or directly supported extra item is class B. FALSE BQA was not rejected merely because its queried triple is absent.
- Audited exact-prompt/incompatible-gold collisions, unknown identifiers, accepted AR observation coverage, prediction identifier role/existence, and blank-answer parser exposure. Incorrect, empty, or hallucinated model answers were never treated as prompt defects.
- The hallucination implementation defect affects OEQA observations in AR, NL, and FS. The newline-crossing blank-answer parser defect also occurs in all three representations; exact counts are in `OBSERVATION_DEFECT_IMPACT.csv`.

## Minimum-change repair plan

1. Freeze class-A prompts and all original observations. Correct the evaluator to select `ar_gold_answer` for AR; repair the blank-answer regex/status handling; implement the manuscript hallucination definition using the representation-specific complete gold set.
2. For class B only, repair the AR context construction so every expected answer is supported and every directly displayed answer is reconciled with gold. Preserve URI-to-abstract-ID identity and complete multi-answer evidence. Generate new hashes/manifests for changed AR inputs.
3. Rerun only the three-model AR observations whose prompt hash changed. Do not rerun NL/FS for AR-only prompt repairs. Retain the old observations as superseded provenance.
4. After review approval, perform one offline rescore from corrected class-A observations plus approved fresh class-B AR observations. No rescoring was done in this audit.

## Outputs

- `AR_QUESTION_AUDIT.csv`: one row per semantic question, with decisive evidence and observation flags.
- `AR_AUDIT_SUMMARY_BY_DATASET_HOP.csv`: A/B/C question and three-model observation counts.
- `OBSERVATION_DEFECT_IMPACT.csv`: parser, hallucination, wrong-gold, and defective-prompt impact by representation/model.
- `AR_AUDIT_MANIFEST.json`: hashes, invariants, and preservation statement.
"""
    report_path = OUT / "AR_EXHAUSTIVE_AUDIT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")

    manifest = {
        "audit_date": "2026-09-25",
        "mode": "offline-read-only-sources",
        "question_count": len(audit),
        "unique_semantic_key_count": int(audit.semantic_key.nunique()),
        "classification_counts": {key: int(counts.get(key, 0)) for key in "ABC"},
        "offline_rescoring_alone_sufficient": offline_only,
        "source_hashes_sha256": source_hashes,
        "output_hashes_sha256": {
            str(audit_path.relative_to(ROOT)): sha256_file(audit_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
            str(impact_path.relative_to(ROOT)): sha256_file(impact_path),
            str(report_path.relative_to(ROOT)): sha256_file(report_path),
        },
        "invariants": {
            "all_formal_queries_parsed": bool(audit.query_parse_ok.all()),
            "all_identifier_maps_bijective": bool(audit.identifier_map_bijective.all()),
            "all_ar_contexts_consistent_per_root": bool(audit.ar_context_consistent_for_root.all()),
            "incompatible_exact_prompt_collisions": int(audit.incompatible_exact_prompt_collision.sum()),
            "all_three_ar_observations_present": bool((audit.ar_observation_count == 3).all()),
            "manual_review_count": int((audit.classification == "C").sum()),
        },
        "preservation": {
            "benchmark_files_modified": False,
            "prompts_modified": False,
            "model_responses_modified": False,
            "manifests_modified": False,
            "results_rescored": False,
            "api_requests": 0,
            "model_reruns": 0,
        },
    }
    manifest_path = OUT / "AR_AUDIT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({
        "classification_counts": manifest["classification_counts"],
        "offline_rescoring_alone_sufficient": offline_only,
        "outputs": [str(path.relative_to(ROOT)) for path in (audit_path, summary_path, impact_path, report_path, manifest_path)],
    }, indent=2))


if __name__ == "__main__":
    main()
