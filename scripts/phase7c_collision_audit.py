#!/usr/bin/env python3
"""Offline Phase 7C collision-safe NL rebuild and exact-input audit.

This script reads the immutable Phase 5 membership and committed Phase 6 staging,
rebuilds only NL surfaces in a separate audit directory, and never calls a model
or provider API.  FS, AR, semantic membership, gold answers, and strata are copied
unchanged and verified before output is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "scripts" / "llm_pipeline"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.llm_pipeline.verbalize_abstract import deduplicate_sentences  # noqa: E402
from scripts.llm_pipeline.verbalize_ontologies import (  # noqa: E402
    clean_entity_name,
    describe_individual_with_domain_independence,
    get_all_individuals,
)
from scripts.phase5_freeze_membership import DATASETS, normalize_answer  # noqa: E402
from scripts.phase6_materialize_release import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    create_context_specific_prompt,
    parse_triple,
    sha256_text,
    stable_json,
    verify_phase5,
)


PHASE6 = ROOT / "release" / "v1.1.0-staging"
DEFAULT_STAGE = ROOT / "release" / "v1.1.0-phase7c-audit"
PREFLIGHT = ROOT / "release" / "v1.1.0-preflight"
REPRESENTATIONS = ("NL", "FS", "AR")
LABEL_PROPERTIES = (
    RDFS.label,
    URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
    URIRef("http://purl.org/dc/elements/1.1/title"),
    URIRef("http://xmlns.com/foaf/0.1/name"),
    URIRef("http://schema.org/name"),
)
STANDARD_NAMESPACES = (
    "http://www.w3.org/",
    "https://www.w3.org/",
    "http://purl.org/dc/",
    "http://xmlns.com/foaf/",
    "http://schema.org/",
    "https://schema.org/",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def local_name(uri: str) -> str:
    return unquote(re.split(r"[#/]", uri.rstrip("/"))[-1])


def identity_surface(uri: str) -> str:
    """Render local-name identity without deleting suffixes."""

    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", local_name(uri))
    value = re.sub(r"[_-]+", " ", value)
    return " ".join(word.capitalize() for word in value.split())


def namespace_surface(uri: str) -> str:
    """Return a compact ontology-grounded namespace qualifier."""

    namespace = uri.rsplit("#", 1)[0] if "#" in uri else uri.rsplit("/", 1)[0]
    parsed = urlsplit(namespace)
    segment = re.split(r"[/#]", parsed.path.rstrip("/"))[-1]
    if not segment:
        segment = parsed.netloc.split(":", 1)[0]
    segment = re.sub(r"\.(owl|rdf|ttl)$", "", segment, flags=re.IGNORECASE)
    return identity_surface(segment) or parsed.netloc or namespace


def extended_namespace_surface(uri: str) -> str:
    namespace = uri.rsplit("#", 1)[0] if "#" in uri else uri.rsplit("/", 1)[0]
    parsed = urlsplit(namespace)
    host = parsed.netloc.split(":", 1)[0].removeprefix("www.")
    parts = [part for part in re.split(r"[/._-]+", host + parsed.path) if part]
    return " ".join(part.capitalize() for part in parts)


def is_source_entity(uri: URIRef) -> bool:
    return not str(uri).startswith(STANDARD_NAMESPACES)


def collision_safe_mapping(
    candidates: dict[str, str],
) -> dict[str, str]:
    """Preserve singleton labels; disambiguate collisions from ontology identity."""

    by_candidate: dict[str, list[str]] = defaultdict(list)
    for uri, candidate in candidates.items():
        by_candidate[candidate].append(uri)

    result: dict[str, str] = {}
    used = {
        candidate
        for candidate, uris in by_candidate.items()
        if len(uris) == 1
    }
    for candidate, uris in sorted(by_candidate.items()):
        uris = sorted(uris)
        if len(uris) == 1:
            result[uris[0]] = candidate
            continue

        preferred = {uri: identity_surface(uri) for uri in uris}
        preferred_counts = Counter(preferred.values())
        for uri in uris:
            identity = preferred[uri] or identity_surface(local_name(uri))
            options = []
            if preferred_counts[preferred[uri]] == 1:
                options.append(identity)
            options.extend(
                (
                    f"{identity} ({namespace_surface(uri)})",
                    f"{identity} ({extended_namespace_surface(uri)})",
                )
            )
            final = next((option for option in options if option not in used), "")
            if not final:
                raise AssertionError(f"Could not create injective label for {uri}")
            result[uri] = final
            used.add(final)

    if len(set(result.values())) != len(result):
        raise AssertionError("Final ontology label mapping is not injective")
    if any("http://" in label or "https://" in label for label in result.values()):
        raise AssertionError("Collision-safe labels must not expose full URIs")
    return result


def entity_role(graph: Graph, uri: URIRef) -> set[str]:
    roles: set[str] = set()
    types = set(graph.objects(uri, RDF.type))
    if OWL.Class in types or RDFS.Class in types:
        roles.add("class")
    if OWL.ObjectProperty in types:
        roles.add("object_property")
    if OWL.DatatypeProperty in types:
        roles.add("data_property")
    if OWL.NamedIndividual in types:
        roles.add("individual")
    return roles or {"referenced_entity"}


def build_dataset_inventory(
    dataset: str, paths: list[Path], query_uris: set[str] | None = None
) -> tuple[dict[str, str], list[dict[str, str]]]:
    query_entities: set[URIRef] = {
        URIRef(uri) for uri in (query_uris or set()) if is_source_entity(URIRef(uri))
    }
    entities = set(query_entities)
    labels: dict[URIRef, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    roles: dict[URIRef, set[str]] = defaultdict(set)
    for path in paths:
        graph = Graph().parse(path, format="turtle")
        classes = {
            subject
            for class_type in (OWL.Class, RDFS.Class)
            for subject in graph.subjects(RDF.type, class_type)
            if isinstance(subject, URIRef) and is_source_entity(subject)
        }
        object_properties = {
            subject
            for subject in graph.subjects(RDF.type, OWL.ObjectProperty)
            if isinstance(subject, URIRef) and is_source_entity(subject)
        }
        data_properties = {
            subject
            for subject in graph.subjects(RDF.type, OWL.DatatypeProperty)
            if isinstance(subject, URIRef) and is_source_entity(subject)
        }
        individuals = {
            subject
            for subject in get_all_individuals(graph, classes, object_properties)
            if isinstance(subject, URIRef) and is_source_entity(subject)
        }
        graph_entities = classes | object_properties | data_properties | individuals
        entities.update(graph_entities)
        for entity in classes:
            roles[entity].add("class")
        for entity in object_properties:
            roles[entity].add("object_property")
        for entity in data_properties:
            roles[entity].add("data_property")
        for entity in individuals:
            roles[entity].add("individual")
        for subject, predicate, obj in graph:
            if subject in graph_entities and predicate in LABEL_PROPERTIES and isinstance(obj, Literal):
                cleaned = clean_entity_name(str(obj))
                if cleaned:
                    labels[subject][LABEL_PROPERTIES.index(predicate)].add(cleaned)
    for entity in query_entities:
        roles[entity].add("formal_query_entity")

    candidates: dict[str, str] = {}
    for entity in sorted(entities, key=str):
        priority_labels = labels.get(entity, {})
        if priority_labels:
            first_priority = min(priority_labels)
            candidate = sorted(priority_labels[first_priority])[0]
        else:
            candidate = clean_entity_name(local_name(str(entity)))
        if not candidate:
            candidate = identity_surface(str(entity))
        candidates[str(entity)] = candidate

    final = collision_safe_mapping(candidates)
    rows = [
        {
            "ontology_scope": dataset,
            "scope_rationale": "dataset-wide frozen-membership ontology inventory spanning both hops",
            "entity_uri": uri,
            "entity_roles": ";".join(sorted(roles[URIRef(uri)] or {"referenced_entity"})),
            "candidate_normalized_label": candidates[uri],
            "final_nl_label": final[uri],
            "collision_resolved": str(final[uri] != candidates[uri]).lower(),
        }
        for uri in sorted(candidates)
    ]
    return final, rows


def overlay_labels(graph: Graph, mapping: dict[str, str]) -> Graph:
    rendered = Graph()
    for triple in graph:
        if triple[1] not in LABEL_PROPERTIES:
            rendered.add(triple)
    graph_entities = {
        term
        for triple in graph
        for term in triple
        if isinstance(term, URIRef) and str(term) in mapping
    }
    for entity in sorted(graph_entities, key=str):
        rendered.add((entity, RDFS.label, Literal(mapping[str(entity)])))
    return rendered


def collision_safe_context(graph: Graph, mapping: dict[str, str]) -> str:
    rendered = overlay_labels(graph, mapping)
    classes = {
        subject
        for subject in rendered.subjects(RDF.type, OWL.Class)
        if isinstance(subject, URIRef)
    }
    obj_properties = {
        subject
        for subject in rendered.subjects(RDF.type, OWL.ObjectProperty)
        if isinstance(subject, URIRef)
    }
    individuals = get_all_individuals(rendered, classes, obj_properties)
    descriptions = deduplicate_sentences(
        describe_individual_with_domain_independence(
            rendered, individual, classes, obj_properties, individuals
        )
        for individual in sorted(individuals, key=lambda item: (mapping.get(str(item), ""), str(item)))
    )
    descriptions.append("")
    return "\n".join(descriptions)


def collision_safe_question(
    existing: str, query: str, candidates: dict[str, str], final: dict[str, str]
) -> str:
    subject, predicate, object_uri = parse_triple(query)
    uris = [subject, predicate] + ([] if object_uri is None else [object_uri])
    required_change = any(
        candidates.get(uri) and final.get(uri) != candidates.get(uri) for uri in uris
    )
    if not required_change:
        return existing

    # Render once from formal positions.  Substring replacement is unsafe because
    # an object candidate such as "Course" can also occur inside a property label.
    is_type = predicate.endswith("#type") or predicate.endswith("/type")
    subject_label = final.get(subject, candidates.get(subject, identity_surface(subject)))
    predicate_label = final.get(predicate, candidates.get(predicate, identity_surface(predicate)))
    if object_uri is not None:
        object_label = final.get(object_uri, candidates.get(object_uri, identity_surface(object_uri)))
        if is_type:
            return f"Is {subject_label} a member of {object_label}?"
        return f"Is {subject_label} related to {object_label} through {predicate_label}?"
    if is_type:
        return f"Which classes does {subject_label} belong to?"
    return f"Which entities are related to {subject_label} through {predicate_label}?"


def evaluated_hash(row: dict[str, Any], representation: str) -> str:
    fields = {
        "NL": ("nl_question", "nl_context", "inline_nl"),
        "FS": ("fs_query", "fs_context", "inline_owl"),
        "AR": ("ar_question", "ar_context", "inline_abs"),
    }
    question, context, mode = fields[representation]
    prompt = create_context_specific_prompt(
        str(row[question]), str(row[context]), mode, str(row["answer_type"])
    )
    payload = stable_json(
        {
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "representation": representation,
            "rendered_prompt": prompt,
        }
    )
    return sha256_text(payload)


def gold_semantics(row: dict[str, Any]) -> str:
    return stable_json(
        {
            "task_group": row["task_group"],
            "answer": normalize_answer(str(row["gold_answer"]), str(row["answer_type"])),
        }
    )


def duplicate_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hash_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for representation in REPRESENTATIONS:
            input_hash = evaluated_hash(row, representation)
            item = {**row, "representation": representation, "input_hash": input_hash}
            grouped[(representation, input_hash)].append(item)
            hash_rows.append(
                {
                    "task_id": row["task_id"],
                    "representation": representation,
                    "input_hash": input_hash,
                    "gold_semantics": gold_semantics(row),
                }
            )

    groups: list[dict[str, Any]] = []
    for (representation, input_hash), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        semantics = sorted({gold_semantics(row) for row in members})
        scopes = sorted({f"{row['dataset']}/{row['hop']}" for row in members})
        formal = {str(row["formal_query"]) for row in members}
        if len(semantics) > 1:
            classification = "incompatible gold semantics"
            category = "1"
        elif len(scopes) > 1:
            classification = "incidental cross-dataset/hop duplication"
            category = "4"
        elif len(formal) == 1:
            classification = "compatible semantic duplicates"
            category = "2"
        else:
            classification = "compatible representational collapse"
            category = "3"
        groups.append(
            {
                "input_equivalence_group_id": f"ieq-{representation.lower()}-{input_hash[:24]}",
                "representation": representation,
                "input_hash": input_hash,
                "classification_category": category,
                "classification": classification,
                "semantic_row_count": len(members),
                "task_ids": ";".join(str(row["task_id"]) for row in members),
                "dataset_hop_distribution": stable_json(dict(Counter(
                    f"{row['dataset']}/{row['hop']}" for row in members
                ))),
                "identical_gold_semantics": semantics[0] if len(semantics) == 1 else "",
                "gold_semantics_count": len(semantics),
                "cause": classification,
            }
        )

    counts: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        rep_groups = [row for row in groups if row["representation"] == representation]
        incompatible = [row for row in rep_groups if row["classification_category"] == "1"]
        compatible = [row for row in rep_groups if row["classification_category"] != "1"]
        counts[representation] = {
            "incompatible_groups": len(incompatible),
            "incompatible_rows": sum(row["semantic_row_count"] for row in incompatible),
            "compatible_groups": len(compatible),
            "compatible_rows": sum(row["semantic_row_count"] for row in compatible),
            "compatible_by_classification": dict(Counter(
                row["classification"] for row in compatible
            )),
        }
    return groups, {"counts": counts, "hash_rows": hash_rows}


def outcomes_for_phase7b(
    corrected: list[dict[str, Any]], post_groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = PREFLIGHT / "phase7b_incompatible_groups.csv"
    if not path.is_file():
        return []
    by_id = {str(row["task_id"]): row for row in corrected}
    fatal_hashes = {
        row["input_hash"]
        for row in post_groups
        if row["representation"] == "NL" and row["classification_category"] == "1"
    }
    outcomes: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for old in csv.DictReader(handle):
            ids = [item for item in old["task_ids"].split(";") if item]
            new_hashes = [evaluated_hash(by_id[item], "NL") for item in ids]
            outcomes.append(
                {
                    "phase7b_input_hash": old["input_hash"],
                    "task_group": old["task_group"],
                    "datasets": old["datasets"],
                    "task_ids": old["task_ids"],
                    "distinct_postfix_input_count": len(set(new_hashes)),
                    "postfix_incompatible": str(any(value in fatal_hashes for value in new_hashes)).lower(),
                    "outcome": "resolved" if len(set(new_hashes)) > 1 and not any(value in fatal_hashes for value in new_hashes) else "unresolved",
                }
            )
    return outcomes


def proposed_config(post_counts: dict[str, Any], row_count: int) -> dict[str, Any]:
    source = json.loads((PREFLIGHT / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    source["config_version"] = "core-llm-bench-v1.1-phase7c-proposed-config-1"
    source["status"] = "proposed_pending_scientific_and_provider-routing_review"
    source["execution_authorized"] = False
    source["ready_for_execution"] = False
    source["primary_experiment_policy"] = {
        "policy": "complete fresh rerun only",
        "historical_v1_0_predictions": "provenance and sensitivity only; never mixed into primary v1.1 results",
        "semantic_rows": row_count,
        "representations_per_row": 3,
        "models": 3,
        "expected_requests": row_count * 3 * 3,
    }
    source["configuration_uncertainties"] = [
        {
            "id": "gpt5-sampling-controls",
            "blocking": False,
            "detail": "temperature and top_p remain omitted: retained history did not send them and model-specific support with reasoning_effort=low was not established. Presence/frequency penalties are explicitly frozen at their API defaults of 0.0.",
        },
        {
            "id": "provider-endpoint-availability-preflight",
            "blocking": True,
            "detail": "Immediately before execution, verify without inference that the proposed OpenRouter endpoint tags still serve the exact model IDs and support every requested parameter; provider availability can change.",
        },
        {
            "id": "historical-gpt5-mini-snapshot-deprecated",
            "blocking": True,
            "detail": "The exact historically verified gpt-5-mini-2025-08-07 snapshot is now marked deprecated in the official model catalog. Scientific review must choose between historical identity and a newly frozen replacement before execution.",
        },
    ]
    if post_counts["NL"]["incompatible_groups"]:
        source["configuration_uncertainties"].append(
            {
                "id": "incompatible-duplicate-inputs",
                "blocking": True,
                "detail": "Phase 7C still found incompatible NL exact-input groups.",
            }
        )
    source["prompt_and_response_contract"]["system_prompt"] = None
    source["prompt_and_response_contract"]["system_prompt_policy"] = "explicitly no system message"
    source["provider_routing_policy"] = {
        "OpenAI": {"endpoint": "https://api.openai.com/v1", "fallback": False},
        "OpenRouter": {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "model_fallbacks": [],
        },
    }
    for model in source["models"].values():
        model["system_prompt"] = None
        model["user_prompt_template_version"] = PROMPT_TEMPLATE_VERSION
        model["timeout_seconds"] = 30
        model["retry_policy"] = source["execution_design"]["technical_retry_policy"]
        model.setdefault("presence_penalty", model.get("provider_specific_parameters", {}).get("presence_penalty"))
        model.setdefault("frequency_penalty", model.get("provider_specific_parameters", {}).get("frequency_penalty"))
    gpt = source["models"]["GPT-5 mini"]
    gpt.update(
        {
            "temperature": None,
            "top_p": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": 0,
            "seed_control": "best-effort deterministic sampling; API does not guarantee exact determinism",
            "provider_routing": {"endpoint": "https://api.openai.com/v1", "fallback": False},
        }
    )
    gemini = source["models"]["Gemini 2.5 Flash-Lite"]
    gemini.update(
        {
            "reasoning_or_thinking": {"enabled": False},
            "seed": 0,
            "seed_control": "supported by the model route; deterministic sampling is best-effort",
            "presence_penalty": None,
            "frequency_penalty": None,
            "unsupported_setting_note": "The model route does not advertise presence/frequency penalties; they are not sent.",
            "provider_routing": {
                "order": ["google-ai-studio"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            },
            "provider_specific_parameters": {},
        }
    )
    qwen = source["models"]["Qwen3-30B-A3B-Instruct"]
    qwen.update(
        {
            "reasoning_or_thinking": "not applicable; exact Instruct-2507 model operates in non-thinking mode",
            "seed": 0,
            "seed_control": "supported by the proposed DekaLLM endpoint; deterministic sampling is best-effort",
            "provider_routing": {
                "order": ["dekallm"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            },
        }
    )
    source["configuration_evidence"] = {
        "checked_utc_date": "2026-09-21",
        "openai_model": "https://developers.openai.com/api/docs/models/gpt-5-mini",
        "openai_chat_parameters": "https://platform.openai.com/docs/api-reference/chat/create",
        "openrouter_routing": "https://openrouter.ai/docs/guides/routing/provider-selection",
        "gemini_model": "https://openrouter.ai/google/gemini-2.5-flash-lite",
        "qwen_model": "https://openrouter.ai/qwen/qwen3-30b-a3b-instruct-2507",
    }
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if ROOT not in stage.parents or stage == PHASE6.resolve():
        raise ValueError("Phase 7C output must be a new repository-local staging directory")

    membership, _manifest = verify_phase5()
    source_table = pq.read_table(PHASE6 / "core_llm_bench_v1_1.parquet")
    source_rows = source_table.to_pylist()
    if len(source_rows) != len(membership) or len(source_rows) != 9048:
        raise AssertionError("Frozen membership/staging cardinality drift")

    mappings: dict[str, dict[str, str]] = {}
    candidates: dict[str, dict[str, str]] = {}
    mapping_rows: list[dict[str, str]] = []
    selected_roots: dict[str, set[tuple[str, str]]] = defaultdict(set)
    selected_query_uris: dict[str, set[str]] = defaultdict(set)
    for row in membership:
        selected_roots[row["dataset"]].add((row["hop"], row["root"]))
        selected_query_uris[row["dataset"]].update(parse_triple(row["normalized_query"])[:2])
        query_object = parse_triple(row["normalized_query"])[2]
        if query_object is not None:
            selected_query_uris[row["dataset"]].add(query_object)
    for dataset, dataset_dir in DATASETS.items():
        paths = sorted(
            ROOT / "data" / "resources" / f"{dataset_dir}_{hop}" / f"{root}.ttl"
            for hop, root in selected_roots[dataset]
        )
        mapping, rows = build_dataset_inventory(dataset, paths, selected_query_uris[dataset])
        mappings[dataset] = mapping
        candidates[dataset] = {row["entity_uri"]: row["candidate_normalized_label"] for row in rows}
        mapping_rows.extend(rows)

    corrected: list[dict[str, Any]] = []
    context_cache: dict[tuple[str, str, str], str] = {}
    question_changes = context_changes = 0
    for source in source_rows:
        dataset = str(source["dataset_key"])
        key = (dataset, str(source["hop"]), str(source["root_entity"]))
        if key not in context_cache:
            graph_path = ROOT / "data" / "resources" / f"{DATASETS[dataset]}_{source['hop']}" / f"{source['root_entity']}.ttl"
            graph = Graph().parse(graph_path, format="turtle")
            context_cache[key] = collision_safe_context(graph, mappings[dataset])
        new_context = context_cache[key]
        new_question = collision_safe_question(
            str(source["nl_question"]), str(source["formal_query"]), candidates[dataset], mappings[dataset]
        )
        row = dict(source)
        row["nl_question"] = new_question
        row["nl_context"] = new_context
        row["nl_question_source"] = "phase7c-collision-safe-ontology-inventory"
        question_changes += new_question != source["nl_question"]
        context_changes += new_context != source["nl_context"]
        for field in ("fs_query", "fs_context", "ar_question", "ar_context", "ar_gold_answer", "gold_answer", "semantic_key"):
            if row[field] != source[field]:
                raise AssertionError(f"Phase 7C altered protected field {field}")
        corrected.append(row)

    pre_groups, pre = duplicate_audit(source_rows)
    post_groups, post = duplicate_audit(corrected)
    outcomes = outcomes_for_phase7b(corrected, post_groups)
    changed_rows = sum(
        (row["nl_question"], row["nl_context"]) != (old["nl_question"], old["nl_context"])
        for row, old in zip(corrected, source_rows, strict=True)
    )

    stage.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(corrected, schema=source_table.schema), stage / "core_llm_bench_v1_1_phase7c.parquet")
    write_csv(stage / "entity_label_mapping.csv", mapping_rows, list(mapping_rows[0]))
    write_csv(stage / "input_equivalence_groups.csv", post_groups, list(post_groups[0]))
    write_csv(stage / "phase7b_conflict_outcomes.csv", outcomes, list(outcomes[0]))
    write_csv(stage / "evaluated_input_hashes.csv", post["hash_rows"], list(post["hash_rows"][0]))

    report = {
        "status": "phase7c_complete_pending_scientific_review",
        "model_or_api_calls": 0,
        "membership_rows": len(corrected),
        "membership_changed": False,
        "mapping_scope": "dataset-wide frozen-membership ontology inventory spanning both hops",
        "mapping_scope_rationale": "A URI receives one stable surface across every ontology root selected by the frozen membership in its generated dataset; Pizza100 and Pizza250 remain distinct source-ontology scopes.",
        "label_mapping_rows": len(mapping_rows),
        "disambiguated_entities": sum(row["collision_resolved"] == "true" for row in mapping_rows),
        "nl_rows_changed": changed_rows,
        "nl_questions_changed": question_changes,
        "nl_contexts_changed": context_changes,
        "pre_fix": pre["counts"],
        "post_fix": post["counts"],
        "phase7b_conflict_outcomes": outcomes,
        "semantic_membership_can_remain_9048": post["counts"]["NL"]["incompatible_groups"] == 0,
        "proposed_removals": [],
        "fresh_primary_experiment_matrix": len(corrected) * 3 * 3,
        "protected_representation_check": "FS and AR fields byte-identical for every row",
    }
    write_json(stage / "phase7c_audit_report.json", report)
    write_json(stage / "proposed_experiment_config_v1_1.json", proposed_config(post["counts"], len(corrected)))
    print(stable_json(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
