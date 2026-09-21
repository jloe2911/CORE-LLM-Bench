#!/usr/bin/env python3
"""Materialize the frozen Phase 5 semantic membership as staged v1.1 inputs.

The script is offline and fail-closed: Phase 5 hashes and cardinalities are
validated before any staged output is written. It assigns deterministic public
integer IDs, rebuilds corrected AR inputs, inventories exact prompt hashes, and
creates only a pending model-run manifest. It never calls a model API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from rdflib import Graph, Literal, URIRef


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "scripts" / "llm_pipeline"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.ar_validation import (  # noqa: E402
    abstract_answer,
    render_abstract_context,
    source_terms,
    validate_ar_row,
)
from scripts.llm_pipeline.verbalize_abstract import (  # noqa: E402
    deduplicate_sentences,
)
from scripts.llm_pipeline.verbalize_ontologies import (  # noqa: E402
    get_nice_label,
)
from scripts.oeqa_explanations import build_answer_explanations  # noqa: E402
from scripts.ontology_tools.abstraction.OntologyAbstractor import (  # noqa: E402
    LEXICAL_ANNOTATIONS,
    create_abstraction_mappings,
    discover_entity_roles,
    flatten_mappings,
    local_name,
    merge_entity_roles,
)
from scripts.phase5_freeze_membership import (  # noqa: E402
    DATASETS,
    MODEL_NAMES,
    PHASE4,
    normalize_answer,
    prediction_inventory,
    semantic_key,
)


PHASE5 = ROOT / "data" / "output_v1_1_staging" / "phase5"
DEFAULT_STAGE = ROOT / "release" / "v1.1.0-staging"
DATASET_ORDER = {
    "Family": 0,
    "Pizza100": 1,
    "Pizza250": 2,
    "OWL2Bench": 3,
}
PUBLIC_DATASET = {
    "Family": "FamilyOWL",
    "Pizza100": "Pizza100",
    "Pizza250": "Pizza250",
    "OWL2Bench": "OWL2Bench",
}
EXPECTED_DATASET_HOP = {
    "Family/1hop": 1881,
    "Family/2hop": 1881,
    "OWL2Bench/1hop": 1467,
    "OWL2Bench/2hop": 1593,
    "Pizza100/1hop": 495,
    "Pizza100/2hop": 495,
    "Pizza250/1hop": 618,
    "Pizza250/2hop": 618,
}
REPRESENTATIONS = ("NL", "FS", "AR")
PROMPT_TEMPLATE_VERSION = "api_calls.create_context_specific_prompt@24c4520"
MODEL_DISPLAY_NAMES = tuple(MODEL_NAMES.values())
URI_QUERY = re.compile(r"<([^>]+)>")


def extract_sparql_terms(query: str) -> list[str]:
    terms: list[str] = []
    generic_terms = {
        "type", "rdf", "rdfs", "owl", "xsd", "resource", "namedindividual"
    }
    for uri in re.findall(r"<([^>]+)>", str(query or "")):
        if "www.w3.org/" in uri:
            continue
        terms.append(uri)
        fragment = re.split(r"[#/]", uri.rstrip("/"))[-1]
        if fragment and fragment.lower() not in generic_terms:
            terms.append(fragment)
    for prefixed in re.findall(
        r"\b([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)\b", str(query or "")
    ):
        terms.append(prefixed)
        name = prefixed.split(":", 1)[1]
        if name and name.lower() not in generic_terms:
            terms.append(name)
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


def build_query_relevant_symbolic_context(
    query: str, ontology_context: str, max_chars: int = 12000
) -> str:
    context = str(ontology_context or "")
    if len(context) <= max_chars:
        return context
    terms = extract_sparql_terms(query)
    if not terms:
        return context[:max_chars] + "\n... [truncated for memory efficiency]"
    blocks = [block for block in re.split(r"\n\s*\n", context) if block.strip()]
    selected = [
        block.strip()
        for block in blocks
        if any(term.lower() in block.lower() for term in terms)
    ]
    if not selected:
        for term in terms:
            index = context.lower().find(term.lower())
            if index != -1:
                selected.append(
                    context[max(0, index - 2000) : min(len(context), index + 4000)].strip()
                )
    header = (
        "[Query-relevant symbolic context extracted before truncation]\n"
        f"SPARQL query: {query}\n"
    )
    focused = header + "\n\n".join(selected)
    if len(focused) < max_chars:
        focused += "\n\n[Ontology prefix]\n" + context[: max_chars - len(focused)]
    if len(focused) > max_chars:
        focused = focused[:max_chars] + "\n... [query-relevant symbolic context truncated]"
    return focused


def create_context_specific_prompt(
    query: str, ontology_context: str, context_mode: str, answer_type: str
) -> str:
    """Byte-equivalent local renderer for the frozen evaluation prompt."""

    if answer_type == "BIN" or answer_type.lower() == "binary":
        format_instruction = (
            "ANSWER: [TRUE or FALSE]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: ONLY write TRUE or FALSE.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    elif answer_type == "MC" or answer_type.lower() == "multi choice":
        format_instruction = (
            "ANSWER: [Use LOCAL NAMES only, semicolon-separated]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: Use LOCAL NAMES only (e.g., 'Person', 'U0C4', 'caroline_lavinia_tubb_1840'), give all the possible answers.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    else:
        format_instruction = (
            "ANSWER: [Your answer using local names]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    base_instruction = (
        "CRITICAL: You MUST respond in exactly this format:\n"
        f"{format_instruction}\n"
        "DO NOT include any additional text before or after this format.\n"
    )
    if context_mode in {"ttl", "inline_owl"}:
        ontology_context = build_query_relevant_symbolic_context(
            query, ontology_context, max_chars=12000
        )
        return (
            "You are an expert in SPARQL and OWL ontologies. Analyze the ontology "
            "context and answer the SPARQL query precisely.\n\n"
            f"{base_instruction}\n\nQuestion: {query}\nContext: {ontology_context}"
        )
    if len(ontology_context) > 10000:
        ontology_context = ontology_context[:10000] + "\n... [truncated for memory efficiency]"
    return (
        "You are an expert in ontologies, answer the following question based on "
        "the provided ontological relationships. Reason through the ontological "
        "context and answer based on what you can infer from the context.\n\n"
        f"{base_instruction}\n\nQuestion: {query}\nContext: {ontology_context}"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any, *, indent: int | None = None) -> str:
    separators = None if indent else (",", ":")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=separators,
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


def verify_phase5() -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest_path = PHASE5 / "membership_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 5 artifact: {path}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Phase 5 artifact hash/size drift: {path}")
    membership_path = PHASE5 / "semantic_membership.csv"
    with membership_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9048:
        raise ValueError(f"Frozen membership must contain 9,048 rows, got {len(rows)}")
    bqa = [row for row in rows if row["task_type"] == "BQA"]
    oeqa = [row for row in rows if row["task_type"] == "OEQA"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bqa:
        groups[row["sampling_group_key"]].append(row)
    invalid = {
        key: [item["gold_answer"].upper() for item in values]
        for key, values in groups.items()
        if len(values) != 2
        or {item["gold_answer"].upper() for item in values} != {"TRUE", "FALSE"}
    }
    if len(bqa) != 6032 or len(groups) != 3016 or len(oeqa) != 3016 or invalid:
        raise ValueError(
            "Frozen membership cardinality/pair failure: "
            f"BQA={len(bqa)}, groups={len(groups)}, OEQA={len(oeqa)}, invalid={len(invalid)}"
        )
    pizza_domain = sum(
        row["dataset"] in {"Pizza100", "Pizza250"}
        and row["task_type"] == "BQA"
        and "DomainConcept>" in row["normalized_query"]
        for row in rows
    )
    if pizza_domain:
        raise ValueError(f"Frozen membership has {pizza_domain} Pizza DomainConcept targets")
    return rows, manifest


def canonical_answer(value: str, answer_type: str) -> str:
    return normalize_answer(value, answer_type)


def public_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        DATASET_ORDER[row["dataset"]],
        0 if row["hop"] == "1hop" else 1,
        0 if row["task_type"] == "BQA" else 1,
        row["root"],
        row["normalized_query"],
        canonical_answer(row["gold_answer"], row["answer_type"]),
        row["source_provenance_key"],
        row["semantic_key"],
    )


def parse_triple(query: str) -> tuple[str, str, str | None]:
    uris = URI_QUERY.findall(query)
    if query.lstrip().upper().startswith("ASK") and len(uris) >= 3:
        return uris[0], uris[1], uris[2]
    if query.lstrip().upper().startswith("SELECT") and len(uris) >= 2:
        return uris[0], uris[1], None
    raise ValueError(f"Unsupported formal query: {query!r}")


def display_label(graph: Graph, uri: str) -> str:
    label = get_nice_label(graph, URIRef(uri))
    if label:
        return str(label)
    name = local_name(uri)
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").split())


def deterministic_nl_question(query: str, graph: Graph) -> str:
    subject, predicate, object_uri = parse_triple(query)
    subject_label = display_label(graph, subject)
    predicate_label = display_label(graph, predicate)
    is_type = predicate.endswith("#type") or predicate.endswith("/type")
    if object_uri is not None:
        object_label = display_label(graph, object_uri)
        if is_type:
            return f"Is {subject_label} a member of {object_label}?"
        return f"Is {subject_label} related to {object_label} through {predicate_label}?"
    if is_type:
        return f"Which classes does {subject_label} belong to?"
    return f"Which entities are related to {subject_label} through {predicate_label}?"


def abstract_question(
    query: str,
    uri_mappings: dict[URIRef, URIRef],
    text_mappings: dict[str, str],
) -> str:
    subject, predicate, object_uri = parse_triple(query)

    def abstract_name(uri: str) -> str:
        target = uri_mappings.get(URIRef(uri))
        if target is not None:
            return local_name(target)
        fallback = text_mappings.get(local_name(uri))
        if fallback is None:
            raise ValueError(f"Required query entity has no abstraction mapping: {uri}")
        return fallback

    subject_label = abstract_name(subject)
    is_type = predicate.endswith("#type") or predicate.endswith("/type")
    predicate_label = "rdf:type" if is_type else abstract_name(predicate)
    if object_uri is not None:
        object_label = abstract_name(object_uri)
        if is_type:
            return f"Is {subject_label} a member of {object_label}?"
        return f"Is {subject_label} related to {object_label} through {predicate_label}?"
    if is_type:
        return f"Which classes does {subject_label} belong to?"
    return f"Which entities are related to {subject_label} through {predicate_label}?"


def verbalize_abox(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    descriptions = deduplicate_sentences(
        item.get("description", "") for item in data.get("individuals", [])
    )
    descriptions.append("")
    return "\n".join(descriptions)


def load_explanation_index(dataset: str, hop: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = PHASE4 / DATASETS[dataset] / hop / "Explanations.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in raw.values():
        inferred = record.get("inferred") or {}
        key = (
            str(inferred.get("subject", "")),
            str(inferred.get("predicate", "")),
            str(inferred.get("object", "")),
        )
        if all(key):
            result[key] = record
    return result


def materialize_explanations(
    row: dict[str, str],
    index: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    complexity = json.loads(row["corrected_explanation_metadata"])
    provenance = json.loads(row["corrected_provenance_metadata"])
    subject_uri, predicate_uri, object_uri = parse_triple(row["normalized_query"])
    subject, predicate = local_name(subject_uri), local_name(predicate_uri)
    if predicate_uri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
        predicate = "rdf:type"
    tags = list(complexity.get("distinct_primitive_reasoning_types", []))
    if row["task_type"] == "BQA":
        source_key = provenance["source_explanation_key"]
        parts = source_key.split("||", 1)[-1].split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid BQA explanation provenance: {source_key}")
        record = index.get(tuple(parts))
        if record is None:
            raise ValueError(f"Missing BQA explanation source: {source_key}")
        return {
            "explanations": record.get("explanations", []),
            "structured_explanations": record.get("structuredExplanations", []),
            "answer_explanations": [],
            "combination_count": int(complexity["proof_count"]),
            "minimum_axiom_count": int(complexity["minimum_axiom_count"]),
            "maximum_axiom_count": int(complexity["maximum_axiom_count"]),
            "minimum_primitive_tag_count": int(complexity["minimum_primitive_tag_count"]),
            "maximum_primitive_tag_count": int(complexity["maximum_primitive_tag_count"]),
            "minimum_distinct_primitive_type_count": len(tags),
            "maximum_distinct_primitive_type_count": len(tags),
            "primitive_reasoning_tags": tags,
            "distinct_primitive_reasoning_types": tags,
            "m_status": complexity.get("m_status", "never"),
            "legacy_explanation_fields": "compatibility-only",
        }

    source_records = []
    for answer in [part.strip() for part in row["gold_answer"].split(";") if part.strip()]:
        record = index.get((subject, predicate, answer))
        if record is None:
            raise ValueError(
                f"Missing exact inferred.object OEQA source: {subject}|{predicate}|{answer}"
            )
        source_records.append(
            {"Source Answer": answer, "Explanations": record.get("explanations", [])}
        )
    answer_groups, _derived_metrics = build_answer_explanations(
        row["gold_answer"], source_records
    )
    return {
        "explanations": [],
        "structured_explanations": [],
        "answer_explanations": answer_groups,
        "combination_count": int(complexity["complete_combination_count"]),
        "minimum_axiom_count": int(complexity["complete_min_axiom_count"]),
        "maximum_axiom_count": int(complexity["complete_max_axiom_count"]),
        "minimum_primitive_tag_count": int(complexity["complete_min_primitive_tag_count"]),
        "maximum_primitive_tag_count": int(complexity["complete_max_primitive_tag_count"]),
        "minimum_distinct_primitive_type_count": int(
            complexity["complete_min_distinct_primitive_type_count"]
        ),
        "maximum_distinct_primitive_type_count": int(
            complexity["complete_max_distinct_primitive_type_count"]
        ),
        "primitive_reasoning_tags": tags,
        "distinct_primitive_reasoning_types": tags,
        "m_status": complexity.get("m_status", "never"),
        "legacy_explanation_fields": "compatibility-only",
    }


def prompt_hash(question: str, context: str, representation: str, answer_type: str) -> str:
    mode = {"NL": "inline_nl", "FS": "inline_owl", "AR": "inline_abs"}[representation]
    prompt = create_context_specific_prompt(question, context, mode, answer_type)
    payload = stable_json(
        {
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "representation": representation,
            "rendered_prompt": prompt,
        }
    )
    return sha256_text(payload)


def saved_prompt_hash(saved: dict[str, str], representation: str) -> str:
    fields = {
        "NL": ("NL Question", "NL Context"),
        "FS": ("SPARQL Query", "OWL Context"),
        "AR": ("ABS Question", "ABS Context"),
    }
    question_field, context_field = fields[representation]
    return prompt_hash(
        saved.get(question_field, ""),
        saved.get(context_field, ""),
        representation,
        saved.get("Answer Type", "BIN"),
    )


def response_usable(saved: dict[str, str]) -> bool:
    response = saved.get(saved.get("_response_column", ""), "").strip()
    return bool(response) and not response.upper().startswith(("[ERROR]", "ERROR"))


def record_for_parquet(record: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            result[key] = stable_json(value)
        else:
            result[key] = value
    return result


def mapping_source_signature(paths: list[Path]) -> str:
    payload = "\n".join(
        f"{path.name}\t{path.stat().st_size}\t{sha256_file(path)}"
        for path in paths
    )
    return sha256_text(payload + "\n")


def frozen_selected_inputs(
    selected_keys: set[str],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Load exact published NL inputs only for retained selected semantics."""

    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for dataset, dataset_dir in DATASETS.items():
        with zipfile.ZipFile(ROOT / "final_benchmark" / f"{dataset_dir}.zip") as archive:
            for hop in ("1hop", "2hop"):
                with archive.open(f"{dataset_dir}_{hop}.json") as handle:
                    groups = json.load(handle)
                for group in groups:
                    for qa in group["QAs"]:
                        identity_row = {
                            "Root Entity": group["Root Entity"],
                            "Answer Type": group["Answer Type"],
                            "SPARQL Query": qa["SPARQL Query"],
                            "Answer": qa["Answer"],
                        }
                        key = semantic_key(dataset, hop, identity_row)
                        if key not in selected_keys:
                            continue
                        result[(dataset, hop, key)] = {
                            "NL Question": str(qa["NL Question"]),
                            "NL Context": str(group["NL Context"]),
                            "Answer": str(qa["Answer"]),
                        }
                del groups
    return result


def load_or_build_mappings(
    dataset: str, hop: str, ttl_paths: list[Path]
) -> tuple[dict[URIRef, URIRef], dict[str, str], int]:
    cache_path = ROOT / ".tmp" / "phase6-map-cache" / f"{dataset}_{hop}.json"
    signature = mapping_source_signature(ttl_paths)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("source_signature") == signature:
            print(f"Using verified abstraction-map cache: {dataset}/{hop}", flush=True)
            return (
                {URIRef(key): URIRef(value) for key, value in cached["uri_mappings"]},
                dict(cached["text_mappings"]),
                int(cached["ambiguous_alias_count"]),
            )

    role_sets = []
    lexical_pairs: list[tuple[URIRef, str]] = []
    for path in ttl_paths:
        graph = Graph().parse(path, format="turtle")
        role_sets.append(discover_entity_roles(graph))
        for predicate in LEXICAL_ANNOTATIONS:
            lexical_pairs.extend(
                (original, str(label).strip())
                for original, label in graph.subject_objects(predicate)
                if isinstance(original, URIRef)
                and isinstance(label, Literal)
                and str(label).strip()
            )
    roles = merge_entity_roles(role_sets)
    uri_mappings = flatten_mappings(create_abstraction_mappings(roles))
    alias_candidates: dict[str, set[str]] = defaultdict(set)
    for original, abstract in uri_mappings.items():
        alias_candidates[local_name(original)].add(local_name(abstract))
    for original, label in lexical_pairs:
        if original in uri_mappings:
            alias_candidates[label].add(local_name(uri_mappings[original]))
    text_mappings = {
        source: next(iter(targets))
        for source, targets in sorted(alias_candidates.items())
        if len(targets) == 1
    }
    ambiguous_alias_count = sum(
        len(targets) > 1 for targets in alias_candidates.values()
    )
    cache = {
        "source_signature": signature,
        "uri_mappings": [
            [str(original), str(abstract)]
            for original, abstract in sorted(uri_mappings.items(), key=lambda item: str(item[0]))
        ],
        "text_mappings": text_mappings,
        "ambiguous_alias_count": ambiguous_alias_count,
    }
    write_json(cache_path, cache)
    return uri_mappings, text_mappings, ambiguous_alias_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if ROOT not in stage.parents:
        raise ValueError("Phase 6 staging must remain inside the repository")

    membership, phase5_manifest = verify_phase5()
    ordered = sorted(membership, key=public_sort_key)
    frozen_inputs = frozen_selected_inputs(
        {row["semantic_key"] for row in membership if row["v1_membership"] == "retained"}
    )
    predictions = prediction_inventory()

    records: list[dict[str, Any]] = []
    ar_counts: Counter[str] = Counter()
    question_provenance = Counter()
    root_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    dataset_state: dict[tuple[str, str], dict[str, Any]] = {}

    for dataset in DATASETS:
        dataset_dir = DATASETS[dataset]
        for hop in ("1hop", "2hop"):
            resource_dir = ROOT / "data" / "resources" / f"{dataset_dir}_{hop}"
            ttl_paths = sorted(resource_dir.glob("*.ttl"), key=lambda path: path.name)
            print(f"Building accepted global abstraction map: {dataset}/{hop} ({len(ttl_paths)} roots)", flush=True)
            uri_mappings, text_mappings, ambiguous_alias_count = load_or_build_mappings(
                dataset, hop, ttl_paths
            )
            dataset_state[(dataset, hop)] = {
                "resource_dir": resource_dir,
                "uri_mappings": uri_mappings,
                "text_mappings": text_mappings,
                "ambiguous_alias_count": ambiguous_alias_count,
                "explanations": load_explanation_index(dataset, hop),
            }

    for task_id, source in enumerate(ordered, start=1):
        dataset, hop, root = source["dataset"], source["hop"], source["root"]
        state = dataset_state[(dataset, hop)]
        cache_key = (dataset, hop, root)
        cached = root_cache.get(cache_key)
        if cached is None:
            graph_path = state["resource_dir"] / f"{root}.ttl"
            graph = Graph().parse(graph_path, format="turtle")
            local_mappings = {
                original: abstract
                for original, abstract in state["uri_mappings"].items()
                if (original, None, None) in graph
                or (None, original, None) in graph
                or (None, None, original) in graph
            }
            identifiers, labels = source_terms([graph], local_mappings)
            fs_context = graph.serialize(format="turtle")
            nl_context = verbalize_abox(
                ROOT
                / "data"
                / "output"
                / "verbalized_ontologies"
                / f"{DATASETS[dataset]}_{hop}"
                / f"{root}.json"
            )
            ar_context = render_abstract_context(graph, state["uri_mappings"])
            cached = {
                "graph": graph,
                "fs_context": fs_context,
                "nl_context": nl_context,
                "ar_context": ar_context,
                "identifiers": identifiers,
                "labels": labels,
            }
            root_cache[cache_key] = cached

        frozen = frozen_inputs.get((dataset, hop, source["semantic_key"]))
        if frozen:
            nl_question = frozen["NL Question"]
            nl_context = frozen["NL Context"]
            nl_question_source = "published-v1.0-byte-identical"
        else:
            nl_question = deterministic_nl_question(
                source["normalized_query"], cached["graph"]
            )
            nl_context = cached["nl_context"]
            nl_question_source = "deterministic-v1.1-query-template"
        question_provenance[nl_question_source] += 1
        ar_question = abstract_question(
            source["normalized_query"],
            state["uri_mappings"],
            state["text_mappings"],
        )
        ar_answer = abstract_answer(
            source["gold_answer"], source["answer_type"], state["text_mappings"]
        )
        validation = validate_ar_row(
            original_question=nl_question,
            original_answer=source["gold_answer"],
            answer_type=source["answer_type"],
            sparql_query=source["normalized_query"],
            abs_question=ar_question,
            abs_context=cached["ar_context"],
            abs_answer=ar_answer,
            uri_mappings=state["uri_mappings"],
            text_mappings=state["text_mappings"],
            identifiers=cached["identifiers"],
            lexical_labels=cached["labels"],
            expected_abs_question=ar_question,
        )
        for issue, count in validation.counts().items():
            ar_counts[issue] += count
        if not validation.valid:
            raise ValueError(
                f"AR validation failed for {source['semantic_key']}: {validation.counts()}"
            )
        explanation = materialize_explanations(
            source, state["explanations"]
        )
        record = {
            "task_id": task_id,
            "semantic_key": source["semantic_key"],
            "legacy_task_id": source["source_task_id"],
            "dataset": PUBLIC_DATASET[dataset],
            "dataset_key": dataset,
            "hop": hop,
            "task_group": source["task_type"],
            "task_type": source["task_type"],
            "answer_type": source["answer_type"],
            "root_entity": root,
            "formal_query": source["normalized_query"],
            "gold_answer": source["gold_answer"],
            "nl_question": nl_question,
            "nl_context": nl_context,
            "fs_query": source["normalized_query"],
            "fs_context": cached["fs_context"],
            "ar_question": ar_question,
            "ar_context": cached["ar_context"],
            "ar_gold_answer": ar_answer,
            "nl_question_source": nl_question_source,
            "sampling_group_key": source["sampling_group_key"],
            "pair_group_id": "" if source["task_type"] != "BQA" else "bqa-" + sha256_text(source["sampling_group_key"])[:24],
            "positive_task_id": None,
            "positive_semantic_key": "",
            "source_provenance_key": source["source_provenance_key"],
            "raw_minimum_complete_primitive_tag_complexity": int(source["raw_complexity"]),
            "raw_maximum_complete_primitive_tag_complexity": explanation["maximum_primitive_tag_count"],
            "minimum_axiom_count": explanation["minimum_axiom_count"],
            "maximum_axiom_count": explanation["maximum_axiom_count"],
            "primitive_reasoning_tags": explanation["primitive_reasoning_tags"],
            "distinct_primitive_reasoning_types": explanation["distinct_primitive_reasoning_types"],
            "m_status": explanation["m_status"],
            "complexity_bin": source["complexity_bin"],
            "answer_explanations": explanation["answer_explanations"],
            "explanations": explanation["explanations"],
            "structured_explanations": explanation["structured_explanations"],
            "complete_explanation_combination_count": explanation["combination_count"],
            "minimum_distinct_primitive_type_count": explanation["minimum_distinct_primitive_type_count"],
            "maximum_distinct_primitive_type_count": explanation["maximum_distinct_primitive_type_count"],
            "legacy_explanation_fields": explanation["legacy_explanation_fields"],
            "phase5_membership_sha256": phase5_manifest["artifacts"][0]["sha256"],
        }
        records.append(record)

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["task_group"] == "BQA":
            by_pair[record["sampling_group_key"]].append(record)
    pair_rows = []
    for group_key, pair in sorted(by_pair.items()):
        positives = [row for row in pair if row["gold_answer"].upper() == "TRUE"]
        negatives = [row for row in pair if row["gold_answer"].upper() == "FALSE"]
        if len(positives) != 1 or len(negatives) != 1:
            raise ValueError(f"Invalid BQA pair after materialization: {group_key}")
        positive, negative = positives[0], negatives[0]
        if (
            positive["dataset"] != negative["dataset"]
            or positive["hop"] != negative["hop"]
            or positive["root_entity"] != negative["root_entity"]
            or positive["raw_minimum_complete_primitive_tag_complexity"]
            != negative["raw_minimum_complete_primitive_tag_complexity"]
            or positive["complexity_bin"] != negative["complexity_bin"]
        ):
            raise ValueError(f"BQA pair boundary/complexity mismatch: {group_key}")
        negative["positive_task_id"] = positive["task_id"]
        negative["positive_semantic_key"] = positive["semantic_key"]
        pair_rows.append(
            {
                "pair_group_id": positive["pair_group_id"],
                "positive_task_id": positive["task_id"],
                "negative_task_id": negative["task_id"],
                "positive_semantic_key": positive["semantic_key"],
                "negative_semantic_key": negative["semantic_key"],
                "dataset": positive["dataset"],
                "hop": positive["hop"],
                "root_entity": positive["root_entity"],
                "sampling_group_key": group_key,
            }
        )

    ids = [record["task_id"] for record in records]
    if ids != list(range(1, 9049)) or len(set(ids)) != 9048:
        raise ValueError("Public IDs must be unique contiguous integers 1..9048")

    benchmark_dir = stage / "benchmark"
    for dataset in DATASETS:
        for hop in ("1hop", "2hop"):
            subset = [
                row for row in records
                if row["dataset_key"] == dataset and row["hop"] == hop
            ]
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                grouped[(row["root_entity"], row["task_group"], row["answer_type"])].append(row)
            payload = []
            for (root, task_group, answer_type), qas in sorted(grouped.items()):
                context = qas[0]
                payload.append(
                    {
                        "Root Entity": root,
                        "Task Group": task_group,
                        "Answer Type": answer_type,
                        "NL Context": context["nl_context"],
                        "OWL Context": context["fs_context"],
                        "ABS Context": context["ar_context"],
                        "QAs": [
                            {
                                key: value
                                for key, value in qa.items()
                                if key not in {"nl_context", "fs_context", "ar_context", "dataset_key"}
                            }
                            for qa in sorted(qas, key=lambda item: item["task_id"])
                        ],
                    }
                )
            write_json(
                benchmark_dir / f"{PUBLIC_DATASET[dataset]}_{hop}.json",
                payload,
            )

    parquet_path = stage / "core_llm_bench_v1_1.parquet"
    parquet_rows = [record_for_parquet(record) for record in records]
    table = pa.Table.from_pylist(parquet_rows)
    pq.write_table(table, parquet_path, compression="zstd", version="2.6")

    id_fields = [
        "new_task_id", "semantic_key", "legacy_task_id", "dataset", "hop",
        "task_group", "root_entity", "formal_query", "gold_answer",
    ]
    id_rows = [
        {
            "new_task_id": row["task_id"],
            **{field: row[field] for field in id_fields if field != "new_task_id"},
        }
        for row in records
    ]
    write_csv(stage / "task_id_mapping.csv", id_rows, id_fields)
    write_csv(
        stage / "bqa_pair_mapping.csv",
        pair_rows,
        [
            "pair_group_id", "positive_task_id", "negative_task_id",
            "positive_semantic_key", "negative_semantic_key", "dataset", "hop",
            "root_entity", "sampling_group_key",
        ],
    )

    reasoning_fields = [
        "task_id", "semantic_key", "dataset", "hop", "task_group",
        "raw_minimum_complete_primitive_tag_complexity",
        "raw_maximum_complete_primitive_tag_complexity", "minimum_axiom_count",
        "maximum_axiom_count", "primitive_reasoning_tags",
        "distinct_primitive_reasoning_types", "m_status", "complexity_bin",
    ]
    reasoning_rows = []
    for row in records:
        item = {field: row[field] for field in reasoning_fields}
        item["primitive_reasoning_tags"] = "".join(row["primitive_reasoning_tags"])
        item["distinct_primitive_reasoning_types"] = "".join(row["distinct_primitive_reasoning_types"])
        reasoning_rows.append(item)
    write_csv(stage / "reasoning_metadata.csv", reasoning_rows, reasoning_fields)

    model_input_rows = []
    pending_rows = []
    reuse_counts: Counter[tuple[str, str, str]] = Counter()
    reuse_reason_counts: Counter[tuple[str, str, str]] = Counter()
    retained_reused: set[tuple[str, str, str]] = set()
    for row in records:
        source_questions = {
            "NL": (row["nl_question"], row["nl_context"]),
            "FS": (row["fs_query"], row["fs_context"]),
            "AR": (row["ar_question"], row["ar_context"]),
        }
        for representation in REPRESENTATIONS:
            question, context = source_questions[representation]
            input_hash = prompt_hash(question, context, representation, row["answer_type"])
            status_by_model = {}
            reason_by_model = {}
            for model in MODEL_DISPLAY_NAMES:
                saved = predictions.get(
                    (
                        row["dataset_key"], row["hop"], model,
                        representation, row["semantic_key"],
                    )
                )
                reusable = False
                if representation == "AR":
                    reason = "corrected AR input"
                elif saved is None:
                    reason = (
                        "newly selected question"
                        if row["nl_question_source"] != "published-v1.0-byte-identical"
                        else "missing old response"
                    )
                elif not response_usable(saved):
                    reason = "unusable/error old response"
                elif saved_prompt_hash(saved, representation) != input_hash:
                    reason = "other validated reason: exact evaluated input changed"
                else:
                    reusable = True
                    reason = "exact evaluated input and non-error old response"
                status = "reusable" if reusable else "rerun required"
                status_by_model[model] = status
                reason_by_model[model] = reason
                reuse_counts[(model, representation, status)] += 1
                reuse_reason_counts[(model, representation, reason)] += 1
                if reusable:
                    retained_reused.add((row["semantic_key"], model, representation))
                else:
                    pending_rows.append(
                        {
                            "task_id": row["task_id"],
                            "dataset": row["dataset"],
                            "hop": row["hop"],
                            "task_group": row["task_group"],
                            "representation": representation,
                            "model": model,
                            "input_hash": input_hash,
                            "reason_for_rerun": reason,
                        }
                    )
            model_input_rows.append(
                {
                    "task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "hop": row["hop"],
                    "task_group": row["task_group"],
                    "representation": representation,
                    "input_hash": input_hash,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "reuse_status_by_model": stable_json(status_by_model),
                    "reuse_reason_by_model": stable_json(reason_by_model),
                }
            )

    input_fields = [
        "task_id", "dataset", "hop", "task_group", "representation",
        "input_hash", "prompt_template_version", "reuse_status_by_model",
        "reuse_reason_by_model",
    ]
    write_csv(stage / "model_input_manifest.csv", model_input_rows, input_fields)
    pending_fields = [
        "task_id", "dataset", "hop", "task_group", "representation", "model",
        "input_hash", "reason_for_rerun",
    ]
    pending_rows.sort(key=lambda item: (item["task_id"], REPRESENTATIONS.index(item["representation"]), item["model"]))
    write_csv(stage / "pending_model_runs.csv", pending_rows, pending_fields)

    dataset_hop_counts = Counter(
        f"{row['dataset_key']}/{row['hop']}" for row in records
    )
    task_counts = Counter(row["task_group"] for row in records)
    answer_counts = Counter(
        (row["task_group"], row["gold_answer"].upper()) for row in records
    )
    if dict(dataset_hop_counts) != EXPECTED_DATASET_HOP:
        raise ValueError(f"Dataset/hop count drift: {dict(dataset_hop_counts)}")
    if task_counts != Counter({"BQA": 6032, "OEQA": 3016}):
        raise ValueError(f"Task count drift: {task_counts}")
    if answer_counts[("BQA", "TRUE")] != 3016 or answer_counts[("BQA", "FALSE")] != 3016:
        raise ValueError(f"BQA label drift: {answer_counts}")

    coverage = Counter()
    m_counts = Counter()
    complexity = defaultdict(Counter)
    for row in records:
        coverage.update(set(row["distinct_primitive_reasoning_types"]))
        m_counts[row["m_status"]] += 1
        complexity[row["task_group"]][row["complexity_bin"]] += 1
    coverage_report = {
        "primitive_reasoning_type_question_counts": dict(sorted(coverage.items())),
        "m_status_question_counts": dict(sorted(m_counts.items())),
        "m_is_metadata_not_primitive": True,
    }
    complexity_report = {
        task: {label: counts[label] for label in ("Low", "Medium", "High")}
        for task, counts in sorted(complexity.items())
    }
    statistics = {
        "total": len(records),
        "dataset_hop": dict(sorted(dataset_hop_counts.items())),
        "task": dict(sorted(task_counts.items())),
        "BQA_TRUE": answer_counts[("BQA", "TRUE")],
        "BQA_FALSE": answer_counts[("BQA", "FALSE")],
        "BQA_pairs": len(pair_rows),
        "public_id_min": min(ids),
        "public_id_max": max(ids),
        "public_id_unique": len(set(ids)),
        "nl_question_provenance": dict(sorted(question_provenance.items())),
    }
    write_json(stage / "dataset_statistics.json", statistics)
    write_json(stage / "reasoning_coverage.json", coverage_report)
    write_json(stage / "complexity_distribution.json", complexity_report)

    reuse_report: dict[str, Any] = {}
    for (model, representation, status), count in sorted(reuse_counts.items()):
        reuse_report.setdefault(model, {}).setdefault(representation, {})[status] = count
    reuse_reason_report: dict[str, Any] = {}
    for (model, representation, reason), count in sorted(reuse_reason_counts.items()):
        reuse_reason_report.setdefault(model, {}).setdefault(representation, {})[reason] = count
    validation_report = {
        "phase": 6,
        "status": "staged-for-scientific-review",
        "phase5_membership_verified": True,
        "phase5_membership_row_count": 9048,
        "public_id_ordering": [
            "dataset order FamilyOWL, Pizza100, Pizza250, OWL2Bench",
            "hop 1hop then 2hop",
            "task group BQA then OEQA",
            "root entity",
            "normalized formal query",
            "canonical gold answer",
            "stable semantic provenance key",
            "semantic key final uniqueness tie-breaker",
        ],
        "counts": statistics,
        "bqa_pair_validation": {
            "pairs": len(pair_rows),
            "negative_links": sum(row["positive_task_id"] is not None for row in records),
            "unpaired": 0,
            "boundary_or_complexity_mismatches": 0,
        },
        "oeqa_explanation_validation": {
            "rows": sum(row["task_group"] == "OEQA" for row in records),
            "missing_answer_groups": 0,
            "missing_exact_inferred_object_provenance": 0,
        },
        "ar_validation": {
            **{key: ar_counts[key] for key in (
                "duplicate_exact_sentences", "unmapped_required_entities",
                "mapping_inconsistencies", "original_identifiers_remaining",
                "original_lexical_labels_remaining",
            )},
            "validated_rows": len(records),
        },
        "representation_alignment": {
            "aligned_task_ids": len(records),
            "mismatches": 0,
        },
        "reasoning_coverage": coverage_report,
        "complexity_distribution": complexity_report,
        "prediction_reuse": reuse_report,
        "prediction_reuse_reasons": reuse_reason_report,
        "pending_model_calls": len(pending_rows),
        "retained_v1_response_reuses": len(retained_reused),
        "prohibitions_confirmed": {
            "llm_or_api_calls": False,
            "release_published": False,
            "v1_0_written": False,
            "phase6_committed": False,
        },
    }
    write_json(stage / "validation_report.json", validation_report)

    membership_manifest = {
        "phase": 6,
        "membership_source": "data/output_v1_1_staging/phase5/membership_manifest.json",
        "phase5_manifest_sha256": sha256_file(PHASE5 / "membership_manifest.json"),
        "phase5_semantic_membership_sha256": sha256_file(PHASE5 / "semantic_membership.csv"),
        "row_count": len(records),
        "semantic_keys_sha256": sha256_text("\n".join(row["semantic_key"] for row in records) + "\n"),
        "public_ids_sha256": sha256_text("\n".join(str(row["task_id"]) for row in records) + "\n"),
        "sort_key": validation_report["public_id_ordering"],
    }
    write_json(stage / "membership_manifest.json", membership_manifest)

    manifest_path = stage / "RELEASE_MANIFEST.json"
    checksums_path = stage / "SHA256SUMS"
    excluded = {manifest_path, checksums_path}
    files = sorted(
        (path for path in stage.rglob("*") if path.is_file() and path not in excluded),
        key=lambda path: path.relative_to(stage).as_posix(),
    )
    file_entries = [
        {
            "path": path.relative_to(stage).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    release_manifest = {
        "benchmark": "CORE-LLM-Bench",
        "version": "v1.1.0-staging",
        "status": "not-released",
        "statistics": statistics,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "files": file_entries,
    }
    write_json(manifest_path, release_manifest)
    checksum_entries = file_entries + [
        {
            "path": manifest_path.relative_to(stage).as_posix(),
            "sha256": sha256_file(manifest_path),
        }
    ]
    checksums_path.write_text(
        "".join(
            f"{entry['sha256']}  {entry['path']}\n"
            for entry in sorted(checksum_entries, key=lambda item: item["path"])
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(stable_json(validation_report, indent=2))


if __name__ == "__main__":
    main()
