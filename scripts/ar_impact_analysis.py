"""Read-only Phase 3 impact analysis against the persisted v1.0 benchmark.

The script writes nothing. It rebuilds corrected abstraction state in memory
and emits one JSON report to stdout. No model or network client is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ar_validation import (  # noqa: E402
    abstract_answer,
    query_specific_text_mappings,
    render_abstract_context,
    replace_text,
    source_terms,
    validate_ar_row,
)
from scripts.llm_pipeline.verbalize_abstract import parse_mapping_file  # noqa: E402
from scripts.ontology_tools.abstraction.OntologyAbstractor import (  # noqa: E402
    build_text_mappings,
    create_abstraction_mappings,
    discover_entity_roles,
    flatten_mappings,
    merge_entity_roles,
)


BENCHMARKS = (
    "FamilyOWL_1hop",
    "FamilyOWL_2hop",
    "OWL2Bench_1hop",
    "OWL2Bench_2hop",
    "pizza_100_1hop",
    "pizza_100_2hop",
    "pizza_250_1hop",
    "pizza_250_2hop",
    "toy_example_1hop",
    "toy_example_2hop",
)
ISSUE_KEYS = (
    "duplicate_exact_sentences",
    "unmapped_required_entities",
    "mapping_inconsistencies",
    "original_identifiers_remaining",
    "original_lexical_labels_remaining",
)


def stream_json_array(path: Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        eof = False
        started = False
        while True:
            if not eof and len(buffer) < chunk_size:
                piece = handle.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        return
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"Expected JSON array: {path}")
                buffer = buffer[1:]
                started = True
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                piece = handle.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True
                continue
            yield value
            buffer = buffer[end:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]


def graphs(paths):
    for path in paths:
        yield Graph().parse(path, format="turtle")


def sha256_text(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def remap_abstract_text(
    text: str, old_text_mappings: dict[str, str], new_text_mappings: dict[str, str]
) -> str:
    token_targets: dict[str, set[str]] = defaultdict(set)
    for source, old_target in old_text_mappings.items():
        new_target = new_text_mappings.get(source)
        if new_target:
            token_targets[old_target].add(new_target)
    token_map = {
        old: next(iter(targets))
        for old, targets in token_targets.items()
        if len(targets) == 1
    }
    return replace_text(replace_text(text, token_map), new_text_mappings)


def is_phase1_disappearance(
    dataset_hop: str, qa: dict, task_type: str, answer_type: str
) -> bool:
    return (
        dataset_hop.startswith(("pizza_100_", "pizza_250_"))
        and answer_type.upper() == "BIN"
        and task_type == "Membership"
        and ("#DomainConcept>" in str(qa.get("SPARQL Query", ""))
             or "/DomainConcept>" in str(qa.get("SPARQL Query", "")))
    )


def prediction_inventory(root: Path):
    inventory = defaultdict(lambda: {"rows": 0, "predictions": 0})
    paths = list(root.glob("**/abs/abs_final_benchmark_results_FINAL.csv"))
    for path in paths:
        dataset_hop = path.relative_to(root).parts[0]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            response_columns = [name for name in reader.fieldnames or [] if name.endswith("_response")]
            for row in reader:
                key = (
                    row.get("Task ID", ""),
                    row.get("Root Entity", ""),
                    row.get("SPARQL Query", ""),
                )
                inventory[(dataset_hop, key)]["rows"] += 1
                inventory[(dataset_hop, key)]["predictions"] += sum(
                    bool(row.get(column, "").strip()) for column in response_columns
                )
    return inventory, [str(path) for path in paths]


def analyze_one(dataset_hop: str, repo: Path, prediction_rows):
    resource_dir = repo / "data" / "resources" / dataset_hop
    benchmark_file = repo / "final_benchmark" / f"{dataset_hop}.json"
    ttl_paths = sorted(resource_dir.glob("*.ttl"), key=lambda path: str(path))
    roles = merge_entity_roles(discover_entity_roles(graph) for graph in graphs(ttl_paths))
    uri_mappings = flatten_mappings(create_abstraction_mappings(roles))
    text_mappings, ambiguous_aliases = build_text_mappings(graphs(ttl_paths), uri_mappings)
    old_mapping_path = (
        repo
        / "data"
        / "output"
        / "abstracted_ontologies"
        / dataset_hop
        / "abstraction_mappings.txt"
    )
    old_text_mappings = parse_mapping_file(old_mapping_path)

    counts = Counter()
    pre_context_failures = Counter()
    post_context_failures = Counter()
    post_issue_terms = Counter()
    post_issue_examples = []
    breakdown = Counter()
    classification_rows = Counter()
    classification_predictions = Counter()
    reusable_fingerprints = []

    for group in stream_json_array(benchmark_file):
        root_entity = str(group["Root Entity"])
        group_task_type = str(group.get("Task Type", ""))
        group_answer_type = str(group.get("Answer Type", ""))
        source_path = resource_dir / f"{root_entity}.ttl"
        graph = Graph().parse(source_path, format="turtle")
        local_mappings = {
            original: abstract
            for original, abstract in uri_mappings.items()
            if (original, None, None) in graph
            or (None, original, None) in graph
            or (None, None, original) in graph
        }
        identifiers, labels = source_terms([graph], local_mappings)
        old_context = str(group.get("ABS Context", ""))
        new_context = render_abstract_context(graph, uri_mappings)
        context_changed = old_context != new_context
        if context_changed:
            counts["contexts_changed"] += 1

        context_common = dict(
            original_question="",
            original_answer="",
            answer_type="BIN",
            sparql_query="",
            abs_question="",
            abs_answer="",
            uri_mappings=uri_mappings,
            identifiers=identifiers,
            lexical_labels=labels,
            expected_abs_question="",
        )
        pre_context_result = validate_ar_row(
            **context_common,
            abs_context=old_context,
            text_mappings=old_text_mappings,
        )
        post_context_result = validate_ar_row(
            **context_common,
            abs_context=new_context,
            text_mappings=text_mappings,
        )
        group_pre_issues = {
            issue for issue in ISSUE_KEYS if getattr(pre_context_result, issue)
        }
        group_post_issues = {
            issue for issue in ISSUE_KEYS if getattr(post_context_result, issue)
        }
        for qa in group.get("QAs", []):
            counts["rows"] += 1
            answer_type = str(qa.get("Answer Type", group_answer_type))
            task_type = str(qa.get("Task Type", group_task_type))
            old_question = str(qa.get("ABS Question", ""))
            old_answer = str(qa.get("ABS Answer", qa.get("Answer", "")))
            original_question = str(qa.get("NL Question", ""))
            original_answer = str(qa.get("Answer", ""))
            new_question = remap_abstract_text(
                old_question, old_text_mappings, text_mappings
            )
            new_question = replace_text(
                new_question,
                query_specific_text_mappings(
                    str(qa.get("SPARQL Query", "")), graph, uri_mappings
                ),
            )
            new_answer = abstract_answer(original_answer, answer_type, text_mappings)
            question_changed = old_question != new_question
            answer_changed = old_answer != new_answer
            if question_changed:
                counts["questions_changed"] += 1
            if answer_changed:
                counts["answers_changed"] += 1
            if question_changed or context_changed:
                counts["prompt_rows_changed"] += 1
            task_family = "BQA" if answer_type.upper() == "BIN" else "OEQA"
            if question_changed or context_changed or answer_changed:
                counts[f"affected_{task_family}"] += 1
                breakdown[task_family] += 1

            common = dict(
                original_question=original_question,
                original_answer=original_answer,
                answer_type=answer_type,
                sparql_query=str(qa.get("SPARQL Query", "")),
                uri_mappings=uri_mappings,
                identifiers=identifiers,
                lexical_labels=labels,
            )
            pre = validate_ar_row(
                **common,
                abs_question=old_question,
                abs_context="",
                abs_answer=old_answer,
                text_mappings=old_text_mappings,
                expected_abs_question=old_question,
            )
            post = validate_ar_row(
                **common,
                abs_question=new_question,
                abs_context="",
                abs_answer=new_answer,
                text_mappings=text_mappings,
                expected_abs_question=new_question,
            )
            for issue in ISSUE_KEYS:
                pre_values = list(getattr(pre, issue)) + list(
                    getattr(pre_context_result, issue)
                )
                post_values = list(getattr(post, issue)) + list(
                    getattr(post_context_result, issue)
                )
                if pre_values:
                    counts[f"pre_rows_{issue}"] += 1
                if post_values:
                    counts[f"post_rows_{issue}"] += 1
                    post_issue_terms.update(
                        f"{issue}:{value}" for value in post_values
                    )
                    if len(post_issue_examples) < 10:
                        post_issue_examples.append(
                            {
                                "task_id": str(qa.get("Task ID", "")),
                                "root_entity": root_entity,
                                "issue": issue,
                                "values": sorted(set(post_values)),
                            }
                        )

            if is_phase1_disappearance(dataset_hop, qa, task_type, answer_type):
                classification = "E"
            elif question_changed:
                classification = "B"
            elif context_changed:
                classification = "C"
            elif answer_changed:
                classification = "D"
            else:
                classification = "A"
            classification_rows[classification] += 1
            key = (
                str(qa.get("Task ID", "")),
                root_entity,
                str(qa.get("SPARQL Query", "")),
            )
            prediction_count = prediction_rows.get((dataset_hop, key), {}).get("predictions", 0)
            classification_predictions[classification] += prediction_count
            if classification == "A":
                reusable_fingerprints.append(
                    (key, sha256_text(old_question, old_context))
                )

        for issue in group_pre_issues:
            pre_context_failures[issue] += 1
        for issue in group_post_issues:
            post_context_failures[issue] += 1

    reusable_digest = hashlib.sha256(
        json.dumps(sorted(reusable_fingerprints), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "counts": dict(counts),
        "pre_context_failures": dict(pre_context_failures),
        "post_context_failures": dict(post_context_failures),
        "post_issue_terms": dict(post_issue_terms.most_common(30)),
        "post_issue_examples": post_issue_examples,
        "row_classification": dict(classification_rows),
        "prediction_classification": dict(classification_predictions),
        "ambiguous_text_aliases": len(ambiguous_aliases),
        "reusable_identity_prompt_hash_manifest_sha256": reusable_digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="*", choices=BENCHMARKS, default=list(BENCHMARKS))
    args = parser.parse_args()
    repo = args.repo.resolve()
    predictions, prediction_files = prediction_inventory(
        repo / "data" / "output" / "final_benchmark_llm_results"
    )
    report = {
        "datasets": {},
        "prediction_source_files": prediction_files,
        "notes": {
            "classification_precedence": "E, then B, then C, then D, else A",
            "E_scope": "deterministically identifiable Pizza DomainConcept BQA removals only",
            "writes": "none",
            "api_calls": "none",
        },
    }
    for dataset_hop in args.datasets:
        report["datasets"][dataset_hop] = analyze_one(
            dataset_hop, repo, predictions
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
