#!/usr/bin/env python3
"""Finalize only the public v1.1 explanation schema from frozen Phase-7D data.

The current staged benchmark is authoritative for every model-facing and
membership field.  Symbolic Phase-4 records are consulted only to rebuild
explanations.  The model-input gate runs before the output directory is made.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.explanation_schema import (  # noqa: E402
    PUBLIC_TAGS,
    answer_group,
    complete_explanation,
    strip_private_semantic_identities,
)
from scripts.phase6_materialize_release import (  # noqa: E402
    DATASETS,
    PROMPT_TEMPLATE_VERSION,
    load_explanation_index,
    local_name,
    parse_triple,
    prompt_hash,
)

DEFAULT_SOURCE = ROOT / "release" / "v1.1.0-staging"
DEFAULT_OUTPUT = ROOT / "release" / "v1.1.0-schema-finalized"
GEMINI_OBSERVATIONS = (
    ROOT / "release" / "v1.1.0-phase7d" / "responses" / "gemini_observations.jsonl"
)
REPRESENTATIONS = ("NL", "FS", "AR")
EXPLANATION_FIELDS = {
    "answer_explanations",
    "complete_explanation",
    "complete_explanation_combination_count",
    "explanations",
    "legacy_explanation_fields",
    "maximum_axiom_count",
    "minimum_axiom_count",
    "structured_explanations",
}
EXPECTED_DATASET_HOP = {
    "FamilyOWL/1hop": 1881,
    "FamilyOWL/2hop": 1881,
    "Pizza100/1hop": 495,
    "Pizza100/2hop": 495,
    "Pizza250/1hop": 618,
    "Pizza250/2hop": 618,
    "OWL2Bench/1hop": 1467,
    "OWL2Bench/2hop": 1593,
}
OWLAPI_VERSION = "5.1.20"
MANCHESTER_RENDERER = "ManchesterOWLSyntaxOWLObjectRendererImpl"
MANCHESTER_PREFIX_POLICY = "full-iri-no-prefixes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_manifest(path: Path) -> tuple[list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), raw


def decoded(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def model_input_hashes(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, str], str]:
    result: dict[tuple[int, str], str] = {}
    for row in rows:
        inputs = {
            "NL": (row["nl_question"], row["nl_context"]),
            "FS": (row["fs_query"], row["fs_context"]),
            "AR": (row["ar_question"], row["ar_context"]),
        }
        for representation, (question, context) in inputs.items():
            result[(int(row["task_id"]), representation)] = prompt_hash(
                str(question), str(context), representation, str(row["answer_type"])
            )
    return result


def prewrite_model_input_gate(
    rows: list[dict[str, Any]], source: Path
) -> tuple[dict[str, int], bytes, dict[tuple[int, str], str]]:
    manifest_rows, manifest_bytes = read_manifest(source / "model_input_manifest.csv")
    baseline = {
        (int(row["task_id"]), row["representation"]): row["input_hash"]
        for row in manifest_rows
    }
    current = model_input_hashes(rows)
    if len(manifest_rows) != 27144 or set(baseline) != set(current):
        raise ValueError("PRE-WRITE STOP: model-input identity/cardinality changed")
    changes = Counter(
        representation
        for (task_id, representation), digest in current.items()
        if baseline[(task_id, representation)] != digest
    )
    report = {name: changes[name] for name in REPRESENTATIONS}
    if any(report.values()):
        raise ValueError(f"PRE-WRITE STOP: model-input hashes changed: {report}")
    versions = {row["prompt_template_version"] for row in manifest_rows}
    if versions != {PROMPT_TEMPLATE_VERSION}:
        raise ValueError(f"PRE-WRITE STOP: unexpected prompt versions: {versions}")
    return report, manifest_bytes, baseline


def semantic_axiom_identities(
    indexes: Iterable[dict[tuple[str, str, str], dict[str, Any]]],
) -> list[str]:
    result = {
        str(axiom["identity"])
        for index in indexes
        for record in index.values()
        for proof in record.get("structuredExplanations", [])
        for axiom in proof.get("semanticAxioms", [])
    }
    if not result:
        raise ValueError("No semantic OWLAPI axiom identities found")
    return sorted(result)


def render_semantic_axioms(identities: list[str]) -> dict[str, str]:
    """Render through OWLAPI; Python never rewrites OWL syntax."""
    temp = ROOT / "tmp_schema_render"
    if temp.exists():
        raise FileExistsError(f"Renderer workspace already exists: {temp}")
    temp.mkdir()
    try:
        input_path = temp / "identities.b64"
        output_path = temp / "renderings.b64.tsv"
        input_path.write_text(
            "".join(
                base64.b64encode(identity.encode()).decode("ascii") + "\n"
                for identity in identities
            ),
            encoding="ascii",
            newline="\n",
        )
        command = [
            "mvn.cmd" if sys.platform == "win32" else "mvn",
            "-q",
            "-DskipTests",
            "-Dexec.mainClass=com.example.explanation.SemanticAxiomRenderer",
            f"-Dexec.args={input_path} {output_path}",
            "compile",
            "org.codehaus.mojo:exec-maven-plugin:3.6.3:java",
        ]
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if completed.returncode:
            raise RuntimeError(
                "OWLAPI Manchester rendering failed:\n"
                + completed.stdout
                + completed.stderr
            )
        result: dict[str, str] = {}
        for line in output_path.read_text(encoding="ascii").splitlines():
            identity, rendering = line.split("\t", 1)
            result[base64.b64decode(identity).decode()] = base64.b64decode(
                rendering
            ).decode()
        if set(result) != set(identities):
            raise ValueError("OWLAPI renderer did not return every semantic identity")
        return result
    finally:
        for path in (temp / "identities.b64", temp / "renderings.b64.tsv"):
            if path.exists():
                path.unlink()
        temp.rmdir()


def explanation_for(
    row: dict[str, Any],
    index: dict[tuple[str, str, str], dict[str, Any]],
    renderings: dict[str, str],
) -> dict[str, Any]:
    render = renderings.__getitem__
    if row["task_group"] == "BQA":
        parts = str(row["source_provenance_key"]).split("||", 1)[-1].split("|")
        if len(parts) != 3 or tuple(parts) not in index:
            raise ValueError(
                f"Missing exact BQA explanation source for task {row['task_id']}"
            )
        metadata: dict[str, Any] = {}
        if str(row["gold_answer"]).upper() == "FALSE":
            if not row["positive_task_id"] or not row["positive_semantic_key"]:
                raise ValueError(f"FALSE BQA lacks paired positive: {row['task_id']}")
            metadata = {
                "explanation_role": "paired_positive_entailment_provenance",
                "proves_label": "TRUE",
                "source_positive_task_id": int(row["positive_task_id"]),
                "source_positive_semantic_key": row["positive_semantic_key"],
            }
        groups = [
            answer_group(
                str(row["gold_answer"]).upper(),
                index[tuple(parts)].get("structuredExplanations", []),
                render,
                **metadata,
            )
        ]
    else:
        subject_uri, predicate_uri, _ = parse_triple(str(row["formal_query"]))
        subject = local_name(subject_uri)
        predicate = (
            "rdf:type"
            if predicate_uri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            else local_name(predicate_uri)
        )
        groups = []
        for answer in [part.strip() for part in str(row["gold_answer"]).split(";") if part.strip()]:
            source = index.get((subject, predicate, answer))
            if source is None:
                raise ValueError(
                    "Missing exact inferred.object OEQA source: "
                    f"{subject}|{predicate}|{answer}"
                )
            inferred = source.get("inferred") or {}
            if str(inferred.get("object")) != answer:
                raise ValueError(f"OEQA inferred.object drift for task {row['task_id']}")
            groups.append(
                answer_group(
                    answer,
                    source.get("structuredExplanations", []),
                    render,
                    source_provenance={"inferred.object": answer},
                )
            )
    return strip_private_semantic_identities(
        {
            "answer_explanations": groups,
            "complete_explanation": complete_explanation(groups),
        }
    )


def transform_rows(
    source_rows: list[dict[str, Any]],
    indexes: dict[tuple[str, str], dict[tuple[str, str, str], dict[str, Any]]],
    renderings: dict[str, str],
) -> list[dict[str, Any]]:
    transformed = []
    for original in source_rows:
        row = dict(original)
        explanation = explanation_for(
            row, indexes[(str(row["dataset_key"]), str(row["hop"]))], renderings
        )
        for field in EXPLANATION_FIELDS:
            row.pop(field, None)
        row["answer_explanations"] = encoded(explanation["answer_explanations"])
        row["complete_explanation"] = encoded(explanation["complete_explanation"])
        transformed.append(row)
    return transformed


def assert_only_explanations_changed(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> None:
    if len(before) != len(after):
        raise ValueError("Row count changed")
    for old, new in zip(before, after):
        old_public = {k: v for k, v in old.items() if k not in EXPLANATION_FIELDS}
        new_public = {k: v for k, v in new.items() if k not in EXPLANATION_FIELDS}
        if old_public != new_public:
            raise ValueError(f"Non-explanation field changed at task {old['task_id']}")


def validate_membership(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [int(row["task_id"]) for row in rows]
    if ids != list(range(1, 9049)):
        raise ValueError("Task IDs are not exactly ordered 1..9048")
    tasks = Counter(str(row["task_group"]) for row in rows)
    answers = Counter(
        str(row["gold_answer"]).upper()
        for row in rows
        if row["task_group"] == "BQA"
    )
    dataset_hop = Counter(f"{row['dataset']}/{row['hop']}" for row in rows)
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task_group"] == "BQA":
            pair_groups[str(row["pair_group_id"])].append(row)
    complete_pairs = sum(
        len(pair) == 2
        and {str(row["gold_answer"]).upper() for row in pair} == {"TRUE", "FALSE"}
        for pair in pair_groups.values()
    )
    if (
        len(rows) != 9048
        or tasks != {"BQA": 6032, "OEQA": 3016}
        or answers != {"TRUE": 3016, "FALSE": 3016}
        or complete_pairs != 3016
        or dict(dataset_hop) != EXPECTED_DATASET_HOP
    ):
        raise ValueError(
            f"Frozen membership mismatch: {tasks}, {answers}, {complete_pairs}, {dataset_hop}"
        )
    return {
        "total": len(rows),
        "BQA": tasks["BQA"],
        "TRUE": answers["TRUE"],
        "FALSE": answers["FALSE"],
        "complete_bqa_pairs": complete_pairs,
        "OEQA": tasks["OEQA"],
        "dataset_hop": dict(dataset_hop),
    }


def validate_explanations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    alternatives: dict[str, list[int]] = defaultdict(list)
    tied_minima: list[int] = []
    combinations: list[int] = []
    exercised: set[str] = set()
    for row in rows:
        groups = decoded(row["answer_explanations"])
        complete = decoded(row["complete_explanation"])
        if not groups or any(not group["alternatives"] for group in groups):
            raise ValueError(f"Missing explanation alternative for task {row['task_id']}")
        task = str(row["task_group"])
        alternatives[task].extend(group["alternative_count"] for group in groups)
        expected_combinations = 1
        for group in groups:
            expected_combinations *= group["alternative_count"]
            for proof in group["alternatives"]:
                if proof["tag_sequence"] != "".join(
                    axiom["tag"] for axiom in proof["axioms"]
                ):
                    raise ValueError("tag_sequence is not stored axiom order")
                exercised.update(proof["tag_sequence"])
        if complete["combination_count"] != expected_combinations:
            raise ValueError(f"Combination count mismatch at task {row['task_id']}")
        if not complete["minimum_explanations"]:
            raise ValueError(f"No global minimum at task {row['task_id']}")
        public_unions: set[tuple[tuple[str, str], ...]] = set()
        public_counts: list[int] = []
        for selection in product(*(group["alternatives"] for group in groups)):
            union = {
                (axiom["axiom"], axiom["tag"])
                for proof in selection
                for axiom in proof["axioms"]
            }
            public_unions.add(tuple(sorted(union)))
            public_counts.append(len(union))
        if (
            min(public_counts) != complete["min_axiom_count"]
            or max(public_counts) != complete["max_axiom_count"]
        ):
            raise ValueError(f"Complete min/max mismatch at task {row['task_id']}")
        counts = [len(item["axioms"]) for item in complete["minimum_explanations"]]
        if set(counts) != {complete["min_axiom_count"]}:
            raise ValueError(f"Minimum axiom count mismatch at task {row['task_id']}")
        semantic_renderings = {
            tuple(sorted((item["axiom"], item["tag"]) for item in proof["axioms"]))
            for proof in complete["minimum_explanations"]
        }
        if len(semantic_renderings) != len(complete["minimum_explanations"]):
            raise ValueError(f"Duplicate tied minimum at task {row['task_id']}")
        expected_minima = {union for union in public_unions if len(union) == min(public_counts)}
        if semantic_renderings != expected_minima:
            raise ValueError(f"Incomplete tied minima at task {row['task_id']}")
        if task == "FALSE":
            raise AssertionError("unreachable")
        if task == "BQA" and str(row["gold_answer"]).upper() == "FALSE":
            group = groups[0]
            if (
                group.get("explanation_role")
                != "paired_positive_entailment_provenance"
                or group.get("proves_label") != "TRUE"
                or group.get("source_positive_task_id") != int(row["positive_task_id"])
            ):
                raise ValueError(f"Invalid FALSE provenance at task {row['task_id']}")
        if task == "OEQA":
            expected_answers = [x.strip() for x in str(row["gold_answer"]).split(";") if x.strip()]
            if [g["source_provenance"]["inferred.object"] for g in groups] != expected_answers:
                raise ValueError(f"OEQA provenance mismatch at task {row['task_id']}")
            tied_minima.append(len(complete["minimum_explanations"]))
            combinations.append(complete["combination_count"])
    if exercised - PUBLIC_TAGS or "M" in exercised or "P" in exercised:
        raise ValueError(f"Invalid public primitive tags: {sorted(exercised)}")

    def stats(values: list[int]) -> dict[str, Any]:
        return {
            "groups": len(values),
            "total_alternatives": sum(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": round(sum(values) / len(values), 6),
        }

    return {
        "primitive_tags_exercised": sorted(exercised),
        "BQA_alternatives": stats(alternatives["BQA"]),
        "OEQA_alternatives": stats(alternatives["OEQA"]),
        "OEQA_rows_with_multiple_tied_global_minima": sum(x > 1 for x in tied_minima),
        "OEQA_maximum_tied_global_minima": max(tied_minima),
        "OEQA_maximum_combination_count": max(combinations),
    }


def validate_gemini(
    baseline: dict[tuple[int, str], str], path: Path
) -> dict[str, Any]:
    observed: dict[tuple[int, str], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["task_id"]), row["representation"])
            if key in observed:
                raise ValueError(f"Duplicate Gemini observation: {key}")
            observed[key] = row["input_hash"]
    mismatches = sum(observed.get(key) != digest for key, digest in baseline.items())
    if len(observed) != 27144 or set(observed) != set(baseline) or mismatches:
        raise ValueError(
            f"Gemini input mismatch: rows={len(observed)}, mismatches={mismatches}"
        )
    return {
        "observations": len(observed),
        "input_hash_mismatches": 0,
        "rerun_required": False,
        "response_files_modified_or_moved": False,
    }


def transform_benchmark_files(
    source: Path, output: Path, rows: list[dict[str, Any]]
) -> None:
    by_id = {int(row["task_id"]): row for row in rows}
    paths = sorted((source / "benchmark").glob("*.json"))
    if len(paths) != 8:
        raise ValueError(f"Expected 8 benchmark files, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for group in payload:
            for qa in group["QAs"]:
                transformed = by_id[int(qa["task_id"])]
                old_public = {k: v for k, v in qa.items() if k not in EXPLANATION_FIELDS}
                for field in EXPLANATION_FIELDS:
                    qa.pop(field, None)
                qa["answer_explanations"] = decoded(transformed["answer_explanations"])
                qa["complete_explanation"] = decoded(transformed["complete_explanation"])
                new_public = {k: v for k, v in qa.items() if k not in EXPLANATION_FIELDS}
                if old_public != new_public:
                    raise ValueError(f"Benchmark non-explanation drift at {qa['task_id']}")
        write_json(output / "benchmark" / path.name, payload)


def refresh_manifests(output: Path, report: dict[str, Any]) -> None:
    validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    validation["explanation_schema_finalization"] = report
    write_json(output / "validation_report.json", validation)
    manifest_path = output / "RELEASE_MANIFEST.json"
    checksums_path = output / "SHA256SUMS"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = {manifest_path.resolve(), checksums_path.resolve()}
    files = sorted(
        (p for p in output.rglob("*") if p.is_file() and p.resolve() not in excluded),
        key=lambda p: p.relative_to(output).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    old_manifest["phase"] = "explanation-schema-finalized"
    old_manifest["files"] = entries
    old_manifest["explanation_schema"] = report["schema"]
    write_json(manifest_path, old_manifest)
    checksum_entries = entries + [
        {"path": "RELEASE_MANIFEST.json", "sha256": sha256_file(manifest_path)}
    ]
    checksums_path.write_text(
        "".join(
            f"{item['sha256']}  {item['path']}\n"
            for item in sorted(checksum_entries, key=lambda x: x["path"])
        ),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gemini-observations", type=Path, default=GEMINI_OBSERVATIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if ROOT.resolve() not in output.parents or source == output:
        raise ValueError("Output must be a distinct directory inside the repository")
    if output.exists():
        raise FileExistsError(f"Finalization output already exists: {output}")
    source_rows = pq.read_table(source / "core_llm_bench_v1_1.parquet").to_pylist()
    membership = validate_membership(source_rows)
    model_changes, manifest_bytes, baseline = prewrite_model_input_gate(source_rows, source)
    print(f"Pre-write model-input gate passed: {model_changes}", flush=True)

    indexes = {
        (dataset, hop): load_explanation_index(dataset, hop)
        for dataset in DATASETS
        for hop in ("1hop", "2hop")
    }
    identities = semantic_axiom_identities(indexes.values())
    print(f"Rendering {len(identities)} semantic axioms with OWLAPI", flush=True)
    renderings = render_semantic_axioms(identities)
    transformed = transform_rows(source_rows, indexes, renderings)
    assert_only_explanations_changed(source_rows, transformed)
    explanation_report = validate_explanations(transformed)
    gemini_report = validate_gemini(baseline, args.gemini_observations.resolve())

    report = {
        "architecture": "post-process immutable Phase-7D staging; symbolic data used only for explanations",
        "source": source.relative_to(ROOT).as_posix(),
        "model_input_changes": model_changes,
        "model_input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "membership": membership,
        "explanations": explanation_report,
        "gemini": gemini_report,
        "schema": {
            "canonical_fields": ["answer_explanations", "complete_explanation"],
            "removed_fields": sorted(EXPLANATION_FIELDS - {"answer_explanations", "complete_explanation"}),
            "primitive_tags": list("DHIRSTN"),
            "tag_sequence_semantics": "primitive tags corresponding to axioms in stored axiom order",
            "semantic_identity": "annotation-free OWLAPI axiom identity",
            "m_status": "derived metadata only; excluded from primitive tags and complexity",
            "owlapi_version": OWLAPI_VERSION,
            "manchester_renderer": MANCHESTER_RENDERER,
            "prefix_policy": MANCHESTER_PREFIX_POLICY,
        },
    }

    shutil.copytree(source, output)
    transform_benchmark_files(source, output, transformed)
    pq.write_table(
        pa.Table.from_pylist(transformed),
        output / "core_llm_bench_v1_1.parquet",
        compression="zstd",
        version="2.6",
    )
    if (output / "model_input_manifest.csv").read_bytes() != manifest_bytes:
        raise ValueError("Copied model_input_manifest.csv changed")
    roundtrip = pq.read_table(output / "core_llm_bench_v1_1.parquet").to_pylist()
    assert_only_explanations_changed(source_rows, roundtrip)
    validate_explanations(roundtrip)
    refresh_manifests(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
