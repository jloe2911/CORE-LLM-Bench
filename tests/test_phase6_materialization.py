from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import phase6_materialize_release as phase6  # noqa: E402
from scripts.llm_pipeline.verbalize_ontologies import (  # noqa: E402
    describe_individual_with_domain_independence,
)


STAGE = ROOT / "release" / "v1.1.0-staging"
PHASE5_MEMBERSHIP = (
    ROOT / "data" / "output_v1_1_staging" / "phase5" / "semantic_membership.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase6MaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = read_csv(STAGE / "task_id_mapping.csv")
        cls.pairs = read_csv(STAGE / "bqa_pair_mapping.csv")
        cls.inputs = read_csv(STAGE / "model_input_manifest.csv")
        cls.pending = read_csv(STAGE / "pending_model_runs.csv")
        cls.report = json.loads(
            (STAGE / "validation_report.json").read_text(encoding="utf-8")
        )

    def test_frozen_phase5_membership_is_hash_verified_and_unchanged(self) -> None:
        rows, manifest = phase6.verify_phase5()
        self.assertEqual(len(rows), 9048)
        membership_record = next(
            record
            for record in manifest["artifacts"]
            if record["path"].endswith("semantic_membership.csv")
        )
        self.assertEqual(sha256_file(PHASE5_MEMBERSHIP), membership_record["sha256"])
        self.assertEqual(
            {row["semantic_key"] for row in rows},
            {row["semantic_key"] for row in self.mapping},
        )

    def test_public_ids_are_exactly_one_through_9048_in_canonical_order(self) -> None:
        ids = [int(row["new_task_id"]) for row in self.mapping]
        self.assertEqual(ids, list(range(1, 9049)))
        phase5_rows, _ = phase6.verify_phase5()
        expected_keys = [
            row["semantic_key"] for row in sorted(phase5_rows, key=phase6.public_sort_key)
        ]
        self.assertEqual(
            [row["semantic_key"] for row in self.mapping], expected_keys
        )
        self.assertTrue(all(row["legacy_task_id"] for row in self.mapping))

    def test_nl_fs_ar_are_aligned_and_exact_prompt_hashes_recompute(self) -> None:
        by_id: dict[int, set[str]] = defaultdict(set)
        manifest_hashes: dict[tuple[int, str], str] = {}
        for row in self.inputs:
            task_id = int(row["task_id"])
            by_id[task_id].add(row["representation"])
            manifest_hashes[(task_id, row["representation"])] = row["input_hash"]
            self.assertRegex(row["input_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(self.inputs), 9048 * 3)
        self.assertEqual(set(by_id), set(range(1, 9049)))
        self.assertTrue(all(value == {"NL", "FS", "AR"} for value in by_id.values()))

        columns = [
            "task_id", "answer_type", "nl_question", "nl_context",
            "fs_query", "fs_context", "ar_question", "ar_context",
        ]
        for row in pq.read_table(
            STAGE / "core_llm_bench_v1_1.parquet", columns=columns
        ).to_pylist():
            sources = {
                "NL": (row["nl_question"], row["nl_context"]),
                "FS": (row["fs_query"], row["fs_context"]),
                "AR": (row["ar_question"], row["ar_context"]),
            }
            for representation, (question, context) in sources.items():
                self.assertEqual(
                    phase6.prompt_hash(
                        question, context, representation, row["answer_type"]
                    ),
                    manifest_hashes[(row["task_id"], representation)],
                )

    def test_bqa_pairs_link_one_true_and_one_false_within_boundaries(self) -> None:
        by_id = {int(row["new_task_id"]): row for row in self.mapping}
        self.assertEqual(len(self.pairs), 3016)
        linked_ids: set[int] = set()
        for pair in self.pairs:
            positive_id = int(pair["positive_task_id"])
            negative_id = int(pair["negative_task_id"])
            positive = by_id[positive_id]
            negative = by_id[negative_id]
            self.assertEqual(positive["gold_answer"], "TRUE")
            self.assertEqual(negative["gold_answer"], "FALSE")
            for field in ("dataset", "hop", "root_entity"):
                self.assertEqual(positive[field], negative[field])
                self.assertEqual(positive[field], pair[field])
            linked_ids.update((positive_id, negative_id))
        self.assertEqual(len(linked_ids), 6032)

    def test_corrected_oeqa_explanations_and_complexity_metadata(self) -> None:
        columns = [
            "task_group", "answer_explanations", "complexity_bin",
            "raw_minimum_complete_primitive_tag_complexity",
            "raw_maximum_complete_primitive_tag_complexity", "m_status",
        ]
        rows = pq.read_table(
            STAGE / "core_llm_bench_v1_1.parquet", columns=columns
        ).to_pylist()
        oeqa_count = 0
        complexity = defaultdict(Counter)
        for row in rows:
            complexity[row["task_group"]][row["complexity_bin"]] += 1
            self.assertIn(row["m_status"], {"always", "never", "selection-dependent"})
            self.assertLessEqual(
                row["raw_minimum_complete_primitive_tag_complexity"],
                row["raw_maximum_complete_primitive_tag_complexity"],
            )
            if row["task_group"] != "OEQA":
                continue
            oeqa_count += 1
            explanations = json.loads(row["answer_explanations"])
            self.assertGreater(len(explanations), 0)
            for answer_group in explanations:
                self.assertIn("answer", answer_group)
                self.assertGreater(len(answer_group["alternatives"]), 0)
                self.assertEqual(
                    answer_group["source_provenance"]["inferred.object"],
                    answer_group["answer"],
                )
        self.assertEqual(oeqa_count, 3016)
        expected = {
            task: Counter(values)
            for task, values in self.report["complexity_distribution"].items()
        }
        self.assertEqual(dict(complexity), expected)

    def test_all_ar_validators_are_zero(self) -> None:
        ar = self.report["ar_validation"]
        self.assertEqual(ar["validated_rows"], 9048)
        for key in (
            "duplicate_exact_sentences", "unmapped_required_entities",
            "mapping_inconsistencies", "original_identifiers_remaining",
            "original_lexical_labels_remaining",
        ):
            self.assertEqual(ar[key], 0)

    def test_multi_type_verbalization_is_independent_of_graph_insertion_order(self) -> None:
        namespace = Namespace("https://example.invalid/abstract#")
        individual = namespace.Individual1
        classes = {namespace.Class117, namespace.Class57}

        def render(order: list) -> str:
            graph = Graph()
            for class_uri in classes:
                graph.add((class_uri, RDF.type, OWL.Class))
            graph.add((individual, RDF.type, OWL.NamedIndividual))
            for class_uri in order:
                graph.add((individual, RDF.type, class_uri))
            return describe_individual_with_domain_independence(
                graph, individual, classes, set(), {individual}
            )

        expected = (
            "Individual1 is an instance of Class117 and an instance of Class57."
        )
        self.assertEqual(render([namespace.Class117, namespace.Class57]), expected)
        self.assertEqual(render([namespace.Class57, namespace.Class117]), expected)

    def test_pending_manifest_is_exactly_derived_and_stably_sorted(self) -> None:
        models = {
            "GPT-5 mini", "Gemini 2.5 Flash-Lite", "Qwen3-30B-A3B-Instruct"
        }
        expected = {
            (row["task_id"], row["representation"], model, row["input_hash"])
            for row in self.inputs
            for model in models
        }
        actual = {
            (
                row["task_id"], row["representation"], row["model"],
                row["input_hash"],
            )
            for row in self.pending
        }
        self.assertEqual(actual, expected)
        self.assertEqual({row["status"] for row in self.pending}, {"pending"})
        self.assertEqual(len(actual), self.report["pending_model_calls"])
        ordering = [
            (
                int(row["task_id"]),
                phase6.REPRESENTATIONS.index(row["representation"]),
                row["model"],
            )
            for row in self.pending
        ]
        self.assertEqual(ordering, sorted(ordering))

    def test_release_checksums_cover_all_payload_files(self) -> None:
        checksum_rows = {}
        for line in (STAGE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            checksum_rows[relative] = digest
        expected_paths = {
            path.relative_to(STAGE).as_posix()
            for path in STAGE.rglob("*")
            if path.is_file()
            and path.name not in {"SHA256SUMS", "LARGE_ARTIFACTS.md"}
        }
        excluded_large = {"benchmark/FamilyOWL_2hop.json"}
        self.assertEqual(set(checksum_rows) - expected_paths, excluded_large)
        self.assertEqual(expected_paths - set(checksum_rows), set())
        for relative, digest in checksum_rows.items():
            path = STAGE / relative
            if path.exists():
                self.assertEqual(sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
