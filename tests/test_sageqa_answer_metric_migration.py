import contextlib
import csv
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.create_paper_results_table import (
    DuplicateModelSettingSourceError,
    discover_validated_sources,
)
from scripts.llm_pipeline.sageqa_answer_metrics import (
    BenchmarkMismatchError,
    SAGEQA_EVALUATOR_CANONICAL_SHA256,
    SAGEQA_EVALUATOR_SHA256,
    SAGEQA_SOURCE_COMMIT,
    evaluate,
    load_sageqa_evaluator,
    read_csv_rows,
    sageqa_evaluator_path,
    score_checkpoint_rows,
    validate_checkpoint_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data/output/final_benchmark_llm_results"


@contextlib.contextmanager
def writable_temp_dir():
    parent = PROJECT_ROOT / ".test_tmp"
    path = parent / str(uuid.uuid4())
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SageQAAnswerMetricMigrationTests(unittest.TestCase):
    def test_authoritative_evaluator_is_pinned(self):
        module = load_sageqa_evaluator()
        self.assertEqual(
            SAGEQA_EVALUATOR_SHA256,
            "1610a67d64c48d46e0530dc71ffe073ca6ddae98d0a9b3c83116b0009a384713",
        )
        self.assertEqual(
            SAGEQA_EVALUATOR_CANONICAL_SHA256,
            "86b9bedfb3784145f3d20e2b9b8b6082ee4252e57918807bf02529b3904eefd6",
        )
        self.assertEqual(
            SAGEQA_SOURCE_COMMIT,
            "dbdbb50708bdc6c686ef82518ec71c1d1bf55985",
        )
        self.assertIn(PROJECT_ROOT, sageqa_evaluator_path().parents)
        self.assertTrue(callable(module.answer_set_scores))
        self.assertTrue(callable(module.evaluate))

    def test_rejects_mismatched_benchmark_ids(self):
        benchmark = [{"Task ID": "q1", "Answer Type": "BIN", "Answer": "TRUE"}]
        checkpoint = [{"Task ID": "q2", "Answer Type": "BIN", "Answer": "TRUE"}]
        with self.assertRaises(BenchmarkMismatchError):
            validate_checkpoint_rows(checkpoint, benchmark)

    def test_duplicate_valid_model_setting_sources_are_an_error(self):
        benchmark = [{"Task ID": "q1", "Answer Type": "BIN", "Answer": "TRUE"}]
        checkpoint = [
            {
                **benchmark[0],
                "model_final_answer": "TRUE",
                "model_confidence_score": "1.0",
            }
        ]
        with writable_temp_dir() as root:
            write_csv(
                root / "data/output/FamilyOWL/1hop/SPARQL_questions_sampling.csv",
                benchmark,
            )
            for source_name in ("source_a", "source_b"):
                write_csv(
                    root
                    / "results/FamilyOWL_1hop"
                    / source_name
                    / "nl/LATEST_checkpoint.csv",
                    checkpoint,
                )
            with self.assertRaises(DuplicateModelSettingSourceError):
                discover_validated_sources(root / "results", project_root=root)

    def test_all_chapter4_predictions_match_sageqa_evaluate_per_example(self):
        sources, rejected = discover_validated_sources(RESULTS_ROOT)
        self.assertEqual(len(sources), 72)
        self.assertTrue(any("checkpoint=489" in message for message in rejected))

        qwen_family_2hop = [
            source
            for key, source in sources.items()
            if key[0] == "FamilyOWL"
            and key[1] == "2hop"
            and key[3] == "openrouter_qwen_qwen3_30b_a3b_instruct_2507"
        ]
        self.assertEqual(len(qwen_family_2hop), 3)
        self.assertTrue(
            all(
                "openrouter_google_gemini_2_5_flash_lite__openrouter_qwen_"
                in str(source.checkpoint_path)
                for source in qwen_family_2hop
            )
        )

        details = []
        answer_rows = []
        adapter_scores = {}
        for key, source in sorted(sources.items()):
            rows = read_csv_rows(source.checkpoint_path)
            scored = score_checkpoint_rows(rows, source.model_name)
            self.assertEqual(len(scored["per_question"]), len(rows))
            for item in scored["per_question"]:
                if item["answer_type"] == "BIN":
                    self.assertEqual(item["answer_em"], item["answer_f1"])
                example_id = "::".join((*key, item["task_id"]))
                details.append(
                    {
                        "example_id": example_id,
                        "answer": item["gold"],
                        "evaluation_scope": "answer_only",
                    }
                )
                answer_rows.append(
                    {
                        "example_id": example_id,
                        "predicted_answer": item["prediction"],
                    }
                )
                adapter_scores[example_id] = (
                    item["answer_em"],
                    item["answer_f1"],
                )

        with writable_temp_dir() as temp:
            details_path = temp / "details.json"
            answers_path = temp / "answers.jsonl"
            details_path.write_text(json.dumps(details), encoding="utf-8")
            answers_path.write_text(
                "".join(json.dumps(row) + "\n" for row in answer_rows),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                sage_result = evaluate(str(details_path), str(answers_path), top_k=3)

        mismatches = []
        for item in sage_result["per_example"]:
            expected = adapter_scores[item["example_id"]]
            actual = (item["answer_em"], item["answer_f1"])
            if actual != expected:
                mismatches.append((item["example_id"], expected, actual))
        self.assertEqual(len(adapter_scores), 81_288)
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
