from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "llm_pipeline"))

import stratified_sampling as sampling  # noqa: E402
from benchmark_corrections import add_group_keys, prepare_eligible_rows  # noqa: E402


def source_row(
    task_id: str,
    *,
    root: str,
    subject: str,
    object_: str,
    answer: str,
    complexity: int,
    answer_type: str = "BIN",
    task_type: str = "Membership",
    abox: int = 10,
) -> dict[str, object]:
    query = (
        f"ASK WHERE {{ <urn:{subject}> <urn:type> <urn:{object_}> }}"
        if answer_type == "BIN"
        else f"SELECT ?x WHERE {{ <urn:{subject}> <urn:type> ?x }}"
    )
    return {
        "Task ID": task_id,
        "Root Entity": root,
        "Size of ontology ABox": abox,
        "Task Type": task_type,
        "Answer Type": answer_type,
        "SPARQL Query": query,
        "Answer": answer,
        "Max Tag Length": 999,
        sampling.CORRECTED_COMPLEXITY_COLUMN: complexity,
    }


class FinalComplexityBinTests(unittest.TestCase):
    def test_bqa_final_bins_are_one_two_three_plus(self) -> None:
        self.assertEqual(sampling.final_complexity_bin("BQA", 1), "Low")
        self.assertEqual(sampling.final_complexity_bin("BQA", 2), "Medium")
        self.assertEqual(sampling.final_complexity_bin("BQA", 3), "High")
        self.assertEqual(sampling.final_complexity_bin("BQA", 99), "High")

    def test_oeqa_final_bins_are_one_to_three_four_to_five_six_plus(self) -> None:
        self.assertEqual(sampling.final_complexity_bin("OEQA", 1), "Low")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 3), "Low")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 4), "Medium")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 5), "Medium")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 6), "High")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 251), "High")

    def test_corrected_oeqa_complexity_not_legacy_max_tag_length_is_used(self) -> None:
        rows = pd.DataFrame(
            [
                source_row(
                    "oeqa-a-MC", root="r1", subject="s1", object_="unused",
                    answer="a", complexity=3, answer_type="MC",
                ),
                source_row(
                    "oeqa-b-MC", root="r2", subject="s2", object_="unused",
                    answer="b", complexity=6, answer_type="MC",
                ),
            ]
        )
        groups = sampling.build_corrected_sampling_groups(
            add_group_keys(rows, "Family", "1hop")
        )
        self.assertEqual(list(groups[sampling.COMPLEXITY_BIN_COLUMN]), ["Low", "High"])

    def test_m_metadata_is_not_an_extra_primitive_unit(self) -> None:
        # Phase 4 supplies the already corrected primitive count. The sampler
        # has no M input and therefore cannot increment it.
        self.assertEqual(sampling.final_complexity_bin("BQA", 2), "Medium")
        self.assertEqual(sampling.final_complexity_bin("OEQA", 5), "Medium")


class HierarchicalFallbackTests(unittest.TestCase):
    @staticmethod
    def _groups() -> pd.DataFrame:
        rows = []
        specs = [
            ("n1", "Low", 0, "normal-low"),
            ("n2", "Low", 0, "normal-low"),
            ("n3", "Medium", 1, "normal-medium"),
            ("n4", "Medium", 1, "normal-medium"),
            ("f1", "High", 2, "fallback-a"),
            ("f2", "High", 3, "fallback-b"),
            ("s1", "Low", 9, "singleton"),
        ]
        raw = {"Low": 1, "Medium": 2, "High": 3}
        for key, complexity_bin, abox_bin, combined in specs:
            rows.append(
                {
                    "Sampling Group Key": ("BQA", key),
                    "Size of ontology ABox": abox_bin,
                    sampling.CORRECTED_COMPLEXITY_COLUMN: raw[complexity_bin],
                    sampling.TASK_CLASS_COLUMN: "BQA",
                    sampling.COMPLEXITY_BIN_COLUMN: complexity_bin,
                    "ABox Bin": abox_bin,
                    "Combined Stratum": combined,
                }
            )
        return pd.DataFrame(rows)

    def test_sparse_combined_strata_use_fallback_without_bin_changes(self) -> None:
        groups = self._groups()
        with patch.object(sampling, "build_corrected_sampling_groups", return_value=groups):
            _, decisions = sampling.select_corrected_groups(
                pd.DataFrame(), benchmark_fraction=0.25, random_state=42
            )
        by_key = {row["Sampling Group Key"]: row for _, row in decisions.iterrows()}
        self.assertEqual(by_key[("BQA", "f1")][sampling.SAMPLING_MODE_COLUMN], "complexity-fallback")
        self.assertEqual(by_key[("BQA", "f2")][sampling.SAMPLING_MODE_COLUMN], "complexity-fallback")
        self.assertEqual(by_key[("BQA", "s1")][sampling.SAMPLING_MODE_COLUMN], "deterministic-singleton-fallback")
        self.assertEqual(by_key[("BQA", "f1")][sampling.COMPLEXITY_BIN_COLUMN], "High")
        self.assertEqual(len(decisions), len(groups))

    def test_singleton_assignment_is_deterministic_and_never_dropped_from_ledger(self) -> None:
        groups = pd.DataFrame(
            [
                {
                    "Sampling Group Key": ("OEQA", key),
                    "Size of ontology ABox": index,
                    sampling.CORRECTED_COMPLEXITY_COLUMN: complexity,
                    sampling.TASK_CLASS_COLUMN: "OEQA",
                    sampling.COMPLEXITY_BIN_COLUMN: bin_name,
                    "ABox Bin": index,
                    "Combined Stratum": f"unique-{index}",
                }
                for index, (key, complexity, bin_name) in enumerate(
                    [("a", 1, "Low"), ("b", 4, "Medium"), ("c", 6, "High")]
                )
            ]
        )
        with patch.object(sampling, "build_corrected_sampling_groups", return_value=groups):
            first, first_decisions = sampling.select_corrected_groups(
                pd.DataFrame(), benchmark_fraction=0.25, random_state=42
            )
            second, second_decisions = sampling.select_corrected_groups(
                pd.DataFrame(), benchmark_fraction=0.25, random_state=42
            )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(first_decisions), 3)
        pd.testing.assert_frame_equal(first_decisions, second_decisions)
        self.assertTrue(
            first_decisions[sampling.SAMPLING_MODE_COLUMN]
            .eq("deterministic-singleton-fallback")
            .all()
        )


class PairAndDomainPolicyTests(unittest.TestCase):
    def test_pair_is_an_indivisible_sampling_group(self) -> None:
        rows = pd.DataFrame(
            [
                source_row(
                    "positive", root="r", subject="s", object_="Pizza",
                    answer="TRUE", complexity=2,
                ),
                source_row(
                    "negative", root="r", subject="s", object_="American",
                    answer="FALSE", complexity=2,
                ),
            ]
        )
        keyed = add_group_keys(rows, "Pizza100", "1hop")
        self.assertEqual(keyed["Sampling Group Key"].nunique(), 1)
        groups = sampling.build_corrected_sampling_groups(keyed)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups.iloc[0][sampling.COMPLEXITY_BIN_COLUMN], "Medium")

    def test_pizza_domainconcept_target_is_removed_but_oeqa_is_retained(self) -> None:
        rows = pd.DataFrame(
            [
                source_row(
                    "domain", root="r", subject="s", object_="DomainConcept",
                    answer="TRUE", complexity=1,
                ),
                source_row(
                    "ordinary", root="r", subject="s", object_="Pizza",
                    answer="TRUE", complexity=1,
                ),
                source_row(
                    "negative", root="r", subject="s", object_="American",
                    answer="FALSE", complexity=1,
                ),
                source_row(
                    "oeqa-MC", root="r", subject="s", object_="unused",
                    answer="Pizza; DomainConcept", complexity=2, answer_type="MC",
                ),
            ]
        )
        eligible, _ = prepare_eligible_rows(rows, "Pizza100", "1hop")
        self.assertNotIn("<urn:DomainConcept>", " ".join(eligible["SPARQL Query"]))
        self.assertIn("Pizza; DomainConcept", set(eligible["Answer"]))


if __name__ == "__main__":
    unittest.main()
