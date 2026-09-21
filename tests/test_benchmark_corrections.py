from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "llm_pipeline"))

from benchmark_corrections import (  # noqa: E402
    add_group_keys,
    assert_no_broken_source_pairs,
    assert_owl2bench_membership_pairs,
    audit_bqa_pairing,
    prepare_eligible_rows,
    sampling_group_key,
)
from stratified_sampling import (  # noqa: E402
    build_sampling_groups,
    limit_rows_by_group,
    select_train_groups,
)


def row(
    task_id: str,
    *,
    root: str,
    subject: str,
    predicate: str = "type",
    object_: str,
    answer: str,
    answer_type: str = "BIN",
    task_type: str = "Membership",
    abox_size: int = 10,
    max_tag_length: int = 2,
) -> dict[str, object]:
    query_type = "ASK" if answer_type == "BIN" else "SELECT"
    query = (
        f"ASK WHERE {{ <urn:{subject}> <urn:{predicate}> <urn:{object_}> }}"
        if query_type == "ASK"
        else f"SELECT ?x WHERE {{ <urn:{subject}> <urn:{predicate}> ?x }}"
    )
    return {
        "Task ID": task_id,
        "Root Entity": root,
        "Size of ontology TBox": 5,
        "Size of ontology ABox": abox_size,
        "Task Type": task_type,
        "Answer Type": answer_type,
        "SPARQL Query": query,
        "Predicate": predicate,
        "Answer": answer,
        "Min Tag Length": 1,
        "Max Tag Length": max_tag_length,
    }


class StructuralPairingTests(unittest.TestCase):
    def test_bqa_key_ignores_task_id_and_query_object(self) -> None:
        positive = row(
            "arbitrary-positive-id",
            root="Root_A",
            subject="subject-a",
            object_="PositiveClass",
            answer="TRUE",
        )
        negative = row(
            "unrelated-negative-id",
            root="Root_A",
            subject="subject-a",
            object_="NegativeClass",
            answer="FALSE",
        )
        self.assertEqual(
            sampling_group_key(positive, "OWL2Bench", "1hop"),
            sampling_group_key(negative, "OWL2Bench", "1hop"),
        )

    def test_sampling_keeps_structural_pairs_with_unrelated_ids(self) -> None:
        rows = []
        for index in range(8):
            rows.extend(
                [
                    row(
                        f"positive-{index}",
                        root=f"Root_{index}",
                        subject=f"subject-{index}",
                        object_="Publication",
                        answer="TRUE",
                    ),
                    row(
                        f"totally-different-{index}",
                        root=f"Root_{index}",
                        subject=f"subject-{index}",
                        object_="Course",
                        answer="FALSE",
                    ),
                ]
            )

        source = pd.DataFrame(rows)
        keyed = add_group_keys(source, "OWL2Bench", "1hop")
        selected_groups = select_train_groups(keyed, test_size=0.5, random_state=42)
        selected = keyed[keyed["Sampling Group Key"].isin(selected_groups)].drop(
            columns=["Sampling Group Key"]
        )

        grouped = selected.groupby(["Root Entity", "Predicate"])["Answer"].apply(set)
        self.assertGreater(len(grouped), 0)
        self.assertTrue(all(answers == {"TRUE", "FALSE"} for answers in grouped))
        report = audit_bqa_pairing(
            pd.DataFrame(rows), selected, "OWL2Bench", "1hop"
        )
        assert_no_broken_source_pairs(report)
        assert_owl2bench_membership_pairs(report)

    def test_pair_stratum_uses_maximum_abox_and_complexity(self) -> None:
        rows = [
            row(
                "positive",
                root="Root",
                subject="subject",
                object_="Positive",
                answer="TRUE",
                abox_size=10,
                max_tag_length=2,
            ),
            row(
                "negative",
                root="Root",
                subject="subject",
                object_="Negative",
                answer="FALSE",
                abox_size=25,
                max_tag_length=5,
            ),
        ]
        groups = build_sampling_groups(
            add_group_keys(pd.DataFrame(rows), "OWL2Bench", "1hop")
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups.iloc[0]["Size of ontology ABox"], 25)
        self.assertEqual(groups.iloc[0]["Max Tag Length"], 5)

    def test_row_cap_keeps_whole_structural_groups(self) -> None:
        rows = []
        for index in range(3):
            rows.extend(
                [
                    row(
                        f"positive-{index}",
                        root=f"Root_{index}",
                        subject=f"subject-{index}",
                        object_="Positive",
                        answer="TRUE",
                    ),
                    row(
                        f"negative-{index}",
                        root=f"Root_{index}",
                        subject=f"subject-{index}",
                        object_="Negative",
                        answer="FALSE",
                    ),
                ]
            )
        keyed = add_group_keys(pd.DataFrame(rows), "OWL2Bench", "1hop")
        limited = limit_rows_by_group(keyed, max_rows=4, random_state=42)
        grouped = limited.groupby("Root Entity")["Answer"].apply(set)
        self.assertEqual(len(limited), 4)
        self.assertTrue(all(answers == {"TRUE", "FALSE"} for answers in grouped))

    def test_validator_distinguishes_missing_source_from_broken_sample(self) -> None:
        pair = [
            row(
                "p",
                root="Root_pair",
                subject="paired",
                object_="Positive",
                answer="TRUE",
            ),
            row(
                "n",
                root="Root_pair",
                subject="paired",
                object_="Negative",
                answer="FALSE",
            ),
        ]
        source_missing = row(
            "source-only-positive",
            root="Root_missing",
            subject="missing",
            object_="Positive",
            answer="TRUE",
        )
        source = pd.DataFrame([*pair, source_missing])

        valid_report = audit_bqa_pairing(
            source, pd.DataFrame([source_missing]), "Pizza100", "1hop"
        )
        self.assertEqual(valid_report[0].missing_source_pair_positives, 1)
        self.assertEqual(valid_report[0].broken_sample_positives, 0)
        assert_no_broken_source_pairs(valid_report)

        broken_report = audit_bqa_pairing(
            source, pd.DataFrame([pair[0]]), "Pizza100", "1hop"
        )
        self.assertEqual(broken_report[0].broken_sample_positives, 1)
        with self.assertRaisesRegex(ValueError, "broke valid source BQA pairs"):
            assert_no_broken_source_pairs(broken_report)


class DomainConceptFilterTests(unittest.TestCase):
    def test_domainconcept_positive_is_replaced_and_pair_is_preserved(self) -> None:
        for dataset in ("Pizza100", "Pizza250"):
            for hop in ("1hop", "2hop"):
                with self.subTest(dataset=dataset, hop=hop):
                    domain_bqa_true = row(
                        f"{hop}-domain-bqa-true",
                        root="Root_pizza",
                        subject="pizza",
                        object_="DomainConcept",
                        answer="TRUE",
                    )
                    paired_false = row(
                        f"{hop}-paired-false",
                        root="Root_pizza",
                        subject="pizza",
                        object_="American",
                        answer="FALSE",
                    )
                    ordinary_bqa = row(
                        f"{hop}-ordinary-bqa",
                        root="Root_pizza",
                        subject="pizza",
                        object_="Pizza",
                        answer="TRUE",
                    )
                    domain_oeqa = row(
                        f"{hop}-domain-oeqa-MC",
                        root="Root_pizza",
                        subject="pizza",
                        object_="unused",
                        answer="Pizza; DomainConcept; NamedPizza",
                        answer_type="MC",
                    )
                    eligible, report = prepare_eligible_rows(
                        pd.DataFrame(
                            [
                                domain_bqa_true,
                                paired_false,
                                ordinary_bqa,
                                domain_oeqa,
                            ]
                        ),
                        dataset,
                        hop,
                    )
                    self.assertEqual(
                        set(eligible["Task ID"]),
                        {
                            paired_false["Task ID"],
                            ordinary_bqa["Task ID"],
                            domain_oeqa["Task ID"],
                        },
                    )
                    self.assertEqual(report.domainconcept_bqa_targets_removed, 1)
                    self.assertEqual(report.domainconcept_positives_replaced, 1)
                    self.assertEqual(report.domainconcept_only_groups_omitted, 0)
                    self.assertEqual(report.oeqa_domainconcept_answers_retained, 1)

                    answers = set(
                        eligible[eligible["Answer Type"] == "BIN"]["Answer"]
                    )
                    self.assertEqual(answers, {"TRUE", "FALSE"})

    def test_domainconcept_only_group_is_omitted_without_orphan_negative(self) -> None:
        domain_true = row(
            "domain-only-true",
            root="Root_domain_only",
            subject="domain-only",
            object_="DomainConcept",
            answer="TRUE",
        )
        paired_false = row(
            "domain-only-false",
            root="Root_domain_only",
            subject="domain-only",
            object_="American",
            answer="FALSE",
        )
        domain_oeqa = row(
            "domain-only-MC",
            root="Root_domain_only",
            subject="domain-only",
            object_="unused",
            answer="DomainConcept",
            answer_type="MC",
        )

        eligible, report = prepare_eligible_rows(
            pd.DataFrame([domain_true, paired_false, domain_oeqa]),
            "Pizza100",
            "1hop",
        )

        self.assertEqual(list(eligible["Task ID"]), [domain_oeqa["Task ID"]])
        self.assertEqual(report.domainconcept_positives_replaced, 0)
        self.assertEqual(report.domainconcept_only_groups_omitted, 1)
        self.assertEqual(report.oeqa_domainconcept_answers_retained, 1)
        pairing = audit_bqa_pairing(
            eligible, eligible, "Pizza100", "1hop"
        )
        self.assertTrue(all(item.unpaired_negatives == 0 for item in pairing))

    def test_no_pizza_membership_bqa_targets_domainconcept(self) -> None:
        rows = [
            row(
                "domain-true",
                root="Root",
                subject="subject",
                object_="DomainConcept",
                answer="TRUE",
            ),
            row(
                "ordinary-true",
                root="Root",
                subject="subject",
                object_="Pizza",
                answer="TRUE",
            ),
            row(
                "domain-false",
                root="Root",
                subject="subject",
                object_="DomainConcept",
                answer="FALSE",
            ),
            row(
                "ordinary-false",
                root="Root",
                subject="subject",
                object_="American",
                answer="FALSE",
            ),
        ]

        eligible, _ = prepare_eligible_rows(
            pd.DataFrame(rows), "Pizza250", "2hop"
        )
        self.assertNotIn("DomainConcept", " ".join(eligible["SPARQL Query"]))

    def test_oeqa_with_domainconcept_is_unchanged(self) -> None:
        domain_oeqa = row(
            "domain-MC",
            root="Root",
            subject="subject",
            object_="unused",
            answer="Pizza; DomainConcept; NamedPizza",
            answer_type="MC",
        )
        source = pd.DataFrame([domain_oeqa])
        eligible, report = prepare_eligible_rows(source, "Pizza100", "1hop")

        pd.testing.assert_frame_equal(eligible.reset_index(drop=True), source)
        self.assertEqual(report.oeqa_domainconcept_answers_retained, 1)
        self.assertEqual(report.domainconcept_positives_replaced, 1)

    def test_audit_recovers_replacement_from_post_selection_source(self) -> None:
        selected_true = row(
            "selected-true",
            root="Root",
            subject="subject",
            object_="Pizza",
            answer="TRUE",
        )
        paired_false = row(
            "paired-false",
            root="Root",
            subject="subject",
            object_="American",
            answer="FALSE",
        )
        domain_oeqa = row(
            "domain-MC",
            root="Root",
            subject="subject",
            object_="unused",
            answer="DomainConcept; Pizza",
            answer_type="MC",
        )

        eligible, report = prepare_eligible_rows(
            pd.DataFrame([selected_true, paired_false, domain_oeqa]),
            "Pizza250",
            "2hop",
        )

        self.assertEqual(len(eligible), 3)
        self.assertEqual(report.domainconcept_bqa_targets_removed, 0)
        self.assertEqual(report.domainconcept_positives_replaced, 1)
        self.assertEqual(report.domainconcept_only_groups_omitted, 0)

    def test_non_pizza_domainconcept_bqa_is_not_filtered(self) -> None:
        domain_bqa = row(
            "owl-domain",
            root="Root",
            subject="subject",
            object_="DomainConcept",
            answer="FALSE",
        )
        eligible, report = prepare_eligible_rows(
            pd.DataFrame([domain_bqa]), "OWL2Bench", "1hop"
        )
        self.assertEqual(len(eligible), 1)
        self.assertEqual(report.domainconcept_bqa_targets_removed, 0)


if __name__ == "__main__":
    unittest.main()
