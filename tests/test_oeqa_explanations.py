import unittest

from scripts.oeqa_explanations import build_answer_explanations


def record(answer, *alternatives):
    return {"Source Answer": answer, "Explanations": list(alternatives)}


class AnswerGroupedExplanationTests(unittest.TestCase):
    def test_one_answer_one_proof(self):
        groups, complete = build_answer_explanations(
            "A", [record("A", ["a fact", "TAG:D"])]
        )
        self.assertEqual(groups[0]["Alternative Count"], 1)
        self.assertEqual(
            groups[0]["Source Provenance"], {"inferred.object": "A"}
        )
        self.assertEqual(complete["Complete Explanation Min Axiom Count"], 1)

    def test_one_answer_multiple_alternatives(self):
        groups, complete = build_answer_explanations(
            "A", [record("A", ["a fact", "TAG:D"], ["a fact", "A SubClassOf B", "TAG:DH"])]
        )
        self.assertEqual(groups[0]["Alternative Count"], 2)
        self.assertEqual(complete["Complete Explanation Max Axiom Count"], 2)

    def test_multiple_answers_are_conjunctive(self):
        groups, complete = build_answer_explanations(
            "A; B",
            [record("A", ["proof A", "TAG:D"]), record("B", ["proof B", "TAG:D"])],
        )
        self.assertEqual([group["Answer"] for group in groups], ["A", "B"])
        self.assertEqual(complete["Complete Explanation Min Axiom Count"], 2)

    def test_multiple_answers_multiple_alternatives(self):
        _, complete = build_answer_explanations(
            "A; B",
            [
                record("A", ["a1", "TAG:D"], ["a2", "TAG:D"]),
                record("B", ["b1", "TAG:D"], ["b2", "TAG:D"]),
            ],
        )
        self.assertEqual(complete["Complete Explanation Combination Count"], 4)

    def test_shared_axiom_is_counted_once_in_complete_union(self):
        _, complete = build_answer_explanations(
            "A; B",
            [
                record("A", ["fact A", "p Domain C", "TAG:DR"]),
                record("B", ["fact B", "p Domain C", "TAG:DR"]),
            ],
        )
        self.assertEqual(complete["Complete Explanation Min Axiom Count"], 3)

    def test_complete_union_m_semantics(self):
        _, complete = build_answer_explanations(
            "A; B",
            [
                record("A", ["x rdf:type A", "A SubClassOf B", "TAG:DH"]),
                record("B", ["x p y", "p Range B", "TAG:DR"]),
            ],
        )
        self.assertEqual(complete["Complete Explanation M Status"], "always")
        self.assertGreaterEqual(
            complete[
                "Complete Explanation Min Distinct Non-Direct Tag Type Count"
            ],
            2,
        )

    def test_professor_collaboration_example_preserves_symmetry_proof(self):
        groups, complete = build_answer_explanations(
            "U0C0D0FP0; U0C2D3UGS28",
            [
                record("U0C0D0FP0", ["U0C0D1UGS36 hasCollaborationWith U0C0D0FP0", "TAG:D"]),
                record(
                    "U0C2D3UGS28",
                    [
                        "U0C2D3UGS28 hasCollaborationWith U0C0D1UGS36",
                        "SymmetricObjectProperty(hasCollaborationWith)",
                        "TAG:DS",
                    ],
                ),
            ],
        )
        self.assertEqual(len(groups), 2)
        self.assertIn("TAG:DS", groups[1]["Alternatives"][0])
        self.assertEqual(complete["Complete Explanation Min Axiom Count"], 3)

    def test_rejects_non_gold_source_and_missing_gold(self):
        with self.assertRaises(ValueError):
            build_answer_explanations("A", [record("B", ["proof", "TAG:D"])])
        with self.assertRaises(ValueError):
            build_answer_explanations("A; B", [record("A", ["proof", "TAG:D"])])


if __name__ == "__main__":
    unittest.main()
