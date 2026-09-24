from __future__ import annotations

import unittest

from scripts.explanation_schema import (
    answer_group,
    complete_explanation,
    strip_private_semantic_identities,
)


def proof(*axioms: tuple[str, str]) -> dict:
    return {
        "semanticAxioms": [
            {"identity": identity, "primitiveTags": [tag]}
            for identity, tag in axioms
        ]
    }


class ExplanationSchemaTests(unittest.TestCase):
    @staticmethod
    def render(identity: str) -> str:
        return f"Manchester({identity})"

    def test_exact_union_enumeration_exposes_every_unique_tied_minimum(self) -> None:
        groups = [
            answer_group(
                "A",
                [proof(("shared", "D"), ("a1", "H")), proof(("shared", "D"), ("a2", "D"))],
                self.render,
            ),
            answer_group(
                "B",
                [proof(("shared", "D"), ("b1", "R")), proof(("b2", "I"), ("b3", "D"))],
                self.render,
            ),
        ]
        complete = complete_explanation(groups)
        self.assertEqual(complete["combination_count"], 4)
        self.assertEqual(complete["min_axiom_count"], 3)
        self.assertEqual(complete["max_axiom_count"], 4)
        self.assertEqual(len(complete["minimum_explanations"]), 2)

    def test_semantic_identity_deduplicates_axioms_and_alternatives(self) -> None:
        group = answer_group(
            "A",
            [proof(("id-1", "D"), ("id-1", "D")), proof(("id-1", "D"))],
            self.render,
        )
        self.assertEqual(group["alternative_count"], 1)
        self.assertEqual(group["alternatives"][0]["axiom_count"], 1)

    def test_only_public_structural_tags_are_accepted(self) -> None:
        for tag in ("M", "P", "A"):
            with self.assertRaisesRegex(ValueError, "Unsupported public primitive tag"):
                answer_group("A", [proof(("id", tag))], self.render)

    def test_tag_sequence_is_stored_axiom_order(self) -> None:
        group = answer_group("A", [proof(("z", "N"), ("a", "D"))], self.render)
        alternative = strip_private_semantic_identities(group)["alternatives"][0]
        self.assertEqual(alternative["tag_sequence"], "ND")
        self.assertNotIn("_semantic_identities", alternative)


if __name__ == "__main__":
    unittest.main()
