---
pretty_name: CORE-LLM-Bench
license: other
task_categories:
  - question-answering
language:
  - en
tags:
  - ontology
  - reasoning
  - neurosymbolic
  - benchmark
---

# CORE-LLM-Bench v1.0

## Summary

CORE-LLM-Bench evaluates verifiable ontology reasoning across Family, Pizza 100, Pizza 250, and OWL2Bench. It contains 9,032 unique question-hop instances. Every row keeps the NL, formal-symbolic (FS), and abstract-representation (AR) views together.

## Tasks and statistics

- BQA: 5,999 binary questions with `TRUE` or `FALSE` labels.
- OEQA: 3,033 open-ended questions with set-valued gold answers.
- Dataset totals: Family 3,760; Pizza 100 984; Pizza 250 1,232; OWL2Bench 3,056.
- Both 1-hop and 2-hop ontology-context variants are included.

## Fields

Identity and labels: `task_id`, `dataset`, `hop`, `task_type`, `reasoning_task`, `answer_type`, `binary_label`, `gold_answer`, `ar_gold_answer`, and `root_entity`.

Representations: `nl_question`, `nl_context`, `fs_query`, `fs_context`, `ar_question`, and `ar_context`.

Explanations and provenance: `minimum_explanation`, `explanations`, `explanation_count`, `min_tag_length`, `max_tag_length`, `reasoning_tags`, `linked_positive_task_id`, `benchmark_version`, `source_package`, and `source_member`.

## Generation and validation

Questions were generated from individual-centered ontology subgraphs. Pellet supplies gold entailments and explanation metadata. Run `python scripts/validate_release.py` in the source repository to verify package hashes, schema, identities, labels, representations, complexity, and reasoning coverage without API keys.

## Intended uses

Use the benchmark to compare models under NL, FS, and AR inputs, or to study performance by ontology-context depth and reasoner-derived explanation complexity. Keep all representations for a `task_id` in the same evaluation split or analysis unit.

## Limitations

The benchmark is English-centered, covers four ontology families, and instantiates eight of the 20 reasoning tags. Context depth is not the same as minimum proof complexity. Generated natural language may contain artifacts. The Family ontology redistribution license still requires clearance; consult `NOTICE.md` before publication or redistribution.

## Licensing and citation

This is a mixed-provenance dataset. Repository code is MIT; Pizza source material declares CC BY 3.0; OWL2Bench is Apache-2.0; Family redistribution terms remain unresolved. See `NOTICE.md` and `CITATION.cff`. Do not replace these notices with a single blanket license.
