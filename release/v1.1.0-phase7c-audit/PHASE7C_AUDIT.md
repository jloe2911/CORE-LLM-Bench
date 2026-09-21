# Phase 7C collision-safe NL and input-uniqueness audit

Status: complete and stopped for scientific review. Phase 7C is uncommitted,
execution-disabled, and separate from committed Phase 6 staging.

## 1. Files changed

- `.gitignore`: exposes the Phase 7C regression test to Git.
- `scripts/phase7c_collision_audit.py`: offline renderer, staged rebuild, duplicate
  audit, equivalence metadata, and proposed configuration generator.
- `tests/test_phase7c_collision_audit.py`: seven focused regression tests.
- This directory: one corrected audit Parquet plus mapping, hash, equivalence,
  conflict-outcome, report, and proposed-configuration artifacts.

## 2. Collision-safe algorithm

The mapping scope is each generated dataset (`Family`, `OWL2Bench`, `Pizza100`,
or `Pizza250`) across both hops and every ontology root selected by the immutable
Phase 5 membership. Formal-query URIs are included even when a false BQA object
does not occur as a triple in its root graph.

For every source URI, the renderer obtains the existing normalized candidate
label using the prior lexical-property priority and local-name fallback. Singleton
candidates are unchanged. For a candidate shared by distinct URIs, the renderer
uses a suffix-preserving, human-readable local-name surface (`Employee_10` becomes
`Employee 10`). If that surface would still collide, it appends ontology-grounded
local identity; full URI identity is the final fail-closed fallback. The complete
mapping is checked for injectivity. No random ID or answer-derived value is used.

The final mapping contains 6,659 semantic entities; 2,921 require disambiguation. Existing
questions are changed only where a formal query URI has a disambiguated surface.
Contexts are regenerated from URI-keyed relations with the dataset mapping.

## 3-8. Row and duplicate outcomes

- NL evaluated inputs changed: **9,048 / 9,048** rows.
- NL questions changed: **6,212**; NL contexts changed: **9,048**.
- NL incompatible groups: **9 groups / 19 rows before**, **0 / 0 after**.
- FS incompatible groups: **0 / 0 before and after**.
- AR incompatible groups: **0 / 0 before and after**.
- All four fatal BQA pairs now have two distinct evaluated NL inputs; TRUE/FALSE
  gold labels are unchanged and no answer label enters the rendering.
- All three Family fatal OEQA groups resolve (2, 3, and 2 distinct post-fix inputs).
- Both Pizza100/Pizza250 fatal OEQA groups resolve to distinct inputs.
- Compatible duplicates retained after correction: NL **12 groups / 24 rows**,
  FS **5 / 10**, AR **7 / 14**; total **24 groups / 48 rows**.

## 9. Input-equivalence metadata

`input_equivalence_groups.csv` records every compatible exact-input group with a
stable ID `ieq-{representation}-{first 24 hex characters of input hash}`, its
representation, semantic-row count, dataset/hop distribution, identical normalized
gold semantics, and cause/classification. NL contains three compatible
representational-collapse groups and ten incidental cross-dataset/hop groups;
FS has five incidental groups and AR has seven. Sampling is unchanged.

## 10-11. Membership impact

Semantic membership remains **9,048** rows: 6,032 BQA rows in 3,016 intact pairs
and 3,016 OEQA rows. No row or group is proposed for removal.

## 12. Proposed v1.1 model configuration

The complete machine-readable proposal is `proposed_experiment_config_v1_1.json`.
All models use the existing user-only prompt template, no system message, 1,024
maximum output tokens, a 30-second timeout, and technical-only retries (three after
the initial attempt, with 2/4/8 second backoff). Nonempty malformed answers are
terminal observations and are not resampled.

- GPT-5 mini: `gpt-5-mini-2025-08-07`, direct OpenAI Chat Completions,
  `reasoning_effort=low`, `verbosity=low`, presence/frequency penalty `0.0`, seed
  `0` (best effort). Temperature and top-p remain explicitly omitted because the
  retained run omitted them and model-specific control with low reasoning was not
  established. The official catalog now marks this historical snapshot deprecated,
  so model identity is review-blocking before execution.
- Gemini 2.5 Flash-Lite: `google/gemini-2.5-flash-lite`, temperature `0.0`, top-p
  `0.9`, seed `0`, reasoning disabled, no unsupported penalties, pinned proposed
  OpenRouter route `google-ai-studio`, `allow_fallbacks=false`, and
  `require_parameters=true`.
- Qwen3-30B-A3B-Instruct: `qwen/qwen3-30b-a3b-instruct-2507`, non-thinking
  instruct model, temperature `0.0`, top-p `0.9`, seed `0`, presence penalty `0.0`,
  frequency penalty `0.1`, pinned proposed OpenRouter route `dekallm`,
  `allow_fallbacks=false`, and `require_parameters=true`.

OpenRouter endpoint availability and parameter support must be rechecked without
inference immediately before execution. The proposal remains `ready_for_execution:
false` and `execution_authorized: false`.

## 13. Complete fresh experiment matrix

The primary experiment is a complete fresh rerun: **9,048 rows x 3
representations x 3 models = 81,432 requests**. Historical v1.0 responses remain
provenance/sensitivity evidence only and are not mixed into primary v1.1 results.

## 14. Tests

- Focused Phase 7C: 10 passed.
- Complete tracked Python suite: 80 passed and 4 subtests passed; one additional test hit a
  Windows scratch-directory ACL error before setup, then passed alone with a fresh
  repository-local `target` scratch root (all 81 tests validated).
- Maven: 9 passed (5 tagger, 4 processor), using a repository-local Maven cache
  after the default and OS-temp caches were inaccessible.
- Ruff 0.15.1: passed for Phase 7C source and tests.
- Python byte compilation: passed.
- `git diff --check`: passed.

## 15-16. Safety boundary

The Phase 7C implementation made **zero model inference/provider-generation API
calls** and executed no experiment. Read-only current provider documentation was
consulted to avoid inventing configuration support. Published v1.0.0 data and the
committed Phase 6 staging artifacts remain untouched. No commit, push, merge, tag,
release, or publication action occurred.
