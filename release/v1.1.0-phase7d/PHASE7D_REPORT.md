# Phase 7D endpoint/configuration validation and experiment freeze

Status: final staging complete; primary experiment not started. Execution is
blocked until the exact GPT snapshot can be revalidated after OpenAI billing is
restored. No substitution was made.

## 1. Accepted Phase 7C commit

- Commit: `44f7c9dc76d1ee664969d37db82ebf8e7e624454`
- Message: `Make NL verbalization collision-safe`

## 2. Final materialization hashes

- `core_llm_bench_v1_1.parquet`: `7285e506483b422acf8e1882da5b5390966ba4896b4bb020820f6bb4a5218859`
- `primary_experiment_manifest.csv`: `22a113624d62e3b1689ec3d083522d4c3e64e4c0d504bff12370d8060a3c49b2`
- `experiment_config_v1_1.json`: `d1a2ce2703e8b7fceb224fa79d004bb00d92663a56edda5ba13c7dd0d8050000`
- Internal configuration hash: `40e56b6e2478b0c8dcacd6eea30848299bd5cc882fa5c29c06d0a9f0e8aec3d6`
- `RELEASE_MANIFEST.json`: `774310997890e3654617fe952147a1bf2baca7470cc9b40a44c551ef9e82286a`
- `SHA256SUMS`: `31735d0f2f7369e15990a23e3b56aa84001732315974dc5c671279c6fc3db1a2`

An independent rebuild produced 26 files with zero path differences and zero
SHA-256 differences.

## 3. Duplicate audit

- NL incompatible groups/rows: 0 / 0; compatible groups/rows: 12 / 24.
- FS incompatible groups/rows: 0 / 0; compatible groups/rows: 5 / 10.
- AR incompatible groups/rows: 0 / 0; compatible groups/rows: 7 / 14.
- The 24 accepted compatible groups are retained in
  `input_equivalence_groups.csv`.

## 4. Endpoint availability and provider pinning

- GPT-5 mini dated snapshot: `gpt-5-mini-2025-08-07` is still retrievable from
  the OpenAI Models API, but its sole diagnostic request was rejected with
  `credit_balance_exhausted`. Actual executability is therefore not established.
  The `gpt-5-mini` alias is also retrievable but is metadata-only and was not
  tested or substituted. The dated snapshot remains marked deprecated in the
  official catalog.
- Gemini: `google/gemini-2.5-flash-lite` succeeded, returned the exact model ID,
  and returned provider `Google AI Studio`. Route pinned to
  `google-ai-studio`; fallback disabled; parameter support required.
- Qwen: `qwen/qwen3-30b-a3b-instruct-2507` succeeded, returned the exact model
  ID, and returned provider `DekaLLM`. Route pinned to `dekallm`; fallback
  disabled; parameter support required. Endpoint metadata identifies the model
  as non-thinking.

## 5. Frozen parameters

- GPT: requested `reasoning_effort=low`, `verbosity=low`, and
  `max_completion_tokens=1024`; none could be execution-validated because the
  request was rejected for account quota. Temperature, top-p, seed, and
  presence/frequency penalties are intentionally omitted because they were not
  historically specified or verified for this run.
- Gemini: accepted `reasoning.enabled=false`, `temperature=0.0`, `top_p=0.9`,
  `max_tokens=1024`, and `seed=0`. The response reported zero reasoning tokens.
  Presence/frequency penalties are unsupported by the pinned endpoint and are
  omitted.
- Qwen: accepted `temperature=0.0`, `top_p=0.9`, `max_tokens=1024`, `seed=0`,
  `presence_penalty=0.0`, and `frequency_penalty=0.1`. The response reported
  zero reasoning tokens.
- All models use the frozen user-only prompt, no system message, 30-second
  timeout, and at most three technical retries with 2/4/8 second backoff.

The machine-readable per-parameter requested/accepted/effective/guaranteed
record is in `release/v1.1.0-staging/experiment_config_v1_1.json`.

## 6. Primary matrix and cost estimate

- Membership: 9,048.
- Input cells: 27,144 (9,048 x 3 representations).
- Model observations: 81,432 (9,048 x 3 x 3), all `pending`.
- Historical v1.0 responses incorporated: 0.
- Calls per model: 27,144.
- Estimated input tokens per model: 63,773,766.
- Maximum output-token exposure per model: 27,795,456.
- Maximum estimated cost: GPT $71.534354; Gemini $17.495559; Qwen $14.078276;
  total $103.108188.

Input tokens are an explicit heuristic estimate,
`ceil(rendered Unicode characters / 4)`, because exact provider tokenizers were
not available offline. Costs use current listed input/output rates and the full
1,024-token maximum for every response; they are conservative exposure estimates,
not expected realized billing.

## 7. Validation and safety

- Python: 96 passed.
- Maven: 9 passed, 0 failures/errors/skips.
- Ruff: passed.
- Python byte compilation: passed.
- `git diff --check`: passed.
- Deterministic rebuild: 26/26 files byte-identical.
- Diagnostic provider requests: exactly 3 (one per candidate endpoint); two
  returned neutral inference and GPT was rejected for quota. One earlier local
  SDK serialization failure occurred before any provider request and is recorded
  separately.
- Benchmark experiment calls: 0.
- Release/tag/push/merge/publication actions: 0.
- Published v1.0.0 modified files: 0; tag target remains
  `ee766caa48bb23905956a14b1ca5836bcfe19e6d`.

## 8. Stop condition

The configuration is deliberately `ready_for_execution=false` and
`execution_authorized=false`. Do not execute the 81,432-row matrix until the GPT
blocker is resolved, the resulting configuration is reviewed/frozen, and a
separate execution authorization is given.
