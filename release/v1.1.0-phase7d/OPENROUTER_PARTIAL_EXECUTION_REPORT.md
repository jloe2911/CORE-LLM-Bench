# OpenRouter partial execution report

Status: Gemini complete; Qwen stopped after failed canary; GPT untouched and
pending. This is an execution-diagnostics report, not a final benchmark or
paper-results report.

## Frozen preflight

- Membership: 9,048 tasks.
- Representations: NL, FS, AR.
- Pending observations before execution: 27,144 per model; 81,432 total.
- Configuration hash:
  `c1562554bd9e252bf97356ef098edde997f813536dd8e547f35f5981a1023df3`.
- Incompatible input/gold groups: 0.
- Fallback: disabled for every model.
- Authorized execution allowlist: `google/gemini-2.5-flash-lite` and
  `qwen/qwen3-30b-a3b-instruct-2507` only.
- GPT was rejected locally by the runner allowlist before credential access or
  networking.

## Gemini

- Canary: passed, 10/10 unique first-pending rows.
- Requested/returned model: `google/gemini-2.5-flash-lite`.
- Returned backend: Google AI Studio for 27,144/27,144 observations.
- Completed: 27,144; unresolved technical failures: 0; pending: 0.
- Schema-usable: 27,139; malformed-but-retained: 5.
- Technical retries: 0.
- Duplicate, missing, unexpected, input-hash, configuration-hash, returned-model,
  and backend mismatches: 0.
- Rows with nonzero reported reasoning tokens: 0.
- Execution interval: 2026-09-21T09:50:09.722376+00:00 through
  2026-09-21T10:54:49.608115+00:00.

Malformed-but-retained Gemini observations:

- task 1627, AR,
  `sem-v1.1-73e51bc8d7b57df19005e2cdfe835800ec20ace987a72645ac498c561d62390d`
- task 1730, NL,
  `sem-v1.1-de2392833bbd2376bcdb365e9371b95e46f9e354732080f1fdb3f26de4dfcd62`
- task 1849, NL,
  `sem-v1.1-148ae4244b9bac67fdeebe5570be25b9686abbd6c4bf9e0950d44422a3003b84`
- task 3211, AR,
  `sem-v1.1-1ad3a53185e4b5b7fb4d492aec6488056e46bdd79556cd1d0b196a5b0ab5c300`
- task 3430, AR,
  `sem-v1.1-06af47cdfdee7e25f87c4d04f974e05d5b626df6ee34db01e88913b77386a72b`

Each was accepted by the existing parser with the frozen default-confidence
behavior, retained verbatim, and not regenerated.

## Qwen

- Canary: failed; full execution was not started.
- Canary rows: 10 unique first-pending rows.
- Usable observations: 3; unresolved technical failures: 7; unattempted/pending:
  27,134.
- Successful responses returned exact model
  `qwen/qwen3-30b-a3b-instruct-2507` and backend DekaLLM.
- All seven terminal failures were DekaLLM upstream shared-pool HTTP 429 rate
  limits.
- Technical retries: 23; provider requests: 33; duplicates: 0; malformed: 0.
- The full Qwen run was stopped after the bounded canary, without exceeding the
  frozen per-cell retry policy.

## Totals and safety

- OpenRouter experiment requests: 27,177 (27,144 Gemini; 33 Qwen).
- Accepted observations preserved: 27,147 (27,144 Gemini; 3 Qwen).
- Expected accepted observations after both models complete: 54,288. This target
  was not reached because Qwen failed its canary.
- GPT observations: 0.
- OpenAI provider requests during this execution: 0.
- `OPENAI_API_KEY` was not loaded or used by an execution process.
- No final paper metrics, cross-model results, or conclusions were computed.
- No release was published; published v1.0.0 was untouched.

Frozen-artifact hashes verified against `SHA256SUMS` after execution:

- `core_llm_bench_v1_1.parquet`:
  `7285e506483b422acf8e1882da5b5390966ba4896b4bb020820f6bb4a5218859`
- `primary_experiment_manifest.csv`:
  `732c0a6306f970dcb7ec4aa36903fc5739f2b5f941f6ce14096fcca1049b0bbc`
- `experiment_config_v1_1.json`:
  `47a60d2536dbfc718658fe91c12599a89d5fda66a03828a6eb20cff24d7a8da7`
