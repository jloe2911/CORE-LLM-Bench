# CORE-LLM-Bench v1.1.0 minimum-rerun correction package

Status: frozen offline plan; **no paid API calls executed**.

The published `release/v1.1.0/` tree is an immutable input. Its canonical
Parquet remains SHA-256
`0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc`.
This directory is an unpublished corrected derivative, not a release.

## Corrected benchmark

- All 9,048 questions were checked against their exact supplied OWL context.
- The 67 published FALSE BQAs found entailed were replaced, not relabelled.
  Their 67 TRUE pair members and pair membership are preserved.
- Openllet 2.6.5 and the independent OWLAPI structural reasoner both find all
  67 replacements non-entailed in consistent contexts.
- Final membership is 3,016 TRUE BQA, 3,016 FALSE BQA, and 3,016 OEQA.
- The corrected benchmark repairs 5,286 non-Family query IRIs and namespace
  labels and adds 2,322 missing OEQA answers. AR mappings are complete.
- `corrected_model_input_manifest.csv` freezes all 27,144 corrected
  question/representation hashes; `false_bqa_replacements.csv` separately
  freezes all 201 replacement prompt hashes.

## Reuse and exact rerun accounting

Every one of the 81,432 frozen observations matches its original frozen input
hash. The final partition is:

- 64,974 reused observation cells;
- 16,458 rerun observation cells;
- 16,443 executable paid calls after exact prompt/model/config deduplication;
- exactly 5,481 calls per model.

Among the reused cells, 15,855 are the three-model observations for 5,285
category-B NL prompts. These are qualified non-exact-prompt transfers; see
`JOURNAL_METHOD_QUALIFICATION.md`. All other reuse is exact-input-hash reuse.

The executable plan is `rerun_manifest.csv`. Rows with
`deduplicated_request=true` are the 16,443 provider requests; the remaining 15
rows map duplicate observation cells to those requests.

## Current model status and cost

`model_revalidation_2026-09-28.json` records the live, non-inference endpoint
check. All three original provider routes and parameter sets remain available.
GPT-5 Mini's dated target is deprecated and scheduled to shut down on
2026-12-11. Gemini remains an undated alias, so exact historical model identity
cannot be reproduced even though its provider-pinned configuration can be.
Qwen retains its dated model ID and Alibaba route.

At current pinned-route prices, the historical-token projection is $13.778190.
The deliberately conservative 1,024-output-token bound is $27.467296, so the
executor enforces the $25 cap request-by-request and may stop early if outputs
are unexpectedly long.

## Offline validation and execution

Run the complete offline gates:

```powershell
.\.venv\Scripts\python.exe scripts\validate_v1_1_minimum_rerun.py
.\.venv\Scripts\python.exe scripts\run_v1_1_minimum_rerun.py
```

The second command is preflight-only. After explicit approval for paid calls:

```powershell
$env:OPENROUTER_API_KEY = '<key>'
.\.venv\Scripts\python.exe scripts\run_v1_1_minimum_rerun.py --execute --cap-usd 25
```

The executor is sequential, checkpointed in SQLite, uses no automatic provider
retry, claims each deduplicated request before submission, and refuses to
resume past an unresolved in-flight request. This prevents silent duplicate
paid submissions after a crash or ambiguous timeout. Use `--max-requests N`
for a deliberately bounded tranche.

Execution, publication, and modification of archived v1.1.0 remain outside
this preparation phase.
