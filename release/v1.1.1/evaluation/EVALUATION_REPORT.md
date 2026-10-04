# Offline corrected v1.1 evaluation review

Status: **complete; frozen; unpublished**. No additional model/API calls were made during release-candidate preparation or final local freezing.

## Integrity

- Checkpoint: 16,443/16,443 completed; zero pending, failed, in-flight, duplicate, hash-mismatched, route-mismatched, or unresolved requests.
- The journal contains 16 schema-nonconformant completed outputs (15 blank/unusable). They are retained as malformed accepted observations in the denominator, consistent with the published evaluation protocol; therefore the requested zero-malformed gate is not satisfied.
- Response journal: 16,443 unique validated records.
- Locally recorded cost: $13.53325078; zero reserved or uncertain charge.
- Final matrix: 64,974 reusable frozen + 16,458 corrected rerun = 81,432 unique observations.
- Qualified non-exact-prompt NL transfers among reused observations: 15,855 = 5,285 questions × 3 models. These are qualified transfers, not exact-prompt reuse.

## Overall corrected results (%)

| Model | N | EM | F1 | Confidence–correctness alignment | OEQA hallucination |
|---|---:|---:|---:|---:|---:|
| GPT-5 mini | 27,144 | 62.65 | 65.86 | 70.89 | 22.23 |
| Gemini 2.5 Flash-Lite | 27,144 | 57.17 | 60.71 | 64.53 | 51.06 |
| Qwen3-30B-A3B-Instruct | 27,144 | 61.50 | 65.09 | 66.17 | 70.62 |

All primary, factorial, explanation-complexity, and published-comparison tables are under `csv/`, `comparison/`, and `latex/`. Full reasoning-tag results are generated separately from this validated per-observation score matrix.

## Interpretation qualifications

- AR is weakest overall, not universally.
- FS is strongest overall and for every model, but not in every fine-grained cell.
- 2-hop is worse overall and for every model/dataset, with three fine-grained reversals.
- Performance decreases monotonically with complexity in aggregate, but not in every detailed stratum.
