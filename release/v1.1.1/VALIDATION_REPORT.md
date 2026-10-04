# v1.1.1 final local release validation

Status: **PASS; frozen; unpublished**. The validated `v1.1.1-rc1` candidate was promoted locally without additional model/API calls.

- Benchmark: 9,048 questions; 6,032 BQA and 3,016 OEQA; 3,016 balanced TRUE/FALSE pairs.
- Proof metadata: 885 stale summaries repaired; zero residual mismatches; shared axioms deduplicated; `M` excluded.
- Observation matrix: 81,432 accepted observations (64,974 reused and 16,458 corrected-rerun cells); 16,443/16,443 deduplicated rerun requests are resolved.
- Qualified non-exact-prompt NL transfers: 15,855 = 5,285 questions × 3 models; these are qualified transfers, not exact-prompt reuse.
- Malformed-accepted policy: 86 retained, including 16 schema-nonconformant rerun outputs (15 blank/unusable).
- Predictions modified: no.
- v1.1.0 modified: no.
- Manuscript modified: no.
- Canonical repository test suite: 139 passed.
- Final release manifest: 84/84 entries verified; zero mismatches.
- Corrected benchmark Parquet SHA-256: `0a051b7ccc021a0fafa1852ee3dd02d49ea79631d67adb86e33b360a7fccb803`.

Scientific interpretation remains qualified: AR is weakest overall, not universally; FS is strongest overall and for every model, but not in every fine-grained cell; 2-hop is worse overall and for every model/dataset, with three fine-grained reversals; and performance decreases monotonically with complexity in aggregate, but not in every detailed stratum. The metric name is standardized to `confidence–correctness alignment`.

All benchmark, response, evaluation, analysis, and provenance artifact hashes are enumerated in `RELEASE_MANIFEST.json` and `SHA256SUMS`.
