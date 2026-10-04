# Migration from v1.1.0 to v1.1.1

v1.1.1 is the corrected, frozen final local release state promoted from the validated `v1.1.1-rc1` candidate. It remains unpublished, and v1.1.0 remains untouched.

- Formal IRI repair: source IRIs replace defective namespace reconstruction; corrected semantic keys are canonical and original keys remain recorded.
- Complete OEQA gold sets: independently entailed answers omitted from v1.1.0 are restored.
- Repaired FALSE BQA pairs: 67 entailed negative members are replaced by independently verified non-entailed peers, preserving 3,016 balanced pairs.
- Identity-preserving NL normalization: verified display qualifiers are removed through an injective dataset-scoped map; digits and suffixes are retained.
- Corrected proofs and complexity metadata: 885 stale cached complete-proof summaries are rebuilt from answer-group alternatives with shared axioms deduplicated and `M` excluded.
- Partial prompt reruns: only correction-affected cells were rerun; validated unaffected observations were reused. The final matrix contains 64,974 reused and 16,458 rerun observations.
- Qualified NL transfers: 15,855 reused observations represent 5,285 questions × 3 models. These are qualified non-exact-prompt transfers, not exact-prompt reuse.
- Distribution: GitHub uses a lean repository/tag distribution; the complete frozen artifact package will be distributed through Hugging Face and Zenodo.

No additional model/API calls were made during release-candidate preparation or final local freezing. Predictions were not edited. The 86 malformed accepted observations, including 16 rerun schema-nonconformant outputs, remain in the evaluation denominator under the documented policy. No manuscript files were changed.

Interpretation is qualified as follows: AR is weakest overall, not universally; FS is strongest overall and for every model, but not in every fine-grained cell; 2-hop is worse overall and for every model/dataset, with three fine-grained reversals; and performance decreases monotonically with complexity in aggregate, but not in every detailed stratum. The standardized metric name is `confidence–correctness alignment`.
