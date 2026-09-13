# Changelog

## 1.0.0 - 2026-09-13

- Define the public benchmark as 9,032 unique question-hop instances from Family, Pizza 100, Pizza 250, and OWL2Bench.
- Package both 1-hop and 2-hop artifacts for each source dataset.
- Document BQA/OEQA tasks and NL/FS/AR representations.
- Add a machine-readable artifact manifest and per-instance reasoning metadata index.
- Add offline release validation and local Hugging Face and Zenodo preparation tools.
- Include the 20-tag reasoning taxonomy audit, with eight instantiated types in v1.0.0.
- Use Answer EM and Answer F1 through the hash-pinned SAGE-QA evaluator adapter for manuscript result processing.
- Add explicit `full` and `public-safe` local release profiles. The fallback
  profile excludes Family/FHKB source-bearing and derived payloads while
  preserving the canonical four-dataset full candidate privately.

Released on 2026-09-13. The published conference paper remains the primary citation; insert its complete confirmed bibliographic record before release. The extended journal manuscript is in preparation.
