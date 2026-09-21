# Large artifacts excluded from the professor-review branch

The full generated v1.1 candidate artifacts listed below are intentionally
excluded from Git because of repository size limits. They remain preserved in
the local full-candidate checkpoint and will be distributed with the eventual
versioned dataset release after scientific approval.

The benchmark candidate is not yet a public v1.1.0 release.

No Zenodo or Hugging Face location has been assigned for these artifacts.

## Artifact inventory

| Filename | Intended relative path | Bytes | SHA-256 | Type and purpose |
|---|---|---:|---|---|
| `FamilyOWL_2hop.json` | `release/v1.1.0-staging/benchmark/FamilyOWL_2hop.json` | 203,849,891 | `b112510a8b784bd70b9b6105e3ab85bfb45fa5c26e0d58e1129f90f3ddb4b927` | Generated canonical JSON benchmark artifact for the FamilyOWL two-hop partition. |
| `core_llm_bench_v1_1_phase7c.parquet` | `release/v1.1.0-phase7c-audit/core_llm_bench_v1_1_phase7c.parquet` | 196,749,988 | `be75f246fcd2ea54ce5fb82411da9a61393cdc3c4ffba0ac12e38b8a084d7431` | Generated Phase 7C audit snapshot with collision-safe natural-language inputs and protected FS/AR fields. |

The staging `RELEASE_MANIFEST.json` and `SHA256SUMS` intentionally continue to
describe the complete materialized candidate, including the excluded
`FamilyOWL_2hop.json`, so a locally materialized copy can be checked against the
accepted candidate. This pointer records the separately excluded Phase 7C audit
snapshot.

## Local reproduction and verification

Start from the frozen Phase 5 membership and the candidate source state. To
materialize the Phase 6 candidate into a repository-local directory, run:

```text
python scripts/phase6_materialize_release.py --stage release/v1.1.0-staging
```

To reproduce the Phase 7C collision audit snapshot after Phase 6 materialization,
run:

```text
python scripts/phase7c_collision_audit.py --stage release/v1.1.0-phase7c-audit
```

These commands write generated artifacts. They are not required merely to
review the committed manifests, mappings, summaries, reports, source code, and
tests. Compare reproduced files with the byte sizes and SHA-256 values above
and with `release/v1.1.0-staging/SHA256SUMS` before treating them as the accepted
candidate artifacts. Neither command performs model or API calls.
