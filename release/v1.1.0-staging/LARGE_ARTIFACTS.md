# Finalized v1.1 artifacts excluded from the explanation-schema finalization commit

> **Historical staging record:** this file records the prepublication state of
> the finalized artifacts. CORE-LLM-Bench v1.1.0 was published on 2026-09-25 at
> the [GitHub release](https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0),
> [Hugging Face v1.1.0 revision](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.0),
> and [Zenodo DOI 10.5281/zenodo.22959957](https://zenodo.org/records/22959957).
> The staging language and paths below are retained for provenance only.

The finalized generated benchmark payload revisions below are intentionally
excluded from the explanation-schema finalization commit. They remain in the
local canonical staging tree and are intended for the eventual GitHub Release
assets, Zenodo deposit, and Hugging Face dataset after scientific approval.

The candidate is not yet a public v1.1.0 release. No Zenodo or Hugging Face
location has been assigned.

## Finalized artifact inventory

| Intended release path | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `release/v1.1.0-staging/benchmark/FamilyOWL_1hop.json` | 89,729,635 | `1fb8d1006eeb5ab2c45ff907aeb2a13b51aa5ce788aa57481b1b5a8f63f23c1c` | Canonical finalized FamilyOWL one-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/FamilyOWL_2hop.json` | 205,352,668 | `78b3d4c7a88efd3218f7cf81393f73fee228323d4110190504b2be1fee5939a1` | Canonical finalized FamilyOWL two-hop JSON benchmark; exceeds GitHub's 100 MB blob limit. |
| `release/v1.1.0-staging/benchmark/OWL2Bench_1hop.json` | 28,326,331 | `0b633af84779ae56eabf72ee94dff896e7e58ceb570dcc46189f766781c3a379` | Canonical finalized OWL2Bench one-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/OWL2Bench_2hop.json` | 63,572,699 | `82d70e2978e6735a08aea8cfa64a85a6ed598f595f540d4d306fd4f649c7fecc` | Canonical finalized OWL2Bench two-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/Pizza100_1hop.json` | 18,102,356 | `6e1729232266b66a1425542c7a0a0013f1af1b83b1a6f17791a88004005d5441` | Canonical finalized Pizza100 one-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/Pizza100_2hop.json` | 38,777,937 | `eacea066e489c7a153ea2965f70376170a610fc1e5dcdd1ed7e02c8e7c80c18c` | Canonical finalized Pizza100 two-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/Pizza250_1hop.json` | 22,817,075 | `43ac2d08ca4d66aa78b92a3bdac9a711d5f9345a30569b350f6eb451cc1124d3` | Canonical finalized Pizza250 one-hop JSON benchmark. |
| `release/v1.1.0-staging/benchmark/Pizza250_2hop.json` | 55,985,540 | `5ea5baa46b538c8713420f63df5e050cf15da49f6f283a9419cdb1b43082be36` | Canonical finalized Pizza250 two-hop JSON benchmark. |
| `release/v1.1.0-staging/core_llm_bench_v1_1.parquet` | 10,223,171 | `0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc` | Canonical finalized unified Parquet benchmark. |

The committed `RELEASE_MANIFEST.json`, `SHA256SUMS`, and
`validation_report.json` describe the complete local finalized candidate,
including the payload revisions excluded from this commit.

## Additional excluded audit artifact

| Intended release path | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `release/v1.1.0-phase7c-audit/core_llm_bench_v1_1_phase7c.parquet` | 196,749,988 | `be75f246fcd2ea54ce5fb82411da9a61393cdc3c4ffba0ac12e38b8a084d7431` | Generated Phase 7C audit snapshot with collision-safe natural-language inputs and protected FS/AR fields. |

## Local reproduction and verification

The finalized tree was produced offline from the frozen pre-schema staging
tree with:

```text
python scripts/finalize_v11_explanation_schema.py --source <pre-schema-staging> --output <fresh-output-directory>
```

The earlier Phase 7C audit snapshot can be reproduced after Phase 6
materialization with:

```text
python scripts/phase7c_collision_audit.py --stage release/v1.1.0-phase7c-audit
```

Neither command performs model or API calls. Compare reproduced files with the
listed sizes and hashes and with `release/v1.1.0-staging/SHA256SUMS` before
treating them as accepted candidate artifacts.
