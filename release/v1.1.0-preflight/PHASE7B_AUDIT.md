# Phase 7B duplicate-input and reuse audit

Status: audit complete; stopped for scientific review. No benchmark repair, resampling, ID rematerialization, experiment execution, model/API call, commit, or v1.0.0 modification occurred.

## Fatal incompatible inputs

The frozen artifacts contain 9 groups / 19 rows, split as 4 BQA groups / 8 rows and 5 OEQA groups / 11 rows. This differs from the requested expected split of 3 BQA and 6 OEQA groups.

| Hash | Type | Task IDs | Datasets | Category |
|---|---:|---|---|---|
| `00f1e2c6b6b437600a9d66e746f224c23c6b4fd53c64f39791efa1043d3df3a4` | OEQA | 4590;5784 | Pizza100;Pizza250 | 6 OEQA ANSWER-SET COLLISION |
| `153117442841edb4e2d039a70e14765deb1ae38d10ce30993e1757d304a8c4ab` | OEQA | 3699;3700 | FamilyOWL | 4 ENTITY-LABEL COLLISION |
| `1fe4c49ebbdc3af806ca8d40668698b042351c75b8831dceb1d88269fe4ca8b5` | OEQA | 4096;5168 | Pizza100;Pizza250 | 6 OEQA ANSWER-SET COLLISION |
| `52751ca28345c1850befe9ce7c38c0b51452950e038ecca207b0c0e0656a71f6` | BQA | 6563;6564 | OWL2Bench | 5 NEGATIVE-BQA CONSTRUCTION COLLISION |
| `97d9481bdd7a39a3f0e632000fb32bb2a376bd04b94453bf40d1291acf08cfb4` | OEQA | 3368;3370;3375 | FamilyOWL | 4 ENTITY-LABEL COLLISION |
| `9fb52fd56c164d54a3cb3d31cb33a39b338e0de934122052674d63b73ddf76d1` | BQA | 6395;6396 | OWL2Bench | 5 NEGATIVE-BQA CONSTRUCTION COLLISION |
| `aa984aa19f6905c4ac3b3dbbd4f62d795f686a4dd9b2b306ae56bf711c01c94b` | OEQA | 3460;3461 | FamilyOWL | 4 ENTITY-LABEL COLLISION |
| `b264c3dcef1cb1a723c36b329bcddb9fada72bc7bfbb4411b2fce61e79ba4c34` | BQA | 7916;7917 | OWL2Bench | 5 NEGATIVE-BQA CONSTRUCTION COLLISION |
| `cdb998ae49d3ee6d5c6884a6205fd0f2c84110e23c813911d6ce6f2db32e9227` | BQA | 6417;6418 | OWL2Bench | 5 NEGATIVE-BQA CONSTRUCTION COLLISION |

The exhaustive row table, including full NL context, reconstructed evaluated prompt, semantic/provenance identities, and FS/AR equivalents, is `phase7b_incompatible_rows.csv`. Group-level causes and proposed corrections are in `phase7b_incompatible_groups.csv`.

## Compatible duplicates

There are 435 compatible groups / 1,629 rows: AR/B 7/14, FS/D 5/10, NL/D 12/24, and NL/C 411/1,581. Category D denotes incidental cross-dataset duplication: the formal query and gold agree, but there is no evidence that repeated inclusion was intentional. Every group is classified in `phase7b_compatible_duplicate_groups.csv`. Retaining them counts identical evaluated inputs repeatedly in per-row aggregates.

## Required scope

Recommended scope is Option 2. If all fatal groups are removed atomically, membership becomes 9,029: 6,024 BQA rows in 3,012 complete pairs and 3,005 OEQA rows. Do not auto-replace rows merely to retain 9,048. Public IDs and all derived manifests would require later deterministic rematerialization after review.

The fatal instances are local, but their causes are systematic generation-policy defects: label normalization loses identity, there is no post-render uniqueness gate, and dataset scope is absent from coincident Pizza surfaces. The 411 compatible NL representational-collapse groups confirm that the mechanism extends beyond the nine fatal groups.

## Representation-specific uniqueness rule

FATAL: within each representation independently, one exact evaluated-input fingerprint maps to more than one normalized gold semantic value. WARNING/POLICY: one exact evaluated input maps to compatible gold semantics. Apply both checks across datasets as well as within each dataset. Current fatal counts are NL 9/19, FS 0/0, AR 0/0.

## Historical configuration and mixed reuse

See `phase7b_historical_parameter_audit.csv` for VERIFIED / IMPLIED BY CLIENT DEFAULT AT THE TIME / UNKNOWN classifications and `phase7b_historical_run_configs.csv` for every retained run configuration. The primary recommendation is Option 3: rerun the complete v1.1 matrix under one newly frozen request and routing configuration; retain historical exact-input responses as provenance or sensitivity data, not primary mixed-condition observations.

## Default-confidence responses

All 25 are OEQA and all expose a syntactically extractable ANSWER line. Eighteen stored responses truncate that ANSWER line at 500 characters and are not complete reusable answer payloads. Five complete responses omit CONFIDENCE, and two complete responses terminate at literal `CONF`. The published v1.0 evaluator did contain the 0.5 fallback; original metrics were computed before long responses were truncated for checkpoint storage. Exact rows are in `phase7b_default_confidence_audit.csv`.
