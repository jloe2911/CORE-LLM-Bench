# CORE-LLM-Bench v1.1: scientific review candidate

> **Historical prepublication review record:** this document describes the
> scientific-review state before v1.1.0 was approved and published. The current
> release is v1.1.0, published on 2026-09-25; see
> [`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md). Status statements below are
> retained as dated provenance and are not current instructions.

Status: **candidate for professor review; not released**. Model experiments are paused pending scientific approval.

## 1. Why v1.1 was necessary

Review of CORE-LLM-Bench v1.0 identified several issues that could affect scientific interpretation: binary questions could be sampled without their intended counterpart; multi-answer explanations did not represent the logic of complete answer justification; reasoning complexity could be inflated; abstract representations could be inconsistent or leak source labels; and distinct entities could collapse to identical natural-language inputs with incompatible gold answers. These were benchmark-construction issues, so the underlying causes were corrected before further comparative experiments.

## 2. Professor comment

The review comments, including the `AsparagusTopping` abstraction example, raised a common concern: do the questions, representations, explanations, and complexity labels preserve the intended ontology semantics consistently? The audit was therefore broadened beyond individual examples to the complete candidate benchmark and all three representations.

## 3. Root cause identified

The audit found related but distinct causes:

- BQA grouping relied on Task-ID manipulation rather than semantic structure.
- Pizza `DomainConcept` could be selected as a binary Membership target even when it was not a meaningful benchmark concept.
- OEQA proofs for different returned answers were flattened as though they were interchangeable alternatives.
- The reasoning tagger could return broad tags before inspecting nested OWL expressions, and duplicate semantic occurrences could inflate tag counts.
- Sparse combined strata could be discarded instead of handled by a controlled fallback.
- Abstract-representation discovery and rendering did not consistently preserve URI identity, include required individuals, or suppress source lexical annotations.
- Natural-language verbalization could give distinct entities, such as `Employee_10` and `Employee_47`, the same rendered label.

## 4. Correction implemented

### BQA positive/negative pairing

Previously, sampling groups were inferred by manipulating Task IDs, which could break intended TRUE/FALSE pairs. BQA questions are now grouped structurally by **dataset, hop, root, subject, and predicate**. Each TRUE/FALSE pair is indivisible during sampling. The candidate contains 3,016 complete pairs, with zero unpaired positives and zero unpaired negatives.

### Pizza `DomainConcept`

`DomainConcept` remains in the ontology and may legitimately occur in OEQA answers. It is excluded only as a Pizza binary Membership target. Where another meaningful entailed class exists, that class is used instead. The candidate has zero Pizza `DomainConcept` BQA targets.

### OEQA explanation semantics

Previously, proofs belonging to different returned answers were flattened as if all proofs were alternatives. The corrected semantics are:

- for one answer, its proof alternatives are **disjunctive**;
- across different correct answers, the evidence requirement is **conjunctive**;
- a complete OEQA explanation must therefore justify every gold answer.

Answer-to-proof provenance is taken exactly from `inferred.object`, not reconstructed by heuristic string matching.

### Reasoning tags

The previous tagger could return broad tags such as `H` or `Q` before examining nested OWL class expressions and could inflate results through duplicate semantic-axiom occurrences. The corrected tagger recursively inspects OWL constructs, permits multiple applicable tags per axiom, and removes artificial duplicates.

The canonical taxonomy remains:

`D H T S A J N E ∩ ¬ I F V Y Q R C L U M`

`M` is metadata for heterogeneous TBox reasoning; it is not a primitive reasoning operation and is excluded from primitive complexity. After genuine symbolic regeneration, the selected benchmark explanations exercise `D, H, T, S, N, I, R`. Other supported constructs occur in source ontologies but are not exercised by the regenerated explanations of these selected tasks. In other words, **present in ontology does not mean used in a benchmark explanation**.

### Explanation complexity and sampling bins

The primary measure is **minimum complete primitive-tag count**.

- BQA: the minimum primitive-tag count of a valid proof.
- OEQA: the minimum primitive-tag count of a complete conjunctive explanation covering the entire gold answer set.
- `M` is excluded from primitive length.

The final fixed-integer bins are:

| Task | Low | Medium | High |
|---|---:|---:|---:|
| BQA | 1 | 2 | 3+ |
| OEQA | 1–3 | 4–5 | 6+ |

These task-specific thresholds were selected after a 40-scheme fixed-integer audit and avoid the pathological sparsity produced by the old thresholds. Identical integer complexity values are never split across bins.

### Sparse-stratum handling

Rare ABox × complexity strata are no longer silently discarded. The fallback first preserves the complexity bin and relaxes the ABox bin. No question changes complexity category merely to make a stratum sampleable, and no eligible group disappears solely because its combined stratum is sparse.

### Abstract representation

The `AsparagusTopping` example exposed a combination of missing individual mappings, source-label leakage from annotations, different URIs collapsing to the same display label, repeated sentences, and inconsistent mappings among question, context, and answer.

The corrected representation uses graph-semantic entity discovery; deterministic `ClassN`, `PropertyN`, `DataPropertyN`, and `IndividualN` mappings; URI-keyed relations; exclusion of source lexical labels from rendering; deterministic sentence deduplication; and aligned question/context/answer mappings.

### Natural-language identity collisions

Previously, distinct entities such as `Employee_10` and `Employee_47` could receive the same rendered label, producing identical NL inputs with incompatible gold answers. Rendering is now collision-safe: unique labels remain unchanged, while only colliding entities receive deterministic, ontology-grounded disambiguation. Compatible duplicate inputs remain documented separately and can be evaluated with a unique-input-weighted sensitivity analysis.

### Sequential public IDs

Public Task IDs are now globally unique integers `1..9048`. Semantic identity and provenance remain in separate fields. IDs no longer encode, or act as authority for, grouping and sampling semantics.

## 5. Validation result

The current candidate was revalidated without rebuilding or resampling:

- membership: 9,048 rows;
- BQA: 6,032 rows — 3,016 TRUE and 3,016 FALSE;
- complete TRUE/FALSE pairs: 3,016;
- unpaired BQA positives/negatives: 0 / 0;
- OEQA: 3,016 rows;
- Pizza `DomainConcept` BQA targets: 0;
- public IDs: 9,048 unique contiguous integers, `1..9048`;
- incompatible exact-input/gold groups: NL 0, FS 0, AR 0;
- AR duplicate sentences: 0;
- AR unmapped required entities: 0;
- AR mapping inconsistencies: 0;
- AR source identifiers remaining: 0;
- AR source lexical-label leakage: 0;
- v1.0.0 changed artifacts: 0.

### Final benchmark statistics

| Dataset | Hop | Questions |
|---|---:|---:|
| Family | 1hop | 1,881 |
| Family | 2hop | 1,881 |
| Pizza100 | 1hop | 495 |
| Pizza100 | 2hop | 495 |
| Pizza250 | 1hop | 618 |
| Pizza250 | 2hop | 618 |
| OWL2Bench | 1hop | 1,467 |
| OWL2Bench | 2hop | 1,593 |
| **Total** |  | **9,048** |

Task totals are BQA 6,032 and OEQA 3,016. BQA contains 3,016 TRUE and 3,016 FALSE rows.

### Compact v1.0 / v1.1 comparison

| Issue | v1.0 behavior | v1.1 correction | Scientific impact |
|---|---|---|---|
| BQA pairing | Task-ID-derived grouping could split pairs | Structural grouping; pairs indivisible | Balanced, semantically valid binary sampling |
| Pizza target selection | `DomainConcept` could be a BQA target | Excluded only from Pizza Membership BQA | Removes uninformative binary targets without altering the ontology |
| OEQA proofs | Proofs across answers flattened as alternatives | Alternatives within answers; conjunction across answers | Explanations justify the complete gold answer set |
| Proof provenance | Heuristic matching could associate proofs | Exact `inferred.object` provenance | Traceable answer-to-proof identity |
| Reasoning tags | Early broad tags and duplicate inflation | Recursive multi-tag inspection and semantic deduplication | More faithful reasoning-type coverage |
| Complexity | Flattened/inflated proof counts | Minimum complete primitive-tag count; `M` excluded | Complexity matches the least complete valid justification |
| Sparse strata | Rare combined strata could disappear | Preserve complexity and relax ABox bin first | Better coverage without relabeling difficulty |
| Abstract representation | Missing mappings, leakage, collisions, repetitions | Graph-semantic, URI-keyed deterministic mapping and deduplication | Consistent, label-independent abstraction |
| NL identity | Distinct entities could render identically | Collision-safe disambiguation only where needed | Zero incompatible exact NL input groups |
| Public IDs | IDs carried implicit grouping semantics | Sequential IDs plus separate semantic/provenance fields | Stable public references without semantic overloading |

## 6. Scientific consequence

The corrected candidate is naturally 9,048 questions; it is not forced back to the old 9,032 total. The changes affect which semantic questions are selected and how explanations, complexity, and representations are interpreted. Consequently, v1.1 should be treated as a corrected benchmark candidate requiring scientific approval, not as a formatting-only revision.

Experiment state is deliberately frozen:

- **Gemini: COMPLETE BUT PROVISIONAL PENDING BENCHMARK APPROVAL** — 27,144 / 27,144 accepted observations, preserved unchanged.
- **Qwen: PAUSED / NOT COMPLETE** — only the already recorded 10-row provider canary is present; no comparative result is computed.
- **GPT: PAUSED / NOT STARTED** — 0 / 27,144 observations.

No final comparative results have been calculated, and v1.1 has not been published, tagged, or released.

## 7. Points for scientific approval

Please confirm whether you agree with:

1. The corrected OEQA explanation semantics: alternatives within one answer and conjunction across different answers.
2. The primary complexity definition: minimum complete primitive-tag count.
3. Retaining `M` as metadata while excluding it from primitive complexity.
4. The task-specific bins: BQA `1 / 2 / 3+`; OEQA `1–3 / 4–5 / 6+`.
5. Retaining reasoning types with zero benchmark-explanation coverage in the documented taxonomy while stating clearly that the current instances do not exercise them.
6. Collision-safe NL rendering rather than deleting otherwise valid semantic questions.
7. Retaining the natural corrected size of 9,048 rather than forcing the previous 9,032 total.
8. Retaining compatible duplicate model inputs, documenting them, and optionally reporting a unique-input-weighted sensitivity analysis.
