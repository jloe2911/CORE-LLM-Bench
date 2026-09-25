# CORE-LLM-Bench v1.1 final offline evaluation

Status: **complete**. The evaluation used 9,048 frozen questions and 81,432 accepted observations. No model/API request is made by the evaluation script.

## Integrity and identity

- All three models contribute exactly 27,144 unique accepted observations; missing, duplicate, extra, semantic, input-hash, provider, returned-model, and unexpected parser-replay mismatches are all zero.
- The corrected parser classifies explicit blank `ANSWER:` fields as empty in every representation. These intentional replay differences are recorded in `observation_integrity.csv`.
- Malformed-but-accepted outputs remain in every relevant denominator (Gemini 5; Qwen 58; GPT 7).
- Qwen's nine non-accepted transient diagnostic rows are not observations and are reported separately in the integrity audit.
- GPT exact scientific identifier: `gpt-5-mini-2025-08-07`. The explicit dated OpenRouter request was `openai/gpt-5-mini-2025-08-07` and succeeded through provider `OpenAI`; response records returned alias `openai/gpt-5-mini`. The alias alone is not used as dated-snapshot evidence.

## Overall results (%)

| Model | N | Answer EM | Answer F1 | Confidence-correctness | OEQA empty | Generated answers | OEQA hallucination | Malformed accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 mini | 27,144 | 59.85 | 61.68 | 66.55 | 2,768 | 14,391 | 28.71 | 7 |
| Gemini 2.5 Flash-Lite | 27,144 | 57.21 | 59.27 | 63.03 | 658 | 23,213 | 57.09 | 5 |
| Qwen3-30B-A3B-Instruct | 27,144 | 60.54 | 62.91 | 63.99 | 15 | 40,159 | 73.39 | 58 |

## Sensitivity excluding confirmed defective AR prompts (%)

| Model | N | Answer EM | Answer F1 | Confidence-correctness | OEQA empty | Generated answers | OEQA hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 mini | 27,141 | 59.85 | 61.69 | 66.55 | 2,767 | 14,389 | 28.70 |
| Gemini 2.5 Flash-Lite | 27,141 | 57.22 | 59.27 | 63.04 | 658 | 23,159 | 57.21 |
| Qwen3-30B-A3B-Instruct | 27,141 | 60.54 | 62.91 | 63.99 | 15 | 40,098 | 73.48 |

Exactly 3 AR questions are explicitly flagged as confirmed defective (task IDs 1274, 4094, 5165). The prevalence of additional semantic-support defects remains unresolved; this sensitivity analysis must not be read as proving that all other AR prompts are semantically supported.

## Comparison with the previous evaluation

AR now uses `ar_gold_answer`. NL/FS answer EM and F1 did not change; their hallucination values changed because the corrected definition uses the complete representation-specific gold set and excludes empty predictions from the generated-answer denominator. The detailed representation-level deltas are in `csv/comparison_with_previous_by_representation.csv`.

## Methods

Answer EM/F1 use the hash-pinned SAGE-QA ontology answer-set evaluator (`86b9bedfb3784145f3d20e2b9b8b6082ee4252e57918807bf02529b3904eefd6`, source commit `dbdbb50708bdc6c686ef82518ec71c1d1bf55985`). AR uses `ar_gold_answer`; NL/FS use `gold_answer`. Confidence-correctness alignment is `1 - abs(confidence - per-question Answer F1)` with the documented 0.5 fallback. OEQA hallucination is the micro-average `unsupported generated answers / generated answers` against the complete representation-specific normalized gold set. Empty predictions are counted separately and have an undefined per-observation rate; an aggregate with zero generated answers is emitted blank rather than assigned a value. BQA rows are excluded. Complexity is the frozen minimum complete primitive-tag count with BQA bins 1/2/3+ and OEQA bins 1-3/4-5/6+. `M` is metadata and is excluded from primitive complexity.

Detailed factorial, complexity, reasoning-tag, dataset-statistics, and per-observation outputs are in `csv/`; manuscript tables are in `latex/`; the machine-readable provenance gate is in `audit/integrity_audit.json`.
