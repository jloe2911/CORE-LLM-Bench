# CORE-LLM-Bench v1.1 reasoning-tag difficulty analysis

## Scope and method

This offline analysis joins 9,048 canonical questions to all 81,432 corrected accepted observations. Corrected minima are reconstructed from the answer-group alternatives (conjunctive across answers, disjunctive within an answer), with shared axioms deduplicated and primitive tags reconstructed by the benchmark's deterministic fallback. The input audit found 0 stale cached scalar/complete-union records. The primary tag definition is presence in every tied minimum; presence in any tied minimum is reported as sensitivity. BQA FALSE rows use their paired TRUE entailment only as structural annotation, never as a proof of FALSE. `M` is excluded.

Intervals for descriptive means cluster BQA at the TRUE/FALSE pair and OEQA at the question. Adjusted linear models cluster on the same units and include tag, model, representation, dataset, hop, task-specific covariates, representation-by-tag interactions, and model-by-tag interactions. Coefficients are associations, not causal effects or evidence that a model executed the tagged operation.

## Descriptive findings

- BQA/NL: lowest tag-specific mean was R (0.646, n=664); highest was I (0.703).
- BQA/FS: lowest tag-specific mean was H (0.795, n=1,586); highest was I (0.893).
- BQA/AR: lowest tag-specific mean was R (0.516, n=664); highest was H (0.580).
- OEQA/NL: lowest tag-specific mean was J (0.081, n=107); highest was S (0.301).
- OEQA/FS: lowest tag-specific mean was R (0.138, n=640); highest was I (0.549).
- OEQA/AR: lowest tag-specific mean was I (0.128, n=1,051); highest was H (0.243).

The comparisons above omit `D` and sparse tags. `D` occurs in every question and is not discriminating. Sparse task-tag cells are BQA-N (n=78), BQA-S (n=40), OEQA-N (n=42), OEQA-T (n=10); they are descriptive only. Raw tag means remain confounded by tag co-occurrence and benchmark composition.

## Adjusted interactions

Holm-corrected representation-by-tag F1 interactions: BQA: tag_H:C(representation, Treatment(reference="NL"))[T.AR] (-0.035); BQA: tag_I:C(representation, Treatment(reference="NL"))[T.AR] (-0.109); BQA: tag_I:C(representation, Treatment(reference="NL"))[T.FS] (+0.028); BQA: tag_R:C(representation, Treatment(reference="NL"))[T.AR] (-0.063); BQA: tag_R:C(representation, Treatment(reference="NL"))[T.FS] (+0.083); OEQA: tag_H:C(representation, Treatment(reference="NL"))[T.FS] (+0.096); OEQA: tag_I:C(representation, Treatment(reference="NL"))[T.AR] (-0.270); OEQA: tag_R:C(representation, Treatment(reference="NL"))[T.AR] (-0.067); OEQA: tag_R:C(representation, Treatment(reference="NL"))[T.FS] (-0.532)

Holm-corrected model-by-tag F1 interactions: BQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (+0.030); BQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.124); BQA: tag_I:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.038); BQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (+0.023); BQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.021); OEQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (+0.101); OEQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.094); OEQA: tag_I:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.024); OEQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (-0.027); OEQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (-0.111)

All adjusted-design diagnostics are recorded in `csv/model_diagnostics.csv`; 16 of 16 planned fits passed the rank, condition-number, and frequency gates. Reference groups are explicit in `csv/adjusted_associations.csv`.

## Sensitivity and limitations

Changing the tag definition from EVERY to ANY tied minimum changed a tag/representation F1 mean by at most 0.064708; full membership and estimate changes are in `csv/tied_minimum_sensitivity.csv`. Excluding the three confirmed defective AR questions (nine observations) changed any primary tag/representation F1 mean by at most 0.000391. The analysis does not imply exhaustive semantic validation of other AR contexts. Corrected proof-derived complexity results are reported in the parent evaluation directory.
