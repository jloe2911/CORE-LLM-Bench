# CORE-LLM-Bench v1.1 reasoning-tag difficulty analysis

## Scope and method

This offline analysis joins 9,048 canonical questions to all 81,432 corrected accepted observations. It uses the final `complete_explanation.minimum_explanations` schema. The primary tag definition is presence in every tied minimum; presence in any tied minimum is reported as sensitivity. BQA FALSE rows use their paired TRUE entailment only as structural annotation, never as a proof of FALSE. OEQA minima cover the complete gold-answer set and retain shared-axiom deduplication. `M` is excluded.

Intervals for descriptive means cluster BQA at the TRUE/FALSE pair and OEQA at the question. Adjusted linear models cluster on the same units and include tag, model, representation, dataset, hop, task-specific covariates, representation-by-tag interactions, and model-by-tag interactions. Coefficients are associations, not causal effects or evidence that a model executed the tagged operation.

## Descriptive findings

- BQA/NL: lowest tag-specific mean was R (0.605, n=664); highest was I (0.701).
- BQA/FS: lowest tag-specific mean was H (0.702, n=1,586); highest was I (0.876).
- BQA/AR: lowest tag-specific mean was R (0.518, n=664); highest was H (0.580).
- OEQA/NL: lowest tag-specific mean was R (0.163, n=381); highest was I (0.319).
- OEQA/FS: lowest tag-specific mean was R (0.016, n=381); highest was I (0.640).
- OEQA/AR: lowest tag-specific mean was R (0.020, n=381); highest was H (0.226).

The comparisons above omit `D` and sparse tags. `D` occurs in every question and is not discriminating. Sparse task-tag cells are BQA-N (n=78), BQA-S (n=40), OEQA-N (n=42), OEQA-S (n=16), OEQA-T (n=1); they are descriptive only. Raw tag means remain confounded by tag co-occurrence and benchmark composition.

## Adjusted interactions

Holm-corrected representation-by-tag F1 interactions: BQA: tag_H:C(representation, Treatment(reference="NL"))[T.AR] (-0.037); BQA: tag_H:C(representation, Treatment(reference="NL"))[T.FS] (-0.030); BQA: tag_I:C(representation, Treatment(reference="NL"))[T.AR] (-0.114); BQA: tag_I:C(representation, Treatment(reference="NL"))[T.FS] (+0.076); BQA: tag_R:C(representation, Treatment(reference="NL"))[T.FS] (+0.154); OEQA: tag_H:C(representation, Treatment(reference="NL"))[T.AR] (-0.076); OEQA: tag_I:C(representation, Treatment(reference="NL"))[T.AR] (-0.293); OEQA: tag_I:C(representation, Treatment(reference="NL"))[T.FS] (+0.090); OEQA: tag_R:C(representation, Treatment(reference="NL"))[T.AR] (-0.163); OEQA: tag_R:C(representation, Treatment(reference="NL"))[T.FS] (-0.498)

Holm-corrected model-by-tag F1 interactions: BQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.054); BQA: tag_I:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (-0.065); OEQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (+0.137); OEQA: tag_H:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.133); OEQA: tag_I:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (-0.025); OEQA: tag_I:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (+0.032); OEQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Gemini 2.5 Flash-Lite] (+0.054); OEQA: tag_R:C(model, Treatment(reference="GPT-5 mini"))[T.Qwen3-30B-A3B-Instruct] (-0.045)

All adjusted-design diagnostics are recorded in `csv/model_diagnostics.csv`; 16 of 16 planned fits passed the rank, condition-number, and frequency gates. Reference groups are explicit in `csv/adjusted_associations.csv`.

## Sensitivity and limitations

Changing the tag definition from EVERY to ANY tied minimum changed a tag/representation F1 mean by at most 0.113328; full membership and estimate changes are in `csv/tied_minimum_sensitivity.csv`. Excluding the three confirmed defective AR questions (nine observations) changed any primary tag/representation F1 mean by at most 0.000674. The analysis does not imply exhaustive semantic validation of other AR contexts. Existing complexity-bin outputs were not modified and remain optional supplementary context.
