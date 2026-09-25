# CORE-LLM-Bench v1.1.0

This release candidate provides the finalized explanation schema, collision-safe natural-language inputs, corrected offline answer and hallucination evaluation, and finalized reasoning-tag association analysis. It contains 9,048 questions (6,032 BQA in 3,016 TRUE/FALSE pairs and 3,016 OEQA) and evaluation of 81,432 accepted observations.

The v1.0.0 GitHub tag/release, Zenodo record 22742977, and Hugging Face tag v1.0.0 remain immutable historical artifacts. See `PRESERVATION.json` for verified identities.

Zenodo DOI `10.5281/zenodo.22959957` and record ID `22959957` are reserved specifically for v1.1.0. The draft is not yet published, so the publication date remains pending.

Known scientific limitation: three AR prompts (task IDs 1274, 4094, and 5165) are confirmed defective; the primary frozen-observation analysis retains them and provides a separately identified exclusion sensitivity. No response was rerun.
