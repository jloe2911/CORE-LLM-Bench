# Journal methodological qualification for NL-response reuse

Of the 5,286 non-Family natural-language prompts affected by the namespace
correction, 5,285 (category B) changed only by deletion of the erroneous
parenthetical label `(Genealogy)`; the entity and relation surface forms, the
complete natural-language context, the task type, and the corrected gold
semantics were otherwise unchanged. We therefore reused the frozen model
responses for these 5,285 prompts and rescored them against the corrected
gold answers. This is a qualified response transfer, not exact-prompt reuse:
the original and corrected prompt hashes differ, and no claim is made that a
fresh request to the corrected wording would necessarily produce the same
response. The reuse assumes that removing this presentation-only namespace
qualifier does not materially alter the model's answer; results involving
these cells should be interpreted with that limitation. The remaining
non-Family NL prompt changed its queried object as part of the FALSE-BQA
replacement and is rerun, as are the 66 additional Family replacement NL
prompts.

The row-level evidence is frozen in `nl_category_b_reuse_audit.csv`; all
15,855 model-observation cells covered by this qualification are explicitly
marked `qualified_non_exact_prompt_transfer` in
`observation_differential_audit.csv`.
