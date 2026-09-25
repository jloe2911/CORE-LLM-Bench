# CORE-LLM-Bench v1.1.0

CORE-LLM-Bench v1.1.0 is the current release, published on 2026-09-25. It
contains 9,048 question-hop instances: 6,032 binary question-answering (BQA)
instances in 3,016 TRUE/FALSE pairs and 3,016 open-ended question-answering
(OEQA) instances.

- GitHub release: https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0
- Hugging Face: https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.0
- Zenodo DOI: https://zenodo.org/records/22959957 (`10.5281/zenodo.22959957`)

The release provides the finalized explanation schema, collision-safe
natural-language inputs, corrected offline answer and hallucination evaluation,
and finalized reasoning-tag association analysis. It evaluates 81,432 frozen
accepted observations. No benchmark question or model response was regenerated
or modified for this post-publication documentation correction, and no LLM API
call was made.

The attached archive is immutable and matches the published copies. Some
checksum-covered files inside it retain prepublication phrases such as
`release candidate`, `prepared-not-published`, `publication_date: null`, or
`published: false`. These values record the package's build-time state and do
not describe its current publication status. Likewise, staging labels in frozen
provenance files are historical workflow identifiers.

Known scientific limitation: three AR prompts (task IDs 1274, 4094, and 5165)
are confirmed defective. The primary frozen-observation analysis retains them
and provides a separately identified exclusion sensitivity; no response was
rerun.

Licensing is component-specific: software is MIT; separable original CORE
material is CC BY 4.0; Pizza-derived material is CC BY 3.0; OWL2Bench material
is Apache-2.0; and modified/adapted Family/FHKB-derived material is CC BY-SA
3.0. Consult `NOTICE.md` in the archive before redistribution.

The v1.0.0 GitHub tag/release, Hugging Face revision, and Zenodo record remain
immutable historical artifacts.
