# CORE-LLM-Bench v1.0.0 release notes

> **Historical release:** v1.0.0 is preserved and remains citable, but v1.1.0
> is the current benchmark version. See `docs/PUBLICATION_STATUS.md`.

Release date: **2026-09-13**
GitHub Release URL: **https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.0.0**
Zenodo DOI: **10.5281/zenodo.22742977**
Zenodo record URL: **https://zenodo.org/records/22742977**
Hugging Face URL: **https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench**

CORE-LLM-Bench is a controlled neurosymbolic benchmark for evaluating
ontology-grounded reasoning in large language models. The intended canonical
v1.0.0 release contains 9,032 unique question-hop instances from Family, Pizza
100, Pizza 250, and OWL2Bench: 5,999 binary QA (BQA) and 3,033 open-ended QA
(OEQA) instances.

The Hugging Face repository provides the canonical tabular 9,032-row dataset
view, with one row per unique question-hop instance.

Each instance aligns natural-language (NL), formal-symbolic (FS), and
entity-abstracted (AR) conditions at 1-hop or 2-hop context depth. The release
preserves reasoner-derived answers and explanations, minimum/maximum explanation
complexity, linked negative BQA provenance, and coverage under the benchmark's
20-type reasoning taxonomy (eight types instantiated in v1.0.0).

The repository provides profile-aware local Hugging Face and Zenodo packaging,
offline hash/schema/count validation, the benchmark-generation pipeline, and
scripts for reproducing the reported evaluation from separately retained saved
predictions. Model responses are not duplicated in the benchmark rows.

## Licensing, provenance, and citation

The sources have mixed provenance. Pizza is CC BY 3.0; OWL2Bench is
Apache-2.0; and Family is a modified/adapted FHKB resource whose derived
material is distributed under the applicable CC BY-SA 3.0 terms. Repository
software is MIT licensed, and original author-created benchmark questions and
metadata are intended for CC BY 4.0 where separable from source-derived
material. Consult `NOTICE.md` for attribution and modification details; do not
apply a single blanket license or relicense FHKB-derived content as CC BY 4.0.

The published CORE-LLM-Bench conference paper is the primary citation. The
v1.0.0 benchmark archive can be cited with DOI `10.5281/zenodo.22742977`. Do not
infer missing conference citation fields or ORCIDs.

The Hugging Face dataset, GitHub v1.0.0 release, and Zenodo archive are already
published. The GitHub and Zenodo archives must not be rebuilt or replaced as
part of post-release metadata maintenance.
