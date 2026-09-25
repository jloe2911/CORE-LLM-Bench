# CORE-LLM-Bench publication status

## Current release

CORE-LLM-Bench **v1.1.0** is the current version. It was published on
**2026-09-25** and contains **9,048** question-hop instances: **6,032 BQA** in
3,016 TRUE/FALSE pairs and **3,016 OEQA**.

- GitHub: https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0
- Hugging Face: https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.0
- Zenodo DOI: https://zenodo.org/records/22959957 (`10.5281/zenodo.22959957`)

## Immutable-package clarification

The published v1.1.0 archive is immutable and has not been rebuilt or replaced.
Its checksum-covered `README.md`, `RELEASE_NOTES.md`, `RELEASE_METADATA.json`,
`RELEASE_MANIFEST.json`, and `VALIDATION_REPORT.json` were created before
publication and therefore contain phrases such as `release candidate`,
`prepared-not-published`, `publication_date: null`, or `published: false`.
Those fields record the package's build-time state. They do not describe the
current public availability of v1.1.0.

Directories and files named `staging`, `preflight`, `phase7*`, and
`PROFESSOR_REVIEW*` are also frozen workflow or provenance records. Their
historical status labels and pending-work statements are not current release
instructions. The authoritative current status is the publication information
above.

## Historical v1.0.0 material

The `final_benchmark/` packages, `release/huggingface/` export commands,
`scripts/prepare_zenodo.py`, `scripts/validate_release.py`,
`RELEASE_AUDIT.md`, and `release/RELEASE_NOTES_v1.0.0*.md` document the
historical v1.0.0 release workflow and its 9,032-question schema. They remain
available for reproducibility but are not the v1.1.0 distribution workflow.
The v1.0.0 tags and public archives remain unchanged.

## Scientific and licensing notes

Three v1.1.0 AR prompts, task IDs 1274, 4094, and 5165, are confirmed
defective. The primary frozen-observation analysis retains them and includes a
separately identified exclusion sensitivity; no response was rerun.

Licensing remains component-specific: software is MIT; separable original CORE
material is CC BY 4.0; Pizza-derived material is CC BY 3.0; OWL2Bench material
is Apache-2.0; and modified/adapted Family/FHKB-derived material is CC BY-SA
3.0. See `NOTICE.md` before redistribution.
