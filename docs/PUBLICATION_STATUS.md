# CORE-LLM-Bench publication status

## Current release

CORE-LLM-Bench **v1.1.1** is the current version. It was published on
**2026-10-04** and contains **9,048** question-hop instances: **6,032 BQA** in
3,016 TRUE/FALSE pairs and **3,016 OEQA**.

- GitHub lean reproducibility/source release: https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.1
- Hugging Face canonical corrected benchmark: https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.1
- Zenodo complete frozen archival package: https://zenodo.org/records/23138373 (`10.5281/zenodo.23138373`)
- Zenodo concept DOI: `10.5281/zenodo.22742976`

Version 1.1.0 remains available as an immutable historical version. Version
1.1.1 supersedes it for future benchmark use.

## Immutable-package clarification

The published v1.1.1 release artifacts are immutable and have not been rebuilt
or replaced. Frozen files under `release/v1.1.1/**` were created before
publication and may therefore contain wording such as `unpublished`, `will be
distributed`, `release candidate`, or similar build-state language. That text
is intentionally preserved as frozen release provenance and does not represent
the current publication status. These frozen files must not be edited to update
their wording.

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
They also do not describe the current v1.1.1 distribution workflow. The v1.0.0
tags and public archives remain unchanged.

## Scientific and licensing notes

Three v1.1.1 AR prompts, task IDs 1274, 4094, and 5165, are confirmed
defective. The released evaluation retains the three prompts and reports the
exclusion sensitivity.

Licensing remains component-specific: software is MIT; separable original CORE
material is CC BY 4.0; Pizza-derived material is CC BY 3.0; OWL2Bench material
is Apache-2.0; and modified/adapted Family/FHKB-derived material is CC BY-SA
3.0. See `NOTICE.md` before redistribution.
