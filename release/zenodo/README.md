# Zenodo deposit preparation

CORE-LLM-Bench v1.0.0 is archived at
https://zenodo.org/records/22742977 under DOI `10.5281/zenodo.22742977`.
The published archive must not be rebuilt or replaced during post-release
metadata maintenance.

Run `python scripts/prepare_zenodo.py --profile full` and
`python scripts/prepare_zenodo.py --profile public-safe`. The commands create
separate local staging directories and deterministic archives with
profile-specific benchmark files, metadata, manifests, checksums, statistics,
and reproducibility documentation. They perform no network requests and do not
create a Zenodo deposit.

The complete conference-paper citation and persistent identifier remain omitted
until they can be copied from a confirmed bibliographic record. The v1.0.0
source release is available at
https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.0.0.
Family/FHKB-derived material is documented as a modified/adapted FHKB component
under CC BY-SA 3.0 in `NOTICE.md`. The extended journal manuscript is only in
preparation.
