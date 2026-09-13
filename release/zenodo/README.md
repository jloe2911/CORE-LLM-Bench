# Zenodo deposit preparation

Run `python scripts/prepare_zenodo.py --profile full` and
`python scripts/prepare_zenodo.py --profile public-safe`. The commands create
separate local staging directories and deterministic archives with
profile-specific benchmark files, metadata, manifests, checksums, statistics,
and reproducibility documentation. They perform no network requests and do not
create a Zenodo deposit.

Before upload, resolve every item marked `TODO` in `zenodo_metadata.json`: copy the complete conference-paper citation and persistent identifier from the confirmed manuscript record, fix the release date, and create the `v1.0.0` source tag/release snapshot. Then rerun validation against the exact archive contents. Family/FHKB-derived material is already documented as a modified/adapted FHKB component under CC BY-SA 3.0 in `NOTICE.md`. Do not add a DOI until Zenodo assigns one. The extended journal manuscript is only in preparation.
