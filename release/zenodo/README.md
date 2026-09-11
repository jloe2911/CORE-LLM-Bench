# Zenodo deposit preparation

Run `python scripts/prepare_zenodo.py` after `python scripts/validate_release.py`. The command creates `release/zenodo/package/` locally and copies the frozen benchmark packages, release metadata, reasoning-coverage results, and reproducibility documentation into it. It performs no network requests and does not create a Zenodo deposit.

Before upload, resolve every item marked `TODO` in `zenodo_metadata.json`, confirm the final author list and release date, clear Family ontology redistribution, create the corresponding source tag/release snapshot, and rerun validation against the exact archive contents. Do not add a DOI until Zenodo assigns one.
