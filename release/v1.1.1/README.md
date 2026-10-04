# CORE-LLM-Bench v1.1.1

Status: frozen final local release state; unpublished. This tree was promoted from the validated `v1.1.1-rc1` candidate.

The benchmark is under `benchmark/`, frozen and corrected-rerun response artifacts under `responses/`, offline evaluation under `evaluation/`, and reasoning-tag analysis under `analysis/`. `RELEASE_MANIFEST.json` and `SHA256SUMS` bind every distributed artifact. No additional model/API calls were made during release-candidate preparation or final local freezing. Predictions, corrected gold answers, evaluation tables, reasoning-tag outputs, and benchmark semantics were not changed during promotion. See `CHANGELOG.md` and `VALIDATION_REPORT.md`.

## Distribution / artifact availability

The `v1.1.1` Git tag will contain a lean reproducibility subset: source code, correction and evaluation scripts, tests, release documentation, aggregate evaluation tables, and compact provenance artifacts. Large generated artifacts are intentionally excluded from ordinary Git history. The complete frozen release inventory is defined by `RELEASE_MANIFEST.json` and `SHA256SUMS`. The canonical corrected benchmark will be distributed through Hugging Face, and the complete archival release—including benchmark JSON files, frozen model responses, full provenance matrices, and per-observation outputs—will be distributed through Zenodo. Hashes in the release manifest identify the exact artifacts across these distribution channels.
