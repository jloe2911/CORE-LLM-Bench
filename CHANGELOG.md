# Changelog

## 1.1.0 - 2026-09-25

- Publish the schema-finalized 9,048-question benchmark with deterministic public IDs and semantic keys.
- Publish the version through the [GitHub release](https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0), [Hugging Face v1.1.0 revision](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.0), and Zenodo DOI [`10.5281/zenodo.22959957`](https://zenodo.org/records/22959957).
- Preserve all aligned NL, FS, and AR inputs used by the frozen 81,432-observation experiment.
- Add complete answer-level explanation objects, tied minimum explanations, primitive reasoning tags, and derived complexity metadata.
- Publish corrected offline Answer EM/F1 and hallucination evaluation without rerunning or modifying model responses.
- Add the final reasoning-tag difficulty analysis and explicit sensitivity results for three confirmed defective AR prompts.
- Preserve v1.0.0 as an immutable historical GitHub, Hugging Face, and Zenodo version.
- Retain prepublication labels inside frozen provenance and the immutable archive as historical build-state metadata; see `docs/PUBLICATION_STATUS.md` for the current status.

## 1.0.0 - 2026-09-13

- Define the public benchmark as 9,032 unique question-hop instances from Family, Pizza 100, Pizza 250, and OWL2Bench.
- Package both 1-hop and 2-hop artifacts for each source dataset.
- Document BQA/OEQA tasks and NL/FS/AR representations.
- Add a machine-readable artifact manifest and per-instance reasoning metadata index.
- Add offline release validation and local Hugging Face and Zenodo preparation tools.
- Include the 20-tag reasoning taxonomy audit, with eight instantiated types in v1.0.0.
- Use Answer EM and Answer F1 through the hash-pinned SAGE-QA evaluator adapter for manuscript result processing.
- Add explicit `full` and `public-safe` local release profiles. The fallback
  profile excludes Family/FHKB source-bearing and derived payloads while
  preserving the canonical four-dataset full candidate privately.

Released on 2026-09-13 and archived on Zenodo under DOI
[`10.5281/zenodo.22742977`](https://doi.org/10.5281/zenodo.22742977). The
canonical tabular 9,032-row dataset view is available on
[Hugging Face](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench). The
published conference paper remains the primary citation; its complete
confirmed bibliographic record has not been inferred here. The extended
journal manuscript is in preparation.
