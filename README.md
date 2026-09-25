# CORE-LLM-Bench

CORE-LLM-Bench is a neurosymbolic benchmark for evaluating language models on questions whose answers and explanations are grounded in OWL ontologies. Symbolic reasoning with Pellet provides entailed gold answers and proof metadata; models are evaluated through natural-language, formal-symbolic, and entity-abstracted views of the same underlying tasks.

> **Current release:** CORE-LLM-Bench v1.1.0 contains 9,048 frozen question-hop
> instances and is available
> on [Hugging Face](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench).
> The complete source and data package is attached to the
> [GitHub v1.1.0 release](https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0).
> Zenodo DOI [10.5281/zenodo.22959957](https://doi.org/10.5281/zenodo.22959957)
> is reserved for the manually published archival copy. The immutable v1.0.0
> archive remains at [10.5281/zenodo.22742977](https://doi.org/10.5281/zenodo.22742977).

Version 1.1.0 contains **9,048 unique question-hop instances** from four source datasets:

| Dataset | 1-hop | 2-hop | Total |
| --- | ---: | ---: | ---: |
| Family / FamilyOWL | 1,881 | 1,881 | 3,762 |
| Pizza 100 | 495 | 495 | 990 |
| Pizza 250 | 618 | 618 | 1,236 |
| OWL2Bench | 1,467 | 1,593 | 3,060 |
| **Total** | **4,461** | **4,587** | **9,048** |

The task totals are 6,032 binary questions (BQA), arranged as 3,016 TRUE/FALSE pairs, and 3,016 open-ended questions (OEQA). `task_id` and `semantic_key` identify a benchmark instance. NL, FS, and AR are aligned representations of that instance and are not separate questions.

## Benchmark design

Two task types are included:

- **BQA** uses an `ASK` query and a `TRUE` or `FALSE` gold label.
- **OEQA** uses a `SELECT` query and a set-valued gold answer.

Each instance carries three representations:

- **NL**: a natural-language question and verbalized ontology context.
- **FS**: the formal SPARQL query and serialized OWL/Turtle context.
- **AR**: an abstracted question and context in which ontology entities are systematically replaced.

The **1-hop/2-hop designation describes ontology-context extraction depth** around a root entity. It does not describe proof length. **Explanation complexity** is derived independently from Pellet explanations and is represented by minimum and maximum reasoning-tag length. The 20-tag taxonomy and coverage analysis are documented under `results/reasoning_coverage/`.

## Repository map

| Purpose | Location |
| --- | --- |
| Ready-to-use benchmark packages | `final_benchmark/*.zip` |
| Version, hashes, counts, and identity definition | `final_benchmark/manifest.json` |
| Per-instance complexity, reasoning tags, and negative-BQA proof links | `final_benchmark/reasoning_metadata.csv` |
| Source ontologies used for generation | `data/input/*.owl` |
| Benchmark generation | `scripts/run_final_benchmark_pipeline.py`, `final_benchmark/create_final_bench.py`, `src/` |
| LLM evaluation | `scripts/llm_pipeline/run_final_benchmark.py` |
| Manuscript and diagnostic analysis | `scripts/create_paper_results_table.py`, `scripts/create_explanation_complexity_analysis.py`, `analysis/` |
| Offline release checks and exports | `scripts/validate_release.py`, `scripts/export_huggingface.py`, `scripts/prepare_zenodo.py` |
| Tests | `tests/` |

## Using the released benchmark

You do **not** need to regenerate the benchmark or use an API to load and validate it.

### Validate the packages

Python 3.10 or newer is recommended. The validator uses only the standard library:

```bash
python scripts/validate_release.py
```

It checks ZIP and JSON readability, recorded SHA-256 hashes, all eight dataset-hop artifacts, required fields, unique identities, aligned representations, labels and gold answers, explanation links, complexity metadata, reasoning tags, and the frozen totals.

### Load one record without API spending

```python
import json
from zipfile import ZipFile

with ZipFile("final_benchmark/pizza_100.zip") as archive:
    groups = json.loads(archive.read("pizza_100_1hop.json"))

group = groups[0]
qa = group["QAs"][0]
print(qa["Task ID"])
print(qa["NL Question"])
print(qa["Answer"])
```

The grouped JSON schema avoids repeating large contexts for related questions. Group fields are `Task Type`, `Answer Type`, `Root Entity`, `OWL Context`, `NL Context`, `ABS Context`, and `QAs`. Each QA provides its task ID, three question/query views, original and abstract gold answers, and explanation fields. Negative BQA explanations are linked in `reasoning_metadata.csv` to the source positive entailment from which the false query was constructed.

### Evaluate a new model

Extract the packages first:

```python
from pathlib import Path
from zipfile import ZipFile

for path in Path("final_benchmark").glob("*.zip"):
    with ZipFile(path) as archive:
        archive.extractall("final_benchmark")
```

Then run one or all representation settings. Provider calls require the corresponding API credentials; loading and validation do not.

```bash
python scripts/llm_pipeline/run_final_benchmark.py \
  --dataset pizza_100 \
  --hop 1hop \
  --setting all \
  --models openai:gpt-4.1-mini \
  --max-workers 1 \
  --batch-size 5 \
  --checkpoint-frequency 25 \
  --silent-mode
```

Use `nl`, `sparql`, or `abs` for a single condition. The evaluation runner is resumable. Use `--limit-questions` and `--max-api-calls` for a deliberately bounded smoke test. Answer EM/F1 manuscript processing delegates through `scripts/llm_pipeline/sageqa_answer_metrics.py` to the exact manuscript-frozen SAGE-QA evaluator vendored under `scripts/llm_pipeline/vendor/`; it does not depend on a mutable sibling checkout or silently substitute Jaccard similarity. The frozen source commit, hashes, and MIT license are recorded in the vendor README.

### Tabular Hugging Face export

The canonical tabular 9,048-row v1.1.0 dataset view is published at
[huggingface.co/datasets/jloe2911/CORE-LLM-Bench](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench).
Each row is one unique question-hop instance, with NL, FS, and AR retained as
aligned columns.

To prepare either release profile locally without uploading it, install the
optional release dependency and run:

```bash
python -m pip install -r requirements-release.txt
python scripts/export_huggingface.py --profile full
python scripts/export_huggingface.py --profile public-safe
```

These commands describe the legacy v1.0.0 export path and write profile-specific Parquet exports, dataset cards, statistics,
manifests, and checksums under `release/huggingface/full/` and
`release/huggingface/public-safe/`. NL, FS, and AR remain columns in one row.
The full profile has 9,032 rows. The public-safe fallback has 5,272 rows and
excludes every Family/FHKB payload. Nothing is uploaded.

## Regenerating the benchmark from source ontologies

Regeneration is optional and separate from benchmark use. It requires Java 17, Maven, the Python requirements under `scripts/`, and API credentials for LLM-based ontology/SPARQL verbalization.

```bash
python scripts/run_final_benchmark_pipeline.py \
  --input-owl data/input/toy_example.owl \
  --dataset toy_example \
  --hops 1hop 2hop \
  --model gpt-4.1-mini
```

For a production source, replace the input and dataset, for example `data/input/family.owl` with `FamilyOWL`, `data/input/pizza_100.owl` with `pizza_100`, `data/input/pizza_250.owl` with `pizza_250`, or `data/input/OWL2DL-1.owl` with `OWL2Bench`. See `python scripts/run_final_benchmark_pipeline.py --help` for deterministic sampling, paired-subgraph, skip, and OWL2Bench size-cap options.

The Family input is a modified/adapted FHKB resource. When redistributing it or
Family-derived benchmark material, preserve the attribution, modification
notice, and CC BY-SA 3.0 terms documented in `NOTICE.md`. The provenance
comparison and a qualified local reconstruction path are recorded in
`docs/FAMILY_RECONSTRUCTION.md`.

The pipeline performs subgraph extraction, Pellet reasoning/explanation generation, SPARQL task creation, stratified sampling, abstraction, verbalization, and final JSON assembly. Do not overwrite the v1.0.0 packages when experimenting; use a separate output checkout or preserve and revalidate the hashes in `final_benchmark/manifest.json`.

## Reproducing manuscript analyses

The release-facing, API-free analyses are:

```bash
python analysis/reasoning_coverage.py
python scripts/validate_release.py --profile full
python scripts/validate_release.py --profile public-safe
```

Reasoning coverage outputs are in `results/reasoning_coverage/`. Version 1.0.0 defines 20 taxonomy tags and instantiates eight: D 9,032; H 3,278; I 2,324; R 1,660; M 1,020; N 168; S 55; T 2. Counts are not mutually exclusive.

The manuscript result and explanation-complexity scripts consume saved model predictions under the local `data/output/final_benchmark_llm_results/` hierarchy:

```bash
python scripts/create_paper_results_table.py
python scripts/create_explanation_complexity_analysis.py
python scripts/create_chapter4_interpretation_audit.py
```

Those scripts do not need new model calls when the saved predictions are present. Large intermediate generation artifacts and saved experimental responses are intentionally not part of the compact benchmark packages; archive them separately if full response-level reproduction is required.

## Preparing archival deposits

Prepare both local, checksum-indexed staging packages with:

```bash
python scripts/prepare_zenodo.py --profile full
python scripts/prepare_zenodo.py --profile public-safe
```

The commands create `release/zenodo/full/` and
`release/zenodo/public-safe/`, plus deterministic local ZIP archives. They do
not publish, reserve a DOI, or contact Zenodo. The full profile is the intended
canonical v1.0.0 release; public-safe is an optional reduced distribution that
excludes all Family/FHKB payload.

## Known limitations

- The four ontology families do not cover every OWL construct; only eight of 20 reasoning tags occur in v1.0.0.
- Ontology-context depth and proof complexity are related but distinct and must not be conflated.
- Natural-language and abstract verbalizations are generated and may contain stylistic or entity-rendering artifacts.
- BQA and OEQA totals are not balanced across datasets or hops.
- Full manuscript reproduction requires the separately retained saved predictions, while benchmark use does not.
- The source materials have mixed provenance. Pizza is CC BY 3.0, OWL2Bench is Apache-2.0, and Family/FHKB-derived material is CC BY-SA 3.0; see `NOTICE.md` for the required component-level attribution and modification notices.

### Citation

Users of CORE-LLM-Bench should currently cite the published CORE-LLM-Bench conference paper. A complete conference BibTeX record is not present in this repository, so no incomplete or inferred BibTeX is reproduced here; the conference publication remains the primary citation until the extended article is published.

An extended version of CORE-LLM-Bench is currently being prepared for submission to the Neurosymbolic AI journal special issue on Neurosymbolic Benchmark Papers.

## Version and license

The current benchmark version is `1.1.0`; the release tag is `v1.1.0`, and the
release date is 2026-09-25. The canonical tabular 9,048-row dataset view is available
on [Hugging Face](https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench).
The v1.1.0 Zenodo DOI is
[`10.5281/zenodo.22959957`](https://doi.org/10.5281/zenodo.22959957), pending
manual publication. The source release is available from the
[GitHub v1.1.0 release](https://github.com/jloe2911/CORE-LLM-Bench/releases/tag/v1.1.0).
The historical v1.0.0 GitHub, Hugging Face, and Zenodo artifacts remain unchanged.
See `VERSION`, `CITATION.cff`, and `final_benchmark/manifest.json`.

Licensing has three distinct layers:

- Software/code is licensed under the MIT License in `LICENSE`.
- Original CORE-LLM-Bench benchmark questions and metadata created by the authors are CC BY 4.0 where separable from source-derived content.
- Third-party ontology material and source-derived content retain source-specific terms: Pizza is CC BY 3.0, OWL2Bench is Apache-2.0, and Family/FHKB-derived material is CC BY-SA 3.0.

Do not treat the repository as uniformly MIT- or CC-BY-licensed. See `NOTICE.md` before redistribution.
