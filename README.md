# CORE-LLM-Bench

CORE-LLM-Bench is a neurosymbolic benchmark for evaluating language models on questions whose answers and explanations are grounded in OWL ontologies. Symbolic reasoning with Pellet provides entailed gold answers and proof metadata; models are evaluated through natural-language, formal-symbolic, and entity-abstracted views of the same underlying tasks.

Version 1.0 contains **9,032 unique question-hop instances** from four source datasets:

| Dataset | 1-hop | 2-hop | Total |
| --- | ---: | ---: | ---: |
| Family / FamilyOWL | 1,880 | 1,880 | 3,760 |
| Pizza 100 | 492 | 492 | 984 |
| Pizza 250 | 616 | 616 | 1,232 |
| OWL2Bench | 1,466 | 1,590 | 3,056 |
| **Total** | **4,454** | **4,578** | **9,032** |

The task totals are 5,999 binary questions (BQA) and 3,033 open-ended questions (OEQA). Dataset, hop, and `Task ID` jointly identify a benchmark instance. NL, FS, and AR are aligned representations of that instance and are not separate questions.

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

Use `nl`, `sparql`, or `abs` for a single condition. The evaluation runner is resumable. Use `--limit-questions` and `--max-api-calls` for a deliberately bounded smoke test. Answer EM/F1 manuscript processing delegates to the hash-pinned SAGE-QA evaluator through `scripts/llm_pipeline/sageqa_answer_metrics.py`; it does not silently substitute Jaccard similarity.

### Tabular Hugging Face export

Install the one optional release dependency, then export locally:

```bash
python -m pip install -r requirements-release.txt
python scripts/export_huggingface.py
```

This writes one Parquet row per unique question-hop instance to `release/huggingface/`, together with `dataset_info.json`. NL, FS, and AR remain columns in one row. The draft dataset card is `release/huggingface/README.md`. Nothing is uploaded.

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

The pipeline performs subgraph extraction, Pellet reasoning/explanation generation, SPARQL task creation, stratified sampling, abstraction, verbalization, and final JSON assembly. Do not overwrite the v1.0 packages when experimenting; use a separate output checkout or preserve and revalidate the hashes in `final_benchmark/manifest.json`.

## Reproducing manuscript analyses

The release-facing, API-free analyses are:

```bash
python analysis/reasoning_coverage.py
python scripts/validate_release.py
```

Reasoning coverage outputs are in `results/reasoning_coverage/`. Version 1.0 defines 20 taxonomy tags and instantiates eight: D 9,032; H 3,278; I 2,324; R 1,660; M 1,020; N 168; S 55; T 2. Counts are not mutually exclusive.

The manuscript result and explanation-complexity scripts consume saved model predictions under the local `data/output/final_benchmark_llm_results/` hierarchy:

```bash
python scripts/create_paper_results_table.py
python scripts/create_explanation_complexity_analysis.py
python scripts/create_chapter4_interpretation_audit.py
```

Those scripts do not need new model calls when the saved predictions are present. Large intermediate generation artifacts and saved experimental responses are intentionally not part of the compact benchmark packages; archive them separately if full response-level reproduction is required.

## Preparing archival deposits

`python scripts/prepare_zenodo.py` builds a local, checksum-indexed package under `release/zenodo/package/`. The draft metadata is `release/zenodo/zenodo_metadata.json`. It does not publish, reserve a DOI, or contact Zenodo.

## Known limitations

- The four ontology families do not cover every OWL construct; only eight of 20 reasoning tags occur in v1.0.
- Ontology-context depth and proof complexity are related but distinct and must not be conflated.
- Natural-language and abstract verbalizations are generated and may contain stylistic or entity-rendering artifacts.
- BQA and OEQA totals are not balanced across datasets or hops.
- Full manuscript reproduction requires the separately retained saved predictions, while benchmark use does not.
- The source materials have mixed provenance. Pizza declares CC BY 3.0 and OWL2Bench is Apache-2.0, but Family redistribution terms remain unresolved. Public release is blocked until that question is cleared; see `NOTICE.md`.

## Citation, version, and license

The benchmark version is `1.0.0`; see `VERSION`, `CITATION.cff`, and `final_benchmark/manifest.json`. Complete the pending paper metadata and release date before publication. Repository code is MIT licensed, but that license does not automatically cover third-party ontology content or all source-derived benchmark fields. See `LICENSE` and `NOTICE.md` before redistribution.
