# CORE-LLM-Bench: Ontology Reasoning Benchmarks for Large Language Models

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CORE-LLM-Bench is a benchmark and generation pipeline for evaluating large language models on verifiable ontology reasoning tasks. It generates individual-centered OWL subgraphs, derives entailed facts with a symbolic reasoner, creates SPARQL `ASK` and `SELECT` questions, produces natural-language and abstracted variants, and evaluates LLM answers against gold labels.

The benchmark is designed to separate three capabilities:

- Reasoning over natural-language questions with verbalized ontology context.
- Reasoning over abstracted natural-language questions where semantic cues are removed.
- Reasoning over formal SPARQL queries with OWL/Turtle context.

Binary `ASK` questions are generated as positive/negative pairs before sampling. Each positive question is generated from an entailed triple, and its paired negative question preserves the subject and predicate while replacing the object with a non-entailed ontology term. Multiple-choice questions are represented as `SELECT` queries.

## Artifact At A Glance

This repository contains:

- `final_benchmark/*.json`: ready-to-use benchmark files with questions, gold answers, contexts, and explanation metadata.
- `data/input/toy_example.owl`: a small ontology for end-to-end artifact checks.
- `data/input/pizza_100.owl`: Pizza ontology input used to produce the checked-in 1-hop and 2-hop Pizza benchmark datasets.
- `scripts/run_final_benchmark_pipeline.py`: one-command benchmark generation from an OWL file.
- `scripts/llm_pipeline/run_final_benchmark.py`: resumable LLM evaluation from a final benchmark JSON.
- Java source in `src/main/java`: subgraph extraction, reasoning, explanation tagging, and SPARQL question generation.
- Python scripts in `scripts/`: sampling, abstraction, verbalization, LLM calls, and metric summaries.

Checked-in final benchmark files:

| File | Groups | Questions | Binary | Multiple-choice |
| --- | ---: | ---: | ---: | ---: |
| `final_benchmark/toy_example_1hop.json` | 13 | 48 | 38 | 10 |
| `final_benchmark/toy_example_2hop.json` | 18 | 77 | 66 | 11 |
| `final_benchmark/FamilyOWL_1hop.json` | 187 | 529 | 446 | 83 |
| `final_benchmark/FamilyOWL_2hop.json` | 121 | 543 | 442 | 101 |
| `final_benchmark/OWL2Bench_1hop.json` | 1,606 | 6,903 | 5,966 | 937 |
| `final_benchmark/pizza_100_1hop.json` | 146 | 1,159 | 1,107 | 52 |
| `final_benchmark/pizza_100_2hop.json` | 86 | 1,224 | 1,123 | 101 |

The checked-in benchmark files are the reproducible artifacts for evaluation. If you regenerate datasets after changing generator code, rerun the count check below and update this table.

## Artifact Evaluation Quickstart

For conference artifact review, the fastest no-API check is:

```powershell
git clone https://github.com/jloe2911/CORE-LLM-Bench.git
cd CORE-LLM-Bench
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r scripts/llm_pipeline/requirements.txt
pip install -r scripts/ontology_tools/requirements.txt
mvn package -DskipTests
python -c "import json; from pathlib import Path; [print(p.name + ':', len(d := json.loads(p.read_text(encoding='utf-8'))), 'groups,', sum(len(g.get('QAs', [])) for g in d), 'questions') for p in sorted(Path('final_benchmark').glob('*.json'))]"
```

This verifies that the repository installs, the Java components build, and the checked-in benchmark JSON files can be parsed without API keys.

To run a credit-safe LLM smoke test, configure `OPENAI_API_KEY` in `.env`, then run:

```powershell
python scripts/llm_pipeline/run_final_benchmark.py `
  --benchmark-json final_benchmark/toy_example_1hop.json `
  --setting nl `
  --models openai:gpt-4.1-mini `
  --limit-questions 5 `
  --max-workers 1 `
  --batch-size 5 `
  --checkpoint-frequency 5 `
  --silent-mode
```

Expected runtime is a few minutes for setup and seconds to minutes for the smoke test, depending on package cache state and API latency. Full benchmark regeneration and full LLM evaluation can take substantially longer and may incur API costs.

## Quick Start

### 1. Install prerequisites

- Java JDK 17 or newer
- Maven 3.6 or newer
- Python 3.8 or newer
- 8 GB RAM minimum, 16 GB recommended for larger ontologies

```powershell
git clone https://github.com/jloe2911/CORE-LLM-Bench.git
cd CORE-LLM-Bench
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r scripts/llm_pipeline/requirements.txt
pip install -r scripts/ontology_tools/requirements.txt
mvn package -DskipTests
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### 2. Validate the checked-in benchmark JSON files

This check requires no API key.

```bash
python -c "import json; from pathlib import Path; [print(p.name + ':', len(d := json.loads(p.read_text(encoding='utf-8'))), 'groups,', sum(len(g.get('QAs', [])) for g in d), 'questions') for p in sorted(Path('final_benchmark').glob('*.json'))]"
```

Expected output includes:

```text
FamilyOWL_1hop.json: 187 groups, 529 questions
FamilyOWL_2hop.json: 121 groups, 543 questions
OWL2Bench_1hop.json: 1606 groups, 6903 questions
pizza_100_1hop.json: 146 groups, 1159 questions
pizza_100_2hop.json: 86 groups, 1224 questions
toy_example_1hop.json: 13 groups, 48 questions
toy_example_2hop.json: 18 groups, 77 questions
```

### 3. Run a dry run of the generation pipeline

This verifies command wiring without making API calls.

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/toy_example.owl `
  --dataset toy_example `
  --hops 1hop 2hop `
  --model gpt-4.1-mini `
  --dry-run
```

## End-To-End Reproduction

Full benchmark generation uses an LLM for ontology verbalization and SPARQL-to-natural-language conversion. Configure the provider keys you need in `.env`:

```dotenv
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key
```

Then run the toy pipeline:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/toy_example.owl `
  --dataset toy_example `
  --hops 1hop 2hop `
  --model gpt-4.1-mini
```

The checked-in FamilyOWL and Pizza-100 benchmark subsets are already small enough for LLM evaluation. For large ontologies such as OWL2Bench, increase `--sampling-test-size` to reduce the sampled training split while preserving stratification. We used this adjusted setting for OWL2Bench to keep the benchmark subset for LLM experiments small enough to run reproducibly, since verbalization and model evaluation incur API cost. The default `0.95` keeps about 5% of eligible task groups; for OWL2Bench 1-hop, `0.9925` produces roughly 2,000 sampled questions.

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/OWL2Bench.owl `
  --dataset OWL2Bench `
  --hops 1hop `
  --model gpt-4.1-mini `
  --sampling-test-size 0.9925
```

To resume a previous run and avoid recomputing completed stages:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/toy_example.owl `
  --dataset toy_example `
  --hops 1hop 2hop `
  --model gpt-4.1-mini `
  --skip-existing `
  --no-maven-build
```

Expected final outputs:

- `final_benchmark/toy_example_1hop.json`
- `final_benchmark/toy_example_2hop.json`
- `data/output/toy_example/1hop/SPARQL_questions.csv`
- `data/output/toy_example/2hop/SPARQL_questions.csv`
- `data/output/toy_example/1hop/Explanations.json`
- `data/output/toy_example/2hop/Explanations.json`

If generator code changes, delete dependent outputs before using `--skip-existing`, especially `SPARQL_questions.csv`, `Explanations.json`, sampled question CSVs, and the corresponding `final_benchmark/*.json` files. The current generator limits repeated base queries to a small number of root contexts, so regenerated artifacts may differ from older files produced with global triple deduplication or unrestricted root-scoped deduplication.

## Pipeline Stages

The one-command pipeline runs these stages:

1. Build Java components with Maven.
2. Extract 1-hop and 2-hop individual-centered subgraphs while preserving the complete TBox.
3. Run Pellet-based symbolic reasoning.
4. Generate explanations and reasoning-complexity tags for positive entailments.
5. Generate SPARQL `ASK` and `SELECT` questions.
6. Stratify and sample questions while keeping positive/negative binary pairs together.
7. Create abstracted ontologies and abstracted questions.
8. Verbalize ontology contexts and SPARQL questions.
9. Assemble final benchmark JSON files.

Manual stage commands are useful for debugging. The examples below use the toy ontology.

### Subgraph Extraction

```powershell
mvn compile
mvn exec:java `
  "-Dexec.mainClass=SmallOntologyExtractor" `
  "-Dexec.args=data/input/toy_example.owl data/resources/toy_example_1hop/ data/resources/toy_example_2hop/"
```

### Core Benchmark Generation

```powershell
java -jar target/llm-orbench-1.0-SNAPSHOT.jar `
  data/resources/toy_example_1hop/ `
  data/output/toy_example/1hop

java -jar target/llm-orbench-1.0-SNAPSHOT.jar `
  data/resources/toy_example_2hop/ `
  data/output/toy_example/2hop
```

Generated files:

- `SPARQL_questions.csv`: complete SPARQL question set with metadata.
- `Explanations.json`: formal explanations and complexity tags for positive entailments.

Negative binary questions are non-entailment checks and therefore do not receive entailment explanations in the final benchmark JSON.

### Sampling, Abstraction, And Verbalization

```powershell
python scripts/llm_pipeline/stratified_sampling.py `
  --input_file data/output/toy_example/1hop/SPARQL_questions.csv `
  --output_file data/output/toy_example/1hop/SPARQL_questions_sampling.csv

python scripts/ontology_tools/abstraction/Usage.py `
  --input-directory data/resources/toy_example_1hop `
  --output-directory data/output/abstracted_ontologies/toy_example_1hop/

python scripts/llm_pipeline/verbalize_ontologies.py `
  --input-dir data/resources/toy_example_1hop/ `
  --output-dir data/output/verbalized_ontologies/toy_example_1hop/ `
  --file-pattern "*.ttl"

python scripts/llm_pipeline/sparql_to_nl.py `
  --input-csv data/output/toy_example/1hop/SPARQL_questions_sampling.csv `
  --output-directory data/output/toy_example/1hop/ `
  --output-file SPARQL_questions_sampling_nl.csv `
  --model gpt-4.1-mini
```

For large inputs, prefer adjusting `--test-size` over using a fixed row cap. This keeps the experimental dataset smaller for LLM evaluation while still sampling through the stratified split. For example, `--test-size 0.9925` produces about 2,000 sampled OWL2Bench 1-hop questions. Use `--max-rows` only as a final budget guard; it preserves complete task groups, including binary positive/negative pairs, but it is less statistically clean than controlling the stratified split size directly.

Repeat the same commands with `2hop` paths for 2-hop data.

## LLM Evaluation

The preferred evaluation entry point reads directly from `final_benchmark/*.json`, writes checkpoints, and resumes completed model responses.

```powershell
python scripts/llm_pipeline/run_final_benchmark.py `
  --dataset toy_example `
  --hop 1hop `
  --setting all `
  --models openai:gpt-4.1-mini `
  --max-workers 1 `
  --batch-size 5 `
  --checkpoint-frequency 5 `
  --max-api-calls 20 `
  --silent-mode
```

For a tiny smoke test:

```powershell
python scripts/llm_pipeline/run_final_benchmark.py `
  --benchmark-json final_benchmark/toy_example_1hop.json `
  --setting nl `
  --models openai:gpt-4.1-mini `
  --limit-questions 5 `
  --max-workers 1 `
  --batch-size 5 `
  --checkpoint-frequency 5 `
  --silent-mode
```

Supported settings:

- `nl`: natural-language question with natural-language ontology context.
- `abs`: abstracted natural-language question with abstracted context.
- `sparql`: SPARQL query with OWL context.
- `all`: run all three settings.

Outputs are written under:

```text
data/output/final_benchmark_llm_results/<benchmark>/<llm>/<setting>/
```

For example, abstracted results for `toy_example_1hop` with `openai:gpt-4.1-mini` are written to `data/output/final_benchmark_llm_results/toy_example_1hop/openai_gpt_4_1_mini/abs/`. Each setting directory contains a `LATEST_checkpoint.csv`, final model outputs, inline explanation metadata, and metric summaries. Re-running the same command resumes from the latest checkpoint. Use `--restart` only when intentionally ignoring previous outputs.

## Final Benchmark JSON Schema

Each final benchmark file is a list of benchmark groups. A group contains ontology context and one or more related QAs:

- `Task Type`: membership or property assertion.
- `Answer Type`: `BIN` for binary questions or `MC` for multiple choice.
- `Root Entity`: individual around which the subgraph was extracted.
- `OWL Context`: original OWL/RDF context.
- `NL Context`: verbalized ontology context.
- `ABS Context`: abstracted verbalized context.
- `QAs`: question-answer records.

Each QA record contains:

- `Task ID`
- `SPARQL Query`
- `NL Question`
- `ABS Question`
- `Answer`
- `Minimum Explanation`
- `Explanations`
- `Explanation Count`
- `Explanation Min`
- `Explanation Max`

For binary tasks, `Answer` is `TRUE` or `FALSE`. For multiple-choice tasks, `Answer` is a semicolon-separated list of gold entities or classes.

## Output File Guide

- `SPARQL_questions.csv`: complete generated query dataset.
- `SPARQL_questions_sampling.csv`: stratified sample used for LLM evaluation.
- `SPARQL_questions_sampling_nl.csv`: natural-language sampled questions.
- `SPARQL_questions_sampling_abs.csv`: abstracted natural-language sampled questions.
- `Explanations.json`: reasoning explanations with complexity tags.
- `final_benchmark/*.json`: final benchmark files.
- `LATEST_checkpoint.csv`: resumable LLM response ledger.
- `*_FINAL.csv`: final LLM outputs.
- `metrics/*_key_findings_summary.json`: setting-level performance summaries.

## Notes

- The checked-in final benchmark JSON files can be inspected and evaluated without regenerating the dataset.
- API keys are only required for LLM-based verbalization, SPARQL-to-NL conversion, and LLM evaluation.
- Use `--max-api-calls`, `--limit-questions`, and `--max-workers 1` for credit-safe smoke tests.
- Large ontologies can require substantially more memory and time than the toy example.
- The toy paths are intended as templates for running the same pipeline on another OWL ontology.
- Do not submit local `.env`, virtual environments, logs, IDE folders, or generated intermediate output directories as part of an artifact archive.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
