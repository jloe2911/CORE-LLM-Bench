# CORE-LLM-Bench

CORE-LLM-Bench is a benchmark and generation pipeline for evaluating large language models on verifiable ontology reasoning tasks. It extracts individual-centered OWL subgraphs, derives entailed facts with Pellet, generates SPARQL `ASK` and `SELECT` questions, creates natural-language and abstracted variants, and evaluates LLM answers against symbolic gold labels.

The benchmark compares three settings:

- `NL`: natural-language questions with verbalized ontology context.
- `AR`: abstracted natural-language questions with abstracted context.
- `FS`: formal SPARQL queries with OWL/Turtle context.

For each subject-predicate reasoning instance, the generator creates one open-ended `SELECT` question, one positive binary `ASK` question, and, when a plausible counterexample exists, one negative binary `ASK` question. The final benchmark subsets are then produced with stratified sampling over ontology size and reasoning-complexity bins.

## Repository Contents

- `final_benchmark/*.json`: ready-to-use benchmark files for evaluation.
- `data/input/*.owl`: source ontologies used by the generation pipeline.
- `scripts/run_final_benchmark_pipeline.py`: one-command benchmark generation.
- `scripts/llm_pipeline/run_final_benchmark.py`: resumable LLM evaluation from final benchmark JSON files.
- `scripts/create_paper_results_table.py`: combines final LLM metrics into paper-ready CSV/LaTeX tables.
- `src/main/java`: Java extraction, reasoning, explanation, and SPARQL generation code.
- `scripts/ontology_tools`: ontology abstraction utilities.

## Final Benchmark Sizes

Current generated and checked-in sampled benchmark files contain:

| Ontology | Hop | BQ (pos/neg) | OEQ | Sampled BQ (pos/neg) | Sampled OEQ | Sampled Questions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FamilyOWL` | 1-hop | 5012 (2506/2506) | 2506 | 1272 (603/669) | 608 | 1880 |
| `FamilyOWL` | 2-hop | 5012 (2506/2506) | 2506 | 1272 (602/670) | 608 | 1880 |
| `toy_example` | 1-hop | 98 (49/49) | 49 | 25 (12/13) | 11 | 36 |
| `toy_example` | 2-hop | 98 (49/49) | 49 | 20 (11/9) | 16 | 36 |

`BQ` means binary questions and `OEQ` means open-ended `SELECT` questions. The positive/negative split is shown for binary questions as `TRUE`/`FALSE`.

The checked-in final benchmark JSON files currently have the following group counts, file sizes, and symbolic-context sizes. `OWL Context` is serialized as Turtle for the FS setting.

| Ontology | Hop | Groups | Questions | JSON size | Avg. OWL context chars | Max OWL context chars |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FamilyOWL` | 1-hop | 1086 | 1880 | 116.6 MB | 57556 | 58290 |
| `FamilyOWL` | 2-hop | 1086 | 1880 | 292.7 MB | 110423 | 110441 |
| `toy_example` | 1-hop | 25 | 36 | 0.7 MB | 26003 | 27302 |
| `toy_example` | 2-hop | 26 | 36 | 0.8 MB | 27734 | 28997 |

To recompute these counts:

```powershell
python -c "import csv,json; from pathlib import Path; names=['FamilyOWL','toy_example']; f=lambda rows: (sum(r.get('Answer Type','').upper()=='BIN' for r in rows),sum(r.get('Answer Type','').upper()=='MC' for r in rows),sum(r.get('Answer Type','').upper()=='BIN' and str(r.get('Answer','')).upper()=='TRUE' for r in rows),sum(r.get('Answer Type','').upper()=='BIN' and str(r.get('Answer','')).upper()=='FALSE' for r in rows)); flat=lambda data: [dict(qa, **{'Answer Type': qa.get('Answer Type', g.get('Answer Type',''))}) for g in data for qa in g.get('QAs',[])]; [print(n,h,'raw BQ/OEQ/pos/neg=',f(list(csv.DictReader(open(Path('data/output')/n/h/'SPARQL_questions.csv',encoding='utf-8-sig')))),'sampled BQ/OEQ/pos/neg=',f(flat(json.loads((Path('final_benchmark')/f'{n}_{h}.json').read_text(encoding='utf-8'))))) for n in names for h in ['1hop','2hop']]"
```

To recompute the JSON file statistics:

```powershell
python -c "import json; from pathlib import Path; names=['FamilyOWL','toy_example']; [print(n,h,'groups=',len(d:=json.loads((p:=Path('final_benchmark')/f'{n}_{h}.json').read_text(encoding='utf-8'))),'questions=',sum(len(g.get('QAs',[])) for g in d),'size_mb=',round(p.stat().st_size/1024/1024,1),'avg_owl_chars=',round(sum(len(str(g.get('OWL Context',''))) for g in d)/len(d)),'max_owl_chars=',max(len(str(g.get('OWL Context',''))) for g in d)) for n in names for h in ['1hop','2hop']]"
```

During FS evaluation, large symbolic contexts are additionally trimmed query-aware at prompt time: SPARQL terms are prioritized before any size limit is applied.


## Quickstart

These steps validate the checked-in benchmark files without using any API credits. If the repository has already been unpacked, start from the `cd CORE-LLM-Bench` step.

```powershell
git clone <repository-url>
cd CORE-LLM-Bench
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r scripts/llm_pipeline/requirements.txt
pip install -r scripts/ontology_tools/requirements.txt
mvn -q -DskipTests compile
python -c "import json; from pathlib import Path; [print(p.name, len(json.loads(p.read_text(encoding='utf-8'))), 'groups') for p in sorted(Path('final_benchmark').glob('*.json'))]"
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Credit-Safe LLM Smoke Test

Configure an API key in `.env`, for example:

```dotenv
OPENAI_API_KEY=your_openai_key
```

Then run a small evaluation:

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

Results are written under:

```text
data/output/final_benchmark_llm_results/<benchmark>/<model>/<setting>/
```

Each setting directory contains resumable `LATEST_checkpoint*` files, final model outputs, inline explanation metadata, and metric summaries. Use `--restart` only when intentionally ignoring previous checkpoints.

## Regenerating Benchmarks

The end-to-end pipeline uses an LLM for ontology/context verbalization and SPARQL-to-natural-language conversion. Configure the provider key for the model you choose.

Toy example:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/toy_example.owl `
  --dataset toy_example `
  --hops 1hop 2hop `
  --model gpt-4.1-mini
```

FamilyOWL example:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/family.owl `
  --dataset FamilyOWL `
  --hops 1hop 2hop `
  --model gpt-4.1-mini `
  --no-maven-build
```

Useful defaults:

- `--sampling-test-size 0.75` is the default; the retained benchmark sample is the remaining 25%.
- `--focus-root-individual-only` is the default; use `--no-focus-root-individual-only` to process all individuals in each extracted ontology.
- Use `--skip-existing` only when dependent outputs are already valid for the current generator code.
- Use `--no-explanations` to reuse an existing valid `Explanations.json` while regenerating SPARQL questions.

If generator code changes, regenerate dependent outputs rather than using stale files:

- `SPARQL_questions.csv`
- `SPARQL_questions_sampling.csv`
- `SPARQL_questions_sampling_nl.csv`
- `SPARQL_questions_sampling_abs.csv`
- `final_benchmark/*.json`

## Pipeline Stages

The one-command pipeline performs:

1. Java build, unless `--no-maven-build` is passed.
2. 1-hop and 2-hop individual-centered subgraph extraction.
3. Pellet reasoning and explanation extraction.
4. SPARQL `ASK`/`SELECT` question generation.
5. Stratified sampling by ontology size and reasoning complexity.
6. Ontology abstraction and abstract question creation.
7. Ontology and SPARQL verbalization.
8. Final benchmark JSON assembly.

Manual debugging entry points:

```powershell
java -jar target/llm-orbench-1.0-SNAPSHOT.jar `
  data/resources/toy_example_1hop `
  data/output/toy_example/1hop

python scripts/llm_pipeline/stratified_sampling.py `
  --input_file data/output/toy_example/1hop/SPARQL_questions.csv `
  --output_file data/output/toy_example/1hop/SPARQL_questions_sampling.csv

python final_benchmark/create_final_bench.py --dataset toy_example --hop 1hop
```

## LLM Evaluation

Run all three settings for a benchmark:

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

Supported model identifiers use `provider:model`, for example:

- `openai:gpt-4.1-mini`
- `openai:gpt-5-mini-2025-08-07`
- `openrouter:google/gemini-2.5-flash-lite`

After evaluation, rebuild the paper table:

```powershell
python scripts/create_paper_results_table.py
```

## Final Benchmark JSON Schema

Each final benchmark file is a list of groups. Each group contains ontology context and one or more related QA records.

Group-level fields include:

- `Task Type`: membership or property assertion.
- `Answer Type`: `BIN` for binary questions or `MC` for open-ended `SELECT` questions.
- `Root Entity`
- `OWL Context`
- `NL Context`
- `ABS Context`
- `QAs`

Each QA contains:

- `Task ID`
- `SPARQL Query`
- `NL Question`
- `ABS Question`
- `ABS Answer`
- `Answer`
- `Minimum Explanation`
- `Explanations`
- `Explanation Count`
- `Explanation Min`
- `Explanation Max`

For binary tasks, `Answer` is `TRUE` or `FALSE`. For open-ended tasks, `Answer` is a semicolon-separated set of gold entities or classes.

## Notes For Conference Evaluation

- Checked-in `final_benchmark/*.json` files can be inspected and evaluated without regenerating datasets.
- API keys are required only for LLM-based verbalization, SPARQL-to-NL conversion, and LLM evaluation.
- Use `--limit-questions`, `--max-api-calls`, and `--max-workers 1` for small, credit-safe runs.
- Large ontology regeneration can take substantial time and memory.
- Do not include local `.env`, virtual environments, logs, IDE folders, or generated intermediate output directories in a submission package.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
