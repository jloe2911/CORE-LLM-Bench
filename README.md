# CORE-LLM-Bench

CORE-LLM-Bench is a benchmark and generation pipeline for evaluating large language models on verifiable ontology reasoning tasks. It extracts individual-centered OWL subgraphs, derives entailed facts with Pellet, generates SPARQL `ASK` and `SELECT` questions, creates natural-language and abstracted variants, and evaluates LLM answers against symbolic gold labels.

The benchmark compares three settings:

- `NL`: natural-language questions with verbalized ontology context; CLI setting `nl`.
- `AR`: abstracted natural-language questions with abstracted context; CLI setting `abs`.
- `FS`: formal SPARQL queries with OWL/Turtle context; CLI setting `sparql`.

For each subject-predicate reasoning instance, the generator creates one open-ended `SELECT` question, one positive binary `ASK` question, and, one negative binary `ASK` question. The final benchmark subsets are then produced with stratified sampling over ontology size and reasoning-complexity bins.

## Repository Contents

- `final_benchmark/*.json`: ready-to-use benchmark files for evaluation, when unpacked.
- `final_benchmark/FamilyOWL.zip`: compressed FamilyOWL final benchmark JSONs (`FamilyOWL_1hop.json` and `FamilyOWL_2hop.json`) for distribution when the large JSON files are not unpacked.
- `final_benchmark/OWL2Bench.zip`: compressed OWL2Bench final benchmark JSONs (`OWL2Bench_1hop.json` and `OWL2Bench_2hop.json`) for distribution when the large JSON files are not unpacked.
- `data/input/*.owl`: source ontologies used by the generation pipeline.
- `scripts/run_final_benchmark_pipeline.py`: one-command benchmark generation.
- `scripts/llm_pipeline/run_final_benchmark.py`: resumable LLM evaluation from final benchmark JSON files.
- `scripts/create_paper_results_table.py`: combines final LLM metrics into paper-ready CSV/LaTeX tables.
- `src/main/java`: Java extraction, reasoning, explanation, and SPARQL generation code.
- `scripts/ontology_tools`: ontology abstraction utilities.

## Reproducing Manuscript Results

There are two levels of reproduction:

1. **Evaluate the released benchmark artifact.** Use the checked-in or zipped files in `final_benchmark/`, run the LLM evaluation script on each dataset/hop/setting/model combination, and rebuild the paper results table.
2. **Regenerate the benchmark from OWL sources.** Run the generation pipeline from `data/input/*.owl`, then rerun the LLM evaluations and table script. This is slower and uses API calls for verbalization.

The main manuscript workflow is:

1. Follow [Quickstart](#quickstart) to install dependencies, compile Java code, extract benchmark zips, and verify that the benchmark JSON files load.
2. Use [Regenerating Benchmarks](#regenerating-benchmarks) only if you want to rebuild `final_benchmark/*.json` from the OWL files.
3. Run [LLM Evaluation](#llm-evaluation) for each reported benchmark: `FamilyOWL`, `OWL2Bench`, and `toy_example`; each hop: `1hop` and `2hop`; each setting: `nl`, `abs`, and `sparql`; and each manuscript model.
4. Run `python scripts/create_paper_results_table.py` to rebuild `data/output/final_benchmark_llm_results/combined_1hop_2hop_results_table.csv` and `.tex`.

The manuscript model identifiers used by the evaluation CLI are:

- `openai:gpt-5-mini-2025-08-07`
- `openrouter:google/gemini-2.5-flash-lite`
- `openrouter:qwen/qwen3-30b-a3b-instruct-2507`

The 5-question toy command in [Credit-Safe LLM Smoke Test](#credit-safe-llm-smoke-test) is only a low-cost sanity check. It does not reproduce the manuscript table.

## Final Benchmark Sizes

Current generated question pools and sampled benchmark subsets contain:

| Ontology | Hop | BQ (pos/neg) | OEQ | Sampled BQ (pos/neg) | Sampled OEQ | Sampled Questions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FamilyOWL` | 1-hop | 5012 (2506/2506) | 2506 | 1272 (603/669) | 608 | 1880 |
| `FamilyOWL` | 2-hop | 5012 (2506/2506) | 2506 | 1272 (602/670) | 608 | 1880 |
| `OWL2Bench` | 1-hop | 3912 (1956/1956) | 1956 | 957 (484/473) | 509 | 1466 |
| `OWL2Bench` | 2-hop | 4246 (2123/2123) | 2123 | 1044 (508/536) | 546 | 1590 |
| `toy_example` | 1-hop | 98 (49/49) | 49 | 25 (12/13) | 11 | 36 |
| `toy_example` | 2-hop | 98 (49/49) | 49 | 20 (11/9) | 16 | 36 |

`BQ` means binary questions and `OEQ` means open-ended `SELECT` questions. The positive/negative split is shown for binary questions as `TRUE`/`FALSE`.

The 1-hop and 2-hop OWL2Bench runs are paired at the root-individual subgraph
level, but the generator does not force identical question counts across hops.
FamilyOWL happens to yield equal 1-hop and 2-hop eligible question pools. For
OWL2Bench, the 2-hop extraction exposes additional inferred `isMemberOf`
property assertions for 167 root-focused subject-predicate groups; each group
contributes one positive `ASK`, one negative `ASK`, and one `SELECT` question,
adding 501 raw questions before independent stratified sampling.

The final benchmark JSON files currently have the following group counts, file sizes, and symbolic-context sizes after any benchmark zips are extracted. `OWL Context` is serialized as Turtle for the FS setting.

| Ontology | Hop | Groups | Questions | JSON size | Avg. OWL context chars | Max OWL context chars |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FamilyOWL` | 1-hop | 1086 | 1880 | 117.0 MB | 57556 | 58290 |
| `FamilyOWL` | 2-hop | 1086 | 1880 | 293.1 MB | 110423 | 110441 |
| `OWL2Bench` | 1-hop | 866 | 1466 | 29.5 MB | 30775 | 55414 |
| `OWL2Bench` | 2-hop | 869 | 1590 | 73.0 MB | 51225 | 123280 |
| `toy_example` | 1-hop | 25 | 36 | 0.2 MB | 6989 | 7424 |
| `toy_example` | 2-hop | 26 | 36 | 0.3 MB | 7602 | 8053 |

To recompute these counts:

```powershell
python -c "import csv; from pathlib import Path; names=['FamilyOWL','OWL2Bench','toy_example']; f=lambda rows: (sum(r.get('Answer Type','').upper()=='BIN' for r in rows),sum(r.get('Answer Type','').upper()=='MC' for r in rows),sum(r.get('Answer Type','').upper()=='BIN' and str(r.get('Answer','')).upper()=='TRUE' for r in rows),sum(r.get('Answer Type','').upper()=='BIN' and str(r.get('Answer','')).upper()=='FALSE' for r in rows)); [print(n,h,'raw BQ/OEQ/pos/neg=',f(list(csv.DictReader(open(Path('data/output')/n/h/'SPARQL_questions.csv',encoding='utf-8-sig')))),'sampled BQ/OEQ/pos/neg=',f(list(csv.DictReader(open(Path('data/output')/n/h/'SPARQL_questions_sampling.csv',encoding='utf-8-sig'))))) for n in names for h in ['1hop','2hop']]"
```

To recompute the JSON file statistics:

```powershell
python -c "import json; from pathlib import Path; pairs=[('FamilyOWL','1hop'),('FamilyOWL','2hop'),('OWL2Bench','1hop'),('OWL2Bench','2hop'),('toy_example','1hop'),('toy_example','2hop')]; [print(n,h,'groups=',len(d:=json.loads((p:=Path('final_benchmark')/f'{n}_{h}.json').read_text(encoding='utf-8'))),'questions=',sum(len(g.get('QAs',[])) for g in d),'size_mb=',round(p.stat().st_size/1024/1024,1),'avg_owl_chars=',round(sum(len(str(g.get('OWL Context',''))) for g in d)/len(d)),'max_owl_chars=',max(len(str(g.get('OWL Context',''))) for g in d)) for n,h in pairs]"
```

During FS evaluation, large symbolic contexts are additionally trimmed query-aware at prompt time: SPARQL terms are prioritized before any size limit is applied.


## Quickstart

These steps validate the benchmark files without using any API credits. The Java components target Java 17; Maven and Python 3.10 or newer are recommended.

```powershell
git clone <repository-url>
cd CORE-LLM-Bench
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r scripts/llm_pipeline/requirements.txt
pip install -r scripts/ontology_tools/requirements.txt
mvn -q -DskipTests compile
Get-ChildItem final_benchmark -Filter *.zip | ForEach-Object { Expand-Archive -Path $_.FullName -DestinationPath final_benchmark -Force }
python -c "import json; from pathlib import Path; [print(p.name, len(json.loads(p.read_text(encoding='utf-8'))), 'groups') for p in sorted(Path('final_benchmark').glob('*.json'))]"
```

On macOS/Linux, activate the environment and extract benchmark zips with:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
for path in Path("final_benchmark").glob("*.zip"):
    with ZipFile(path) as archive:
        archive.extractall("final_benchmark")
PY
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

Each setting directory contains resumable `LATEST_checkpoint*` files and metric summaries. Add `--write-final-artifacts` when you also want the legacy `FINAL` CSV/log/detailed-metrics files used for paper-table inspection. Use `--restart` only when intentionally ignoring previous checkpoints.

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

OWL2Bench reproducibility command used for the reported 2-hop benchmark:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/OWL2DL-1.owl `
  --dataset OWL2Bench `
  --hops 2hop `
  --max-2hop-subgraphs 500 `
  --max-subgraph-file-size-mb 0.5 `
  --no-maven-build
```

This deterministically samples 500 paired root-individual subgraphs from the
1-hop/2-hop filename intersection after excluding TTL files larger than 0.5 MB.
The default seed is `13`. To regenerate the matching 1-hop OWL2Bench benchmark
after the 2-hop run, omit the subgraph cap; the pipeline reuses the paired
sample manifest:

```powershell
python scripts/run_final_benchmark_pipeline.py `
  --input-owl data/input/OWL2DL-1.owl `
  --dataset OWL2Bench `
  --hops 1hop `
  --no-maven-build
```

Useful defaults:

- `--sampling-test-size 0.75` is the default; the retained benchmark sample is the remaining 25%.
- `--focus-root-individual-only` is the default; use `--no-focus-root-individual-only` to process all individuals in each extracted ontology.
- Use `--max-2hop-subgraphs N` to deterministically pre-sample extracted 2-hop TTL subgraphs before Java reasoning/explanation generation. The pipeline samples matching TTL filenames so the same root individuals can be used for both hops. The paired sample is copied to `data/resources/<dataset>_<hop>_paired_sampled_N_seed_<seed>` and reused by downstream stages. If one hop is run later without a subgraph cap, it reuses the newest existing paired sample manifest from the other hop.
- Use `--subgraph-sample-seed N` to change the deterministic pre-sampling seed.
- Use `--max-subgraph-file-size-mb N` with OWL2Bench 2-hop if explanation extraction stalls on very large sampled TTLs.
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

To reproduce the manuscript-scale evaluation from the released benchmark JSON files, run all reported datasets, hops, settings, and models:

```powershell
$datasets = @("FamilyOWL", "OWL2Bench", "toy_example")
$hops = @("1hop", "2hop")
$models = @(
  "openai:gpt-5-mini-2025-08-07",
  "openrouter:google/gemini-2.5-flash-lite",
  "openrouter:qwen/qwen3-30b-a3b-instruct-2507"
)

foreach ($dataset in $datasets) {
  foreach ($hop in $hops) {
    python scripts/llm_pipeline/run_final_benchmark.py `
      --dataset $dataset `
      --hop $hop `
      --setting all `
      --models $models `
      --max-workers 1 `
      --batch-size 5 `
      --checkpoint-frequency 25 `
      --silent-mode `
      --write-final-artifacts
  }
}
```

Supported model identifiers use `provider:model`, for example:

- `openai:gpt-4.1-mini`
- `openai:gpt-5-mini-2025-08-07`
- `openrouter:google/gemini-2.5-flash-lite`
- `openrouter:qwen/qwen3-30b-a3b-instruct-2507`

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

## Notes

- Final benchmark JSON files, whether checked in directly or distributed through `final_benchmark/*.zip`, can be inspected and evaluated without regenerating datasets.
- API keys are required only for LLM-based verbalization, SPARQL-to-NL conversion, and LLM evaluation.
- Use `--limit-questions`, `--max-api-calls`, and `--max-workers 1` for small, credit-safe runs.
- Large ontology regeneration can take substantial time and memory.
- Do not include local `.env`, virtual environments, logs, IDE folders, or generated intermediate output directories in a submission package.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
