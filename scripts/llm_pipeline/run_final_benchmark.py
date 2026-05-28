"""
Run LLM evaluation directly from final_benchmark/*.json files.

This is the credit-safe entry point for completed benchmark datasets:
1. Load a final benchmark JSON.
2. Flatten nested QAs into one row per question.
3. Run selected LLMs with inline context from the JSON.
4. Save checkpoints and final model outputs.
5. Run the metrics summarizer.
"""

import argparse
import ast
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from api_calls import (
    calculate_model_performance_summary,
    check_api_clients,
    get_model_params,
    run_llm_reasoning,
)


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)


SETTING_CONFIG = {
    "nl": {
        "question_column": "NL Question",
        "context_mode": "inline_nl",
        "prefix": "nl",
        "description": "Natural-language question with natural-language context",
    },
    "abs": {
        "question_column": "ABS Question",
        "context_mode": "inline_abs",
        "prefix": "abs",
        "description": "Abstracted natural-language question with abstracted context",
    },
    "sparql": {
        "question_column": "SPARQL Query",
        "context_mode": "inline_owl",
        "prefix": "sparql",
        "description": "SPARQL query with OWL context",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a complete, resumable LLM pipeline from a final benchmark JSON."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--benchmark-json",
        type=str,
        help="Path to a final benchmark JSON file.",
    )
    input_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name, e.g. toy_example, FamilyOWL, OWL2Bench.",
    )
    parser.add_argument(
        "--hop",
        choices=["1hop", "2hop"],
        help="Hop level used with --dataset.",
    )
    parser.add_argument(
        "--setting",
        choices=["nl", "abs", "sparql", "all"],
        default="all",
        help="Evaluation setting to run.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="data/output/final_benchmark_llm_results",
        help=(
            "Base directory where outputs, checkpoints, and metrics are stored. "
            "Runs are written to <base>/<benchmark>/<model>/<setting>/."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Models in provider:model_id format, e.g. "
            "openrouter:meta-llama/llama-4-maverick openrouter:deepseek/deepseek-chat"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=5)
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help="Maximum number of new API calls per setting for this run.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument("--silent-mode", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore previous LATEST_checkpoint.csv for this output directory.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only run LLM calls and write final CSV; do not run metrics.",
    )
    return parser.parse_args()


def parse_models(model_args):
    models = {}
    for item in model_args:
        if ":" not in item:
            raise ValueError(
                f"Invalid model specification '{item}'. Use provider:model_id"
            )

        provider, model_id = item.split(":", 1)
        provider = provider.strip().lower()
        model_id = model_id.strip()

        if provider not in {"openai", "openrouter", "deepseek"}:
            raise ValueError(
                f"Unsupported provider '{provider}' in '{item}'. "
                "Supported providers: openai, openrouter, deepseek"
            )

        display_name = f"{provider}_{model_id.replace('/', '_').replace('-', '_').replace('.', '_')}"
        models[display_name] = {"provider": provider, "model_id": model_id}

    return models


def slugify_path_part(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "unknown"


def model_output_slug(models_config):
    model_names = [slugify_path_part(name) for name in models_config.keys()]
    if len(model_names) == 1:
        return model_names[0]
    return "__".join(model_names)


def resolve_benchmark_path(args):
    if args.benchmark_json:
        return Path(args.benchmark_json)

    if not args.hop:
        raise ValueError("--hop is required when using --dataset")

    return Path("final_benchmark") / f"{args.dataset}_{args.hop}.json"


def load_final_benchmark(path):
    if not path.exists():
        raise FileNotFoundError(f"Final benchmark JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected final benchmark JSON to contain a top-level list")
    return data


def normalize_answer_type(value):
    text = str(value).strip().upper()
    if text in {"BINARY", "BIN", "TRUE/FALSE"}:
        return "BIN"
    if text in {"MC", "MULTI CHOICE", "MULTIPLE CHOICE", "MULTICHOICE"}:
        return "MC"
    return text or "BIN"


def flatten_final_benchmark(records):
    rows = []
    for group_idx, group in enumerate(records):
        qas = group.get("QAs", [])
        for qa_idx, qa in enumerate(qas):
            task_id = qa.get("Task ID", f"group{group_idx}_qa{qa_idx}")
            rows.append(
                {
                    "Benchmark Group Index": group_idx,
                    "QA Index": qa_idx,
                    "Task ID": task_id,
                    "Task Type": qa.get("Task Type", group.get("Task Type", "")),
                    "Answer Type": normalize_answer_type(
                        qa.get("Answer Type", group.get("Answer Type", "BIN"))
                    ),
                    "Root Entity": qa.get("Root Entity", group.get("Root Entity", "")),
                    "SPARQL Query": qa.get("SPARQL Query", ""),
                    "NL Question": qa.get("NL Question", ""),
                    "ABS Question": qa.get("ABS Question", ""),
                    "Question": qa.get("NL Question", ""),
                    "Answer": qa.get("Answer", ""),
                    "OWL Context": group.get("OWL Context", ""),
                    "NL Context": group.get("NL Context", ""),
                    "ABS Context": group.get("ABS Context", ""),
                    "Minimum Explanation": qa.get("Minimum Explanation"),
                    "Explanations": qa.get("Explanations"),
                    "Explanation Count": qa.get("Explanation Count"),
                    "Explanation Min": qa.get("Explanation Min"),
                    "Explanation Max": qa.get("Explanation Max"),
                }
            )

    return pd.DataFrame(rows)


def write_inline_explanations(df, path):
    def normalize_explanations(value):
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                return None
            try:
                parsed = ast.literal_eval(text)
                return parsed if isinstance(parsed, list) else None
            except (ValueError, SyntaxError):
                return None
        return None

    explanations = {}
    for idx, row in df.iterrows():
        row_explanations = normalize_explanations(row.get("Explanations"))
        if not row_explanations:
            continue

        key = f"{row.get('Task ID', 'task')}_{idx}"
        explanations[key] = {
            "sparqlQueries": [row.get("SPARQL Query", "")],
            "inferred": {},
            "explanations": row_explanations,
            "explanationCount": row.get("Explanation Count", 0),
            "size": {},
            "taskIds": [row.get("Task ID", "")],
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, default=str)
    return path


def load_resume_dataframe(output_dir, base_df, restart):
    latest_checkpoint = output_dir / "LATEST_checkpoint.csv"
    if not restart and latest_checkpoint.exists():
        print(f"Resuming from {latest_checkpoint}")
        return pd.read_csv(latest_checkpoint)
    return base_df.copy()


def save_final_outputs(
    results_df,
    logs,
    detailed_metrics,
    output_dir,
    setting_prefix,
    config,
    models_config,
):
    results_file = output_dir / f"{setting_prefix}_final_benchmark_results_FINAL.csv"
    logs_file = output_dir / f"{setting_prefix}_final_benchmark_logs_FINAL.csv"
    metrics_file = output_dir / f"{setting_prefix}_final_benchmark_metrics_FINAL.json"
    summary_file = output_dir / f"{setting_prefix}_experiment_summary.json"

    results_df.to_csv(results_file, index=False)
    pd.DataFrame(logs).to_csv(logs_file, index=False)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2, default=str)

    summary = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results_df),
        "total_models": len(models_config),
        "performance_summary": calculate_model_performance_summary(
            results_df, models_config
        ),
        "files_created": {
            "results": str(results_file),
            "logs": str(logs_file),
            "metrics": str(metrics_file),
            "summary": str(summary_file),
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return results_file


def run_metrics(results_file, explanations_file, output_dir, setting_prefix):
    from complete_evaluation import CompleteEvaluator

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    evaluator = CompleteEvaluator(
        csv_file=str(results_file),
        explanations_file=str(explanations_file),
        output_dir=str(metrics_dir),
    )
    evaluator.file_prefix = setting_prefix
    evaluator.evaluate_all_models()
    return metrics_dir


def run_setting(args, benchmark_path, base_df, setting, models_config):
    setting_config = SETTING_CONFIG[setting]
    dataset_stem = benchmark_path.stem
    llm_stem = model_output_slug(models_config)
    output_dir = Path(args.output_directory) / dataset_stem / llm_stem / setting
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_resume_dataframe(output_dir, base_df, args.restart)
    if args.limit_questions is not None:
        df = df.head(args.limit_questions).copy()

    question_column = setting_config["question_column"]
    missing_questions = df[question_column].astype(str).str.strip().eq("").sum()
    if missing_questions:
        print(
            f"Warning: {missing_questions} rows have empty {question_column}; they will still be checkpointed if called."
        )

    explanations_file = (
        output_dir / f"{setting_config['prefix']}_inline_explanations.json"
    )
    write_inline_explanations(df, explanations_file)

    config = {
        "benchmark_json": str(benchmark_path),
        "setting": setting,
        "description": setting_config["description"],
        "question_column": question_column,
        "context_mode": setting_config["context_mode"],
        "models": models_config,
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "checkpoint_frequency": args.checkpoint_frequency,
        "max_api_calls": args.max_api_calls,
        "limit_questions": args.limit_questions,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Running {setting.upper()} setting for {benchmark_path.name}")
    print(f"Questions: {len(df)} | Models: {', '.join(models_config.keys())}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    start_time = time.time()
    results_df, logs, detailed_metrics, _ = run_llm_reasoning(
        df,
        ontology_base_path="",
        models=models_config,
        model_params=get_model_params(models_config),
        context_mode=setting_config["context_mode"],
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        checkpoint_frequency=args.checkpoint_frequency,
        max_api_calls=args.max_api_calls,
        question_column=question_column,
        output_dir=output_dir,
        silent_mode=args.silent_mode,
    )
    config["runtime_seconds"] = round(time.time() - start_time, 3)

    results_file = save_final_outputs(
        results_df,
        logs,
        detailed_metrics,
        output_dir,
        setting_config["prefix"],
        config,
        models_config,
    )

    metrics_dir = None
    if not args.skip_metrics:
        metrics_dir = run_metrics(
            results_file,
            explanations_file,
            output_dir,
            setting_config["prefix"],
        )

    return {"results_file": results_file, "metrics_dir": metrics_dir}


def main():
    args = parse_args()
    benchmark_path = resolve_benchmark_path(args)
    models_config = parse_models(args.models)

    if not check_api_clients(models_config):
        print("Fix API client issues before running the benchmark pipeline.")
        return

    records = load_final_benchmark(benchmark_path)
    base_df = flatten_final_benchmark(records)
    if base_df.empty:
        raise ValueError(f"No QAs found in {benchmark_path}")

    settings = list(SETTING_CONFIG.keys()) if args.setting == "all" else [args.setting]
    outputs = {}
    for setting in settings:
        outputs[setting] = run_setting(
            args, benchmark_path, base_df, setting, models_config
        )

    print("\nCompleted final benchmark LLM pipeline.")
    for setting, paths in outputs.items():
        print(f"  {setting}: results={paths['results_file']}")
        if paths["metrics_dir"]:
            print(f"  {setting}: metrics={paths['metrics_dir']}")


if __name__ == "__main__":
    main()
