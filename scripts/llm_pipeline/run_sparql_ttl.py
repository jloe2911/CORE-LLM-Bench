"""
SPARQL + TTL execution: Tests LLM ability to understand formal SPARQL queries with TTL ontology context
Enhanced version with improved memory management for heavy ontologies and thousands of questions
"""

import pandas as pd
import os
import time
import json
import gc
import psutil
from pathlib import Path
from datetime import datetime
from api_calls import (
    run_llm_reasoning,
    calculate_model_performance_summary,
    log_models_metadata,
    check_api_clients,
    openai_client,
    deepseek_client,
    openrouter_client,
)

# Navigate to project root if necessary
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)

print(f"Working directory: {os.getcwd()}")


def monitor_memory():
    """Monitor system memory usage"""
    return psutil.virtual_memory().percent


def force_cleanup():
    """Force garbage collection"""
    gc.collect()
    time.sleep(1)


# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(
    f"output/llm_results/OWL2Bench_2hop/sparql_ttl_experiment_{timestamp}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"📂 Output directory: {output_dir}")

# Configuration - Optimized for heavy processing
MODELS_CONFIG = {
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "deepseek-chat": "deepseek-chat",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
}

CONFIG = {
    "experiment_type": "sparql_ttl",
    "description": "SPARQL queries with TTL ontology context - Memory Optimized",
    "questions_csv": "output/OWL2Bench/2hop/SPARQL_questions_sampling2.csv",
    "ttl_ontology_dir": "src/main/resources/OWL2Bench_2hop",
    "context_mode": "ttl",
    "question_column": "SPARQL Query",
    "models_used": MODELS_CONFIG,
    "max_workers": 4,  # Reduced for better memory management
    "batch_size": 15,  # Smaller batches for heavy ontologies
    "checkpoint_frequency": 50,  # More frequent saves
    "silent_mode": False,
    "memory_threshold": 85,  # Memory threshold for cleanup
}


def print_experiment_header():
    """Print a nice experiment header"""
    print("\n" + "=" * 80)
    print("🚀 SPARQL + TTL EXPERIMENT (MEMORY OPTIMIZED)")
    print("=" * 80)
    print(f"📋 Experiment: {CONFIG['description']}")
    print(f"🤖 Models: {', '.join(CONFIG['models_used'].keys())}")
    print(f"🔧 Max Workers: {CONFIG['max_workers']}")
    print(f"📦 Batch Size: {CONFIG['batch_size']}")
    print(f"💾 Checkpoint: Every {CONFIG['checkpoint_frequency']} questions")
    print(f"🔇 Silent Mode: {'ON' if CONFIG['silent_mode'] else 'OFF'}")
    print(f"🧠 Memory Threshold: {CONFIG['memory_threshold']}%")
    print(f"📂 Output: {output_dir}")
    print("=" * 80)


def validate_setup():
    """Validate experiment setup with memory monitoring"""
    print("🔍 Validating setup...")
    print(f"🧠 Initial memory usage: {monitor_memory():.1f}%")

    # Load questions
    try:
        df = pd.read_csv(CONFIG["questions_csv"])
        print(f"✅ Loaded {len(df)} questions from {CONFIG['questions_csv']}")
    except FileNotFoundError:
        print(f"❌ Error: Questions CSV not found at {CONFIG['questions_csv']}")
        return None

    # Check TTL directory
    ttl_dir = Path(CONFIG["ttl_ontology_dir"])
    if not ttl_dir.exists():
        print(f"❌ Error: TTL directory does not exist: {ttl_dir}")
        return None

    # Filter questions based on available TTL files
    available_ttl_files = {p.stem for p in ttl_dir.glob("*.ttl")}
    original_count = len(df)
    df = df[df["Root Entity"].isin(available_ttl_files)].copy()

    print(f"🔍 Found {len(available_ttl_files)} TTL files")
    print(f"📢 Filtered dataset to {len(df)} questions (from {original_count})")

    if df.empty:
        print(
            "❌ Error: No questions remain after filtering. Check TTL file names and 'Root Entity' column."
        )
        return None

    # Check ontology sizes for memory estimation
    total_size = 0
    large_ontologies = 0
    for ttl_file in ttl_dir.glob("*.ttl"):
        size = ttl_file.stat().st_size
        total_size += size
        if size > 500000:  # 500KB threshold for TTL files
            large_ontologies += 1

    print(f"📊 Ontology Analysis:")
    print(f"   Total TTL data: {total_size / 1024 / 1024:.1f} MB")
    print(f"   Large ontologies (>500KB): {large_ontologies}")
    print(f"   Average size: {total_size / len(available_ttl_files) / 1024:.1f} KB")

    if total_size > 1000 * 1024 * 1024:  # 1GB threshold
        print(
            "⚠️ Warning: Very large ontology dataset detected. Consider reducing batch size further."
        )
        CONFIG["batch_size"] = min(CONFIG["batch_size"], 15)
        CONFIG["max_workers"] = min(CONFIG["max_workers"], 6)
        print(
            f"   Auto-adjusted: batch_size={CONFIG['batch_size']}, max_workers={CONFIG['max_workers']}"
        )

    return df


def estimate_experiment_time(df, models_config, max_workers):
    """Estimate experiment duration with memory considerations"""
    # Higher time estimate for TTL processing (more complex parsing)
    estimated_time_per_question = 3.5  # Higher for TTL ontologies
    total_calls = len(df) * len(models_config)
    estimated_total_time = (total_calls * estimated_time_per_question) / max_workers

    print(f"\n⏱️ EXPERIMENT ESTIMATES:")
    print(f"   Total questions: {len(df)}")
    print(f"   Total API calls: {total_calls}")
    print(f"   Estimated time: {estimated_total_time / 60:.1f} minutes")
    print(
        f"   Checkpoints will be saved every {CONFIG['checkpoint_frequency']} questions"
    )
    print(f"   Memory monitoring: Every {CONFIG['batch_size']} questions")
    return estimated_total_time


def check_for_previous_run():
    """Check if there's a previous incomplete run"""
    recovery_file = output_dir / "LATEST_recovery_info.json"
    if recovery_file.exists():
        try:
            with open(recovery_file, "r") as f:
                recovery_info = json.load(f)

            print(f"\n🔄 Found previous incomplete run:")
            print(
                f"   Completed: {recovery_info['questions_completed']}/{recovery_info['total_questions']} questions"
            )
            print(f"   Progress: {recovery_info['completion_percentage']:.1f}%")
            if "memory_usage_percent" in recovery_info:
                print(
                    f"   Last memory usage: {recovery_info['memory_usage_percent']:.1f}%"
                )

            response = (
                input("Do you want to continue from where you left off? (y/n): ")
                .lower()
                .strip()
            )
            if response == "y":
                latest_csv = output_dir / "LATEST_checkpoint.csv"
                if latest_csv.exists():
                    df = pd.read_csv(latest_csv)
                    return df, recovery_info

        except Exception as e:
            print(f"⚠️ Could not load previous run: {e}")

    return None, None


def analyze_sparql_patterns(df):
    """Analyze SPARQL query patterns with memory efficiency"""
    patterns = {
        "ask_queries": 0,
        "select_queries": 0,
        "simple_queries": 0,
        "complex_queries": 0,
        "type_queries": 0,
        "property_queries": 0,
    }

    for idx, row in df.iterrows():
        query = str(row.get("SPARQL Query", "")).upper()

        if "ASK" in query:
            patterns["ask_queries"] += 1
        elif "SELECT" in query:
            patterns["select_queries"] += 1

        if "RDF:TYPE" in query or "A " in query:
            patterns["type_queries"] += 1
        else:
            patterns["property_queries"] += 1

        # Count complexity by number of triples
        triple_count = query.count(".") + query.count(";")
        if triple_count <= 1:
            patterns["simple_queries"] += 1
        else:
            patterns["complex_queries"] += 1

    return patterns


def main():
    """Main experiment execution with memory optimization"""
    print_experiment_header()

    if not check_api_clients():
        print("❌ Fix API client issues before proceeding")
        return

    # NEW: Log actual model metadata
    models_metadata_log = log_models_metadata(
        MODELS_CONFIG, output_dir, openai_client, deepseek_client, openrouter_client
    )

    # Check for previous run
    previous_df, recovery_info = check_for_previous_run()

    if previous_df is not None:
        df = previous_df
        print(f"✅ Resuming from previous run with {len(df)} questions")
        start_question = recovery_info["questions_completed"]
    else:
        # Validate setup for new run
        df = validate_setup()
        if df is None:
            return
        start_question = 0

    # Check initial memory
    initial_memory = monitor_memory()
    if initial_memory > 80:
        print(f"⚠️ Warning: High initial memory usage ({initial_memory:.1f}%)")
        print("   Consider closing other applications or reducing batch size")

    # Analyze SPARQL patterns
    sparql_patterns = analyze_sparql_patterns(df)
    print(f"\n📊 SPARQL Query Analysis:")
    for pattern, count in sparql_patterns.items():
        percentage = (count / len(df)) * 100
        print(f"   {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    # Estimate time
    estimated_time = estimate_experiment_time(df, MODELS_CONFIG, CONFIG["max_workers"])

    # Confirmation prompt
    print(f"\n⚠️ Ready to start experiment")
    print(f"📊 Processing {len(df)} questions across {len(MODELS_CONFIG)} models")
    print(f"🎯 Starting from question {start_question + 1}")
    print(
        f"🔇 Silent mode: {'ON (faster)' if CONFIG['silent_mode'] else 'OFF (shows responses)'}"
    )
    print(f"🧠 Current memory: {monitor_memory():.1f}%")

    # Auto-start after brief pause
    print("Starting in 3 seconds... (Ctrl+C to cancel)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Experiment cancelled by user")
        return

    # Run experiment with memory monitoring
    print(f"\n🧠 Starting SPARQL reasoning experiment...")
    start_time = time.time()

    try:
        results_df, logs, detailed_metrics, _ = run_llm_reasoning(
            df,
            ontology_base_path=CONFIG["ttl_ontology_dir"],
            models=CONFIG["models_used"],
            context_mode=CONFIG["context_mode"],
            max_workers=CONFIG["max_workers"],
            question_column=CONFIG["question_column"],
            batch_size=CONFIG["batch_size"],
            output_dir=output_dir,
            silent_mode=CONFIG["silent_mode"],
        )

        experiment_time = time.time() - start_time

        # Print completion summary
        print_completion_summary(
            results_df, experiment_time, estimated_time, sparql_patterns
        )

        # Save final results
        save_final_results(
            results_df, logs, detailed_metrics, experiment_time, sparql_patterns
        )

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        experiment_time = time.time() - start_time
        print(f"⏱️ Ran for {experiment_time:.1f}s ({experiment_time / 60:.1f}min)")
        print(f"💾 Progress has been saved in checkpoints")
        print(f"🧠 Final memory usage: {monitor_memory():.1f}%")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        print(f"🧠 Memory usage at failure: {monitor_memory():.1f}%")


def print_completion_summary(results_df, actual_time, estimated_time, sparql_patterns):
    """Print experiment completion summary with memory info"""
    print("\n" + "=" * 80)
    print("🎉 SPARQL EXPERIMENT COMPLETED!")
    print("=" * 80)

    # Time analysis
    print(f"⏱️ Time Analysis:")
    print(f"   Actual time: {actual_time:.1f}s ({actual_time / 60:.1f}min)")
    print(f"   Estimated time: {estimated_time:.1f}s ({estimated_time / 60:.1f}min)")
    time_diff = ((actual_time - estimated_time) / estimated_time) * 100
    print(f"   Difference: {time_diff:+.1f}%")

    # Memory analysis
    final_memory = monitor_memory()
    print(f"🧠 Memory Analysis:")
    print(f"   Final memory usage: {final_memory:.1f}%")

    # Response analysis
    total_questions = len(results_df)
    print(f"\n📊 Response Analysis:")
    print(f"   Total questions: {total_questions}")

    for model in MODELS_CONFIG.keys():
        response_col = f"{model}_response"
        if response_col in results_df.columns:
            responses = results_df[response_col]
            non_empty = (responses != "").sum()
            errors = responses.astype(str).str.startswith("[ERROR]").sum()
            success = non_empty - errors
            success_rate = (success / len(responses)) * 100 if len(responses) > 0 else 0

            # Correctness analysis
            correctness_col = f"{model}_quality_correctness"
            if correctness_col in results_df.columns:
                correct_answers = (results_df[correctness_col] > 0.5).sum()
                correctness_rate = (correct_answers / len(results_df)) * 100
                avg_correctness = results_df[correctness_col].mean()
                avg_response_time = pd.to_numeric(
                    results_df[f"{model}_response_time"], errors="coerce"
                ).mean()

                print(f"   🤖 {model}:")
                print(
                    f"     Completed: {success}/{len(responses)} ({success_rate:.1f}%)"
                )
                print(
                    f"     Correct: {correct_answers}/{len(results_df)} ({correctness_rate:.1f}%)"
                )
                print(f"     Avg correctness: {avg_correctness:.3f}")
                print(f"     Avg response time: {avg_response_time:.2f}s")
                if errors > 0:
                    print(f"     Errors: {errors}")


def save_final_results(
    results_df, logs, detailed_metrics, experiment_time, sparql_patterns
):
    """Save all final results with enhanced metadata and memory cleanup"""
    print(f"\n💾 Saving final results...")

    # Main results files
    results_file = output_dir / "sparql_ttl_results_FINAL.csv"
    logs_file = output_dir / "sparql_ttl_logs_FINAL.csv"
    metrics_file = output_dir / "sparql_ttl_metrics_FINAL.json"
    config_file = output_dir / "experiment_summary.json"

    # Save main files with memory efficiency
    results_df.to_csv(results_file, index=False)

    # Save logs in chunks if large
    if len(logs) > 1000:
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(logs_file, index=False, chunksize=500)
        del logs_df
    else:
        pd.DataFrame(logs).to_csv(logs_file, index=False)

    # Save essential metrics only
    essential_metrics = []
    for metric in detailed_metrics:
        essential_metric = {
            "query_index": metric.get("query_index"),
            "model_display_name": metric.get("model_display_name"),
            "final_answer_extracted": metric.get("final_answer_extracted"),
            "quality_correctness": metric.get("quality_correctness"),
            "response_time_seconds": metric.get("response_time_seconds"),
            "error_occurred": metric.get("error_occurred"),
        }
        essential_metrics.append(essential_metric)

    with open(metrics_file, "w") as f:
        json.dump(essential_metrics, f, indent=2, default=str)

    # Enhanced experiment summary
    performance_summary = calculate_model_performance_summary(results_df, MODELS_CONFIG)

    experiment_summary = {
        "config": CONFIG,
        "sparql_patterns": sparql_patterns,
        "experiment_time_seconds": experiment_time,
        "experiment_time_minutes": experiment_time / 60,
        "total_questions_processed": len(results_df),
        "total_api_calls": len(results_df) * len(MODELS_CONFIG),
        "performance_summary": performance_summary,
        "timestamp": datetime.now().isoformat(),
        "output_directory": str(output_dir),
        "memory_info": {
            "final_memory_usage_percent": monitor_memory(),
            "memory_threshold": CONFIG["memory_threshold"],
        },
        "files_created": {
            "results": str(results_file),
            "logs": str(logs_file),
            "metrics": str(metrics_file),
            "summary": str(config_file),
        },
        "checkpoint_info": {
            "frequency": CONFIG["checkpoint_frequency"],
            "checkpoints_dir": str(output_dir / "checkpoints"),
        },
    }

    with open(config_file, "w") as f:
        json.dump(experiment_summary, f, indent=2, default=str)

    print(f"✅ Results saved to: {output_dir}")
    print(f"📄 Main results: {results_file}")
    print(f"📊 Summary: {config_file}")
    print(f"💾 Checkpoints: {output_dir / 'checkpoints'}")

    # Print final model performance summary
    print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
    for model_name, summary in performance_summary.items():
        response_metrics = summary.get("response_metrics", {})
        quality_metrics = summary.get("quality_metrics", {})

        success_rate = response_metrics.get("success_rate", 0) * 100
        avg_correctness = quality_metrics.get("correctness", {}).get("mean", 0)

        print(f"  🤖 {model_name}:")
        print(f"     Success Rate: {success_rate:.1f}%")
        print(f"     Avg Correctness: {avg_correctness:.3f}")

    # Final cleanup
    force_cleanup()


if __name__ == "__main__":
    main()
