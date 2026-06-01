"""
api_calls.py - Universal evaluation metrics for all models with enhanced prompting and model parameters
Optimized for heavy ontologies and thousands of questions
"""

import os
import errno
import time
import sys
import json
import re
from datetime import datetime
import openai
from dotenv import load_dotenv
import concurrent.futures
from threading import Lock
import pandas as pd
import hashlib
import numpy as np
import functools
from pathlib import Path
import threading
from collections import defaultdict
import traceback
import json
import math
import gc
import psutil
import requests
import platform
from typing import Dict, Any

# Add progress tracking
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)

load_dotenv()

# Enhanced model dictionary with latest models
MODELS = {
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "deepseek-chat": "deepseek-chat",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
}

# Model to client mapping for cleaner client selection
MODEL_TO_CLIENT = {
    "gpt-5-mini": "openai",
    "gpt-4o-mini": "openai",
    "deepseek-chat": "deepseek",
    "llama-4-maverick": "openrouter",
}

# Global variables for progress tracking and response display
progress_lock = Lock()
display_lock = Lock()
save_lock = Lock()
memory_lock = Lock()
completed_tasks = 0
total_tasks = 0
display_results = defaultdict(list)
question_counter = 0
questions_completed = 0
stop_requested = threading.Event()
stop_reason = None


CREDIT_ERROR_PATTERNS = (
    "insufficient credit",
    "insufficient credits",
    "not enough credit",
    "not enough credits",
    "out of credit",
    "out of credits",
    "insufficient balance",
    "balance too low",
    "quota exceeded",
    "billing",
    "payment required",
    "credit limit",
    "rate limit exceeded",
    "temporarily rate-limited",
    "too-many-requests",
    "free-models-per-day",
    "free-models-per-min",
    "429",
    "402",
)


def is_insufficient_credit_error(value):
    """Detect provider billing/credit failures that should stop a run."""
    if value is None:
        return False
    text = str(value).lower()
    return any(pattern in text for pattern in CREDIT_ERROR_PATTERNS)


def is_no_space_left_error(error):
    """Detect disk-full failures raised by pandas/Python on different platforms."""
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return True
    if getattr(error, "winerror", None) == 112:
        return True
    text = str(error).lower()
    return "no space left on device" in text or "not enough space on the disk" in text


def request_stop(reason):
    """Request an early stop and keep the first reason for the final summary."""
    global stop_reason
    if not stop_requested.is_set():
        stop_reason = reason
    stop_requested.set()


def resolve_model_info(display_name, model_entry):
    """
    Supports both:
      old format:  "gpt-5-mini": "gpt-5-mini-2025-08-07"
      new format:  "openai_gpt_4_1_mini": {"provider": "openai", "model_id": "gpt-4.1-mini"}
    """
    if isinstance(model_entry, dict):
        provider = model_entry.get("provider")
        model_id = model_entry.get("model_id")
        if not provider or not model_id:
            raise ValueError(
                f"Invalid model config for {display_name}. Expected provider and model_id."
            )
        return provider, model_id

    # old fallback behavior
    model_id = model_entry
    provider = MODEL_TO_CLIENT.get(display_name)

    if not provider:
        # best-effort fallback from model id
        if str(model_id).startswith("gpt-"):
            provider = "openai"
        elif "/" in str(model_id):
            provider = "openrouter"
        elif "deepseek" in str(model_id).lower():
            provider = "deepseek"
        else:
            raise ValueError(
                f"Could not infer provider for model '{display_name}' with id '{model_id}'"
            )

    return provider, model_id


def get_client_for_provider(provider):
    clients = {
        "openai": openai_client,
        "deepseek": deepseek_client,
        "openrouter": openrouter_client,
    }
    return clients.get(provider)


def get_model_params(models=None):
    """
    Returns parameters for each model.
    Supports both:
      old format:  {"gpt-5-mini": "gpt-5-mini-2025-08-07"}
      new format:  {"openai_gpt_4_1_mini": {"provider":"openai","model_id":"gpt-4.1-mini"}}
    """

    if models is None:
        models = MODELS

    model_params = {}

    for display_name, model_entry in models.items():
        # Detect provider
        if isinstance(model_entry, dict):
            provider = model_entry.get("provider")
        else:
            provider = MODEL_TO_CLIENT.get(display_name)

        if provider == "openai":
            model_params[display_name] = {"max_completion_tokens": 1024}

        elif provider == "deepseek":
            model_params[display_name] = {
                "max_tokens": 1024,
                "temperature": 0.0,
                "top_p": 0.9,
            }

        elif provider == "openrouter":
            model_params[display_name] = {
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 1024,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.1,
            }

        else:
            model_params[display_name] = {"max_tokens": 1024}

    return model_params


def monitor_memory():
    """Monitor system memory usage"""
    return psutil.virtual_memory().percent


def force_cleanup():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    time.sleep(0.5)


def get_model_params(models=None):
    if models is None:
        models = MODELS

    model_params = {}
    for display_name, model_entry in models.items():
        if isinstance(model_entry, dict):
            provider = model_entry.get("provider")
            model_id = model_entry.get("model_id", display_name)
        else:
            provider = MODEL_TO_CLIENT.get(display_name)
            model_id = str(model_entry)

        if provider == "openai" and "gpt-5" in model_id:
            model_params[display_name] = {
                "max_completion_tokens": 1024,  # Increased
                "reasoning_effort": "low",
                "verbosity": "low",
            }
        elif provider == "openai":
            model_params[display_name] = {"max_completion_tokens": 1024}
        elif provider == "deepseek":
            model_params[display_name] = {
                "max_tokens": 1024,  # Increased
                "temperature": 0.0,  # Now supported!
                "top_p": 0.9,
            }
        else:
            model_params[display_name] = {
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 1024,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.1,
            }
    return model_params


def is_completed_response(value):
    """Only successful non-empty responses should be skipped on resume."""
    if value is None:
        return False
    if pd.isna(value):
        return False
    text = str(value).strip()
    if text.startswith("[ERROR]"):
        return False
    if is_insufficient_credit_error(text):
        return False
    return text != ""


def count_completed_questions(df, models):
    completed = 0
    for _, row in df.iterrows():
        if all(
            is_completed_response(row.get(f"{model}_response", "")) for model in models
        ):
            completed += 1
    return completed


def save_checkpoint_csv(
    df,
    logs,
    detailed_metrics,
    output_dir,
    questions_completed,
    total_questions,
    model_names=None,
):
    """Save a rolling checkpoint without accumulating historical CSV copies."""
    with save_lock:
        try:
            model_names = list(model_names or MODELS.keys())
            latest_file = output_dir / "LATEST_checkpoint.csv"
            latest_logs = output_dir / "LATEST_checkpoint_logs.csv"
            latest_metrics = output_dir / "LATEST_checkpoint_metrics.json"

            df.to_csv(latest_file, index=False)
            if logs:
                logs_df = pd.DataFrame(logs)
                logs_df.to_csv(latest_logs, index=False)
                del logs_df
            if detailed_metrics:
                essential_metrics = []
                for metric in detailed_metrics:
                    essential_metrics.append(
                        {
                            "query_index": metric.get("query_index"),
                            "model_display_name": metric.get("model_display_name"),
                            "final_answer_extracted": metric.get(
                                "final_answer_extracted"
                            ),
                            "quality_correctness": metric.get("quality_correctness"),
                            "response_time_seconds": metric.get(
                                "response_time_seconds"
                            ),
                            "error_occurred": metric.get("error_occurred"),
                        }
                    )

                with open(latest_metrics, "w") as f:
                    json.dump(essential_metrics, f, indent=2, default=str)
                del essential_metrics

            # Create recovery info
            recovery_info = {
                "questions_completed": questions_completed,
                "total_questions": total_questions,
                "completion_percentage": (questions_completed / total_questions) * 100,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_file": str(latest_file),
                "total_api_calls_completed": completed_tasks,
                "total_api_calls_expected": total_questions * len(model_names),
                "memory_usage_percent": monitor_memory(),
            }

            recovery_file = output_dir / "LATEST_recovery_info.json"
            with open(recovery_file, "w") as f:
                json.dump(recovery_info, f, indent=2, default=str)

            # Calculate and display statistics
            model_stats = {}
            for model in model_names:
                response_col = f"{model}_response"
                if response_col in df.columns:
                    non_empty = (df[response_col] != "").sum()
                    errors = (
                        df[response_col].astype(str).str.startswith("[ERROR]").sum()
                    )
                    success = non_empty - errors

                    # Calculate correctness
                    correctness_col = f"{model}_quality_correctness"
                    if correctness_col in df.columns:
                        correct_values = pd.to_numeric(
                            df[correctness_col], errors="coerce"
                        )
                        correct_answers = (correct_values > 0.5).sum()
                        accuracy = (
                            (correct_answers / non_empty * 100) if non_empty > 0 else 0
                        )
                    else:
                        accuracy = 0

                    model_stats[model] = {
                        "responses": int(non_empty),
                        "successful": int(success),
                        "errors": int(errors),
                        "correct": int(correct_answers)
                        if correctness_col in df.columns
                        else 0,
                        "accuracy": float(accuracy),
                    }

            memory_usage = monitor_memory()
            print(
                f"\nCHECKPOINT SAVED - Question {questions_completed}/{total_questions} ({(questions_completed / total_questions) * 100:.1f}%)"
            )
            print(f"Saved to: {latest_file}")
            print(f"Memory usage: {memory_usage:.1f}%")
            print(f"Progress Summary:")

            for model, stats in model_stats.items():
                print(
                    f"   {model:20} | {stats['successful']:3d}/{stats['responses']:3d} success | {stats['correct']:3d} correct ({stats['accuracy']:5.1f}%)"
                )

            # Force cleanup after checkpoint
            force_cleanup()

            return latest_file

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if is_no_space_left_error(e):
                request_stop("No space left on device while saving checkpoint.")
                print("Stopping run before scheduling more API calls.")
            traceback.print_exc()
            return None


def display_question_and_response(
    question_idx,
    question_text,
    model_name,
    expected_answer,
    model_response,
    final_answer,
    is_correct,
    response_time,
    confidence_score=None,
    reasoning_steps=None,
    error=None,
    silent_mode=False,
):
    """Display question and model response with memory-efficient formatting"""
    if silent_mode:
        return

    with display_lock:
        print("\n" + "=" * 80)  # Reduced width for better performance
        print(f"QUESTION #{question_idx + 1} | MODEL: {model_name.upper()}")
        print("=" * 80)

        # Display question (truncated if too long for memory efficiency)
        question_display = (
            question_text[:150] + "..." if len(question_text) > 150 else question_text
        )
        print(f"QUESTION: {question_display}")
        print(f"EXPECTED: {expected_answer}")

        # Display result with new metrics
        status_icon = "âœ…" if is_correct else "âŒ" if not error else "âš ï¸"
        print(f"{status_icon} MODEL ANSWER: {final_answer}")
        print(f"RESPONSE TIME: {response_time:.2f}s")

        if confidence_score is not None and confidence_score > 0:
            confidence_bar = "â–ˆ" * int(confidence_score * 5) + "â–‘" * (
                5 - int(confidence_score * 5)
            )  # Smaller bar
            print(f"ðŸŽ¯ CONFIDENCE: {confidence_score:.2f} [{confidence_bar}]")

        if reasoning_steps is not None and reasoning_steps > 0:
            complexity_bar = "ðŸ§ " * min(reasoning_steps, 3) + "ðŸ’­" * max(
                0, min(reasoning_steps - 3, 2)
            )
            print(f"ðŸ§  REASONING: {reasoning_steps}/10 {complexity_bar}")

        if error:
            print(f"ðŸ’¥ ERROR: {error}")
        else:
            # Show response preview (shortened for memory)
            if model_response:
                response_preview = (
                    model_response[:200] + "..."
                    if len(model_response) > 200
                    else model_response
                )
                print(f"ðŸ’¬ RESPONSE:\n{response_preview}")

        print("-" * 80)


def display_recent_summary(
    display_results, models, sample_size=3
):  # Reduced sample size
    """Display recent responses summary with memory optimization"""
    print(f"\nðŸ“Š RECENT {sample_size} RESPONSES SUMMARY")
    print("=" * 60)

    for model in models:
        if model in display_results and display_results[model]:
            samples = display_results[model][-sample_size:]
            correct_count = sum(1 for s in samples if s["is_correct"])

            print(
                f"\nðŸ¤– {model.upper()}: {correct_count}/{len(samples)} correct ({(correct_count / len(samples) * 100):.1f}%)"
            )

            for sample in samples:
                status = "âœ…" if sample["is_correct"] else "âŒ"
                print(
                    f"   {status} Q{sample['question_idx']:3d}: {sample['final_answer'][:30]:30} | {sample['response_time']:.2f}s"
                )


def update_progress(
    future,
    pbar,
    model_name,
    question_text,
    expected_answer,
    question_idx,
    df,
    logs,
    detailed_metrics,
    output_dir,
    models_list,
    total_questions,
    checkpoint_frequency,
    silent_mode=False,
):
    """Callback function to update progress and save checkpoints with memory management"""
    global completed_tasks, display_results, questions_completed

    if future.cancelled():
        return

    with progress_lock:
        completed_tasks += 1
        pbar.update(1)

        # Get result for display and processing
        try:
            result = future.result()
            if not result:
                completed_tasks -= 1
                return

            if result:
                final_answer = result.get("final_answer_extracted", "ERROR")
                is_correct = result.get("quality_correctness", 0.0) > 0.5
                response_time = result.get("response_time_seconds", 0)
                reasoning = result.get("reasoning_extracted", "")
                model_response = result.get("full_response", "")
                error = result.get("error_message", None)

                if error and is_insufficient_credit_error(error):
                    if not stop_requested.is_set():
                        print(
                            "\nStopping run: provider reported a billing, quota, or rate-limit error."
                        )
                        print(f"Model: {model_name}")
                        print(f"Error: {error}")
                    request_stop(
                        "Provider reported a billing, quota, or rate-limit error."
                    )

                # Display the question and response with reduced frequency for performance
                if not silent_mode and (
                    question_idx % 5 == 0 or error
                ):  # Show every 5th question or errors
                    display_question_and_response(
                        question_idx,
                        question_text,
                        model_name,
                        expected_answer,
                        model_response,
                        final_answer,
                        is_correct,
                        response_time,
                        confidence_score=result.get("confidence_score", 0.5),
                        reasoning_steps=result.get("reasoning_steps_complexity", 1),
                        error=error,
                        silent_mode=silent_mode,
                    )

                # Store for summary display with memory limit
                display_results[model_name].append(
                    {
                        "question": question_text[:100],  # Truncate for memory
                        "question_idx": question_idx,
                        "expected_answer": expected_answer,
                        "final_answer": final_answer,
                        "is_correct": is_correct,
                        "response_time": response_time,
                    }
                )

                # Keep only last 5 for memory efficiency
                if len(display_results[model_name]) > 5:
                    display_results[model_name] = display_results[model_name][-5:]

                # Add to detailed metrics and logs (simplified for memory)
                essential_result = {
                    "query_index": result.get("query_index"),
                    "model_display_name": result.get("model_display_name"),
                    "final_answer_extracted": final_answer,
                    "quality_correctness": result.get("quality_correctness", 0.0),
                    "response_time_seconds": response_time,
                    "error_occurred": result.get("error_occurred", False),
                }
                detailed_metrics.append(essential_result)

                # Simplified log entry
                logs.append(
                    {
                        "Query_index": question_idx,
                        "model": model_name,
                        "final_answer": final_answer,
                        "quality_correctness": result.get("quality_correctness", 0.0),
                        "response_time": response_time,
                    }
                )

                # Update DataFrame
                dn, idx = result["model_display_name"], result["query_index"]
                df.at[idx, f"{dn}_response"] = model_response
                if not result["error_occurred"]:
                    df.at[idx, f"{dn}_final_answer"] = final_answer
                    df.at[idx, f"{dn}_confidence_score"] = result.get(
                        "confidence_score", 0.5
                    )
                    df.at[idx, f"{dn}_reasoning_steps_complexity"] = result.get(
                        "reasoning_steps_complexity", 1
                    )
                    df.at[idx, f"{dn}_response_time"] = response_time
                    df.at[idx, f"{dn}_token_count"] = result.get("total_tokens", 0)
                    df.at[idx, f"{dn}_quality_correctness"] = result.get(
                        "quality_correctness", 0.0
                    )

                # Check if we completed all models for this question
                question_complete = True
                for model in models_list:
                    if not is_completed_response(
                        df.at[question_idx, f"{model}_response"]
                    ):
                        question_complete = False
                        break

                # If question is complete, increment counter and maybe save checkpoint
                if question_complete:
                    questions_completed += 1

                    if questions_completed % checkpoint_frequency == 0:
                        checkpoint_file = save_checkpoint_csv(
                            df,
                            logs,
                            detailed_metrics,
                            output_dir,
                            questions_completed,
                            total_questions,
                            models_list,
                        )

                        # Clear logs and metrics after checkpoint to free memory
                        if checkpoint_file is not None:
                            logs.clear()
                            detailed_metrics.clear()

                        if not silent_mode:
                            display_recent_summary(
                                display_results, models_list, sample_size=2
                            )

                    # Show progress for completed questions
                    if questions_completed % 10 == 0:
                        memory_usage = monitor_memory()
                        print(
                            f"\nCompleted {questions_completed}/{total_questions} questions ({(questions_completed / total_questions) * 100:.1f}%)"
                        )
                        print(f"Memory usage: {memory_usage:.1f}%")

                        # Force cleanup if memory is high
                        if memory_usage > 85:
                            print("High memory usage, forcing cleanup...")
                            force_cleanup()

        except Exception as e:
            print(f"Error processing result for {model_name}: {e}")


# Initialize API clients
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not all([openai_api_key, deepseek_api_key, openrouter_api_key]):
    print("Some API keys are missing. Only keys for selected providers are required.")

openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
deepseek_client = (
    openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    if deepseek_api_key
    else None
)
openrouter_client = (
    openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
    if openrouter_api_key
    else None
)


def extract_final_answer(response_content, question_type):
    """Extract just the final answer"""
    if not response_content or "[ERROR]" in response_content:
        return "ERROR"

    # Try structured ANSWER: format
    answer_match = re.search(r"ANSWER:\s*([^\n\r]+)", response_content, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback to first line
        lines = response_content.strip().split("\n")
        answer = lines[0].strip() if lines else response_content.strip()

    # Clean up URIs for MC questions
    if question_type == "MC" or question_type.lower() == "multi choice":
        answer = re.sub(r"<[^#]*#([^>]+)>", r"\1", answer)
        answer = re.sub(r"[^#]*#([^,\s]+)", r"\1", answer)
        answer = re.sub(r"ns1:([^,\s]+)", r"\1", answer)
        answer = re.sub(r"rdf:([^,\s]+)", r"\1", answer)
        answer = re.sub(r"rdfs:([^,\s]+)", r"\1", answer)
        answer = re.sub(r"owl:([^,\s]+)", r"\1", answer)
    elif question_type == "BIN" or question_type.lower() == "binary":
        if "true" in answer.lower():
            return "TRUE"
        elif "false" in answer.lower():
            return "FALSE"

    return answer.strip()


def extract_confidence_score(response_content):
    """Extract confidence score from response with safe parsing"""
    if not response_content:
        return 0.5  # Default uncertain

    # Look for CONFIDENCE: pattern
    confidence_match = re.search(
        r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", response_content, re.IGNORECASE
    )
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            return max(0.0, min(1.0, confidence))  # Clamp to [0,1]
        except (ValueError, AttributeError):
            pass

    return 0.5  # Default to uncertain


def extract_reasoning_steps(response_content):
    """Extract reasoning steps complexity from response with safe parsing"""
    if not response_content:
        return 1  # Default simple

    # Look for REASONING_STEPS: pattern
    steps_match = re.search(
        r"REASONING_STEPS:\s*([0-9]+)", response_content, re.IGNORECASE
    )
    if steps_match:
        try:
            steps = int(steps_match.group(1))
            return max(1, min(10, steps))  # Clamp to [1,10]
        except (ValueError, AttributeError):
            pass

    return 1  # Default to simple


def evaluate_response_quality_universal(
    response_content, expected_answer, question_type, extracted_answer
):
    quality_metrics = {
        "correctness": 0.0,
        "completeness": 0.0,
        "structure_compliance": 0.0,
        "reasoning_quality": 0.0,
    }
    if not response_content:
        return quality_metrics

    expected_lower, extracted_lower = (
        str(expected_answer).lower().strip(),
        str(extracted_answer).lower().strip(),
    )

    if question_type == "BIN":
        if expected_lower == extracted_lower:
            quality_metrics["correctness"] = 1.0
    elif question_type == "MC":
        # Enhanced MC evaluation for multiple answers
        if expected_lower in extracted_lower or extracted_lower in expected_lower:
            quality_metrics["correctness"] = 1.0
        else:
            # Semantic similarity for multiple choice answers
            expected_terms = set(
                re.findall(r"\b\w+\b", expected_lower.replace(",", " "))
            )
            extracted_terms = set(
                re.findall(r"\b\w+\b", extracted_lower.replace(",", " "))
            )

            if expected_terms and extracted_terms:
                intersection = expected_terms.intersection(extracted_terms)
                union = expected_terms.union(extracted_terms)
                if union:
                    quality_metrics["correctness"] = len(intersection) / len(union)

    # Completeness based on conciseness preference
    word_count = len(response_content.split())
    if 20 <= word_count <= 50:
        quality_metrics["completeness"] = 1.0  # Sweet spot for concise answers
    elif 10 <= word_count < 20:
        quality_metrics["completeness"] = 0.8
    elif word_count > 50:
        quality_metrics["completeness"] = 0.6  # Penalize verbosity
    else:
        quality_metrics["completeness"] = 0.3

    # Structure compliance
    has_answer_format = bool(
        re.search(r"ANSWER:\s*[^\n]+", response_content, re.IGNORECASE)
    )
    has_reasoning_format = bool(
        re.search(r"REASONING:\s*[^\n]+", response_content, re.IGNORECASE)
    )

    if has_answer_format and has_reasoning_format:
        quality_metrics["structure_compliance"] = 1.0
    elif has_answer_format or has_reasoning_format:
        quality_metrics["structure_compliance"] = 0.5

    # Reasoning quality (check for bullet points and logical structure)
    bullet_indicators = ["â€¢", "-", "*", "1.", "2.", "3."]
    logical_indicators = ["because", "therefore", "since", "thus", "shows", "indicates"]

    bullet_count = sum(
        1 for indicator in bullet_indicators if indicator in response_content
    )
    logical_count = sum(
        1 for indicator in logical_indicators if indicator in response_content.lower()
    )

    quality_metrics["reasoning_quality"] = min(
        (bullet_count * 0.3 + logical_count * 0.2), 1.0
    )

    return quality_metrics


def create_enhanced_metrics_entry(
    idx,
    row,
    query,
    question_column,
    display_name,
    model_id,
    response,
    content,
    start_time,
    end_time,
    attempt,
    model_params,
    ontology_context="",
    error=None,
):
    response_time = round(end_time - start_time, 3)
    answer_type = row.get("Answer Type", "BIN")
    expected_answer = row.get("Answer", "Unknown")

    metrics = {
        "query_index": idx,
        "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
        "query_text": query[:200] + "..."
        if len(query) > 200
        else query,  # Truncate for memory
        "question_column_used": question_column,
        "ontology_name": row.get("Root Entity", "Unknown"),
        "answer_type": answer_type,
        "expected_answer": expected_answer,
        "model_display_name": display_name,
        "model_api_id": model_id,
        "attempt_number": attempt + 1,
        "timestamp_request": datetime.now().isoformat(),
        "response_time_seconds": response_time,
        "response_length_chars": len(content) if content else 0,
        "response_length_words": len(content.split()) if content else 0,
        "full_response": content[:500] + "..."
        if content and len(content) > 500
        else content,  # Truncate for memory
        "response_preview": content[:200] if content else "",
        "error_occurred": error is not None,
        "error_message": str(error) if error else None,
    }

    if response and not error and content:
        # Extract our new metrics
        final_answer = extract_final_answer(content, answer_type)
        confidence_score = extract_confidence_score(content)
        reasoning_steps = extract_reasoning_steps(content)

        metrics.update(
            {
                "final_answer_extracted": final_answer,
                "confidence_score": confidence_score,
                "reasoning_steps_complexity": reasoning_steps,
            }
        )

        # Evaluate correctness
        quality_metrics = evaluate_response_quality_universal(
            content, expected_answer, answer_type, final_answer
        )
        for quality_metric, score in quality_metrics.items():
            metrics[f"quality_{quality_metric}"] = round(score, 3)

        # Token usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            metrics.update(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "tokens_per_second": round(
                        (usage.completion_tokens or 0) / max(response_time, 0.001), 2
                    ),
                }
            )

    return metrics


@functools.lru_cache(maxsize=128)  # Reduced cache size for memory efficiency
def load_ontology_context(ontology_base_path, ontology_name, context_mode):
    try:
        if context_mode == "ttl":
            ontology_path = os.path.join(ontology_base_path, f"{ontology_name}.ttl")
            if not os.path.exists(ontology_path):
                return f"[ERROR: No ontology file found for {ontology_name} in {ontology_base_path}]"
            with open(ontology_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Truncate very large ontologies for memory efficiency
                if len(content) > 15000:
                    content = (
                        content[:15000] + "\n... [truncated for memory efficiency]"
                    )
                return content
        elif context_mode == "json":
            ontology_path = os.path.join(ontology_base_path, f"{ontology_name}.json")
            with open(ontology_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                json_content = json.dumps(data, indent=2)
                # Truncate very large JSON for memory efficiency
                if len(json_content) > 15000:
                    json_content = (
                        json_content[:15000] + "\n... [truncated for memory efficiency]"
                    )
                return json_content
        else:
            return f"[ERROR: Unknown context mode: {context_mode}]"
    except Exception as e:
        return f"[ERROR: Failed to load ontology {ontology_name}: {str(e)}]"


def get_inline_context_from_row(row, context_mode):
    context_columns = {
        "inline_owl": "OWL Context",
        "inline_nl": "NL Context",
        "inline_abs": "ABS Context",
    }
    context_column = context_columns.get(context_mode)
    if not context_column:
        return None
    return row.get(context_column, "")


def extract_sparql_terms(query):
    """Return URI fragments/local names that should be prioritized in context."""
    if not query:
        return []

    terms = []
    text = str(query)
    generic_terms = {
        "type",
        "rdf",
        "rdfs",
        "owl",
        "xsd",
        "resource",
        "namedindividual",
    }

    for uri in re.findall(r"<([^>]+)>", text):
        if "www.w3.org/" in uri:
            continue
        terms.append(uri)
        fragment = re.split(r"[#/]", uri.rstrip("/"))[-1]
        if fragment and fragment.lower() not in generic_terms:
            terms.append(fragment)

    for prefixed in re.findall(r"\b([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)\b", text):
        terms.append(prefixed)
        local_name = prefixed.split(":", 1)[1]
        if local_name and local_name.lower() not in generic_terms:
            terms.append(local_name)

    seen = set()
    unique_terms = []
    for term in terms:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms


def split_symbolic_context_blocks(context):
    """Split Turtle/RDF/XML context into chunks that keep related facts together."""
    text = str(context or "")
    if "</rdf:Description>" in text:
        blocks = re.findall(
            r"<rdf:Description\b.*?</rdf:Description>",
            text,
            flags=re.DOTALL,
        )
        if blocks:
            return blocks

    blocks = re.split(r"\n\s*\n", text)
    return [block for block in blocks if block.strip()]


def build_query_relevant_symbolic_context(query, ontology_context, max_chars=12000):
    """Keep SPARQL-relevant symbolic evidence before applying any size limit."""
    context = str(ontology_context or "")
    if len(context) <= max_chars:
        return context

    terms = extract_sparql_terms(query)
    if not terms:
        return context[:max_chars] + "\n... [truncated for memory efficiency]"

    lower_terms = [term.lower() for term in terms]
    blocks = split_symbolic_context_blocks(context)
    selected = []
    selected_ids = set()

    for block in blocks:
        block_lower = block.lower()
        if any(term in block_lower for term in lower_terms):
            block_id = id(block)
            if block_id not in selected_ids:
                selected.append(block.strip())
                selected_ids.add(block_id)

    if not selected:
        snippets = []
        for term in terms:
            index = context.lower().find(term.lower())
            if index == -1:
                continue
            start = max(0, index - 2000)
            end = min(len(context), index + 4000)
            snippets.append(context[start:end].strip())
        selected = snippets

    header = (
        "[Query-relevant symbolic context extracted before truncation]\n"
        f"SPARQL query: {query}\n"
    )
    focused_context = header + "\n\n".join(selected)

    if len(focused_context) < max_chars:
        remaining = max_chars - len(focused_context)
        prefix = context[: max(0, remaining)]
        focused_context = focused_context + "\n\n[Ontology prefix]\n" + prefix

    if len(focused_context) > max_chars:
        focused_context = (
            focused_context[:max_chars]
            + "\n... [query-relevant symbolic context truncated]"
        )

    return focused_context


def create_context_specific_prompt(query, ontology_context, context_mode, answer_type):
    """Enhanced prompting for confidence and reasoning steps with memory optimization"""

    if answer_type == "BIN" or answer_type.lower() == "binary":
        format_instruction = (
            "ANSWER: [TRUE or FALSE]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: ONLY write TRUE or FALSE.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    elif answer_type == "MC" or answer_type.lower() == "multi choice":
        format_instruction = (
            "ANSWER: [Use LOCAL NAMES only, semicolon-separated]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "ANSWER section: Use LOCAL NAMES only (e.g., 'Person', 'U0C4', 'caroline_lavinia_tubb_1840'), give all the possible answers.\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )
    else:
        format_instruction = (
            "ANSWER: [Your answer using local names]\n"
            "CONFIDENCE: [score ranging from 0.0 to 1.0 indicating how certain you are]\n"
            "CONFIDENCE section: 1.0 = completely certain, 0.0 = pure guess.\n"
        )

    # Move this outside the if/else blocks so all answer types can use it
    base_instruction = (
        "CRITICAL: You MUST respond in exactly this format:\n"
        f"{format_instruction}\n"
        "DO NOT include any additional text before or after this format.\n"
    )

    if context_mode in {"ttl", "inline_owl"}:
        ontology_context = build_query_relevant_symbolic_context(
            query, ontology_context, max_chars=12000
        )
        return f"""You are an expert in SPARQL and OWL ontologies. Analyze the ontology context and answer the SPARQL query precisely.

{base_instruction}

Question: {query}
Context: {ontology_context}"""

    else:  # Natural language mode
        if len(ontology_context) > 10000:
            ontology_context = (
                ontology_context[:10000] + "\n... [truncated for memory efficiency]"
            )
        return f"""You are an expert in ontologies, answer the following question based on the provided ontological relationships. Reason through the ontological context and answer based on what you can infer from the context.

{base_instruction}

Question: {query}
Context: {ontology_context}"""


def normalize_answer_type(answer_type):
    """Normalize answer type to handle both old and new formats"""
    if not answer_type:
        return "BIN"

    answer_type_lower = str(answer_type).lower().strip()

    if answer_type_lower in ["binary", "bin"]:
        return "BIN"
    elif answer_type_lower in ["multi choice", "mc", "multichoice"]:
        return "MC"
    else:
        return "BIN"  # Default fallback


def process_single_model_request(args):
    (
        idx,
        row,
        display_name,
        model_entry,
        question_column,
        ontology_base_path,
        context_mode,
        model_params,
    ) = args

    if stop_requested.is_set():
        return None

    query = row.get(question_column, "")
    ontology_name = row.get("Root Entity", "Unknown")
    answer_type = normalize_answer_type(row.get("Answer Type", "BIN"))
    ontology_context = ""

    try:
        provider, model_id = resolve_model_info(display_name, model_entry)
    except Exception as e:
        return create_enhanced_metrics_entry(
            idx,
            row,
            query,
            question_column,
            display_name,
            str(model_entry),
            None,
            f"[ERROR] Invalid model configuration: {str(e)}",
            time.time(),
            time.time(),
            0,
            model_params,
            ontology_context,
            error=e,
        )

    TIMEOUTS = {
        "openai": 30,
        "deepseek": 30,
        "openrouter": 30,
    }

    timeout = TIMEOUTS.get(provider, 45)
    start_time = time.time()
    error_msg = None

    try:
        ontology_context = get_inline_context_from_row(row, context_mode)
        if ontology_context is None:
            ontology_context = load_ontology_context(
                ontology_base_path, ontology_name, context_mode
            )
        full_prompt = create_context_specific_prompt(
            query, ontology_context, context_mode, answer_type
        )

        client = get_client_for_provider(provider)
        if not client:
            raise ValueError(f"No client available for provider: {provider}")

        current_params = model_params.get(display_name, {}).copy()

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": full_prompt}],
            timeout=timeout,
            **current_params,
        )

        end_time = time.time()
        content = response.choices[0].message.content

        return create_enhanced_metrics_entry(
            idx,
            row,
            query,
            question_column,
            display_name,
            model_id,
            response,
            content,
            start_time,
            end_time,
            0,
            model_params,
            ontology_context,
        )

    except Exception as e:
        error_msg = str(e)
        end_time = time.time()

        return create_enhanced_metrics_entry(
            idx,
            row,
            query,
            question_column,
            display_name,
            model_id,
            None,
            f"[ERROR] {error_msg}",
            start_time,
            end_time,
            0,
            model_params,
            ontology_context,
            error=e,
        )


def check_api_clients(models_config=None):
    """
    Validate only the API clients needed for the selected models.
    models_config format:
    {
        "openai_gpt_4_1_mini": {"provider": "openai", "model_id": "gpt-4.1-mini"},
        "openrouter_deepseek_chat": {"provider": "openrouter", "model_id": "deepseek/deepseek-chat"},
    }
    """
    issues = []

    required_providers = set()

    if models_config is None:
        # fallback: old behavior if nothing is passed
        required_providers = {"openai", "deepseek", "openrouter"}
    else:
        for model_info in models_config.values():
            if isinstance(model_info, dict):
                provider = model_info.get("provider")
                if provider:
                    required_providers.add(provider)

    if "openai" in required_providers and openai_client is None:
        issues.append("OpenAI client not initialized - check OPENAI_API_KEY")

    if "deepseek" in required_providers and deepseek_client is None:
        issues.append("DeepSeek client not initialized - check DEEPSEEK_API_KEY")

    if "openrouter" in required_providers and openrouter_client is None:
        issues.append("OpenRouter client not initialized - check OPENROUTER_API_KEY")

    if issues:
        print("⚠️ API Client Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("✅ Required API clients are initialized")
    return True


def get_actual_model_metadata(
    client_type: str, model_id: str, client
) -> Dict[str, Any]:
    """
    Retrieve actual model metadata directly from API providers
    """
    metadata = {
        "model_id": model_id,
        "client_type": client_type,
        "retrieval_timestamp": datetime.now().isoformat(),
        "api_accessible": False,
        "error": None,
        "raw_metadata": {},
    }

    try:
        if client_type == "openai":
            metadata.update(get_openai_actual_metadata(client, model_id))
        elif client_type == "deepseek":
            metadata.update(get_deepseek_actual_metadata(client, model_id))
        elif client_type == "openrouter":
            metadata.update(get_openrouter_actual_metadata(client, model_id))

        metadata["api_accessible"] = True

    except Exception as e:
        metadata["error"] = str(e)
        metadata["api_accessible"] = False

    return metadata


def get_openai_actual_metadata(client, model_id: str) -> Dict[str, Any]:
    """Get actual OpenAI model metadata from API"""
    try:
        # Try to get model info from OpenAI's models endpoint
        models_response = client.models.list()
        model_info = None

        for model in models_response.data:
            if model.id == model_id:
                model_info = model
                break

        if model_info:
            return {
                "raw_metadata": {
                    "id": model_info.id,
                    "object": model_info.object,
                    "created": getattr(model_info, "created", None),
                    "owned_by": getattr(model_info, "owned_by", None),
                }
            }
        else:
            # If model not in list, try a test call to get response metadata
            test_response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )

            return {
                "raw_metadata": {
                    "model_used": test_response.model,
                    "usage": test_response.usage.model_dump()
                    if test_response.usage
                    else None,
                    "system_fingerprint": getattr(
                        test_response, "system_fingerprint", None
                    ),
                    "created": getattr(test_response, "created", None),
                }
            }
    except Exception as e:
        return {"raw_metadata": {}, "metadata_error": str(e)}


def get_deepseek_actual_metadata(client, model_id: str) -> Dict[str, Any]:
    """Get actual DeepSeek model metadata from API"""
    try:
        # Try to get models list
        models_response = client.models.list()
        model_info = None

        for model in models_response.data:
            if model.id == model_id:
                model_info = model
                break

        if model_info:
            return {
                "raw_metadata": {
                    "id": model_info.id,
                    "object": model_info.object,
                    "owned_by": getattr(model_info, "owned_by", None),
                }
            }
        else:
            # Test call to get response metadata
            test_response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )

            return {
                "raw_metadata": {
                    "model_used": test_response.model,
                    "usage": test_response.usage.model_dump()
                    if test_response.usage
                    else None,
                    "created": getattr(test_response, "created", None),
                }
            }
    except Exception as e:
        return {"raw_metadata": {}, "metadata_error": str(e)}


def get_openrouter_actual_metadata(client, model_id: str) -> Dict[str, Any]:
    """Get actual OpenRouter model metadata from API"""
    try:
        # OpenRouter provides a models endpoint
        try:
            api_key = client.api_key
            headers = {"Authorization": f"Bearer {api_key}"}

            models_response = requests.get(
                "https://openrouter.ai/api/v1/models", headers=headers, timeout=10
            )

            if models_response.status_code == 200:
                models_data = models_response.json()
                for model in models_data.get("data", []):
                    if model.get("id") == model_id:
                        return {
                            "raw_metadata": {
                                "id": model.get("id"),
                                "name": model.get("name"),
                                "description": model.get("description"),
                                "pricing": model.get("pricing"),
                                "context_length": model.get("context_length"),
                                "architecture": model.get("architecture"),
                                "top_provider": model.get("top_provider"),
                                "per_request_limits": model.get("per_request_limits"),
                            }
                        }
        except:
            pass

        # Fallback to test call
        test_response = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": "test"}], max_tokens=1
        )

        return {
            "raw_metadata": {
                "model_used": test_response.model,
                "usage": test_response.usage.model_dump()
                if test_response.usage
                else None,
                "created": getattr(test_response, "created", None),
            }
        }

    except Exception as e:
        return {"raw_metadata": {}, "metadata_error": str(e)}


def log_models_metadata(
    models_config: Dict, output_dir, openai_client, deepseek_client, openrouter_client
):
    """
    Log actual model metadata from APIs to JSON file.
    Supports both old and new model config formats.
    """
    metadata_log = {
        "experiment_timestamp": datetime.now().isoformat(),
        "models_metadata": {},
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "library_versions": get_library_versions(),
        },
    }

    print("🔍 Retrieving actual model metadata from APIs...")

    for display_name, model_entry in models_config.items():
        try:
            client_type, model_id = resolve_model_info(display_name, model_entry)
            client = get_client_for_provider(client_type)

            if not client:
                print(f"   ❌ {display_name}: No client available")
                metadata_log["models_metadata"][display_name] = {
                    "error": "No client available",
                    "model_id": model_id,
                    "client_type": client_type,
                }
                continue

            print(f"   📋 Fetching metadata for {display_name}...")
            model_metadata = get_actual_model_metadata(client_type, model_id, client)
            metadata_log["models_metadata"][display_name] = model_metadata

            status = "✅" if model_metadata["api_accessible"] else "❌"
            print(f"      {status} {display_name} ({model_id})")

            if model_metadata.get("error"):
                print(f"         Error: {model_metadata['error']}")
            elif model_metadata.get("raw_metadata"):
                raw_meta = model_metadata["raw_metadata"]
                if "context_length" in raw_meta:
                    print(f"         Context: {raw_meta['context_length']:,} tokens")
                if "pricing" in raw_meta:
                    print("         Pricing info available")

        except Exception as e:
            metadata_log["models_metadata"][display_name] = {
                "error": str(e),
                "model_entry": str(model_entry),
            }
            print(f"   ❌ {display_name}: {e}")

    metadata_file = output_dir / "models_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_log, f, indent=2, default=str)

    print(f"📄 Model metadata saved to: {metadata_file}")
    return metadata_log


def get_library_versions():
    """Get key library versions"""
    versions = {}

    libraries = ["openai", "requests", "pandas", "numpy", "tqdm"]

    for lib in libraries:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "Unknown")
        except ImportError:
            versions[lib] = "Not installed"

    return versions


def run_llm_reasoning(
    df,
    ontology_base_path,
    models=None,
    model_params=None,
    context_mode="ttl",
    max_workers=8,
    batch_size=25,
    checkpoint_frequency=50,
    max_api_calls=None,
    question_column="Question",
    save_detailed_metrics=True,
    output_dir=None,
    silent_mode=False,
):
    """Optimized for heavy ontologies and thousands of questions"""
    global \
        completed_tasks, \
        total_tasks, \
        display_results, \
        questions_completed, \
        stop_reason

    if models is None:
        models = MODELS
    if model_params is None:
        model_params = get_model_params(models)
    checkpoint_frequency = max(1, int(checkpoint_frequency))
    stop_requested.clear()
    stop_reason = None

    for display_name in models:
        base_cols = [
            "_response",
            "_final_answer",
            "_confidence_score",
            "_reasoning_steps_complexity",
            "_response_time",
            "_token_count",
            "_quality_correctness",
        ]

        for col in base_cols:
            column_name = f"{display_name}{col}"
            if column_name not in df.columns:
                if "response" in col or "answer" in col:
                    df[column_name] = ""
                else:
                    df[column_name] = 0.0
            elif "response" in col or "answer" in col:
                df[column_name] = df[column_name].fillna("")
            else:
                df[column_name] = df[column_name].fillna(0.0)

    total_questions = len(df)
    pending_tasks = []
    for idx, row in df.iterrows():
        for display_name, model_entry in models.items():
            if is_completed_response(row.get(f"{display_name}_response", "")):
                continue
            pending_tasks.append((idx, row, display_name, model_entry))

    if max_api_calls is not None:
        pending_tasks = pending_tasks[:max_api_calls]

    total_tasks = len(pending_tasks)
    completed_tasks = 0
    questions_completed = count_completed_questions(df, models.keys())
    display_results.clear()

    print(f"🚀 Processing {total_questions} questions with {len(models)} models...")
    print(f"📊 Pending API calls: {total_tasks}")
    print(f"💾 Already completed questions: {questions_completed}/{total_questions}")
    print(f"⚙️ Max workers: {max_workers}")
    print(f"💾 Checkpoint frequency: Every {checkpoint_frequency} questions")
    if max_api_calls is not None:
        print(f"🧯 API call cap for this run: {max_api_calls}")
    print(f"🔇 Silent mode: {'ON' if silent_mode else 'OFF'}")
    print(f"🧠 Initial memory usage: {monitor_memory():.1f}%")

    pbar = tqdm(total=total_tasks, desc="API Calls", unit="calls")
    total_batches = (len(pending_tasks) + batch_size - 1) // batch_size
    detailed_metrics, logs = [], []

    for batch_idx in range(total_batches):
        if stop_requested.is_set():
            print(f"\nStopping before next batch: {stop_reason or 'stop requested.'}")
            break

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pending_tasks))
        df_batch_tasks = pending_tasks[start_idx:end_idx]

        print(
            f"\n🔄 Processing batch {batch_idx + 1}/{total_batches} (pending calls {start_idx + 1}-{end_idx})"
        )
        print(f"🧠 Memory usage: {monitor_memory():.1f}%")

        batch_tasks = []
        for idx, row, display_name, model_entry in df_batch_tasks:
            task_args = (
                idx,
                row,
                display_name,
                model_entry,
                question_column,
                ontology_base_path,
                context_mode,
                model_params,
            )
            batch_tasks.append(task_args)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task_args in batch_tasks:
                if stop_requested.is_set():
                    break

                idx, row, display_name, model_entry = (
                    task_args[0],
                    task_args[1],
                    task_args[2],
                    task_args[3],
                )
                query = row.get(question_column, "")
                expected_answer = row.get("Answer", "Unknown")

                future = executor.submit(process_single_model_request, task_args)
                future.add_done_callback(
                    lambda f, model=display_name, question=query, expected=expected_answer, q_idx=idx: (
                        update_progress(
                            f,
                            pbar,
                            model,
                            question,
                            expected,
                            q_idx,
                            df,
                            logs,
                            detailed_metrics,
                            output_dir,
                            list(models.keys()),
                            total_questions,
                            checkpoint_frequency,
                            silent_mode,
                        )
                    )
                )
                future_to_task[future] = task_args

            for future in concurrent.futures.as_completed(future_to_task):
                if stop_requested.is_set():
                    for pending_future in future_to_task:
                        if pending_future is not future and not pending_future.done():
                            pending_future.cancel()
                    break

        force_cleanup()

        memory_usage = monitor_memory()
        if memory_usage > 85:
            print(f"⚠️ High memory usage ({memory_usage:.1f}%), forcing cleanup...")
            force_cleanup()
            time.sleep(2)

    pbar.close()

    if output_dir:
        if stop_requested.is_set() and stop_reason and "No space left" in stop_reason:
            print("\nSkipping final save because the device is out of space.")
            final_file = None
        else:
            final_file = save_checkpoint_csv(
                df,
                logs,
                detailed_metrics,
                output_dir,
                questions_completed,
                total_questions,
                list(models.keys()),
            )
        if final_file is not None:
            print(f"\nFinal results saved to: {final_file}")
        else:
            print("\nFinal results were not saved because checkpoint saving failed.")

    if not silent_mode:
        display_recent_summary(display_results, list(models.keys()), sample_size=5)

    if stop_requested.is_set():
        print(f"\nSTOPPED EARLY: {stop_reason or 'stop requested.'}")
    else:
        print("\n🎉 EXPERIMENT COMPLETED!")
    print(f"📊 Total: {questions_completed}/{total_questions} questions processed")
    print(f"⏱️ Total API calls: {completed_tasks}/{total_tasks}")
    print(f"🧠 Final memory usage: {monitor_memory():.1f}%")

    return df, logs, detailed_metrics, {}


# Keep existing functions for compatibility
def resume_failed_queries(
    df,
    failed_indices,
    ontology_base_path,
    models=None,
    model_params=None,
    context_mode="ttl",
    question_column="Question",
    enable_deepeval=False,
):
    # Implementation remains the same
    pass


def calculate_model_performance_summary(df, models):
    """Calculate comprehensive performance summary"""
    summary = {}

    for model_name in models.keys():
        response_col = f"{model_name}_response"
        if response_col in df.columns:
            responses = df[response_col].astype(str)
            non_empty = (responses != "").sum()
            errors = responses.str.startswith("[ERROR]").sum()
            success = non_empty - errors

            correctness_col = f"{model_name}_quality_correctness"
            completeness_col = f"{model_name}_quality_completeness"
            structure_col = f"{model_name}_quality_structure_compliance"
            reasoning_col = f"{model_name}_quality_reasoning_quality"
            time_col = f"{model_name}_response_time"

            summary[model_name] = {
                "response_metrics": {
                    "total_questions": len(df),
                    "responses": int(non_empty),
                    "successful": int(success),
                    "errors": int(errors),
                    "success_rate": float(success / len(df)) if len(df) > 0 else 0.0,
                },
                "quality_metrics": {},
                "timing_metrics": {},
            }

            for col, metric_name in [
                (correctness_col, "correctness"),
                (completeness_col, "completeness"),
                (structure_col, "structure_compliance"),
                (reasoning_col, "reasoning_quality"),
            ]:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce")
                    values = values[values > 0]
                    if len(values) > 0:
                        summary[model_name]["quality_metrics"][metric_name] = {
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }

            if time_col in df.columns:
                times = pd.to_numeric(df[time_col], errors="coerce")
                times = times[times > 0]
                if len(times) > 0:
                    summary[model_name]["timing_metrics"] = {
                        "avg_response_time": float(times.mean()),
                        "median_response_time": float(times.median()),
                        "min_response_time": float(times.min()),
                        "max_response_time": float(times.max()),
                    }

    return summary


# Export key functions for use in experiment scripts
__all__ = [
    "run_llm_reasoning",
    "resume_failed_queries",
    "calculate_model_performance_summary",
    "log_models_metadata",
    "check_api_clients",
    "openai_client",
    "deepseek_client",
    "openrouter_client",
    "MODELS",
    "get_model_params",
]
