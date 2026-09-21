#!/usr/bin/env python3
"""Resumable, fail-closed runner for the frozen v1.1 pending manifest.

The default invocation is validation-only. Calls require ``--execute`` and an
explicit, authorized OpenRouter model ID. Frozen global authorization flags stay
false so they cannot accidentally authorize GPT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
sys.path.insert(0, str(ROOT / "scripts"))
from phase7a_preflight import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    parse_response,
    read_csv,
)
from phase6_materialize_release import create_context_specific_prompt  # noqa: E402
from phase7d_finalize_experiment import CONFIG_VERSION  # noqa: E402

DEFAULT_OUTPUT = ROOT / "release" / "v1.1.0-phase7d" / "responses" / "new_observations.jsonl"
EXPECTED_CONFIGURATION_HASH = "c1562554bd9e252bf97356ef098edde997f813536dd8e547f35f5981a1023df3"


MODEL_KEYS = {
    "GPT-5 mini": "OPENAI_API_KEY",
    "Gemini 2.5 Flash-Lite": "OPENROUTER_API_KEY",
    "Qwen3-30B-A3B-Instruct": "OPENROUTER_API_KEY",
}
ROUTING_CONTRACT = {
    "GPT-5 mini": ("gpt-5-mini-2025-08-07", "OpenAI", "OPENAI_API_KEY"),
    "Gemini 2.5 Flash-Lite": ("google/gemini-2.5-flash-lite", "OpenRouter", "OPENROUTER_API_KEY"),
    "Qwen3-30B-A3B-Instruct": ("qwen/qwen3-30b-a3b-instruct-2507", "OpenRouter", "OPENROUTER_API_KEY"),
}
AUTHORIZED_EXECUTION_MODEL_IDS = frozenset({
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3-30b-a3b-instruct-2507",
})
EXPECTED_OPENROUTER_BACKENDS = {
    "google/gemini-2.5-flash-lite": "Google AI Studio",
    "qwen/qwen3-30b-a3b-instruct-2507": "DekaLLM",
}
write_lock = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_fingerprint(row: dict[str, str], model_config: dict[str, Any]) -> str:
    payload = {
        "input_hash": row["input_hash"],
        "config_version": CONFIG_VERSION,
        "model": model_config,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def observation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["task_id"]), str(row["representation"]), str(row["model"])


def load_terminal_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") in {"usable", "malformed_response"}:
                key = observation_key(record)
                if key in keys:
                    raise ValueError(f"Duplicate accepted observation at line {line_number}: {key}")
                keys.add(key)
    return keys


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(record) + "\n"
    with write_lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def task_payloads() -> dict[tuple[str, str], tuple[str, str, str]]:
    import pyarrow.parquet as pq

    columns = [
        "task_id", "task_group", "answer_type", "nl_question", "nl_context",
        "fs_query", "fs_context", "ar_question", "ar_context",
    ]
    result: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in pq.read_table(STAGE / "core_llm_bench_v1_1.parquet", columns=columns).to_pylist():
        for representation, q_field, c_field in (
            ("NL", "nl_question", "nl_context"),
            ("FS", "fs_query", "fs_context"),
            ("AR", "ar_question", "ar_context"),
        ):
            result[(str(row["task_id"]), representation)] = (
                row[q_field], row[c_field], row["answer_type"]
            )
    return result


def classify_exception(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    if "timeout" in text:
        return "timeout"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if any(token in text for token in ("500", "502", "503", "504", "connection", "temporar")):
        return "provider_transient_error"
    return "nonretryable_provider_error"


def client_for(model_config: dict[str, Any]):
    from openai import OpenAI

    provider = model_config["api_provider"]
    key_name = model_config["credential_source"]
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable {key_name}")
    if provider == "OpenAI":
        return OpenAI(api_key=api_key, max_retries=0)
    if provider == "OpenRouter":
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    raise ValueError(f"Unsupported API provider: {provider}")


def request_parameters(model_config: dict[str, Any]) -> dict[str, Any]:
    return dict(model_config["request_parameters"])


def audit_manifest_integrity(config: dict[str, Any]) -> dict[str, Any]:
    pending = read_csv(STAGE / "primary_experiment_manifest.csv")
    if len(pending) != 81432:
        raise ValueError(f"Primary manifest must contain 81,432 rows, got {len(pending)}")
    if {row["configuration_hash"] for row in pending} != {config["configuration_hash"]}:
        raise ValueError("Primary manifest configuration hash drift")
    if {row["status"] for row in pending} != {"pending"}:
        raise ValueError("Primary manifest contains a non-pending observation")
    for row in pending:
        expected_route = ROUTING_CONTRACT[row["model"]]
        actual_route = (row["model_id"], row["api_provider"], row["credential_source"])
        if actual_route != expected_route:
            raise ValueError(f"Primary manifest routing violation for {row['model']}")
    expected = {
        (str(task_id), representation, model)
        for task_id in range(1, 9049)
        for representation in ("NL", "FS", "AR")
        for model in MODEL_KEYS
    }
    if {observation_key(row) for row in pending} != expected:
        raise ValueError("Primary manifest is not a complete task/representation/model matrix")
    return {"pending_rows": len(pending)}


def execute_one(
    row: dict[str, str],
    payload: tuple[str, str, str],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    model_config = config["models"][row["model"]]
    if model_config["model_id"] not in AUTHORIZED_EXECUTION_MODEL_IDS:
        raise RuntimeError(
            f"Execution model is not authorized: {model_config['model_id']}"
        )
    question, context, answer_type = payload
    context_mode = "inline_owl" if row["representation"] == "FS" else "inline_nl"
    prompt = create_context_specific_prompt(question, context, context_mode, answer_type)
    client = client_for(model_config)
    retry = config["retry_policy"]
    maximum_attempts = 1 + int(retry["maximum_retries_after_initial_attempt"])
    backoff = list(retry["backoff_seconds"])
    fingerprint = request_fingerprint(row, model_config)
    attempt_output = output.with_name(f"{output.stem}_attempts{output.suffix}")
    last_record: dict[str, Any] = {}
    for attempt in range(1, maximum_attempts + 1):
        requested_at = datetime.now(timezone.utc).isoformat()
        try:
            response = client.chat.completions.create(
                model=model_config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                timeout=config["timeout_seconds"],
                **request_parameters(model_config),
            )
            raw_provider_response = response.model_dump(mode="json")
            content = response.choices[0].message.content or ""
            if not content.strip():
                last_record = {
                    **{key: row[key] for key in (
                        "task_id", "semantic_key", "dataset", "hop", "task",
                        "representation", "model", "input_hash",
                    )},
                    "config_version": config["config_version"],
                    "configuration_hash": config["configuration_hash"],
                    "requested_model_identifier": model_config["model_id"],
                    "api_provider": model_config["api_provider"],
                    "credential_source": model_config["credential_source"],
                    "request_fingerprint": fingerprint,
                    "request_timestamp": requested_at,
                    "attempt": attempt,
                    "technical_retry_count": attempt - 1,
                    "returned_model_identifier": getattr(response, "model", None),
                    "returned_provider": raw_provider_response.get("provider"),
                    "raw_provider_response": raw_provider_response,
                    "raw_response_text": "",
                    "parsed_response": None,
                    "status": "technical_failure",
                    "error_type": "empty_response",
                    "error": "Provider returned an empty response",
                }
                append_checkpoint(attempt_output, last_record)
                if attempt == maximum_attempts:
                    append_checkpoint(output, last_record)
                    return last_record
                delay = backoff[min(attempt - 1, len(backoff) - 1)]
                time.sleep(delay)
                continue
            parsed = parse_response(content, row["task"])
            status = (
                "usable"
                if parsed["status"] == "requested_schema_conformant"
                else "malformed_response"
            )
            record = {
                **{key: row[key] for key in (
                    "task_id", "semantic_key", "dataset", "hop", "task",
                    "representation", "model", "input_hash",
                )},
                "config_version": config["config_version"],
                "configuration_hash": config["configuration_hash"],
                "requested_model_identifier": model_config["model_id"],
                "api_provider": model_config["api_provider"],
                "credential_source": model_config["credential_source"],
                "request_fingerprint": fingerprint,
                "request_timestamp": requested_at,
                "attempt": attempt,
                "technical_retry_count": attempt - 1,
                "returned_model_identifier": getattr(response, "model", None),
                "returned_provider": raw_provider_response.get("provider"),
                "raw_provider_response": raw_provider_response,
                "raw_response_text": content,
                "parsed_response": parsed,
                "status": status,
                "error_type": None,
                "error": None,
            }
            append_checkpoint(output, record)
            return record
        except Exception as error:  # provider SDK exception hierarchy varies
            failure_type = classify_exception(error)
            retryable = failure_type in set(retry["retryable"])
            last_record = {
                **{key: row[key] for key in (
                    "task_id", "semantic_key", "dataset", "hop", "task",
                    "representation", "model", "input_hash",
                )},
                "config_version": config["config_version"],
                "configuration_hash": config["configuration_hash"],
                "requested_model_identifier": model_config["model_id"],
                "api_provider": model_config["api_provider"],
                "credential_source": model_config["credential_source"],
                "request_fingerprint": fingerprint,
                "request_timestamp": requested_at,
                "attempt": attempt,
                "technical_retry_count": attempt - 1,
                "returned_model_identifier": None,
                "returned_provider": None,
                "raw_provider_response": None,
                "raw_response_text": "",
                "parsed_response": None,
                "status": "technical_failure",
                "error_type": failure_type,
                "error": str(error),
            }
            append_checkpoint(attempt_output, last_record)
            if not retryable or attempt == maximum_attempts:
                append_checkpoint(output, last_record)
                return last_record
            delay = backoff[min(attempt - 1, len(backoff) - 1)]
            time.sleep(delay)
    return last_record


def validate_config(
    config: dict[str, Any],
    execute: bool,
    selected_model_ids: set[str] | None = None,
) -> None:
    if config.get("config_version") != CONFIG_VERSION:
        raise ValueError("Unexpected experiment configuration version")
    if config.get("configuration_hash") != EXPECTED_CONFIGURATION_HASH:
        raise ValueError("Unexpected frozen configuration hash")
    if execute:
        requested = selected_model_ids or set()
        if not requested:
            raise RuntimeError("Execution requires an explicit model allowlist")
        unauthorized = requested - AUTHORIZED_EXECUTION_MODEL_IDS
        if unauthorized:
            raise RuntimeError(
                f"Execution model is not authorized: {sorted(unauthorized)!r}"
            )
        if config.get("execution_authorized") or config.get("ready_for_execution"):
            raise RuntimeError(
                "Global execution flags must remain false for partial model authorization"
            )
    if config["prompt"]["version"] != PROMPT_TEMPLATE_VERSION:
        raise ValueError("Prompt template version drift")
    for model_name, expected in ROUTING_CONTRACT.items():
        model = config["models"][model_name]
        actual = (model.get("model_id"), model.get("api_provider"), model.get("credential_source"))
        if actual != expected:
            raise ValueError(f"Fail-closed routing violation for {model_name}: {actual!r}")
        if model.get("provider") != expected[1]:
            raise ValueError(f"Provider/API disagreement for {model_name}")
        if model.get("fallback") is not False:
            raise ValueError(f"Fallback must remain disabled for {model_name}")
    for model_id in AUTHORIZED_EXECUTION_MODEL_IDS:
        model = next(value for value in config["models"].values() if value["model_id"] == model_id)
        provider_policy = model.get("request_parameters", {}).get("extra_body", {}).get("provider", {})
        if provider_policy.get("allow_fallbacks") is not False:
            raise ValueError(f"OpenRouter fallback is not fail-closed for {model_id}")
        if provider_policy.get("order") != [model["provider_tag"]]:
            raise ValueError(f"Pinned OpenRouter provider mismatch for {model_id}")


def validate_frozen_preflight(config: dict[str, Any]) -> dict[str, Any]:
    validation = json.loads((STAGE / "validation_report.json").read_text(encoding="utf-8"))
    required = {
        "membership_rows": 9048,
        "matrix_rows": 81432,
        "pending_rows": 81432,
        "configuration_hash": EXPECTED_CONFIGURATION_HASH,
        "all_matrix_input_hashes_match": True,
        "all_rows_reference_one_configuration_hash": True,
        "release_published": False,
        "v1_0_modified": False,
    }
    drift = {
        key: {"expected": expected, "actual": validation.get(key)}
        for key, expected in required.items()
        if validation.get(key) != expected
    }
    incompatible = sum(
        value.get("incompatible_groups", -1)
        for value in validation.get("duplicate_audit", {}).values()
    )
    if incompatible != 0:
        drift["incompatible_input_gold_groups"] = {"expected": 0, "actual": incompatible}
    integrity = audit_manifest_integrity(config)
    rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    counts = {
        model_id: sum(row["model_id"] == model_id for row in rows)
        for model_id in {route[0] for route in ROUTING_CONTRACT.values()}
    }
    if set(counts.values()) != {27144}:
        drift["per_model_pending_rows"] = {"expected": 27144, "actual": counts}
    if drift:
        raise ValueError(f"Frozen preflight validation drift: {drift!r}")
    return {
        **integrity,
        "membership_rows": 9048,
        "representations": ["NL", "FS", "AR"],
        "pending_by_model_id": counts,
        "configuration_hash": config["configuration_hash"],
        "incompatible_input_gold_groups": incompatible,
    }


def validate_canary(path: Path, model_id: str) -> dict[str, Any]:
    if model_id not in AUTHORIZED_EXECUTION_MODEL_IDS:
        raise RuntimeError(f"Canary model is not authorized: {model_id}")
    rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    expected_rows = [row for row in rows if row["model_id"] == model_id][:10]
    expected_by_key = {observation_key(row): row for row in expected_rows}
    records: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    failures: list[str] = []
    if len(records) != 10:
        failures.append(f"expected 10 checkpoint rows, found {len(records)}")
    seen: set[tuple[str, str, str]] = set()
    expected_backend = EXPECTED_OPENROUTER_BACKENDS[model_id]
    for index, record in enumerate(records, 1):
        record_key = observation_key(record)
        manifest_row = expected_by_key.get(record_key)
        if manifest_row is None:
            failures.append(f"row {index}: not one of the first 10 pending manifest rows")
            continue
        if record_key in seen:
            failures.append(f"row {index}: duplicate accepted observation {record_key!r}")
        seen.add(record_key)
        checks = {
            "status": record.get("status") in {"usable", "malformed_response"},
            "requested model": record.get("requested_model_identifier") == model_id,
            "returned model": record.get("returned_model_identifier") == model_id,
            "provider": record.get("api_provider") == "OpenRouter",
            "credential": record.get("credential_source") == "OPENROUTER_API_KEY",
            "returned provider": record.get("returned_provider") == expected_backend,
            "input hash": record.get("input_hash") == manifest_row["input_hash"],
            "configuration hash": record.get("configuration_hash") == EXPECTED_CONFIGURATION_HASH,
            "raw response": bool(record.get("raw_provider_response")),
            "nonempty response": bool(str(record.get("raw_response_text", "")).strip()),
            "parsed answer": bool((record.get("parsed_response") or {}).get("answer")),
            "parsed confidence": (record.get("parsed_response") or {}).get("confidence") is not None,
        }
        for label, passed in checks.items():
            if not passed:
                failures.append(f"row {index}: {label} check failed")
        raw = record.get("raw_provider_response") or {}
        usage = raw.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        message = ((raw.get("choices") or [{}])[0].get("message") or {})
        if reasoning_tokens not in {None, 0}:
            failures.append(f"row {index}: unexpected reasoning tokens {reasoning_tokens}")
        if message.get("reasoning") not in {None, ""}:
            failures.append(f"row {index}: unexpected reasoning output")
    missing = set(expected_by_key) - seen
    if missing:
        failures.append(f"missing first-10 manifest keys: {sorted(missing)!r}")
    report = {
        "status": "passed" if not failures else "failed",
        "model_id": model_id,
        "expected_backend": expected_backend,
        "checkpoint_rows": len(records),
        "unique_expected_rows": len(seen & set(expected_by_key)),
        "failures": failures,
    }
    if failures:
        raise ValueError(f"Canary validation failed: {report!r}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=STAGE / "experiment_config_v1_1.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Explicit execution allowlist; repeat once per requested model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Execute only the first N pending rows in frozen manifest order",
    )
    parser.add_argument(
        "--validate-canary",
        action="store_true",
        help="Validate that the output is exactly the successful first-10 canary",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_model_ids = set(args.model_id)
    validate_config(config, args.execute, selected_model_ids)
    integrity = validate_frozen_preflight(config)
    if (args.execute or args.validate_canary) and len(selected_model_ids) != 1:
        raise RuntimeError("Exactly one authorized model ID is required per independent run")
    if args.validate_canary:
        print(json.dumps(validate_canary(args.output, next(iter(selected_model_ids))), indent=2))
        return 0
    pending = read_csv(STAGE / "primary_experiment_manifest.csv")
    terminal = load_terminal_keys(args.output)
    selected_rows = [
        row for row in pending
        if not selected_model_ids or row["model_id"] in selected_model_ids
    ]
    remaining = [row for row in selected_rows if observation_key(row) not in terminal]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        remaining = remaining[:args.limit]
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "validation-only",
                "pending_manifest_rows": integrity["pending_rows"],
                "selected_model_ids": sorted(selected_model_ids),
                "selected_manifest_rows": len(selected_rows),
                "already_terminal": len(terminal),
                "remaining": len(remaining),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not args.execute:
        return 0
    payloads = task_payloads()
    workers = args.max_workers or 1
    if not 1 <= workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                execute_one, row, payloads[(row["task_id"], row["representation"])],
                config, args.output,
            ): observation_key(row)
            for row in remaining
        }
        for future in as_completed(futures):
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
