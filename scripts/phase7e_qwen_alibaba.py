#!/usr/bin/env python3
"""Fresh, provider-specific Phase 7E Qwen execution through OpenRouter Alibaba.

This script never reads ``OPENAI_API_KEY``. Preparation resolves live OpenRouter
metadata and freezes a Qwen-only configuration/manifest. Execution uses only
``OPENROUTER_API_KEY`` and appends every provider attempt durably.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
PHASE7D = ROOT / "release" / "v1.1.0-phase7d"
PHASE7E = ROOT / "release" / "v1.1.0-phase7e"
RESPONSES = PHASE7E / "responses" / "qwen_alibaba_observations.jsonl"
ATTEMPTS = PHASE7E / "responses" / "qwen_alibaba_attempts.jsonl"
CONFIG_PATH = PHASE7E / "qwen_alibaba_config.json"
MANIFEST_PATH = PHASE7E / "qwen_alibaba_primary_manifest.csv"
METADATA_PATH = PHASE7E / "openrouter_alibaba_endpoint.json"
REPORT_PATH = PHASE7E / "PHASE7E_REPORT.json"
RUN_METADATA_PATH = PHASE7E / "qwen_alibaba_execution_metadata.json"

sys.path.insert(0, str(ROOT / "scripts"))
from phase6_materialize_release import create_context_specific_prompt, prompt_hash  # noqa: E402
from phase7a_preflight import parse_response, read_csv  # noqa: E402
from run_v1_1_experiments import task_payloads  # noqa: E402


MODEL_NAME = "Qwen3-30B-A3B-Instruct"
MODEL_ID = "qwen/qwen3-30b-a3b-instruct-2507"
PROVIDER_TAG = "alibaba"
PROVIDER_METADATA_NAME = "Alibaba"
PROVIDER_REQUESTED_NAME = "Alibaba Cloud International"
BASE_CONFIGURATION_HASH = "c1562554bd9e252bf97356ef098edde997f813536dd8e547f35f5981a1023df3"
CONFIG_VERSION = "core-llm-bench-v1.1-qwen-alibaba-phase7e-2"
PRE_SCHEMA_PARQUET_SHA256 = "7285e506483b422acf8e1882da5b5390966ba4896b4bb020820f6bb4a5218859"
CANONICAL_PARQUET_SHA256 = "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc"
MODEL_INPUT_MANIFEST_SHA256 = "191c1c0dc221a5829dfa361f86fd1ebf8bbfd54e8f89eeaf0621810001715d18"
CANARY_SIZE = 12
ACCEPTED = {"usable", "malformed_response"}
REQUIRED_PARAMETERS = {
    "max_tokens", "temperature", "top_p", "seed",
    "presence_penalty", "frequency_penalty",
}
WRITE_LOCK = threading.Lock()


class ProviderIdentityError(RuntimeError):
    """A successful response came from an unauthorized model/provider."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["task_id"]), str(row["representation"]), str(row["model"])


def openrouter_metadata() -> dict[str, Any]:
    url = f"https://openrouter.ai/api/v1/models/{MODEL_ID}/endpoints"
    request = urllib.request.Request(url, headers={"User-Agent": "CORE-LLM-Bench/phase7e"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)["data"]


def selected_endpoint(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("id") != MODEL_ID:
        raise RuntimeError("OpenRouter metadata returned a different model")
    matches = [item for item in metadata.get("endpoints", []) if item.get("tag") == PROVIDER_TAG]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {PROVIDER_TAG!r} endpoint, got {len(matches)}")
    endpoint = matches[0]
    if endpoint.get("provider_name") != PROVIDER_METADATA_NAME:
        raise RuntimeError(f"Unexpected provider name: {endpoint.get('provider_name')!r}")
    if endpoint.get("status") != 0:
        raise RuntimeError(f"Alibaba endpoint is not available: status={endpoint.get('status')!r}")
    supported = set(endpoint.get("supported_parameters") or [])
    missing = REQUIRED_PARAMETERS - supported
    if missing:
        raise RuntimeError(f"Alibaba endpoint lacks frozen parameters: {sorted(missing)!r}")
    if int(endpoint.get("max_completion_tokens") or 0) < 1024:
        raise RuntimeError("Alibaba endpoint cannot provide 1,024 output tokens")
    return endpoint


def configuration_basis(endpoint: dict[str, Any]) -> dict[str, Any]:
    base = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    if base.get("configuration_hash") != BASE_CONFIGURATION_HASH:
        raise RuntimeError("Frozen Phase 7D configuration hash drift")
    qwen = base["models"][MODEL_NAME]
    expected_parameters = {
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_p": 0.9,
        "seed": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.1,
    }
    actual_parameters = {
        key: qwen["request_parameters"].get(key) for key in expected_parameters
    }
    if actual_parameters != expected_parameters:
        raise RuntimeError("Frozen Qwen request parameter drift")
    if base.get("timeout_seconds") != 30:
        raise RuntimeError("Frozen timeout drift")
    return {
        "config_version": CONFIG_VERSION,
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "api_provider": "OpenRouter",
        "credential_source": "OPENROUTER_API_KEY",
        "provider_requested_name": PROVIDER_REQUESTED_NAME,
        "provider_tag": PROVIDER_TAG,
        "provider_metadata_name": PROVIDER_METADATA_NAME,
        "fallback": False,
        "require_parameters": True,
        "model_behavior": "non-thinking instruct model",
        "request_parameters": {
            **expected_parameters,
            "extra_body": {
                "provider": {
                    "order": [PROVIDER_TAG],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "data_collection": "deny",
                }
            },
        },
        "timeout_seconds": base["timeout_seconds"],
        "retry_policy": base["retry_policy"],
        "prompt": base["prompt"],
        "parser": "phase7a_preflight.parse_response",
        "source_configuration_hash": BASE_CONFIGURATION_HASH,
        "endpoint_capability_snapshot": {
            "tag": endpoint.get("tag"),
            "provider_name": endpoint.get("provider_name"),
            "status": endpoint.get("status"),
            "supported_parameters": sorted(endpoint.get("supported_parameters") or []),
            "context_length": endpoint.get("context_length"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "pricing": endpoint.get("pricing"),
        },
        "benchmark_hash_transition": {
            "pre_schema_parquet_sha256": PRE_SCHEMA_PARQUET_SHA256,
            "canonical_v1_1_parquet_sha256": CANONICAL_PARQUET_SHA256,
            "model_input_manifest_sha256": MODEL_INPUT_MANIFEST_SHA256,
            "change_scope": "schema-only; model-facing NL/FS/AR inputs unchanged",
            "model_inputs_regenerated": False,
        },
    }


def frozen_files() -> dict[str, str]:
    paths = {
        "benchmark_parquet": STAGE / "core_llm_bench_v1_1.parquet",
        "primary_manifest": STAGE / "primary_experiment_manifest.csv",
        "model_input_manifest": STAGE / "model_input_manifest.csv",
        "staging_config": STAGE / "experiment_config_v1_1.json",
        "gemini_observations": PHASE7D / "responses" / "gemini_observations.jsonl",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def relevant_runner_status() -> list[str]:
    output = git_output(
        "status", "--short", "--", "scripts/phase7e_qwen_alibaba.py",
        "tests/test_phase7e_qwen_alibaba.py", "release/v1.1.0-phase7e",
    )
    return output.splitlines() if output else []


def write_execution_metadata(config: dict[str, Any], authenticated: bool = False) -> dict[str, Any]:
    previous = (
        json.loads(RUN_METADATA_PATH.read_text(encoding="utf-8"))
        if RUN_METADATA_PATH.is_file() else {}
    )
    runner_hash = sha256_file(Path(__file__))
    runner_history = list(previous.get("execution_runner_sha256_history") or [])
    previous_runner = previous.get("runner_sha256")
    if previous_runner and previous_runner not in runner_history:
        runner_history.append(previous_runner)
    if runner_hash not in runner_history:
        runner_history.append(runner_hash)
    metadata = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "relevant_runner_files_modified": relevant_runner_status(),
        "runner_sha256": runner_hash,
        "execution_runner_sha256_history": runner_history,
        "config_sha256": sha256_file(CONFIG_PATH),
        "pre_schema_parquet_sha256": PRE_SCHEMA_PARQUET_SHA256,
        "canonical_v1_1_parquet_sha256": sha256_file(STAGE / "core_llm_bench_v1_1.parquet"),
        "model_input_manifest_sha256": sha256_file(STAGE / "model_input_manifest.csv"),
        "parquet_hash_change_scope": "schema-only; model-facing NL/FS/AR inputs unchanged",
        "authentication_material_loaded": authenticated,
        "non_secret_effective_config": {
            "model": config["model_id"],
            "provider_order": [config["provider_tag"]],
            "fallback": config["fallback"],
            "request_parameters": config["request_parameters"],
            "timeout_seconds": config["timeout_seconds"],
            "retry_policy": config["retry_policy"],
        },
    }
    write_json(RUN_METADATA_PATH, metadata)
    return metadata


def select_canary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    required = {
        *(f"representation:{value}" for value in ("NL", "FS", "AR")),
        *(f"task:{value}" for value in ("BQA", "OEQA")),
        *(f"hop:{value}" for value in ("1hop", "2hop")),
        *(f"dataset:{value}" for value in ("FamilyOWL", "Pizza100", "Pizza250", "OWL2Bench")),
    }

    def features(row: dict[str, str]) -> set[str]:
        return {
            f"representation:{row['representation']}", f"task:{row['task']}",
            f"hop:{row['hop']}", f"dataset:{row['dataset']}",
        }

    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    uncovered = set(required)
    while uncovered:
        candidate = max(rows, key=lambda row: len(features(row) & uncovered))
        gain = features(candidate) & uncovered
        if not gain:
            raise RuntimeError(f"Canary coverage unavailable: {sorted(uncovered)!r}")
        selected.append(candidate)
        selected_keys.add(observation_key(candidate))
        uncovered -= gain
    for index in range(CANARY_SIZE):
        candidate = rows[(index * len(rows)) // CANARY_SIZE]
        if observation_key(candidate) not in selected_keys:
            selected.append(candidate)
            selected_keys.add(observation_key(candidate))
        if len(selected) == CANARY_SIZE:
            break
    if len(selected) < CANARY_SIZE:
        for candidate in rows:
            if observation_key(candidate) not in selected_keys:
                selected.append(candidate)
                selected_keys.add(observation_key(candidate))
            if len(selected) == CANARY_SIZE:
                break
    return selected


def prepare() -> dict[str, Any]:
    if RESPONSES.exists() or ATTEMPTS.exists():
        raise FileExistsError("Refusing to replace an existing Phase 7E checkpoint")
    metadata = openrouter_metadata()
    endpoint = selected_endpoint(metadata)
    basis = configuration_basis(endpoint)
    config_hash = hashlib.sha256(canonical_json(basis).encode("utf-8")).hexdigest()
    config = {
        **basis,
        "configuration_hash": config_hash,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "frozen_file_sha256": frozen_files(),
        "openai_api_key_loaded": False,
    }
    base_rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    rows = [row for row in base_rows if row["model_id"] == MODEL_ID]
    if len(rows) != 27144 or len({observation_key(row) for row in rows}) != 27144:
        raise RuntimeError("Frozen Qwen manifest is not a unique 27,144-row matrix")
    canary_keys = {observation_key(row) for row in select_canary_rows(rows)}
    fields = list(rows[0]) + ["requested_provider", "requested_provider_name", "canary"]
    PHASE7E.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["configuration_hash"] = config_hash
            item["requested_provider"] = PROVIDER_TAG
            item["requested_provider_name"] = PROVIDER_REQUESTED_NAME
            item["canary"] = "true" if observation_key(row) in canary_keys else "false"
            writer.writerow(item)
    safe_endpoint = {
        key: endpoint.get(key) for key in (
            "name", "model_id", "provider_name", "tag", "status", "context_length",
            "max_completion_tokens", "supported_parameters", "pricing",
        )
    }
    write_json(METADATA_PATH, {
        "resolved_at": config["prepared_at"],
        "requested_model": MODEL_ID,
        "requested_provider_name": PROVIDER_REQUESTED_NAME,
        "resolved_provider_identifier": PROVIDER_TAG,
        "selected_endpoint": safe_endpoint,
        "required_parameters": sorted(REQUIRED_PARAMETERS),
        "availability_check": "passed",
    })
    write_json(CONFIG_PATH, config)
    write_json(PHASE7E / "qwen_alibaba_canary.json", {
        "status": "not_run",
        "accepted": 0,
        "configuration_hash": config_hash,
        "primary_keys": [list(key) for key in sorted(canary_keys)],
        "failures": [],
    })
    write_execution_metadata(config)
    return config


def validate_preflight(config: dict[str, Any]) -> list[dict[str, str]]:
    stored_hash = config["configuration_hash"]
    basis = {key: value for key, value in config.items() if key not in {
        "configuration_hash", "prepared_at", "frozen_file_sha256", "openai_api_key_loaded"
    }}
    if hashlib.sha256(canonical_json(basis).encode("utf-8")).hexdigest() != stored_hash:
        raise RuntimeError("Phase 7E configuration hash is invalid")
    if config.get("credential_source") != "OPENROUTER_API_KEY":
        raise RuntimeError("Unauthorized credential source")
    if config.get("provider_tag") != PROVIDER_TAG or config.get("fallback") is not False:
        raise RuntimeError("Provider pin/fallback drift")
    if config.get("frozen_file_sha256") != frozen_files():
        raise RuntimeError("Frozen benchmark, Gemini, manifest, or staging config changed")
    if sha256_file(STAGE / "core_llm_bench_v1_1.parquet") != CANONICAL_PARQUET_SHA256:
        raise RuntimeError("Canonical v1.1 Parquet hash mismatch")
    if sha256_file(STAGE / "model_input_manifest.csv") != MODEL_INPUT_MANIFEST_SHA256:
        raise RuntimeError("Frozen model-input manifest hash mismatch")
    rows = read_csv(MANIFEST_PATH)
    if len(rows) != 27144 or {row["configuration_hash"] for row in rows} != {stored_hash}:
        raise RuntimeError("Phase 7E primary manifest drift")
    if {row["requested_provider"] for row in rows} != {PROVIDER_TAG}:
        raise RuntimeError("Phase 7E manifest provider drift")
    expected_counts = {
        "rows": 27144, "tasks": 9048,
        "NL": 9048, "FS": 9048, "AR": 9048,
        "BQA": 18096, "OEQA": 9048,
    }
    actual_counts = {
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        **{rep: sum(row["representation"] == rep for row in rows) for rep in ("NL", "FS", "AR")},
        **{task: sum(row["task"] == task for row in rows) for task in ("BQA", "OEQA")},
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"Frozen matrix count mismatch: {actual_counts!r}")
    input_rows = read_csv(STAGE / "model_input_manifest.csv")
    input_hashes = {(row["task_id"], row["representation"]): row["input_hash"] for row in input_rows}
    mismatches = sum(
        row["input_hash"] != input_hashes.get((row["task_id"], row["representation"]))
        for row in rows
    )
    if mismatches:
        raise RuntimeError(f"Frozen matrix has {mismatches} input-hash mismatches")
    return rows


def read_openrouter_key() -> str:
    value = os.getenv("OPENROUTER_API_KEY")
    if value:
        return value
    env_path = ROOT / ".env"
    with env_path.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                candidate = candidate.strip()
                if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                    candidate = candidate[1:-1]
                if candidate:
                    return candidate
                break
    raise RuntimeError("OPENROUTER_API_KEY is unavailable")


def classify_exception(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    if "timeout" in text:
        return "timeout"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if any(token in text for token in ("500", "502", "503", "504", "connection", "temporar")):
        return "provider_transient_error"
    return "nonretryable_provider_error"


def validate_response_identity(returned_model: Any, provider: Any) -> None:
    if returned_model != MODEL_ID:
        raise ProviderIdentityError(
            f"model_mismatch: requested={MODEL_ID!r}, returned={returned_model!r}"
        )
    if provider != PROVIDER_METADATA_NAME:
        raise ProviderIdentityError(
            f"provider_mismatch: requested={PROVIDER_TAG!r}, returned={provider!r}"
        )


def execute_one(
    row: dict[str, str], payload: tuple[str, str, str], config: dict[str, Any], client: Any,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Run stopped before request submission")
    question, context, answer_type = payload
    context_mode = "inline_owl" if row["representation"] == "FS" else "inline_nl"
    prompt = create_context_specific_prompt(question, context, context_mode, answer_type)
    actual_input_hash = prompt_hash(question, context, row["representation"], answer_type)
    if actual_input_hash != row["input_hash"]:
        raise RuntimeError(
            f"Input hash mismatch before request: {row['task_id']}/{row['representation']}"
        )
    retry = config["retry_policy"]
    maximum_attempts = 1 + int(retry["maximum_retries_after_initial_attempt"])
    backoff = list(retry["backoff_seconds"])
    common = {
        **{key: row[key] for key in (
            "task_id", "semantic_key", "dataset", "hop", "task", "representation",
            "model", "input_hash", "configuration_hash",
        )},
        "config_version": config["config_version"],
        "requested_model": MODEL_ID,
        "requested_provider": PROVIDER_TAG,
        "requested_provider_name": PROVIDER_REQUESTED_NAME,
        "api_provider": "OpenRouter",
        "credential_source": "OPENROUTER_API_KEY",
        "seed": config["request_parameters"]["seed"],
        "generation_config": {
            key: value for key, value in config["request_parameters"].items()
            if key != "extra_body"
        },
    }
    last: dict[str, Any] = {}
    for attempt in range(1, maximum_attempts + 1):
        requested_at = datetime.now(timezone.utc).isoformat()
        failure_content = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                timeout=config["timeout_seconds"],
                **config["request_parameters"],
            )
            raw = response.model_dump(mode="json")
            content = response.choices[0].message.content or ""
            returned_model = getattr(response, "model", None)
            observed_provider = raw.get("provider")
            try:
                validate_response_identity(returned_model, observed_provider)
            except ProviderIdentityError as error:
                diagnostic = {
                    **common, "timestamp": requested_at, "attempt": attempt,
                    "technical_retry_count": attempt - 1,
                    "returned_model": returned_model,
                    "observed_provider_backend": observed_provider,
                    "raw_provider_response": raw, "raw_response": content,
                    "parsed_answer": None, "parsed_confidence": None,
                    "parsed_response": None, "status": "rejected_provider_identity",
                    "error_type": "provider_or_model_mismatch", "error": str(error),
                }
                append_jsonl(ATTEMPTS, diagnostic)
                if stop_event is not None:
                    stop_event.set()
                raise
            choice = (raw.get("choices") or [{}])[0]
            embedded_error = choice.get("error")
            if embedded_error or choice.get("finish_reason") == "error":
                failure_type = "provider_transient_error"
                message = f"Provider returned an errored completion: {canonical_json(embedded_error)}"
                raw_response = raw
                failure_content = content
            elif content.strip():
                parsed = parse_response(content, row["task"])
                status = "usable" if parsed["status"] == "requested_schema_conformant" else "malformed_response"
                usage = raw.get("usage") or {}
                last = {
                    **common, "timestamp": requested_at, "attempt": attempt,
                    "technical_retry_count": attempt - 1,
                    "returned_model": returned_model,
                    "observed_provider_backend": observed_provider,
                    "raw_provider_response": raw, "raw_response": content,
                    "parsed_answer": parsed.get("answer"),
                    "parsed_confidence": parsed.get("confidence"),
                    "parsed_response": parsed, "status": status,
                    "parse_schema_status": parsed.get("status"),
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "provider_reported_cost": usage.get("cost"),
                    "finish_reason": response.choices[0].finish_reason,
                    "error_type": None, "error": None,
                }
                append_jsonl(ATTEMPTS, last)
                append_jsonl(RESPONSES, last)
                return last
            else:
                failure_type, message = "empty_response", "Provider returned an empty response"
                raw_response = raw
        except ProviderIdentityError:
            raise
        except Exception as error:  # SDK exception hierarchy varies.
            failure_type, message = classify_exception(error), str(error)
            returned_model, observed_provider, raw_response = None, None, None
        last = {
            **common, "timestamp": requested_at, "attempt": attempt,
            "technical_retry_count": attempt - 1,
            "returned_model": returned_model,
            "observed_provider_backend": observed_provider,
            "raw_provider_response": raw_response, "raw_response": failure_content,
            "parsed_answer": None, "parsed_confidence": None,
            "parsed_response": None, "status": "technical_failure",
            "error_type": failure_type, "error": message,
        }
        append_jsonl(ATTEMPTS, last)
        retryable = failure_type in set(retry["retryable"])
        if not retryable or attempt == maximum_attempts:
            append_jsonl(RESPONSES, last)
            return last
        time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    return last


def validate_canary(config: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    records = load_jsonl(RESPONSES)
    attempts = load_jsonl(ATTEMPTS)
    canary_rows = [row for row in rows if row.get("canary") == "true"]
    expected = {observation_key(row): row for row in canary_rows}
    failures: list[str] = []
    if len(canary_rows) != CANARY_SIZE:
        failures.append(f"expected {CANARY_SIZE} canary manifest rows, found {len(canary_rows)}")
    if len(records) != CANARY_SIZE:
        failures.append(f"expected {CANARY_SIZE} terminal rows, found {len(records)}")
    rate_limits = sum(item.get("error_type") == "rate_limit" for item in attempts)
    if rate_limits >= 2:
        failures.append(f"repeated 429/rate-limit attempts: {rate_limits}")
    if any(item.get("error_type") == "nonretryable_provider_error" for item in attempts):
        failures.append("authentication or other nonretryable provider failure")
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records, 1):
        key = observation_key(record)
        manifest = expected.get(key)
        if manifest is None:
            failures.append(f"row {index}: not in deterministic representative canary")
            continue
        if key in seen:
            failures.append(f"row {index}: duplicate observation")
        seen.add(key)
        raw = record.get("raw_provider_response") or {}
        usage = raw.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        message = ((raw.get("choices") or [{}])[0].get("message") or {})
        checks = {
            "accepted": record.get("status") in ACCEPTED,
            "requested model": record.get("requested_model") == MODEL_ID,
            "returned model": record.get("returned_model") == MODEL_ID,
            "requested provider": record.get("requested_provider") == PROVIDER_TAG,
            "observed provider": record.get("observed_provider_backend") == PROVIDER_METADATA_NAME,
            "input hash": record.get("input_hash") == manifest["input_hash"],
            "configuration hash": record.get("configuration_hash") == config["configuration_hash"],
            "nonempty response": bool(str(record.get("raw_response", "")).strip()),
            "parsing infrastructure": record.get("parsed_response") is not None,
            "zero reasoning tokens": details.get("reasoning_tokens") in {None, 0},
            "no reasoning output": message.get("reasoning") in {None, ""},
            "not truncated": record.get("finish_reason") != "length",
        }
        failures.extend(f"row {index}: {label} failed" for label, ok in checks.items() if not ok)
    if set(expected) != seen:
        failures.append("deterministic representative-canary key set mismatch")
    report = {
        "status": "passed" if not failures else "failed",
        "accepted": sum(record.get("status") in ACCEPTED for record in records),
        "configuration_hash": config["configuration_hash"],
        "primary_keys": [list(observation_key(row)) for row in canary_rows],
        "coverage": {
            field: sorted({row[field] for row in canary_rows})
            for field in ("representation", "task", "hop", "dataset")
        },
        "rate_limit_attempts": rate_limits,
        "failures": failures,
    }
    write_json(PHASE7E / "qwen_alibaba_canary.json", report)
    return report


def execute(limit: int | None, workers: int) -> int:
    from openai import OpenAI

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = validate_preflight(config)
    terminal_records = load_jsonl(RESPONSES)
    terminal: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in terminal_records:
        key = observation_key(record)
        if record.get("status") in ACCEPTED:
            if key in terminal:
                raise RuntimeError(f"Duplicate accepted observation: {key!r}")
            terminal[key] = record
    remaining = [row for row in rows if observation_key(row) not in terminal]
    fresh_canary = not terminal and limit == CANARY_SIZE
    if fresh_canary:
        remaining = [row for row in remaining if row.get("canary") == "true"]
    if limit is not None:
        remaining = remaining[:limit]
    key = read_openrouter_key()
    metadata = write_execution_metadata(config, authenticated=True)
    print(json.dumps(metadata["non_secret_effective_config"], indent=2))
    print("authentication_material_loaded=true")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    payloads = task_payloads()
    stop_event = threading.Event()
    fatal: ProviderIdentityError | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                execute_one, row, payloads[(row["task_id"], row["representation"])],
                config, client, stop_event,
            )
            for row in remaining
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except ProviderIdentityError as error:
                stop_event.set()
                fatal = error
                for pending in futures:
                    pending.cancel()
            except RuntimeError as error:
                if str(error) != "Run stopped before request submission":
                    raise
    if fatal is not None:
        raise fatal
    if fresh_canary:
        canary = validate_canary(config, rows)
        print(json.dumps(canary, indent=2))
        return 0 if canary["status"] == "passed" else 2
    return 0


def report() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = validate_preflight(config)
    expected = {observation_key(row): row for row in rows}
    records = load_jsonl(RESPONSES)
    attempts = load_jsonl(ATTEMPTS)
    accepted = [record for record in records if record.get("status") in ACCEPTED]
    accepted_keys = [observation_key(record) for record in accepted]
    duplicates = len(accepted_keys) - len(set(accepted_keys))
    missing = set(expected) - set(accepted_keys)
    unresolved = [
        record for record in records
        if record.get("status") == "technical_failure" and observation_key(record) in missing
    ]
    malformed = [record for record in accepted if record.get("status") == "malformed_response"]
    providers = sorted({str(record.get("observed_provider_backend")) for record in accepted})
    returned_models = sorted({str(record.get("returned_model")) for record in accepted})
    provider_mismatches = sum(record.get("observed_provider_backend") != PROVIDER_METADATA_NAME for record in accepted)
    model_mismatches = sum(record.get("returned_model") != MODEL_ID for record in accepted)
    input_hash_mismatches = sum(
        expected.get(observation_key(record), {}).get("input_hash") != record.get("input_hash")
        for record in accepted
    )
    representation_counts = {
        value: sum(row.get("representation") == value for row in accepted)
        for value in ("NL", "FS", "AR")
    }
    task_type_counts = {
        value: sum(row.get("task") == value for row in accepted)
        for value in ("BQA", "OEQA")
    }
    dataset_counts = {
        value: sum(row.get("dataset") == value for row in accepted)
        for value in sorted({row["dataset"] for row in rows})
    }
    hop_counts = {
        value: sum(row.get("hop") == value for row in accepted)
        for value in sorted({row["hop"] for row in rows})
    }
    usage_rows = [(row.get("raw_provider_response") or {}).get("usage") or {} for row in attempts]
    input_tokens = sum(int(usage.get("prompt_tokens") or 0) for usage in usage_rows)
    output_tokens = sum(int(usage.get("completion_tokens") or 0) for usage in usage_rows)
    provider_cost_values = [usage.get("cost") for usage in usage_rows if usage.get("cost") is not None]
    provider_reported_cost = sum(float(value) for value in provider_cost_values) if provider_cost_values else None
    pricing = config.get("endpoint_capability_snapshot", {}).get("pricing") or {}
    calculated_cost = None
    if pricing.get("prompt") is not None and pricing.get("completion") is not None:
        calculated_cost = input_tokens * float(pricing["prompt"]) + output_tokens * float(pricing["completion"])
    timestamps = [row.get("timestamp") for row in attempts if row.get("timestamp")]
    runtime_seconds = None
    if timestamps:
        start = min(datetime.fromisoformat(value) for value in timestamps)
        end = max(datetime.fromisoformat(value) for value in timestamps)
        runtime_seconds = (end - start).total_seconds()
    gemini_rows = load_jsonl(PHASE7D / "responses" / "gemini_observations.jsonl")
    gemini_accepted = sum(row.get("status") in ACCEPTED for row in gemini_rows)
    complete = (
        len(accepted) == 27144 and not duplicates and not unresolved and not missing
        and not provider_mismatches and not model_mismatches and not input_hash_mismatches
    )
    result = {
        "status": "complete" if complete else "incomplete",
        "resolved_openrouter_provider_identifier": PROVIDER_TAG,
        "provider_metadata_name": PROVIDER_METADATA_NAME,
        "provider_model_availability_check": "passed",
        "configuration_hash": config["configuration_hash"],
        "canary": json.loads((PHASE7E / "qwen_alibaba_canary.json").read_text(encoding="utf-8")) if (PHASE7E / "qwen_alibaba_canary.json").is_file() else None,
        "qwen_accepted_observations": len(accepted),
        "pending_observations": len(missing),
        "unique_primary_keys": len(set(accepted_keys)),
        "representation_counts": representation_counts,
        "task_type_counts": task_type_counts,
        "dataset_counts": dataset_counts,
        "hop_counts": hop_counts,
        "schema_usable_outputs": sum(row.get("status") == "usable" for row in accepted),
        "technical_retries": len(attempts) - len(accepted),
        "historical_exhausted_retry_rows_resolved_on_resume": sum(
            record.get("status") == "technical_failure"
            and observation_key(record) in set(accepted_keys)
            for record in records
        ),
        "permanent_technical_failures": len(unresolved),
        "malformed_but_retained": len(malformed),
        "malformed_keys": [observation_key(record) for record in malformed],
        "observed_provider_backends": providers,
        "returned_models": returned_models,
        "alibaba_provider_requests": len(attempts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_reported_cost_usd": provider_reported_cost,
        "calculated_cost_usd": calculated_cost,
        "runtime_seconds_between_first_and_last_request": runtime_seconds,
        "duplicate_accepted_observations": duplicates,
        "input_hash_mismatches": input_hash_mismatches,
        "provider_mismatches": provider_mismatches,
        "model_mismatches": model_mismatches,
        "dekallm_primary_contribution": 0,
        "old_alibaba_diagnostic_observations_imported": 0,
        "gemini_accepted_observations": gemini_accepted,
        "gpt_observations": 0,
        "openai_api_key_loaded": False,
        "benchmark_and_gemini_hashes_unchanged": config["frozen_file_sha256"] == frozen_files(),
        "release_published": False,
        "v1_0_modified": False,
        "output_paths": {
            "observations": str(RESPONSES.relative_to(ROOT)),
            "attempts": str(ATTEMPTS.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "execution_metadata": str(RUN_METADATA_PATH.relative_to(ROOT)),
        },
    }
    write_json(REPORT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--validate-canary", action="store_true")
    actions.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
        return 0
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = validate_preflight(config)
    if args.validate_canary:
        result = validate_canary(config, rows)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.report:
        print(json.dumps(report(), indent=2))
        return 0
    return execute(args.limit, args.max_workers)


if __name__ == "__main__":
    raise SystemExit(main())
