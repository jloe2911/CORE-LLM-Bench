#!/usr/bin/env python3
"""Final fresh GPT-5 Mini v1.1 experiment through OpenRouter -> OpenAI only.

All provider payloads are kept under ignored ``data/output``. Preparation is
offline except for public OpenRouter endpoint metadata. Paid execution requires
``--execute-canary`` or ``--execute-full`` and never reads historical GPT rows.
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
OUTPUT_ROOT = ROOT / "data" / "output" / "v1.1.0-gpt-openrouter-primary"
RESPONSES = OUTPUT_ROOT / "responses" / "gpt_observations.jsonl"
ATTEMPTS = OUTPUT_ROOT / "responses" / "gpt_attempts.jsonl"
PROBE_PATH = OUTPUT_ROOT / "gpt_identity_probe.json"
CONFIG_PATH = OUTPUT_ROOT / "gpt_openrouter_config.json"
MANIFEST_PATH = OUTPUT_ROOT / "gpt_primary_manifest.csv"
ENDPOINT_PATH = OUTPUT_ROOT / "openrouter_openai_endpoint.json"
CANARY_PATH = OUTPUT_ROOT / "gpt_canary.json"
PREFLIGHT_PATH = OUTPUT_ROOT / "gpt_preflight.json"
REPORT_PATH = OUTPUT_ROOT / "GPT_OPENROUTER_FINAL_REPORT.json"
RUN_METADATA_PATH = OUTPUT_ROOT / "gpt_execution_metadata.json"

sys.path.insert(0, str(ROOT / "scripts"))
from phase6_materialize_release import create_context_specific_prompt, prompt_hash  # noqa: E402
from phase7a_preflight import parse_response, read_csv  # noqa: E402
from run_v1_1_experiments import task_payloads  # noqa: E402


MODEL_NAME = "GPT-5 mini"
ALIAS_MODEL_ID = "openai/gpt-5-mini"
SNAPSHOT_MODEL_ID = "openai/gpt-5-mini-2025-08-07"
UPSTREAM_SNAPSHOT_ID = "gpt-5-mini-2025-08-07"
PROVIDER_TAG = "openai"
PROVIDER_NAME = "OpenAI"
CONFIG_VERSION = "core-llm-bench-v1.1-gpt-openrouter-final-1"
CANONICAL_PARQUET_SHA256 = "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc"
MODEL_INPUT_MANIFEST_SHA256 = "191c1c0dc221a5829dfa361f86fd1ebf8bbfd54e8f89eeaf0621810001715d18"
CANARY_SIZE = 12
ACCEPTED = {"usable", "malformed_response"}
WRITE_LOCK = threading.Lock()


class ProviderIdentityError(RuntimeError):
    """A successful response came from an unauthorized provider or model."""


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


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def openrouter_metadata(model_id: str = ALIAS_MODEL_ID) -> dict[str, Any]:
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    request = urllib.request.Request(
        url, headers={"User-Agent": "CORE-LLM-Bench/gpt-final"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)["data"]


def selected_endpoint(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("id") != ALIAS_MODEL_ID:
        raise RuntimeError("OpenRouter metadata returned a different model")
    matches = [
        endpoint for endpoint in metadata.get("endpoints", [])
        if endpoint.get("tag") == PROVIDER_TAG
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {PROVIDER_TAG!r} endpoint, got {len(matches)}")
    endpoint = matches[0]
    if endpoint.get("provider_name") != PROVIDER_NAME or endpoint.get("status") != 0:
        raise RuntimeError(f"OpenAI endpoint unavailable or mislabeled: {endpoint!r}")
    name = str(endpoint.get("name") or "")
    if UPSTREAM_SNAPSHOT_ID not in name:
        raise RuntimeError(f"Endpoint metadata does not identify the dated snapshot: {name!r}")
    supported = set(endpoint.get("supported_parameters") or [])
    if not {"reasoning_effort", "max_tokens"}.issubset(supported):
        raise RuntimeError("OpenAI endpoint lacks required Chat Completions controls")
    if int(endpoint.get("max_completion_tokens") or 1024) < 1024:
        raise RuntimeError("OpenAI endpoint cannot provide 1,024 output tokens")
    return endpoint


def safe_endpoint(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        key: endpoint.get(key) for key in (
            "name", "model_id", "provider_name", "tag", "status", "context_length",
            "max_completion_tokens", "supported_parameters", "pricing",
        )
    }


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
    keys: set[tuple[str, str, str]] = set()
    uncovered = set(required)
    while uncovered:
        candidate = max(rows, key=lambda row: len(features(row) & uncovered))
        gain = features(candidate) & uncovered
        if not gain:
            raise RuntimeError(f"Canary coverage unavailable: {sorted(uncovered)!r}")
        selected.append(candidate)
        keys.add(observation_key(candidate))
        uncovered -= gain
    for index in range(CANARY_SIZE):
        candidate = rows[(index * len(rows)) // CANARY_SIZE]
        if observation_key(candidate) not in keys:
            selected.append(candidate)
            keys.add(observation_key(candidate))
        if len(selected) == CANARY_SIZE:
            break
    for candidate in rows:
        if len(selected) == CANARY_SIZE:
            break
        if observation_key(candidate) not in keys:
            selected.append(candidate)
            keys.add(observation_key(candidate))
    return selected


def frozen_hashes() -> dict[str, str]:
    return {
        "benchmark_parquet": sha256_file(STAGE / "core_llm_bench_v1_1.parquet"),
        "model_input_manifest": sha256_file(STAGE / "model_input_manifest.csv"),
    }


def base_gpt_contract() -> dict[str, Any]:
    config = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    gpt = config["models"][MODEL_NAME]
    expected = {
        "max_completion_tokens": 1024,
        "reasoning_effort": "low",
        "extra_body": {"verbosity": "low"},
    }
    if gpt.get("request_parameters") != expected:
        raise RuntimeError(f"Frozen GPT parameter drift: {gpt.get('request_parameters')!r}")
    omitted = ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty")
    if any(gpt.get("parameters", {}).get(name, {}).get("requested") is not None for name in omitted):
        raise RuntimeError("A frozen GPT omitted parameter is unexpectedly requested")
    return {
        "source_config_hash": config["configuration_hash"],
        "timeout_seconds": config["timeout_seconds"],
        "retry_policy": config["retry_policy"],
        "prompt": config["prompt"],
        "frozen_direct_openai_parameters": expected,
    }


def config_basis(endpoint: dict[str, Any]) -> dict[str, Any]:
    base = base_gpt_contract()
    return {
        "config_version": CONFIG_VERSION,
        "model": MODEL_NAME,
        "scientific_target": UPSTREAM_SNAPSHOT_ID,
        "requested_openrouter_model": ALIAS_MODEL_ID,
        "explicit_snapshot_probe_model": SNAPSHOT_MODEL_ID,
        "api_provider": "OpenRouter",
        "upstream_provider": PROVIDER_NAME,
        "credential_source": "OPENROUTER_API_KEY",
        "provider_tag": PROVIDER_TAG,
        "allow_fallbacks": False,
        "transport_change": "direct OpenAI API -> OpenRouter -> OpenAI provider only",
        "request_parameters": {
            "max_tokens": 1024,
            "reasoning_effort": "low",
            "extra_body": {
                "verbosity": "low",
                "provider": {
                    "order": [PROVIDER_TAG],
                    "allow_fallbacks": False,
                },
            },
        },
        "omitted_parameters": [
            "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
        ],
        "timeout_seconds": base["timeout_seconds"],
        "retry_policy": base["retry_policy"],
        "prompt": base["prompt"],
        "source_config_hash": base["source_config_hash"],
        "source_direct_openai_parameters": base["frozen_direct_openai_parameters"],
        "endpoint_capability_snapshot": safe_endpoint(endpoint),
        "pricing_usd_per_token": endpoint.get("pricing") or {},
        "canonical_parquet_sha256": CANONICAL_PARQUET_SHA256,
        "model_input_manifest_sha256": MODEL_INPUT_MANIFEST_SHA256,
    }


def estimate_cost(rows: int, endpoint: dict[str, Any]) -> dict[str, Any]:
    pricing = endpoint.get("pricing") or {}
    prompt_rate = float(pricing.get("prompt") or 0.00000025)
    completion_rate = float(pricing.get("completion") or 0.000002)
    reference_input_tokens = 82_397_683  # same frozen prompts, completed Qwen primary run
    reference_output_tokens = 631_440
    scale = rows / 27144
    return {
        "basis": "Qwen primary-run token volume over the identical frozen prompts; tokenizer-dependent estimate",
        "estimated_input_tokens": round(reference_input_tokens * scale),
        "estimated_output_tokens_if_qwen_like": round(reference_output_tokens * scale),
        "estimated_cost_usd_if_qwen_like_output": round(
            scale * (reference_input_tokens * prompt_rate + reference_output_tokens * completion_rate), 6
        ),
        "worst_case_cost_usd_at_1024_output_tokens_each": round(
            scale * reference_input_tokens * prompt_rate + rows * 1024 * completion_rate, 6
        ),
        "prompt_usd_per_million": prompt_rate * 1_000_000,
        "completion_usd_per_million": completion_rate * 1_000_000,
    }


def prepare() -> dict[str, Any]:
    if RESPONSES.exists() or ATTEMPTS.exists():
        raise FileExistsError("Refusing to replace an existing GPT primary checkpoint")
    endpoint = selected_endpoint(openrouter_metadata())
    basis = config_basis(endpoint)
    configuration_hash = hashlib.sha256(canonical_json(basis).encode()).hexdigest()
    config = {
        **basis,
        "configuration_hash": configuration_hash,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "frozen_hashes": frozen_hashes(),
    }
    base_rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    rows = [row for row in base_rows if row["model"] == MODEL_NAME]
    if len(rows) != 27144 or len({observation_key(row) for row in rows}) != 27144:
        raise RuntimeError("Frozen GPT manifest is not a unique 27,144-row matrix")
    canary_keys = {observation_key(row) for row in select_canary_rows(rows)}
    fields = list(rows[0]) + ["requested_openrouter_model", "requested_provider", "canary"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["model_id"] = ALIAS_MODEL_ID
            row["api_provider"] = "OpenRouter"
            row["credential_source"] = "OPENROUTER_API_KEY"
            row["configuration_hash"] = configuration_hash
            row["requested_openrouter_model"] = ALIAS_MODEL_ID
            row["requested_provider"] = PROVIDER_TAG
            row["canary"] = "true" if observation_key(row) in canary_keys else "false"
            writer.writerow(row)
    endpoint_record = {
        "resolved_at": config["prepared_at"],
        "requested_model": ALIAS_MODEL_ID,
        "selected_endpoint": safe_endpoint(endpoint),
        "snapshot_identity_claim_from_endpoint": UPSTREAM_SNAPSHOT_ID in str(endpoint.get("name")),
    }
    write_json(ENDPOINT_PATH, endpoint_record)
    write_json(CONFIG_PATH, config)
    write_json(CANARY_PATH, {
        "status": "not_run", "accepted": 0, "configuration_hash": configuration_hash,
        "primary_keys": [list(key) for key in sorted(canary_keys)], "failures": [],
    })
    preflight = validate_preflight(config)
    write_json(PREFLIGHT_PATH, preflight)
    write_execution_metadata(config, authenticated=False)
    return {**preflight, "cost_estimate": estimate_cost(27144, endpoint)}


def validate_preflight(config: dict[str, Any]) -> dict[str, Any]:
    stored_hash = config["configuration_hash"]
    basis = {key: value for key, value in config.items() if key not in {
        "configuration_hash", "prepared_at", "frozen_hashes",
    }}
    if hashlib.sha256(canonical_json(basis).encode()).hexdigest() != stored_hash:
        raise RuntimeError("GPT OpenRouter configuration hash is invalid")
    hashes = frozen_hashes()
    if hashes != config.get("frozen_hashes"):
        raise RuntimeError("Frozen benchmark or model-input manifest changed")
    if hashes["benchmark_parquet"] != CANONICAL_PARQUET_SHA256:
        raise RuntimeError("Canonical v1.1 Parquet hash mismatch")
    if hashes["model_input_manifest"] != MODEL_INPUT_MANIFEST_SHA256:
        raise RuntimeError("Frozen model-input manifest hash mismatch")
    params = config.get("request_parameters") or {}
    expected_params = {
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "extra_body": {
            "verbosity": "low",
            "provider": {"order": [PROVIDER_TAG], "allow_fallbacks": False},
        },
    }
    if params != expected_params:
        raise RuntimeError("Effective GPT request parameter drift")
    rows = read_csv(MANIFEST_PATH)
    actual = {
        "total_observations": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        **{rep: sum(row["representation"] == rep for row in rows) for rep in ("NL", "FS", "AR")},
        **{task: sum(row["task"] == task for row in rows) for task in ("BQA", "OEQA")},
    }
    expected = {
        "total_observations": 27144, "tasks": 9048,
        "NL": 9048, "FS": 9048, "AR": 9048, "BQA": 18096, "OEQA": 9048,
    }
    if actual != expected:
        raise RuntimeError(f"Frozen count mismatch: {actual!r}")
    if len({observation_key(row) for row in rows}) != 27144:
        raise RuntimeError("GPT primary manifest contains duplicate observation keys")
    input_rows = read_csv(STAGE / "model_input_manifest.csv")
    input_hashes = {(row["task_id"], row["representation"]): row["input_hash"] for row in input_rows}
    mismatches = sum(
        row["input_hash"] != input_hashes.get((row["task_id"], row["representation"]))
        for row in rows
    )
    if mismatches:
        raise RuntimeError(f"Frozen GPT manifest has {mismatches} input-hash mismatches")
    payloads = task_payloads()
    recomputed = 0
    for row in rows:
        question, context, answer_type = payloads[(row["task_id"], row["representation"])]
        recomputed += prompt_hash(question, context, row["representation"], answer_type) != row["input_hash"]
    if recomputed:
        raise RuntimeError(f"Recomputed model inputs have {recomputed} hash mismatches")
    return {
        **actual,
        "input_hash_mismatches": 0,
        "canonical_parquet_sha256": hashes["benchmark_parquet"],
        "model_input_manifest_sha256": hashes["model_input_manifest"],
        "gpt_output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "output_separate_from_gemini_qwen": True,
        "historical_gpt_responses_scheduled_for_reuse": 0,
        "effective_parameters": params,
    }


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
    raise RuntimeError("OPENROUTER_API_KEY is unavailable")


def write_execution_metadata(config: dict[str, Any], authenticated: bool) -> None:
    write_json(RUN_METADATA_PATH, {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "authentication_material_loaded": authenticated,
        "transport_change": config["transport_change"],
        "non_secret_effective_config": {
            "requested_model": config["requested_openrouter_model"],
            "scientific_target": config["scientific_target"],
            "provider_order": [PROVIDER_TAG],
            "allow_fallbacks": False,
            "request_parameters": config["request_parameters"],
        },
    })


def classify_exception(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    if "timeout" in text:
        return "timeout"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if any(token in text for token in ("500", "502", "503", "504", "connection", "temporar")):
        return "provider_transient_error"
    return "nonretryable_provider_error"


def response_content(response: Any, raw: dict[str, Any]) -> tuple[str, str | None]:
    choices = raw.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return str(message.get("content") or ""), choices[0].get("finish_reason")
    return str(getattr(response, "output_text", "") or raw.get("output_text") or ""), raw.get("status")


def validate_response_identity(returned_model: Any, provider: Any) -> None:
    if returned_model not in {ALIAS_MODEL_ID, SNAPSHOT_MODEL_ID}:
        raise ProviderIdentityError(
            f"model_mismatch: approved={ALIAS_MODEL_ID!r}/{SNAPSHOT_MODEL_ID!r}, returned={returned_model!r}"
        )
    if provider != PROVIDER_NAME:
        raise ProviderIdentityError(
            f"provider_mismatch: requested={PROVIDER_TAG!r}, returned={provider!r}"
        )


def request_kwargs(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    params = json.loads(json.dumps(config["request_parameters"]))
    return {
        "model": model_id,
        "timeout": config["timeout_seconds"],
        **params,
    }


def execute_one(
    row: dict[str, str], payload: tuple[str, str, str], config: dict[str, Any], client: Any,
    stop_event: threading.Event | None = None, model_id: str = ALIAS_MODEL_ID,
) -> dict[str, Any]:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Run stopped before request submission")
    question, context, answer_type = payload
    context_mode = "inline_owl" if row["representation"] == "FS" else "inline_nl"
    prompt = create_context_specific_prompt(question, context, context_mode, answer_type)
    if prompt_hash(question, context, row["representation"], answer_type) != row["input_hash"]:
        raise RuntimeError(f"Input hash mismatch before request: {row['task_id']}/{row['representation']}")
    retry = config["retry_policy"]
    maximum_attempts = 1 + int(retry["maximum_retries_after_initial_attempt"])
    backoff = list(retry["backoff_seconds"])
    common = {
        **{key: row[key] for key in (
            "task_id", "semantic_key", "dataset", "hop", "task", "representation",
            "model", "input_hash", "configuration_hash",
        )},
        "config_version": config["config_version"],
        "requested_openrouter_model": model_id,
        "scientific_target": UPSTREAM_SNAPSHOT_ID,
        "requested_provider": PROVIDER_TAG,
        "api_provider": "OpenRouter",
        "credential_source": "OPENROUTER_API_KEY",
        "generation_config": {
            "reasoning_effort": "low", "verbosity": "low", "max_tokens": 1024,
            "temperature": "omitted", "top_p": "omitted", "seed": "omitted",
            "presence_penalty": "omitted", "frequency_penalty": "omitted",
        },
    }
    last: dict[str, Any] = {}
    for attempt in range(1, maximum_attempts + 1):
        requested_at = datetime.now(timezone.utc).isoformat()
        content = ""
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs(config, model_id),
            )
            raw = response.model_dump(mode="json")
            content, finish_reason = response_content(response, raw)
            returned_model = getattr(response, "model", None)
            observed_provider = raw.get("provider")
            try:
                validate_response_identity(returned_model, observed_provider)
            except ProviderIdentityError as error:
                rejected = {
                    **common, "timestamp": requested_at, "attempt": attempt,
                    "technical_retry_count": attempt - 1, "returned_model": returned_model,
                    "observed_provider_backend": observed_provider,
                    "raw_provider_response": raw, "raw_response": content,
                    "parsed_response": None, "status": "rejected_provider_identity",
                    "error_type": "provider_or_model_mismatch", "error": str(error),
                }
                append_jsonl(ATTEMPTS, rejected)
                if stop_event is not None:
                    stop_event.set()
                raise
            choice = (raw.get("choices") or [{}])[0]
            if choice.get("error") or finish_reason == "error":
                failure_type = "provider_transient_error"
                message = f"Provider returned errored completion: {canonical_json(choice.get('error'))}"
                raw_response = raw
            elif content.strip():
                parsed = parse_response(content, row["task"])
                status = "usable" if parsed["status"] == "requested_schema_conformant" else "malformed_response"
                usage = raw.get("usage") or {}
                last = {
                    **common, "timestamp": requested_at, "attempt": attempt,
                    "technical_retry_count": attempt - 1, "returned_model": returned_model,
                    "observed_provider_backend": observed_provider,
                    "raw_provider_response": raw, "raw_response": content,
                    "parsed_answer": parsed.get("answer"),
                    "parsed_confidence": parsed.get("confidence"),
                    "parsed_response": parsed, "parse_schema_status": parsed.get("status"),
                    "status": status, "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "provider_reported_cost": usage.get("cost"),
                    "finish_reason": finish_reason, "error_type": None, "error": None,
                }
                append_jsonl(ATTEMPTS, last)
                append_jsonl(RESPONSES, last)
                return last
            else:
                failure_type, message = "empty_response", "Provider returned an empty response"
                raw_response = raw
        except ProviderIdentityError:
            raise
        except Exception as error:  # provider SDK exception hierarchy varies
            failure_type, message = classify_exception(error), str(error)
            returned_model, observed_provider, raw_response, finish_reason = None, None, None, None
        last = {
            **common, "timestamp": requested_at, "attempt": attempt,
            "technical_retry_count": attempt - 1, "returned_model": returned_model,
            "observed_provider_backend": observed_provider,
            "raw_provider_response": raw_response, "raw_response": content,
            "parsed_response": None, "status": "technical_failure",
            "finish_reason": finish_reason, "error_type": failure_type, "error": message,
        }
        append_jsonl(ATTEMPTS, last)
        if failure_type not in set(retry["retryable"]) or attempt == maximum_attempts:
            append_jsonl(RESPONSES, last)
            return last
        time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    return last


def client():
    from openai import OpenAI
    return OpenAI(
        api_key=read_openrouter_key(), base_url="https://openrouter.ai/api/v1", max_retries=0,
    )


def identity_probe(config: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    if PROBE_PATH.exists():
        return json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["canary"] == "true")
    payload = task_payloads()[(row["task_id"], row["representation"])]
    prompt = create_context_specific_prompt(
        payload[0], payload[1], "inline_owl" if row["representation"] == "FS" else "inline_nl", payload[2]
    )
    probe_client = client()
    attempts = []
    successful_raw = None
    for model_id in (SNAPSHOT_MODEL_ID, ALIAS_MODEL_ID):
        try:
            response = probe_client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                **{key: value for key, value in request_kwargs(config, model_id).items() if key != "model"},
            )
            raw = response.model_dump(mode="json")
            attempts.append({"requested_model": model_id, "success": True, "raw_provider_response": raw})
            successful_raw = raw
            break
        except Exception as error:  # preserve rejection without retrying a model alias
            attempts.append({
                "requested_model": model_id, "success": False,
                "error_type": type(error).__name__, "error": str(error),
            })
    endpoint = json.loads(ENDPOINT_PATH.read_text(encoding="utf-8"))["selected_endpoint"]
    returned_model = (successful_raw or {}).get("model")
    provider = (successful_raw or {}).get("provider")
    explicit_succeeded = bool(attempts and attempts[0]["success"])
    endpoint_names_snapshot = UPSTREAM_SNAPSHOT_ID in str(endpoint.get("name"))
    exact_established = bool(
        successful_raw and provider == PROVIDER_NAME
        and returned_model in {ALIAS_MODEL_ID, SNAPSHOT_MODEL_ID}
        and (explicit_succeeded or endpoint_names_snapshot)
    )
    result = {
        "status": "passed" if exact_established else "failed",
        "attempts": attempts,
        "explicit_snapshot_request_succeeded": explicit_succeeded,
        "successful_requested_model": next(
            (item["requested_model"] for item in attempts if item["success"]), None
        ),
        "returned_model": returned_model,
        "returned_provider": provider,
        "endpoint_metadata": endpoint,
        "revision_metadata": {
            "endpoint_name": endpoint.get("name"),
            "explicit_snapshot_id": UPSTREAM_SNAPSHOT_ID if endpoint_names_snapshot else None,
        },
        "exact_august_7_snapshot_identity_established": exact_established,
        "identity_basis": (
            "successful explicit dated OpenRouter model request"
            if explicit_succeeded else
            "OpenAI-only pinned response plus live endpoint name explicitly identifying gpt-5-mini-2025-08-07"
            if exact_established else "insufficient model revision evidence"
        ),
    }
    write_json(PROBE_PATH, result)
    return result


def validate_canary(config: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    records = load_jsonl(RESPONSES)
    attempts = load_jsonl(ATTEMPTS)
    canary_rows = [row for row in rows if row["canary"] == "true"]
    expected = {observation_key(row): row for row in canary_rows}
    failures = []
    if len(records) != CANARY_SIZE:
        failures.append(f"expected {CANARY_SIZE} terminal canary rows, found {len(records)}")
    rate_limits = sum(row.get("error_type") == "rate_limit" for row in attempts)
    if rate_limits >= 2:
        failures.append(f"repeated rate-limit failures: {rate_limits}")
    seen = set()
    for index, record in enumerate(records, 1):
        key = observation_key(record)
        manifest = expected.get(key)
        if manifest is None:
            failures.append(f"row {index}: not in representative canary")
            continue
        seen.add(key)
        checks = {
            "accepted": record.get("status") in ACCEPTED,
            "OpenAI provider": record.get("observed_provider_backend") == PROVIDER_NAME,
            "approved returned model": record.get("returned_model") in {ALIAS_MODEL_ID, SNAPSHOT_MODEL_ID},
            "input hash": record.get("input_hash") == manifest["input_hash"],
            "raw response": bool(record.get("raw_provider_response")),
            "nonempty response": bool(str(record.get("raw_response") or "").strip()),
            "parser ran": record.get("parsed_response") is not None,
            "not truncated": record.get("finish_reason") != "length",
        }
        failures.extend(f"row {index}: {label} failed" for label, ok in checks.items() if not ok)
    if set(expected) != seen:
        failures.append("representative canary key-set mismatch")
    probe = identity_probe(config, rows)
    if not probe["exact_august_7_snapshot_identity_established"]:
        failures.append("exact August-7 snapshot identity not established")
    report = {
        "status": "passed" if not failures else "failed",
        "accepted": sum(row.get("status") in ACCEPTED for row in records),
        "configuration_hash": config["configuration_hash"],
        "identity_probe": probe,
        "coverage": {
            field: sorted({row[field] for row in canary_rows})
            for field in ("representation", "task", "hop", "dataset")
        },
        "primary_keys": [list(observation_key(row)) for row in canary_rows],
        "technical_attempts": len(attempts),
        "rate_limit_attempts": rate_limits,
        "failures": failures,
    }
    write_json(CANARY_PATH, report)
    return report


def execute(canary_only: bool, workers: int) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_preflight(config)
    rows = read_csv(MANIFEST_PATH)
    write_execution_metadata(config, authenticated=True)
    probe = identity_probe(config, rows)
    if probe["status"] != "passed":
        print(json.dumps(probe, indent=2))
        return 3
    accepted_records = [row for row in load_jsonl(RESPONSES) if row.get("status") in ACCEPTED]
    terminal = {observation_key(row) for row in accepted_records}
    selected = [row for row in rows if observation_key(row) not in terminal]
    if canary_only:
        selected = [row for row in selected if row["canary"] == "true"]
    else:
        canary = validate_canary(config, rows)
        if canary["status"] != "passed":
            print(json.dumps(canary, indent=2))
            return 4
    api_client = client()
    payloads = task_payloads()
    stop_event = threading.Event()
    fatal: ProviderIdentityError | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                execute_one, row, payloads[(row["task_id"], row["representation"])],
                config, api_client, stop_event,
            )
            for row in selected
        ]
        for completed, future in enumerate(as_completed(futures), 1):
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
            if completed % 250 == 0:
                print(f"completed_futures={completed}/{len(futures)}", flush=True)
    if fatal:
        raise fatal
    if canary_only:
        canary = validate_canary(config, rows)
        print(json.dumps(canary, indent=2))
        return 0 if canary["status"] == "passed" else 4
    return 0


def final_report() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    preflight = validate_preflight(config)
    rows = read_csv(MANIFEST_PATH)
    expected = {observation_key(row): row for row in rows}
    records = load_jsonl(RESPONSES)
    attempts = load_jsonl(ATTEMPTS)
    accepted = [row for row in records if row.get("status") in ACCEPTED]
    keys = [observation_key(row) for row in accepted]
    missing = set(expected) - set(keys)
    duplicates = len(keys) - len(set(keys))
    unresolved = [
        row for row in records
        if row.get("status") == "technical_failure" and observation_key(row) in missing
    ]
    usage = [(row.get("raw_provider_response") or {}).get("usage") or {} for row in attempts]
    input_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage)
    output_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage)
    reasoning_tokens = sum(
        int((row.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        for row in usage
    )
    costs = [row.get("cost") for row in usage if row.get("cost") is not None]
    pricing = config["pricing_usd_per_token"]
    calculated_cost = input_tokens * float(pricing["prompt"]) + output_tokens * float(pricing["completion"])
    timestamps = [row["timestamp"] for row in attempts if row.get("timestamp")]
    runtime = None
    if timestamps:
        start = min(datetime.fromisoformat(value) for value in timestamps)
        end = max(datetime.fromisoformat(value) for value in timestamps)
        runtime = (end - start).total_seconds()
    provider_mismatches = sum(row.get("observed_provider_backend") != PROVIDER_NAME for row in accepted)
    model_mismatches = sum(row.get("returned_model") not in {ALIAS_MODEL_ID, SNAPSHOT_MODEL_ID} for row in accepted)
    input_mismatches = sum(
        expected.get(observation_key(row), {}).get("input_hash") != row.get("input_hash")
        for row in accepted
    )
    complete = bool(
        len(accepted) == 27144 and len(set(keys)) == 27144 and not missing and not duplicates
        and not unresolved and not provider_mismatches and not model_mismatches and not input_mismatches
    )
    result = {
        "status": "complete" if complete else "incomplete",
        "gpt_primary_run_complete_and_valid": complete,
        "requested_openrouter_model_id": ALIAS_MODEL_ID,
        "scientific_target": UPSTREAM_SNAPSHOT_ID,
        "exact_returned_models": sorted({str(row.get("returned_model")) for row in accepted}),
        "exact_returned_revision_metadata": json.loads(PROBE_PATH.read_text(encoding="utf-8")),
        "upstream_provider": PROVIDER_NAME,
        "observed_providers": sorted({str(row.get("observed_provider_backend")) for row in accepted}),
        "fallback_configuration": {"order": [PROVIDER_TAG], "allow_fallbacks": False},
        "effective_generation_parameters": preflight["effective_parameters"],
        "canonical_benchmark_sha256": CANONICAL_PARQUET_SHA256,
        "model_input_manifest_sha256": MODEL_INPUT_MANIFEST_SHA256,
        "expected_observations": 27144,
        "accepted_observations": len(accepted),
        "unique_observations": len(set(keys)),
        "missing_observations": len(missing),
        "duplicate_observations": duplicates,
        "representation_counts": {
            value: sum(row.get("representation") == value for row in accepted)
            for value in ("NL", "FS", "AR")
        },
        "task_counts": {
            value: sum(row.get("task") == value for row in accepted)
            for value in ("BQA", "OEQA")
        },
        "dataset_counts": {
            value: sum(row.get("dataset") == value for row in accepted)
            for value in sorted({row["dataset"] for row in rows})
        },
        "hop_counts": {
            value: sum(row.get("hop") == value for row in accepted)
            for value in sorted({row["hop"] for row in rows})
        },
        "schema_usable_outputs": sum(row.get("status") == "usable" for row in accepted),
        "malformed_but_retained_outputs": sum(row.get("status") == "malformed_response" for row in accepted),
        "technical_retries": len(attempts) - len(records),
        "permanent_failures": len(unresolved),
        "total_provider_requests": len(attempts) + sum(
            len(json.loads(PROBE_PATH.read_text(encoding="utf-8")).get("attempts") or []) for _ in [0]
        ),
        "provider_mismatches": provider_mismatches,
        "model_mismatches": model_mismatches,
        "input_hash_mismatches": input_mismatches,
        "reasoning_tokens": reasoning_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_reported_cost_usd": sum(float(value) for value in costs) if costs else None,
        "calculated_cost_usd": calculated_cost,
        "runtime_seconds_between_first_and_last_primary_attempt": runtime,
        "output_paths": {
            "observations": str(RESPONSES.relative_to(ROOT)),
            "attempts": str(ATTEMPTS.relative_to(ROOT)),
            "identity_probe": str(PROBE_PATH.relative_to(ROOT)),
            "canary": str(CANARY_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
        "exact_august_7_snapshot_identity_established": json.loads(
            PROBE_PATH.read_text(encoding="utf-8")
        )["exact_august_7_snapshot_identity_established"],
        "historical_gpt_responses_imported": 0,
    }
    write_json(REPORT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--execute-canary", action="store_true")
    actions.add_argument("--validate-canary", action="store_true")
    actions.add_argument("--execute-full", action="store_true")
    actions.add_argument("--report", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
        return 0
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = read_csv(MANIFEST_PATH)
    if args.execute_canary:
        return execute(True, args.max_workers)
    if args.validate_canary:
        result = validate_canary(config, rows)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 4
    if args.execute_full:
        return execute(False, args.max_workers)
    print(json.dumps(final_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
