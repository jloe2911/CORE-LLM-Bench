#!/usr/bin/env python3
"""One-shot Phase 7D endpoint/configuration availability validation.

The script uses one neutral diagnostic inference per configured candidate and
never reads benchmark content.  It writes only sanitized response metadata and
refuses to run if its output already exists, preventing accidental repeat calls.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "v1.1.0-phase7d" / "endpoint_validation.json"
PROMPT = "Reply with exactly: OK"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def read_keys() -> dict[str, str]:
    values = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value}
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if os.getenv(key):
            values[key] = os.environ[key]
        if not values.get(key):
            raise RuntimeError(f"Missing {key}")
    return values


def openrouter_metadata(model_id: str) -> dict[str, Any]:
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return json.load(response)["data"]


def sanitized_response(response: Any) -> dict[str, Any]:
    raw = response.model_dump(mode="json")
    text = response.choices[0].message.content or ""
    return {
        "success": True,
        "response_exactly_ok": text.strip() == "OK",
        "response_text": text,
        "returned_model_identifier": raw.get("model"),
        "returned_provider": raw.get("provider"),
        "response_id": raw.get("id"),
        "usage": raw.get("usage"),
    }


def diagnostic(name: str, request: Any) -> dict[str, Any]:
    requested_at = datetime.now(timezone.utc).isoformat()
    try:
        result = sanitized_response(request())
    except Exception as error:  # Provider SDK exception classes vary.
        result = {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "candidate": name,
        "requested_at": requested_at,
        "neutral_prompt": PROMPT,
        "inference_call_count": 1,
        **result,
    }


def select_endpoint(metadata: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [endpoint for endpoint in metadata["endpoints"] if endpoint["tag"] == tag]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one endpoint tagged {tag!r}, got {len(matches)}")
    endpoint = matches[0]
    return {
        field: endpoint.get(field)
        for field in (
            "name", "model_id", "provider_name", "tag", "status",
            "context_length", "max_completion_tokens", "supported_parameters",
            "pricing", "uptime_last_30m", "uptime_last_5m", "uptime_last_1d",
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recover-local-gpt-serialization", action="store_true")
    return parser.parse_args()


def recover_local_gpt_serialization(output: Path, api_key: str) -> int:
    """Submit GPT once after a provably local pre-request SDK keyword failure."""

    report = json.loads(output.read_text(encoding="utf-8"))
    diagnostics = report.get("diagnostics", [])
    if len(diagnostics) != 3:
        raise RuntimeError("Recovery requires the original three-candidate diagnostic log")
    gpt = diagnostics[0]
    expected = "unexpected keyword argument 'verbosity'"
    if (
        gpt.get("candidate") != "gpt-5-mini-2025-08-07"
        or gpt.get("error_type") != "TypeError"
        or expected not in gpt.get("error", "")
        or "provider_request_submitted" in gpt
    ):
        raise RuntimeError("Log does not prove the eligible local GPT serialization failure")
    if not all(item.get("success") for item in diagnostics[1:]):
        raise RuntimeError("OpenRouter diagnostics must already be complete; they will not be repeated")

    client = OpenAI(api_key=api_key)
    recovered = diagnostic(
        gpt["candidate"],
        lambda: client.chat.completions.create(
            model=gpt["candidate"],
            messages=[{"role": "user", "content": PROMPT}],
            max_completion_tokens=1024,
            reasoning_effort="low",
            timeout=30,
            extra_body={"verbosity": "low"},
        ),
    )
    recovered["provider_request_submitted"] = True
    recovered["client_serialization_note"] = (
        "verbosity passed via extra_body because the installed SDK does not expose "
        "it as a direct chat.completions keyword"
    )
    recovered["supersedes_local_pre_request_attempt"] = gpt
    diagnostics[0] = recovered
    report["execution_date_utc"] = datetime.now(timezone.utc).isoformat()
    report["safety"]["local_pre_request_failures"] = 1
    report["safety"]["total_diagnostic_inference_calls"] = 3
    report["safety"]["total_provider_requests_submitted"] = 3
    output.write_text(stable_json(report), encoding="utf-8", newline="\n")
    print(stable_json(report))
    return 0


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.recover_local_gpt_serialization:
        if not output.is_file():
            raise FileNotFoundError("Recovery requires the original diagnostic log")
        return recover_local_gpt_serialization(output, read_keys()["OPENAI_API_KEY"])
    if output.exists():
        raise FileExistsError(
            f"Refusing repeat diagnostic calls because the log already exists: {output}"
        )
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Diagnostic log must remain inside the repository") from error

    keys = read_keys()
    openai_client = OpenAI(api_key=keys["OPENAI_API_KEY"])
    openrouter_client = OpenAI(
        api_key=keys["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"
    )

    gpt_snapshot = "gpt-5-mini-2025-08-07"
    gpt_alias = "gpt-5-mini"
    openai_metadata: dict[str, Any] = {}
    for model_id in (gpt_snapshot, gpt_alias):
        try:
            model = openai_client.models.retrieve(model_id)
            openai_metadata[model_id] = {
                "retrievable": True,
                "returned_id": model.id,
                "created": getattr(model, "created", None),
                "owned_by": getattr(model, "owned_by", None),
            }
        except Exception as error:
            openai_metadata[model_id] = {
                "retrievable": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    gemini_id = "google/gemini-2.5-flash-lite"
    qwen_id = "qwen/qwen3-30b-a3b-instruct-2507"
    gemini_metadata = openrouter_metadata(gemini_id)
    qwen_metadata = openrouter_metadata(qwen_id)
    selected = {
        gemini_id: select_endpoint(gemini_metadata, "google-ai-studio"),
        qwen_id: select_endpoint(qwen_metadata, "dekallm"),
    }

    provider_policy = {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "require_parameters": True,
    }
    diagnostics = [
        diagnostic(
            gpt_snapshot,
            lambda: openai_client.chat.completions.create(
                model=gpt_snapshot,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=1024,
                reasoning_effort="low",
                verbosity="low",
                timeout=30,
            ),
        ),
        diagnostic(
            gemini_id,
            lambda: openrouter_client.chat.completions.create(
                model=gemini_id,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=1024,
                temperature=0.0,
                top_p=0.9,
                seed=0,
                timeout=30,
                extra_body={
                    "provider": {**provider_policy, "order": ["google-ai-studio"]},
                    "reasoning": {"enabled": False},
                },
            ),
        ),
        diagnostic(
            qwen_id,
            lambda: openrouter_client.chat.completions.create(
                model=qwen_id,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=1024,
                temperature=0.0,
                top_p=0.9,
                seed=0,
                presence_penalty=0.0,
                frequency_penalty=0.1,
                timeout=30,
                extra_body={
                    "provider": {**provider_policy, "order": ["dekallm"]},
                },
            ),
        ),
    ]

    report = {
        "phase": "7D endpoint validation",
        "execution_date_utc": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "benchmark_question_used": False,
            "diagnostics_are_experiment_observations": False,
            "automatic_retries": 0,
            "total_diagnostic_inference_calls": sum(
                item["inference_call_count"] for item in diagnostics
            ),
        },
        "openai_model_metadata": openai_metadata,
        "openrouter_selected_endpoint_metadata": selected,
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(report), encoding="utf-8", newline="\n")
    print(stable_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
