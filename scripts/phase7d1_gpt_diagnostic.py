#!/usr/bin/env python3
"""Submit exactly one direct-OpenAI neutral GPT snapshot diagnostic.

The output is create-once and contains only sanitized metadata.  The OpenAI
client has SDK retries disabled so one invocation means one provider request.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "v1.1.0-phase7d" / "phase7d1_gpt_diagnostic.json"
MODEL_ID = "gpt-5-mini-2025-08-07"
PROMPT = "Reply with exactly: OK"
CREDENTIAL_SOURCE = "OPENAI_API_KEY"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def api_key() -> str:
    value = os.getenv(CREDENTIAL_SOURCE)
    if not value:
        value = dotenv_values(ROOT / ".env").get(CREDENTIAL_SOURCE)
    if not value:
        raise RuntimeError(f"Missing {CREDENTIAL_SOURCE}")
    return str(value)


def safe_headers(headers: Any) -> dict[str, str]:
    allowed = {
        "openai-organization",
        "openai-processing-ms",
        "openai-project",
        "openai-version",
        "x-request-id",
    }
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if str(key).lower() in allowed
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing a repeat diagnostic: {output}")
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Diagnostic log must remain inside the repository") from error

    requested_at = datetime.now(timezone.utc).isoformat()
    request = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": 1024,
        "reasoning_effort": "low",
        "extra_body": {"verbosity": "low"},
        "timeout": 30,
    }
    base = {
        "phase": "7D.1",
        "requested_at": requested_at,
        "requested_model_identifier": MODEL_ID,
        "api_provider": "OpenAI",
        "api_endpoint": "https://api.openai.com/v1/chat/completions",
        "credential_source": CREDENTIAL_SOURCE,
        "neutral_prompt": PROMPT,
        "request_parameters": {
            "reasoning_effort": "low",
            "verbosity": "low",
            "max_completion_tokens": 1024,
            "temperature": "omitted",
            "top_p": "omitted",
        },
        "automatic_retries": 0,
        "diagnostic_model_calls": 1,
        "provider_requests_submitted": 1,
        "benchmark_experiment_calls": 0,
    }
    try:
        client = OpenAI(api_key=api_key(), max_retries=0)
        raw = client.chat.completions.with_raw_response.create(**request)
        response = raw.parse()
        text = response.choices[0].message.content or ""
        result = {
            **base,
            "success": True,
            "response_status": raw.status_code,
            "response_exactly_ok": text.strip() == "OK",
            "response_text": text,
            "returned_model_identifier": response.model,
            "response_id": response.id,
            "provider_backend_metadata": safe_headers(raw.headers),
            "parameter_acceptance": {
                "reasoning_effort": "accepted",
                "verbosity": "accepted",
                "max_completion_tokens": "accepted",
                "temperature": "omitted_not_tested",
                "top_p": "omitted_not_tested",
            },
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
        }
    except Exception as error:  # SDK exception hierarchy varies.
        status_code = getattr(error, "status_code", None)
        body = getattr(error, "body", None)
        result = {
            **base,
            "success": False,
            "response_status": status_code,
            "returned_model_identifier": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "provider_error_body": body if isinstance(body, dict) else None,
            "provider_backend_metadata": safe_headers(
                getattr(getattr(error, "response", None), "headers", {})
            ),
            "parameter_acceptance": {
                "reasoning_effort": "not_established",
                "verbosity": "not_established",
                "max_completion_tokens": "not_established",
                "temperature": "omitted_not_tested",
                "top_p": "omitted_not_tested",
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(result), encoding="utf-8", newline="\n")
    print(stable_json(result))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
