import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.phase7f_gpt_openrouter as runner


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"


def endpoint():
    return {
        "name": "OpenAI | openai/gpt-5-mini-2025-08-07",
        "provider_name": "OpenAI",
        "tag": "openai",
        "status": 0,
        "supported_parameters": ["reasoning_effort", "max_tokens", "seed"],
        "max_completion_tokens": 128000,
        "pricing": {"prompt": "0.00000025", "completion": "0.000002"},
    }


def test_endpoint_proves_openai_dated_snapshot_identity():
    metadata = {"id": runner.ALIAS_MODEL_ID, "endpoints": [endpoint()]}
    assert runner.selected_endpoint(metadata)["tag"] == "openai"
    bad = endpoint()
    bad["name"] = "OpenAI | unknown revision"
    with pytest.raises(RuntimeError, match="dated snapshot"):
        runner.selected_endpoint({"id": runner.ALIAS_MODEL_ID, "endpoints": [bad]})


def test_basis_preserves_frozen_gpt_controls_and_omissions():
    basis = runner.config_basis(endpoint())
    assert basis["request_parameters"] == {
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "extra_body": {
            "verbosity": "low",
            "provider": {"order": ["openai"], "allow_fallbacks": False},
        },
    }
    assert set(basis["omitted_parameters"]) == {
        "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
    }
    assert len(hashlib.sha256(runner.canonical_json(basis).encode()).hexdigest()) == 64


def test_response_identity_fails_closed():
    runner.validate_response_identity(runner.ALIAS_MODEL_ID, "OpenAI")
    runner.validate_response_identity(runner.SNAPSHOT_MODEL_ID, "OpenAI")
    with pytest.raises(runner.ProviderIdentityError, match="model_mismatch"):
        runner.validate_response_identity("openai/gpt-5.6-terra", "OpenAI")
    with pytest.raises(runner.ProviderIdentityError, match="provider_mismatch"):
        runner.validate_response_identity(runner.ALIAS_MODEL_ID, "Azure")


def test_provider_mismatch_is_never_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ATTEMPTS", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(runner, "RESPONSES", tmp_path / "responses.jsonl")
    question, context, answer_type = "Is A B?", "A is B.", "BIN"
    row = {
        "task_id": "1", "semantic_key": "sem", "dataset": "FamilyOWL",
        "hop": "1hop", "task": "BQA", "representation": "NL",
        "model": runner.MODEL_NAME,
        "input_hash": runner.prompt_hash(question, context, "NL", answer_type),
        "configuration_hash": "config",
    }
    config = {
        "config_version": "test", "timeout_seconds": 30,
        "retry_policy": {"maximum_retries_after_initial_attempt": 0, "backoff_seconds": [0], "retryable": []},
        "request_parameters": {
            "max_tokens": 1024, "reasoning_effort": "low",
            "extra_body": {"verbosity": "low", "provider": {"order": ["openai"], "allow_fallbacks": False}},
        },
    }
    response = SimpleNamespace(
        model=runner.ALIAS_MODEL_ID,
        choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER: TRUE\nCONFIDENCE: 1"), finish_reason="stop")],
        model_dump=lambda mode: {
            "model": runner.ALIAS_MODEL_ID, "provider": "Azure", "usage": {},
            "choices": [{"message": {"content": "ANSWER: TRUE\nCONFIDENCE: 1"}, "finish_reason": "stop"}],
        },
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(runner.ProviderIdentityError, match="provider_mismatch"):
        runner.execute_one(row, (question, context, answer_type), config, client)
    assert runner.ATTEMPTS.is_file()
    assert not runner.RESPONSES.exists()


def test_representative_canary_covers_all_required_axes():
    rows = [
        row for row in runner.read_csv(STAGE / "primary_experiment_manifest.csv")
        if row["model"] == runner.MODEL_NAME
    ]
    selected = runner.select_canary_rows(rows)
    assert len(selected) == 12
    assert selected == runner.select_canary_rows(rows)
    assert {row["representation"] for row in selected} == {"NL", "FS", "AR"}
    assert {row["task"] for row in selected} == {"BQA", "OEQA"}
    assert {row["hop"] for row in selected} == {"1hop", "2hop"}
    assert {row["dataset"] for row in selected} == {"FamilyOWL", "Pizza100", "Pizza250", "OWL2Bench"}
