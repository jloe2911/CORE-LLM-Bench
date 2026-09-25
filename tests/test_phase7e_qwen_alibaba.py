import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.phase7e_qwen_alibaba as runner

from scripts.phase7e_qwen_alibaba import (
    BASE_CONFIGURATION_HASH,
    MODEL_ID,
    PROVIDER_TAG,
    REQUIRED_PARAMETERS,
    canonical_json,
    configuration_basis,
    execute_one,
    select_canary_rows,
    selected_endpoint,
    validate_response_identity,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"


def endpoint():
    return {
        "provider_name": "Alibaba",
        "tag": "alibaba",
        "status": 0,
        "supported_parameters": sorted(REQUIRED_PARAMETERS | {"stop"}),
        "max_completion_tokens": 32768,
        "context_length": 131072,
    }


def test_alibaba_endpoint_selection_is_exact_and_capable():
    metadata = {"id": MODEL_ID, "endpoints": [endpoint()]}
    assert selected_endpoint(metadata)["tag"] == PROVIDER_TAG


def test_phase7e_basis_changes_only_qwen_provider_choice():
    base = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    assert base["configuration_hash"] == BASE_CONFIGURATION_HASH
    basis = configuration_basis(endpoint())
    qwen = base["models"]["Qwen3-30B-A3B-Instruct"]
    assert basis["model_id"] == qwen["model_id"]
    assert basis["request_parameters"]["temperature"] == 0.0
    assert basis["request_parameters"]["top_p"] == 0.9
    assert basis["request_parameters"]["max_tokens"] == 1024
    assert basis["request_parameters"]["seed"] == 0
    assert basis["request_parameters"]["presence_penalty"] == 0.0
    assert basis["request_parameters"]["frequency_penalty"] == 0.1
    assert basis["request_parameters"]["extra_body"]["provider"] == {
        "order": ["alibaba"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert basis["timeout_seconds"] == base["timeout_seconds"]
    assert basis["retry_policy"] == base["retry_policy"]
    assert basis["prompt"] == base["prompt"]
    assert basis["benchmark_hash_transition"] == {
        "pre_schema_parquet_sha256": "7285e506483b422acf8e1882da5b5390966ba4896b4bb020820f6bb4a5218859",
        "canonical_v1_1_parquet_sha256": "0c39f84abb7f5a44af7496862317ea761bc41809cc1cdd490cbaed19a281bccc",
        "model_input_manifest_sha256": "191c1c0dc221a5829dfa361f86fd1ebf8bbfd54e8f89eeaf0621810001715d18",
        "change_scope": "schema-only; model-facing NL/FS/AR inputs unchanged",
        "model_inputs_regenerated": False,
    }
    assert len(hashlib.sha256(canonical_json(basis).encode()).hexdigest()) == 64


def test_response_identity_requires_exact_model_and_alibaba():
    validate_response_identity(MODEL_ID, "Alibaba")
    with pytest.raises(runner.ProviderIdentityError, match="model_mismatch"):
        validate_response_identity("qwen/another-model", "Alibaba")
    with pytest.raises(runner.ProviderIdentityError, match="provider_mismatch"):
        validate_response_identity(MODEL_ID, "DekaLLM")


def test_provider_mismatch_is_diagnostic_only_and_never_accepted(tmp_path, monkeypatch):
    attempts = tmp_path / "attempts.jsonl"
    responses = tmp_path / "observations.jsonl"
    monkeypatch.setattr(runner, "ATTEMPTS", attempts)
    monkeypatch.setattr(runner, "RESPONSES", responses)
    question, context, answer_type = "Is A B?", "A is B.", "BIN"
    input_hash = runner.prompt_hash(question, context, "NL", answer_type)
    row = {
        "task_id": "1", "semantic_key": "sem", "dataset": "FamilyOWL",
        "hop": "1hop", "task": "BQA", "representation": "NL",
        "model": "Qwen3-30B-A3B-Instruct", "input_hash": input_hash,
        "configuration_hash": "config",
    }
    config = {
        "config_version": "test", "timeout_seconds": 30,
        "retry_policy": {"maximum_retries_after_initial_attempt": 0, "backoff_seconds": [0], "retryable": []},
        "request_parameters": {"seed": 0, "temperature": 0.0, "top_p": 0.9,
                               "presence_penalty": 0.0, "frequency_penalty": 0.1,
                               "max_tokens": 1024, "extra_body": {}},
    }
    response = SimpleNamespace(
        model=MODEL_ID,
        choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER: TRUE\nCONFIDENCE: 1.0"), finish_reason="stop")],
        model_dump=lambda mode: {"model": MODEL_ID, "provider": "DekaLLM", "usage": {}},
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(runner.ProviderIdentityError, match="provider_mismatch"):
        execute_one(row, (question, context, answer_type), config, client)
    assert attempts.is_file()
    assert json.loads(attempts.read_text(encoding="utf-8"))["status"] == "rejected_provider_identity"
    assert not responses.exists()


def test_embedded_provider_error_is_technical_and_never_accepted(tmp_path, monkeypatch):
    attempts = tmp_path / "attempts.jsonl"
    responses = tmp_path / "observations.jsonl"
    monkeypatch.setattr(runner, "ATTEMPTS", attempts)
    monkeypatch.setattr(runner, "RESPONSES", responses)
    question, context, answer_type = "Is A B?", "A is B.", "BIN"
    row = {
        "task_id": "1", "semantic_key": "sem", "dataset": "FamilyOWL",
        "hop": "1hop", "task": "BQA", "representation": "NL",
        "model": "Qwen3-30B-A3B-Instruct",
        "input_hash": runner.prompt_hash(question, context, "NL", answer_type),
        "configuration_hash": "config",
    }
    config = {
        "config_version": "test", "timeout_seconds": 30,
        "retry_policy": {"maximum_retries_after_initial_attempt": 0, "backoff_seconds": [0],
                         "retryable": ["provider_transient_error"]},
        "request_parameters": {"seed": 0, "temperature": 0.0, "top_p": 0.9,
                               "presence_penalty": 0.0, "frequency_penalty": 0.1,
                               "max_tokens": 1024, "extra_body": {}},
    }
    choice = {"finish_reason": "error", "error": {"code": 502, "message": "Network connection lost."}}
    response = SimpleNamespace(
        model=MODEL_ID,
        choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER"), finish_reason="error")],
        model_dump=lambda mode: {"model": MODEL_ID, "provider": "Alibaba", "choices": [choice], "usage": {}},
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = execute_one(row, (question, context, answer_type), config, client)
    assert result["status"] == "technical_failure"
    assert result["error_type"] == "provider_transient_error"
    assert result["raw_response"] == "ANSWER"
    assert json.loads(attempts.read_text(encoding="utf-8"))["status"] == "technical_failure"
    assert json.loads(responses.read_text(encoding="utf-8"))["status"] == "technical_failure"


def test_representative_canary_is_deterministic_and_covers_required_axes():
    rows = runner.read_csv(STAGE / "primary_experiment_manifest.csv")
    rows = [row for row in rows if row["model_id"] == MODEL_ID]
    selected = select_canary_rows(rows)
    assert len(selected) == 12
    assert selected == select_canary_rows(rows)
    assert {row["representation"] for row in selected} == {"NL", "FS", "AR"}
    assert {row["task"] for row in selected} == {"BQA", "OEQA"}
    assert {row["hop"] for row in selected} == {"1hop", "2hop"}
    assert {row["dataset"] for row in selected} == {"FamilyOWL", "Pizza100", "Pizza250", "OWL2Bench"}
