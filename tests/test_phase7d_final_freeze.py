import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.phase7d_finalize_experiment import canonical_json
from scripts.run_v1_1_experiments import audit_manifest_integrity, validate_config


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "v1.1.0-staging"
PHASE7D = ROOT / "release" / "v1.1.0-phase7d"


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_final_config_hash_and_execution_gate():
    config = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    expected = config.pop("configuration_hash")
    assert hashlib.sha256(canonical_json(config).encode()).hexdigest() == expected
    config["configuration_hash"] = expected
    assert config["ready_for_execution"] is False
    assert config["execution_authorized"] is False
    validate_config(config, execute=False)


def test_primary_matrix_is_complete_fresh_and_hash_pinned():
    config = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    report = audit_manifest_integrity(config)
    rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    assert report["pending_rows"] == 81432
    assert {row["status"] for row in rows} == {"pending"}
    assert {row["configuration_hash"] for row in rows} == {config["configuration_hash"]}


def test_membership_and_duplicate_gates():
    assert pq.read_table(STAGE / "core_llm_bench_v1_1.parquet").num_rows == 9048
    report = json.loads((STAGE / "validation_report.json").read_text(encoding="utf-8"))
    assert report["membership_rows"] == 9048
    assert report["matrix_rows"] == 81432
    assert report["historical_v1_0_responses_in_primary"] == 0
    for representation in ("NL", "FS", "AR"):
        assert report["duplicate_audit"][representation]["incompatible_groups"] == 0


def test_provider_pinning_and_diagnostic_safety():
    config = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    endpoint = json.loads((PHASE7D / "endpoint_validation.json").read_text(encoding="utf-8"))
    assert config["models"]["Gemini 2.5 Flash-Lite"]["provider_tag"] == "google-ai-studio"
    assert config["models"]["Qwen3-30B-A3B-Instruct"]["provider_tag"] == "dekallm"
    assert all(model["fallback"] is False for model in config["models"].values())
    assert endpoint["safety"]["total_provider_requests_submitted"] == 3
    assert endpoint["safety"]["benchmark_question_used"] is False
    assert config["models"]["GPT-5 mini"]["api_provider"] == "OpenAI"
    assert config["models"]["GPT-5 mini"]["credential_source"] == "OPENAI_API_KEY"
    for name in ("Gemini 2.5 Flash-Lite", "Qwen3-30B-A3B-Instruct"):
        assert config["models"][name]["api_provider"] == "OpenRouter"
        assert config["models"][name]["credential_source"] == "OPENROUTER_API_KEY"


def test_manifest_routes_are_explicit_and_fail_closed():
    config = json.loads((STAGE / "experiment_config_v1_1.json").read_text(encoding="utf-8"))
    rows = read_csv(STAGE / "primary_experiment_manifest.csv")
    assert {(row["api_provider"], row["credential_source"]) for row in rows if row["model"] == "GPT-5 mini"} == {("OpenAI", "OPENAI_API_KEY")}
    assert {(row["api_provider"], row["credential_source"]) for row in rows if row["model"] != "GPT-5 mini"} == {("OpenRouter", "OPENROUTER_API_KEY")}
    import pytest
    for model_name, field, invalid in (
        ("GPT-5 mini", "api_provider", "OpenRouter"),
        ("GPT-5 mini", "credential_source", "OPENROUTER_API_KEY"),
        ("Gemini 2.5 Flash-Lite", "api_provider", "OpenAI"),
        ("Qwen3-30B-A3B-Instruct", "api_provider", "OpenAI"),
        ("Gemini 2.5 Flash-Lite", "credential_source", "OPENAI_API_KEY"),
        ("Qwen3-30B-A3B-Instruct", "credential_source", "OPENAI_API_KEY"),
    ):
        bad = json.loads(json.dumps(config))
        bad["models"][model_name][field] = invalid
        with pytest.raises(ValueError, match="routing violation"):
            validate_config(bad, execute=False)


def test_release_manifest_and_checksums_cover_final_files():
    manifest = json.loads((STAGE / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in manifest["files"]}
    for required in (
        "core_llm_bench_v1_1.parquet",
        "experiment_config_v1_1.json",
        "primary_experiment_manifest.csv",
        "input_equivalence_groups.csv",
        "cost_estimate.json",
    ):
        path = STAGE / required
        assert required in entries
        assert entries[required]["bytes"] == path.stat().st_size
        assert entries[required]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
