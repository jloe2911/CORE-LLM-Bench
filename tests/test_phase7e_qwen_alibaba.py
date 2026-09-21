import hashlib
import json
from pathlib import Path

from scripts.phase7e_qwen_alibaba import (
    BASE_CONFIGURATION_HASH,
    MODEL_ID,
    PROVIDER_TAG,
    REQUIRED_PARAMETERS,
    canonical_json,
    configuration_basis,
    selected_endpoint,
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
    assert len(hashlib.sha256(canonical_json(basis).encode()).hexdigest()) == 64
