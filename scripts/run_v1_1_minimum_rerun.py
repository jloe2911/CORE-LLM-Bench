#!/usr/bin/env python3
"""Resumable executor for the frozen 16,443-call v1.1 correction plan.

Default operation is offline preflight.  Paid requests require the explicit
``--execute`` flag.  A SQLite journal claims each deduplicated request before
submission, retains unresolved in-flight claims after crashes, and never
automatically retries them.  This avoids silently duplicating a paid request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "correction" / "v1.1.0-semantic-audit"
MANIFEST = AUDIT / "rerun_manifest.csv"
BENCHMARK = AUDIT / "core_llm_bench_v1_1_corrected.parquet"
INPUT_MANIFEST = AUDIT / "corrected_model_input_manifest.csv"
OUTPUT = ROOT / "data" / "output" / "v1.1.0-minimum-rerun"
DATABASE = OUTPUT / "checkpoint.sqlite3"
RESPONSES = OUTPUT / "responses.jsonl"
MAX_CAP_USD = 25.0
EXPECTED_CALLS = 16443

sys.path.insert(0, str(ROOT / "scripts"))
from phase6_materialize_release import create_context_specific_prompt, prompt_hash  # noqa: E402
from phase7a_preflight import parse_response  # noqa: E402


MODELS: dict[str, dict[str, Any]] = {
    "GPT-5 mini": {
        "model_id": "openai/gpt-5-mini",
        "provider": "OpenAI",
        "provider_tag": "openai",
        "configuration_hash": "4c7b1f65090715fbc99756b7d3b3c869c467e921d93976bce19cb85f91c81475",
        "input_per_million": 0.25,
        "output_per_million": 2.0,
        "parameters": {
            "max_tokens": 1024,
            "reasoning_effort": "low",
            "extra_body": {
                "verbosity": "low",
                "provider": {"order": ["openai"], "allow_fallbacks": False},
            },
        },
        "returned_models": {"openai/gpt-5-mini", "openai/gpt-5-mini-2025-08-07"},
    },
    "Gemini 2.5 Flash-Lite": {
        "model_id": "google/gemini-2.5-flash-lite",
        "provider": "Google AI Studio",
        "provider_tag": "google-ai-studio",
        "configuration_hash": "c1562554bd9e252bf97356ef098edde997f813536dd8e547f35f5981a1023df3",
        "input_per_million": 0.10,
        "output_per_million": 0.40,
        "parameters": {
            "max_tokens": 1024, "temperature": 0.0, "top_p": 0.9, "seed": 0,
            "extra_body": {
                "reasoning": {"enabled": False},
                "provider": {
                    "order": ["google-ai-studio"], "allow_fallbacks": False,
                    "require_parameters": True, "data_collection": "deny",
                },
            },
        },
        "returned_models": {"google/gemini-2.5-flash-lite"},
    },
    "Qwen3-30B-A3B-Instruct": {
        "model_id": "qwen/qwen3-30b-a3b-instruct-2507",
        "provider": "Alibaba",
        "provider_tag": "alibaba",
        "configuration_hash": "68ba5917f3197234f41316cf765e9581b3421981041ccb899218968346d3b031",
        "input_per_million": 0.13,
        "output_per_million": 0.52,
        "parameters": {
            "max_tokens": 1024, "temperature": 0.0, "top_p": 0.9, "seed": 0,
            "presence_penalty": 0.0, "frequency_penalty": 0.1,
            "extra_body": {
                "provider": {
                    "order": ["alibaba"], "allow_fallbacks": False,
                    "require_parameters": True, "data_collection": "deny",
                },
            },
        },
        "returned_models": {"qwen/qwen3-30b-a3b-instruct-2507"},
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def payload(row: pd.Series, representation: str) -> tuple[str, str]:
    if representation == "NL":
        return str(row.nl_question), str(row.nl_context)
    if representation == "FS":
        return str(row.fs_query), str(row.fs_context)
    if representation == "AR":
        return str(row.ar_question), str(row.ar_context)
    raise ValueError(representation)


def load_plan() -> tuple[list[dict[str, str]], dict[str, pd.Series]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["deduplicated_request"].lower() == "true"]
    if len(rows) != EXPECTED_CALLS or len({row["deduplication_group"] for row in rows}) != EXPECTED_CALLS:
        raise RuntimeError("Executable manifest is not the frozen 16,443-request plan")
    benchmark = pd.read_parquet(BENCHMARK)
    by_id = {str(row.task_id): row for _, row in benchmark.iterrows()}
    corrected_inputs = pd.read_csv(INPUT_MANIFEST, dtype=str).fillna("")
    hashes = corrected_inputs.set_index(["task_id", "representation"]).input_hash.to_dict()
    if len(hashes) != 27144:
        raise RuntimeError("Corrected model-input manifest is incomplete")
    for row in rows:
        model = MODELS[row["model"]]
        if row["configuration_hash"] != model["configuration_hash"]:
            raise RuntimeError(f"Configuration drift in {row['deduplication_group']}")
        question, context = payload(by_id[row["task_id"]], row["representation"])
        actual = prompt_hash(question, context, row["representation"], row["answer_type"])
        if actual != row["corrected_prompt_hash"] or actual != hashes[(row["task_id"], row["representation"])]:
            raise RuntimeError(f"Input hash drift in {row['deduplication_group']}")
    return rows, by_id


def connect(rows: list[dict[str, str]]) -> sqlite3.Connection:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS requests ("
        "request_id TEXT PRIMARY KEY, model TEXT NOT NULL, task_id TEXT NOT NULL, "
        "representation TEXT NOT NULL, input_hash TEXT NOT NULL, status TEXT NOT NULL, "
        "reserved_usd REAL NOT NULL DEFAULT 0, charged_usd REAL NOT NULL DEFAULT 0, "
        "claimed_at TEXT, completed_at TEXT, response_sha256 TEXT, error TEXT)"
    )
    for row in rows:
        db.execute(
            "INSERT OR IGNORE INTO requests(request_id,model,task_id,representation,input_hash,status) "
            "VALUES(?,?,?,?,?,'pending')",
            (row["deduplication_group"], row["model"], row["task_id"],
             row["representation"], row["corrected_prompt_hash"]),
        )
    db.commit()
    if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] != EXPECTED_CALLS:
        raise RuntimeError("Checkpoint contains request IDs outside the frozen manifest")
    return db


def reservation(row: dict[str, str]) -> float:
    model = MODELS[row["model"]]
    input_tokens = int(row["historical_input_tokens"])
    return (input_tokens * model["input_per_million"] + 1024 * model["output_per_million"]) / 1_000_000


def status(db: sqlite3.Connection, cap: float) -> dict[str, Any]:
    counts = dict(db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status").fetchall())
    charged, reserved = db.execute(
        "SELECT COALESCE(SUM(charged_usd),0),COALESCE(SUM(CASE WHEN status='in_flight' THEN reserved_usd ELSE 0 END),0) FROM requests"
    ).fetchone()
    return {"counts": counts, "charged_usd": charged, "in_flight_reserved_usd": reserved,
            "cap_usd": cap, "remaining_unreserved_usd": cap - charged - reserved}


def claim(db: sqlite3.Connection, row: dict[str, str], cap: float) -> float:
    reserve = reservation(row)
    db.execute("BEGIN IMMEDIATE")
    current = db.execute("SELECT status FROM requests WHERE request_id=?", (row["deduplication_group"],)).fetchone()[0]
    charged, held = db.execute(
        "SELECT COALESCE(SUM(charged_usd),0),COALESCE(SUM(CASE WHEN status='in_flight' THEN reserved_usd ELSE 0 END),0) FROM requests"
    ).fetchone()
    if current != "pending":
        db.rollback()
        raise RuntimeError(f"Request is not pending: {row['deduplication_group']}={current}")
    if charged + held + reserve > cap:
        db.rollback()
        raise RuntimeError("$25 spending cap gate stopped before the next request")
    changed = db.execute(
        "UPDATE requests SET status='in_flight',reserved_usd=?,claimed_at=? WHERE request_id=? AND status='pending'",
        (reserve, datetime.now(timezone.utc).isoformat(), row["deduplication_group"]),
    ).rowcount
    if changed != 1:
        db.rollback()
        raise RuntimeError("Atomic request claim failed")
    db.commit()
    return reserve


def execute_one(client: Any, row: dict[str, str], bench: pd.Series, reserve: float) -> tuple[dict[str, Any], float]:
    model = MODELS[row["model"]]
    question, context = payload(bench, row["representation"])
    mode = "inline_owl" if row["representation"] == "FS" else "inline_nl"
    prompt = create_context_specific_prompt(question, context, mode, row["answer_type"])
    response = client.chat.completions.create(
        model=model["model_id"], messages=[{"role": "user", "content": prompt}],
        timeout=30, **model["parameters"],
    )
    raw = response.model_dump(mode="json")
    returned_model, provider = str(response.model), str(raw.get("provider"))
    if returned_model not in model["returned_models"] or provider != model["provider"]:
        raise RuntimeError(f"Provider/model mismatch: {returned_model}/{provider}")
    content = response.choices[0].message.content or ""
    usage = raw.get("usage") or {}
    reported = usage.get("cost")
    computed = (
        int(usage.get("prompt_tokens") or 0) * model["input_per_million"]
        + int(usage.get("completion_tokens") or 0) * model["output_per_million"]
    ) / 1_000_000
    charged = float(reported) if reported is not None else (computed if computed else reserve)
    parsed = parse_response(content, row["task_type"]) if content.strip() else None
    record = {
        "request_id": row["deduplication_group"], "task_id": row["task_id"],
        "representation": row["representation"], "model": row["model"],
        "input_hash": row["corrected_prompt_hash"], "timestamp": datetime.now(timezone.utc).isoformat(),
        "returned_model": returned_model, "returned_provider": provider,
        "raw_provider_response": raw, "raw_response": content, "parsed_response": parsed,
        "provider_reported_cost_usd": reported, "charged_cost_usd": charged,
        "status": "usable" if parsed and parsed.get("status") == "requested_schema_conformant" else "malformed_response",
    }
    return record, charged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="authorize paid requests")
    parser.add_argument("--cap-usd", type=float, default=25.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if not (0 < args.cap_usd <= MAX_CAP_USD):
        raise RuntimeError("The enforced cap must be greater than $0 and no more than $25")
    rows, benchmark = load_plan()
    db = connect(rows)
    current = status(db, args.cap_usd)
    if not args.execute:
        print(json.dumps({"status": "offline-preflight-passed", "paid_requests_made": 0,
                          "manifest_sha256": sha256_file(MANIFEST), **current}, indent=2, sort_keys=True))
        return 0
    if current["counts"].get("in_flight", 0):
        raise RuntimeError("Unresolved in-flight requests exist; reconcile them before resuming")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required only for --execute")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    pending = {value[0] for value in db.execute("SELECT request_id FROM requests WHERE status='pending'")}
    sent = 0
    for row in rows:
        request_id = row["deduplication_group"]
        if request_id not in pending:
            continue
        if args.max_requests is not None and sent >= args.max_requests:
            break
        reserve = claim(db, row, args.cap_usd)
        try:
            record, charged = execute_one(client, row, benchmark[row["task_id"]], reserve)
            append_jsonl(RESPONSES, record)
            digest = hashlib.sha256(canonical(record).encode()).hexdigest()
            db.execute(
                "UPDATE requests SET status='completed',charged_usd=?,reserved_usd=0,completed_at=?,response_sha256=? WHERE request_id=?",
                (charged, datetime.now(timezone.utc).isoformat(), digest, request_id),
            )
            db.commit()
        except Exception as error:
            # Deliberately retain in_flight: a timeout may have reached the provider.
            db.execute("UPDATE requests SET error=? WHERE request_id=?", (repr(error), request_id))
            db.commit()
            raise
        sent += 1
    print(json.dumps({"status": "execution-paused-or-complete", "requests_this_process": sent,
                      **status(db, args.cap_usd)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
