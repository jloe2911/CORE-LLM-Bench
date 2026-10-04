#!/usr/bin/env python3
"""Resume only unresolved calls in the frozen v1.1 minimum-rerun manifest.

Execution is sequential and ordered Gemini, GPT-5 mini, then Qwen. Clearly
transient transport/provider failures are retried while retaining the same
atomic request claim and the frozen provider/model route. All other failures
remain in flight and stop the process for explicit reconciliation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v1_1_minimum_rerun as core  # noqa: E402


FREEZE_MANIFEST = core.AUDIT / "artifact_freeze_manifest.json"
ATTEMPT_JOURNAL = core.OUTPUT / "retry_attempts.jsonl"
MODEL_ORDER = (
    "Gemini 2.5 Flash-Lite",
    "GPT-5 mini",
    "Qwen3-30B-A3B-Instruct",
)
CAP_USD = 25.0
RETRY_BACKOFF_SECONDS = (5, 10, 20, 40, 80)
RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503, 504})
RETRYABLE_TIMEOUT_TYPES = frozenset({"ConnectTimeout", "ReadTimeout"})
PILOT_429_REQUEST_IDS = {
    "request-006243",
    "request-007828",
    "request-006320",
    "request-006905",
    "request-008623",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dotenv_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    path = ROOT / ".env"
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        if key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return ""


def verify_frozen_artifacts() -> dict[str, Any]:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            mismatches.append({"path": item["path"], "problem": "missing"})
            continue
        actual_hash = core.sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != item["sha256"] or actual_size != item["bytes"]:
            mismatches.append({
                "path": item["path"], "problem": "hash_or_size_mismatch",
                "actual_sha256": actual_hash, "actual_bytes": actual_size,
            })
    if mismatches:
        raise RuntimeError(f"Frozen artifact validation failed: {mismatches}")
    return {"checked": len(manifest["files"]), "mismatches": 0}


def ensure_attempt_table(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS execution_attempts ("
        "request_id TEXT NOT NULL, attempt_number INTEGER NOT NULL, "
        "model TEXT NOT NULL, representation TEXT NOT NULL, "
        "started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, "
        "cause TEXT, backoff_seconds REAL NOT NULL DEFAULT 0, "
        "PRIMARY KEY(request_id,attempt_number))"
    )
    columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
    if "uncertain_charged_usd" not in columns:
        db.execute(
            "ALTER TABLE requests ADD COLUMN uncertain_charged_usd "
            "REAL NOT NULL DEFAULT 0"
        )
    db.commit()


def append_retry_event(value: dict[str, Any]) -> None:
    core.append_jsonl(ATTEMPT_JOURNAL, value)


def next_attempt_number(db: sqlite3.Connection, request_id: str) -> int:
    value = db.execute(
        "SELECT COALESCE(MAX(attempt_number),0)+1 FROM execution_attempts WHERE request_id=?",
        (request_id,),
    ).fetchone()[0]
    return int(value)


def record_historical_pilot_rejections(db: sqlite3.Connection) -> int:
    """Explicitly reconcile only the five known pilot 429 failures."""
    failures = db.execute(
        "SELECT request_id,model,representation,claimed_at,completed_at,error "
        "FROM requests WHERE status='failed'"
    ).fetchall()
    if not failures:
        placeholders = ",".join("?" for _ in PILOT_429_REQUEST_IDS)
        historical = {
            row[0] for row in db.execute(
                f"SELECT request_id FROM execution_attempts WHERE attempt_number=1 "
                f"AND status='rate_limited' AND request_id IN ({placeholders})",
                tuple(sorted(PILOT_429_REQUEST_IDS)),
            ).fetchall()
        }
        if historical == PILOT_429_REQUEST_IDS:
            return 0
        raise RuntimeError(
            "The five original pilot 429 reconciliation records are incomplete: "
            f"{sorted(historical)}"
        )
    if len(failures) != 5:
        raise RuntimeError(f"Expected exactly five pilot failures, found {len(failures)}")
    reconciled = 0
    for request_id, model, representation, claimed_at, completed_at, error in failures:
        text = str(error or "")
        if model != "Gemini 2.5 Flash-Lite" or not (
            text.startswith("RateLimitError(") and "code': 429" in text
        ):
            raise RuntimeError(f"Non-reconcilable failed request: {request_id}")
        present = db.execute(
            "SELECT COUNT(*) FROM execution_attempts WHERE request_id=?",
            (request_id,),
        ).fetchone()[0]
        if not present:
            db.execute(
                "INSERT INTO execution_attempts(request_id,attempt_number,model,representation,"
                "started_at,finished_at,status,cause) VALUES(?,?,?,?,?,?,?,?)",
                (request_id, 1, model, representation, claimed_at or utc_now(),
                 completed_at or utc_now(), "rate_limited", text),
            )
            append_retry_event({
                "request_id": request_id, "attempt_number": 1, "model": model,
                "representation": representation, "status": "rate_limited",
                "cause": "HTTP 429 upstream_provider_shared_pool",
                "historical_pilot_attempt": True,
            })
        changed = db.execute(
            "UPDATE requests SET status='pending',reserved_usd=0,claimed_at=NULL,"
            "completed_at=NULL WHERE request_id=? AND status='failed'",
            (request_id,),
        ).rowcount
        reconciled += changed
    db.commit()
    return reconciled


def grouped_status(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {"model": model, "representation": representation, "status": status, "count": count}
        for model, representation, status, count in db.execute(
            "SELECT model,representation,status,COUNT(*) FROM requests "
            "GROUP BY model,representation,status ORDER BY model,representation,status"
        ).fetchall()
    ]


def checkpoint_status(db: sqlite3.Connection, cap: float) -> dict[str, Any]:
    counts = dict(db.execute(
        "SELECT status,COUNT(*) FROM requests GROUP BY status"
    ).fetchall())
    charged, uncertain, reserved = db.execute(
        "SELECT COALESCE(SUM(charged_usd),0),"
        "COALESCE(SUM(uncertain_charged_usd),0),"
        "COALESCE(SUM(CASE WHEN status='in_flight' THEN reserved_usd ELSE 0 END),0) "
        "FROM requests"
    ).fetchone()
    return {
        "counts": counts,
        "charged_usd": charged,
        "uncertain_charged_usd": uncertain,
        "in_flight_reserved_usd": reserved,
        "cap_usd": cap,
        "remaining_unreserved_usd": cap - charged - uncertain - reserved,
    }


def claim(db: sqlite3.Connection, row: dict[str, str], cap: float) -> float:
    """Atomically claim one logical request, including conservative timeout holds."""
    reserve = core.reservation(row)
    db.execute("BEGIN IMMEDIATE")
    current = db.execute(
        "SELECT status FROM requests WHERE request_id=?",
        (row["deduplication_group"],),
    ).fetchone()[0]
    charged, uncertain, held = db.execute(
        "SELECT COALESCE(SUM(charged_usd),0),"
        "COALESCE(SUM(uncertain_charged_usd),0),"
        "COALESCE(SUM(CASE WHEN status='in_flight' THEN reserved_usd ELSE 0 END),0) "
        "FROM requests"
    ).fetchone()
    if current != "pending":
        db.rollback()
        raise RuntimeError(
            f"Request is not pending: {row['deduplication_group']}={current}"
        )
    if charged + uncertain + held + reserve > cap:
        db.rollback()
        raise RuntimeError("$25 spending cap gate stopped before the next request")
    changed = db.execute(
        "UPDATE requests SET status='in_flight',reserved_usd=?,claimed_at=? "
        "WHERE request_id=? AND status='pending'",
        (reserve, utc_now(), row["deduplication_group"]),
    ).rowcount
    if changed != 1:
        db.rollback()
        raise RuntimeError("Atomic request claim failed")
    db.commit()
    return reserve


def exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_transient_error(error: BaseException) -> str | None:
    """Return a narrow retry reason; ambiguous failures are non-retryable."""
    status_code = getattr(error, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code in RETRYABLE_HTTP_STATUSES:
        return f"http_{status_code}"
    for item in exception_chain(error):
        if type(item).__name__ in RETRYABLE_TIMEOUT_TYPES:
            return type(item).__name__.removesuffix("Timeout").lower() + "_timeout"
    return None


def reserve_uncertain_timeout_charge(
    db: sqlite3.Connection, request_id: str, reserve: float, cap: float
) -> bool:
    """Conservatively hold a possibly charged timed-out attempt before retrying."""
    db.execute("BEGIN IMMEDIATE")
    row = db.execute(
        "SELECT status,reserved_usd FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    charged, uncertain, held = db.execute(
        "SELECT COALESCE(SUM(charged_usd),0),"
        "COALESCE(SUM(uncertain_charged_usd),0),"
        "COALESCE(SUM(CASE WHEN status='in_flight' THEN reserved_usd ELSE 0 END),0) "
        "FROM requests"
    ).fetchone()
    if row is None or row[0] != "in_flight" or abs(float(row[1]) - reserve) > 1e-12:
        db.rollback()
        raise RuntimeError("Atomic retry claim changed before timeout accounting")
    if charged + uncertain + held + reserve > cap:
        db.rollback()
        return False
    db.execute(
        "UPDATE requests SET uncertain_charged_usd=uncertain_charged_usd+? "
        "WHERE request_id=? AND status='in_flight'",
        (reserve, request_id),
    )
    db.commit()
    return True


def response_record_count(path: Path, request_id: str) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Malformed response journal line {line_number}"
                ) from error
            count += record.get("request_id") == request_id
    return count


def reconcile_transient_unpersisted_request(
    db: sqlite3.Connection,
    request_id: str,
    responses_path: Path = core.RESPONSES,
    attempt_journal_path: Path = ATTEMPT_JOURNAL,
) -> dict[str, Any]:
    """Return one explicit transient rejection to pending after local proof."""
    active = db.execute(
        "SELECT model,status,reserved_usd,charged_usd,uncertain_charged_usd,error "
        "FROM requests WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if active is None or active[1] != "in_flight":
        raise RuntimeError(f"Request is not an unresolved in-flight claim: {request_id}")
    other_in_flight = db.execute(
        "SELECT COUNT(*) FROM requests WHERE status='in_flight' AND request_id!=?",
        (request_id,),
    ).fetchone()[0]
    if other_in_flight:
        raise RuntimeError("More than the selected request is in flight")
    if float(active[3]) != 0 or float(active[4]) != 0:
        raise RuntimeError("Request has charged or uncertain cost; refusing reconciliation")
    persisted = response_record_count(responses_path, request_id)
    if persisted:
        raise RuntimeError("A response is already persisted; refusing reconciliation")
    attempt = db.execute(
        "SELECT attempt_number,status,cause FROM execution_attempts "
        "WHERE request_id=? ORDER BY attempt_number DESC LIMIT 1",
        (request_id,),
    ).fetchone()
    if attempt is None or attempt[1] != "unresolved":
        raise RuntimeError("Latest attempt is not an unresolved failure")
    match = re.search(r"(?:code[=: ]+|code['\"]?:\s*)(429|502|503|504)\b", str(attempt[2]))
    if match is None:
        raise RuntimeError("Failure is not an explicit retryable HTTP rejection")
    reconciled_at = utc_now()
    db.execute("BEGIN IMMEDIATE")
    changed = db.execute(
        "UPDATE requests SET status='pending',reserved_usd=0,claimed_at=NULL,"
        "completed_at=NULL,response_sha256=NULL,error=NULL WHERE request_id=? "
        "AND status='in_flight' AND charged_usd=0 AND uncertain_charged_usd=0",
        (request_id,),
    ).rowcount
    if changed != 1:
        db.rollback()
        raise RuntimeError("Atomic reconciliation failed")
    db.execute(
        "UPDATE execution_attempts SET status='transient_reconciled' "
        "WHERE request_id=? AND attempt_number=? AND status='unresolved'",
        (request_id, attempt[0]),
    )
    db.commit()
    event = {
        "request_id": request_id,
        "attempt_number": attempt[0],
        "model": active[0],
        "status": "transient_reconciled",
        "cause": f"HTTP {match.group(1)} with no persisted response or local charge",
        "released_reservation_usd": float(active[2]),
        "timestamp": reconciled_at,
    }
    core.append_jsonl(attempt_journal_path, event)
    return event


def execute(db: sqlite3.Connection, rows: list[dict[str, str]], benchmark: Any, cap: float) -> int:
    key = load_dotenv_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unavailable; no paid request made")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    ordered = [row for model in MODEL_ORDER for row in rows if row["model"] == model]
    completed_this_process = 0
    network_attempts_this_process = 0

    for row in ordered:
        request_id = row["deduplication_group"]
        current = db.execute(
            "SELECT status FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()[0]
        if current == "completed":
            continue
        if current != "pending":
            raise RuntimeError(f"Unresolved non-pending request: {request_id}={current}")

        reserve = claim(db, row, cap)
        for retry_index in range(len(RETRY_BACKOFF_SECONDS) + 1):
            attempt_number = next_attempt_number(db, request_id)
            started_at = utc_now()
            db.execute(
                "INSERT INTO execution_attempts(request_id,attempt_number,model,representation,"
                "started_at,status) VALUES(?,?,?,?,?,?)",
                (request_id, attempt_number, row["model"], row["representation"],
                 started_at, "in_flight"),
            )
            db.commit()
            network_attempts_this_process += 1
            try:
                record, charged = core.execute_one(
                    client, row, benchmark[row["task_id"]], reserve
                )
                core.append_jsonl(core.RESPONSES, record)
                digest = hashlib.sha256(core.canonical(record).encode()).hexdigest()
                finished_at = utc_now()
                db.execute(
                    "UPDATE requests SET status='completed',charged_usd=?,reserved_usd=0,"
                    "completed_at=?,response_sha256=?,error=NULL WHERE request_id=?",
                    (charged, finished_at, digest, request_id),
                )
                db.execute(
                    "UPDATE execution_attempts SET finished_at=?,status='completed' "
                    "WHERE request_id=? AND attempt_number=?",
                    (finished_at, request_id, attempt_number),
                )
                db.commit()
                completed_this_process += 1
                if completed_this_process % 100 == 0:
                    print(json.dumps({
                        "event": "progress", "completed_this_process": completed_this_process,
                        "network_attempts_this_process": network_attempts_this_process,
                        "current_model": row["model"], **checkpoint_status(db, cap),
                    }, sort_keys=True), flush=True)
                break
            except Exception as error:
                reason = classify_transient_error(error)
                exhausted = retry_index == len(RETRY_BACKOFF_SECONDS)
                finished_at = utc_now()
                if reason is None:
                    db.execute(
                        "UPDATE requests SET error=? WHERE request_id=?",
                        (repr(error), request_id),
                    )
                    db.execute(
                        "UPDATE execution_attempts SET finished_at=?,status='unresolved',cause=? "
                        "WHERE request_id=? AND attempt_number=?",
                        (utc_now(), repr(error), request_id, attempt_number),
                    )
                    db.commit()
                    raise
                backoff = 0 if exhausted else RETRY_BACKOFF_SECONDS[retry_index]
                if not exhausted and reason in {"connect_timeout", "read_timeout"}:
                    if not reserve_uncertain_timeout_charge(db, request_id, reserve, cap):
                        exhausted = True
                        backoff = 0
                        reason += "_cap_blocked"
                db.execute(
                    "UPDATE requests SET error=? WHERE request_id=?",
                    (repr(error), request_id),
                )
                db.execute(
                    "UPDATE execution_attempts SET finished_at=?,status=?,cause=?,"
                    "backoff_seconds=? WHERE request_id=? AND attempt_number=?",
                    (finished_at, "retry_exhausted" if exhausted else "transient_retryable",
                     repr(error), backoff, request_id, attempt_number),
                )
                db.commit()
                append_retry_event({
                    "request_id": request_id, "attempt_number": attempt_number,
                    "model": row["model"], "representation": row["representation"],
                    "requested_model": core.MODELS[row["model"]]["model_id"],
                    "requested_provider": core.MODELS[row["model"]]["provider"],
                    "status": "retry_exhausted" if exhausted else "transient_retryable",
                    "cause": reason, "backoff_seconds": backoff,
                    "terminal": exhausted, "timestamp": finished_at,
                })
                print(json.dumps({
                    "event": "transient_failure", "request_id": request_id,
                    "attempt_number": attempt_number, "cause": reason,
                    "terminal": exhausted, "backoff_seconds": backoff,
                }, sort_keys=True), flush=True)
                if exhausted:
                    raise RuntimeError(
                        f"Transient retry policy exhausted for {request_id}; "
                        "claim retained in flight for reconciliation"
                    ) from error
                time.sleep(backoff)

    print(json.dumps({
        "status": "execution-complete-or-exhausted",
        "completed_this_process": completed_this_process,
        "network_attempts_this_process": network_attempts_this_process,
        "grouped_status": grouped_status(db), **checkpoint_status(db, cap),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cap-usd", type=float, default=CAP_USD)
    parser.add_argument(
        "--reconcile-request",
        help="offline-only reconciliation of one explicit transient rejection",
    )
    args = parser.parse_args()
    if args.cap_usd != CAP_USD:
        raise RuntimeError("This controlled resume requires the frozen $25 cap")
    freeze = verify_frozen_artifacts()
    rows, benchmark = core.load_plan()
    db = core.connect(rows)
    ensure_attempt_table(db)
    reconciliation = None
    if args.reconcile_request:
        if args.execute:
            raise RuntimeError("Reconciliation and paid execution must be separate invocations")
        reconciliation = reconcile_transient_unpersisted_request(
            db, args.reconcile_request
        )
    if db.execute("SELECT COUNT(*) FROM requests WHERE status='in_flight'").fetchone()[0]:
        raise RuntimeError("Unresolved in-flight request exists; refusing resume")
    reconciled = record_historical_pilot_rejections(db)
    preflight = {
        "status": "resume-preflight-passed", "freeze": freeze,
        "reconciled_pilot_429_failures": reconciled,
        "reconciled_transient_request": reconciliation,
        "model_order": MODEL_ORDER, "concurrency": 1,
        "transient_retry_backoff_seconds": list(RETRY_BACKOFF_SECONDS),
        "retryable_http_statuses": sorted(RETRYABLE_HTTP_STATUSES),
        "retryable_timeout_types": sorted(RETRYABLE_TIMEOUT_TYPES),
        "maximum_retries_after_initial_attempt": len(RETRY_BACKOFF_SECONDS),
        **checkpoint_status(db, args.cap_usd),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    return execute(db, rows, benchmark, args.cap_usd)


if __name__ == "__main__":
    raise SystemExit(main())
