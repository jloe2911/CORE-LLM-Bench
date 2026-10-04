#!/usr/bin/env python3
"""Execute only the balanced 90-call pilot from the frozen rerun plan.

The pilot contains the same 15 NL and 15 FS prompts for each of the three
models.  It reuses the full-run SQLite checkpoint and response journal, so a
later explicitly authorized run can resume without repeating pilot requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v1_1_minimum_rerun as core  # noqa: E402


PILOT_REPRESENTATIONS = ("NL", "FS")
PILOT_PER_MODEL_REPRESENTATION = 15
PILOT_CALLS = 90
PILOT_MAX_CAP_USD = 1.0
PILOT_MANIFEST = core.OUTPUT / "pilot_manifest.csv"


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


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if size < count:
        raise RuntimeError(f"Cannot select {count} rows from {size}")
    if count == 1:
        return [0]
    return [(index * (size - 1)) // (count - 1) for index in range(count)]


def select_pilot(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    models = tuple(core.MODELS)
    selected: list[dict[str, str]] = []
    for representation in PILOT_REPRESENTATIONS:
        reference = sorted(
            (
                row for row in rows
                if row["model"] == models[0] and row["representation"] == representation
            ),
            key=lambda row: (
                row["dataset"], row["task_type"], int(row["task_id"]),
                row["deduplication_group"],
            ),
        )
        prompts = [reference[index] for index in evenly_spaced_indices(
            len(reference), PILOT_PER_MODEL_REPRESENTATION
        )]
        for prompt in prompts:
            matches = [
                row for row in rows
                if row["task_id"] == prompt["task_id"]
                and row["representation"] == representation
                and row["corrected_prompt_hash"] == prompt["corrected_prompt_hash"]
            ]
            by_model = {row["model"]: row for row in matches}
            if set(by_model) != set(models):
                raise RuntimeError(
                    f"Pilot prompt lacks a three-model executable triplet: "
                    f"{prompt['task_id']}/{representation}"
                )
            selected.extend(by_model[model] for model in models)

    if len(selected) != PILOT_CALLS:
        raise RuntimeError(f"Pilot selection is {len(selected)} calls, expected {PILOT_CALLS}")
    if len({row["deduplication_group"] for row in selected}) != PILOT_CALLS:
        raise RuntimeError("Pilot request IDs are not unique")
    counts: dict[tuple[str, str], int] = {}
    for row in selected:
        key = (row["model"], row["representation"])
        counts[key] = counts.get(key, 0) + 1
    expected = {
        (model, representation): PILOT_PER_MODEL_REPRESENTATION
        for model in models for representation in PILOT_REPRESENTATIONS
    }
    if counts != expected:
        raise RuntimeError(f"Pilot is not balanced: {counts}")
    return selected


def write_pilot_manifest(rows: list[dict[str, str]]) -> None:
    core.OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with PILOT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pilot_status(db: Any, rows: list[dict[str, str]], cap: float) -> dict[str, Any]:
    ids = [row["deduplication_group"] for row in rows]
    placeholders = ",".join("?" for _ in ids)
    counts = dict(db.execute(
        f"SELECT status,COUNT(*) FROM requests WHERE request_id IN ({placeholders}) GROUP BY status",
        ids,
    ).fetchall())
    charged = db.execute(
        f"SELECT COALESCE(SUM(charged_usd),0) FROM requests WHERE request_id IN ({placeholders})",
        ids,
    ).fetchone()[0]
    return {
        "pilot_counts": counts,
        "pilot_charged_usd": charged,
        "cap_usd": cap,
        "pilot_manifest": str(PILOT_MANIFEST.relative_to(ROOT)),
        "pilot_manifest_sha256": core.sha256_file(PILOT_MANIFEST),
    }


def reconcile_definitive_rejections(db: Any, pilot_ids: set[str]) -> int:
    """Close only explicit HTTP 429 rejections; never retry them."""
    reconciled = 0
    for request_id, error in db.execute(
        "SELECT request_id,error FROM requests WHERE status='in_flight'"
    ).fetchall():
        text = str(error or "")
        if request_id not in pilot_ids:
            raise RuntimeError("Non-pilot request is in flight")
        if not (text.startswith("RateLimitError(") and "code': 429" in text):
            continue
        db.execute(
            "UPDATE requests SET status='failed',charged_usd=0,reserved_usd=0,"
            "completed_at=? WHERE request_id=? AND status='in_flight'",
            (datetime.now(timezone.utc).isoformat(), request_id),
        )
        reconciled += 1
    db.commit()
    return reconciled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="authorize the paid pilot")
    parser.add_argument("--cap-usd", type=float, default=PILOT_MAX_CAP_USD)
    args = parser.parse_args()
    if not (0 < args.cap_usd <= PILOT_MAX_CAP_USD):
        raise RuntimeError("The pilot cap must be greater than $0 and no more than $1")

    rows, benchmark = core.load_plan()
    pilot = select_pilot(rows)
    write_pilot_manifest(pilot)
    db = core.connect(rows)
    pilot_ids = {row["deduplication_group"] for row in pilot}
    reconciled_rejections = reconcile_definitive_rejections(db, pilot_ids)
    nonpilot_active = db.execute(
        "SELECT COUNT(*) FROM requests WHERE status!='pending' AND request_id NOT IN ("
        + ",".join("?" for _ in pilot_ids) + ")",
        tuple(pilot_ids),
    ).fetchone()[0]
    if nonpilot_active:
        raise RuntimeError("Checkpoint contains non-pilot activity; refusing controlled pilot")
    unresolved = db.execute("SELECT COUNT(*) FROM requests WHERE status='in_flight'").fetchone()[0]
    if unresolved:
        raise RuntimeError("Unresolved in-flight request exists; reconcile before resuming")

    reserved_bound = sum(core.reservation(row) for row in pilot)
    summary = {
        "status": "pilot-preflight-passed",
        "paid_requests_this_process": 0,
        "selected_calls": len(pilot),
        "maximum_reserved_cost_usd": reserved_bound,
        "definitive_rejections_reconciled": reconciled_rejections,
        **pilot_status(db, pilot, args.cap_usd),
    }
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    key = load_dotenv_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable; no paid request was made")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)

    sent = 0
    for row in pilot:
        request_id = row["deduplication_group"]
        current = db.execute(
            "SELECT status FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()[0]
        if current in {"completed", "failed"}:
            continue
        if current != "pending":
            raise RuntimeError(f"Pilot request is not resumable: {request_id}={current}")
        reserve = core.claim(db, row, args.cap_usd)
        try:
            record, charged = core.execute_one(
                client, row, benchmark[row["task_id"]], reserve
            )
            core.append_jsonl(core.RESPONSES, record)
            digest = hashlib.sha256(core.canonical(record).encode()).hexdigest()
            db.execute(
                "UPDATE requests SET status='completed',charged_usd=?,reserved_usd=0,"
                "completed_at=?,response_sha256=? WHERE request_id=?",
                (charged, datetime.now(timezone.utc).isoformat(), digest, request_id),
            )
            db.commit()
        except Exception as error:
            definitive_rejection = (
                error.__class__.__name__ == "RateLimitError"
                and getattr(error, "status_code", None) == 429
            )
            if definitive_rejection:
                db.execute(
                    "UPDATE requests SET status='failed',charged_usd=0,reserved_usd=0,"
                    "completed_at=?,error=? WHERE request_id=?",
                    (datetime.now(timezone.utc).isoformat(), repr(error), request_id),
                )
                db.commit()
                sent += 1
                continue
            db.execute("UPDATE requests SET error=? WHERE request_id=?", (repr(error), request_id))
            db.commit()
            print(json.dumps({
                "status": "pilot-stopped-on-unresolved-request",
                "failed_request": request_id,
                "error": repr(error),
                "paid_requests_this_process": sent + 1,
                **pilot_status(db, pilot, args.cap_usd),
            }, indent=2, sort_keys=True))
            return 2
        sent += 1

    print(json.dumps({
        "status": "pilot-complete",
        "paid_requests_this_process": sent,
        **pilot_status(db, pilot, args.cap_usd),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
