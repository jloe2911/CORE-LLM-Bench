from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resume_v1_1_minimum_rerun as resume  # noqa: E402


class HttpFailure(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class ConnectTimeout(Exception):
    pass


class ReadTimeout(Exception):
    pass


class APIConnectionError(Exception):
    pass


def test_retry_classification_is_narrow_and_explicit() -> None:
    for status in (429, 502, 503, 504):
        assert resume.classify_transient_error(HttpFailure(status)) == f"http_{status}"
    for status in (400, 401, 403, 404, 409, 422, 500, 505):
        assert resume.classify_transient_error(HttpFailure(status)) is None

    connect_wrapper = APIConnectionError("request failed")
    connect_wrapper.__cause__ = ConnectTimeout("connect timed out")
    assert resume.classify_transient_error(connect_wrapper) == "connect_timeout"

    read_wrapper = APIConnectionError("request failed")
    read_wrapper.__cause__ = ReadTimeout("read timed out")
    assert resume.classify_transient_error(read_wrapper) == "read_timeout"

    assert resume.classify_transient_error(
        RuntimeError("Provider/model mismatch: wrong/wrong")
    ) is None
    assert resume.classify_transient_error(
        RuntimeError("connection timeout text without a typed timeout")
    ) is None


def make_checkpoint() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE requests (request_id TEXT PRIMARY KEY, model TEXT, task_id TEXT, "
        "representation TEXT, input_hash TEXT, status TEXT, reserved_usd REAL, "
        "charged_usd REAL, claimed_at TEXT, completed_at TEXT, response_sha256 TEXT, "
        "error TEXT, uncertain_charged_usd REAL NOT NULL DEFAULT 0)"
    )
    db.execute(
        "CREATE TABLE execution_attempts (request_id TEXT, attempt_number INTEGER, "
        "model TEXT, representation TEXT, started_at TEXT, finished_at TEXT, "
        "status TEXT, cause TEXT, backoff_seconds REAL DEFAULT 0, "
        "PRIMARY KEY(request_id,attempt_number))"
    )
    db.execute(
        "INSERT INTO requests VALUES "
        "('request-test','GPT-5 mini','1','FS','hash','in_flight',0.002,0,"
        "'start',NULL,NULL,'503 failure',0)"
    )
    db.execute(
        "INSERT INTO execution_attempts VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "request-test", 1, "GPT-5 mini", "FS", "start", "finish",
            "unresolved", "InternalServerError(\"code': 503\")", 0,
        ),
    )
    db.commit()
    return db


def test_reconcile_only_unpersisted_uncharged_transient(tmp_path: Path) -> None:
    db = make_checkpoint()
    responses = tmp_path / "responses.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    event = resume.reconcile_transient_unpersisted_request(
        db, "request-test", responses, attempts
    )
    row = db.execute(
        "SELECT status,reserved_usd,charged_usd,claimed_at,error FROM requests"
    ).fetchone()
    assert row == ("pending", 0.0, 0.0, None, None)
    assert db.execute(
        "SELECT status FROM execution_attempts"
    ).fetchone()[0] == "transient_reconciled"
    assert event["cause"].startswith("HTTP 503")
    assert json.loads(attempts.read_text(encoding="utf-8"))["request_id"] == "request-test"


@pytest.mark.parametrize("unsafe", ["charged", "persisted"])
def test_reconcile_refuses_charge_or_persisted_response(
    tmp_path: Path, unsafe: str
) -> None:
    db = make_checkpoint()
    responses = tmp_path / "responses.jsonl"
    if unsafe == "charged":
        db.execute("UPDATE requests SET charged_usd=0.001")
        db.commit()
    else:
        responses.write_text(
            json.dumps({"request_id": "request-test"}) + "\n", encoding="utf-8"
        )
    with pytest.raises(RuntimeError):
        resume.reconcile_transient_unpersisted_request(
            db, "request-test", responses, tmp_path / "attempts.jsonl"
        )


def test_timeout_retry_reservation_preserves_cap() -> None:
    db = make_checkpoint()
    assert resume.reserve_uncertain_timeout_charge(db, "request-test", 0.002, 0.01)
    assert db.execute(
        "SELECT uncertain_charged_usd,reserved_usd FROM requests"
    ).fetchone() == (0.002, 0.002)
    assert not resume.reserve_uncertain_timeout_charge(db, "request-test", 0.002, 0.005)
    assert db.execute(
        "SELECT uncertain_charged_usd FROM requests"
    ).fetchone()[0] == 0.002


def test_retry_policy_is_bounded() -> None:
    assert resume.RETRY_BACKOFF_SECONDS == (5, 10, 20, 40, 80)
    assert len(resume.RETRY_BACKOFF_SECONDS) == 5


def make_execution_checkpoint() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE requests (request_id TEXT PRIMARY KEY, model TEXT, task_id TEXT, "
        "representation TEXT, input_hash TEXT, status TEXT, reserved_usd REAL DEFAULT 0, "
        "charged_usd REAL DEFAULT 0, claimed_at TEXT, completed_at TEXT, "
        "response_sha256 TEXT, error TEXT, uncertain_charged_usd REAL DEFAULT 0)"
    )
    db.execute(
        "INSERT INTO requests(request_id,model,task_id,representation,input_hash,status) "
        "VALUES ('request-execute','GPT-5 mini','1','FS','hash','pending')"
    )
    resume.ensure_attempt_table(db)
    return db


def test_execute_retries_same_claim_route_and_then_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_execution_checkpoint()
    row = {
        "deduplication_group": "request-execute",
        "model": "GPT-5 mini",
        "task_id": "1",
        "representation": "FS",
        "historical_input_tokens": "100",
    }
    calls: list[tuple[str, str]] = []
    sleeps: list[int] = []

    def fake_execute_one(client, observed_row, bench, reserve):
        calls.append((observed_row["model"], resume.core.MODELS[observed_row["model"]]["model_id"]))
        if len(calls) < 3:
            raise HttpFailure(503)
        return {"request_id": "request-execute", "status": "usable"}, 0.001

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kwargs: object())
    )
    monkeypatch.setattr(resume.core, "execute_one", fake_execute_one)
    monkeypatch.setattr(resume.core, "RESPONSES", tmp_path / "responses.jsonl")
    monkeypatch.setattr(resume, "ATTEMPT_JOURNAL", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(resume.time, "sleep", sleeps.append)
    assert resume.execute(db, [row], {"1": object()}, 25.0) == 0

    assert calls == [
        ("GPT-5 mini", "openai/gpt-5-mini"),
        ("GPT-5 mini", "openai/gpt-5-mini"),
        ("GPT-5 mini", "openai/gpt-5-mini"),
    ]
    assert sleeps == [5, 10]
    assert db.execute(
        "SELECT status,reserved_usd,charged_usd FROM requests"
    ).fetchone() == ("completed", 0.0, 0.001)
    assert db.execute(
        "SELECT status FROM execution_attempts ORDER BY attempt_number"
    ).fetchall() == [("transient_retryable",), ("transient_retryable",), ("completed",)]


def test_execute_exhaustion_retains_atomic_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = make_execution_checkpoint()
    row = {
        "deduplication_group": "request-execute",
        "model": "GPT-5 mini",
        "task_id": "1",
        "representation": "FS",
        "historical_input_tokens": "100",
    }
    calls = 0
    sleeps: list[int] = []

    def always_503(client, observed_row, bench, reserve):
        nonlocal calls
        calls += 1
        raise HttpFailure(503)

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kwargs: object())
    )
    monkeypatch.setattr(resume.core, "execute_one", always_503)
    monkeypatch.setattr(resume.core, "RESPONSES", tmp_path / "responses.jsonl")
    monkeypatch.setattr(resume, "ATTEMPT_JOURNAL", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(resume.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="claim retained in flight"):
        resume.execute(db, [row], {"1": object()}, 25.0)

    assert calls == 6
    assert sleeps == [5, 10, 20, 40, 80]
    assert db.execute(
        "SELECT status,reserved_usd,charged_usd FROM requests"
    ).fetchone()[0] == "in_flight"
    assert db.execute(
        "SELECT status FROM execution_attempts ORDER BY attempt_number DESC LIMIT 1"
    ).fetchone()[0] == "retry_exhausted"
