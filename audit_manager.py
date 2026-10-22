"""
Secure Jarvis - Audit Manager (System 3: Observability & Compliance)
==================================================================

Provides durable, tamper-evident-ish local persistence for every decision
made by the System 1 decision gate. This module is intentionally
dependency-free (Python standard library `sqlite3` only) so that the
compliance substrate remains available even if the AI provider is down.

Storage model
-------------
``command_logs``      -> one row per evaluated command (the System 1 verdict).
``security_alerts``  -> one row per *violation* (blocked destructive ops).

All timestamps are ISO-8601 UTC strings so that the log is trivially
consumable by SIEM tooling / watsonx.governance pipelines.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "DEFAULT_DB_PATH",
    "get_db_path",
    "init_db",
    "log_event",
    "log_security_alert",
    "fetch_recent_logs",
    "fetch_security_alerts",
    "fetch_stats",
    "utc_now_iso",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "audit.db"

# Serializes writes so the REPL thread and the API thread never collide.
_WRITE_LOCK = threading.Lock()

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS command_logs (
        id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        raw_command TEXT NOT NULL,
        intent TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        reasoning TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS security_alerts (
        id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        raw_command TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        intent TEXT,
        confidence REAL,
        guardrail_rule TEXT,
        reasoning TEXT,
        disposition TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_command_logs_timestamp ON command_logs (timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_command_logs_status ON command_logs (status)",
    "CREATE INDEX IF NOT EXISTS idx_security_alerts_timestamp ON security_alerts (timestamp DESC)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def get_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve the audit database path.

    Precedence: explicit argument > ``SECURE_JARVIS_DB`` env var > default
    ``<project>/audit.db``.
    """
    if db_path:
        return Path(db_path).expanduser().resolve()
    env_override = _read_env("SECURE_JARVIS_DB")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return DEFAULT_DB_PATH


def _read_env(key: str) -> str | None:
    """Read an environment variable without requiring python-dotenv to be present."""
    try:
        from dotenv import dotenv_values  # local import keeps this module stdlib-only
    except Exception:  # pragma: no cover - dotenv is a hard requirement in practice
        import os

        return os.environ.get(key)
    value = dotenv_values().get(key)
    return value if value else None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema lifecycle
# ---------------------------------------------------------------------------
def init_db(db_path: Optional[str | Path] = None) -> Path:
    """Create the audit database and its tables if they do not exist.

    Safe to call repeatedly (idempotent). Returns the resolved DB path.
    """
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        conn = _connect(path)
        try:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.commit()
        finally:
            conn.close()
    return path


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------
def log_event(
    raw_command: str,
    intent: str,
    risk_level: str,
    confidence: float,
    status: str,
    latency_ms: float,
    reasoning: str,
    db_path: Optional[str | Path] = None,
    timestamp: Optional[str] = None,
) -> int:
    """Persist a single System 1 decision verdict to ``command_logs``.

    Returns the row id of the inserted record.
    """
    path = get_db_path(db_path)
    init_db(path)
    ts = timestamp or utc_now_iso()
    with _WRITE_LOCK:
        conn = _connect(path)
        try:
            cur = conn.execute(
                """
                INSERT INTO command_logs
                    (timestamp, raw_command, intent, risk_level, confidence,
                     status, latency_ms, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    raw_command,
                    str(intent),
                    str(risk_level),
                    float(confidence),
                    str(status),
                    float(latency_ms),
                    reasoning or "",
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()


def log_security_alert(
    raw_command: str,
    alert_type: str,
    severity: str = "CRITICAL",
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    guardrail_rule: Optional[str] = None,
    reasoning: Optional[str] = None,
    disposition: str = "BLOCKED",
    db_path: Optional[str | Path] = None,
    timestamp: Optional[str] = None,
) -> int:
    """Persist an enterprise compliance violation to ``security_alerts``.

    Returns the row id of the inserted alert.
    """
    path = get_db_path(db_path)
    init_db(path)
    ts = timestamp or utc_now_iso()
    with _WRITE_LOCK:
        conn = _connect(path)
        try:
            cur = conn.execute(
                """
                INSERT INTO security_alerts
                    (timestamp, raw_command, alert_type, severity, intent,
                     confidence, guardrail_rule, reasoning, disposition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    raw_command,
                    alert_type,
                    severity,
                    intent,
                    float(confidence) if confidence is not None else None,
                    guardrail_rule,
                    reasoning,
                    disposition,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------
def fetch_recent_logs(limit: int = 20, db_path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Return the ``limit`` most recent command verdicts, newest first."""
    path = get_db_path(db_path)
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, raw_command, intent, risk_level, confidence,
                   status, latency_ms, reasoning
            FROM command_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_security_alerts(limit: int = 20, db_path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Return the ``limit`` most recent compliance violations, newest first."""
    path = get_db_path(db_path)
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, raw_command, alert_type, severity, intent,
                   confidence, guardrail_rule, reasoning, disposition
            FROM security_alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_stats(db_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Aggregate command counts by status and by risk level for the dashboard."""
    path = get_db_path(db_path)
    empty = {
        "total": 0,
        "approved": 0,
        "blocked": 0,
        "requires_approval": 0,
        "error": 0,
        "critical": 0,
        "avg_latency_ms": 0.0,
        "avg_confidence": 0.0,
        "alerts": 0,
    }
    if not path.exists():
        return empty
    conn = _connect(path)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(AVG(confidence), 0) AS avg_confidence,
                   COALESCE(SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END), 0) AS approved,
                   COALESCE(SUM(CASE WHEN status = 'BLOCKED' THEN 1 ELSE 0 END), 0) AS blocked,
                   COALESCE(SUM(CASE WHEN status = 'REQUIRES_APPROVAL' THEN 1 ELSE 0 END), 0) AS requires_approval,
                   COALESCE(SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END), 0) AS error,
                   COALESCE(SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END), 0) AS critical
            FROM command_logs
            """
        ).fetchone()
        alerts = conn.execute("SELECT COUNT(*) FROM security_alerts").fetchone()[0]
        stats = {k: row[k] for k in row.keys()}
        stats["alerts"] = int(alerts)
        return stats
    finally:
        conn.close()

