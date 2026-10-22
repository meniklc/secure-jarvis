"""
Secure Jarvis - Mock Tool Layer (System 2: Operations Execution)
================================================================

These functions represent the *only* capabilities an approved agent
command may reach. They are deterministic simulations: no real cluster is
touched, no real database is mutated. Their purpose is to prove that
System 2 is strictly downstream of an APPROVED System 1 verdict.

Every function returns a plain ``dict`` so the CLI can render it in a
Rich table without additional adapters.
"""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "get_cluster_metrics",
    "deploy_staging",
    "query_analytics",
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(text: str, length: int = 12) -> str:
    """Deterministic pseudo-commit hash so repeated runs look consistent."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def get_cluster_metrics() -> dict[str, Any]:
    """Simulate reading live infrastructure telemetry for the platform fleet.

    Returns CPU/memory utilisation, uptime and per-node status for the
    simulated Kubernetes control plane.
    """
    time.sleep(0.05)  # simulate network round-trip to the monitoring API
    nodes = [
        {"node": "ip-10-0-1-14", "role": "control-plane", "status": "Healthy"},
        {"node": "ip-10-0-2-31", "role": "worker", "status": "Healthy"},
        {"node": "ip-10-0-2-44", "role": "worker", "status": "Degraded"},
        {"node": "ip-10-0-3-09", "role": "worker", "status": "Healthy"},
    ]
    healthy = sum(1 for n in nodes if n["status"] == "Healthy")
    return {
        "tool": "get_cluster_metrics",
        "queried_at": _iso_now(),
        "cluster": "jarvis-prod-sim",
        "version": "v1.29.4",
        "cpu_usage_pct": round(random.uniform(28.0, 62.0), 1),
        "memory_usage_pct": round(random.uniform(41.0, 74.0), 1),
        "disk_usage_pct": round(random.uniform(35.0, 68.0), 1),
        "uptime_hours": round(random.uniform(120.0, 900.0), 1),
        "nodes_total": len(nodes),
        "nodes_healthy": healthy,
        "nodes_degraded": len(nodes) - healthy,
        "nodes": nodes,
    }


def deploy_staging(service_name: str) -> dict[str, Any]:
    """Simulate a rolling deployment of ``service_name`` into the staging tier."""
    time.sleep(0.08)  # simulate CI/CD pipeline latency
    service = (service_name or "unknown-service").strip() or "unknown-service"
    commit = _stable_hash(service + _iso_now(), 10)
    build_id = f"build-{commit[:6]}-{int(time.time() * 1000) % 100000}"
    return {
        "tool": "deploy_staging",
        "service_name": service,
        "build_id": build_id,
        "commit_hash": commit,
        "environment": "staging",
        "status": "SUCCESS",
        "replicas_ready": 3,
        "replicas_desired": 3,
        "rolled_out_at": _iso_now(),
        "pipeline": [
            {"stage": "lint", "result": "PASS", "duration_s": 4.2},
            {"stage": "unit-test", "result": "PASS", "duration_s": 31.8},
            {"stage": "build-image", "result": "PASS", "duration_s": 22.5},
            {"stage": "deploy-staging", "result": "SUCCESS", "duration_s": 9.7},
        ],
        "confirmed": True,
    }


def query_analytics() -> dict[str, Any]:
    """Simulate a read-only analytics query against the enterprise data warehouse."""
    time.sleep(0.06)  # simulate warehouse round-trip
    return {
        "tool": "query_analytics",
        "executed_at": _iso_now(),
        "warehouse": "snowflake-sim",
        "query": "SELECT date_trunc('day', created_at) AS d, count(*) FROM events GROUP BY 1",
        "mode": "READ_ONLY",
        "rows_returned": 7,
        "scan_bytes": 41_286_144,
        "duration_ms": 184,
        "results": [
            {"day": "2026-10-16", "events": 18_402},
            {"day": "2026-10-17", "events": 19_115},
            {"day": "2026-10-18", "events": 17_998},
            {"day": "2026-10-19", "events": 21_340},
            {"day": "2026-10-20", "events": 23_876},
            {"day": "2026-10-21", "events": 20_551},
            {"day": "2026-10-22", "events": 24_119},
        ],
    }
