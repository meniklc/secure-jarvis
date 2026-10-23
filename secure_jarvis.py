"""
Secure Jarvis - Enterprise Agent Governance CLI
================================================

A dual-system autonomous agent governance framework.

    System 1  (decision_engine.py)  Semantic AI classification + deterministic
                                   guardrail enforcement. Nothing executes
                                   until this stage returns APPROVED.
    System 2  (mock_tools.py)      Safe simulated infrastructure / database
                                   operations. Only reachable via APPROVED.
    System 3  (audit_manager.py)   SQLite persistence of every decision,
                                   latency metric and compliance violation.

Usage
-----
    python secure_jarvis.py "show me cluster cpu metrics"   # argument mode
    python secure_jarvis.py                                    # interactive REPL
    python secure_jarvis.py --audit                            # audit viewer
    python secure_jarvis.py --setup                            # configure API key
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Confirm, Prompt

from audit_manager import (
    fetch_recent_logs,
    fetch_security_alerts,
    fetch_stats,
    init_db,
    log_event,
    log_security_alert,
)
from decision_engine import (
    AIDecisionEngine,
    Decision,
    DecisionEngineError,
    DecisionStatus,
    GuardrailCategory,
    Intent,
    RiskLevel,
    load_api_key,
)
from mock_tools import deploy_staging, get_cluster_metrics, query_analytics

BASE_DIR = Path(__file__).resolve().parent

__all__ = ["main", "SecureJarvisApp"]

# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------
THEME = Theme(
    {
        "approved": "bold green",
        "blocked": "bold red",
        "warning": "bold yellow",
        "critical": "bold red",
        "info": "bold cyan",
        "muted": "dim white",
        "metric": "bold magenta",
        "header": "bold white on dark_blue",
    }
)

console = Console(theme=THEME)

# Colour mapping for each verdict.
STATUS_STYLE = {
    DecisionStatus.APPROVED: "approved",
    DecisionStatus.BLOCKED: "blocked",
    DecisionStatus.REQUIRES_APPROVAL: "warning",
    DecisionStatus.ERROR: "blocked",
}

RISK_STYLE = {
    RiskLevel.SAFE: "approved",
    RiskLevel.REQUIRES_APPROVAL: "warning",
    RiskLevel.CRITICAL: "blocked",
}

# Friendly labels for the guardrail categories shown in the violation box.
GUARDRAIL_LABELS = {
    GuardrailCategory.NONE: "None",
    GuardrailCategory.DATA_WIPE: "Data Wipe / Destruction",
    GuardrailCategory.SCHEMA_DESTRUCTION: "Schema Destruction",
    GuardrailCategory.CREDENTIAL_EXFILTRATION: "Credential Exfiltration",
    GuardrailCategory.UNAUTHORIZED_PROD_RESTART: "Unauthorized Production Restart",
    GuardrailCategory.PRIVILEGE_ESCALATION: "Privilege Escalation",
    GuardrailCategory.OTHER_DESTRUCTIVE: "Other Destructive Operation",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _short_timestamp(raw: str) -> str:
    """Compact an ISO-8601 timestamp for narrow table columns (MM-DD HH:MM)."""
    text = str(raw or "")
    if "T" in text:
        text = text.replace("T", " ")
    if len(text) >= 16:
        return text[5:16]  # MM-DD HH:MM
    return text[:16]


# Compact display labels for the audit table so it fits an 80-column terminal.
INTENT_SHORT = {
    "INFRASTRUCTURE_METRICS": "INFRA_METRICS",
    "STAGING_DEPLOYMENT": "STAGING_DEPLOY",
    "DATABASE_QUERY": "DB_QUERY",
    "DESTRUCTIVE_OPS": "DESTRUCTIVE",
    "UNKNOWN": "UNKNOWN",
}

RISK_SHORT = {"SAFE": "SAFE", "REQUIRES_APPROVAL": "REVIEW", "CRITICAL": "CRITICAL"}

STATUS_SHORT = {
    "APPROVED": "APPROVED",
    "BLOCKED": "BLOCKED",
    "REQUIRES_APPROVAL": "REVIEW",
    "ERROR": "ERROR",
}


def _style_for_status(status: DecisionStatus) -> str:
    return STATUS_STYLE.get(status, "info")


def _build_metric_table(decision: Decision) -> Table:
    """Build the live execution-metrics table (intent / risk / conf / latency)."""
    table = Table(
        title="System 1 Decision Metrics",
        title_style="header",
        show_header=True,
        header_style="bold cyan",
        box=rich_box(),
        expand=False,
        padding=(0, 1),
    )
    table.add_column("Field", style="bold white", no_wrap=True)
    table.add_column("Value", overflow="fold")

    intent_style = {
        Intent.INFRASTRUCTURE_METRICS: "info",
        Intent.STAGING_DEPLOYMENT: "metric",
        Intent.DATABASE_QUERY: "metric",
        Intent.DESTRUCTIVE_OPS: "blocked",
        Intent.UNKNOWN: "muted",
    }.get(decision.intent, "info")

    table.add_row("Detected Intent", Text(decision.intent.value, style=intent_style))
    table.add_row(
        "Risk Classification",
        Text(decision.risk_level.value, style=RISK_STYLE.get(decision.risk_level, "info")),
    )
    table.add_row(
        "Calibrated Confidence",
        Text(f"{decision.confidence * 100:.1f}%", style="metric"),
    )
    table.add_row(
        "Processing Latency",
        Text(f"{decision.latency_ms:.1f} ms", style="metric"),
    )
    table.add_row("Inference Provider", Text(decision.provider, style="muted"))
    table.add_row("Model", Text(decision.model, style="muted"))
    return table


def rich_box():
    """Return the box style used for panels and tables."""
    from rich import box

    return box.ROUNDED


# ---------------------------------------------------------------------------
# System 2: tool routing
# ---------------------------------------------------------------------------
def route_tool(intent: Intent, raw_command: str) -> tuple[Optional[str], Optional[Any]]:
    """Map an APPROVED intent onto the appropriate simulated tool.

    Returns ``(tool_name, tool_result)``. ``tool_result`` is ``None`` when
    the intent has no tool binding (e.g. ``UNKNOWN``).

    This function is ONLY ever called with an APPROVED verdict.
    """
    command = (raw_command or "").strip()

    if intent is Intent.INFRASTRUCTURE_METRICS:
        return "get_cluster_metrics", get_cluster_metrics()

    if intent is Intent.DATABASE_QUERY:
        return "query_analytics", query_analytics()

    if intent is Intent.STAGING_DEPLOYMENT:
        service = _extract_service_name(command)
        return "deploy_staging", deploy_staging(service)

    return None, None


def _extract_service_name(command: str) -> str:
    """Extract a plausible service name from a natural-language deploy command.

    Uses simple linguistic heuristics only for *labelling* the mock
    deployment - never for the safety decision.
    """
    lowered = command.lower()
    candidate = command
    for prefix in ("deploy ", "release ", "ship ", "roll out ", "rollout of "):
        if lowered.startswith(prefix):
            candidate = command[len(prefix) :].strip()
            break

    # Strip common trailing qualifiers: "to staging", "into staging", etc.
    for suffix in (
        " to staging",
        " into staging",
        " to the staging env",
        " to staging environment",
    ):
        if candidate.lower().endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break

    candidate = candidate.strip().strip("\"'").strip()

    # Strip leading determiners / possessives left over from natural speech.
    for lead in ("the ", "our ", "my ", "that ", "this "):
        if candidate.lower().startswith(lead):
            candidate = candidate[len(lead) :].strip()
            break

    # Drop a generic trailing noun so "auth service" becomes "auth".
    for noun in (" service", " application", " microservice"):
        if candidate.lower().endswith(noun):
            candidate = candidate[: -len(noun)].strip()
            break

    return candidate or "jarvis-core"


def _load_style(value: float) -> str:
    """Colour a utilisation percentage by threshold."""
    if value >= 85:
        return "blocked"
    if value >= 70:
        return "warning"
    return "approved"


def _render_cluster_metrics(data: dict[str, Any]) -> Any:
    table = Table(
        title="Cluster Metrics (simulated)",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        expand=False,
    )
    table.add_column("Metric", style="bold white", no_wrap=True)
    table.add_column("Value", overflow="fold")

    table.add_row("Cluster", str(data.get("cluster", "-")))
    table.add_row("Version", str(data.get("version", "-")))
    table.add_row(
        "CPU Usage",
        Text(f"{data.get('cpu_usage_pct', 0)}%", style=_load_style(data.get("cpu_usage_pct", 0))),
    )
    table.add_row(
        "Memory Usage",
        Text(f"{data.get('memory_usage_pct', 0)}%", style=_load_style(data.get("memory_usage_pct", 0))),
    )
    table.add_row(
        "Disk Usage",
        Text(f"{data.get('disk_usage_pct', 0)}%", style=_load_style(data.get("disk_usage_pct", 0))),
    )
    table.add_row("Uptime", f"{data.get('uptime_hours', 0)} hours")
    table.add_row(
        "Node Health",
        f"{data.get('nodes_healthy', 0)}/{data.get('nodes_total', 0)} healthy"
        f" ({data.get('nodes_degraded', 0)} degraded)",
    )

    node_table = Table(
        title="Active Nodes",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        show_edge=False,
    )
    node_table.add_column("Node", style="bold white")
    node_table.add_column("Role")
    node_table.add_column("Status")
    for node in data.get("nodes", []):
        status = str(node.get("status", ""))
        style = "approved" if status == "Healthy" else "warning"
        node_table.add_row(
            str(node.get("node", "-")), str(node.get("role", "-")), Text(status, style=style)
        )

    return Group(table, node_table)


def _render_deploy(data: dict[str, Any]) -> Any:
    table = Table(
        title=f"Staging Deployment - {data.get('service_name', '-')}",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        expand=False,
    )
    table.add_column("Field", style="bold white", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Build ID", str(data.get("build_id", "-")))
    table.add_row("Commit Hash", str(data.get("commit_hash", "-")))
    table.add_row("Environment", str(data.get("environment", "-")))
    table.add_row("Status", Text(str(data.get("status", "-")), style="approved"))
    table.add_row(
        "Replicas Ready",
        f"{data.get('replicas_ready', 0)}/{data.get('replicas_desired', 0)}",
    )
    table.add_row("Rolled Out At", str(data.get("rolled_out_at", "-")))

    pipeline = Table(
        title="CI/CD Pipeline",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        show_edge=False,
    )
    pipeline.add_column("Stage", style="bold white")
    pipeline.add_column("Result")
    pipeline.add_column("Duration", justify="right")
    for stage in data.get("pipeline", []):
        result = str(stage.get("result", ""))
        style = "approved" if result in ("PASS", "SUCCESS") else "warning"
        pipeline.add_row(
            str(stage.get("stage", "-")),
            Text(result, style=style),
            f"{stage.get('duration_s', 0)}s",
        )

    return Group(table, pipeline)


def _render_analytics(data: dict[str, Any]) -> Any:
    table = Table(
        title="Analytics Query Summary",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        expand=False,
    )
    table.add_column("Field", style="bold white", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Warehouse", str(data.get("warehouse", "-")))
    table.add_row("Mode", Text(str(data.get("mode", "-")), style="approved"))
    table.add_row("Rows Returned", str(data.get("rows_returned", 0)))
    table.add_row("Bytes Scanned", f"{int(data.get('scan_bytes', 0)):,}")
    table.add_row("Duration", f"{data.get('duration_ms', 0)} ms")

    results = Table(
        title="Daily Event Volume",
        title_style="header",
        header_style="bold cyan",
        box=rich_box(),
        show_edge=False,
    )
    results.add_column("Day", style="bold white")
    results.add_column("Events", justify="right")
    for row in data.get("results", []):
        results.add_row(str(row.get("day", "-")), f"{int(row.get('events', 0)):,}")

    return Group(table, results)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class SecureJarvisApp:
    """Orchestrates System 1 -> System 2 -> System 3 for a single session."""

    def __init__(self, engine: Optional[AIDecisionEngine] = None) -> None:
        self.engine = engine
        self.db_path = init_db()

    # -- setup -------------------------------------------------------------
    @staticmethod
    def run_setup() -> int:
        """Interactively write the inference API key into the local ``.env``."""
        console.print(
            Panel(
                Text(
                    "Secure Jarvis - Inference Credential Setup\n\n"
                    "Your key is written to a local .env file (gitignored) and is\n"
                    "loaded at runtime via python-dotenv. It is never committed.",
                    style="info",
                ),
                title="Setup",
                border_style="info",
                box=rich_box(),
            )
        )
        provider = Prompt.ask(
            "Choose provider",
            choices=["groq", "openrouter", "gemini"],
            default="groq",
        )
        key = Prompt.ask(f"Enter your {provider.upper()} API key", password=False)
        key = (key or "").strip()
        if not key:
            console.print("[blocked]No key entered. Aborting setup.[/blocked]")
            return 1

        env_path = BASE_DIR / ".env"
        var_name = {
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(provider, "GROQ_API_KEY")
        _write_env_value(env_path, var_name, key)
        console.print(f"[approved]Credential saved to {env_path} ({var_name}).[/approved]")
        console.print(
            "[muted]Note: if you already exported the variable in your shell, "
            "that value takes precedence.[/muted]"
        )
        return 0

    # -- command processing ------------------------------------------------
    def process_command(self, raw_command: str, auto_approve: bool = False) -> Decision:
        """Run the full System 1 -> System 2 -> System 3 pipeline."""
        console.print()
        console.rule(Text("SYSTEM 1 :: DECISION GATE", style="info"), style="cyan")

        if self.engine is None or not self.engine.is_ready:
            decision = Decision(
                intent=Intent.UNKNOWN,
                risk_level=RiskLevel.CRITICAL,
                confidence=0.0,
                reasoning=(
                    "No inference credential configured - fail-closed. "
                    "Run `python secure_jarvis.py --setup` or set GROQ_API_KEY "
                    "in your .env file."
                ),
                guardrail_category=GuardrailCategory.OTHER_DESTRUCTIVE,
                action=DecisionStatus.ERROR,
                latency_ms=0.0,
            )
        else:
            decision = self.engine.evaluate_safe(raw_command)

        self._render_verdict(decision, raw_command)

        # -- System 3: persist the verdict BEFORE any execution -------------
        log_event(
            raw_command=raw_command,
            intent=decision.intent.value,
            risk_level=decision.risk_level.value,
            confidence=decision.confidence,
            status=decision.action.value,
            latency_ms=decision.latency_ms,
            reasoning=decision.reasoning,
        )

        if decision.is_blocked:
            self._record_violation(decision, raw_command)
            return decision

        if decision.action is DecisionStatus.REQUIRES_APPROVAL:
            approved = self._request_human_approval(decision, raw_command, auto_approve)
            if not approved:
                console.print(
                    "[warning]Human approval withheld - command terminated without execution.[/warning]"
                )
                return decision
            decision = decision.model_copy(update={"action": DecisionStatus.APPROVED})

        # -- System 2: execute only on APPROVED ----------------------------
        console.print()
        console.rule(Text("SYSTEM 2 :: OPERATIONS EXECUTION", style="approved"), style="green")
        self._execute_tool(decision, raw_command)
        return decision

    def _record_violation(self, decision: Decision, raw_command: str) -> None:
        """Write a security alert for a blocked command."""
        log_security_alert(
            raw_command=raw_command,
            alert_type=decision.guardrail_category.value,
            severity="CRITICAL" if decision.risk_level is RiskLevel.CRITICAL else "HIGH",
            intent=decision.intent.value,
            confidence=decision.confidence,
            guardrail_rule=(
                "DESTRUCTIVE_OPERATION_BLOCKED"
                if decision.guardrail_category is not GuardrailCategory.NONE
                else "RISK_POLICY_BLOCKED"
            ),
            reasoning=decision.reasoning,
            disposition="BLOCKED",
        )

    def _request_human_approval(
        self, decision: Decision, raw_command: str, auto_approve: bool
    ) -> bool:
        """Prompt the operator for explicit approval of a medium-risk action."""
        if auto_approve:
            return True
        console.print(
            Panel(
                Text(
                    f"This operation was classified as {decision.risk_level.value} "
                    "and requires explicit human approval before execution.",
                    style="warning",
                ),
                title="Human Approval Required",
                border_style="yellow",
                box=rich_box(),
            )
        )
        try:
            return Confirm.ask("Approve execution?", default=False)
        except (EOFError, KeyboardInterrupt):
            return False


    # -- rendering ---------------------------------------------------------
    def _render_verdict(self, decision: Decision, raw_command: str) -> None:
        """Render the bordered security audit result panel + metrics."""
        style = _style_for_status(decision.action)
        badge = Text(decision.action.value, style=f"bold {style}")

        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="right")
        header.add_row(Text("SECURITY AUDIT RESULT", style="bold white"), badge)

        body = Group(
            header,
            Text(""),
            Text(f"Command:  {raw_command}", style="bold white"),
            Text(f"Intent:   {decision.intent.value}", style="info"),
            Text(
                f"Risk:     {decision.risk_level.value}",
                style=RISK_STYLE.get(decision.risk_level, "info"),
            ),
            Text(f"Confidence: {decision.confidence * 100:.1f}%", style="metric"),
            Text(f"Latency:  {decision.latency_ms:.1f} ms", style="metric"),
            Text(""),
            Text(f"Reasoning: {decision.reasoning}", style="muted"),
        )

        console.print(
            Panel(
                body,
                title="[bold]SYSTEM 1 :: VERDICT[/bold]",
                border_style=style,
                box=rich_box(),
                padding=(1, 2),
            )
        )

        console.print(_build_metric_table(decision))

        if decision.is_blocked:
            self._render_violation_box(decision, raw_command)
        elif decision.action is DecisionStatus.REQUIRES_APPROVAL:
            console.print(
                Panel(
                    Text(
                        "Medium-risk operation: routed to a human approver. "
                        "System 2 execution is gated on explicit approval.",
                        style="warning",
                    ),
                    border_style="yellow",
                    box=rich_box(),
                )
            )

    def _render_violation_box(self, decision: Decision, raw_command: str) -> None:
        """Render the enterprise compliance violation box for blocked commands."""
        label = GUARDRAIL_LABELS.get(decision.guardrail_category, "Policy Violation")
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold white", justify="right")
        details.add_column(style="white")
        details.add_row("Alert Type", Text(label, style="bold red"))
        details.add_row("Severity", Text("CRITICAL", style="bold red"))
        details.add_row(
            "Policy Rule",
            Text(
                "Enterprise Agent Safety Policy v1.0 :: DESTRUCTIVE_OPERATION_PROHIBITED",
                style="red",
            ),
        )
        details.add_row("Disposition", Text("BLOCKED - NO TOOL INVOKED", style="bold red"))
        details.add_row("Audit Log", Text(str(self.db_path), style="muted"))

        console.print(
            Panel(
                Group(
                    Text("SECURITY VIOLATION DETECTED", style="bold red"),
                    Text(""),
                    details,
                ),
                title="[bold red]ENTERPRISE COMPLIANCE VIOLATION[/bold red]",
                border_style="red",
                box=rich_box(),
                padding=(1, 2),
            )
        )

    def _execute_tool(self, decision: Decision, raw_command: str) -> None:
        """Invoke the routed mock tool and render its output."""
        tool_name, result = route_tool(decision.intent, raw_command)
        if tool_name is None or result is None:
            console.print(
                Panel(
                    Text(
                        "The command was approved but no tool binding exists for "
                        f"intent {decision.intent.value}.",
                        style="warning",
                    ),
                    border_style="yellow",
                    box=rich_box(),
                )
            )
            return

        renderers = {
            "get_cluster_metrics": _render_cluster_metrics,
            "deploy_staging": _render_deploy,
            "query_analytics": _render_analytics,
        }
        renderer = renderers.get(tool_name)
        console.print(
            Panel(
                Text(f"Executed simulated tool: {tool_name}", style="approved"),
                border_style="green",
                box=rich_box(),
            )
        )
        if renderer is not None:
            console.print(renderer(result))


    # -- audit viewer ------------------------------------------------------
    def render_audit(self, limit: int = 20) -> None:
        """Render the persisted audit trail in a Rich table."""
        logs = fetch_recent_logs(limit=limit)
        alerts = fetch_security_alerts(limit=limit)
        stats = fetch_stats()

        console.print(
            Panel(
                Text(
                    "System 3 Observability - persisted decision records\n"
                    f"Database: {self.db_path}",
                    style="info",
                ),
                title="[bold]AUDIT TRAIL[/bold]",
                border_style="cyan",
                box=rich_box(),
            )
        )

        summary = Table.grid(padding=(0, 3))
        summary.add_column()
        summary.add_column()
        summary.add_column()
        summary.add_column()
        summary.add_row(
            Text(f"Total: {stats.get('total', 0)}", style="bold white"),
            Text(f"Approved: {stats.get('approved', 0)}", style="approved"),
            Text(f"Blocked: {stats.get('blocked', 0)}", style="blocked"),
            Text(f"Alerts: {stats.get('alerts', 0)}", style="warning"),
        )
        console.print(summary)
        console.print()

        if not logs:
            console.print(
                "[muted]No audit records yet. Run a command to populate the trail.[/muted]"
            )
            return

        # Compute column widths from the live console width so the table always
        # fits the terminal (no overflow, no awkward mid-word wrapping).
        available = max(60, console.width)
        n_cols = 7
        overhead = 3 * n_cols + 1  # borders + per-column padding
        content = available - overhead
        fixed = 2 + 11 + 8 + 3 + 8  # #, Time, Risk, Cnf, Status
        flex = max(10, content - fixed)
        cmd_w = max(10, int(flex * 0.5))
        intent_w = max(9, flex - cmd_w)

        table = Table(
            title="Recent Command Verdicts",
            title_style="header",
            header_style="bold cyan",
            box=rich_box(),
            show_lines=True,
            expand=False,
            pad_edge=False,
        )
        table.add_column("#", justify="right", style="muted", no_wrap=True, width=2)
        table.add_column("Time", style="muted", no_wrap=True, width=11)
        table.add_column("Command", overflow="fold", no_wrap=False, width=cmd_w)
        table.add_column("Intent", overflow="fold", no_wrap=False, width=intent_w)
        table.add_column("Risk", overflow="fold", no_wrap=True, width=8)
        table.add_column("Cnf", justify="right", no_wrap=True, width=3)
        table.add_column("Status", overflow="fold", no_wrap=True, width=8)

        for row in logs:
            status = str(row.get("status", ""))
            style = {
                "APPROVED": "approved",
                "BLOCKED": "blocked",
                "REQUIRES_APPROVAL": "warning",
                "ERROR": "blocked",
            }.get(status, "info")
            risk = str(row.get("risk_level", ""))
            risk_style = {
                "SAFE": "approved",
                "REQUIRES_APPROVAL": "warning",
                "CRITICAL": "blocked",
            }.get(risk, "info")
            table.add_row(
                str(row.get("id", "")),
                _short_timestamp(row.get("timestamp", "")),
                str(row.get("raw_command", "")),
                INTENT_SHORT.get(str(row.get("intent", "")), str(row.get("intent", ""))),
                Text(RISK_SHORT.get(risk, risk), style=risk_style),
                f"{float(row.get('confidence', 0)) * 100:.0f}%",
                Text(STATUS_SHORT.get(status, status), style=style),
            )
        console.print(table)

        if alerts:
            console.print()
            alert_available = max(60, console.width)
            a_fixed = 11 + 8 + 8  # Time, Severity, Disposition
            a_flex = max(10, alert_available - (3 * 5 + 1) - a_fixed)
            a_type_w = max(9, int(a_flex * 0.4))
            a_cmd_w = max(10, a_flex - a_type_w)

            alert_table = Table(
                title="Security Alerts (Compliance Violations)",
                title_style="header",
                header_style="bold red",
                box=rich_box(),
                expand=False,
                pad_edge=False,
            )
            alert_table.add_column("Time", style="muted", no_wrap=True, width=11)
            alert_table.add_column("Alert Type", overflow="fold", width=a_type_w)
            alert_table.add_column("Severity", overflow="fold", no_wrap=True, width=8)
            alert_table.add_column("Command", overflow="fold", width=a_cmd_w)
            alert_table.add_column("Action", overflow="fold", no_wrap=True, width=8)
            for alert in alerts:
                alert_table.add_row(
                    _short_timestamp(alert.get("timestamp", "")),
                    Text(str(alert.get("alert_type", "")), style="blocked"),
                    Text(str(alert.get("severity", "")), style="blocked"),
                    str(alert.get("raw_command", "")),
                    Text(str(alert.get("disposition", "")), style="blocked"),
                )
            console.print(alert_table)


    # -- interactive REPL --------------------------------------------------
    def run_repl(self) -> int:
        """Continuous interactive shell until the operator types 'exit'."""
        self._print_banner()

        while True:
            try:
                command = Prompt.ask(
                    "[bold cyan]jarvis[/bold cyan] [muted]>[/muted]",
                    default="",
                )
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print("[muted]Session ended.[/muted]")
                return 0

            command = (command or "").strip()
            if not command:
                continue
            if command.lower() in ("exit", "quit", "q", ":q"):
                console.print("[muted]Session ended. Audit trail preserved.[/muted]")
                return 0
            if command.lower() in ("help", "?"):
                self._print_help()
                continue
            if command.lower() == "audit":
                self.render_audit()
                continue

            try:
                self.process_command(command)
            except Exception as exc:  # noqa: BLE001 - REPL must never crash
                console.print(f"[blocked]Unhandled error: {exc}[/blocked]")

    def _print_banner(self) -> None:
        provider_info = (
            self.engine.describe() if self.engine is not None else "unconfigured (no API key)"
        )
        banner = Text()
        banner.append("SECURE JARVIS\n", style="bold cyan")
        banner.append("Enterprise Agent Governance Framework\n", style="bold white")
        banner.append(
            "System 1 semantic decision gate  >  System 2 gated execution  >  "
            "System 3 audit trail\n",
            style="muted",
        )
        banner.append(f"Inference provider: {provider_info}\n", style="info")
        banner.append(
            "Type 'help' for commands, 'audit' for the audit trail, 'exit' to quit.",
            style="muted",
        )
        console.print(
            Panel(
                banner,
                border_style="cyan",
                box=rich_box(),
                padding=(1, 2),
            )
        )

    def _print_help(self) -> None:
        table = Table(
            title="Interactive Commands",
            title_style="header",
            header_style="bold cyan",
            box=rich_box(),
            show_edge=False,
        )
        table.add_column("Command", style="bold white")
        table.add_column("Description")
        table.add_row("help", "Show this help panel.")
        table.add_row("audit", "Render the persisted System 3 audit trail.")
        table.add_row("exit", "Terminate the session (audit records are preserved).")
        table.add_row("<any command>", "Evaluated by the System 1 semantic decision gate.")
        console.print(table)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _write_env_value(env_path: Path, key: str, value: str) -> None:
    """Write or replace ``key=value`` inside the ``.env`` file, preserving others."""
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                output.append(f"{key}={value}")
                replaced = True
                continue
        output.append(line)

    if not replaced:
        output.append(f"{key}={value}")

    env_path.write_text("\n".join(output).strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secure_jarvis",
        description=(
            "Secure Jarvis - Enterprise agent governance with a semantic "
            "System 1 decision gate, gated System 2 execution and a "
            "System 3 SQLite audit trail."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python secure_jarvis.py "show cluster cpu and memory metrics"\n'
            '  python secure_jarvis.py "deploy the payments-api service to staging"\n'
            '  python secure_jarvis.py "clean up the old staging records"\n'
            "  python secure_jarvis.py --audit\n"
            "  python secure_jarvis.py --setup\n"
        ),
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to evaluate. Omit to start the interactive REPL.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Render the persisted audit trail and exit.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Interactively configure the inference API key in .env and exit.",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openrouter", "gemini"],
        default=None,
        help="Force a specific inference provider (default: auto-detect).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model identifier used for inference.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip the interactive human-approval prompt for medium-risk commands.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of audit records to display in --audit mode (default: 20).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Secure Jarvis 1.0.0 :: Enterprise Agent Governance Framework",
    )
    return parser


def _build_engine(provider: Optional[str], model: Optional[str]) -> Optional[AIDecisionEngine]:
    """Instantiate the System 1 engine, or ``None`` if no credential exists."""
    try:
        return AIDecisionEngine(provider=provider, model=model)
    except DecisionEngineError:
        return None
    except Exception:  # noqa: BLE001 - never block CLI start-up on engine issues
        return None


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- setup mode --------------------------------------------------------
    if args.setup:
        return SecureJarvisApp.run_setup()

    # -- audit viewer mode -------------------------------------------------
    if args.audit:
        app = SecureJarvisApp()
        app.render_audit(limit=args.limit)
        return 0

    # -- argument mode -----------------------------------------------------
    if args.command:
        engine = _build_engine(args.provider, args.model)
        if engine is None:
            console.print(
                Panel(
                    Text(
                        "No inference credential found. Run `python secure_jarvis.py --setup` "
                        "or set GROQ_API_KEY / GEMINI_API_KEY in your .env file.\n"
                        "The system will fail closed: the command will be BLOCKED and logged.",
                        style="blocked",
                    ),
                    title="[bold red]CREDENTIAL REQUIRED[/bold red]",
                    border_style="red",
                    box=rich_box(),
                )
            )
        app = SecureJarvisApp(engine=engine)
        decision = app.process_command(" ".join(args.command), auto_approve=args.auto_approve)
        return 0 if decision.action is DecisionStatus.APPROVED else 1

    # -- interactive REPL mode ---------------------------------------------
    engine = _build_engine(args.provider, args.model)
    app = SecureJarvisApp(engine=engine)
    return app.run_repl()


if __name__ == "__main__":
    sys.exit(main())
