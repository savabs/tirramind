"""
Tool: Pipeline Query (Agent ↔ Pipeline Bridge)

Lets the LLM-driven Agent Layer query the Pipeline Store's accumulated
data, signals, and run history without making network calls.

Three modes:
    data    — query stored tool outputs (pipeline_data table)
    signals — query computed signal values (signals table)
    runs    — query DAG execution history (dag_runs table)

All queries hit local SQLite — fast, free, no rate limits.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from agent.pipeline.store import PipelineStore
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# Seconds per unit for since/until relative time parsing
_UNITS = {"h": 3600, "d": 86400, "w": 604800}


def _parse_relative_time(value: str) -> float | None:
    """Parse relative time like '7d', '24h', '2w' into epoch seconds.

    Returns epoch timestamp (now - duration), or None if not a relative format.
    """
    if not value:
        return None
    value = value.strip().lower()
    if len(value) >= 2 and value[-1] in _UNITS and value[:-1].replace(".", "").isdigit():
        seconds = float(value[:-1]) * _UNITS[value[-1]]
        return time.time() - seconds
    return None


def _format_ts(epoch: float | None) -> str:
    """Format epoch seconds to human-readable UTC string."""
    if epoch is None:
        return "—"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class PipelineQueryTool(Tool):
    """Query the Pipeline Store for accumulated data, signals, and run history."""

    def __init__(self, store: PipelineStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "pipeline_query"

    @property
    def description(self) -> str:
        return (
            "Query the pipeline's local database for stored tool outputs, "
            "computed signals, or DAG run history. "
            "Mode 'data': query stored data by source (e.g. 'cftc', 'power_grid'). "
            "Mode 'signals': query computed signal values by name. "
            "Mode 'runs': query DAG execution history. "
            "Supports relative time filters like '7d', '24h', '2w'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["data", "signals", "runs"],
                    "description": "Query type: 'data' for tool outputs, 'signals' for computed values, 'runs' for execution history",
                },
                "source": {
                    "type": "string",
                    "description": "Data source name (e.g. 'cftc', 'fetch_gdelt'). Required for mode='data'.",
                },
                "signal_name": {
                    "type": "string",
                    "description": "Signal name to query. Required for mode='signals'.",
                },
                "dag_name": {
                    "type": "string",
                    "description": "Filter runs by DAG name. Optional for mode='runs'.",
                },
                "since": {
                    "type": "string",
                    "description": "Relative time filter: '24h', '7d', '2w'. Only data older than this is excluded.",
                },
                "until": {
                    "type": "string",
                    "description": "Relative time filter for upper bound: '1h', '1d'. Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 20, max 500).",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in ("data", "signals", "runs"):
            return ToolResult(False, f"Invalid mode '{mode}'. Use 'data', 'signals', or 'runs'.")

        limit = max(1, min(500, int(kwargs.get("limit", 20))))
        since = _parse_relative_time(kwargs.get("since", ""))
        until = _parse_relative_time(kwargs.get("until", ""))

        if mode == "data":
            return self._query_data(kwargs, since, until, limit)
        elif mode == "signals":
            return self._query_signals(kwargs, since, until, limit)
        else:
            return self._query_runs(kwargs, limit)

    def _query_data(
        self, kwargs: dict, since: float | None, until: float | None, limit: int
    ) -> ToolResult:
        source = (kwargs.get("source") or "").strip()
        if not source:
            return ToolResult(False, "mode='data' requires 'source' parameter (e.g. 'cftc', 'fetch_gdelt').")

        rows = self._store.query_data(source=source, since=since, until=until, limit=limit)
        if not rows:
            return ToolResult(
                True,
                f"No data found for source '{source}'" + (f" (since {_format_ts(since)})" if since else "") + ".",
                data={"source": source, "count": 0, "rows": []},
            )

        # Build human-readable summary
        lines = [f"Pipeline data for '{source}': {len(rows)} row(s)"]
        for r in rows[:5]:  # Show first 5 in text
            ts = _format_ts(r.get("fetched_at"))
            data_preview = json.dumps(r.get("data", {}), default=str)
            if len(data_preview) > 200:
                data_preview = data_preview[:200] + "…"
            lines.append(f"  [{ts}] {data_preview}")
        if len(rows) > 5:
            lines.append(f"  ... and {len(rows) - 5} more rows")

        return ToolResult(
            True,
            "\n".join(lines),
            data={"source": source, "count": len(rows), "rows": rows},
        )

    def _query_signals(
        self, kwargs: dict, since: float | None, until: float | None, limit: int
    ) -> ToolResult:
        signal_name = (kwargs.get("signal_name") or "").strip()
        if not signal_name:
            return ToolResult(False, "mode='signals' requires 'signal_name' parameter.")

        rows = self._store.query_signals(signal_name=signal_name, since=since, until=until, limit=limit)
        if not rows:
            return ToolResult(
                True,
                f"No signals found for '{signal_name}'" + (f" (since {_format_ts(since)})" if since else "") + ".",
                data={"signal_name": signal_name, "count": 0, "rows": []},
            )

        lines = [f"Signal '{signal_name}': {len(rows)} value(s)"]
        for r in rows[:10]:
            ts = _format_ts(r.get("computed_at"))
            lines.append(f"  [{ts}] {r.get('value', '?')}")

        return ToolResult(
            True,
            "\n".join(lines),
            data={"signal_name": signal_name, "count": len(rows), "rows": rows},
        )

    def _query_runs(self, kwargs: dict, limit: int) -> ToolResult:
        dag_name = (kwargs.get("dag_name") or "").strip() or None
        rows = self._store.get_runs(dag_name=dag_name, limit=limit)

        if not rows:
            label = f"for DAG '{dag_name}'" if dag_name else ""
            return ToolResult(
                True,
                f"No pipeline runs found {label}.".strip(),
                data={"count": 0, "runs": []},
            )

        lines = [f"Pipeline runs: {len(rows)}"]
        for r in rows[:10]:
            ts = _format_ts(r.get("started_at"))
            status = r.get("status", "?")
            name = r.get("dag_name", "?")
            run_id = r.get("run_id", "?")[:12]
            lines.append(f"  [{ts}] {name} — {status} ({run_id}…)")

        return ToolResult(
            True,
            "\n".join(lines),
            data={"count": len(rows), "runs": rows},
        )
