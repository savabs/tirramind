#!/usr/bin/env python3
"""Run the daily_collection DAG manually.

Usage:
    python scripts/run_collection.py [--db-path PATH] [--workers N]

Builds a ToolRegistry with PipelineStore wired, loads the daily_collection
DAG, and executes it. Prints per-node status and timing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel

from agent.cli import build_tool_registry
from agent.config.settings import AgentConfig
from agent.pipeline.executor import DAGExecutor
from agent.pipeline.registry import DAGRegistry
from agent.pipeline.store import PipelineStore

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily_collection DAG")
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max parallel workers (default: 4)",
    )
    parser.add_argument(
        "--dag",
        default="daily_collection",
        help="DAG name to run (default: daily_collection)",
    )
    args = parser.parse_args()

    # Build config with specified db_path
    import os

    # Load .env so tools that need API keys (FRED, etc.) pick them up.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    os.environ["TIRRA_PIPELINE_DB"] = args.db_path
    os.environ["TIRRA_PIPELINE_WORKERS"] = str(args.workers)
    config = AgentConfig.from_env()

    console.print(
        Panel(
            f"[bold cyan]Running DAG:[/] {args.dag}\n" f"[dim]DB: {args.db_path} | Workers: {args.workers}[/]",
            title="TirraMind Pipeline",
            border_style="blue",
        )
    )

    # Build registry with PipelineStore wired to all tools
    registry = build_tool_registry(config)

    # Load DAGs
    dag_registry = DAGRegistry()
    dag_registry.load_defaults(registry)
    dag = dag_registry.get(args.dag)
    if dag is None:
        available = ", ".join(d.name for d in dag_registry.list_all())
        console.print(f"[red]DAG not found: {args.dag}[/]")
        console.print(f"[dim]Available: {available}[/]")
        sys.exit(1)

    console.print(f"[dim]Nodes: {', '.join(dag.nodes.keys())}[/]")

    # Execute
    store = PipelineStore(db_path=args.db_path)
    executor = DAGExecutor(
        tool_registry=registry,
        store=store,
        max_workers=args.workers,
    )

    t0 = time.monotonic()
    result = executor.execute(dag, trigger="manual")
    elapsed = time.monotonic() - t0

    # Print results
    console.print()
    status_color = "green" if result.status in ("success", "completed") else "red"
    console.print(f"[{status_color}]Run {result.run_id}: {result.status}[/]")

    for nid, nr in result.node_results.items():
        ok = nr.status in ("success", "completed")
        icon = "[green]✓[/]" if ok else "[red]✗[/]"
        duration = ""
        if hasattr(nr, "duration_s") and nr.duration_s is not None:
            duration = f" ({nr.duration_s:.1f}s)"
        console.print(f"  {icon} {nid}: {nr.status}{duration}")
        if not ok and nr.error:
            # NodeResult.error is the actual failure reason. The executor only
            # logs a warning on non-final retry attempts (see
            # DAGExecutor._execute_node) — the final attempt's exception is
            # captured on the result but was never surfaced here, so a failed
            # node with retries=1 (the default) used to print with no
            # diagnostic at all.
            console.print(f"      [dim red]{nr.error}[/]")

    console.print(f"\n[dim]Total elapsed: {elapsed:.1f}s[/]")

    # Quick DB stats
    try:
        import sqlite3

        conn = sqlite3.connect(args.db_path)
        cur = conn.cursor()
        entity_count = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        obs_count = cur.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
        link_count = cur.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
        conn.close()
        console.print(f"[dim]DB stats: {entity_count} entities, " f"{obs_count} observations, {link_count} links[/]")
    except Exception:
        pass

    sys.exit(0 if result.status in ("success", "completed") else 1)


if __name__ == "__main__":
    main()
