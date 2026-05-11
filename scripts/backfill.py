#!/usr/bin/env python3
"""Phase 47 Historical Backfill Runner.

Calls every relevant tool with a historical date range to populate
entity_observations over 5 years. Idempotent via checkpoint file.

Usage:
    python scripts/backfill.py [--dry-run] [--tool LABEL] [--days-back N]
    python scripts/backfill.py --verify          # probe Group B tools first
    python scripts/backfill.py --skip-group-b    # Group A only
    python scripts/backfill.py --group-b-only    # Group B only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Allow running from repo root: python scripts/backfill.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before importing config so os.getenv picks up keys
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env

from agent.cli import build_tool_registry
from agent.config.settings import AgentConfig, PipelineConfig
from agent.tools.base import ToolRegistry

log = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = ".tirra_pipeline/pipeline.db"
_DEFAULT_DAYS = 1825  # 5 years
_CHECKPOINT_DIR = ".tirra_pipeline"
_CHECKPOINT_FILE = ".tirra_pipeline/backfill_checkpoint.json"
_RATE_LIMIT_SLEEP = 1.5  # seconds between tool invocations
_HTTP_429_SLEEP = 60.0
_DB_RETRY_SLEEP = 5.0

_TODAY = datetime.now(timezone.utc).date().isoformat()
_START_5Y = (
    (datetime.now(timezone.utc) - timedelta(days=_DEFAULT_DAYS)).date().isoformat()
)

# Macro series to backfill from FRED
_MACRO_SERIES = [
    "GDP",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "T10Y2Y",
    "DTWEXBGS",
    "M2SL",
    "DGS10",
    "DGS2",
    "VIXCLS",
    "BAMLH0A0HYM2",
]

# ---------------------------------------------------------------------------
# BACKFILL_PLAN — single source of truth for all tool calls
#
# Fields:
#   label     : unique checkpoint key
#   tool      : ToolRegistry name
#   kwargs    : passed directly to registry.execute(tool, **kwargs)
#   group     : "A" (default), "B" (verify first), or "C" (skip — metadata only)
#   skip      : if True, entry is skipped silently
#   reason    : why entry is skipped
#   chunk     : dict with 'mode' key controlling how kwargs are expanded
#               "repeat": call the same kwargs N times (e.g. rolling windows)
# ---------------------------------------------------------------------------


def _build_plan(days_back: int) -> list[dict[str, Any]]:
    """Build the complete backfill plan. `days_back` overrides the default 1825."""
    start_date = (
        (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    )

    plan: list[dict[str, Any]] = []

    # ── Group A: run unconditionally ────────────────────────────────────────

    plan.append(
        {
            "label": "academic_preprints",
            "tool": "academic_preprints",
            "kwargs": {"mode": "trending"},
        }
    )
    # AIS vessel tracking: Baltic Sea focus (Russian oil / sanction evasion monitoring)
    plan.append(
        {
            "label": "ais_vessel",
            "tool": "ais_vessel_tracking",
            "kwargs": {"mode": "area", "area_name": "full_baltic"},
        }
    )
    plan.append(
        {
            "label": "bankruptcy_court",
            "tool": "bankruptcy_court",
            "kwargs": {"mode": "us_bankruptcy", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "capital_flows",
            "tool": "capital_flows",
            "kwargs": {"mode": "flows", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "central_bank_balance",
            "tool": "central_bank_balance",
            "kwargs": {"mode": "balance_sheets", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "comtrade",
            "tool": "comtrade",
            "kwargs": {"mode": "flows", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "consumer_sentiment",
            "tool": "consumer_sentiment",
            "kwargs": {"mode": "us_sentiment", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "creditor_filings",
            "tool": "creditor_filings",
            "kwargs": {"mode": "stress_scan", "days_back": days_back},
        }
    )

    # disease_surveillance: 4 modes — each produces different observations
    for mode in ("wastewater", "outbreaks", "eu_surveillance"):
        plan.append(
            {
                "label": f"disease_surveillance_{mode}",
                "tool": "disease_surveillance",
                "kwargs": {"mode": mode, "days_back": days_back, "_backfill": True},
            }
        )

    plan.append(
        {
            "label": "earthquake_proximity",
            "tool": "earthquake_proximity",
            "kwargs": {"days_back": min(days_back, 1825), "_backfill": True},
            # Note: no L2 persistence — fetches data but no entity_observations written
        }
    )
    # food_security: WLD = world aggregate; covers all countries in FAO data
    plan.append(
        {
            "label": "food_security",
            "tool": "food_security",
            "kwargs": {"mode": "production", "country": "WLD"},
        }
    )

    # form144: EDGAR EFTS supports arbitrary date range via _backfill bypass
    plan.append(
        {
            "label": "form144",
            "tool": "form144",
            "kwargs": {"days_back": days_back, "_backfill": True},
        }
    )

    plan.append(
        {
            "label": "gov_contracts",
            "tool": "gov_contracts",
            # gov_contracts uses start_date/end_date, not days_back.
            # Compute start_date from days_back so historical contracts are fetched.
            "kwargs": {
                "mode": "recent",
                "start_date": (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "end_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "limit": 50,
            },
        }
    )
    # Second gov_contracts pass: top mode to get high-value awards (different agencies)
    plan.append(
        {
            "label": "gov_contracts_top",
            "tool": "gov_contracts",
            "kwargs": {
                "mode": "top",
                "start_date": (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "end_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "limit": 50,
            },
        }
    )

    # insider_filings: EDGAR — code cap=90d; bypass for 5yr historical
    plan.append(
        {
            "label": "insider_filings",
            "tool": "insider_filings",
            "kwargs": {"days_back": days_back, "_backfill": True},
        }
    )

    plan.append(
        {
            "label": "job_postings",
            "tool": "job_postings",
            "kwargs": {"mode": "jolts", "days_back": min(days_back, 1095)},
        }
    )
    plan.append(
        {
            "label": "labor_disruptions",
            "tool": "labor_disruptions",
            "kwargs": {"mode": "work_stoppages", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "migration_flows",
            "tool": "migration_flows",
            "kwargs": {"mode": "displacement", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "satellite_activity",
            "tool": "satellite_activity",
            "kwargs": {"mode": "events", "days_back": min(days_back, 1095)},
        }
    )

    # sanctions_monitor: code cap=365d; bypass for 5yr historical
    plan.append(
        {
            "label": "sanctions_monitor",
            "tool": "sanctions_monitor",
            "kwargs": {"mode": "recent", "days_back": days_back, "_backfill": True},
        }
    )

    plan.append(
        {
            "label": "supply_chain_monitor",
            "tool": "supply_chain_prices",
            "kwargs": {"mode": "producer_prices", "days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "transport_throughput",
            "tool": "transport_throughput",
            "kwargs": {"days_back": days_back},
        }
    )
    plan.append(
        {
            "label": "building_permits",
            "tool": "building_permits",
            "kwargs": {"mode": "permits", "days_back": days_back},
        }
    )

    # macro_data: per-series calls; FRED supports 20yr history
    for sid in _MACRO_SERIES:
        plan.append(
            {
                "label": f"macro_{sid}",
                "tool": "macro_data",
                "kwargs": {
                    "source": "fred",
                    "series_id": sid,
                    "start_date": start_date,
                    "end_date": _TODAY,
                },
            }
        )

    # GDELT: weekly-sampled historical backfill (one 15-min batch per 7 days)
    plan.append(
        {
            "label": "gdelt_backfill",
            "tool": "gdelt",
            "kwargs": {
                "_backfill": True,
                "days_back": days_back,
                "sample_every_days": 7,
            },
        }
    )

    # CFTC: year-loop mode — each year is one API call.
    # Floor at 2011 (earliest confirmed CFTC disagg ZIP). With days_back=3650
    # this gives 2011 or max(2011, current_year - 10 - 1), whichever is later.
    for year in range(
        max(2011, datetime.now().year - (days_back // 365) - 1), datetime.now().year + 1
    ):
        plan.append(
            {
                "label": f"cftc_{year}",
                "tool": "cftc",
                "kwargs": {"mode": "historical", "year": year},
            }
        )

    # ── Group B: verify first, then run ─────────────────────────────────────

    plan.append(
        {
            "label": "defi_flows",
            "tool": "defi_flows",
            "kwargs": {"mode": "tvl", "days_back": min(days_back, 730)},
            "group": "B",
        }
    )
    plan.append(
        {
            # history mode fetches daily TVL time-series per protocol going back
            # up to 5 years. This is the primary fix for 'protocol' entity density.
            "label": "defi_flows_history",
            "tool": "defi_flows",
            "kwargs": {"mode": "history", "days_back": min(days_back, 1825), "limit": 50},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "drug_regulatory",
            "tool": "drug_regulatory",
            "kwargs": {"mode": "approvals", "days_back": days_back, "limit": 1000},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "electricity_monitor",
            "tool": "electricity_monitor",
            "kwargs": {
                "mode": "demand",
                "region": "PJM",
                "days_back": min(days_back, 730),
            },
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "energy_supply",
            "tool": "energy_supply",
            "kwargs": {"mode": "petroleum_stocks", "days_back": days_back},
            "group": "B",
        }
    )

    # finra_short_volume: FINRA files only available ~20 trading days — hard API limit
    plan.append(
        {
            "label": "finra_short_volume",
            "tool": "finra_short_volume",
            "kwargs": {"mode": "short_volume", "days_back": 20},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "foia_requests",
            "tool": "foia_requests",
            "kwargs": {
                "mode": "agency_activity",
                "agency": "SEC",
                "days_back": min(days_back, 1095),
            },
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "global_pmi",
            "tool": "global_pmi",
            "kwargs": {"mode": "cli", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "interconnection_queue",
            "tool": "interconnection_queue",
            "kwargs": {"mode": "summary", "days_back": min(days_back, 1095)},
            "group": "B",
        }
    )

    # internet_infrastructure: code cap=90d; _backfill=True for OONI historical
    plan.append(
        {
            "label": "internet_infrastructure",
            "tool": "internet_infrastructure",
            "kwargs": {"mode": "outages", "days_back": days_back, "_backfill": True},
            "group": "B",
            # Note: no L2 persistence — no entity_observations written
        }
    )

    plan.append(
        {
            "label": "lobbying",
            "tool": "lobbying",
            "kwargs": {"mode": "issues", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "patent_filings",
            "tool": "patent_filings",
            "kwargs": {"mode": "trends", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "political_risk",
            "tool": "political_risk",
            "kwargs": {"mode": "expenditures", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "polymarket",
            "tool": "polymarket",
            "kwargs": {"mode": "resolved", "days_back": min(days_back, 730)},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "polymarket_whales",
            "tool": "polymarket_whales",
            "kwargs": {"mode": "top_wallets", "days_back": min(days_back, 730)},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "power_grid",
            "tool": "power_grid",
            "kwargs": {"mode": "demand", "days_back": min(days_back, 730)},
            "group": "B",
        }
    )

    # regulatory_gazette: code cap=365d; _backfill=True bypasses
    plan.append(
        {
            "label": "regulatory_gazette",
            "tool": "regulatory_gazette",
            "kwargs": {"days_back": days_back, "_backfill": True},
            "group": "B",
        }
    )

    plan.append(
        {
            "label": "sovereign_debt",
            "tool": "sovereign_debt",
            "kwargs": {"mode": "us_yields", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "treasury_receipts",
            "tool": "treasury_receipts",
            "kwargs": {"mode": "cash_balance", "days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "weather_alerts",
            "tool": "weather_alerts",
            "kwargs": {"days_back": days_back},
            "group": "B",
        }
    )
    plan.append(
        {
            "label": "whale_alert",
            "tool": "whale_alert",
            "kwargs": {"days_back": min(days_back, 730)},
            "group": "B",
        }
    )

    # wikipedia_pageviews: tool has a 365d per-call limit (no bypass needed — API returns
    # up to 365d naturally). Call once per year of history.
    wikipedia_calls = min(5, days_back // 365 + 1)
    for i in range(wikipedia_calls):
        plan.append(
            {
                "label": f"wikipedia_pageviews_{i}",
                "tool": "wikipedia_pageviews",
                "kwargs": {"days_back": 365},
                "group": "B",
            }
        )

    # ── Group C: skip entries (live-only or internal utility) ────────────────
    for label, reason in [
        ("cert_transparency", "live-only: crt.sh current state only"),
        ("dns_monitor", "live-only: bulk resolve only"),
        ("internet_outages", "live-only: RIPE/Cloudflare real-time only"),
        ("backtest", "internal engine — not a data source"),
        ("code_executor", "utility — not a data source"),
        ("file_manager", "utility — not a data source"),
        ("liquidity_regime", "computed signal — runs after market_data backfill"),
        ("pipeline_query", "internal query tool"),
        ("web_browse", "live-only"),
        ("web_search", "live-only"),
        # gdelt is now handled via the gdelt_backfill group-A entry above
    ]:
        plan.append(
            {
                "label": label,
                "tool": label,
                "kwargs": {},
                "group": "C",
                "skip": True,
                "reason": reason,
            }
        )

    return plan


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class BackfillCheckpoint:
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    days_back: int = _DEFAULT_DAYS
    completed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "BackfillCheckpoint":
        try:
            raw = json.loads(Path(path).read_text())
            cp = cls(
                started_at=raw.get(
                    "started_at", datetime.now(timezone.utc).isoformat()
                ),
                days_back=raw.get("days_back", _DEFAULT_DAYS),
                completed=set(raw.get("completed", [])),
                failed=raw.get("failed", {}),
                skipped=raw.get("skipped", []),
            )
            return cp
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "started_at": self.started_at,
                    "days_back": self.days_back,
                    "completed": sorted(self.completed),
                    "failed": self.failed,
                    "skipped": self.skipped,
                },
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# DB observation counting (uses direct SQLite for global counts)
# ---------------------------------------------------------------------------


def _count_obs_by_tool(db_path: str) -> dict[str, int]:
    """Return {source_tool: count} for all tools in entity_observations."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT source_tool, COUNT(*) FROM entity_observations GROUP BY source_tool"
        ).fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except sqlite3.OperationalError:
        return {}


def _count_total_obs(db_path: str) -> int:
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# Tool dispatch with retry logic
# ---------------------------------------------------------------------------


def _run_one(
    registry: ToolRegistry,
    entry: dict[str, Any],
    db_path: str,
    dry_run: bool,
    delay: float,
) -> tuple[bool, int, str]:
    """
    Execute one backfill plan entry.

    Returns:
        (success: bool, obs_added: int, error_msg: str)
    """
    label = entry["label"]
    tool_name = entry["tool"]
    kwargs = entry.get("kwargs", {})

    if dry_run:
        log.info("[DRY-RUN] Would call: %s %s", tool_name, kwargs)
        return True, 0, ""

    before = _count_total_obs(db_path)
    time.sleep(delay)

    last_error = ""
    for attempt in range(2):
        try:
            result = registry.execute(tool_name, **kwargs)
            after = _count_total_obs(db_path)
            if not result.success:
                log.warning(
                    "  %s returned success=False: %s", label, result.output[:120]
                )
            return True, max(0, after - before), ""
        except Exception as exc:
            err_str = str(exc)
            last_error = err_str
            if "429" in err_str or "rate limit" in err_str.lower():
                log.warning(
                    "  %s HTTP 429 — sleeping %ss before retry", label, _HTTP_429_SLEEP
                )
                time.sleep(_HTTP_429_SLEEP)
                continue
            if (
                "OperationalError" in type(exc).__name__
                or "database is locked" in err_str.lower()
            ):
                if attempt == 0:
                    log.warning(
                        "  %s DB locked — sleeping %ss before retry",
                        label,
                        _DB_RETRY_SLEEP,
                    )
                    time.sleep(_DB_RETRY_SLEEP)
                    continue
            break

    return False, 0, last_error


# ---------------------------------------------------------------------------
# Verify Group B tools (probe with small window)
# ---------------------------------------------------------------------------


def _verify_group_b(
    registry: ToolRegistry,
    plan: list[dict[str, Any]],
    db_path: str,
) -> set[str]:
    """Probe each Group B tool with days_back=30. Return labels that respond OK."""
    ok: set[str] = set()
    group_b = [e for e in plan if e.get("group") == "B" and not e.get("skip")]
    log.info("Verifying %d Group B tools ...", len(group_b))
    for entry in group_b:
        kwargs = {**entry.get("kwargs", {}), "days_back": 30}
        try:
            result = registry.execute(entry["tool"], **kwargs)
            data = getattr(result, "data", None) or {}
            has_data = bool(
                data.get("records")
                or data.get("entries")
                or data.get("count", 0)
                or data.get("results")
                or data.get("markets")
                or (result.success and result.output.strip())
            )
            status = "OK" if has_data else "EMPTY"
            log.info("  %-40s %s", entry["label"], status)
            if has_data:
                ok.add(entry["label"])
        except Exception as exc:
            log.info("  %-40s ERROR: %s", entry["label"], str(exc)[:80])
        time.sleep(_RATE_LIMIT_SLEEP)
    return ok


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 47 Historical Backfill Runner")
    parser.add_argument("--db-path", default=_DEFAULT_DB, help="Pipeline DB path")
    parser.add_argument("--days-back", type=int, default=_DEFAULT_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tool", default="", help="Run only this label")
    parser.add_argument(
        "--verify", action="store_true", help="Probe Group B tools first"
    )
    parser.add_argument(
        "--no-retry", action="store_true", help="Skip previously-failed tools"
    )
    parser.add_argument("--delay", type=float, default=_RATE_LIMIT_SLEEP)
    parser.add_argument("--skip-group-b", action="store_true")
    parser.add_argument("--group-b-only", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build the tool registry using the specified DB path
    config = AgentConfig.from_env()
    # Override only the db_path so all other keys (FRED etc.) come from env
    config = AgentConfig(
        llm=config.llm,
        pipeline=PipelineConfig(db_path=args.db_path),
        fred_api_key=config.fred_api_key,
        max_steps=config.max_steps,
        max_plan_depth=config.max_plan_depth,
        memory_dir=config.memory_dir,
        tool_timeout=config.tool_timeout,
        lesson_min_support=config.lesson_min_support,
        lesson_min_runs=config.lesson_min_runs,
        episode_ttl_days=config.episode_ttl_days,
        verbose=config.verbose,
    )
    registry = build_tool_registry(config)

    plan = _build_plan(args.days_back)

    # Load checkpoint
    cp = BackfillCheckpoint.load(_CHECKPOINT_FILE)
    cp.days_back = args.days_back

    # Optional: verify Group B first
    verified_b: set[str] | None = None
    if args.verify:
        verified_b = _verify_group_b(registry, plan, args.db_path)
        log.info(
            "Verified Group B: %d/%d OK",
            len(verified_b),
            sum(1 for e in plan if e.get("group") == "B" and not e.get("skip")),
        )

    # Filter plan
    active = []
    for entry in plan:
        label = entry["label"]
        group = entry.get("group", "A")

        if entry.get("skip"):
            if label not in cp.skipped:
                cp.skipped.append(label)
            continue

        if args.tool and label != args.tool:
            continue

        if args.skip_group_b and group == "B":
            continue
        if args.group_b_only and group != "B":
            continue

        if label in cp.completed:
            continue

        if args.no_retry and label in cp.failed:
            continue

        # If we ran --verify, skip Group B tools that didn't respond
        if verified_b is not None and group == "B" and label not in verified_b:
            log.info("  Skipping %s (verify probe returned empty)", label)
            cp.skipped.append(label)
            cp.save(_CHECKPOINT_FILE)
            continue

        active.append(entry)

    n = len(active)
    if n == 0:
        log.info("Nothing to run (all labels completed or filtered).")
        return 0

    total_before = _count_total_obs(args.db_path)
    wall_start = time.time()

    log.info("Phase 47 Backfill — %s  (%d days)", args.db_path, args.days_back)
    log.info("─" * 70)
    log.info("Plan: %d entries to run | Total obs before: %d", n, total_before)
    if args.dry_run:
        log.info("[DRY-RUN MODE — no API calls will be made]")
    log.info("")

    ok_count = 0
    fail_count = 0
    total_added = 0

    for idx, entry in enumerate(active, 1):
        label = entry["label"]
        group = entry.get("group", "A")
        grp_tag = f"[GRP-{group}]" if group != "A" else ""
        bypass_tag = "[BYPASS]" if entry.get("kwargs", {}).get("_backfill") else ""
        prefix = f"[{idx:3d}/{n}] {label:<42} {bypass_tag:<9} {grp_tag:<9}"

        t0 = time.time()
        success, added, error = _run_one(
            registry, entry, args.db_path, args.dry_run, args.delay
        )
        elapsed = time.time() - t0

        if success:
            ok_count += 1
            total_added += added
            if not args.dry_run:
                cp.completed.add(label)
                cp.failed.pop(label, None)
            log.info("%s ✓  +%d obs  %.1fs", prefix, added, elapsed)
        else:
            fail_count += 1
            if not args.dry_run:
                cp.failed[label] = error
            log.warning("%s ✗  %s", prefix, error[:80])

        if not args.dry_run:
            cp.save(_CHECKPOINT_FILE)  # flush immediately after each entry

    wall_elapsed = time.time() - wall_start
    total_after = _count_total_obs(args.db_path)

    log.info("")
    log.info("─" * 70)
    log.info(
        "Done: %d completed, %d failed, %d skipped",
        ok_count,
        fail_count,
        len(cp.skipped),
    )
    log.info(
        "Total new observations: %d  (before=%d, after=%d)",
        total_after - total_before,
        total_before,
        total_after,
    )
    log.info("Wall time: %.1f min", wall_elapsed / 60)

    if fail_count:
        log.info("")
        log.info("Failed labels (re-run with --no-retry to skip):")
        for lbl, err in cp.failed.items():
            log.info("  %-40s %s", lbl, err[:80])

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
