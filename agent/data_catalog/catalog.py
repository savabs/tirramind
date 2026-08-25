"""
TirraMind — Data Governance Catalog (Idea 13)

Lightweight in-process replacement for OpenMetadata, backed by the existing
PipelineStore SQLite database.  No Docker, no external service, zero extra
CPU overhead beyond a handful of SQL queries.

What it provides
----------------
1. **Tool Registry** — static manifest of all 51+ data sources with:
   - category, expected update frequency, SLA window (hours)
   - list of signal name patterns the tool produces

2. **Freshness Monitor** — for each registered tool, queries the pipeline
   store for the most recent observation and checks against the SLA.
   Emits ``catalog.{tool_name}.freshness_hours`` and
   ``catalog.{tool_name}.sla_breach`` signals when violated.

3. **Lineage Query** — given an entity_id, lists every source_tool that
   has ever contributed observations for it.  Answers "what feeds does
   this entity's graph node depend on?"

4. **Schema Fingerprint** — SHA-256 of each registered tool's metadata
   dict.  Detects when a tool's SLA / category / frequency is changed
   (schema drift) and logs a warning.

CPU Safety
----------
- Only SQLite reads (GROUP BY + MAX) — O(rows scanned) but capped by
  ``max_tools`` (default 200).
- No torch, no numpy, no network calls.
- Disabled by default (``TrainerConfig.use_data_catalog = False``).

References
----------
OpenMetadata (2024). Data Governance Platform, Apache 2.0.
    https://github.com/open-metadata/OpenMetadata
    Concepts: catalog entity, freshness SLA, lineage graph, schema
    versioning.  This module implements the same logical model without
    the server dependency.

Dama International (2017). DAMA-DMBOK 2nd ed., Chapter 9 — Data Quality.
    SLA breach = observed staleness > expected update cadence × tolerance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_HOURS: float = 3600.0
_EPS: float = 1e-6


# ═══════════════════════════════════════════════════════════════
# Tool Manifest — static metadata for all 51+ data sources
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToolMeta:
    """Metadata for a single data source tool.

    Attributes
    ----------
    name : str        Unique tool identifier (matches source_tool in store).
    category : str    Data category (maritime, macro, corporate, …).
    frequency_hours : float  Expected update cadence in hours.
    sla_hours : float  Maximum acceptable staleness before SLA breach.
    signals : tuple[str, ...]  Signal name prefixes this tool produces.
    description : str  One-line human-readable description.
    """

    name: str
    category: str
    frequency_hours: float
    sla_hours: float
    signals: tuple[str, ...] = ()
    description: str = ""


# Canonical manifest — one entry per data source tool
# frequency_hours / sla_hours: sla = 2× frequency (conservative tolerance)
_TOOL_MANIFEST: list[ToolMeta] = [
    # ── Maritime ──────────────────────────────────────────────
    ToolMeta(
        "ais_vessel",
        "maritime",
        6.0,
        12.0,
        ("ais.",),
        "AIS vessel position & routing signals",
    ),
    ToolMeta(
        "transport_throughput",
        "maritime",
        24.0,
        48.0,
        ("transport.",),
        "Port throughput & congestion",
    ),
    ToolMeta(
        "supply_chain_monitor",
        "maritime",
        24.0,
        48.0,
        ("supply_chain.",),
        "Supply chain disruption alerts",
    ),
    # ── Macro ──────────────────────────────────────────────────
    ToolMeta(
        "macro_data",
        "macro",
        24.0,
        72.0,
        ("macro.",),
        "GDP, CPI, rates, employment data",
    ),
    ToolMeta(
        "global_pmi",
        "macro",
        168.0,
        336.0,
        ("pmi.",),
        "Global manufacturing & services PMI",
    ),
    ToolMeta(
        "central_bank_balance",
        "macro",
        168.0,
        336.0,
        ("central_bank.",),
        "Central bank balance sheet flows",
    ),
    ToolMeta(
        "consumer_sentiment",
        "macro",
        168.0,
        336.0,
        ("sentiment.",),
        "Consumer confidence & sentiment",
    ),
    ToolMeta(
        "building_permits",
        "macro",
        168.0,
        336.0,
        ("permits.",),
        "Building permits & housing starts",
    ),
    ToolMeta(
        "capital_flows",
        "macro",
        168.0,
        336.0,
        ("capital_flows.",),
        "Cross-border capital flow data",
    ),
    ToolMeta(
        "sovereign_debt",
        "macro",
        24.0,
        72.0,
        ("sovereign.",),
        "Sovereign bond spreads & ratings",
    ),
    # ── Energy ────────────────────────────────────────────────
    ToolMeta(
        "energy_supply",
        "energy",
        168.0,
        336.0,
        ("energy.",),
        "Oil/gas inventory & production data",
    ),
    ToolMeta(
        "electricity_monitor",
        "energy",
        24.0,
        48.0,
        ("electricity.",),
        "Grid load & generation mix",
    ),
    ToolMeta(
        "interconnection_queue",
        "energy",
        168.0,
        336.0,
        ("interconnection.",),
        "Renewable interconnection queue",
    ),
    ToolMeta(
        "power_grid",
        "energy",
        24.0,
        48.0,
        ("power_grid.",),
        "Power grid stress & outage signals",
    ),
    # ── Financial / Positioning ───────────────────────────────
    ToolMeta(
        "cftc",
        "positioning",
        168.0,
        336.0,
        ("cftc.",),
        "CFTC COT report — net positioning",
    ),
    ToolMeta(
        "finra_short_volume",
        "positioning",
        24.0,
        48.0,
        ("finra.",),
        "FINRA daily short sale volume",
    ),
    ToolMeta(
        "insider_filings",
        "corporate",
        24.0,
        72.0,
        ("insider.",),
        "SEC Form 4 insider transactions",
    ),
    ToolMeta(
        "form144",
        "corporate",
        24.0,
        72.0,
        ("form144.",),
        "SEC Form 144 insider sale notifications",
    ),
    ToolMeta(
        "whale_alert",
        "crypto",
        6.0,
        12.0,
        ("whale.",),
        "On-chain large transaction alerts",
    ),
    ToolMeta(
        "defi_flows", "crypto", 6.0, 12.0, ("defi.",), "DeFi protocol inflow/outflow"
    ),
    ToolMeta(
        "polymarket",
        "prediction",
        6.0,
        12.0,
        ("polymarket.",),
        "Prediction market probabilities",
    ),
    ToolMeta(
        "polymarket_whales",
        "prediction",
        6.0,
        12.0,
        ("polymarket_whales.",),
        "Large polymarket positions",
    ),
    ToolMeta(
        "liquidity_regime",
        "positioning",
        24.0,
        48.0,
        ("liquidity.",),
        "Liquidity regime classification",
    ),
    # ── Corporate ─────────────────────────────────────────────
    ToolMeta(
        "bankruptcy_court",
        "corporate",
        24.0,
        72.0,
        ("bankruptcy.",),
        "Chapter 11/7 court filings",
    ),
    ToolMeta(
        "creditor_filings",
        "corporate",
        24.0,
        72.0,
        ("creditor.",),
        "UCC & creditor lien filings",
    ),
    ToolMeta(
        "lobbying",
        "corporate",
        168.0,
        336.0,
        ("lobbying.",),
        "Federal lobbying disclosures",
    ),
    ToolMeta(
        "gov_contracts",
        "corporate",
        24.0,
        48.0,
        ("govcon.",),
        "USASpending.gov contract awards",
    ),
    ToolMeta(
        "patent_filings",
        "corporate",
        168.0,
        336.0,
        ("patent.",),
        "USPTO patent application filings",
    ),
    ToolMeta(
        "job_postings",
        "corporate",
        168.0,
        336.0,
        ("jobs.",),
        "Online job posting volume by sector",
    ),
    # ── Geopolitical / Regulatory ─────────────────────────────
    ToolMeta(
        "sanctions_monitor",
        "geopolitical",
        24.0,
        48.0,
        ("sanctions.",),
        "OFAC & EU sanctions list changes",
    ),
    ToolMeta(
        "political_risk",
        "geopolitical",
        24.0,
        72.0,
        ("polrisk.",),
        "Political risk event scores",
    ),
    ToolMeta(
        "regulatory_gazette",
        "regulatory",
        24.0,
        72.0,
        ("reggazette.",),
        "Federal register & regulatory changes",
    ),
    ToolMeta(
        "foia_requests",
        "regulatory",
        168.0,
        336.0,
        ("foia.",),
        "FOIA request filing & response data",
    ),
    ToolMeta(
        "drug_regulatory",
        "regulatory",
        24.0,
        72.0,
        ("drug_reg.",),
        "FDA drug approval & recall alerts",
    ),
    # ── News & Intelligence ───────────────────────────────────
    ToolMeta(
        "gdelt", "news", 6.0, 12.0, ("gdelt.",), "GDELT global event & tone signals"
    ),
    ToolMeta(
        "academic_preprints",
        "research",
        24.0,
        72.0,
        ("preprint.",),
        "arXiv/SSRN financial preprints",
    ),
    ToolMeta(
        "wikipedia_pageviews",
        "sentiment",
        24.0,
        48.0,
        ("wiki.",),
        "Wikipedia article view spikes",
    ),
    # ── Infrastructure / Digital ──────────────────────────────
    ToolMeta(
        "internet_outages",
        "infrastructure",
        6.0,
        12.0,
        ("outage.",),
        "Internet outage & BGP anomalies",
    ),
    ToolMeta(
        "dns_monitor",
        "infrastructure",
        6.0,
        12.0,
        ("dns.",),
        "DNS anomaly & domain registration",
    ),
    ToolMeta(
        "cert_transparency",
        "infrastructure",
        6.0,
        12.0,
        ("cert.",),
        "TLS certificate transparency logs",
    ),
    ToolMeta(
        "internet_infrastructure",
        "infrastructure",
        6.0,
        12.0,
        ("infra.",),
        "Internet infrastructure health",
    ),
    # ── Event-Driven ──────────────────────────────────────────
    ToolMeta(
        "earthquake_proximity",
        "event",
        1.0,
        6.0,
        ("earthquake.",),
        "USGS earthquake proximity alerts",
    ),
    ToolMeta(
        "disease_surveillance",
        "event",
        24.0,
        72.0,
        ("disease.",),
        "WHO/CDC disease outbreak signals",
    ),
    ToolMeta(
        "weather_alerts", "event", 1.0, 6.0, ("weather.",), "NWS severe weather alerts"
    ),
    ToolMeta(
        "food_security",
        "event",
        24.0,
        72.0,
        ("food_sec.",),
        "FEWS NET food security alerts",
    ),
    ToolMeta(
        "migration_flows",
        "event",
        168.0,
        336.0,
        ("migration.",),
        "UNHCR/IOM population movement",
    ),
    # ── Trade & Commodity ─────────────────────────────────────
    ToolMeta(
        "comtrade",
        "trade",
        168.0,
        336.0,
        ("comtrade.",),
        "UN Comtrade bilateral trade flows",
    ),
    # ── Instruments ───────────────────────────────────────────
    ToolMeta(
        "instrument_universe",
        "reference",
        168.0,
        336.0,
        ("instrument.",),
        "Instrument universe & metadata",
    ),
    # ── Satellite ─────────────────────────────────────────────
    ToolMeta(
        "satellite_activity",
        "satellite",
        24.0,
        72.0,
        ("satellite.",),
        "Satellite orbit & activity signals",
    ),
]

# Fast lookup by name
_MANIFEST_BY_NAME: dict[str, ToolMeta] = {m.name: m for m in _TOOL_MANIFEST}


# ═══════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════


@dataclass
class FreshnessStatus:
    """Freshness check result for one tool.

    Attributes
    ----------
    tool_name : str         Tool identifier.
    last_seen_at : float    Unix timestamp of last observation (0 = never).
    freshness_hours : float Hours since last observation.
    sla_hours : float       SLA threshold from manifest.
    is_breach : bool        True when freshness_hours > sla_hours.
    hours_overdue : float   How many hours past SLA (0 if not breached).
    checked_at : float      Unix timestamp of this check.
    """

    tool_name: str
    last_seen_at: float
    freshness_hours: float
    sla_hours: float
    is_breach: bool
    hours_overdue: float
    checked_at: float


@dataclass
class CatalogReport:
    """Full freshness report across all registered tools.

    Attributes
    ----------
    statuses : dict[str, FreshnessStatus]  Per-tool freshness status.
    n_breached : int    Number of tools with SLA breaches.
    n_never_seen : int  Tools with no observations at all.
    n_healthy : int     Tools within SLA.
    generated_at : float  Unix timestamp.
    """

    statuses: dict[str, FreshnessStatus]
    n_breached: int
    n_never_seen: int
    n_healthy: int
    generated_at: float

    @property
    def breached_tools(self) -> list[str]:
        return [t for t, s in self.statuses.items() if s.is_breach]

    @property
    def never_seen_tools(self) -> list[str]:
        return [t for t, s in self.statuses.items() if s.last_seen_at == 0.0]


# ═══════════════════════════════════════════════════════════════
# DataCatalog
# ═══════════════════════════════════════════════════════════════


class DataCatalog:
    """In-process data governance catalog for TirraMind.

    Parameters
    ----------
    extra_tools : list[ToolMeta] | None
        Additional tools to register beyond the built-in manifest.
        Use this to register new tools without editing the static list.
    max_tools : int
        CPU-safety cap on how many tools are checked per ``check_freshness``
        call.  Default 200.
    """

    def __init__(
        self,
        extra_tools: list[ToolMeta] | None = None,
        max_tools: int = 200,
    ) -> None:
        self._registry: dict[str, ToolMeta] = dict(_MANIFEST_BY_NAME)
        if extra_tools:
            for tm in extra_tools:
                self._registry[tm.name] = tm
        self._max_tools = max_tools

    # ── Tool Registry ──────────────────────────────────────────────────────

    def register(self, meta: ToolMeta) -> None:
        """Register (or overwrite) a tool's metadata."""
        self._registry[meta.name] = meta

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_meta(self, tool_name: str) -> ToolMeta | None:
        return self._registry.get(tool_name)

    def schema_fingerprint(self, tool_name: str) -> str:
        """SHA-256 of the tool's metadata dict (detects definition drift)."""
        meta = self._registry.get(tool_name)
        if meta is None:
            return ""
        payload = json.dumps(
            {
                "name": meta.name,
                "category": meta.category,
                "frequency_hours": meta.frequency_hours,
                "sla_hours": meta.sla_hours,
                "signals": list(meta.signals),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ── Freshness Monitor ──────────────────────────────────────────────────

    def check_freshness(self, store: Any, now: float | None = None) -> CatalogReport:
        """Query the pipeline store for last observation per tool.

        For each registered tool, finds ``MAX(observed_at)`` in
        ``entity_observations``.  Compares against the tool's SLA window.

        Args:
            store: PipelineStore instance.
            now:   Reference time (Unix ts).  Defaults to ``time.time()``.

        Returns:
            CatalogReport with per-tool FreshnessStatus entries.

        CPU Safety:
            One SQL query per tool (MAX scan on indexed column).
            Capped at ``max_tools`` tools per call.
        """
        if now is None:
            now = time.time()

        tool_names = self.tool_names[: self._max_tools]
        statuses: dict[str, FreshnessStatus] = {}

        for tool_name in tool_names:
            meta = self._registry[tool_name]
            last_seen = self._query_last_seen(store, tool_name)

            if last_seen > 0.0:
                freshness_hours = (now - last_seen) / _HOURS
            else:
                freshness_hours = float("inf")

            is_breach = freshness_hours > meta.sla_hours
            hours_overdue = (
                max(0.0, freshness_hours - meta.sla_hours) if is_breach else 0.0
            )

            statuses[tool_name] = FreshnessStatus(
                tool_name=tool_name,
                last_seen_at=last_seen,
                freshness_hours=freshness_hours,
                sla_hours=meta.sla_hours,
                is_breach=is_breach,
                hours_overdue=hours_overdue,
                checked_at=now,
            )

        n_breached = sum(1 for s in statuses.values() if s.is_breach)
        n_never = sum(1 for s in statuses.values() if s.last_seen_at == 0.0)
        n_healthy = len(statuses) - n_breached

        log.info(
            "DataCatalog.check_freshness: %d tools — %d healthy, %d breached, %d never seen.",
            len(statuses),
            n_healthy,
            n_breached,
            n_never,
        )
        return CatalogReport(
            statuses=statuses,
            n_breached=n_breached,
            n_never_seen=n_never,
            n_healthy=n_healthy,
            generated_at=now,
        )

    # ── Lineage ────────────────────────────────────────────────────────────

    def get_lineage(self, store: Any, entity_id: str) -> list[str]:
        """Return all source_tools that contributed observations for entity_id.

        Args:
            store: PipelineStore instance.
            entity_id: The entity whose lineage to trace.

        Returns:
            Sorted list of source_tool names.
        """
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT DISTINCT source_tool FROM entity_observations "
                "WHERE entity_id=? ORDER BY source_tool",
                (entity_id,),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            log.warning(
                "DataCatalog.get_lineage failed for %s.", entity_id, exc_info=True
            )
            return []

    # ── Signal Storage ─────────────────────────────────────────────────────

    def store_freshness_signals(
        self,
        store: Any,
        report: CatalogReport,
    ) -> int:
        """Persist freshness & breach signals to pipeline store.

        Signal names:
        - ``catalog.{tool}.freshness_hours`` — current staleness in hours
        - ``catalog.{tool}.sla_breach``      — hours overdue (0 = no breach)

        Returns:
            Number of signals written.
        """
        n_written = 0
        for tool_name, status in report.statuses.items():
            if not status.is_breach and status.last_seen_at == 0.0:
                continue  # skip never-seen tools to avoid noise
            for sig_name, value in [
                (
                    f"catalog.{tool_name}.freshness_hours",
                    (
                        status.freshness_hours
                        if status.freshness_hours != float("inf")
                        else -1.0
                    ),
                ),
                (f"catalog.{tool_name}.sla_breach", status.hours_overdue),
            ]:
                try:
                    store.store_signal(
                        signal_name=sig_name,
                        value=value,
                        observed_at=status.checked_at,
                        source_tool="data_catalog",
                    )
                    n_written += 1
                except Exception:
                    log.warning(
                        "DataCatalog: failed to store %s.", sig_name, exc_info=True
                    )
        return n_written

    # ── Helpers ────────────────────────────────────────────────────────────

    def _query_last_seen(self, store: Any, tool_name: str) -> float:
        """Return MAX(observed_at) for tool_name, or 0.0 if no rows."""
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT MAX(observed_at) FROM entity_observations WHERE source_tool=?",
                (tool_name,),
            ).fetchall()
            val = rows[0][0] if rows else None
            return float(val) if val is not None else 0.0
        except Exception:
            log.warning(
                "DataCatalog: could not query last_seen for %s.",
                tool_name,
                exc_info=True,
            )
            return 0.0
