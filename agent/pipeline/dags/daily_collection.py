"""
TirraMind — Daily Collection DAG

Fetches data from all stateless surveillance tools once daily.
All nodes are independent → single parallel layer → maximum throughput.

Schedule: weekdays at 18:00 UTC (after US market close, CFTC/FINRA publish windows).

Nodes:
    fetch_cftc          — CFTC Commitments of Traders, latest report
    derive_cftc_features — CFTC positioning features derived from fetch_cftc's rows
    fetch_finra_scan    — FINRA Reg SHO short volume, all-ticker scan
    fetch_power_demand  — NYISO power grid actual demand by zone
    fetch_power_fuel    — NYISO generation by fuel type
    fetch_gdelt         — GDELT geopolitical events, last 24h
    fetch_polymarket    — Polymarket prediction market odds, all categories
    fetch_cert_domains  — CT log recent issuances for 20 financial domains (callable)
    fetch_dns_domains   — DNS bulk_resolve for 20 financial domains

Change 12: Optional ``tool_router`` parameter.  When provided, the bandit
decides which optional tools to enable before DAG execution.

Change 15 (Tier 8): Quarantine cycle for discovered tools.  Newly discovered
sources run in quarantine (shadowed observations) for a configurable number
of successful cycles before promotion to active.  Three consecutive failures
disable the source.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent.pipeline.dag import DAG
from agent.pipeline.operators import DOMAIN_TABLES_PARAM_KEY

if TYPE_CHECKING:
    from agent.learning.tool_router import ToolContext, ToolRoutingBandit
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# ── Financial domain watchlist (cert_transparency + dns_monitor) ──────────
# 20 major financial institution domains — banks, brokers, exchanges, regulators.
# Used for both CT log surveillance (cert issuance patterns) and DNS change
# monitoring (infrastructure migration, SaaS adoption signals).
# Max 20 enforced by dns_monitor's bulk_resolve limit.
FINANCIAL_DOMAINS: list[str] = [
    "jpmorgan.com",
    "goldmansachs.com",
    "morganstanley.com",
    "blackrock.com",
    "vanguard.com",
    "fidelity.com",
    "schwab.com",
    "citadel.com",
    "bridgewater.com",
    "aqr.com",
    "sec.gov",
    "cftc.gov",
    "federalreserve.gov",
    "bis.org",
    "nyse.com",
    "nasdaq.com",
    "cboe.com",
    "ice.com",
    "bloomberg.com",
    "refinitiv.com",
]


def _count_cert_observations(store: Any) -> int:
    """``COUNT(*)`` of cert_transparency rows in ``entity_observations``.

    The guard below needs real rows, not the collector's own opinion of how
    many certs it saw.  Counted through the store's existing connection —
    read-only SQL, no write.
    """
    row = (
        store._get_conn()
        .execute("SELECT COUNT(*) AS n FROM entity_observations WHERE source_tool = 'cert_transparency'")
        .fetchone()
    )
    return int(row["n"]) if row is not None else 0


def run_cert_domain_collection(
    params: dict[str, Any],
    upstream_results: dict[str, Any],
) -> dict[str, Any]:
    """FunctionOperator callback: fetch recent CT certs for all FINANCIAL_DOMAINS.

    Calls CertTransparencyTool(mode='recent') once per domain and persists
    domain entities + cert_issued observations via PipelineStore.

    **This node used to be the textbook silent failure.** It read ``count`` off
    each ToolResult, summed it, and returned a dict unconditionally — so when
    crt.sh dropped ``entry_timestamp`` and the collector's recency filter began
    discarding 100% of records, ``fetch_cert_domains`` reported
    ``status=completed, error=null`` after 221 seconds of doing nothing (live
    run 55233b8478c4, 2026-09-23). ``entity_observations`` sat at 19 rows with
    a max ``ingested_at`` of 2026-08-27. cert_transparency is TIME-GATED: crt.sh
    serves the log, but the *recency window* this node samples is gone the day
    after. Those 26 days are not backfillable.

    So this now measures a real ``COUNT(*)`` delta on ``entity_observations``
    and **raises** when the run produced nothing, rather than returning a
    payload that looks like a quiet day.  Three outcomes:

    * every domain errored, or not one cert came back → raise (node goes red)
    * certs came back but no new rows landed → ``status="skipped"`` with a
      reason, which ``classify_payload_status`` records as a skip, not a check.
      This is the honest reading of an idempotent re-run: crt.sh returns the
      same certs and the store's unique key collapses them.
    * new rows landed → ``status="completed"`` with the row count

    params:
        db_path       : str   — PipelineStore database path (injected by DAG builder)
        domains       : list  — override FINANCIAL_DOMAINS (optional, for tests)
        days_back     : int   — lookback window in days (default 30)
        request_delay : float — seconds to wait between domains (default 2.5)
    """
    from agent.pipeline.store import PipelineStore
    from agent.tools.cert_transparency import CertTransparencyTool

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    domains: list[str] = params.get("domains", FINANCIAL_DOMAINS)
    days_back: int = params.get("days_back", 30)
    # crt.sh rate-limits by source address. Measured 2026-09-23: 40 back-to-back
    # requests (2 per domain) got 1 of 4 domains through; a 2-3s gap got 4 of 4.
    request_delay: float = float(params.get("request_delay", 2.5))

    store = PipelineStore(db_path)
    try:
        rows_before = _count_cert_observations(store)
        tool = CertTransparencyTool(pipeline_store=store)
        per_domain: dict[str, dict[str, Any]] = {}
        total_certs = 0
        successes = 0
        for index, domain in enumerate(domains):
            if index and request_delay > 0:
                time.sleep(request_delay)
            result = tool.execute(
                mode="recent",
                domain=domain,
                days_back=days_back,
                limit=50,
            )
            data = result.data or {}
            count = data.get("count", 0)
            entry: dict[str, Any] = {
                "success": result.success,
                "count": count,
                "rows_persisted": data.get("rows_persisted", 0),
            }
            if not result.success:
                entry["error"] = (result.output or "")[:300]
            if data.get("base_error"):
                entry["base_error"] = str(data["base_error"])[:300]
            per_domain[domain] = entry
            if result.success:
                successes += 1
                total_certs += count
        rows_written = _count_cert_observations(store) - rows_before
    finally:
        store.close()

    failures = {d: e for d, e in per_domain.items() if not e["success"]}

    if successes == 0 or total_certs == 0:
        # Raise, do not return. A FunctionOperator that returns is a green
        # check; the whole point of this node's history is that a green check
        # for zero rows is indistinguishable from a quiet day.
        detail = "; ".join(
            f"{d}: " + (e.get("error") or e.get("base_error") or f"{e['count']} certs")
            for d, e in sorted(per_domain.items())
        )
        raise RuntimeError(
            f"fetch_cert_domains wrote nothing: {successes}/{len(domains)} domains "
            f"succeeded, {total_certs} certs in the last {days_back}d, "
            f"{rows_written} rows written to entity_observations. "
            f"cert_transparency is time-gated — a zero-row day is unrecoverable, "
            f"not a quiet day. Per-domain: {detail}"
        )

    payload: dict[str, Any] = {
        "status": "completed" if rows_written > 0 else "skipped",
        "domains_scanned": len(domains),
        "domains_succeeded": successes,
        "total_certs": total_certs,
        "rows_written": rows_written,
        "per_domain": per_domain,
    }
    if failures:
        payload["failed_domains"] = sorted(failures)
    if rows_written <= 0:
        payload["reason"] = (
            f"{total_certs} certs returned across {successes} domain(s) but every "
            "observation was already stored — no new certificates since the last run"
        )
    return payload


def run_cftc_feature_derivation(
    params: dict[str, Any],
    upstream_results: dict[str, Any],
) -> dict[str, Any]:
    """FunctionOperator callback: derive CFTC positioning features (Layer 2).

    Re-derives ``futures_positioning_derived`` from the ``futures_positioning``
    rows ``fetch_cftc`` just stored.  Pure re-derivation over data already in
    the database — no network call — so it also refills any gap left by a
    stopped collector on the first run after one.

    Until this node existed the derivation only ever ran by hand: the math sat
    in ``scripts/add_cftc_derived_features.py``, no DAG imported it, and
    ``cftc_derived`` froze at 2026-04-14 while ``cftc`` kept collecting for
    another 162 days with nothing red to show for it (audit P6.4).

    The returned ``status`` is the point: a run that derives nothing says
    ``"skipped"``, which ``classify_payload_status`` records as a skip instead
    of a green check, and the node declares ``entity_observations`` as its
    domain table so the executor's zero-rows guard measures real rows rather
    than the per-node result envelope.

    params:
        db_path : str — PipelineStore database path (injected by DAG builder)
    """
    from agent.pipeline.store import PipelineStore
    from agent.quant.cftc_features import derive_and_store

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")

    store = PipelineStore(db_path)
    try:
        stats = derive_and_store(store)
    finally:
        store.close()

    rows_written = int(stats.get("rows_written") or 0)
    result: dict[str, Any] = {
        "status": "completed" if rows_written else "skipped",
        "rows_written": rows_written,
        "rows_derived": stats.get("rows_derived", 0),
        "entities": stats.get("entities", 0),
        "max_observed_at": stats.get("max_observed_at"),
        "deduplicated": stats.get("deduplicated", 0),
    }
    if not rows_written:
        result["reason"] = (
            "every derived row was already stored — no new futures_positioning observations"
            if stats.get("rows_derived")
            else "no futures_positioning observations in the database"
        )
    return result


def run_evidence_ingest_from_gdelt(
    params: dict[str, Any],
    upstream_results: dict[str, Any],
) -> dict[str, Any]:
    """FunctionOperator callback: turn this cycle's GDELT events into an
    Entity Graph (evidence) document.

    Builds one deterministic sentence per event
    ("{actor1} {event_description} {actor2} in {location}.") from real,
    already-fetched GDELT rows, then ingests the day's events as a single
    document via the existing deterministic extractor (agent/evidence/). This
    is what actually populates the Entity Graph tier with live content — the
    same cross-document co-occurrence signal (a recurring entity pair across
    many days' events is the alpha) now backed by continuously-refreshed real
    data instead of a handful of static demo documents.

    params:
        db_path         : str — PipelineStore database path (injected by DAG builder)
        evidence_db_path: str — EvidenceGraphStore database path
        max_events      : int — cap events processed per cycle (default 200)
    """
    from agent.evidence import EvidenceGraphStore, EvidenceIngestor, ingest_to_store
    from agent.pipeline.store import PipelineStore

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    evidence_db_path = params.get("evidence_db_path", ".tirra_pipeline/evidence.db")
    max_events = params.get("max_events", 200)

    store = PipelineStore(db_path)
    rows = store.query_data("gdelt", limit=1)  # this cycle's fetch_gdelt result
    if not rows:
        return {"ingested_docs": 0, "reason": "no gdelt data yet"}

    evidence_store = EvidenceGraphStore(evidence_db_path)
    ingestor = EvidenceIngestor.from_registry(db_path=db_path)

    ingested = 0
    total_sentences = 0
    for row in rows:
        events = (row.get("data") or {}).get("events", [])
        sentences = []
        for e in events[:max_events]:
            a1 = (e.get("actor1") or {}).get("name")
            a2 = (e.get("actor2") or {}).get("name")
            if not a1 or not a2:
                continue
            desc = (e.get("event_description") or "interacted with").lower()
            loc = (e.get("location") or {}).get("name") or ""
            sentence = f"{a1} {desc} {a2}"
            if loc:
                sentence += f" in {loc}"
            sentences.append(sentence + ".")
        if not sentences:
            continue

        doc_id = f"gdelt_{row.get('fetched_at')}"
        new = ingest_to_store(
            evidence_store,
            ingestor,
            doc_id=doc_id,
            text=" ".join(sentences),
            source="gdelt",
            title=f"GDELT events @ {row.get('fetched_at')}",
            doc_type="text",
        )
        if new:
            ingested += 1
            total_sentences += len(sentences)

    return {"ingested_docs": ingested, "sentences": total_sentences, "stats": evidence_store.stats()}


# ── Quarantine constants ──────────────────────────────────
_QUARANTINE_CYCLES_TO_PROMOTE = 5
_MAX_CONSECUTIVE_FAILURES = 3


def run_quarantine_cycle(store: PipelineStore) -> dict[str, str]:
    """Execute one quarantine check cycle for discovered sources.

    For each source in 'quarantine' status:
    - Try loading its tool config and executing a probe
    - On success: decrement remaining quarantine cycles; promote to 'active'
      after *_QUARANTINE_CYCLES_TO_PROMOTE* consecutive successes
    - On failure: increment consecutive_failures; move to 'disabled' after
      *_MAX_CONSECUTIVE_FAILURES*

    Returns a mapping of ``{source_id: new_status}`` for each source that
    changed status, or ``"quarantine"`` if unchanged.
    """
    results: dict[str, str] = {}
    try:
        sources = store.query_discovered_sources(status="quarantine")
    except Exception:
        return results

    for src in sources:
        source_id = src["source_id"]
        try:
            from agent.discovery.tool_factory import ToolFactory

            factory = ToolFactory()
            config_path = factory._config_dir / f"discovered_{source_id[:8]}.json"
            if not config_path.exists():
                # No config on disk — can't probe, treat as failure
                raise FileNotFoundError(f"No config for {source_id}")

            tools = factory.load_all_configs()
            tool = next((t for t in tools if source_id[:8] in t.name), None)
            if tool is None:
                raise FileNotFoundError(f"Tool not found for {source_id}")

            # Execute a probe — if it doesn't raise, consider it a success
            tool.execute()

            # Success: reset failure counter
            store.reset_source_failures(source_id)

            # Track quarantine progress via metadata
            meta = src.get("metadata_json") or {}
            if isinstance(meta, str):
                import json

                meta = json.loads(meta)
            q_successes = meta.get("quarantine_successes", 0) + 1
            meta["quarantine_successes"] = q_successes

            if q_successes >= _QUARANTINE_CYCLES_TO_PROMOTE:
                store.update_source_status(source_id, "active")
                results[source_id] = "active"
                log.info("Promoted discovered source %s to active", source_id)
            else:
                results[source_id] = "quarantine"

        except Exception as exc:
            store.increment_source_failures(source_id)
            # Refresh to get updated failure count
            updated = store.query_discovered_sources(status="quarantine")
            updated_src = next((s for s in updated if s["source_id"] == source_id), src)
            if updated_src.get("consecutive_failures", 0) >= _MAX_CONSECUTIVE_FAILURES:
                store.update_source_status(source_id, "disabled")
                results[source_id] = "disabled"
                log.warning(
                    "Disabled discovered source %s after %d failures: %s",
                    source_id,
                    _MAX_CONSECUTIVE_FAILURES,
                    exc,
                )
            else:
                results[source_id] = "quarantine"
                log.debug("Quarantine failure for %s: %s", source_id, exc)

    return results


def build_daily_collection_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
    tool_router: ToolRoutingBandit | None = None,
    tool_context: ToolContext | None = None,
) -> DAG:
    """Build the daily_collection DAG. Pure data declaration, no side effects.

    Parameters
    ----------
    db_path : str
        Path to the pipeline SQLite database.
    tool_router : ToolRoutingBandit, optional
        If provided, the bandit decides which optional tools are enabled.
    tool_context : ToolContext, optional
        Context for the routing decision (regime, day, staleness).
    """
    dag = DAG(
        name="daily_collection",
        schedule="0 18 * * 1-5",
        description="Daily surveillance: fetch all stateless data sources after US market close",
    )

    # ── CFTC Commitments of Traders ────────────────────────
    dag.add(
        "fetch_cftc",
        operator="cftc",
        table_name="cftc",
        params={"mode": "latest"},
        timeout=120,
        retries=2,
    )

    # ── CFTC derived positioning features (Layer 2) ────────
    # The derivation existed only as a hand-run script, so it ran once, on
    # 2026-05-03, and then never again (audit P6.4). Wiring it here is what
    # makes it run daily — and what makes a zero-row run visible: the node
    # declares the table it exists to populate, so the executor's guard
    # counts real rows instead of the pipeline_data envelope.
    dag.add(
        "derive_cftc_features",
        operator=run_cftc_feature_derivation,
        depends_on=["fetch_cftc"],
        params={
            "db_path": db_path,
            DOMAIN_TABLES_PARAM_KEY: ["entity_observations"],
        },
        timeout=120,
        retries=1,
        store_result=False,
    )

    # ── FINRA Short Volume (all-ticker scan) ───────────────
    dag.add(
        "fetch_finra_scan",
        operator="finra_short_volume",
        table_name="finra_short_volume",
        params={"mode": "short_volume"},  # no ticker → scan mode
        timeout=180,
        retries=2,
    )

    # ── NYISO Power Grid: Actual Demand ────────────────────
    dag.add(
        "fetch_power_demand",
        operator="power_grid",
        table_name="power_grid",
        params={"mode": "demand"},
        timeout=60,
        retries=2,
    )

    # ── NYISO Power Grid: Fuel Mix ─────────────────────────
    dag.add(
        "fetch_power_fuel",
        operator="power_grid",
        table_name="power_grid",
        params={"mode": "fuel_mix"},
        timeout=60,
        retries=2,
    )

    # ── GDELT Geopolitical Events (24h lookback) ───────────
    dag.add(
        "fetch_gdelt",
        operator="gdelt",
        table_name="gdelt",
        params={"mode": "events", "hours_back": 24, "limit": 500},
        timeout=120,
        retries=2,
    )

    # ── Entity Graph: ingest today's GDELT events as evidence ──
    dag.add(
        "ingest_evidence_from_gdelt",
        operator=run_evidence_ingest_from_gdelt,
        depends_on=["fetch_gdelt"],
        params={"db_path": db_path},
        timeout=60,
        retries=1,
        store_result=False,
    )

    # ── Polymarket Prediction Markets ──────────────────────
    dag.add(
        "fetch_polymarket",
        operator="polymarket",
        table_name="polymarket",
        params={"category": "all", "limit": 100},
        timeout=60,
        retries=2,
    )

    # ── BTC Whale Alert (confirmed-block, L2 entity persistence) ──
    # Phase 41: scheduled to widen observation diversity. The tool is
    # free (blockchain.info), no auth, and persists wallet entities +
    # wallet→instrument links for BTC-USD (see whale_alert L2).
    dag.add(
        "fetch_whale_alert",
        operator="whale_alert",
        table_name="whale_alert",
        params={"mode": "confirmed", "min_btc": 10.0, "limit": 100},
        timeout=60,
        retries=2,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 42 — Entity Diversity Expansion
    # Wires 8 dormant L2-capable tools into the daily schedule to lift
    # entity-type observation entropy from ~0.17 → ≥1.0 nats and
    # activate every dead entity type (company, protocol, person).
    # See [[phase42_entity_diversity_expansion]].
    # ═══════════════════════════════════════════════════════════════

    # ── SEC Form 4 insider purchases (company + person entities) ──
    # Flagship "cheap+weird" signal per project memory: executives
    # revealing private info through their own open-market buys.
    # SEC EDGAR is rate-limited to 10 req/s; 14 days at min_cluster_size=3
    # reliably finishes under 5 minutes.
    dag.add(
        "fetch_insider_filings",
        operator="insider_filings",
        table_name="insider_filings",
        params={"days_back": 14, "min_cluster_size": 3},
        timeout=300,
        retries=2,
    )

    # ── Central bank balance sheets (country entities, monetary state) ──
    # Fed/ECB/BOJ/BOE/PBOC balance sheets FX-normalised. Densifies the
    # 82 sparsely-observed country nodes beyond GDELT event counts.
    dag.add(
        "fetch_central_bank_balance",
        operator="central_bank_balance",
        table_name="central_bank_balance",
        params={"mode": "balance_sheets", "period": "1y"},
        timeout=120,
        retries=2,
    )

    # ── Sovereign yield curves — US (growth/inflation expectations) ──
    dag.add(
        "fetch_sovereign_debt_us",
        operator="sovereign_debt",
        table_name="sovereign_debt",
        params={"mode": "us_yields"},
        timeout=120,
        retries=2,
    )

    # ── Sovereign yield curves — EU (Bund, Gilt, OAT) ──
    dag.add(
        "fetch_sovereign_debt_eu",
        operator="sovereign_debt",
        table_name="sovereign_debt",
        params={"mode": "eu_yields"},
        timeout=120,
        retries=2,
    )

    # ── OECD Composite Leading Indicators (country-level leading econ) ──
    # CLI is the broadest single OECD series; covers G7 + major EM by default.
    dag.add(
        "fetch_global_pmi",
        operator="global_pmi",
        table_name="global_pmi",
        params={"mode": "cli"},
        timeout=120,
        retries=2,
    )

    # ── US TIC capital flows (foreign holdings of Treasuries by country) ──
    # Bilateral concentration — who holds whose debt — cross-country edges.
    dag.add(
        "fetch_capital_flows",
        operator="capital_flows",
        table_name="capital_flows",
        params={"mode": "holdings"},
        timeout=120,
        retries=2,
    )

    # ── DeFi TVL (activates dead protocol nodes: Uniswap, Aave, etc.) ──
    dag.add(
        "fetch_defi_flows",
        operator="defi_flows",
        table_name="defi_flows",
        params={"mode": "tvl", "limit": 20},
        timeout=120,
        retries=2,
    )

    # ── Wikipedia pageview spikes (attention leading indicator on topics) ──
    # Densifies the 729 topic nodes beyond GDELT event counts.
    dag.add(
        "fetch_wikipedia_pageviews",
        operator="wikipedia_pageviews",
        table_name="wikipedia_pageviews",
        params={
            "mode": "spike",
            "days_back": 30,
            "z_threshold": 2.0,
            "limit": 50,
        },
        timeout=120,
        retries=2,
    )

    # ── US Senate LDA lobbying filings (company strategic-intent signal) ──
    # Use `search` mode with a year filter — broad coverage of all registrants
    # and clients filing in the current year. `spending` mode requires a
    # specific registrant/client and would narrow coverage to one entity.
    dag.add(
        "fetch_lobbying",
        operator="lobbying",
        table_name="lobbying",
        params={"mode": "search", "year": datetime.now(UTC).year},
        timeout=120,
        retries=2,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 43 — High-Volume Entity DAG Wiring
    # Adds 4 already-L2-ready tools that generate high entity volume per
    # run: vessel entities (500+/run), company/org entities (gov awards),
    # person/company entities (sanctions), company entities (patents).
    # See [[phase43_high_volume_dag_wiring]].
    # ═══════════════════════════════════════════════════════════════

    # ── AIS vessel positions (vessel entity type, 500+/run) ────────
    # Digitraffic Baltic AIS: 18K+ vessel source pool, zero cost/key.
    # area=full_baltic: bbox (54–66°N, 9–31°E). Persists vessel entities
    # with vessel_position obs. timeout=180 covers per-vessel metadata.
    dag.add(
        "fetch_ais_vessel",
        operator="ais_vessel_tracking",
        table_name="ais_vessel_tracking",
        params={"mode": "area_daily_snapshot", "area_name": "full_baltic", "ship_type": "tanker"},
        timeout=180,
        retries=2,
    )

    # ── US federal contract awards (company + organization entities) ──
    # USASpending.gov: latest 100 award records → company + agency nodes
    # with contract_award obs and awarded_by links.
    dag.add(
        "fetch_gov_contracts",
        operator="gov_contracts",
        table_name="gov_contracts",
        params={"mode": "recent", "limit": 100},
        timeout=120,
        retries=2,
    )

    # ── OFAC/UN sanctions designations (person + company entities) ──
    # Full-roster snapshot across OFAC SDN + UN consolidated lists.
    #
    # Was mode="recent", days_back=90, limit=100 until 2026-09-23. That
    # discarded 95% of every fetch: the OFAC SDN CSV carries no per-entry
    # dates, so the date filter dropped all 19,393 OFAC records and only the
    # 11 dated UN rows inside the window could ever persist — which is
    # exactly what the live DB held for this tool's entire life. Worse, those
    # 11 never change, and observed_at came from the source listing date, so
    # every subsequent run hit the store's idempotency key and updated rows
    # instead of inserting: SUCCESS, zero rows, indistinguishable from "no
    # new data". See the module docstring in agent/tools/sanctions_monitor.py.
    #
    # snapshot mode is first-seen gated (steady-state rows == new
    # designations) and always writes one roster observation stamped with
    # collection time, which cannot dedup away.
    #
    # __domain_tables__ makes the executor's zero-rows guard take a real
    # COUNT(*) delta for this node instead of trusting its green checkmark.
    # Only entity_observations is declared: the delta is measured DAG-wide,
    # not per node, and `entities` legitimately gains nothing on a day when
    # no new entity is designated anywhere — declaring it would fail the whole
    # daily run on a normal day, which is how a guard gets ignored.
    dag.add(
        "fetch_sanctions_monitor",
        operator="sanctions_monitor",
        table_name="sanctions_monitor",
        params={
            "mode": "snapshot",
            "__domain_tables__": ["entity_observations"],
        },
        # Two downloads (5.7MB + 2.1MB) plus a first-seen lookup per record.
        # Measured end-to-end on a throwaway store 2026-09-23: 22s for the
        # full 20,404-record backfill, 41s for the steady-state re-run (same
        # fetch, 20k first-seen lookups, 1 row written). 300 is headroom.
        timeout=300,
        retries=2,
    )

    # ── USPTO AI/ML patent filings (company entities) ──────────────
    # PatentsView search mode with CPC class G06N (machine learning).
    # Creates company entities for major tech assignees with patent_filing
    # obs. search is the only mode that calls _persist_entities.
    dag.add(
        "fetch_patent_filings",
        operator="patent_filings",
        table_name="patent_filings",
        params={"mode": "search", "cpc_class": "G06N", "limit": 50},
        timeout=120,
        retries=2,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 44 — Batch 2 Entity DAG Wiring
    # Wires 5 more L2-ready tools: regulatory_gazette, form144,
    # supply_chain_monitor, political_risk, comtrade.
    # No new tool code — all tools already have _persist_entities.
    # Observation types already in schema: regulatory_velocity,
    # sell_intent, price_movement, campaign_finance, trade_flow.
    # See [[phase44_batch2_dag_wiring]].
    # ═══════════════════════════════════════════════════════════════

    # ── Federal Register regulatory filings (org entities) ────────
    dag.add(
        "fetch_regulatory_gazette",
        operator="regulatory_gazette",
        table_name="regulatory_gazette",
        params={"days_back": 7, "limit": 50},
        timeout=120,
        retries=2,
    )

    # ── SEC Form 144 pre-trade intent notices (company + person) ──
    # Form 144 = insider pre-announces intent to sell restricted stock.
    # Signals near-term insider selling pressure 90 days in advance.
    dag.add(
        "fetch_form144",
        operator="form144",
        table_name="form144",
        params={"days_back": 14},
        timeout=180,
        retries=2,
    )

    # ── BLS Producer Price Index (industry-sector org entities) ───
    dag.add(
        "fetch_supply_chain",
        operator="supply_chain_prices",
        table_name="supply_chain_prices",
        params={"mode": "producer_prices"},
        timeout=120,
        retries=2,
    )

    # ── FEC campaign finance — candidate filings (person entities) ─
    # Candidates mode: broadest FEC coverage, creates person entities
    # for candidates with campaign_finance observations.
    dag.add(
        "fetch_political_risk",
        operator="political_risk",
        table_name="political_risk",
        params={"mode": "candidates"},
        timeout=120,
        retries=2,
    )

    # ── UN Comtrade — US top trading partners (country entities) ──
    # partners mode: requires only reporter; returns top bilateral
    # trade flows by value. Creates trade_flow obs on country nodes.
    dag.add(
        "fetch_comtrade",
        operator="comtrade",
        table_name="comtrade",
        params={"mode": "partners", "reporter": "USA"},
        timeout=120,
        retries=2,
    )

    # ── FRED Macro Data (rates, yields, balance sheet) ─────
    dag.add(
        "fetch_macro",
        operator="macro_data",
        table_name="macro_data",
        params={"series_id": "DFF,GS10,GS2,WALCL"},
        timeout=120,
        retries=2,
    )

    # ── Instrument Universe (daily prices) ─────────────────
    from agent.tools.instrument_universe import run_instrument_ingest

    dag.add(
        "fetch_instruments",
        operator=run_instrument_ingest,
        params={"db_path": db_path},
        timeout=300,
        retries=1,
    )

    # ── M15 quant data (options, dividends, US yield curve) ─────
    from agent.tools.dividend_data import run_dividend_ingest
    from agent.tools.options_chain import run_options_chain_ingest

    def run_us_yield_curve_ingest(
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from datetime import datetime

        from agent.pipeline.store import PipelineStore
        from agent.tools.sovereign_debt import SovereignDebtTool

        db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
        month = params.get("month") or datetime.now(tz=UTC).strftime("%Y-%m")
        store = PipelineStore(db_path)
        try:
            tool = SovereignDebtTool(pipeline_store=store)
            result = tool.execute(mode="us_yields", month=month)
            n = 0
            if result.success and result.data:
                n = len(result.data.get("records", []))
            return {"success": result.success, "month": month, "days_persisted": n}
        finally:
            store.close()

    dag.add(
        "fetch_options_chains",
        operator=run_options_chain_ingest,
        params={"db_path": db_path, "include_should": True},
        timeout=180,
        retries=1,
    )
    dag.add(
        "fetch_dividends",
        operator=run_dividend_ingest,
        params={"db_path": db_path, "include_should": True},
        timeout=180,
        retries=1,
    )
    dag.add(
        "fetch_us_yield_curve",
        operator=run_us_yield_curve_ingest,
        params={"db_path": db_path},
        timeout=120,
        retries=2,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 45 — cert_transparency + dns_monitor wiring
    # Both tools monitor FINANCIAL_DOMAINS (20 major bank/broker/
    # exchange/regulator domains). cert_transparency uses a callable
    # operator that iterates per-domain (single-domain API constraint).
    # dns_monitor uses bulk_resolve (native multi-domain support).
    # Domain entities + cert_issued / dns_record observations accumulate
    # daily. See [[phase45_cert_dns_wiring]].
    # ═══════════════════════════════════════════════════════════════

    # ── CT log surveillance — recent cert issuances (domain entities) ─
    # Calls CertTransparencyTool(mode='recent', days_back=30) for each
    # of the 20 FINANCIAL_DOMAINS. Persists domain entities + cert_issued
    # obs. Signals: cert surge = scaling, new subdomain = product launch,
    # issuer switch = security posture change.
    #
    # timeout was 300 while _TIMEOUT was 30 — 20 domains x 2 crt.sh requests
    # cannot fit in either. crt.sh now gets 120s per request, up to 3 attempts,
    # plus a 2.5s inter-domain gap (19 x 2.5 = 48s), so the node needs real
    # headroom. It sits in the parallel layer; a slow node here blocks nothing.
    dag.add(
        "fetch_cert_domains",
        operator=run_cert_domain_collection,
        params={
            "db_path": db_path,
            "domains": FINANCIAL_DOMAINS,
            "days_back": 30,
            "request_delay": 2.5,
        },
        timeout=1200,
        retries=1,
    )

    # ── DNS change monitoring — bulk_resolve for all financial domains ─
    # DNSMonitorTool(mode='bulk_resolve') resolves A/AAAA/MX/NS/TXT/CNAME
    # for all 20 FINANCIAL_DOMAINS in one call. Persists domain entities
    # + dns_record obs. Signals: MX change = email migration, TXT tokens
    # = SaaS adoption, TTL drop = imminent infra change, NS switch = CDN.
    dag.add(
        "fetch_dns_domains",
        operator="dns_monitor",
        table_name="dns_monitor",
        params={"mode": "bulk_resolve", "domains": FINANCIAL_DOMAINS},
        timeout=120,
        retries=2,
    )

    # ══════════════════════════════════════════════════════════════
    # Phase 45.3 — Remaining 23 unwired tools
    # All nodes are independent (no deps); single parallel layer.
    # L2-ready tools (have _persist_entities): entity observations
    # accumulate daily. L1 aggregate tools: global conditioning
    # variables consumed as country/market-level node features.
    # ══════════════════════════════════════════════════════════════

    # ── Academic preprints — research entity tracking ──────────
    dag.add(
        "fetch_academic_preprints",
        operator="academic_preprints",
        table_name="academic_preprints",
        # "papers" mode requires a specific search query (no sensible daily
        # default without a defined watchlist topic); "trending" needs no
        # query — it returns broadly trending arXiv papers, a genuinely
        # useful zero-argument daily signal.
        params={"mode": "trending"},
        timeout=60,
        retries=1,
    )

    # ── Bankruptcy / insolvency — L2 entity observations ───────
    # us_bankruptcy = PACER RSS from 6 major courts (SDNY, Delaware,
    # S.D. Texas, …). Covers ~90% of large corporate Chapter 11.
    dag.add(
        "fetch_bankruptcy_court",
        operator="bankruptcy_court",
        table_name="bankruptcy_court",
        params={"mode": "us_bankruptcy"},
        timeout=60,
        retries=2,
    )

    # ── Building permits — FRED housing cycle indicator ─────────
    dag.add(
        "fetch_building_permits",
        operator="building_permits",
        table_name="building_permits",
        params={"mode": "permits"},
        timeout=60,
        retries=1,
    )

    # ── Consumer sentiment — macro conditioning variable ────────
    dag.add(
        "fetch_consumer_sentiment",
        operator="consumer_sentiment",
        table_name="consumer_sentiment",
        params={"mode": "us_sentiment"},
        timeout=60,
        retries=1,
    )

    # ── Creditor filings — entity stress scan ──────────────────
    dag.add(
        "fetch_creditor_filings",
        operator="creditor_filings",
        table_name="creditor_filings",
        params={"mode": "stress_scan"},
        timeout=90,
        retries=1,
    )

    # ── Disease surveillance — wastewater pathogen tracking ─────
    # CDC NWSS wastewater: pathogen PCR concentrations. Physics
    # that can't be faked. Detects waves 2-3 weeks before hospitals.
    dag.add(
        "fetch_disease_surveillance",
        operator="disease_surveillance",
        table_name="disease_surveillance",
        params={"mode": "wastewater"},
        timeout=60,
        retries=2,
    )

    # ── Drug regulatory — FDA approval entity tracking ──────────
    dag.add(
        "fetch_drug_regulatory",
        operator="drug_regulatory",
        table_name="drug_regulatory",
        params={"mode": "approvals"},
        timeout=60,
        retries=1,
    )

    # ── Earthquake proximity — USGS seismic + infrastructure ───
    dag.add(
        "fetch_earthquake_proximity",
        operator="earthquake_proximity",
        table_name="earthquake_proximity",
        params={"mode": "recent"},
        timeout=45,
        retries=2,
    )

    # ── Electricity monitor — EIA regional demand entities ─────
    dag.add(
        "fetch_electricity_monitor",
        operator="electricity_monitor",
        table_name="electricity_monitor",
        # "region" is required — without it the tool returns success=False and
        # ToolOperator turns that into a RuntimeError, which is why this node
        # has never written a row. PJM is the largest US balancing authority
        # (~65 GW peak, 13 states), so its hourly load is the single best
        # macro-activity proxy of the set this tool can reach.
        params={"mode": "demand", "region": "PJM"},
        timeout=60,
        retries=1,
    )

    # ── Energy supply — EIA petroleum stocks (macro L1) ────────
    dag.add(
        "fetch_energy_supply",
        operator="energy_supply",
        table_name="energy_supply",
        params={"mode": "petroleum_stocks"},
        timeout=60,
        retries=1,
    )

    # ── FOIA requests — entity cluster (L2 entity-level) ───────
    dag.add(
        "fetch_foia_requests",
        operator="foia_requests",
        table_name="foia_requests",
        # Every mode needs a specific target (query, agency, or entity) — no
        # zero-argument mode exists. SEC is the one agency whose FOIA request
        # volume is a direct market signal per this tool's own docstring
        # ("FOIA surge + insider selling + DNS changes = crisis").
        params={"mode": "agency_activity", "agency": "SEC"},
        timeout=90,
        retries=1,
    )

    # ── Food security — FAO production entity tracking ──────────
    dag.add(
        "fetch_food_security",
        operator="food_security",
        table_name="food_security",
        # "country" is required; "WLD" (world aggregate) is the tool's own
        # documented option for a global daily signal, not one arbitrary
        # country.
        params={"mode": "production", "country": "WLD"},
        timeout=60,
        retries=1,
    )

    # ── Grid interconnection queue — energy project entities ────
    dag.add(
        "fetch_interconnection_queue",
        operator="interconnection_queue",
        table_name="interconnection_queue",
        params={"mode": "queue"},
        timeout=90,
        retries=1,
    )

    # ── Internet infrastructure — IODA country-level outages ───
    dag.add(
        "fetch_internet_infrastructure",
        operator="internet_infrastructure",
        table_name="internet_infrastructure",
        params={"mode": "outages"},
        timeout=60,
        retries=2,
    )

    # ── Internet outages — network health entity tracking ───────
    dag.add(
        "fetch_internet_outages",
        operator="internet_outages",
        table_name="internet_outages",
        # "network_health" (RIPE probes) requires a specific country with no
        # global option. "outage_detection" (OONI) works with no country —
        # a global web_connectivity aggregate — the right zero-argument
        # default for unconditional daily collection.
        params={"mode": "outage_detection"},
        timeout=60,
        retries=2,
    )

    # ── Job postings — BLS JOLTS via FRED ──────────────────────
    dag.add(
        "fetch_job_postings",
        operator="job_postings",
        table_name="job_postings",
        params={"mode": "jolts"},
        timeout=60,
        retries=1,
    )

    # ── Labor disruptions — BLS work stoppages (macro L1) ──────
    dag.add(
        "fetch_labor_disruptions",
        operator="labor_disruptions",
        table_name="labor_disruptions",
        params={"mode": "work_stoppages"},
        timeout=45,
        retries=1,
    )

    # ── Migration flows — UNHCR displacement entity tracking ───
    dag.add(
        "fetch_migration_flows",
        operator="migration_flows",
        table_name="migration_flows",
        params={"mode": "displacement"},
        timeout=90,
        retries=1,
    )

    # ── Polymarket whales — smart money wallet tracking ─────────
    dag.add(
        "fetch_polymarket_whales",
        operator="polymarket_whales",
        table_name="polymarket_whales",
        params={"mode": "recent_signals"},
        timeout=60,
        retries=2,
    )

    # ── Satellite activity — NASA FIRMS fire near infrastructure
    # 'area' is REQUIRED: the tool hard-fails without it and the FIRMS area
    # endpoint takes a bbox only ("west,south,east,north"), never a bare
    # country code. This node registered params={"mode": "fire"} alone from
    # the start and therefore never wrote a single row — the operator raised
    # "Parameter 'area' required for fire mode" on every run. Keep the 'area'
    # key here; tests/test_satellite_activity_edge.py asserts it is non-empty.
    dag.add(
        "fetch_satellite_activity",
        operator="satellite_activity",
        table_name="satellite_activity",
        params={
            "mode": "fire",
            "area": "-125,24,-66,50",  # CONUS
            "source": "VIIRS_NOAA20_NRT",
            "days": 1,
        },
        timeout=60,
        retries=2,
    )

    # ── Transport throughput — BTS border crossing entities ─────
    dag.add(
        "fetch_transport_throughput",
        operator="transport_throughput",
        table_name="transport_throughput",
        params={"mode": "recent"},
        timeout=90,
        retries=1,
    )

    # ── Treasury receipts — US Daily Treasury Statement ─────────
    dag.add(
        "fetch_treasury_receipts",
        operator="treasury_receipts",
        table_name="treasury_receipts",
        params={"mode": "cash_balance"},
        timeout=45,
        retries=1,
    )

    # ── Weather alerts — NOAA NWS severe + NASA FIRMS fire ─────
    dag.add(
        "fetch_weather_alerts",
        operator="weather_alerts",
        table_name="weather_alerts",
        params={"mode": "summary"},
        timeout=60,
        retries=2,
    )

    # ── Change 12: Apply tool routing decisions ────────────
    if tool_router is not None:
        decisions = tool_router.decide(tool_context)
        for node_id, node in dag.nodes.items():
            if node_id in decisions:
                node.enabled = decisions[node_id]

    return dag
