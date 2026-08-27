"""Intelligence Brief — deterministic anomaly detection over public data.

Niche decision (owner, 2026-08-27): the brief is POSITIONING & FLOW ANOMALIES
ONLY. The former "Contract Opportunities" section is cut — it ranked
already-awarded USASpending contracts (nothing biddable) by an expected value
that reduced to contract size, because P(win) was a hardcoded, never-learned
Beta prior (0.5/(0.5+1.0) = 0.3333) for every row. See
docs/specs/nineteen_dollar_tier_spec.md Step 2.

What remains is real: z-scores computed over genuine historical baselines
(169-week CFTC COT series, 3,000+ point DeFi TVL series, etc.) combined with
Bayesian Online Changepoint Detection (Adams & MacKay 2007) — a correct
Normal-Inverse-Gamma conjugate model with a Student-t predictive, not a
threshold heuristic. Deterministic math on live public data. No LLM, no
prediction claim.

Usage:
    .venv/bin/python scripts/intelligence_brief.py --anomalies 8
    .venv/bin/python scripts/intelligence_brief.py --md --out brief.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.live_intelligence_digest import build_digest  # noqa: E402

METHODOLOGY = (
    "Deterministic z-score anomaly detection + Bayesian Online Changepoint "
    "Detection (Adams & MacKay 2007) over public positioning and flow data. "
    "No LLM, no prediction — this surfaces statistically anomalous moves "
    "against each series' own historical baseline (per-finding baseline "
    "length shown below). Every number here is reproducible from the source "
    "APIs; nothing is inferred by a model."
)

# Human-readable section headings, grouped by source. Order here is the
# render order (strongest signal sources first is handled by the sort in
# build_digest / fetch_anomalies; this is just section grouping).
SOURCE_LABELS: dict[str, str] = {
    "cftc": "CFTC — Futures Positioning (Commitment of Traders)",
    "defi_flows": "DeFi Flows — Total Value Locked",
    "instrument_universe": "Instrument Universe — Volatility",
    "sovereign_debt": "Sovereign Debt — Bond Yields",
    "polymarket": "Polymarket — Market-Implied Probability",
}

# Human-readable field labels. Falls back to a de-snake_cased field name for
# anything not listed here so a newly wired-up source never renders blank.
FIELD_LABELS: dict[str, str] = {
    "yield_pct": "yield",
    "mm_net": "managed-money net position",
    "open_interest": "open interest",
    "realized_vol_20d": "20-day realized volatility",
    "intraday_range": "intraday range",
    "tvl_usd": "TVL",
    "yes_price": "market-implied probability",
    "volume_24h": "24h volume",
}


def fetch_anomalies(
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_rows: int = 8,
) -> list[dict[str, Any]]:
    """Top current anomalies from real stored observations (digest)."""
    from agent.pipeline.store import PipelineStore

    store = PipelineStore(db_path)
    report = build_digest(store, top_n=max_rows)
    return report["digest"]


def _edition_id(edition_date: str, anomalies: list[dict[str, Any]]) -> str:
    """Stable identifier: same date + same content -> same id.

    Lets a delivery log distinguish real editions from re-deliveries of an
    unchanged run (previously indistinguishable — checksum 5f0d435ed7c9ebc0
    was delivered three times with no date or id anywhere in the payload).
    """
    fingerprint = json.dumps(
        [
            {
                "source": a.get("source"),
                "entity_id": a.get("entity_id"),
                "field": a.get("field"),
                "zscore": a.get("zscore"),
            }
            for a in anomalies
        ],
        sort_keys=True,
    )
    digest = hashlib.sha256(f"{edition_date}|{fingerprint}".encode()).hexdigest()[:12]
    return f"tirra-brief-{edition_date}-{digest}"


def build_brief(
    *,
    anomalies_limit: int = 8,
    db_path: str = ".tirra_pipeline/pipeline.db",
    # Deprecated — accepted-but-ignored so callers this file does not own
    # (scripts/deliver_brief.py, scripts/tirra_engine.py, scripts/run_scheduled.sh)
    # keep working unmodified after the contract-opportunities section was cut.
    # Contract opportunities are gone per the niche decision (see module
    # docstring); remove these call sites' flags once those files are updated.
    contracts_limit: int = 10,
    learner_path: str = ".tirra_opportunities/win_learner.jsonl",
    max_contract_rows: int = 5,
) -> dict[str, Any]:
    """Build the anomaly-only intelligence brief."""
    del contracts_limit, learner_path, max_contract_rows  # deprecated, unused

    anomalies = fetch_anomalies(db_path=db_path, max_rows=anomalies_limit)
    now = datetime.now(UTC)
    edition_date = now.strftime("%Y-%m-%d")
    edition_id = _edition_id(edition_date, anomalies)

    return {
        "brief_type": "intelligence",
        "edition_date": edition_date,
        "edition_id": edition_id,
        "generated_at_utc": now.isoformat(),
        "methodology": METHODOLOGY,
        "live_anomalies": anomalies,
    }


def _weeks_ago_text(weeks: float | None) -> str:
    if weeks is None:
        return ""
    if weeks < 1:
        return "this week"
    if weeks < 2:
        return "last week"
    return f"{int(round(weeks))} weeks ago"


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))


def _render_finding(a: dict[str, Any]) -> list[str]:
    name = a.get("entity_name") or a["entity_id"]
    field_label = _field_label(a["field"])
    direction_word = "above" if a.get("direction", "up" if a["zscore"] >= 0 else "down") == "up" else "below"
    n_points = a.get("n_points", 0)

    lines = [f"- **{name}** · {field_label}"]
    detail = f"    z={a['zscore']:+.2f} {direction_word} its {n_points}-point baseline"
    if a.get("changepoint"):
        weeks_text = _weeks_ago_text(a.get("changepoint_weeks_ago"))
        detail += f" · structural break {weeks_text}" if weeks_text else " · structural break detected"
    lines.append(detail)
    return lines


def render_markdown(brief: dict[str, Any]) -> str:
    lines: list[str] = ["# TirraMind Intelligence Brief", ""]
    lines.append(f"**Edition:** {brief.get('edition_date', 'unknown')} · `{brief.get('edition_id', 'unknown')}`")
    lines.append("")
    lines.append(f"_{brief.get('methodology', METHODOLOGY)}_")
    lines.append("")

    anomalies = brief.get("live_anomalies", [])
    if not anomalies:
        lines.append("## Positioning & Flow Anomalies")
        lines.append("")
        lines.append("_(none currently flagged — z-score below threshold across all scored sources)_")
        lines.append("")
        return "\n".join(lines)

    # Group by source, preserving the strength-ranked order within each group.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for a in anomalies:
        by_source.setdefault(a["source"], []).append(a)

    # Stable section order: known sources first (in SOURCE_LABELS order),
    # then anything unrecognized so a newly wired-up source is never dropped.
    ordered_sources = [s for s in SOURCE_LABELS if s in by_source]
    ordered_sources += [s for s in by_source if s not in SOURCE_LABELS]

    for source in ordered_sources:
        heading = SOURCE_LABELS.get(source, source)
        lines.append(f"## {heading}")
        lines.append("")
        for a in by_source[source]:
            lines.extend(_render_finding(a))
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Intelligence brief (positioning & flow anomalies)")
    parser.add_argument("--anomalies", type=int, default=8, help="anomalies to include")
    parser.add_argument("--out", type=str, default=None, help="write JSON to path")
    parser.add_argument("--md", action="store_true", help="render markdown too")
    parser.add_argument("--db", type=str, default=".tirra_pipeline/pipeline.db")
    args = parser.parse_args()

    brief = build_brief(anomalies_limit=args.anomalies, db_path=args.db)

    if args.out:
        Path(args.out).write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote brief -> {args.out}")
    else:
        print(json.dumps(brief, indent=2, ensure_ascii=False))

    if args.md:
        md_path = args.out + ".md" if args.out else None
        text = render_markdown(brief)
        if md_path:
            Path(md_path).write_text(text, encoding="utf-8")
            print(f"Wrote markdown -> {md_path}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
