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
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EMPTY_ANOMALIES_NOTE = "No anomalies crossed the z-score threshold across any scored source this edition."

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.brief_sections import (  # noqa: E402
    attach_sections,
    render_credit_stress_section,
    render_insider_sell_intent_section,
    render_wow_diff_section,
)
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
#
# "polymarket" removed (2026-08-29): market_probability tops out at 15
# points/entity across all 1,493 entities — a hard structural ceiling below
# the 20-point z-score floor, not a quiet source. See _SCORABLE in
# live_intelligence_digest.py for the measured detail. Dropped from
# _SCORABLE entirely, so this label would never render; removed here too so
# the two files can't drift.
SOURCE_LABELS: dict[str, str] = {
    "cftc": "CFTC — Futures Positioning (Commitment of Traders)",
    "defi_flows": "DeFi Flows — Total Value Locked",
    "instrument_universe": "Instrument Universe — Volatility & Volume",
    "sovereign_debt": "Sovereign Debt — Bond Yields",
}

# Human-readable field labels. Falls back to a de-snake_cased field name for
# anything not listed here so a newly wired-up source never renders blank
# (degrades to "conc top4 long" instead of breaking) — but real labels read
# far better than the de-snake_cased fallback, so anything we actually score
# gets a real one here.
FIELD_LABELS: dict[str, str] = {
    "yield_pct": "yield",
    "mm_net": "managed-money net position",
    "open_interest": "open interest",
    "realized_vol_20d": "20-day realized volatility",
    "intraday_range": "intraday range",
    "volume": "trading volume",
    "avg_volume_20d": "20-day average volume",
    "tvl_usd": "TVL",
    # CFTC COT — the seven fields beyond mm_net/open_interest (2026-08-29).
    # mm_net is the number anyone can read off the public CFTC site; these
    # are not.
    "swap_net": "swap-dealer net position",
    "pm_net": "producer/merchant net position",
    "conc_top4_long": "top-4 long concentration",
    "conc_top4_short": "top-4 short concentration",
    "mm_net_pct_oi": "managed-money net (% of open interest)",
    "mm_weekly_flow": "managed-money weekly flow",
    "oi_change": "open interest, weekly change",
}


def fetch_report(
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_rows: int = 8,
) -> dict[str, Any]:
    """Full digest report (findings + which sources were actually queryable).

    Prefer this over `fetch_anomalies` when the caller needs to render an
    honest "N of M sources responded" note — see `build_brief`. Raises if
    every scorable source failed (see `build_digest`'s docstring): that is
    deliberately NOT swallowed into an empty, falsely-quiet-looking report.
    """
    from agent.pipeline.store import PipelineStore

    store = PipelineStore(db_path)
    return build_digest(store, top_n=max_rows)


def fetch_anomalies(
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_rows: int = 8,
) -> list[dict[str, Any]]:
    """Top current anomalies from real stored observations (digest)."""
    return fetch_report(db_path=db_path, max_rows=max_rows)["digest"]


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
    """Build the anomaly-only intelligence brief.

    Uses `fetch_report` (not the thinner `fetch_anomalies`) so a partial
    outage — some scored sources unreachable this run, others fine — can be
    disclosed honestly in the delivered brief rather than silently producing
    an edition that looks identical to "every source checked, all quiet."
    `fetch_report`/`build_digest` already raise rather than return if EVERY
    source failed, so reaching this line means at least a partial, real read.
    """
    del contracts_limit, learner_path, max_contract_rows  # deprecated, unused

    report = fetch_report(db_path=db_path, max_rows=anomalies_limit)
    anomalies = report["digest"]
    now = datetime.now(UTC)
    edition_date = now.strftime("%Y-%m-%d")
    edition_id = _edition_id(edition_date, anomalies)

    brief: dict[str, Any] = {
        "brief_type": "intelligence",
        "edition_date": edition_date,
        "edition_id": edition_id,
        "generated_at_utc": now.isoformat(),
        "methodology": METHODOLOGY,
        "live_anomalies": anomalies,
    }

    sources_failed = report.get("sources_failed") or []
    if sources_failed:
        surface_scored = report.get("surface_scored", len(sources_failed))
        brief["degraded_sources"] = sources_failed
        brief["degraded_note"] = (
            f"{len(sources_failed)} of {surface_scored} scored sources could not be queried this "
            "edition (see degraded_sources) — findings above reflect only the sources that responded."
        )
    if not anomalies:
        brief["anomalies_note"] = _DEFAULT_EMPTY_ANOMALIES_NOTE

    # Insider sell intent, credit stress, and the week-over-week anomaly diff
    # (2026-08-29, $29 tier pass) — see scripts/brief_sections.py for the
    # fetch logic and the section contract each renderer satisfies.
    brief = attach_sections(brief, db_path=db_path)

    return brief


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


def _render_finding(a: dict[str, Any]) -> list[str] | None:
    """Render one finding's markdown lines, or None if the row is too
    malformed to render honestly.

    Was direct subscript access (`a["field"]`, `a["zscore"]` — twice, once
    inside a `.get(..., default=...)` call whose default expression Python
    evaluates unconditionally regardless of whether "direction" was even
    present) — a single malformed row (missing key, non-numeric zscore) threw
    an uncaught KeyError/TypeError all the way up through `render_markdown`
    and killed the entire weekly delivery, not just that one line. All access
    below is defensive; a row that can't be rendered honestly is logged and
    skipped rather than taking the rest of the brief down with it.
    """
    zscore = a.get("zscore")
    if not isinstance(zscore, int | float) or isinstance(zscore, bool):
        logger.warning("[brief] skipping malformed finding (non-numeric zscore): %r", a)
        return None

    entity_id = a.get("entity_id", "unknown-entity")
    name = a.get("entity_name") or entity_id
    field_label = _field_label(a.get("field", "unknown_field"))
    direction = a.get("direction") or ("up" if zscore >= 0 else "down")
    direction_word = "above" if direction == "up" else "below"
    n_points = a.get("n_points", 0)

    lines = [f"- **{name}** · {field_label}"]
    detail = f"    z={zscore:+.2f} {direction_word} its {n_points}-point baseline"
    if a.get("changepoint"):
        weeks_text = _weeks_ago_text(a.get("changepoint_weeks_ago"))
        detail += f" · structural break {weeks_text}" if weeks_text else " · structural break detected"
    lines.append(detail)
    return lines


def _render_anomalies_section(brief: dict[str, Any]) -> list[str] | None:
    """The "Positioning & Flow Anomalies" section.

    Returns markdown lines including its own heading and trailing blank line.
    Never returns None — an anomaly-free edition is a genuine, honest result
    that must still say so (see `_DEFAULT_EMPTY_ANOMALIES_NOTE`), not silently
    vanish. This is one section among several composed by `render_markdown`;
    it must never gate whether sections after it render (see the "structural
    blocker" note on `render_markdown`).
    """
    lines: list[str] = ["## Positioning & Flow Anomalies", ""]

    anomalies = brief.get("live_anomalies", [])
    if not anomalies:
        note = (
            brief.get("anomalies_note")
            or "(none currently flagged — z-score below threshold across all scored sources)"
        )
        lines.append(f"_{note}_" if not note.startswith("_") else note)
        lines.append("")
        return lines

    # Group by source, preserving the strength-ranked order within each group.
    # a.get(...), not a["source"] — one malformed row (missing key) must not
    # take the whole render down; it's bucketed under "unknown" instead.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for a in anomalies:
        if not isinstance(a, dict):
            logger.warning("[brief] skipping non-dict finding entry: %r", a)
            continue
        by_source.setdefault(a.get("source", "unknown"), []).append(a)

    # Stable section order: known sources first (in SOURCE_LABELS order),
    # then anything unrecognized so a newly wired-up source is never dropped.
    ordered_sources = [s for s in SOURCE_LABELS if s in by_source]
    ordered_sources += [s for s in by_source if s not in SOURCE_LABELS]

    for source in ordered_sources:
        heading = SOURCE_LABELS.get(source, source)
        lines.append(f"### {heading}")
        lines.append("")
        for a in by_source[source]:
            rendered = _render_finding(a)
            if rendered is None:
                continue  # malformed row — logged in _render_finding, skipped here
            lines.extend(rendered)
        lines.append("")

    return lines


# Registry a new section plugs into. Contract for adding a section (see
# module docstring / render_markdown for the full write-up):
#
#   def _render_my_section(brief: dict[str, Any]) -> list[str] | None:
#       ...
#
# - Signature: takes the full `brief` dict (whatever `build_brief` returned,
#   e.g. `live_anomalies`, `degraded_sources`, plus any new key your build
#   step adds), returns EITHER a list of markdown lines (your own "## Heading"
#   first, a trailing "" blank line last) OR None to render nothing this
#   edition (e.g. no data this run) — returning `[]` also renders nothing,
#   both are valid "skip" signals.
# - Registers by appending to `_SECTION_RENDERERS` below, in the order you
#   want it to appear. Order in this list IS render order.
# - Renders unconditionally: `render_markdown` calls every entry in this list
#   after the header regardless of what any other section contains. No
#   section may gate whether a later section renders — that was the bug
#   fixed 2026-08-29 (an anomaly-free edition used to `return` before any
#   section after it ran, silently dropping everything added later on
#   exactly the quiet week when it mattered most).
_SECTION_RENDERERS: list[Any] = [
    _render_anomalies_section,
    render_wow_diff_section,
    render_insider_sell_intent_section,
    render_credit_stress_section,
]


def render_markdown(brief: dict[str, Any]) -> str:
    """Render the full brief as markdown by composing registered sections.

    STRUCTURAL FIX (2026-08-29): this used to render the anomalies section
    and, if `live_anomalies` was empty, `return` immediately — any section
    appended after it in this function was silently dropped on precisely the
    quiet week when extra sections matter most (nothing else to fill the
    edition with). Sections are now independent entries in
    `_SECTION_RENDERERS`, each rendering (or skipping) on its own; no section
    can gate another. See `_SECTION_RENDERERS` for the contract a new section
    must satisfy.
    """
    lines: list[str] = ["# TirraMind Intelligence Brief", ""]
    lines.append(f"**Edition:** {brief.get('edition_date', 'unknown')} · `{brief.get('edition_id', 'unknown')}`")
    lines.append("")
    lines.append(f"_{brief.get('methodology', METHODOLOGY)}_")
    lines.append("")

    degraded_note = brief.get("degraded_note")
    if degraded_note:
        lines.append(f"> ⚠ {degraded_note}")
        lines.append("")

    for renderer in _SECTION_RENDERERS:
        section_lines = renderer(brief)
        if section_lines:
            lines.extend(section_lines)

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
