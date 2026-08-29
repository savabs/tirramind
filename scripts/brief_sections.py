"""Brief tier — three additional sections: insider sell intent, credit
stress, and a week-over-week anomaly diff.

Registers into `scripts/intelligence_brief.py`'s `_SECTION_RENDERERS` (see
that module's docstring for the section contract). This file owns:
  - the data fetches (`fetch_insider_sell_intent`, `fetch_credit_stress`,
    `compute_wow_diff`) that populate new keys on the `brief` dict, and
  - the pure renderers (`render_insider_sell_intent_section`,
    `render_credit_stress_section`, `render_wow_diff_section`) that turn
    those keys into markdown.
`attach_sections()` is the single entry point `intelligence_brief.build_brief`
calls to populate the brief dict; the renderers never touch the database —
they only read whatever `attach_sections` already put on the brief.

DO NOT change extraction logic in scripts/live_intelligence_digest.py. These
three sections are cross-sectional / event-listing, not z-score series, so
they read entity_observations directly here rather than going through
`_extract_series`/`_SCORABLE` (which discards non-numeric payload fields
these sections need — ticker, company, urgency, relationship, items[]).

--- Insider Sell Intent (form144/sell_intent) ------------------------------
Measured 2026-08-29 against the real DB: 1,195 deduped rows (dedupe on
(entity_id, observed_at), max(rowid) wins — identical rule to
live_intelligence_digest._extract_series), 950 entities, max 9 points/entity.
CRITICAL: this is CROSS-SECTIONAL. 9 points/entity is below the 20-point
z-score floor everywhere in this codebase — the anomaly-detection path is
structurally the wrong tool here, full stop, not a "make the window shorter"
fix. There is nothing to detect a change *against* per entity; the signal is
in ranking issuers against EACH OTHER in the current week, so this aggregates
to issuer-week totals instead of scoring a time series.

Fill rate: of the deduped rows, only rows with dollar_value > 0 AND
urgency == "immediate" are counted (measured: 555/1,195 = 46% overall,
consistent with the 47% cited in the ground truth — the other 638 rows are
zero-value/blank filings that would understate every issuer's total if
summed in). "Zero dollar_value" was measured to fully account for the
excluded rows (no immediate-urgency row has a null dollar_value that isn't
also zero) — filtering on `> 0` alone captures the same set as "non-null +
immediate" would.

entity_links has 953 works_for edges connecting form144 filer -> issuer
(measured, confirmed). NOT used for the ticker/company/issuer mapping here:
the sell_intent payload already carries a clean ticker + company name
directly (e.g. "Apple Inc." / "AAPL"), so joining through entity_links would
just re-derive what's already in the row. It IS used implicitly for filer
identity: the *observation's own* entity_id is the filer (a person entity),
and store.get_entity() on that id already resolves to their canonical_name
without needing the entity_links join — entity_links exists for OTHER
consumers that don't have the payload in hand, which isn't our situation
here. Evaluated per the brief, not blindly joined for the sake of joining.

--- Credit Stress (creditor_filings/creditor_filing) -----------------------
16 rows total, 16 entities, 2.5 days stale (measured). is_stress_signal is
True on all 16 — pre-filtered at collection, so the flag itself carries zero
discrimination; every stored row already IS a stress event. The actual signal
is WHICH SEC 8-K item codes co-occur (1.01 = material definitive agreement,
2.03 = creation of a direct financial obligation, etc.) — item codes alone
are unreadable to a buyer, so each is rendered with its plain-English SEC
meaning via ITEM_CODE_LABELS.

Only 16 rows ever, spanning 2026-07-29 to 2026-08-25 measured — genuinely
possible for a given week to have zero. Per the brief: this section must
render an honest "no filings this week" rather than being omitted, so
`fetch_credit_stress` always returns a dict (never None) once the store
opened successfully; only a store-open failure propagates to `None` (handled
in `attach_sections`).

--- Week-over-week diff -----------------------------------------------------
Keys anomalies by (source, entity_id, field) across editions, using the
dated archive scripts/tirra_engine.py:192-251 (_archive_delivery) already
writes to `.tirra_delivery/archive/intelligence_brief_<UTC-date>.json`.

Two real caveats handled, not ignored:
  1. Same-day reruns OVERWRITE the dated file (measured: index.jsonl has
     three 2026-08-27 rows with three different checksums pointing at one
     file). Selecting the prior edition by filename date and requiring
     STRICTLY `< current edition_date` means a same-day rerun can never be
     mistaken for "the prior edition" — there is exactly one file per date
     regardless of how many times it was overwritten, and today's own date
     is excluded by construction, not by deduping index.jsonl rows.
  2. Only two editions exist as of 2026-08-29, so "is there a prior edition"
     must degrade gracefully on the first-ever run (and the second run,
     which only has one prior). `compute_wow_diff` returns
     `{"first_edition": True, ...}` whenever no archive file with a strictly
     earlier date exists — including "archive dir doesn't exist yet" and
     "directory exists but is empty" — and the renderer says so honestly
     instead of raising or fabricating an empty diff that looks like "no
     changes" (a very different claim from "nothing to compare against").
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# SEC Form 8-K item codes that appear in creditor_filings payloads, with their
# plain-English meaning. Not exhaustive of every possible 8-K item — only
# codes actually observed need to render with a label; anything unseen falls
# back to the bare code (see _item_label) rather than breaking.
ITEM_CODE_LABELS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def _item_label(code: str) -> str:
    return ITEM_CODE_LABELS.get(code, code)


def _display_company_name(canonical_name: str) -> str:
    """Best-effort human-readable name from a canonical_name that embeds a
    ticker and " cik <digits>" suffix (e.g.
    "credit acceptance cacc cik 0000885550" — measured shape of every
    creditor_filings entity's canonical_name). Strips the " cik ..." suffix
    and title-cases the rest. The embedded ticker token is left in place
    (not guessed out): there is no reliable boundary between "company name"
    and "ticker" in the raw string, and a wrong guess reads worse than a
    slightly redundant ticker.
    """
    lower = canonical_name.lower()
    idx = lower.find(" cik ")
    name = canonical_name[:idx] if idx != -1 else canonical_name
    name = name.strip()
    return name.title() if name else canonical_name


def _dedupe_raw_rows(store: Any, source: str, obs_type: str) -> list[tuple[str, float, dict[str, Any]]]:
    """(entity_id, observed_at, value_dict) rows, deduped on
    (entity_id, observed_at) via max(rowid) — same dedupe rule as
    live_intelligence_digest._extract_series (collection re-ingests the same
    report without an upsert). Kept separate from that function because it
    needs the FULL payload dict (ticker, company, urgency, items[], ...), not
    just a caller-selected set of numeric fields.
    """
    cur = store._conn.cursor()
    rows = cur.execute(
        "select entity_id, observed_at, value_json from entity_observations "
        "where source_tool=? and observation_type=? "
        "and rowid in ("
        "  select max(rowid) from entity_observations"
        "  where source_tool=? and observation_type=?"
        "  group by entity_id, observed_at"
        ") order by observed_at asc",
        (source, obs_type, source, obs_type),
    ).fetchall()
    out: list[tuple[str, float, dict[str, Any]]] = []
    for entity_id, ts, value_json in rows:
        try:
            value = json.loads(value_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        out.append((entity_id, float(ts or 0.0), value))
    return out


# ---------------------------------------------------------------------------
# Insider Sell Intent (form144 / sell_intent)
# ---------------------------------------------------------------------------


def fetch_insider_sell_intent(store: Any, top_n: int = 8, now: datetime | None = None) -> dict[str, Any]:
    """Issuer-week ranking of declared Form 144 sell intent.

    Cross-sectional by construction (see module docstring) — never scored as
    a per-entity z-score series. Filters to the CURRENT ISO week (of `now`)
    so the section reflects a real weekly cadence rather than "all history
    ever collected"; if the current week has zero qualifying filings, the
    returned dict still has `top_issuers: []` and the renderer says so
    honestly rather than the section vanishing.
    """
    now = now or datetime.now(UTC)
    year, week, _ = now.isocalendar()
    week_start = date.fromisocalendar(year, week, 1)
    week_end = date.fromisocalendar(year, week, 7)

    rows = _dedupe_raw_rows(store, "form144", "sell_intent")

    total_rows_this_week = 0
    qualifying: list[tuple[str, dict[str, Any]]] = []
    for entity_id, ts, value in rows:
        d = datetime.fromtimestamp(ts, tz=UTC).date()
        if not (week_start <= d <= week_end):
            continue
        total_rows_this_week += 1
        dollar_value = value.get("dollar_value")
        urgency = value.get("urgency")
        if (
            isinstance(dollar_value, int | float)
            and not isinstance(dollar_value, bool)
            and dollar_value > 0
            and urgency == "immediate"
        ):
            qualifying.append((entity_id, value))

    result: dict[str, Any] = {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_rows_this_week": total_rows_this_week,
        "qualifying_count": len(qualifying),
        "total_declared_usd": 0.0,
        "top_issuers": [],
    }
    if not qualifying:
        return result

    agg: dict[str, float] = defaultdict(float)
    company: dict[str, str] = {}
    count: dict[str, int] = defaultdict(int)
    # ticker -> (largest single dollar_value, relationship, filer entity_id)
    top_single: dict[str, tuple[float, str, str]] = {}
    for entity_id, value in qualifying:
        ticker = value.get("ticker") or "UNKNOWN"
        dv = float(value["dollar_value"])
        agg[ticker] += dv
        company.setdefault(ticker, value.get("company") or ticker)
        count[ticker] += 1
        prev = top_single.get(ticker)
        if prev is None or dv > prev[0]:
            top_single[ticker] = (dv, value.get("relationship") or "", entity_id)

    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    top_issuers = []
    for ticker, total in ranked:
        biggest_dv, biggest_rel, biggest_entity = top_single[ticker]
        filer_name = None
        try:
            row = store.get_entity(biggest_entity)
            if row and row.get("canonical_name"):
                filer_name = row["canonical_name"]
        except Exception:
            pass
        top_issuers.append(
            {
                "ticker": ticker,
                "company": company[ticker],
                "total_declared_usd": round(total, 2),
                "filing_count": count[ticker],
                "largest_single_filing_usd": round(biggest_dv, 2),
                "largest_single_filer_name": filer_name,
                "largest_single_filer_relationship": biggest_rel,
            }
        )

    result["total_declared_usd"] = round(sum(agg.values()), 2)
    result["top_issuers"] = top_issuers
    return result


def render_insider_sell_intent_section(brief: dict[str, Any]) -> list[str] | None:
    """Section contract renderer — see intelligence_brief._SECTION_RENDERERS."""
    data = brief.get("insider_sell_intent")
    if not data:
        return None  # source never collected / DB unreachable this run — not "quiet"

    lines = ["## Insider Sell Intent — Form 144", ""]
    lines.append(
        f"_Declared intent to sell restricted/control stock, week of "
        f"{data['week_start']} to {data['week_end']} — SEC Form 144. This is "
        "cross-sectional (ranked by issuer against each other this week), not "
        "a per-entity anomaly score._"
    )
    lines.append("")

    if not data.get("top_issuers"):
        lines.append(
            f"_No qualifying declared-sale filings this week "
            f"({data.get('total_rows_this_week', 0)} raw Form 144 filing(s) seen, "
            "none with both a non-zero dollar value and `urgency: immediate`)._"
        )
        lines.append("")
        return lines

    lines.append(
        f"**${data['total_declared_usd']:,.0f}** declared across "
        f"**{data['qualifying_count']}** filing(s) this week "
        f"({data['qualifying_count']}/{data['total_rows_this_week']} raw filings "
        "cleared the dollar-value + urgency filter)."
    )
    lines.append("")
    for issuer in data["top_issuers"]:
        lines.append(
            f"- **{issuer['company']}** ({issuer['ticker']}) · "
            f"${issuer['total_declared_usd']:,.0f} across {issuer['filing_count']} filing(s)"
        )
        filer_bits = [
            b for b in (issuer.get("largest_single_filer_name"), issuer.get("largest_single_filer_relationship")) if b
        ]
        detail = f"    largest single filing ${issuer['largest_single_filing_usd']:,.0f}"
        if filer_bits:
            detail += f" ({' — '.join(filer_bits)})"
        lines.append(detail)
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Credit Stress (creditor_filings / creditor_filing)
# ---------------------------------------------------------------------------


def fetch_credit_stress(store: Any, now: datetime | None = None) -> dict[str, Any]:
    """8-K filings pre-flagged as credit-stress signals, current ISO week.

    Always returns a dict (never None once the store opened) — 16 rows ever
    means a given week genuinely can be empty, and per the brief this must
    render honestly rather than the section disappearing.
    """
    now = now or datetime.now(UTC)
    year, week, _ = now.isocalendar()
    week_start = date.fromisocalendar(year, week, 1)
    week_end = date.fromisocalendar(year, week, 7)

    rows = _dedupe_raw_rows(store, "creditor_filings", "creditor_filing")

    filings = []
    for entity_id, ts, value in rows:
        file_date_str = value.get("file_date")
        fd: date | None
        try:
            fd = date.fromisoformat(file_date_str) if file_date_str else datetime.fromtimestamp(ts, tz=UTC).date()
        except (ValueError, TypeError):
            fd = None
        if fd is None or not (week_start <= fd <= week_end):
            continue

        entity_name = entity_id
        try:
            row = store.get_entity(entity_id)
            if row and row.get("canonical_name"):
                entity_name = _display_company_name(row["canonical_name"])
        except Exception:
            pass

        items = value.get("items") or []
        if not isinstance(items, list):
            items = []
        filings.append(
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "cik": value.get("cik"),
                "form": value.get("form") or "8-K",
                "file_date": file_date_str,
                "items": items,
                "item_labels": [_item_label(c) for c in items],
            }
        )

    filings.sort(key=lambda f: f.get("file_date") or "", reverse=True)

    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "filing_count": len(filings),
        "filings": filings,
    }


def render_credit_stress_section(brief: dict[str, Any]) -> list[str] | None:
    """Section contract renderer — see intelligence_brief._SECTION_RENDERERS."""
    data = brief.get("credit_stress")
    if not data:
        return None  # DB unreachable this run

    lines = ["## Credit Stress — 8-K Filings", ""]
    lines.append(
        f"_Material-event 8-K filings flagged as credit-stress signals, week of "
        f"{data['week_start']} to {data['week_end']}. Every stored row is "
        "pre-filtered as a stress event — the signal is in which item codes "
        "co-occur, not in whether the row exists._"
    )
    lines.append("")

    if not data.get("filings"):
        lines.append("_No credit-stress 8-K filings this week._")
        lines.append("")
        return lines

    for f in data["filings"]:
        lines.append(f"- **{f['entity_name']}** · {f['form']} filed {f['file_date']}")
        item_bits = [f"{code} ({label})" for code, label in zip(f["items"], f["item_labels"], strict=True)]
        lines.append(f"    items: {', '.join(item_bits) if item_bits else 'none listed'}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Week-over-week anomaly diff
# ---------------------------------------------------------------------------

_ARCHIVE_FILENAME_RE = re.compile(r"^intelligence_brief_(\d{4}-\d{2}-\d{2})\.json$")


def _anomaly_key(a: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (a.get("source"), a.get("entity_id"), a.get("field"))


def compute_wow_diff(
    archive_dir: str,
    current_edition_date: str,
    current_anomalies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Diff this edition's live_anomalies against the most recent PRIOR
    edition on disk, keyed by (source, entity_id, field).

    See module docstring for the two caveats this handles: same-day
    overwrites can't be mistaken for a prior edition (strict `<` on the
    filename date), and a missing/empty archive degrades to
    `{"first_edition": True, ...}` rather than raising or faking a diff.
    """
    archive_path = Path(archive_dir)
    empty: dict[str, Any] = {
        "first_edition": True,
        "prior_edition_date": None,
        "new": [],
        "resolved": [],
        "held": [],
    }
    if not archive_path.is_dir():
        return empty

    candidates: list[tuple[str, Path]] = []
    for p in archive_path.glob("intelligence_brief_*.json"):
        m = _ARCHIVE_FILENAME_RE.match(p.name)
        if m:
            candidates.append((m.group(1), p))

    prior_candidates = [(d, p) for d, p in candidates if d < current_edition_date]
    if not prior_candidates:
        return empty

    prior_date, prior_path = max(prior_candidates, key=lambda dp: dp[0])
    try:
        prior_brief = json.loads(prior_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[brief_sections] could not read prior edition %s: %s", prior_path, exc)
        return empty

    prior_anomalies = prior_brief.get("live_anomalies") or []
    prior_map = {_anomaly_key(a): a for a in prior_anomalies if isinstance(a, dict)}
    current_map = {_anomaly_key(a): a for a in current_anomalies if isinstance(a, dict)}

    new_keys = current_map.keys() - prior_map.keys()
    resolved_keys = prior_map.keys() - current_map.keys()
    held_keys = current_map.keys() & prior_map.keys()

    return {
        "first_edition": False,
        "prior_edition_date": prior_date,
        "new": [current_map[k] for k in new_keys],
        "resolved": [prior_map[k] for k in resolved_keys],
        "held": [current_map[k] for k in held_keys],
    }


def render_wow_diff_section(brief: dict[str, Any]) -> list[str] | None:
    """Section contract renderer — see intelligence_brief._SECTION_RENDERERS."""
    wow = brief.get("wow_diff")
    if wow is None:
        return None  # archive unreadable this run

    lines = ["## Week-over-Week — Anomaly Changes", ""]

    if wow.get("first_edition"):
        lines.append("_No prior edition on record yet — nothing to compare this edition against._")
        lines.append("")
        return lines

    prior_date = wow.get("prior_edition_date")
    new_ = wow.get("new", [])
    resolved = wow.get("resolved", [])
    held = wow.get("held", [])

    lines.append(f"_Compared against the {prior_date} edition (same source/entity/field triple)._")
    lines.append("")

    if not new_ and not resolved and not held:
        lines.append("_Neither edition had any anomalies to compare._")
        lines.append("")
        return lines

    lines.append(f"**{len(new_)} new** · **{len(resolved)} resolved** · **{len(held)} held**")
    lines.append("")

    for label, group in (("New", new_), ("Resolved", resolved), ("Held", held)):
        if not group:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for item in group:
            entity_name = item.get("entity_name") or item.get("entity_id", "unknown-entity")
            source = item.get("source", "unknown")
            field = item.get("field", "unknown_field")
            zscore = item.get("zscore")
            z_text = f" (z={zscore:+.2f})" if isinstance(zscore, int | float) and not isinstance(zscore, bool) else ""
            lines.append(f"- **{entity_name}** · {source}/{field}{z_text}")
        lines.append("")

    return lines


def attach_sections(
    brief: dict[str, Any],
    *,
    db_path: str = ".tirra_pipeline/pipeline.db",
    archive_dir: str = ".tirra_delivery/archive",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Populate `brief` with the data these three sections render from.

    Mutates and returns `brief`. Each fetch is isolated (same isolation
    pattern as build_digest's per-source try/except in
    live_intelligence_digest.py) — one section's data failing to fetch must
    not take the other two, or the anomalies section above them, down with
    it. A failed fetch sets that key to `None`, which the corresponding
    renderer treats as "nothing to render this edition" per the section
    contract — not the same claim as "checked, found nothing" (which is
    represented by a real dict with empty lists), but the safest fallback
    when the underlying read itself couldn't be trusted.
    """
    now = now or datetime.now(UTC)

    store = None
    try:
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path)
    except Exception as exc:  # pragma: no cover - defensive, mirrors build_digest's isolation
        logger.warning("[brief_sections] could not open store at %s: %s", db_path, exc)

    if store is not None:
        try:
            brief["insider_sell_intent"] = fetch_insider_sell_intent(store, now=now)
        except Exception as exc:
            logger.warning("[brief_sections] insider_sell_intent fetch failed: %s", exc)
            brief["insider_sell_intent"] = None
        try:
            brief["credit_stress"] = fetch_credit_stress(store, now=now)
        except Exception as exc:
            logger.warning("[brief_sections] credit_stress fetch failed: %s", exc)
            brief["credit_stress"] = None
    else:
        brief["insider_sell_intent"] = None
        brief["credit_stress"] = None

    try:
        brief["wow_diff"] = compute_wow_diff(
            archive_dir,
            brief.get("edition_date", ""),
            brief.get("live_anomalies", []),
        )
    except Exception as exc:
        logger.warning("[brief_sections] wow_diff compute failed: %s", exc)
        brief["wow_diff"] = None

    return brief
