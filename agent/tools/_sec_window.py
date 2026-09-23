"""Date-window helper for the SEC EDGAR collectors (Layer 1).

Both ``form144`` and ``insider_filings`` query EDGAR full-text search over a
``[startdt, enddt]`` range, but until now both hardcoded ``enddt = today``.
That made historical backfill impossible: EDGAR returns newest-first, so
asking for a longer ``days_back`` just re-fetched the same recent page.

Measured 2026-09-23 against the live API:

    1-day window  (2026-09-22)  → total=87    filings
    30-day window               → total=3,282 filings

So the window has to move, not just widen — and it has to stay narrow enough
that the collector's own ``page_from >= 500`` cap doesn't silently truncate
the result. ~3-day windows sit around 260 hits, comfortably inside it.

Silent truncation is the risk this module exists to make visible: a window
that returns exactly the cap looks identical to one that returned everything.
Callers should treat ``hits == cap`` as "this window was too wide", not as
success.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

__all__ = ["resolve_end_date", "window_is_truncated", "EDGAR_PAGE_CAP"]

# The collector's own pagination stop (``page_from >= 500``). Not an EDGAR
# limit — EDGAR paginates well past this — but raising it would make each call
# slower without making it complete, because every hit costs one extra HTTP
# request to fetch its XML. Narrow windows are the correct fix.
EDGAR_PAGE_CAP = 500


def resolve_end_date(end_date: str | date | None) -> date:
    """Return the window's end date, defaulting to today.

    Accepts ``""``/``None`` (meaning "now"), a ``date``, or an ISO
    ``YYYY-MM-DD`` string.

    A future end date is clamped to today. This is not politeness — a window
    ending in the future is the shape of F-04 (data leakage): it would let a
    backfill pull filings dated after the window it claims to represent, and
    every downstream split derives its boundaries from observed_at.
    """
    if end_date is None or end_date == "":
        return date.today()

    if isinstance(end_date, datetime):
        resolved = end_date.date()
    elif isinstance(end_date, date):
        resolved = end_date
    else:
        try:
            resolved = datetime.strptime(str(end_date).strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"end_date must be ISO YYYY-MM-DD, got {end_date!r}") from exc

    today = date.today()
    return today if resolved > today else resolved


def window_is_truncated(hits: int, cap: int = EDGAR_PAGE_CAP) -> bool:
    """True when a window returned exactly the cap, i.e. it was too wide.

    Use this rather than trusting a full-looking result: a truncated window
    and a complete one are indistinguishable from the row count alone, and
    treating the first as the second is how a backfill silently loses days.
    """
    return hits >= cap


def walk_back(
    *,
    oldest: date,
    newest: date,
    window_days: int,
) -> list[tuple[date, date]]:
    """Yield ``(start, end)`` windows from *newest* back to *oldest*.

    Newest-first because EDGAR is newest-first: if a long backfill is
    interrupted, the data you already have is contiguous with the present
    rather than stranded in the past.
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if oldest > newest:
        raise ValueError(f"oldest ({oldest}) is after newest ({newest})")

    windows: list[tuple[date, date]] = []
    cursor = newest
    while cursor >= oldest:
        start = max(oldest, cursor - timedelta(days=window_days - 1))
        windows.append((start, cursor))
        cursor = start - timedelta(days=1)
    return windows


# ── Paginator ────────────────────────────────────────────────────────────


class EdgarTruncated(RuntimeError):
    """Raised when EDGAR reported more hits than we managed to collect.

    Deliberately an exception rather than a log line. The bug this replaces
    returned a partial page as a successful result: a 3-day window in March
    2026 held 360 filings, a transient 5xx on page 2 ended the loop, and the
    caller received 99 rows with ``success=True``. Losing 72% of a window is
    indistinguishable from a quiet week unless somebody raises.
    """


def fetch_edgar_hits(
    client,
    url: str,
    *,
    forms: str,
    start_dt: date,
    end_dt: date,
    delay: float,
    page_size: int = 100,
    max_hits: int = EDGAR_PAGE_CAP,
    sleep=None,
    strict: bool = True,
) -> list[dict]:
    """Page through EDGAR full-text search, retrying transient failures.

    EDGAR intermittently answers 5xx mid-pagination and rate-limits with 429.
    Both are retried with backoff; only a genuinely exhausted result set or a
    non-transient status ends the walk.

    ``strict=True`` raises :class:`EdgarTruncated` when fewer hits were
    collected than EDGAR reported available, so a caller can narrow the window
    rather than quietly accept a partial answer. Pass ``strict=False`` only
    when a partial page is genuinely acceptable to the caller.
    """
    naptime = sleep or time.sleep
    all_hits: list[dict] = []
    page_from = 0
    reported_total = 0
    attempts = 0
    MAX_ATTEMPTS_PER_PAGE = 4

    while True:
        naptime(delay)
        params = {
            "forms": forms,
            "dateRange": "custom",
            "startdt": str(start_dt),
            "enddt": str(end_dt),
            "from": str(page_from),
            "size": str(page_size),
        }
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:  # noqa: BLE001 — re-raised below unless transient
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = status == 429 or (status is not None and status >= 500)
            attempts += 1
            if transient and attempts < MAX_ATTEMPTS_PER_PAGE:
                naptime(min(2.0 * attempts, 8.0))
                continue
            if transient:
                raise EdgarTruncated(
                    f"EDGAR kept returning {status} at offset {page_from} for "
                    f"{start_dt}..{end_dt}; collected {len(all_hits)} of "
                    f"{reported_total or 'unknown'} hits"
                ) from exc
            raise

        attempts = 0
        hits = result.get("hits", {}).get("hits", [])
        reported_total = result.get("hits", {}).get("total", {}).get("value", 0) or reported_total
        if not hits:
            break

        all_hits.extend(hits)
        page_from += page_size
        if page_from >= reported_total or page_from >= max_hits:
            break

    if strict and reported_total and len(all_hits) < min(reported_total, max_hits):
        raise EdgarTruncated(
            f"EDGAR reported {reported_total} hits for {start_dt}..{end_dt} but only "
            f"{len(all_hits)} were collected — narrow the window and retry"
        )
    return all_hits
