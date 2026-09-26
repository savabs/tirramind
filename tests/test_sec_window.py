"""Tests for agent/tools/_sec_window.py — the SEC EDGAR window + paginator.

The bug these guard (2026-09-23): ``form144._fetch_recent_144s`` paged through
EDGAR full-text search and, on a transient 5xx, ``break``-ed out of the loop
and returned whatever it had collected **as a successful result**. A 3-day
window in March 2026 held 360 filings; a 5xx on page 2 meant the caller got 99
and no error. 72% of the window vanished, and a truncated window is
indistinguishable from a quiet one by row count alone.

So the tests that matter here are not "does it paginate" — the old code
paginated fine on the happy path. They are:
    1. a transient 5xx mid-walk must be retried, not swallowed
    2. a persistently short result must RAISE, never return quietly
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent.tools._sec_window import (
    EDGAR_PAGE_CAP,
    EdgarTruncated,
    fetch_edgar_hits,
    resolve_end_date,
    walk_back,
    window_is_truncated,
)

# ── Fakes ──────────────────────────────────────────────────────


class _Resp:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _HTTPError(self)

    def json(self):
        return self._payload


class _HTTPError(Exception):
    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def _page(hits: int, total: int) -> dict:
    return {"hits": {"hits": [{"_id": f"h{i}"} for i in range(hits)], "total": {"value": total}}}


class _FakeClient:
    """Serves a scripted sequence of responses and records the offsets asked for."""

    def __init__(self, script):
        self._script = list(script)
        self.offsets: list[int] = []

    def get(self, url, params=None):
        self.offsets.append(int((params or {}).get("from", 0)))
        if not self._script:
            return _Resp(200, _page(0, 0))
        item = self._script.pop(0)
        return item() if callable(item) else item


def _call(client, **kw):
    return fetch_edgar_hits(
        client,
        "https://efts.example/search",
        forms="144",
        start_dt=date(2026, 3, 18),
        end_dt=date(2026, 3, 20),
        delay=0,
        sleep=lambda _s: None,  # no real waiting in tests
        **kw,
    )


# ── resolve_end_date ───────────────────────────────────────────


class TestResolveEndDate:
    def test_empty_means_today(self):
        assert resolve_end_date("") == date.today()

    def test_none_means_today(self):
        assert resolve_end_date(None) == date.today()

    def test_iso_string(self):
        assert resolve_end_date("2026-03-20") == date(2026, 3, 20)

    def test_date_passthrough(self):
        assert resolve_end_date(date(2025, 1, 2)) == date(2025, 1, 2)

    def test_future_is_clamped_to_today(self):
        """A window ending in the future is the shape of F-04 (leakage)."""
        assert resolve_end_date("2099-01-01") == date.today()

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            resolve_end_date("March 20th")


# ── walk_back ──────────────────────────────────────────────────


class TestWalkBack:
    def test_windows_are_contiguous_and_cover_the_range(self):
        w = walk_back(oldest=date(2026, 9, 1), newest=date(2026, 9, 10), window_days=3)
        assert w[0][1] == date(2026, 9, 10), "must start at the newest date"
        assert w[-1][0] == date(2026, 9, 1), "must reach the oldest date"
        # every day covered exactly once
        days = [d for s, e in w for d in _days(s, e)]
        assert sorted(days) == _days(date(2026, 9, 1), date(2026, 9, 10))
        assert len(days) == len(set(days)), "windows must not overlap"

    def test_newest_first(self):
        w = walk_back(oldest=date(2026, 1, 1), newest=date(2026, 1, 9), window_days=3)
        assert [s for s, _ in w] == sorted([s for s, _ in w], reverse=True)

    def test_single_day_range(self):
        assert walk_back(oldest=date(2026, 5, 5), newest=date(2026, 5, 5), window_days=7) == [
            (date(2026, 5, 5), date(2026, 5, 5))
        ]

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="after"):
            walk_back(oldest=date(2026, 5, 5), newest=date(2026, 1, 1), window_days=3)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="window_days"):
            walk_back(oldest=date(2026, 1, 1), newest=date(2026, 1, 5), window_days=0)


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# ── The actual bug ─────────────────────────────────────────────


class TestTransientFailuresAreRetriedNotSwallowed:
    """The 2026-09-23 regression, reproduced.

    Old behaviour: 5xx on page 2 → break → return 100 of 360 with no error.
    """

    def test_a_5xx_mid_walk_is_retried_and_the_window_completes(self):
        client = _FakeClient(
            [
                _Resp(200, _page(100, 360)),  # page 1 OK
                _Resp(503),  # page 2 fails transiently  <-- old code stopped here
                _Resp(200, _page(100, 360)),  # retry of page 2 succeeds
                _Resp(200, _page(100, 360)),  # page 3
                _Resp(200, _page(60, 360)),  # page 4
            ]
        )
        hits = _call(client)
        assert len(hits) == 360, (
            "a transient 5xx truncated the window — this is the bug that lost "
            "72% of 2026-03-18..2026-03-20 while reporting success"
        )
        assert client.offsets == [0, 100, 100, 200, 300], "must retry the SAME offset, not skip it"

    def test_429_is_retried(self):
        client = _FakeClient([_Resp(200, _page(100, 200)), _Resp(429), _Resp(200, _page(100, 200))])
        assert len(_call(client)) == 200

    def test_persistent_5xx_raises_rather_than_returning_partial(self):
        client = _FakeClient([_Resp(200, _page(100, 360))] + [_Resp(503)] * 6)
        with pytest.raises(EdgarTruncated, match="503"):
            _call(client)

    def test_short_result_raises_even_without_an_error(self):
        """EDGAR claims 360, serves 100, then goes empty. Silence is not success."""
        client = _FakeClient([_Resp(200, _page(100, 360)), _Resp(200, _page(0, 360))])
        with pytest.raises(EdgarTruncated, match="only 100"):
            _call(client)

    def test_non_transient_status_propagates(self):
        client = _FakeClient([_Resp(404)])
        with pytest.raises(_HTTPError):
            _call(client)

    def test_strict_false_allows_a_partial_page(self):
        client = _FakeClient([_Resp(200, _page(100, 360)), _Resp(200, _page(0, 360))])
        assert len(_call(client, strict=False)) == 100

    def test_complete_small_window_does_not_raise(self):
        client = _FakeClient([_Resp(200, _page(87, 87))])
        assert len(_call(client)) == 87

    def test_hitting_the_cap_is_not_treated_as_short(self):
        """At the cap we stop deliberately — that is truncation, but not an error."""
        pages = [_Resp(200, _page(100, 5000)) for _ in range(EDGAR_PAGE_CAP // 100)]
        hits = _call(_FakeClient(pages))
        assert len(hits) == EDGAR_PAGE_CAP
        assert window_is_truncated(len(hits)), "caller must be able to detect this and narrow"
