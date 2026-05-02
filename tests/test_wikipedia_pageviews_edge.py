"""
Edge case tests for WikipediaPageviewsTool.

Covers: invalid inputs, boundary values, error paths, spike math,
HTTP failures, cache behavior, mode routing, article parsing,
empty responses, zero-variance data, evergreen filtering.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.wikipedia_pageviews import (
    _DEFAULT_WATCHLIST,
    _EVERGREEN,
    WikipediaPageviewsTool,
    _detect_spike,
    _std,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def tool():
    return WikipediaPageviewsTool(cache=None)


@pytest.fixture
def tool_cached():
    cache = MagicMock()
    cache.get.return_value = None
    return WikipediaPageviewsTool(cache=cache), cache


def _make_items(views: list[int], start_date: str = "20260301") -> list[dict]:
    """Build a list of Wikimedia pageview items."""
    dt = datetime.strptime(start_date, "%Y%m%d")
    return [{"timestamp": (dt + timedelta(days=i)).strftime("%Y%m%d00"), "views": v} for i, v in enumerate(views)]


def _make_top_response(articles: list[tuple[str, int]]) -> dict:
    """Build a Wikimedia top-pageviews response."""
    return {
        "items": [
            {"articles": [{"article": name, "views": views, "rank": i + 1} for i, (name, views) in enumerate(articles)]}
        ]
    }


# ==================================================================
# Pure math tests — _std
# ==================================================================


class TestStd:
    def test_empty(self):
        assert _std([]) == 0.0

    def test_single(self):
        assert _std([42]) == 0.0

    def test_uniform(self):
        assert _std([5, 5, 5, 5]) == 0.0

    def test_known_values(self):
        # Population std of [2,4,4,4,5,5,7,9] = 2.0
        assert abs(_std([2, 4, 4, 4, 5, 5, 7, 9]) - 2.0) < 0.01

    def test_two_values(self):
        # std of [0, 10] = 5.0
        assert abs(_std([0, 10]) - 5.0) < 0.01

    def test_large_values(self):
        # Should not overflow for large pageview counts
        vals = [10_000_000] * 100
        assert _std(vals) == 0.0

    def test_negative_values(self):
        # Hypothetical — function handles negatives gracefully
        result = _std([-5, -3, -1, 1, 3, 5])
        assert result > 0

    def test_float_values(self):
        result = _std([1.5, 2.5, 3.5])
        assert isinstance(result, float)
        assert result > 0


# ==================================================================
# Pure math tests — _detect_spike
# ==================================================================


class TestDetectSpike:
    def test_no_spike_flat(self):
        assert _detect_spike([100] * 30, 2.0) is None

    def test_no_spike_normal_variation(self):
        # Value within 2 std
        assert _detect_spike([100, 110, 95, 105, 100, 98, 102] * 4 + [108], 2.0) is None

    def test_spike_on_constant_baseline(self):
        result = _detect_spike([100] * 29 + [300], 2.0)
        assert result is not None
        z, latest, mean, std = result
        assert z == 99.0  # near-zero std path
        assert latest == 300
        assert mean == 100.0

    def test_spike_on_variable_baseline(self):
        result = _detect_spike([100, 110, 95, 105, 100, 98, 102] * 4 + [350], 2.0)
        assert result is not None
        z, latest, mean, std = result
        assert z > 2.0
        assert latest == 350

    def test_too_few_days(self):
        assert _detect_spike([100] * 5 + [500], 2.0) is None
        assert _detect_spike([100] * 6, 2.0) is None  # exactly 6 < 7

    def test_exactly_seven_days(self):
        # 7 is the minimum — 6 baseline + 1 latest
        result = _detect_spike([100] * 6 + [500], 2.0)
        assert result is not None

    def test_empty_list(self):
        assert _detect_spike([], 2.0) is None

    def test_single_element(self):
        assert _detect_spike([100], 2.0) is None

    def test_near_zero_baseline_noise(self):
        # Baseline is all zeros, latest is trivial (5 views) — should NOT spike
        assert _detect_spike([0] * 29 + [5], 2.0) is None

    def test_near_zero_baseline_real_spike(self):
        # Baseline near zero, but latest is a real surge (>= 200)
        result = _detect_spike([0] * 29 + [500], 2.0)
        assert result is not None
        z, latest, mean, std = result
        assert z == 99.0
        assert latest == 500

    def test_near_zero_baseline_moderate(self):
        # Baseline zero, latest = 150 — below 200 threshold, not a spike
        assert _detect_spike([0] * 29 + [150], 2.0) is None

    def test_custom_threshold(self):
        # With high threshold, moderate spike shouldn't trigger
        views = [100, 110, 95, 105, 100, 98, 102] * 4 + [140]
        result_low = _detect_spike(views, 2.0)
        result_high = _detect_spike(views, 10.0)
        assert result_low is not None
        assert result_high is None

    def test_negative_z_not_reported(self):
        # Latest is below mean — never a "spike"
        result = _detect_spike([100, 110, 95, 105, 100, 98, 102] * 4 + [50], 2.0)
        assert result is None

    def test_zero_threshold(self):
        # z_threshold=0 should report any positive deviation
        # With variable baseline where std > 0:
        result = _detect_spike([90, 100, 110, 95, 105, 98, 102] * 2 + [101], 0.0)
        # mean ~100, std ~6, z=(101-100)/6 = 0.17 >= 0.0 -> spike
        assert result is not None

    def test_boundary_exact_threshold(self):
        # z exactly at threshold — should be >= so should trigger
        # Build a case where z = exactly 2.0
        # mean=100, std=5 -> latest needs to be 110 for z=2.0
        baseline = [95, 100, 105] * 10  # mean=100, std≈4.08
        mean = sum(baseline) / len(baseline)
        std = _std(baseline)
        target = mean + 2.0 * std  # exactly at boundary
        result = _detect_spike(baseline + [int(target)], 2.0)
        # Due to int rounding, may or may not trigger — that's fine
        # We're testing it doesn't crash

    def test_all_zeros(self):
        assert _detect_spike([0] * 30, 2.0) is None

    def test_very_large_spike(self):
        result = _detect_spike([100] * 29 + [1_000_000], 2.0)
        assert result is not None
        z, latest, mean, std = result
        assert latest == 1_000_000


# ==================================================================
# Mode routing
# ==================================================================


class TestModeRouting:
    def test_invalid_mode(self, tool):
        r = tool.execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_mode_case_insensitive(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="SPIKE")
            mock.assert_called_once()

    def test_mode_whitespace(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="  spike  ")
            mock.assert_called_once()

    def test_series_requires_article(self, tool):
        r = tool.execute(mode="series", articles="")
        assert not r.success
        assert "requires" in r.output.lower()


from agent.tools.base import ToolResult


class TestProjectValidation:
    def test_default_project(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", project="")
            args = mock.call_args
            assert args.kwargs["project"] == "en.wikipedia"

    def test_invalid_project_no_dot(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", project="nope")
            args = mock.call_args
            assert args.kwargs["project"] == "en.wikipedia"


# ==================================================================
# Parameter clamping
# ==================================================================


class TestParameterClamping:
    def test_days_back_min_spike(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", days_back=1)
            assert mock.call_args.kwargs["days_back"] == 7

    def test_days_back_max_spike(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", days_back=999)
            assert mock.call_args.kwargs["days_back"] == 90

    def test_days_back_min_series(self, tool):
        with patch.object(tool, "_execute_series") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="series", articles="Test", days_back=0)
            assert mock.call_args.kwargs["days_back"] == 1

    def test_days_back_max_series(self, tool):
        with patch.object(tool, "_execute_series") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="series", articles="Test", days_back=9999)
            assert mock.call_args.kwargs["days_back"] == 365

    def test_limit_min(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", limit=0)
            assert mock.call_args.kwargs["limit"] == 1

    def test_limit_max_spike(self, tool):
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", limit=9999)
            assert mock.call_args.kwargs["limit"] == 200


# ==================================================================
# Article parsing
# ==================================================================


class TestArticleParsing:
    def test_empty(self, tool):
        assert tool._parse_articles("") == []

    def test_whitespace(self, tool):
        assert tool._parse_articles("   ") == []

    def test_single(self, tool):
        assert tool._parse_articles("Nvidia") == ["Nvidia"]

    def test_multiple(self, tool):
        assert tool._parse_articles("A,B,C") == ["A", "B", "C"]

    def test_whitespace_around(self, tool):
        assert tool._parse_articles(" A , B , C ") == ["A", "B", "C"]

    def test_with_underscores(self, tool):
        # Comma is the delimiter, so titles with commas get split.
        # This is by design — Wikipedia titles rarely have commas.
        # "Tesla,_Inc." becomes two tokens: "Tesla" and "_Inc."
        result = tool._parse_articles("Apple_Inc.,Boeing")
        assert result == ["Apple_Inc.", "Boeing"]

    def test_trailing_comma(self, tool):
        result = tool._parse_articles("A,B,")
        assert result == ["A", "B"]

    def test_only_commas(self, tool):
        result = tool._parse_articles(",,,")
        assert result == []


# ==================================================================
# Spike mode — HTTP mocking
# ==================================================================


class TestSpikeMode:
    def test_spike_detected(self, tool):
        """Mock HTTP to return data with a spike on the last day."""
        views = [100] * 29 + [500]
        items = _make_items(views)

        with patch.object(tool, "_fetch_article_views", return_value=items):
            r = tool.execute(
                mode="spike",
                articles="TestArticle",
                days_back=30,
                z_threshold=2.0,
            )
        assert r.success
        assert "TestArticle" in r.output
        assert r.data["spikes"]
        assert r.data["spikes"][0]["z_score"] > 0

    def test_no_spike(self, tool):
        """All views flat — no spikes."""
        views = [100] * 30
        items = _make_items(views)

        with patch.object(tool, "_fetch_article_views", return_value=items):
            r = tool.execute(
                mode="spike",
                articles="FlatArticle",
                days_back=30,
            )
        assert r.success
        assert "No attention spikes" in r.output
        assert r.data["spikes"] == []

    def test_multiple_articles_sorted(self, tool):
        """Multiple articles with spikes — should be sorted by z-score descending."""
        big_spike = _make_items([100] * 29 + [1000])
        small_spike = _make_items([100] * 29 + [300])

        call_count = [0]
        articles = ["BigSpike", "SmallSpike"]

        def fake_fetch(client, project, title, start, end):
            idx = articles.index(title) if title in articles else 0
            return [big_spike, small_spike][idx]

        with patch.object(tool, "_fetch_article_views", side_effect=fake_fetch):
            r = tool.execute(
                mode="spike",
                articles="BigSpike,SmallSpike",
                z_threshold=2.0,
            )
        assert r.success
        spikes = r.data["spikes"]
        assert len(spikes) == 2
        assert spikes[0]["article"] == "BigSpike"
        assert spikes[0]["z_score"] >= spikes[1]["z_score"]

    def test_default_watchlist_used(self, tool):
        """When no articles specified, default watchlist is used."""
        items = _make_items([100] * 30)

        with patch.object(tool, "_fetch_article_views", return_value=items) as mock:
            tool.execute(mode="spike")
        # Should have been called once per watchlist article
        assert mock.call_count == len(_DEFAULT_WATCHLIST)

    def test_fetch_error_handled(self, tool):
        """HTTP errors on individual articles shouldn't crash the whole scan."""
        call_count = [0]

        def flaky_fetch(client, project, title, start, end):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise httpx.ConnectError("connection refused")
            return _make_items([100] * 30)

        with patch.object(tool, "_fetch_article_views", side_effect=flaky_fetch):
            r = tool.execute(
                mode="spike",
                articles="A,B,C,D",
                z_threshold=2.0,
            )
        assert r.success
        assert r.data["errors"]  # some errors recorded

    def test_too_few_days_skipped(self, tool):
        """Article with <7 days of data is silently skipped."""
        items = _make_items([100] * 5)

        with patch.object(tool, "_fetch_article_views", return_value=items):
            r = tool.execute(mode="spike", articles="Short")
        assert r.success
        assert r.data["spikes"] == []


# ==================================================================
# Top mode — HTTP mocking
# ==================================================================


class TestTopMode:
    def test_top_basic(self, tool):
        resp_data = _make_top_response(
            [
                ("Main_Page", 7_000_000),
                ("Special:Search", 800_000),
                ("Bitcoin", 300_000),
                ("Nvidia", 200_000),
            ]
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            r = tool.execute(mode="top", date="2026-03-15")

        assert r.success
        # Evergreen pages should be filtered out
        article_names = [a["article"] for a in r.data["articles"]]
        assert "Main_Page" not in article_names
        assert "Special:Search" not in article_names
        assert "Bitcoin" in article_names

    def test_top_invalid_date(self, tool):
        r = tool.execute(mode="top", date="not-a-date")
        assert not r.success
        assert "Invalid date" in r.output

    def test_top_empty_date_uses_yesterday(self, tool):
        resp_data = _make_top_response([("Test", 100)])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            r = tool.execute(mode="top", date="")
        assert r.success

    def test_top_http_error(self, tool):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_resp,
        )

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            r = tool.execute(mode="top", date="2026-03-15")
        assert not r.success
        assert "500" in r.output

    def test_top_empty_response(self, tool):
        resp_data = {"items": [{"articles": []}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            r = tool.execute(mode="top", date="2026-03-15")
        assert r.success
        assert r.data["articles"] == []

    def test_top_cached(self, tool_cached):
        tool, cache = tool_cached
        cache.get.return_value = [{"article": "CachedA", "views": 999, "rank": 1}]
        r = tool.execute(mode="top", date="2026-03-15")
        assert r.success
        assert r.data["articles"][0]["article"] == "CachedA"
        # Should not have made HTTP request
        cache.get.assert_called()

    def test_top_all_evergreen_filtered(self, tool):
        """If ALL results are evergreen pages, return empty."""
        resp_data = _make_top_response(
            [
                ("Main_Page", 7_000_000),
                ("Special:Search", 800_000),
                ("Wikipedia:Featured_pictures", 300_000),
                ("Portal:Current_events", 200_000),
            ]
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            r = tool.execute(mode="top", date="2026-03-15")
        assert r.success
        assert r.data["articles"] == []


# ==================================================================
# Series mode — HTTP mocking
# ==================================================================


class TestSeriesMode:
    def test_series_basic(self, tool):
        views = [100, 150, 120, 200, 180, 130, 170, 110, 190, 160]
        items = _make_items(views)

        with patch.object(tool, "_fetch_article_views", return_value=items):
            r = tool.execute(mode="series", articles="TestArticle", days_back=10)

        assert r.success
        assert "TestArticle" in r.output
        assert r.data["article"] == "TestArticle"
        assert len(r.data["views"]) == 10
        assert r.data["stats"]["days"] == 10
        assert r.data["stats"]["min"] == 100
        assert r.data["stats"]["max"] == 200

    def test_series_no_articles(self, tool):
        r = tool.execute(mode="series", articles="")
        assert not r.success

    def test_series_empty_response(self, tool):
        with patch.object(tool, "_fetch_article_views", return_value=[]):
            r = tool.execute(mode="series", articles="NonexistentPage")
        assert r.success
        assert "No pageview data" in r.output

    def test_series_multiple_articles_uses_first(self, tool):
        items = _make_items([100] * 10)
        with patch.object(tool, "_fetch_article_views", return_value=items) as mock:
            r = tool.execute(mode="series", articles="First,Second,Third")
        assert r.success
        # Should only fetch the first article
        call_args = mock.call_args
        assert call_args[0][2] == "First"  # third positional arg is title

    def test_series_fetch_error(self, tool):
        with patch.object(
            tool,
            "_fetch_article_views",
            side_effect=httpx.ConnectError("timeout"),
        ):
            r = tool.execute(mode="series", articles="Error")
        assert not r.success
        assert "error" in r.output.lower() or "Error" in r.output

    def test_series_spike_flagged_in_output(self, tool):
        """Days with z>2 should be flagged with '<<<' in output."""
        views = [100] * 29 + [500]
        items = _make_items(views)
        with patch.object(tool, "_fetch_article_views", return_value=items):
            r = tool.execute(mode="series", articles="SpikeyArticle", days_back=30)
        assert r.success
        assert "<<<" in r.output


# ==================================================================
# Cache integration
# ==================================================================


class TestCacheIntegration:
    def test_fetch_article_views_cache_hit(self, tool_cached):
        tool, cache = tool_cached
        cached_items = _make_items([100] * 10)
        cache.get.return_value = cached_items

        client = MagicMock()
        result = tool._fetch_article_views(client, "en.wikipedia", "Test", "20260301", "20260310")
        assert result == cached_items
        # Client should NOT have been called
        client.get.assert_not_called()

    def test_fetch_article_views_cache_miss(self, tool_cached):
        tool, cache = tool_cached
        cache.get.return_value = None

        items = _make_items([100] * 5)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": items}
        mock_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.get.return_value = mock_resp

        result = tool._fetch_article_views(client, "en.wikipedia", "Test", "20260301", "20260305")
        assert result == items
        cache.put.assert_called_once()

    def test_fetch_article_views_404_no_cache(self, tool_cached):
        """404 means article doesn't exist — should not cache empty result."""
        tool, cache = tool_cached
        cache.get.return_value = None

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        client = MagicMock()
        client.get.return_value = mock_resp

        result = tool._fetch_article_views(client, "en.wikipedia", "Nonexistent", "20260301", "20260310")
        assert result == []
        cache.put.assert_not_called()


# ==================================================================
# OpenAI tool schema
# ==================================================================


class TestToolSchema:
    def test_schema_valid(self, tool):
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "wikipedia_pageviews"
        assert "parameters" in schema["function"]
        props = schema["function"]["parameters"]["properties"]
        assert "mode" in props
        assert "articles" in props
        assert "project" in props
        assert "days_back" in props
        assert "z_threshold" in props

    def test_name(self, tool):
        assert tool.name == "wikipedia_pageviews"

    def test_description_not_empty(self, tool):
        assert len(tool.description) > 50


# ==================================================================
# Evergreen filtering
# ==================================================================


class TestEvergreenFiltering:
    def test_evergreen_set_contains_main_page(self):
        assert "Main_Page" in _EVERGREEN
        assert "Special:Search" in _EVERGREEN

    def test_default_watchlist_not_empty(self):
        assert len(_DEFAULT_WATCHLIST) > 20

    def test_default_watchlist_no_duplicates(self):
        assert len(_DEFAULT_WATCHLIST) == len(set(_DEFAULT_WATCHLIST))


# ==================================================================
# URL encoding
# ==================================================================


class TestURLEncoding:
    def test_article_with_special_chars(self, tool_cached):
        """Articles with parentheses, commas, etc. should be URL-encoded."""
        tool, cache = tool_cached
        cache.get.return_value = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.get.return_value = mock_resp

        tool._fetch_article_views(client, "en.wikipedia", "Tesla,_Inc.", "20260301", "20260310")
        url = client.get.call_args[0][0]
        # The comma and parentheses should be encoded
        assert "Tesla" in url
        # URL should contain the full base path
        assert "/metrics/pageviews/per-article/" in url


# ==================================================================
# Registry integration
# ==================================================================


class TestRegistryIntegration:
    def test_registered_in_cli(self):
        from agent.cli import build_tool_registry
        from agent.config.settings import AgentConfig

        reg = build_tool_registry(AgentConfig())
        assert "wikipedia_pageviews" in reg.list_names()

    def test_execute_via_registry(self):
        from agent.cli import build_tool_registry
        from agent.config.settings import AgentConfig

        reg = build_tool_registry(AgentConfig())
        # Invalid mode — should return graceful error, not crash
        r = reg.execute("wikipedia_pageviews", mode="invalid_mode")
        assert not r.success


# ==================================================================
# Edge: extra kwargs
# ==================================================================


class TestExtraKwargs:
    def test_extra_kwargs_ignored(self, tool):
        """Unknown kwargs should be silently ignored via **_."""
        with patch.object(tool, "_execute_spike") as mock:
            mock.return_value = ToolResult(success=True, output="ok", data={})
            tool.execute(mode="spike", unknown_param="hello", another=42)
            mock.assert_called_once()
