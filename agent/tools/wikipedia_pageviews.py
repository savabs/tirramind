"""
Tool: Wikipedia Pageviews — Global Attention Anomaly Detection

Wikimedia REST API — zero cost, no auth, global coverage across 300+ language editions.
User-Agent header required (Wikimedia enforces this).

Three modes:
  spike   — Detect anomalous attention spikes across a list of tracked articles.
            Computes z-scores against 30-day trailing mean/std. A z-score > 2 on
            a company Wikipedia page means "someone knows something" — interest
            surge precedes news by 1-3 days.
  top     — Top trending articles for a given date. Surface what the world is
            looking at right now. Filter out evergreen pages (Main_Page, etc.).
  series  — Raw daily pageview timeseries for a single article. Feed into
            spectral analysis, changepoint detection, or correlation with price.

API docs: https://wikimedia.org/api/rest_v1/
Rate limits: ~200 req/s (undocumented, very generous). We add 50ms politeness delay.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_BASE = "https://wikimedia.org/api/rest_v1"
_UA = "TirraMind/0.1 (research; https://github.com/tirramind)"
_TIMEOUT = 12
_POLITENESS_DELAY = 0.05  # 50ms between requests

# Pages that always dominate Top and are never interesting
_EVERGREEN = frozenset(
    {
        "Main_Page",
        "Special:Search",
        "Wikipedia:Featured_pictures",
        "-",
        "Special:RecentChanges",
        "Wikipedia:Main_Page",
        "Special:Watchlist",
        "Special:CreateAccount",
        "Wikipedia",
        "Portal:Current_events",
    }
)

# Default watchlist: major companies, geopolitical entities, commodities.
# The user can override this, but this provides out-of-the-box signal.
_DEFAULT_WATCHLIST = [
    # US mega-cap
    "Apple_Inc.",
    "Microsoft",
    "Alphabet_Inc.",
    "Amazon_(company)",
    "Meta_Platforms",
    "Nvidia",
    "Tesla_(company)",
    "Berkshire_Hathaway",
    # Global mega-cap
    "TSMC",
    "Samsung",
    "Alibaba_Group",
    "Tencent",
    "ASML",
    "Toyota",
    "LVMH",
    "Novo_Nordisk",
    # Finance / crypto
    "Bitcoin",
    "Ethereum",
    "JPMorgan_Chase",
    "Goldman_Sachs",
    "BlackRock",
    "Binance",
    # Defense / geopolitical
    "Lockheed_Martin",
    "Raytheon_Technologies",
    "BAE_Systems",
    "NATO",
    "BRICS",
    # Energy / commodities
    "Saudi_Aramco",
    "ExxonMobil",
    "Chevron_Corporation",
    "Lithium",
    "Uranium",
    "Rare-earth_element",
    # Pharma / health
    "Pfizer",
    "Moderna",
    "Eli_Lilly_and_Company",
    # Banks (stress detection)
    "Silicon_Valley_Bank",
    "Credit_Suisse",
    "Deutsche_Bank",
]


class WikipediaPageviewsTool(Tool):
    name = "wikipedia_pageviews"
    description = (
        "Detect attention anomalies via Wikipedia pageview data. "
        "Mode 'spike' checks a watchlist of company/entity articles for unusual "
        "traffic surges (z-score > 2) — a leading indicator that someone knows "
        "something before news breaks. Mode 'top' returns trending articles. "
        "Mode 'series' returns raw daily pageview timeseries for one article. "
        "Global coverage: 300+ language editions. Zero cost, no API key."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["spike", "top", "series"],
                "default": "spike",
                "description": (
                    "spike = detect attention anomalies across watchlist. "
                    "top = trending articles for a date. "
                    "series = raw daily views for one article."
                ),
            },
            "articles": {
                "type": "string",
                "description": (
                    "Comma-separated Wikipedia article titles (underscore for "
                    "spaces). Used in 'spike' mode to override default watchlist, "
                    "or in 'series' mode for the target article. "
                    "Example: 'Nvidia,Tesla,_Inc.,Boeing'"
                ),
                "default": "",
            },
            "project": {
                "type": "string",
                "description": (
                    "Wikimedia project. Default: en.wikipedia. "
                    "Use ja.wikipedia, de.wikipedia, zh.wikipedia etc. for "
                    "non-English coverage."
                ),
                "default": "en.wikipedia",
            },
            "days_back": {
                "type": "integer",
                "description": (
                    "How many days of history to fetch. 'spike' uses this as "
                    "the baseline window (default 30). 'series' returns this "
                    "many days. 'top' ignores this."
                ),
                "default": 30,
            },
            "date": {
                "type": "string",
                "description": ("Date for 'top' mode (YYYY-MM-DD). Defaults to yesterday."),
                "default": "",
            },
            "z_threshold": {
                "type": "number",
                "description": ("Spike mode: minimum z-score to report. Default: 2.0"),
                "default": 2.0,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 50.",
                "default": 50,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    def execute(
        self,
        *,
        mode: str = "spike",
        articles: str = "",
        project: str = "en.wikipedia",
        days_back: int = 30,
        date: str = "",
        z_threshold: float = 2.0,
        limit: int = 50,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("spike", "top", "series"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'spike', 'top', or 'series'.",
            )
        project = project.strip().lower()
        if not project or "." not in project:
            project = "en.wikipedia"

        if mode == "spike":
            return self._execute_spike(
                articles=articles,
                project=project,
                days_back=max(7, min(days_back, 90)),
                z_threshold=z_threshold,
                limit=max(1, min(limit, 200)),
            )
        if mode == "top":
            return self._execute_top(
                project=project,
                date=date,
                limit=max(1, min(limit, 500)),
            )
        return self._execute_series(
            articles=articles,
            project=project,
            days_back=max(1, min(days_back, 365)),
        )

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def _execute_spike(
        self,
        *,
        articles: str,
        project: str,
        days_back: int,
        z_threshold: float,
        limit: int,
    ) -> ToolResult:
        watchlist = self._parse_articles(articles) or list(_DEFAULT_WATCHLIST)

        yesterday = datetime.now(UTC) - timedelta(days=1)
        start = yesterday - timedelta(days=days_back)
        start_str = start.strftime("%Y%m%d")
        end_str = yesterday.strftime("%Y%m%d")

        spikes: list[dict[str, Any]] = []
        errors: list[str] = []

        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            for title in watchlist:
                try:
                    items = self._fetch_article_views(
                        client,
                        project,
                        title,
                        start_str,
                        end_str,
                    )
                except Exception as exc:
                    errors.append(f"{title}: {exc}")
                    continue

                if len(items) < 7:
                    continue  # not enough data for meaningful stats

                views = [it["views"] for it in items]
                result = _detect_spike(views, z_threshold)
                if result is not None:
                    z, latest, mean, std = result
                    spikes.append(
                        {
                            "article": title,
                            "project": project,
                            "latest_views": latest,
                            "mean_views": round(mean, 1),
                            "std_views": round(std, 1),
                            "z_score": round(z, 2),
                            "spike_ratio": round(latest / mean, 2) if mean > 0 else 0,
                            "days_analyzed": len(views),
                            "date": items[-1].get("timestamp", "")[:8],
                        }
                    )
                time.sleep(_POLITENESS_DELAY)

        spikes.sort(key=lambda s: s["z_score"], reverse=True)
        spikes = spikes[:limit]

        if not spikes and not errors:
            return ToolResult(
                success=True,
                output=(
                    f"Wikipedia Pageviews: No attention spikes detected "
                    f"(z>{z_threshold}) across {len(watchlist)} articles "
                    f"({days_back}d baseline on {project})."
                ),
                data={"spikes": [], "watchlist_size": len(watchlist)},
            )

        lines = [
            f"Wikipedia Attention Spikes (z>{z_threshold}, "
            f"{len(watchlist)} articles, {days_back}d baseline, {project}):",
            "",
        ]
        for s in spikes:
            lines.append(
                f"  {s['article']:40s}  z={s['z_score']:+5.2f}  "
                f"views={s['latest_views']:>8,}  "
                f"mean={s['mean_views']:>8,.0f}  "
                f"ratio={s['spike_ratio']:.1f}x"
            )
        if errors:
            lines.append(f"\n  ({len(errors)} articles failed to fetch)")

        self._persist_entities(spikes)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "spikes": spikes,
                "errors": errors,
                "watchlist_size": len(watchlist),
            },
        )

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, spikes: list[dict[str, Any]]) -> None:
        """Register topic entities and store L2 pageview spike observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not spikes:
            return
        try:
            self._persist_entities_inner(spikes)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, spikes: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        seen: set[str] = set()
        for spike in spikes:
            article = spike.get("article", "")
            if not article or article in seen:
                continue
            seen.add(article)

            topic_eid = entity_id_from_key("topic", article)
            store.register_entity(
                entity_type="topic",
                canonical_name=article.replace("_", " "),
                entity_id=topic_eid,
            )
            store.add_entity_alias(topic_eid, "wikipedia_article", article)

            # Parse date string (YYYYMMDD) to timestamp
            date_str = spike.get("date", "")
            try:
                ts = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC).timestamp()
            except (ValueError, AttributeError):
                ts = time.time()

            store.store_entity_observation(
                entity_id=topic_eid,
                source_tool="wikipedia_pageviews",
                observed_at=ts,
                observation_type="pageview_spike",
                depth_level=2,
                value={
                    "z_score": spike.get("z_score", 0.0),
                    "latest_views": spike.get("latest_views", 0),
                    "mean_views": spike.get("mean_views", 0.0),
                    "spike_ratio": spike.get("spike_ratio", 0.0),
                    "project": spike.get("project", ""),
                },
            )

    # ------------------------------------------------------------------
    # Top trending articles
    # ------------------------------------------------------------------

    def _execute_top(
        self,
        *,
        project: str,
        date: str,
        limit: int,
    ) -> ToolResult:
        if date:
            try:
                dt = datetime.strptime(date.strip(), "%Y-%m-%d")
            except ValueError:
                return ToolResult(
                    success=False,
                    output=f"Invalid date '{date}'. Use YYYY-MM-DD format.",
                )
        else:
            dt = datetime.now(UTC) - timedelta(days=1)

        year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
        url = f"{_BASE}/metrics/pageviews/top/{project}/all-access/{year}/{month}/{day}"

        cache_key = {"top": project, "date": f"{year}{month}{day}"}
        if self._cache:
            cached = self._cache.get("wikipedia_pageviews", cache_key)
            if cached is not None:
                return self._format_top(cached, project, f"{year}-{month}-{day}", limit)

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output=f"Wikipedia top pageviews failed: HTTP {exc.response.status_code}",
            )
        except Exception as exc:
            log.exception("Wikipedia top pageviews fetch failed")
            return ToolResult(success=False, output=f"Fetch error: {exc}")

        raw_articles = data.get("items", [{}])[0].get("articles", [])
        # Filter evergreen pages
        articles = [
            a
            for a in raw_articles
            if a.get("article", "") not in _EVERGREEN
            and not a.get("article", "").startswith("Special:")
            and not a.get("article", "").startswith("Wikipedia:")
            and not a.get("article", "").startswith("Portal:")
            and not a.get("article", "").startswith("File:")
        ]

        if self._cache:
            self._cache.put("wikipedia_pageviews", cache_key, articles)

        return self._format_top(articles, project, f"{year}-{month}-{day}", limit)

    def _format_top(
        self,
        articles: list[dict[str, Any]],
        project: str,
        date_str: str,
        limit: int,
    ) -> ToolResult:
        articles = articles[:limit]
        lines = [
            f"Wikipedia Top Articles ({project}, {date_str}):",
            "",
        ]
        for a in articles:
            name = a.get("article", "?").replace("_", " ")
            views = a.get("views", 0)
            rank = a.get("rank", 0)
            lines.append(f"  {rank:>4}. {name:50s} {views:>12,} views")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"articles": articles, "project": project, "date": date_str},
        )

    # ------------------------------------------------------------------
    # Timeseries mode
    # ------------------------------------------------------------------

    def _execute_series(
        self,
        *,
        articles: str,
        project: str,
        days_back: int,
    ) -> ToolResult:
        parsed = self._parse_articles(articles)
        if not parsed:
            return ToolResult(
                success=False,
                output="Series mode requires an 'articles' parameter with one article title.",
            )
        title = parsed[0]

        yesterday = datetime.now(UTC) - timedelta(days=1)
        start = yesterday - timedelta(days=days_back)
        start_str = start.strftime("%Y%m%d")
        end_str = yesterday.strftime("%Y%m%d")

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            ) as client:
                items = self._fetch_article_views(
                    client,
                    project,
                    title,
                    start_str,
                    end_str,
                )
        except Exception as exc:
            log.exception("Wikipedia series fetch failed")
            return ToolResult(success=False, output=f"Fetch error: {exc}")

        if not items:
            return ToolResult(
                success=True,
                output=f"No pageview data for '{title}' on {project}.",
                data={"article": title, "views": []},
            )

        views = [it["views"] for it in items]
        mean = sum(views) / len(views)
        std = _std(views)

        lines = [
            f"Wikipedia Pageviews: {title.replace('_', ' ')} ({project})",
            f"Period: {items[0]['timestamp'][:8]} — {items[-1]['timestamp'][:8]} ({len(items)} days)",
            f"Mean: {mean:,.0f}/day  Std: {std:,.0f}  Min: {min(views):,}  Max: {max(views):,}",
            "",
        ]
        for it in items:
            v = it["views"]
            z = (v - mean) / std if std > 0 else 0
            bar = "#" * max(1, int(v / mean * 20)) if mean > 0 else ""
            flag = " <<<" if z > 2.0 else ""
            lines.append(f"  {it['timestamp'][:8]}  {v:>8,}  {bar}{flag}")

        series_data = [{"date": it["timestamp"][:8], "views": it["views"]} for it in items]

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "article": title,
                "project": project,
                "views": series_data,
                "stats": {
                    "mean": round(mean, 1),
                    "std": round(std, 1),
                    "min": min(views),
                    "max": max(views),
                    "days": len(views),
                },
            },
        )

    # ------------------------------------------------------------------
    # HTTP fetching
    # ------------------------------------------------------------------

    def _fetch_article_views(
        self,
        client: httpx.Client,
        project: str,
        title: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Fetch daily pageviews for a single article. Returns list of {timestamp, views}."""
        cache_key = {"article": title, "project": project, "start": start, "end": end}
        if self._cache:
            cached = self._cache.get("wikipedia_pageviews", cache_key)
            if cached is not None:
                return cached

        encoded = quote(title, safe="")
        url = f"{_BASE}/metrics/pageviews/per-article/{project}/all-access/user/{encoded}/daily/{start}/{end}"
        resp = client.get(url)
        if resp.status_code == 404:
            # Article doesn't exist in this project
            return []
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if self._cache and items:
            self._cache.put("wikipedia_pageviews", cache_key, items)

        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_articles(raw: str) -> list[str]:
        """Parse comma-separated article titles, strip whitespace."""
        if not raw or not raw.strip():
            return []
        return [a.strip() for a in raw.split(",") if a.strip()]


# ------------------------------------------------------------------
# Pure math — no state, no IO
# ------------------------------------------------------------------


def _std(values: list[int | float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _detect_spike(
    views: list[int],
    z_threshold: float,
) -> tuple[float, int, float, float] | None:
    """Check if the most recent day is a spike relative to the rest.

    Uses all-but-last as baseline to compute mean/std, then z-score
    the last observation. Returns (z_score, latest, mean, std) or None.
    """
    if len(views) < 7:
        return None

    baseline = views[:-1]
    latest = views[-1]
    mean = sum(baseline) / len(baseline)
    std = _std(baseline)

    if std < 1.0:
        # Near-zero variance — any nonzero latest is technically infinite z.
        # Only flag if latest is meaningfully above baseline.
        if latest >= 200 and (mean < 1.0 or latest >= mean * 2):
            return (99.0, latest, mean, std)
        return None

    z = (latest - mean) / std
    if z >= z_threshold:
        return (z, latest, mean, std)
    return None
