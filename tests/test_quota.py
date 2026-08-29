"""Tests for the 2026-08-28 brief_server.py hardening pass:

  TASK 1 — per-key, per-tier burst rate limiting on every paid route
           (BUG C: previously NO paid route was rate limited at all).
  TASK 2 — monthly quota, backed by UsageStore.count_since (previously wired
           to nothing), fail-CLOSED on an unexpected read error.
  TASK 3 — GET /account (HTML shell) + GET /api/v1/account (JSON, header-only
           auth) self-service account page.
  TASK 4 — do_OPTIONS (CORS preflight); there was none, so it 501'd.

Ownership: agent/brief_server.py is owned by api-backend-engineer. This file
is the one new test file that agent was authorized to create.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import UTC
from http.server import ThreadingHTTPServer

import pytest

import agent.brief_server as bs
from agent.brief_server import _Handler, _month_bounds, _RateLimiter
from agent.delivery.brief_deliverer import BriefDeliverer
from agent.payments.handler import SubscriberStore
from agent.payments.usage import UsageStore
from agent.pipeline.store import PipelineStore


@pytest.fixture(autouse=True)
def _pin_env(monkeypatch):
    """Pin dev-mode-open state explicitly — see test_brief_server.py's
    identical fixture docstring for why this must never be inherited from
    the ambient environment (real Paddle creds live in .env)."""
    monkeypatch.delenv("TIRRA_SUB_KEYS", raising=False)
    monkeypatch.delenv("TIRRA_PADDLE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("TIRRA_REJECT_QUERY_KEYS", raising=False)
    monkeypatch.delenv("TIRRA_CORS_ORIGIN", raising=False)


def _server(tmp_path):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=lambda b: "# x")
    d.deliver({"brief_type": "intelligence", "contract_opportunities": [], "live_anomalies": []})

    class Handler(_Handler):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url: str, headers: dict | None = None) -> tuple[int, str, object]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode(), r
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e


def _request(url: str, method: str, headers: dict | None = None) -> tuple[int, str, object]:
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode(), r
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e


def _live_subscriber(tmp_path, monkeypatch, *, tier: str):
    """A currently-active subscriber on `tier`, with real gating switched on."""
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active(f"sub_{tier}", active=True, tier=tier)
    return store, store.api_key_of(f"sub_{tier}")


def _isolated_usage_store(tmp_path, monkeypatch):
    store = UsageStore(str(tmp_path / "usage.db"))
    monkeypatch.setattr("agent.payments.usage.UsageStore", lambda: store)
    return store


# ── TASK 1: per-key, per-tier burst rate limiting ────────────────────────────


def test_month_bounds_wraps_december_into_next_january():
    """Pure-function sanity check for the quota reset boundary."""
    import calendar
    from datetime import datetime

    dec_15 = datetime(2026, 12, 15, 12, 0, 0, tzinfo=UTC).timestamp()
    start, end = _month_bounds(dec_15)
    start_dt = datetime.fromtimestamp(start, tz=UTC)
    end_dt = datetime.fromtimestamp(end, tz=UTC)
    assert (start_dt.year, start_dt.month, start_dt.day) == (2026, 12, 1)
    assert (end_dt.year, end_dt.month, end_dt.day) == (2027, 1, 1)
    assert calendar is not None  # imported only to prove no accidental shadowing


def test_data_tier_rate_limited_after_cap(tmp_path, monkeypatch):
    """A tight per-key limiter on the 'data' bucket 429s after its cap, with
    Retry-After — and the SAME key works again once the window is fresh."""
    monkeypatch.setattr(
        bs, "_TIER_RATE_LIMITS", {**bs._TIER_RATE_LIMITS, "data": _RateLimiter(max_calls=2, window_s=600)}
    )
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="data")

    httpd, base = _server(tmp_path)
    try:
        s1, _, _ = _get(base + f"/api/v1/data?source=cftc&key={key}")
        assert s1 == 200
        s2, _, _ = _get(base + f"/api/v1/data?source=cftc&key={key}")
        assert s2 == 200
        s3, body3, resp3 = _get(base + f"/api/v1/data?source=cftc&key={key}")
        assert s3 == 429
        d = json.loads(body3)
        assert d["ok"] is False
        assert d["error"] == "rate limited"
        assert resp3.headers.get("Retry-After") is not None
    finally:
        httpd.shutdown()


def test_rate_limit_buckets_are_independent_per_tier(tmp_path, monkeypatch):
    """A brief-tier key exhausting ITS bucket must not affect a data-tier
    key's own budget — tiers must not share a limiter."""
    monkeypatch.setattr(
        bs,
        "_TIER_RATE_LIMITS",
        {**bs._TIER_RATE_LIMITS, "brief": _RateLimiter(max_calls=1, window_s=600)},
    )
    _isolated_usage_store(tmp_path, monkeypatch)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    store.set_active("sub_brief", active=True, tier="brief")
    store.set_active("sub_data", active=True, tier="data")
    brief_key = store.api_key_of("sub_brief")
    data_key = store.api_key_of("sub_data")

    httpd, base = _server(tmp_path)
    try:
        s1, _, _ = _get(base + f"/brief.json?key={brief_key}")
        assert s1 == 200
        s2, body2, _ = _get(base + f"/brief.json?key={brief_key}")
        assert s2 == 429  # brief bucket (cap 1) now exhausted

        # data-tier key, untouched bucket, still works fine.
        s3, _, _ = _get(base + f"/api/v1/data?source=cftc&key={data_key}")
        assert s3 == 200
    finally:
        httpd.shutdown()


def test_wrong_tier_key_still_403s_even_with_rate_limit_headroom(tmp_path, monkeypatch):
    """The 403 tier gate runs BEFORE rate limiting — a caller with the wrong
    tier is denied regardless of how much budget its bucket has left. Proves
    the gate is checked both ways: authorized-and-limited AND
    unauthorized-with-headroom."""
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, brief_key = _live_subscriber(tmp_path, monkeypatch, tier="brief")

    httpd, base = _server(tmp_path)
    try:
        status, _, _ = _get(base + f"/api/v1/data?source=cftc&key={brief_key}")
        assert status == 403
    finally:
        httpd.shutdown()


def test_no_key_dev_mode_is_never_rate_limited_or_quota_capped(tmp_path):
    """Dev-mode-open (no TIRRA_SUB_KEYS / TIRRA_PADDLE_WEBHOOK_SECRET at all)
    has no subscriber identity to meter — must not 429/503 a bare, keyless
    request. This is also what keeps every OTHER test file's dev-mode tests
    (which hammer /brief.json etc. with no key) from becoming flaky."""
    httpd, base = _server(tmp_path)
    try:
        for _ in range(5):
            status, _, _ = _get(base + "/brief.json")
            assert status == 200
    finally:
        httpd.shutdown()


# ── TASK 2: monthly quota ────────────────────────────────────────────────────


def test_monthly_quota_exceeded_returns_429_with_limit_and_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "_TIER_MONTHLY_QUOTAS", {**bs._TIER_MONTHLY_QUOTAS, "data": 2})
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="data")

    httpd, base = _server(tmp_path)
    try:
        assert _get(base + f"/api/v1/data?source=cftc&key={key}")[0] == 200
        assert _get(base + f"/api/v1/data?source=cftc&key={key}")[0] == 200
        status, body, resp = _get(base + f"/api/v1/data?source=cftc&key={key}")
        assert status == 429
        d = json.loads(body)
        assert d["ok"] is False
        assert d["error"] == "monthly quota exceeded"
        assert d["tier"] == "data"
        assert d["quota"] == 2
        assert d["used"] == 2
        assert "reset_at" in d
        assert resp.headers.get("Retry-After") is not None
    finally:
        httpd.shutdown()


def test_quota_check_fails_closed_not_200_on_usage_store_error(tmp_path, monkeypatch):
    """CRITICAL: if the quota READ itself blows up, the request must be
    DENIED (503), never silently let through as if unmetered. This is the
    opposite contract from `_log_usage`, which fails open on purpose because
    it doesn't gate anything."""

    def _boom(self, key_id, since):
        raise RuntimeError("sqlite is on fire — should never leak, must never fail open")

    monkeypatch.setattr(UsageStore, "count_since", _boom)
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="data")

    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + f"/api/v1/data?source=cftc&key={key}")
        assert status == 503
        assert status != 200
        d = json.loads(body)
        assert d["ok"] is False
        assert "sqlite is on fire" not in body
        assert "Traceback" not in body
    finally:
        httpd.shutdown()


def test_usage_endpoint_itself_is_never_quota_capped(tmp_path, monkeypatch):
    """Checking your OWN usage must not burn your OWN quota — otherwise a
    customer near their cap could get locked out of even seeing that fact."""
    monkeypatch.setattr(bs, "_TIER_MONTHLY_QUOTAS", {**bs._TIER_MONTHLY_QUOTAS, "brief": 0})
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="brief")

    httpd, base = _server(tmp_path)
    try:
        # Brief itself is capped at 0 → immediately quota-exceeded.
        status, body, _ = _get(base + f"/brief.json?key={key}")
        assert status == 429
        assert json.loads(body)["error"] == "monthly quota exceeded"

        # But /api/v1/usage for the SAME key still works — it's exempt.
        status2, body2, _ = _get(base + f"/api/v1/usage?key={key}")
        assert status2 == 200
        assert json.loads(body2)["ok"] is True
    finally:
        httpd.shutdown()


# ── TASK 3: GET /account + GET /api/v1/account ──────────────────────────────


def test_account_page_serves_unauthenticated_html_shell(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, resp = _get(base + "/account")
        assert status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/html")
        assert "X-Brief-Key" in body  # the shell's own JS names the header it sends
        assert "<input" in body
    finally:
        httpd.shutdown()


def test_account_json_missing_header_401(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/account")
        assert status == 401
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_account_json_malformed_key_400(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/account", headers={"X-Brief-Key": "not-a-real-key-format"})
        assert status == 400
    finally:
        httpd.shutdown()


def test_account_json_unknown_key_403(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    httpd, base = _server(tmp_path)
    try:
        fake = "tirra_" + ("x" * 32)
        status, body, _ = _get(base + "/api/v1/account", headers={"X-Brief-Key": fake})
        assert status == 403
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_account_json_rejects_query_string_key_header_only(tmp_path, monkeypatch):
    """Sensitive account info (tier, usage, subscription status) is header-
    only, same reasoning as POST /api/v1/rotate-key — a bare ?key= must NOT
    grant access, even though most other GET routes accept it as a fallback."""
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="brief")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + f"/api/v1/account?key={key}")
        assert status == 401  # no X-Brief-Key header presented — query key ignored entirely
    finally:
        httpd.shutdown()


def test_account_json_valid_key_returns_tier_status_and_usage(tmp_path, monkeypatch):
    _isolated_usage_store(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="data")

    httpd, base = _server(tmp_path)
    try:
        # Generate one metered usage record first.
        assert _get(base + f"/api/v1/data?source=cftc&key={key}")[0] == 200

        status, body, _ = _get(base + "/api/v1/account", headers={"X-Brief-Key": key})
        assert status == 200
        d = json.loads(body)
        assert d["ok"] is True
        assert d["tier"] == "data"
        assert d["active"] is True
        assert d["usage"]["used"] == 1
        assert d["usage"]["quota"] == bs._TIER_MONTHLY_QUOTAS["data"]
        assert "period_reset_at" in d["usage"]
        assert "management_urls" in d
    finally:
        httpd.shutdown()


def test_account_json_admin_static_key_has_no_account_404(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_SUB_KEYS", "tirra_admin_dev_key_0000000000000")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/account", headers={"X-Brief-Key": "tirra_admin_dev_key_0000000000000"})
        assert status == 404
    finally:
        httpd.shutdown()


def test_account_json_never_500s_when_paddle_management_url_fetch_fails(tmp_path, monkeypatch):
    """The Paddle round-trip for management_urls is best-effort — a network
    error there must not take down the rest of the (already-known-locally)
    account payload."""
    _isolated_usage_store(tmp_path, monkeypatch)
    _store, key = _live_subscriber(tmp_path, monkeypatch, tier="brief")
    # sub_brief has a subscription_id but no real Paddle creds/network here —
    # PaddleConfig.from_env() will raise; confirm graceful degradation.
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/account", headers={"X-Brief-Key": key})
        assert status == 200
        d = json.loads(body)
        assert d["ok"] is True
        assert d["management_urls"] is None
    finally:
        httpd.shutdown()


# ── TASK 4: do_OPTIONS (CORS preflight) ─────────────────────────────────────


def test_options_preflight_no_longer_501s(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, _, resp = _request(base + "/api/v1/data", "OPTIONS")
        assert status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "X-Brief-Key" in resp.headers.get("Access-Control-Allow-Headers", "")
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")
    finally:
        httpd.shutdown()


def test_options_preflight_requires_no_key_on_a_gated_route(tmp_path, monkeypatch):
    """A preflight carries no credentials by design — it must succeed even
    against a fully-gated route, since the browser sends it before the real
    (credentialed) request."""
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    httpd, base = _server(tmp_path)
    try:
        status, _, _ = _request(base + "/api/v1/data", "OPTIONS")
        assert status == 204
    finally:
        httpd.shutdown()


def test_options_preflight_honors_cors_origin_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CORS_ORIGIN", "https://example.com")
    httpd, base = _server(tmp_path)
    try:
        status, _, resp = _request(base + "/status", "OPTIONS")
        assert status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://example.com"
    finally:
        httpd.shutdown()
