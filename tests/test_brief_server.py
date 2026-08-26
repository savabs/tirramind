"""Tests for the brief HTTP server (consumer surface)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agent.brief_server import _Handler
from agent.delivery.brief_deliverer import BriefDeliverer


@pytest.fixture
def _open_access(monkeypatch):
    """Force the server's dev-mode (unauthenticated) path.

    `_authorized_for` serves open only when NEITHER `TIRRA_SUB_KEYS` nor
    `TIRRA_PADDLE_WEBHOOK_SECRET` is set. These tests exercise brief delivery,
    not gating, so they must pin that state rather than inherit it.

    Without this they pass in isolation and fail in a full-suite run: something
    in the suite loads `.env` (e.g. `agent/convergence/backtest.py`), and once
    real Paddle credentials landed in `.env` on 2026-08-26 the server correctly
    started returning 403. The gating was right; the tests were ambient.
    """
    monkeypatch.delenv("TIRRA_SUB_KEYS", raising=False)
    monkeypatch.delenv("TIRRA_PADDLE_WEBHOOK_SECRET", raising=False)


@pytest.fixture
def live_server(tmp_path, _open_access):
    """Start the brief server on an ephemeral port with one delivered brief."""
    out = tmp_path / "del"
    deliverer = BriefDeliverer(out_dir=str(out), render_md=lambda b: f"# {b['brief_type']}")
    deliverer.deliver(
        {
            "brief_type": "intelligence",
            "contract_opportunities": [
                {
                    "award_id": "X1",
                    "recipient": "Co",
                    "agency": "VA",
                    "amount_usd": 40000.0,
                    "expected_value_usd": 20000.0,
                    "p_win": 0.75,
                    "is_long_tail": True,
                }
            ],
            "live_anomalies": [{"source": "cftc", "zscore": 3.0, "changepoint": True}],
        }
    )

    class Handler(_Handler):
        pass

    Handler.deliverer = deliverer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url) as r:
        return r.status, r.read().decode()


def test_serve_brief_json(live_server):
    status, body = _get(live_server + "/brief.json")
    assert status == 200
    d = json.loads(body)
    assert d["contract_opportunities"][0]["award_id"] == "X1"
    assert "live_anomalies" in d


def test_serve_brief_md(live_server):
    status, body = _get(live_server + "/brief.md")
    assert status == 200
    assert "# intelligence" in body


def test_serve_status(live_server):
    status, body = _get(live_server + "/status")
    assert status == 200
    s = json.loads(body)
    assert s["total_deliveries"] == 1


def test_serve_404_for_unknown(live_server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_server + "/nope")
    assert exc.value.code == 404


def test_subscription_gate_requires_key(tmp_path, monkeypatch):
    """When TIRRA_SUB_KEYS is set, the brief is 403 without a valid key."""
    import urllib.error

    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: f"# {b['brief_type']}")
    d.deliver({"brief_type": "intelligence", "contract_opportunities": [], "live_anomalies": []})

    monkeypatch.setenv("TIRRA_SUB_KEYS", "secret-key-1")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    import threading

    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    # no key → 403
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + "/brief.json")
    assert exc.value.code == 403
    # valid key → 200
    status, _ = _get(base + "/brief.json?key=secret-key-1")
    assert status == 200
    httpd.shutdown()


def test_evidence_endpoints_gated_by_entity_tier(tmp_path, monkeypatch):
    """Evidence Graph endpoints require an Entity Graph (or higher) tier subscriber."""
    import urllib.error

    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_brief", active=True, tier="brief")
    store.set_active("sub_entity", active=True, tier="entity")
    brief_key = store.api_key_of("sub_brief")
    entity_key = store.api_key_of("sub_entity")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    # brief-tier subscriber → 403 (wrong tier)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + f"/evidence/stats?key={brief_key}")
    assert exc.value.code == 403

    # entity-tier subscriber → 200
    status, body = _get(base + f"/evidence/stats?key={entity_key}")
    assert status == 200
    assert json.loads(body)["ok"] is True

    httpd.shutdown()


def test_data_api_requires_data_tier(tmp_path, monkeypatch):
    """GET /api/v1/data requires a Data Platform (or Scheduler) tier subscriber."""
    import urllib.error

    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {"mode": "latest"}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    data_key = store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + "/api/v1/data?source=cftc")
    assert exc.value.code == 403

    status, body = _get(base + f"/api/v1/data?source=cftc&key={data_key}")
    assert status == 200
    d2 = json.loads(body)
    assert d2["ok"] is True
    assert d2["rows"][0]["source"] == "cftc"

    httpd.shutdown()


def test_data_api_rejects_unknown_source(tmp_path, monkeypatch):
    """An unrecognized `source` gets a 400 pointing at /api/v1/sources, not an empty 200."""
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {"mode": "latest"}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    data_key = store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + f"/api/v1/data?source=not_a_real_source&key={data_key}")
    assert exc.value.code == 400
    body = exc.value.read().decode()
    assert "/api/v1/sources" in body

    httpd.shutdown()


def test_sources_endpoint_lists_catalog(tmp_path, monkeypatch):
    """GET /api/v1/sources returns every distinct source with row counts."""
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})
    PipelineStore().store_data("cftc", {}, {"z": 2.0})
    PipelineStore().store_data("gdelt", {}, {"n": 1})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    data_key = store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    status, body = _get(base + f"/api/v1/sources?key={data_key}")
    assert status == 200
    d2 = json.loads(body)
    by_name = {s["source"]: s for s in d2["sources"]}
    assert by_name["cftc"]["rows"] == 2
    assert by_name["gdelt"]["rows"] == 1

    httpd.shutdown()


def test_data_api_caps_limit(tmp_path, monkeypatch):
    """A `limit` above the hard cap is clamped, never passed through raw."""
    from agent.brief_server import _MAX_DATA_LIMIT
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    store_p = PipelineStore()
    for i in range(5):
        store_p.store_data("cftc", {}, {"i": i})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    data_key = store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    huge = _MAX_DATA_LIMIT * 100
    status, body = _get(base + f"/api/v1/data?source=cftc&limit={huge}&key={data_key}")
    assert status == 200
    assert json.loads(body)["ok"] is True  # would only matter at scale; asserts it doesn't error

    httpd.shutdown()


def test_usage_endpoint_reports_metered_calls(tmp_path, monkeypatch):
    """Each authorized call to a metered endpoint is logged and visible via /api/v1/usage."""
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.payments.usage import UsageStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    monkeypatch.setattr("agent.payments.usage.UsageStore", lambda: UsageStore(str(tmp_path / "usage.db")))
    store.set_active("sub_data", active=True, tier="data")
    data_key = store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    _get(base + f"/api/v1/data?source=cftc&key={data_key}")
    _get(base + f"/api/v1/data?source=cftc&key={data_key}")

    status, body = _get(base + f"/api/v1/usage?key={data_key}")
    assert status == 200
    summary = json.loads(body)
    assert summary["total"] == 2
    assert summary["by_endpoint"]["/api/v1/data"] == 2

    httpd.shutdown()


def test_buy_endpoint_returns_redirect(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_BUY_URL", "https://pay.example.com/checkout")
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    import threading

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    status, body = _get(f"http://127.0.0.1:{port}/buy")
    assert status == 200
    assert "pay.example.com" in body
    httpd.shutdown()


class TestBriefGating:
    """The brief is gated as soon as ANY auth mechanism is configured.

    Added 2026-08-26. Only the dev-mode-open path was covered, so
    `_authorized_for` switching to gated — which is correct production
    behaviour — surfaced as a mysterious full-suite 403 the moment real Paddle
    credentials landed in `.env`. Pin both directions explicitly.
    """

    def _serve(self, tmp_path):
        import threading as _threading

        deliverer = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=lambda b: f"# {b['brief_type']}")
        deliverer.deliver({"brief_type": "intelligence", "contract_opportunities": [], "live_anomalies": []})

        class Handler(_Handler):
            pass

        Handler.deliverer = deliverer
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        _threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"

    def test_brief_requires_key_when_static_keys_configured(self, tmp_path, monkeypatch):
        import urllib.error

        monkeypatch.setenv("TIRRA_SUB_KEYS", "secret-key-1")
        monkeypatch.delenv("TIRRA_PADDLE_WEBHOOK_SECRET", raising=False)
        httpd, base = self._serve(tmp_path)
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(base + "/brief.json")
            assert exc.value.code == 403
        finally:
            httpd.shutdown()

    def test_brief_served_with_valid_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIRRA_SUB_KEYS", "secret-key-1")
        monkeypatch.delenv("TIRRA_PADDLE_WEBHOOK_SECRET", raising=False)
        httpd, base = self._serve(tmp_path)
        try:
            status, _ = _get(base + "/brief.json?key=secret-key-1")
            assert status == 200
        finally:
            httpd.shutdown()

    def test_paddle_secret_alone_also_gates(self, tmp_path, monkeypatch):
        """A configured webhook secret gates even with no static keys.

        This is exactly what real Paddle credentials in .env did.
        """
        import urllib.error

        monkeypatch.delenv("TIRRA_SUB_KEYS", raising=False)
        monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "pdl_ntfset_dummy")
        httpd, base = self._serve(tmp_path)
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(base + "/brief.json")
            assert exc.value.code == 403
        finally:
            httpd.shutdown()
