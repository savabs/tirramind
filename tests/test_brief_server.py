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


def test_internal_telemetry_sources_not_customer_queryable(tmp_path, monkeypatch):
    """Internal DAG-stage telemetry (source="train_gnn" etc.) must not be
    reachable through the paid Data Platform API, in the catalog or by name.

    Regression for a real leak found 2026-08-27: GET /api/v1/data?source=
    train_gnn returned {"trained": false, "loss_ewc": 579753920.0, ...} —
    the model's own untrained-state defect, readable through the endpoint a
    Data Platform customer pays $500/mo for.
    """
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    store = PipelineStore()
    store.store_data("cftc", {"mode": "latest"}, {"z": 1.0})
    store.store_data("train_gnn", {}, {"trained": False, "loss_ewc": 579753920.0})

    sub_store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: sub_store)
    sub_store.set_active("sub_data", active=True, tier="data")
    data_key = sub_store.api_key_of("sub_data")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    import urllib.error

    # Not in the catalog.
    _, body = _get(base + f"/api/v1/sources?key={data_key}")
    parsed = json.loads(body)
    assert "train_gnn" not in {s["source"] for s in parsed["sources"]}
    assert "cftc" in {s["source"] for s in parsed["sources"]}

    # Rejected by name exactly like an unknown source — not a quiet 200.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + f"/api/v1/data?source=train_gnn&key={data_key}")
    assert exc.value.code == 400

    # A real source still works.
    _, real_body = _get(base + f"/api/v1/data?source=cftc&key={data_key}")
    assert json.loads(real_body)["ok"] is True

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


def test_entity_graph_endpoints_gated_and_scoped(tmp_path, monkeypatch):
    """/api/v1/entity-graph/* requires the Entity Graph tier and serves the
    REAL pipeline graph (entities + entity_links), scoped down: no
    entity_observations, no metadata_json — see
    docs/research/entity_graph_tier_mismatch.md."""
    import urllib.error

    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    p = PipelineStore()
    p.register_entity("company", "Acme Corp", "company:acme", metadata={"secret_note": "should never leak"})
    p.register_entity("country", "USA", "country:usa")
    p.link_entities("company:acme", "country:usa", "headquartered_in", "seed", confidence=0.9, metadata={"cik": "0001"})
    p.store_entity_observation("company:acme", "some_tool", 1.0, "signal", {"alpha": 42})

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

    # Wrong tier → 403, on every entity-graph path.
    for path in (
        "/api/v1/entity-graph/entities",
        "/api/v1/entity-graph/entity?id=company:acme",
        "/api/v1/entity-graph/links",
    ):
        sep = "&" if "?" in path else "?"
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base + path + f"{sep}key={brief_key}")
        assert exc.value.code == 403

    # Entity tier → 200, real data, scoped fields only.
    status, body = _get(base + f"/api/v1/entity-graph/entities?key={entity_key}")
    assert status == 200
    d1 = json.loads(body)
    assert d1["total"] == 2
    by_id = {e["entity_id"]: e for e in d1["entities"]}
    assert by_id["company:acme"]["canonical_name"] == "Acme Corp"
    assert set(by_id["company:acme"]) == {"entity_id", "entity_type", "canonical_name", "created_at"}
    assert "metadata" not in by_id["company:acme"]
    assert "secret_note" not in body  # entity metadata_json must never leak
    assert d1["dataset_scope"]["dataset"] == "production_entity_graph"

    status, body = _get(base + f"/api/v1/entity-graph/entity?id=company:acme&key={entity_key}")
    assert status == 200
    d2 = json.loads(body)
    assert d2["entity"]["canonical_name"] == "Acme Corp"
    assert len(d2["links"]) == 1
    link = d2["links"][0]
    assert link["link_type"] == "headquartered_in"
    assert set(link) == {"link_id", "entity_id_a", "entity_id_b", "link_type", "confidence", "source", "created_at"}
    assert "metadata" not in link  # cik must not leak — unreviewed per-source metadata is stripped

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + f"/api/v1/entity-graph/entity?id=does-not-exist&key={entity_key}")
    assert exc.value.code == 404

    status, body = _get(base + f"/api/v1/entity-graph/links?key={entity_key}")
    assert status == 200
    d3 = json.loads(body)
    assert d3["total"] == 1

    # The raw observation VALUE is never reachable through this surface —
    # only the dataset_scope disclosure is allowed to *name* entity_observations
    # (to explain why it's excluded), never its content.
    assert "some_tool" not in body
    assert '"alpha"' not in body

    httpd.shutdown()


def test_entity_graph_entities_pagination(tmp_path, monkeypatch):
    """limit/offset page the real entity list rather than silently truncating."""
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    out = tmp_path / "del"
    d = BriefDeliverer(out_dir=str(out), render_md=lambda b: "# x")

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    p = PipelineStore()
    for i in range(5):
        p.register_entity("company", f"Company {i}", f"company:{i}")

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_entity", active=True, tier="entity")
    entity_key = store.api_key_of("sub_entity")

    class Handler(H):
        deliverer = d

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    status, body = _get(base + f"/api/v1/entity-graph/entities?limit=2&offset=0&key={entity_key}")
    page1 = json.loads(body)
    assert page1["total"] == 5
    assert len(page1["entities"]) == 2

    status, body = _get(base + f"/api/v1/entity-graph/entities?limit=2&offset=2&key={entity_key}")
    page2 = json.loads(body)
    assert len(page2["entities"]) == 2
    assert {e["entity_id"] for e in page1["entities"]}.isdisjoint({e["entity_id"] for e in page2["entities"]})

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


# ── /evidence/ingest hardening ──────────────────────────────────────────────
# Regression tests for security-auditor findings 1 (CRITICAL, arbitrary file
# read via `path`) and 2 (HIGH, ingest gate fails open on empty token).
# Spec: docs/specs/evidence_ingest_hardening_spec.md


def _ingest_server(tmp_path):
    """Start a server whose evidence store lives under tmp_path."""
    from agent.brief_server import _Handler as H
    from agent.delivery.brief_deliverer import BriefDeliverer

    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=lambda b: "# x")

    class Handler(H):
        deliverer = d
        evidence_db = str(tmp_path / "evidence.db")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post_ingest(base, payload, token=None):
    import urllib.error

    req = urllib.request.Request(
        base + "/evidence/ingest",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token is not None:
        req.add_header("X-Ingest-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_ingest_denies_all_when_token_empty_and_auth_required(tmp_path, monkeypatch):
    """Finding 2: empty TIRRA_INGEST_TOKEN must DENY under TIRRA_REQUIRE_AUTH.

    The original gate was `if admin_token and token != admin_token`, which
    short-circuited to open when the token was empty — and the prod template
    ships it empty.
    """
    monkeypatch.setenv("TIRRA_REQUIRE_AUTH", "1")
    monkeypatch.delenv("TIRRA_INGEST_TOKEN", raising=False)
    httpd, base = _ingest_server(tmp_path)
    try:
        status, _ = _post_ingest(base, {"doc_id": "d1", "text": "Acme Corp met Beta Inc."})
        assert status == 403, "empty ingest token must fail CLOSED, not open"
    finally:
        httpd.shutdown()


def test_ingest_token_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "correct-token")
    httpd, base = _ingest_server(tmp_path)
    try:
        assert _post_ingest(base, {"doc_id": "d", "text": "x"}, token="wrong")[0] == 403
        assert _post_ingest(base, {"doc_id": "d", "text": "Acme Corp met Beta Inc."}, token="correct-token")[0] == 200
    finally:
        httpd.shutdown()


def test_path_ingest_refused_when_ingest_dir_unset(tmp_path, monkeypatch):
    """Finding 1, secure default: path mode is off unless opted into."""
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "t")
    monkeypatch.delenv("TIRRA_INGEST_DIR", raising=False)
    target = tmp_path / "readable.txt"
    target.write_text("Acme Corp met Beta Inc.")
    httpd, base = _ingest_server(tmp_path)
    try:
        status, body = _post_ingest(base, {"doc_id": "d", "path": str(target)}, token="t")
        assert status == 400
        assert str(target) not in body, "400 must not echo the path (existence oracle)"
    finally:
        httpd.shutdown()


def test_path_ingest_rejects_traversal_and_symlink_escape(tmp_path, monkeypatch):
    """`..` and symlinks must not escape TIRRA_INGEST_DIR."""
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "t")
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Acme Corp met Beta Inc.")
    monkeypatch.setenv("TIRRA_INGEST_DIR", str(ingest_dir))

    escape = ingest_dir / "escape.txt"
    escape.symlink_to(outside)

    httpd, base = _ingest_server(tmp_path)
    try:
        traversal = str(ingest_dir / ".." / "outside.txt")
        assert _post_ingest(base, {"doc_id": "a", "path": traversal}, token="t")[0] == 400
        assert _post_ingest(base, {"doc_id": "b", "path": str(escape)}, token="t")[0] == 400
        assert _post_ingest(base, {"doc_id": "c", "path": "/etc/passwd"}, token="t")[0] == 400
    finally:
        httpd.shutdown()


def test_path_ingest_allows_file_inside_ingest_dir(tmp_path, monkeypatch):
    """The allowed case still works — confinement, not removal."""
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "t")
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    doc = ingest_dir / "ok.txt"
    doc.write_text("Acme Corp met Beta Inc. Gamma Ltd signed with Delta Co.")
    monkeypatch.setenv("TIRRA_INGEST_DIR", str(ingest_dir))
    httpd, base = _ingest_server(tmp_path)
    try:
        assert _post_ingest(base, {"doc_id": "ok", "path": str(doc)}, token="t")[0] == 200
    finally:
        httpd.shutdown()


def test_env_file_cannot_be_ingested_and_read_back(tmp_path, monkeypatch):
    """THE EXPLOIT, encoded. Finding 1's full chain, end to end.

    Original attack: POST an env-file path with doc_type=csv (no '.' to split
    on, so the whole file lands in one "sentence"), then read the secret back
    out through the tier-gated evidence read routes.

    Asserts both halves: the ingest is refused, AND the secret never reaches
    the store. Without this test a refactor can silently restore the chain.
    """
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "t")
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    monkeypatch.setenv("TIRRA_INGEST_DIR", str(ingest_dir))

    canary = "CANARY-" + "VALUE-12345"  # noqa: S105 - fabricated, not a credential
    envfile = tmp_path / ".env.production"
    envfile.write_text(f"TIRRA_PADDLE_API_KEY={canary}\nTIRRA_PADDLE_WEBHOOK_SECRET={canary}\n")

    httpd, base = _ingest_server(tmp_path)
    try:
        status, body = _post_ingest(base, {"doc_id": "pwn", "path": str(envfile), "doc_type": "csv"}, token="t")
        assert status == 400, "env file outside TIRRA_INGEST_DIR must be refused"
        assert canary not in body

        from agent.evidence.store import EvidenceGraphStore

        store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
        dumped = json.dumps(store.graph_export())
        assert canary not in dumped, "canary reached the evidence graph — chain is OPEN"
    finally:
        httpd.shutdown()
