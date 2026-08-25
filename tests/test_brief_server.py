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
def live_server(tmp_path):
    """Start the brief server on an ephemeral port with one delivered brief."""
    out = tmp_path / "del"
    deliverer = BriefDeliverer(out_dir=str(out), render_md=lambda b: f"# {b['brief_type']}")
    deliverer.deliver(
        {
            "brief_type": "intelligence",
            "contract_opportunities": [
                {"award_id": "X1", "recipient": "Co", "agency": "VA",
                 "amount_usd": 40000.0, "expected_value_usd": 20000.0,
                 "p_win": 0.75, "is_long_tail": True}
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
