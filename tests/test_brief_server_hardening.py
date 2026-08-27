"""Regression tests for the 2026-08-27 brief_server.py hardening pass
(findings C2-C6) plus the two new routes: GET /api/v1/claim and
POST /api/v1/contact.

Ownership: agent/brief_server.py is owned by api-backend-engineer. This file
is the ONLY test file that agent may create — tests/test_brief_server.py is
owned by another agent and must never be edited here.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agent.brief_server import _Handler, _RateLimiter
from agent.delivery.brief_deliverer import BriefDeliverer


@pytest.fixture(autouse=True)
def _open_access(monkeypatch):
    """Pin dev-mode-open state explicitly — see test_brief_server.py's
    identical fixture docstring for why this must never be inherited from
    the ambient environment."""
    monkeypatch.delenv("TIRRA_SUB_KEYS", raising=False)
    monkeypatch.delenv("TIRRA_PADDLE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("TIRRA_REJECT_QUERY_KEYS", raising=False)
    monkeypatch.delenv("TIRRA_CORS_ORIGIN", raising=False)
    monkeypatch.delenv("TIRRA_WEB_ORIGIN", raising=False)
    monkeypatch.delenv("TIRRA_MAX_CONCURRENT_REQUESTS", raising=False)


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


def _post(url: str, payload: bytes | dict, headers: dict | None = None) -> tuple[int, str, object]:
    body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode(), r
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e


# ── C2: query-string keys ────────────────────────────────────────────────


def test_query_key_still_works_by_default(tmp_path, monkeypatch):
    """Backward compat: tests/test_brief_server.py uses ?key= extensively."""
    from agent.payments.handler import SubscriberStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_brief", active=True, tier="brief")
    key = store.api_key_of("sub_brief")
    try:
        status, _, _ = _get(base + f"/brief.json?key={key}")
        assert status == 200
    finally:
        httpd.shutdown()


def test_reject_query_keys_flag_hard_rejects(tmp_path, monkeypatch):
    """TIRRA_REJECT_QUERY_KEYS=1 refuses a ?key= even if it would otherwise
    be valid — the header is the only accepted channel."""
    from agent.payments.handler import SubscriberStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.setenv("TIRRA_REJECT_QUERY_KEYS", "1")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_brief", active=True, tier="brief")
    key = store.api_key_of("sub_brief")
    try:
        status, body, _ = _get(base + f"/brief.json?key={key}")
        assert status == 400
        assert "X-Brief-Key" in body

        # The header channel still works under the same flag.
        status2, _, _ = _get(base + "/brief.json", headers={"X-Brief-Key": key})
        assert status2 == 200
    finally:
        httpd.shutdown()


# ── C3: source allowlist (not a denylist) ────────────────────────────────


def test_unknown_new_source_name_rejected_even_though_not_denylisted(tmp_path, monkeypatch):
    """A brand-new internal stage that writes under its own name is excluded
    by construction (no DAG table_name), not because someone remembered to
    add it to a list. Simulate that with a made-up source no denylist would
    ever have known about."""
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("brand_new_internal_stage_nobody_denylisted", {}, {"secret": "leak"})
    PipelineStore().store_data("cftc", {}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    key = store.api_key_of("sub_data")

    try:
        status, body, _ = _get(base + f"/api/v1/data?source=brand_new_internal_stage_nobody_denylisted&key={key}")
        assert status == 400
        assert "secret" not in body

        _, sources_body, _ = _get(base + f"/api/v1/sources?key={key}")
        names = {s["source"] for s in json.loads(sources_body)["sources"]}
        assert "brand_new_internal_stage_nobody_denylisted" not in names
        assert "cftc" in names
    finally:
        httpd.shutdown()


def test_known_manual_store_product_source_reachable(tmp_path, monkeypatch):
    """pm_trades etc. have no DAG table_name (manual store_data call inside a
    FunctionOperator body) but ARE a real, documented product surface — must
    not be collateral damage from switching to an allowlist."""
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("pm_trades", {}, {"tx_hash": "0x1"})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    key = store.api_key_of("sub_data")

    try:
        status, body, _ = _get(base + f"/api/v1/data?source=pm_trades&key={key}")
        assert status == 200
        assert json.loads(body)["ok"] is True
    finally:
        httpd.shutdown()


# ── C4: guarded int() parsing ─────────────────────────────────────────────


def test_malformed_limit_does_not_500_on_data_api(tmp_path, monkeypatch):
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    key = store.api_key_of("sub_data")

    try:
        status, body, _ = _get(base + f"/api/v1/data?source=cftc&limit=not-a-number&key={key}")
        assert status == 200
        assert json.loads(body)["ok"] is True
    finally:
        httpd.shutdown()


def test_malformed_limit_does_not_500_on_dag_runs(tmp_path, monkeypatch):
    from agent.payments.handler import SubscriberStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_sched", active=True, tier="scheduler")
    key = store.api_key_of("sub_sched")

    try:
        status, body, _ = _get(base + f"/api/v1/dag/runs?limit=%F0%9F%92%A9&key={key}")
        assert status == 200
        assert json.loads(body)["ok"] is True
    finally:
        httpd.shutdown()


def test_malformed_since_does_not_500_on_data_api(tmp_path, monkeypatch):
    from agent.payments.handler import SubscriberStore
    from agent.pipeline.store import PipelineStore

    httpd, base = _server(tmp_path)
    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    monkeypatch.chdir(tmp_path)
    PipelineStore().store_data("cftc", {}, {"z": 1.0})

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_data", active=True, tier="data")
    key = store.api_key_of("sub_data")

    try:
        status, _, _ = _get(base + f"/api/v1/data?source=cftc&since=not-a-timestamp&key={key}")
        assert status == 200
    finally:
        httpd.shutdown()


def test_malformed_content_length_does_not_500_on_webhook(tmp_path, monkeypatch):
    """C4 applies to POST's Content-Length header too, not just GET query ints."""
    httpd, base = _server(tmp_path)
    req = urllib.request.Request(base + "/webhook", data=b"{}", method="POST")
    req.add_header("Content-Length", "not-a-number")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    # Must not be a raw 500 — malformed signature/body still fails cleanly.
    assert status in (400, 403)
    httpd.shutdown()


# ── C6: version banner + configurable CORS ───────────────────────────────


def test_server_header_does_not_leak_python_version(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        _, _, resp = _get(base + "/status")
        server_header = resp.headers.get("Server", "")
        assert "Python/" not in server_header
    finally:
        httpd.shutdown()


def test_cors_origin_configurable_defaults_to_star(tmp_path, monkeypatch):
    httpd, base = _server(tmp_path)
    try:
        _, _, resp = _get(base + "/status")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
    finally:
        httpd.shutdown()


def test_cors_origin_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CORS_ORIGIN", "https://example.com")
    httpd, base = _server(tmp_path)
    try:
        _, _, resp = _get(base + "/status")
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://example.com"
    finally:
        httpd.shutdown()


# ── Bounded concurrency ───────────────────────────────────────────────────


def test_bounded_concurrency_does_not_break_correctness(tmp_path, monkeypatch):
    """A tiny semaphore serializes handling but every request still succeeds
    — bounding concurrency must queue, not drop or error."""
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_REQUEST_SEMAPHORE", threading.BoundedSemaphore(1))
    httpd, base = _server(tmp_path)
    try:
        results = []

        def _hit():
            status, _, _ = _get(base + "/status")
            results.append(status)

        threads = [threading.Thread(target=_hit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert results == [200] * 10
    finally:
        httpd.shutdown()


# ── GET /api/v1/claim ─────────────────────────────────────────────────────


def test_claim_missing_txn_is_400(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim")
        assert status == 400
        d = json.loads(body)
        assert d["ok"] is False
        assert d["status"] == "bad_request"
    finally:
        httpd.shutdown()


def test_claim_malformed_txn_is_400(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=" + urllib.parse.quote("../../etc/passwd"))
        assert status == 400
        assert json.loads(body)["status"] == "bad_request"
    finally:
        httpd.shutdown()


def test_claim_sets_cors_header_to_web_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_WEB_ORIGIN", "https://tirramind.com")
    httpd, base = _server(tmp_path)
    try:
        status, body, resp = _get(base + "/api/v1/claim")
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://tirramind.com"
    finally:
        httpd.shutdown()


def _patch_claim_result(monkeypatch, status, **fields):
    from agent.payments.claim import ClaimResult

    def _fake(*args, **kwargs):
        return ClaimResult(status=status, **fields)

    monkeypatch.setattr("agent.payments.claim.claim_transaction", _fake)


def test_claim_unknown_transaction_404(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "unknown_transaction")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_doesnotexist000000")
        assert status == 404
        assert json.loads(body)["status"] == "unknown_transaction"
    finally:
        httpd.shutdown()


def test_claim_not_completed_422(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "not_completed", transaction_status="past_due")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_pastdue0000000000")
        assert status == 422
        d = json.loads(body)
        assert d["status"] == "not_completed"
        assert d["transaction_status"] == "past_due"
    finally:
        httpd.shutdown()


def test_claim_pending_202_with_retry_after(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "pending")
    httpd, base = _server(tmp_path)
    try:
        status, body, resp = _get(base + "/api/v1/claim?txn=txn_pending00000000000")
        assert status == 202
        assert json.loads(body)["status"] == "pending"
        assert resp.headers.get("Retry-After") == "3"
    finally:
        httpd.shutdown()


def test_claim_subscriber_inactive_409(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "subscriber_inactive", subscription_id="sub_1")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_inactive000000000")
        assert status == 409
        assert json.loads(body)["status"] == "subscriber_inactive"
    finally:
        httpd.shutdown()


def test_claim_claimed_200_returns_key_once(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "claimed", api_key="tirra_abc123", tier="brief", subscription_id="sub_1")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_claimme000000000000")
        assert status == 200
        d = json.loads(body)
        assert d["status"] == "claimed"
        assert d["api_key"] == "tirra_abc123"
        assert d["tier"] == "brief"
    finally:
        httpd.shutdown()


def test_claim_already_claimed_200_without_key(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "already_claimed", subscription_id="sub_1")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_alreadyclaimed00000")
        assert status == 200
        d = json.loads(body)
        assert d["status"] == "already_claimed"
        assert "api_key" not in d
    finally:
        httpd.shutdown()


def test_claim_upstream_error_502(tmp_path, monkeypatch):
    _patch_claim_result(monkeypatch, "upstream_error")
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_upstreamerr00000000")
        assert status == 502
        assert json.loads(body)["status"] == "upstream_error"
    finally:
        httpd.shutdown()


def test_claim_per_txn_rate_limit_429(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_CLAIM_TXN_LIMITER", _RateLimiter(max_calls=2, window_s=600))
    _patch_claim_result(monkeypatch, "pending")
    httpd, base = _server(tmp_path)
    try:
        txn = "?txn=txn_ratelimited0000000"
        assert _get(base + "/api/v1/claim" + txn)[0] == 202
        assert _get(base + "/api/v1/claim" + txn)[0] == 202
        status, body, resp = _get(base + "/api/v1/claim" + txn)
        assert status == 429
        assert json.loads(body)["status"] == "rate_limited"
        assert resp.headers.get("Retry-After") is not None
    finally:
        httpd.shutdown()


def test_claim_per_ip_rate_limit_429_across_different_txns(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_CLAIM_IP_LIMITER", _RateLimiter(max_calls=2, window_s=600))
    _patch_claim_result(monkeypatch, "pending")
    httpd, base = _server(tmp_path)
    try:
        assert _get(base + "/api/v1/claim?txn=txn_ipratelimited00001")[0] == 202
        assert _get(base + "/api/v1/claim?txn=txn_ipratelimited00002")[0] == 202
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_ipratelimited00003")
        assert status == 429
        assert json.loads(body)["status"] == "rate_limited"
    finally:
        httpd.shutdown()


def test_claim_upstream_exception_is_502_not_500(tmp_path, monkeypatch):
    """PaddleConfig/PaddleClient construction blowing up must never surface
    as a bare 500 with a stack trace."""

    def _boom(*args, **kwargs):
        raise RuntimeError("boom — should never leak to the client")

    monkeypatch.setattr("agent.payments.config.PaddleConfig.from_env", staticmethod(_boom))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/claim?txn=txn_configboom00000000")
        assert status == 502
        d = json.loads(body)
        assert d["status"] == "upstream_error"
        assert "boom" not in body
        assert "Traceback" not in body
    finally:
        httpd.shutdown()


# ── POST /api/v1/contact ──────────────────────────────────────────────────


def test_contact_valid_submission_persists_and_returns_200(tmp_path, monkeypatch):
    log_path = tmp_path / "contact_messages.jsonl"
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(log_path))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(
            base + "/api/v1/contact",
            {"name": "Ada", "email": "ada@example.com", "subject": "hi", "message": "hello there"},
        )
        assert status == 200
        assert json.loads(body)["ok"] is True
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["email"] == "ada@example.com"
        assert record["message"] == "hello there"
    finally:
        httpd.shutdown()


def test_contact_missing_field_400(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(tmp_path / "contact_messages.jsonl"))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(
            base + "/api/v1/contact",
            {"name": "Ada", "email": "ada@example.com", "subject": "", "message": "hello"},
        )
        assert status == 400
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_contact_bad_json_400(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(tmp_path / "contact_messages.jsonl"))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(base + "/api/v1/contact", b"not json")
        assert status == 400
    finally:
        httpd.shutdown()


def test_contact_oversized_message_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(tmp_path / "contact_messages.jsonl"))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(
            base + "/api/v1/contact",
            {"name": "Ada", "email": "a@example.com", "subject": "hi", "message": "x" * 20000},
        )
        assert status in (400, 413)
    finally:
        httpd.shutdown()


def test_contact_oversized_body_rejected_before_storing(tmp_path, monkeypatch):
    log_path = tmp_path / "contact_messages.jsonl"
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(log_path))
    httpd, base = _server(tmp_path)
    try:
        huge = json.dumps({"name": "A", "email": "a@b.com", "subject": "s", "message": "x" * 100000}).encode()
        status, body, _ = _post(base + "/api/v1/contact", huge)
        assert status == 413
        assert not log_path.exists()
    finally:
        httpd.shutdown()


def test_contact_rate_limited_after_cap(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_CONTACT_IP_LIMITER", _RateLimiter(max_calls=2, window_s=600))
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(tmp_path / "contact_messages.jsonl"))
    httpd, base = _server(tmp_path)
    payload = {"name": "Ada", "email": "ada@example.com", "subject": "hi", "message": "hello"}
    try:
        assert _post(base + "/api/v1/contact", payload)[0] == 200
        assert _post(base + "/api/v1/contact", payload)[0] == 200
        status, body, resp = _post(base + "/api/v1/contact", payload)
        assert status == 429
        assert resp.headers.get("Retry-After") is not None
    finally:
        httpd.shutdown()


# ── GET /api/v1/admin/contact-messages ────────────────────────────────────


def test_admin_contact_messages_empty_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(tmp_path / "does_not_exist.jsonl"))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/admin/contact-messages")
        assert status == 200
        d = json.loads(body)
        assert d == {"ok": True, "messages": [], "count": 0, "total": 0}
    finally:
        httpd.shutdown()


def test_admin_contact_messages_reads_submitted_messages_newest_first(tmp_path, monkeypatch):
    import agent.brief_server as bs

    # Isolated limiter: the module-level default is shared (and already
    # exercised) across every other /api/v1/contact test in this same
    # process/IP — a fresh instance here keeps this test's 3 posts from
    # colliding with that shared budget.
    monkeypatch.setattr(bs, "_CONTACT_IP_LIMITER", _RateLimiter(max_calls=10, window_s=3600))
    log_path = tmp_path / "contact_messages.jsonl"
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(log_path))
    httpd, base = _server(tmp_path)
    try:
        for i in range(3):
            status, _, _ = _post(
                base + "/api/v1/contact",
                {"name": "Ada", "email": "ada@example.com", "subject": f"msg-{i}", "message": "hi"},
            )
            assert status == 200

        status, body, _ = _get(base + "/api/v1/admin/contact-messages")
        assert status == 200
        d = json.loads(body)
        assert d["total"] == 3
        assert d["count"] == 3
        # Newest first.
        assert [m["subject"] for m in d["messages"]] == ["msg-2", "msg-1", "msg-0"]
    finally:
        httpd.shutdown()


def test_admin_contact_messages_gated_by_ingest_token(tmp_path, monkeypatch):
    """When TIRRA_INGEST_TOKEN is configured, the admin route requires it —
    same gate as POST /evidence/ingest, and messages must not leak without
    it."""
    monkeypatch.setenv("TIRRA_INGEST_TOKEN", "secret-admin-token")
    log_path = tmp_path / "contact_messages.jsonl"
    log_path.write_text(json.dumps({"subject": "should not leak"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(log_path))
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _get(base + "/api/v1/admin/contact-messages")
        assert status == 403
        assert "should not leak" not in body

        status2, body2, _ = _get(
            base + "/api/v1/admin/contact-messages", headers={"X-Ingest-Token": "secret-admin-token"}
        )
        assert status2 == 200
        assert json.loads(body2)["total"] == 1
    finally:
        httpd.shutdown()


def test_admin_contact_messages_pagination(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_CONTACT_IP_LIMITER", _RateLimiter(max_calls=10, window_s=3600))
    log_path = tmp_path / "contact_messages.jsonl"
    monkeypatch.setenv("TIRRA_CONTACT_LOG", str(log_path))
    httpd, base = _server(tmp_path)
    try:
        for i in range(5):
            _post(
                base + "/api/v1/contact",
                {"name": "Ada", "email": "ada@example.com", "subject": f"msg-{i}", "message": "hi"},
            )
        status, body, _ = _get(base + "/api/v1/admin/contact-messages?limit=2&offset=1")
        assert status == 200
        d = json.loads(body)
        assert d["total"] == 5
        assert d["count"] == 2
        assert [m["subject"] for m in d["messages"]] == ["msg-3", "msg-2"]
    finally:
        httpd.shutdown()


# ── POST /api/v1/rotate-key ───────────────────────────────────────────────


def _active_store(tmp_path, monkeypatch, *, tier="brief"):
    """Note: sets TIRRA_PADDLE_WEBHOOK_SECRET so gated routes (e.g. GET
    /brief.json) actually enforce the key rather than serving dev-mode-open —
    rotate-key itself doesn't consult dev-mode at all (it always requires a
    real, currently-active SubscriberStore entry), but tests that check the
    OLD key stops working elsewhere need the gate genuinely on."""
    from agent.payments.handler import SubscriberStore

    monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "whatever")
    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_1", active=True, tier=tier)
    return store, store.api_key_of("sub_1")


def test_rotate_key_missing_header_401(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(base + "/api/v1/rotate-key", b"")
        assert status == 401
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_rotate_key_malformed_key_400(tmp_path):
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": "' OR 1=1 --"})
        assert status == 400
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_rotate_key_invalid_key_403(tmp_path, monkeypatch):
    _active_store(tmp_path, monkeypatch)
    httpd, base = _server(tmp_path)
    try:
        fake = "tirra_" + ("x" * 32)
        status, body, _ = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": fake})
        assert status == 403
        assert json.loads(body)["ok"] is False
    finally:
        httpd.shutdown()


def test_rotate_key_success_mints_new_key_and_kills_old(tmp_path, monkeypatch):
    store, old_key = _active_store(tmp_path, monkeypatch)
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": old_key})
        assert status == 200
        d = json.loads(body)
        assert d["ok"] is True
        new_key = d["api_key"]
        assert new_key != old_key
        assert new_key.startswith("tirra_")

        # Old key is dead immediately.
        assert store.is_active_key(old_key) is False
        # New key works for gated routes.
        assert store.is_active_key(new_key) is True

        status2, _, _ = _get(base + "/brief.json", headers={"X-Brief-Key": old_key})
        assert status2 == 403
        status3, _, _ = _get(base + "/brief.json", headers={"X-Brief-Key": new_key})
        assert status3 == 200
    finally:
        httpd.shutdown()


def test_rotate_key_rejects_query_string_key(tmp_path, monkeypatch):
    """Rotation is destructive — unlike other routes, ?key= is never accepted
    as a fallback; only the X-Brief-Key header authorises it."""
    _store, old_key = _active_store(tmp_path, monkeypatch)
    httpd, base = _server(tmp_path)
    try:
        status, body, _ = _post(base + f"/api/v1/rotate-key?key={old_key}", b"")
        assert status == 401
    finally:
        httpd.shutdown()


def test_rotate_key_per_key_rate_limited_429(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_ROTATE_KEY_LIMITER", _RateLimiter(max_calls=1, window_s=600))
    _store, old_key = _active_store(tmp_path, monkeypatch)
    httpd, base = _server(tmp_path)
    try:
        status1, body1, _ = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": old_key})
        assert status1 == 200
        new_key = json.loads(body1)["api_key"]
        # Same original key presented again (already dead) — still counted
        # against the per-key limiter before the store lookup even matters.
        status2, body2, resp = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": old_key})
        assert status2 == 429
        assert resp.headers.get("Retry-After") is not None
        assert new_key  # sanity: rotation before the limit did succeed
    finally:
        httpd.shutdown()


def test_rotate_key_per_ip_rate_limited_429_across_different_keys(tmp_path, monkeypatch):
    import agent.brief_server as bs

    monkeypatch.setattr(bs, "_ROTATE_IP_LIMITER", _RateLimiter(max_calls=1, window_s=600))
    from agent.payments.handler import SubscriberStore

    store = SubscriberStore(str(tmp_path / "subs.json"))
    monkeypatch.setattr("agent.payments.handler.SubscriberStore", lambda: store)
    store.set_active("sub_a", active=True, tier="brief")
    store.set_active("sub_b", active=True, tier="brief")
    key_a = store.api_key_of("sub_a")
    key_b = store.api_key_of("sub_b")

    httpd, base = _server(tmp_path)
    try:
        status1, _, _ = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": key_a})
        assert status1 == 200
        status2, body2, resp = _post(base + "/api/v1/rotate-key", b"", headers={"X-Brief-Key": key_b})
        assert status2 == 429
        assert json.loads(body2)["ok"] is False
    finally:
        httpd.shutdown()


# ── C3 follow-up: source allowlist gap cannot silently reappear ─────────────


def test_every_local_pipeline_source_is_allowlisted_or_classified_internal():
    """Regression guard for the 2026-08-27 P0 #3 triage: every distinct
    `source` value actually present in the repo-local pipeline DB must be
    either a real external source in `_external_source_allowlist()` OR
    explicitly classified as non-external in `_KNOWN_NON_EXTERNAL_SOURCES`
    (the exact 23-source set difference the triage enumerated). A source in
    neither bucket is NEW and UNCLASSIFIED — that is exactly the shape of
    gap that let an internal stage's raw defect state (train_gnn) become
    customer-queryable with zero code change, and it must fail the suite
    instead of silently reappearing.

    `.tirra_pipeline/pipeline.db` is gitignored local state, not a CI
    fixture — skip cleanly if it isn't present (fresh clone / CI box)
    rather than asserting on nothing.
    """
    import sqlite3

    import agent.brief_server as bs

    db_path = ".tirra_pipeline/pipeline.db"
    try:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute("SELECT DISTINCT source FROM pipeline_data").fetchall()
        finally:
            con.close()
    except sqlite3.OperationalError:
        pytest.skip(f"{db_path} not present locally — nothing to check")

    sources = {r[0] for r in rows}
    allowlist = bs._external_source_allowlist()
    known_internal = bs._KNOWN_NON_EXTERNAL_SOURCES

    unclassified = sources - allowlist - known_internal
    assert not unclassified, (
        f"Unclassified pipeline_data source(s) found: {sorted(unclassified)!r} — each must be "
        "either added to _external_source_allowlist() (if it's real, customer-facing data) or "
        "added to _KNOWN_NON_EXTERNAL_SOURCES with a comment explaining why it's internal/dead "
        "(see the 2026-08-27 P0 #3 triage classes A/B/C above _MANUAL_STORE_EXTERNAL_SOURCES)."
    )
