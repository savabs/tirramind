"""Tests for the Paddle payments integration (config, webhook, handler)."""

from __future__ import annotations

import json
import time

import pytest

from agent.payments.config import PaddleConfig, PaddleConfigError
from agent.payments.handler import PaddleWebhookHandler, SubscriberStore
from agent.payments.webhook import (
    WebhookVerificationError,
    verify_webhook_signature,
)

_SECRET = "pdl_ntfset_test_01secretkeybytes"  # noqa: S105 — test fixture, not a real credential


def _sign_body(secret: str, body: bytes, ts: int) -> str:
    import hashlib
    import hmac as hmac_mod

    message = f"{ts}:{body.decode()}".encode()
    sig = hmac_mod.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"ts={ts};h1={sig}"


class TestConfig:
    def test_sandbox_default(self, monkeypatch):
        monkeypatch.delenv("TIRRA_PADDLE_MODE", raising=False)
        c = PaddleConfig.from_env()
        assert c.mode == "sandbox"
        assert c.api_base == "https://sandbox-api.paddle.com"
        assert not c.is_live

    def test_live_endpoint_and_requires_secret(self, monkeypatch):
        monkeypatch.setenv("TIRRA_PADDLE_MODE", "live")
        monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "")
        with pytest.raises(PaddleConfigError, match="WEBHOOK_SECRET"):
            PaddleConfig.from_env()

    def test_live_endpoint(self, monkeypatch):
        monkeypatch.setenv("TIRRA_PADDLE_MODE", "live")
        monkeypatch.setenv("TIRRA_PADDLE_WEBHOOK_SECRET", "x" * 64)
        monkeypatch.setenv("TIRRA_PADDLE_API_KEY", "k")
        monkeypatch.setenv("TIRRA_PADDLE_CLIENT_TOKEN", "live_123")
        c = PaddleConfig.from_env()
        assert c.api_base == "https://api.paddle.com"
        assert c.is_live
        assert "api_key" not in c.to_public_dict()

    def test_invalid_mode(self, monkeypatch):
        monkeypatch.setenv("TIRRA_PADDLE_MODE", "staging")
        with pytest.raises(PaddleConfigError):
            PaddleConfig.from_env()


class TestWebhookSignature:
    def test_valid_signature_passes(self):
        body = b'{"event_type":"subscription.activated"}'
        ts = int(time.time())
        sig = _sign_body(_SECRET, body, ts)
        assert verify_webhook_signature(body=body, signature_header=sig, secret=_SECRET) is True

    def test_missing_secret_raises(self):
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(body=b"x", signature_header="ts=1;h1=2", secret="")

    def test_missing_header_raises(self):
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(body=b"x", signature_header="", secret=_SECRET)

    def test_tampered_body_rejected(self):
        body = b'{"event_type":"subscription.activated"}'
        ts = int(time.time())
        sig = _sign_body(_SECRET, body, ts)
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(
                body=b'{"event_type":"subscription.canceled"}', signature_header=sig, secret=_SECRET
            )

    def test_stale_timestamp_rejected(self):
        body = b"x"
        old_ts = int(time.time()) - 10000
        sig = _sign_body(_SECRET, body, old_ts)
        with pytest.raises(WebhookVerificationError, match="stale"):
            verify_webhook_signature(body=body, signature_header=sig, secret=_SECRET, now=time.time())


class TestHandler:
    def test_activated_grants_access(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        body = json.dumps(
            {
                "event_type": "subscription.activated",
                "data": {"id": "sub_123", "customer": {"id": "ctm_1", "email": "a@b.com"}},
            }
        ).encode()
        sig = _sign_body(_SECRET, body, int(time.time()))
        res = handler.handle(body=body, signature_header=sig)
        assert res["handled"] is True
        assert res["active"] is True
        assert store.is_active("sub_123") is True

    def test_canceled_revokes_access(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        store.set_active("sub_9", active=True)
        body = json.dumps({"event_type": "subscription.canceled", "data": {"id": "sub_9", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        assert store.is_active("sub_9") is False

    def test_skips_verification_when_no_secret(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        body = json.dumps({"event_type": "subscription.activated", "data": {"id": "sub_5", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        assert store.is_active("sub_5") is True

    def test_invalid_signature_rejected(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        body = json.dumps({"event_type": "subscription.activated", "data": {"id": "sub_7", "customer": {}}}).encode()
        with pytest.raises(WebhookVerificationError):
            handler.handle(body=body, signature_header="ts=1;h1=deadbeef")

    def test_activation_mints_api_key_returned_in_result(self, tmp_path):
        """The webhook result carries the freshly-minted opaque key (for
        support/ops tooling) — but never the raw subscription_id as the
        customer-facing credential."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        body = json.dumps({"event_type": "subscription.activated", "data": {"id": "sub_42", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["api_key"].startswith("tirra_")
        assert res["api_key"] == store.api_key_of("sub_42")

    def test_tier_resolved_from_price_map(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIRRA_TIER_PRICE_MAP", "pri_data123:data,pri_entity456:entity")
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        body = json.dumps(
            {
                "event_type": "subscription.activated",
                "data": {"id": "sub_data_1", "customer": {}, "items": [{"price": {"id": "pri_data123"}}]},
            }
        ).encode()
        handler.handle(body=body, signature_header="")
        assert store.tier_of("sub_data_1") == "data"

    def test_unmapped_price_defaults_to_brief_tier(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        body = json.dumps(
            {"event_type": "subscription.activated", "data": {"id": "sub_legacy", "customer": {}}}
        ).encode()
        handler.handle(body=body, signature_header="")
        assert store.tier_of("sub_legacy") == "brief"


class TestApiKeyLookup:
    """Customer-facing auth resolves by the opaque api_key, not subscription_id."""

    def test_is_active_key_true_for_active_subscriber(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is True
        assert store.tier_of_key(key) == "data"

    def test_is_active_key_false_after_deactivation(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        key = store.api_key_of("sub_1")
        store.set_active("sub_1", active=False)
        assert store.is_active_key(key) is False

    def test_is_active_key_false_for_unknown_key(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        assert store.is_active_key("tirra_not_a_real_key") is False

    def test_raw_subscription_id_is_not_a_valid_key(self, tmp_path):
        """The subscription_id itself must never work as an API key."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        assert store.is_active_key("sub_1") is False

    def test_key_stable_across_reactivation(self, tmp_path):
        """Canceling and reactivating keeps the same key — no silent rotation."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        key = store.api_key_of("sub_1")
        store.set_active("sub_1", active=False)
        store.set_active("sub_1", active=True)
        assert store.api_key_of("sub_1") == key

    def test_active_keys_returns_api_keys_not_subscription_ids(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        store.set_active("sub_2", active=False, tier="entity")
        keys = store.active_keys()
        assert keys == [store.api_key_of("sub_1")]
