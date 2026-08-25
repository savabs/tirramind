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


def _ed25519_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes_raw().hex()
    return private, public_hex


def _sign_body(private_key, body: bytes, ts: int) -> str:
    message = f"{ts}:{body.decode()}".encode()
    sig = private_key.sign(message).hex()
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
        private, pub = _ed25519_keypair()
        body = b'{"event_type":"subscription.activated"}'
        ts = int(time.time())
        sig = _sign_body(private, body, ts)
        assert verify_webhook_signature(body=body, signature_header=sig, secret=pub) is True

    def test_missing_secret_raises(self):
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(body=b"x", signature_header="ts=1;h1=2", secret="")

    def test_missing_header_raises(self):
        _, pub = _ed25519_keypair()
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(body=b"x", signature_header="", secret=pub)

    def test_tampered_body_rejected(self):
        private, pub = _ed25519_keypair()
        body = b'{"event_type":"subscription.activated"}'
        ts = int(time.time())
        sig = _sign_body(private, body, ts)
        with pytest.raises(WebhookVerificationError):
            verify_webhook_signature(body=b'{"event_type":"subscription.canceled"}',
                                     signature_header=sig, secret=pub)

    def test_stale_timestamp_rejected(self):
        private, pub = _ed25519_keypair()
        body = b"x"
        old_ts = int(time.time()) - 10000
        sig = _sign_body(private, body, old_ts)
        with pytest.raises(WebhookVerificationError, match="stale"):
            verify_webhook_signature(body=body, signature_header=sig, secret=pub,
                                     now=time.time())


class TestHandler:
    def test_activated_grants_access(self, tmp_path):
        private, pub = _ed25519_keypair()
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=pub, store=store)
        body = json.dumps({
            "event_type": "subscription.activated",
            "data": {"id": "sub_123", "customer": {"id": "ctm_1", "email": "a@b.com"}},
        }).encode()
        sig = _sign_body(private, body, int(time.time()))
        res = handler.handle(body=body, signature_header=sig)
        assert res["handled"] is True
        assert res["active"] is True
        assert store.is_active("sub_123") is True

    def test_canceled_revokes_access(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        store.set_active("sub_9", active=True)
        body = json.dumps({"event_type": "subscription.canceled",
                           "data": {"id": "sub_9", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        assert store.is_active("sub_9") is False

    def test_skips_verification_when_no_secret(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret="", store=store)
        body = json.dumps({"event_type": "subscription.activated",
                           "data": {"id": "sub_5", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        assert store.is_active("sub_5") is True

    def test_invalid_signature_rejected(self, tmp_path):
        _, pub = _ed25519_keypair()
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=pub, store=store)
        body = json.dumps({"event_type": "subscription.activated",
                           "data": {"id": "sub_7", "customer": {}}}).encode()
        with pytest.raises(WebhookVerificationError):
            handler.handle(body=body, signature_header="ts=1;h1=deadbeef")
