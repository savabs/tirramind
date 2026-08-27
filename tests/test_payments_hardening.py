"""Hardening tests: webhook replay guard, SubscriberStore read cache,
key rotation/revocation, and claim-flow store support.

Owner: payments (agent/payments/*). This is the ONLY test file this agent
is authorized to create — do not extend/modify tests/test_payments.py here.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import time
from typing import Any

import pytest

from agent.payments.claim import ClaimStore, claim_transaction
from agent.payments.delivery import (
    DeliveryResult,
    attempt_deliver_key_email,
    is_configured,
    send_api_key_email,
)
from agent.payments.event_ledger import ProcessedEventLedger
from agent.payments.handler import PaddleWebhookHandler, SubscriberStore
from agent.payments.webhook import WebhookVerificationError, verify_webhook_signature

_SECRET = "pdl_ntfset_test_01secretkeybytes"  # noqa: S105 — test fixture, not a real credential


def _sign_body(secret: str, body: bytes, ts: int) -> str:
    message = f"{ts}:{body.decode()}".encode()
    sig = hmac_mod.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"ts={ts};h1={sig}"


# ── C5: webhook replay ──────────────────────────────────────────────────────


class TestReplayWindowTightened:
    def test_default_max_age_is_300s_not_3600s(self):
        """The old 1-hour default let a captured, still-valid-looking webhook
        be replayed up to an hour later. Confirm the tightened default."""
        body = b"x"
        ts = int(time.time()) - 600  # 10 minutes old
        sig = _sign_body(_SECRET, body, ts)
        with pytest.raises(WebhookVerificationError, match="stale"):
            verify_webhook_signature(body=body, signature_header=sig, secret=_SECRET)

    def test_within_300s_still_accepted(self):
        body = b"x"
        ts = int(time.time()) - 200  # within the new default window
        sig = _sign_body(_SECRET, body, ts)
        assert verify_webhook_signature(body=body, signature_header=sig, secret=_SECRET) is True

    def test_old_hour_long_window_no_longer_the_default(self):
        """A webhook 50 minutes old (would have passed the old 3600s default)
        must now be rejected by the DEFAULT (no explicit override)."""
        body = b"x"
        ts = int(time.time()) - (50 * 60)
        sig = _sign_body(_SECRET, body, ts)
        with pytest.raises(WebhookVerificationError, match="stale"):
            verify_webhook_signature(body=body, signature_header=sig, secret=_SECRET)


class TestProcessedEventLedger:
    def test_first_seen_returns_true_and_marks(self, tmp_path):
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        assert ledger.check_and_mark("evt_1") is True
        assert ledger.seen("evt_1") is True

    def test_replay_of_same_event_id_returns_false(self, tmp_path):
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        assert ledger.check_and_mark("evt_1") is True
        assert ledger.check_and_mark("evt_1") is False  # replay/retry — do not reprocess

    def test_survives_reinstantiation_ie_process_restart(self, tmp_path):
        path = str(tmp_path / "events.json")
        ledger1 = ProcessedEventLedger(path)
        ledger1.mark_seen("evt_1")
        # Simulate a process restart: brand-new instance, same file.
        ledger2 = ProcessedEventLedger(path)
        assert ledger2.seen("evt_1") is True
        assert ledger2.check_and_mark("evt_1") is False

    def test_bounded_by_age(self, tmp_path):
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"), retention_s=10.0)
        base = 1_000_000.0
        ledger.mark_seen("evt_old", now=base)
        # Long after retention_s has elapsed, the old event should be pruned
        # and therefore treated as unseen (bounded growth, not infinite).
        assert ledger.seen("evt_old", now=base + 1000.0) is False
        assert ledger.check_and_mark("evt_old", now=base + 1000.0) is True

    def test_bounded_by_max_entries(self, tmp_path):
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"), retention_s=1e9, max_entries=5)
        base = 1_000_000.0
        for i in range(10):
            ledger.mark_seen(f"evt_{i}", now=base + i)
        assert len(ledger.all()) <= 5
        # The oldest entries should have been evicted first.
        assert "evt_0" not in ledger.all()
        assert "evt_9" in ledger.all()

    def test_missing_event_id_never_blocks_processing(self, tmp_path):
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        # No event_id to key on — must not raise, must not block.
        assert ledger.check_and_mark("") is True
        assert ledger.check_and_mark("") is True


class TestHandlerRejectsReplayedEvent:
    def test_replayed_event_id_is_a_noop_not_a_reactivation(self, tmp_path):
        """The scenario from the brief: a captured subscription.activated is
        replayed after a cancellation. Confirm it does NOT re-activate."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        handler = PaddleWebhookHandler(secret="", store=store, event_ledger=ledger)

        activate_body = json.dumps(
            {
                "event_id": "evt_captured_1",
                "event_type": "subscription.activated",
                "data": {"id": "sub_1", "customer": {}},
            }
        ).encode()
        res1 = handler.handle(body=activate_body, signature_header="")
        assert res1["handled"] is True
        assert store.is_active("sub_1") is True

        # Legitimate cancellation happens next (different event_id).
        cancel_body = json.dumps(
            {
                "event_id": "evt_cancel_1",
                "event_type": "subscription.canceled",
                "data": {"id": "sub_1", "customer": {}},
            }
        ).encode()
        handler.handle(body=cancel_body, signature_header="")
        assert store.is_active("sub_1") is False

        # Attacker replays the captured activation webhook byte-for-byte.
        res2 = handler.handle(body=activate_body, signature_header="")
        assert res2["handled"] is False
        assert "duplicate" in res2["reason"]
        assert store.is_active("sub_1") is False  # must NOT have been re-activated

    def test_benign_retry_of_same_event_is_idempotent_noop(self, tmp_path):
        """Paddle retries webhooks; a retry re-sends the SAME event_id. That
        must not double-anything (still just one active subscriber row)."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        handler = PaddleWebhookHandler(secret="", store=store, event_ledger=ledger)
        body = json.dumps(
            {"event_id": "evt_1", "event_type": "subscription.activated", "data": {"id": "sub_1", "customer": {}}}
        ).encode()
        first = handler.handle(body=body, signature_header="")
        second = handler.handle(body=body, signature_header="")
        assert first["handled"] is True
        assert second["handled"] is False
        assert len(store.all()) == 1

    def test_payload_without_event_id_still_processes(self, tmp_path):
        """Backward compatible: a payload with no event_id can't be deduped,
        but must still be applied (existing behavior/tests rely on this)."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        handler = PaddleWebhookHandler(secret="", store=store, event_ledger=ledger)
        body = json.dumps({"event_type": "subscription.activated", "data": {"id": "sub_1", "customer": {}}}).encode()
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        assert store.is_active("sub_1") is True


# ── SubscriberStore read cache ──────────────────────────────────────────────


class TestSubscriberStoreCache:
    def test_write_is_immediately_visible_to_a_new_instance(self, tmp_path):
        """A newly activated subscriber must not be locked out by a stale
        cache: a brand-new SubscriberStore() created right after a write
        must see the write immediately (write-through, not TTL-gated)."""
        path = str(tmp_path / "subs.json")
        writer = SubscriberStore(path)
        writer.set_active("sub_1", active=True, tier="data")
        key = writer.api_key_of("sub_1")

        reader = SubscriberStore(path)  # fresh instance, same process
        assert reader.is_active_key(key) is True
        assert reader.tier_of_key(key) == "data"

    def test_cancellation_is_immediately_visible_to_a_new_instance(self, tmp_path):
        """A cancelled subscriber must lose access promptly, not after a
        cache TTL expires."""
        path = str(tmp_path / "subs.json")
        writer = SubscriberStore(path)
        writer.set_active("sub_1", active=True, tier="data")
        key = writer.api_key_of("sub_1")
        assert SubscriberStore(path).is_active_key(key) is True

        writer.set_active("sub_1", active=False)
        reader = SubscriberStore(path)
        assert reader.is_active_key(key) is False

    def test_cache_avoids_a_disk_read_within_ttl(self, tmp_path, monkeypatch):
        """Within the TTL, a new instance must reuse the cached parse rather
        than re-reading the file from disk."""
        path = tmp_path / "subs.json"
        writer = SubscriberStore(str(path))
        writer.set_active("sub_1", active=True, tier="data")

        original_read_text = type(path).read_text
        calls = {"n": 0}

        def counting_read_text(self, *args, **kwargs):
            calls["n"] += 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(type(path), "read_text", counting_read_text)
        # Two more instances within the TTL window should not re-read the file.
        SubscriberStore(str(path))
        SubscriberStore(str(path))
        assert calls["n"] == 0

    def test_disabling_cache_ttl_always_reads_disk(self, tmp_path):
        """cache_ttl_s=0 opts out of caching entirely (used where strict
        read-your-writes-from-disk semantics matter more than avoiding a
        read, or for tests wanting zero cross-instance cache aliasing)."""
        path = str(tmp_path / "subs.json")
        writer = SubscriberStore(path, cache_ttl_s=0)
        writer.set_active("sub_1", active=True)
        # Mutate the file out from under the store directly.
        import json as _json

        raw = _json.loads((tmp_path / "subs.json").read_text())
        raw["sub_1"]["active"] = False
        (tmp_path / "subs.json").write_text(_json.dumps(raw))

        reader = SubscriberStore(path, cache_ttl_s=0)
        assert reader.is_active("sub_1") is False

    def test_mutating_one_instance_entry_does_not_corrupt_a_cached_sibling(self, tmp_path):
        """Entries are mutated in place (set_active does entry[...] = ...).
        A shallow cache copy would let that mutation leak into another
        instance's view before either has saved — must be deep-copied."""
        path = str(tmp_path / "subs.json")
        a = SubscriberStore(path)
        a.set_active("sub_1", active=True, tier="data")

        b = SubscriberStore(path)  # may come from cache
        b_entry_tier_before = b.tier_of("sub_1")
        # Mutate via a fresh instance c, must not silently alter b's view.
        c = SubscriberStore(path)
        c.set_active("sub_1", active=True, tier="entity")

        assert b.tier_of("sub_1") == b_entry_tier_before  # b's snapshot is untouched


# ── Key rotation / revocation ────────────────────────────────────────────────


class TestKeyRotation:
    def test_rotate_key_by_subscription_id_changes_the_key(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        old_key = store.api_key_of("sub_1")

        new_key = store.rotate_key("sub_1")
        assert new_key is not None
        assert new_key != old_key
        assert store.api_key_of("sub_1") == new_key

    def test_old_key_stops_working_after_rotation(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        old_key = store.api_key_of("sub_1")
        store.rotate_key("sub_1")
        assert store.is_active_key(old_key) is False

    def test_new_key_works_after_rotation(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        new_key = store.rotate_key("sub_1")
        assert store.is_active_key(new_key) is True
        assert store.tier_of_key(new_key) == "data"

    def test_rotate_unknown_subscription_returns_none(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        assert store.rotate_key("sub_does_not_exist") is None

    def test_self_service_rotate_by_api_key(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        old_key = store.api_key_of("sub_1")

        new_key = store.rotate_key_for_api_key(old_key)
        assert new_key is not None
        assert new_key != old_key
        assert store.is_active_key(old_key) is False
        assert store.is_active_key(new_key) is True

    def test_self_service_rotate_rejects_unknown_key(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        assert store.rotate_key_for_api_key("tirra_not_real") is None

    def test_self_service_rotate_rejects_inactive_subscriber(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        key = store.api_key_of("sub_1")
        store.set_active("sub_1", active=False)
        assert store.rotate_key_for_api_key(key) is None

    def test_revoke_key_invalidates_without_minting_replacement(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        key = store.api_key_of("sub_1")
        assert store.revoke_key("sub_1") is True
        assert store.is_active_key(key) is False
        assert store.api_key_of("sub_1") is None
        # Subscriber remains active (revocation ≠ deactivation).
        assert store.is_active("sub_1") is True

    def test_revoke_unknown_subscription_returns_false(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        assert store.revoke_key("sub_nope") is False


# ── Claim-flow store support ─────────────────────────────────────────────────


class _FakePaddleClient:
    """Stand-in for PaddleClient in claim tests — no network calls."""

    def __init__(self, transactions: dict[str, dict]):
        self._transactions = transactions

    def get_transaction(self, transaction_id: str) -> dict:
        import httpx

        if transaction_id not in self._transactions:
            request = httpx.Request("GET", f"https://sandbox-api.paddle.com/transactions/{transaction_id}")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        return self._transactions[transaction_id]


class TestClaimTransaction:
    def test_unknown_transaction(self, tmp_path):
        client = _FakePaddleClient({})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_ghost", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "unknown_transaction"

    def test_not_completed(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "billed", "subscription_id": None}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "not_completed"
        assert result.transaction_status == "billed"

    def test_completed_but_no_subscription_id_yet_is_pending(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": None}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "pending"

    def test_completed_subscription_not_yet_in_store_is_pending(self, tmp_path):
        """The webhook race: transaction completed with a subscription_id,
        but the webhook hasn't landed in SubscriberStore yet."""
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "pending"

    def test_subscriber_inactive(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=False, tier="data")
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "subscriber_inactive"

    def test_first_claim_returns_api_key(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        expected_key = store.api_key_of("sub_1")
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.status == "claimed"
        assert result.api_key == expected_key
        assert result.tier == "data"
        assert result.subscription_id == "sub_1"

    def test_repeat_claim_within_window_returns_same_key(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        expected_key = store.api_key_of("sub_1")
        claims = ClaimStore(str(tmp_path / "claims.json"))
        first = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        second = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert first.status == "claimed"
        assert second.status == "claimed"
        assert second.api_key == expected_key

    def test_claim_outside_window_is_already_claimed_without_key(self, tmp_path):
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        claims = ClaimStore(str(tmp_path / "claims.json"))
        base = 1_000_000.0
        first = claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims, now=base)
        assert first.status == "claimed"
        later = claim_transaction(
            "txn_1", paddle_client=client, subscriber_store=store, claim_store=claims, now=base + 16 * 60
        )
        assert later.status == "already_claimed"
        assert later.api_key is None

    def test_hammering_within_window_eventually_already_claimed(self, tmp_path):
        """Even inside the 15-minute window, a script re-claiming the same
        txn_id many times must eventually be refused the key."""
        client = _FakePaddleClient({"txn_1": {"status": "completed", "subscription_id": "sub_1"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data")
        claims = ClaimStore(str(tmp_path / "claims.json"))
        base = 1_000_000.0
        results = [
            claim_transaction("txn_1", paddle_client=client, subscriber_store=store, claim_store=claims, now=base + i)
            for i in range(8)
        ]
        assert results[0].status == "claimed"
        assert any(r.status == "already_claimed" for r in results)
        # Once already_claimed, no result may leak the api_key.
        assert all(r.api_key is None for r in results if r.status == "already_claimed")

    def test_unknown_transaction_never_leaks_a_key(self, tmp_path):
        client = _FakePaddleClient({})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_ghost", paddle_client=client, subscriber_store=store, claim_store=claims)
        assert result.api_key is None

    def test_upstream_error_on_unexpected_exception(self, tmp_path, monkeypatch):
        class _BrokenClient:
            def get_transaction(self, transaction_id: str) -> dict:
                raise RuntimeError("connection reset")

        store = SubscriberStore(str(tmp_path / "subs.json"))
        claims = ClaimStore(str(tmp_path / "claims.json"))
        result = claim_transaction("txn_1", paddle_client=_BrokenClient(), subscriber_store=store, claim_store=claims)
        assert result.status == "upstream_error"
        assert result.api_key is None


class TestClaimStore:
    def test_get_unknown_returns_none(self, tmp_path):
        store = ClaimStore(str(tmp_path / "claims.json"))
        assert store.get("txn_nope") is None

    def test_record_claim_creates_entry_with_count_1(self, tmp_path):
        store = ClaimStore(str(tmp_path / "claims.json"))
        entry = store.record_claim("txn_1", subscription_id="sub_1", now=1000.0)
        assert entry["claim_count"] == 1
        assert entry["first_claimed_at"] == 1000.0

    def test_record_claim_bumps_count_and_preserves_first_claimed_at(self, tmp_path):
        store = ClaimStore(str(tmp_path / "claims.json"))
        store.record_claim("txn_1", subscription_id="sub_1", now=1000.0)
        entry = store.record_claim("txn_1", subscription_id="sub_1", now=1010.0)
        assert entry["claim_count"] == 2
        assert entry["first_claimed_at"] == 1000.0  # anchors the idempotency window

    def test_survives_reinstantiation(self, tmp_path):
        path = str(tmp_path / "claims.json")
        ClaimStore(path).record_claim("txn_1", subscription_id="sub_1", now=1000.0)
        reloaded = ClaimStore(path)
        entry = reloaded.get("txn_1")
        assert entry is not None
        assert entry["subscription_id"] == "sub_1"


# ── Email delivery groundwork (P2 #11 second channel) ────────────────────────


class TestDeliveryUnconfigured:
    """No SMTP env vars exist in this repo/CI — delivery MUST no-op loudly,
    never raise, and never be mistaken for success."""

    def test_is_configured_false_with_empty_env(self):
        assert is_configured(env={}) is False

    def test_is_configured_false_with_partial_env(self):
        partial = {
            "TIRRA_SMTP_HOST": "smtp.example.com",
            "TIRRA_SMTP_USERNAME": "user",
            # missing TIRRA_SMTP_PASSWORD and TIRRA_SMTP_FROM
        }
        assert is_configured(env=partial) is False

    def test_is_configured_true_with_full_env(self):
        full = {
            "TIRRA_SMTP_HOST": "smtp.example.com",
            "TIRRA_SMTP_USERNAME": "user",
            "TIRRA_SMTP_PASSWORD": "hunter2",
            "TIRRA_SMTP_FROM": "no-reply@tirramind.com",
        }
        assert is_configured(env=full) is True

    def test_send_with_no_env_returns_unconfigured_not_an_exception(self, caplog):
        result = send_api_key_email("customer@example.com", api_key="tirra_abc", tier="data", env={})
        assert isinstance(result, DeliveryResult)
        assert result.status == "unconfigured"
        assert result.ok is False

    def test_send_with_no_recipient_returns_unconfigured(self):
        result = send_api_key_email("", api_key="tirra_abc", tier="data", env={"TIRRA_SMTP_HOST": "x"})
        assert result.status == "unconfigured"

    def test_missing_vars_are_named_in_the_result_detail(self):
        result = send_api_key_email(
            "customer@example.com",
            api_key="tirra_abc",
            tier="data",
            env={"TIRRA_SMTP_HOST": "smtp.example.com"},
        )
        assert result.status == "unconfigured"
        assert "TIRRA_SMTP_USERNAME" in result.detail
        assert "TIRRA_SMTP_PASSWORD" in result.detail
        assert "TIRRA_SMTP_FROM" in result.detail

    def test_never_raises_even_with_garbage_env(self):
        garbage = {
            "TIRRA_SMTP_HOST": "smtp.example.com",
            "TIRRA_SMTP_USERNAME": "user",
            "TIRRA_SMTP_PASSWORD": "pw",
            "TIRRA_SMTP_FROM": "no-reply@tirramind.com",
            "TIRRA_SMTP_PORT": "not-a-port",
        }
        # Should return an error DeliveryResult, not raise.
        result = send_api_key_email("customer@example.com", api_key="tirra_abc", tier="data", env=garbage)
        assert result.status == "error"


class TestDeliverySendPath:
    """When fully configured, the module must actually attempt an SMTP send
    (mocked here — no real network, no real credentials) rather than silently
    treating 'configured' as 'sent'."""

    _FULL_ENV = {
        "TIRRA_SMTP_HOST": "smtp.example.com",
        "TIRRA_SMTP_PORT": "587",
        "TIRRA_SMTP_USERNAME": "user",
        "TIRRA_SMTP_PASSWORD": "pw",  # noqa: S105 — test fixture, not a real credential
        "TIRRA_SMTP_FROM": "no-reply@tirramind.com",
    }

    def test_configured_send_calls_smtp_login_and_send_message(self, monkeypatch):
        calls: dict[str, Any] = {}

        class _FakeSMTP:
            def __init__(self, host, port, timeout=None):
                calls["host"] = host
                calls["port"] = port

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def starttls(self):
                calls["starttls"] = True

            def login(self, username, password):
                calls["login"] = (username, password)

            def send_message(self, message):
                calls["sent_to"] = message["To"]
                calls["sent_from"] = message["From"]
                calls["body"] = message.get_content()

        import agent.payments.delivery as delivery_mod

        monkeypatch.setattr(delivery_mod.smtplib, "SMTP", _FakeSMTP)

        result = send_api_key_email("customer@example.com", api_key="tirra_secretkey", tier="data", env=self._FULL_ENV)
        assert result.status == "sent"
        assert result.ok is True
        assert calls["host"] == "smtp.example.com"
        assert calls["port"] == 587
        assert calls["starttls"] is True
        assert calls["login"] == ("user", "pw")
        assert calls["sent_to"] == "customer@example.com"
        assert "tirra_secretkey" in calls["body"]

    def test_smtp_failure_is_caught_and_reported_as_error_not_raised(self, monkeypatch):
        class _ExplodingSMTP:
            def __init__(self, host, port, timeout=None):
                pass

            def __enter__(self):
                raise ConnectionRefusedError("no relay reachable")

            def __exit__(self, *exc):
                return False

        import agent.payments.delivery as delivery_mod

        monkeypatch.setattr(delivery_mod.smtplib, "SMTP", _ExplodingSMTP)

        result = send_api_key_email("customer@example.com", api_key="tirra_secretkey", tier="data", env=self._FULL_ENV)
        assert result.status == "error"
        assert result.ok is False

    def test_use_tls_false_skips_starttls(self, monkeypatch):
        calls: dict[str, Any] = {}

        class _FakeSMTP:
            def __init__(self, host, port, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def starttls(self):
                calls["starttls_called"] = True

            def login(self, username, password):
                pass

            def send_message(self, message):
                pass

        import agent.payments.delivery as delivery_mod

        monkeypatch.setattr(delivery_mod.smtplib, "SMTP", _FakeSMTP)

        env = dict(self._FULL_ENV, TIRRA_SMTP_USE_TLS="false")
        result = send_api_key_email("customer@example.com", api_key="tirra_secretkey", tier="data", env=env)
        assert result.status == "sent"
        assert "starttls_called" not in calls


class TestAttemptDeliverKeyEmailGuard:
    """attempt_deliver_key_email is the webhook-path entry point — it must be
    unconditionally safe to call even with hostile/missing inputs."""

    def test_no_api_key_short_circuits_without_touching_smtp(self, monkeypatch):
        import agent.payments.delivery as delivery_mod

        def _boom(*a, **kw):
            raise AssertionError("send_api_key_email should not have been called")

        monkeypatch.setattr(delivery_mod, "send_api_key_email", _boom)
        result = attempt_deliver_key_email("customer@example.com", api_key=None, tier="data")
        assert result.status == "unconfigured"

    def test_none_email_does_not_raise(self):
        result = attempt_deliver_key_email(None, api_key="tirra_abc", tier="data", env={})
        assert result.status == "unconfigured"

    def test_unexpected_exception_from_send_is_swallowed(self, monkeypatch):
        import agent.payments.delivery as delivery_mod

        def _boom(*a, **kw):
            raise RuntimeError("something totally unexpected")

        monkeypatch.setattr(delivery_mod, "send_api_key_email", _boom)
        result = attempt_deliver_key_email("customer@example.com", api_key="tirra_abc", tier="data")
        assert result.status == "error"


class TestHandlerDeliveryWiring:
    """The webhook handler must attempt delivery exactly once, on the event
    that actually mints a key — never crash the activation path even if
    delivery itself fails, and never fire again on a later renewal/update
    webhook for a subscriber that already has a key."""

    def test_activation_attempts_delivery_exactly_once(self, tmp_path, monkeypatch):
        calls = []

        def _fake_deliver(email, *, api_key, tier):
            calls.append((email, api_key, tier))
            from agent.payments.delivery import DeliveryResult

            return DeliveryResult(status="unconfigured", detail="test stub")

        import agent.payments.handler as handler_mod

        monkeypatch.setattr(handler_mod, "attempt_deliver_key_email", _fake_deliver)

        store = SubscriberStore(str(tmp_path / "subs.json"))
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        handler = PaddleWebhookHandler(secret="", store=store, event_ledger=ledger)

        body = json.dumps(
            {
                "event_id": "evt_1",
                "event_type": "subscription.activated",
                "data": {"id": "sub_1", "customer": {"email": "customer@example.com"}},
            }
        ).encode()
        handler.handle(body=body, signature_header="")
        assert len(calls) == 1
        assert calls[0][0] == "customer@example.com"

        # A subsequent renewal/update event for the SAME subscriber (already
        # has a key) must NOT attempt delivery again.
        update_body = json.dumps(
            {
                "event_id": "evt_2",
                "event_type": "subscription.updated",
                "data": {"id": "sub_1", "customer": {"email": "customer@example.com"}},
            }
        ).encode()
        handler.handle(body=update_body, signature_header="")
        assert len(calls) == 1  # unchanged

    def test_delivery_failure_never_breaks_activation(self, tmp_path, monkeypatch):
        """Even if the delivery layer somehow raised (it shouldn't — see its
        own guard), the webhook handler's job (store write, ack) must not be
        endangered by it. This wires in a raising stub directly at the
        handler's import site to prove the activation path survives it."""

        def _raising_deliver(*a, **kw):
            raise RuntimeError("smtp relay is on fire")

        import agent.payments.handler as handler_mod

        # Real attempt_deliver_key_email already swallows everything itself;
        # this test simulates the pathological case where that guarantee is
        # somehow violated, to prove the store write already happened first
        # (delivery is attempted AFTER the store write, not before/blocking).
        store = SubscriberStore(str(tmp_path / "subs.json"))
        ledger = ProcessedEventLedger(str(tmp_path / "events.json"))
        handler = PaddleWebhookHandler(secret="", store=store, event_ledger=ledger)

        monkeypatch.setattr(handler_mod, "attempt_deliver_key_email", _raising_deliver)

        body = json.dumps(
            {
                "event_id": "evt_1",
                "event_type": "subscription.activated",
                "data": {"id": "sub_1", "customer": {"email": "customer@example.com"}},
            }
        ).encode()
        with pytest.raises(RuntimeError):
            handler.handle(body=body, signature_header="")
        # The store write must have already happened before delivery blew up —
        # a real customer's activation state is not lost even in this
        # pathological scenario.
        assert store.is_active("sub_1") is True
        assert store.api_key_of("sub_1") is not None
