"""Subscriber lifecycle correctness tests — the money-correctness checklist.

Owner: payments (agent/payments/handler.py, agent/payments/client.py). This
file is authorized in addition to tests/test_payments.py and
tests/test_payments_hardening.py (which this agent does not modify).

Every test class below states, in its docstring, whether it would FAIL
against the pre-fix code — a test that passes before and after a fix proves
nothing (per the money-correctness standard of evidence).
"""

from __future__ import annotations

import json
import time
from datetime import UTC

import httpx
import pytest

from agent.payments.client import PaddleClient
from agent.payments.config import PaddleConfig
from agent.payments.handler import PaddleWebhookHandler, SubscriberStore

_SECRET = ""  # verification disabled for these tests — webhook.py/verify logic is covered elsewhere


def _body(event_type: str, data: dict) -> bytes:
    return json.dumps({"event_type": event_type, "data": data}).encode()


def _real_shape_activated(subscription_id: str, customer_id: str, *, period_ends_at: str | None = None) -> dict:
    """A payload shaped exactly like Paddle's real subscription.activated
    example (developer.paddle.com, verified 2026-08-28): a FLAT
    `customer_id`, no nested `customer` object, no email anywhere.
    """
    d: dict = {"id": subscription_id, "status": "active", "customer_id": customer_id}
    if period_ends_at:
        d["current_billing_period"] = {"starts_at": "2024-04-12T10:18:47Z", "ends_at": period_ends_at}
    return d


# ── TASK 1: customer_id / email extraction ──────────────────────────────────


class TestCustomerIdExtractionRealShape:
    """BUG (A): handler.py read data.get("customer", {}).get("id") — real
    Paddle webhooks carry a flat data.customer_id and no nested `customer`
    object at all. Against the UNFIXED code, `customer_id` extraction from
    this exact real-shaped payload silently resolves to None on every
    delivery ever received. These tests FAIL on the unfixed code (assertion
    on store.get(...)["customer_id"] would be None instead of "ctm_1").
    """

    def test_flat_customer_id_is_captured(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        body = _body("subscription.activated", _real_shape_activated("sub_1", "ctm_1"))
        handler.handle(body=body, signature_header="")
        entry = store.get("sub_1")
        assert entry is not None
        assert entry["customer_id"] == "ctm_1", (
            "customer_id must be read from the flat data.customer_id field "
            f"(real Paddle shape) — got {entry.get('customer_id')!r}"
        )

    def test_flat_customer_id_survives_a_renewal_update_too(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1")), signature_header=""
        )
        handler.handle(body=_body("subscription.updated", _real_shape_activated("sub_1", "ctm_1")), signature_header="")
        assert store.get("sub_1")["customer_id"] == "ctm_1"


class _FakePaddleClient:
    """Stand-in for PaddleClient — records get_customer calls, no network."""

    def __init__(self, customers: dict[str, dict]):
        self._customers = customers
        self.calls: list[str] = []

    def get_customer(self, customer_id: str) -> dict:
        self.calls.append(customer_id)
        if customer_id not in self._customers:
            request = httpx.Request("GET", f"https://sandbox-api.paddle.com/customers/{customer_id}")
            raise httpx.HTTPStatusError("not found", request=request, response=httpx.Response(404, request=request))
        return self._customers[customer_id]


class TestEmailResolvedViaCustomerFetch:
    """TASK 1 second half: since no webhook carries an email, the handler
    must resolve it via PaddleClient.get_customer(customer_id). Against the
    unfixed code there is no such call at all (email is simply always None
    for a real-shaped payload) — this FAILS on the unfixed code.
    """

    def test_activation_fetches_email_from_customer_api(self, tmp_path):
        fake_client = _FakePaddleClient({"ctm_1": {"id": "ctm_1", "email": "real-customer@example.com"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store, paddle_client=fake_client)
        body = _body("subscription.activated", _real_shape_activated("sub_1", "ctm_1"))
        handler.handle(body=body, signature_header="")
        entry = store.get("sub_1")
        assert entry["email"] == "real-customer@example.com"
        assert fake_client.calls == ["ctm_1"]

    def test_email_is_fetched_only_once_not_on_every_renewal(self, tmp_path):
        """Once an email is on file, a renewal/update webhook must not
        re-fetch it — mirrors the existing "mint key exactly once" guard."""
        fake_client = _FakePaddleClient({"ctm_1": {"id": "ctm_1", "email": "real-customer@example.com"}})
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store, paddle_client=fake_client)
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1")), signature_header=""
        )
        handler.handle(body=_body("subscription.updated", _real_shape_activated("sub_1", "ctm_1")), signature_header="")
        assert fake_client.calls == ["ctm_1"]  # not called twice

    def test_customer_fetch_failure_never_blocks_activation(self, tmp_path):
        """A Paddle API hiccup fetching the customer must not prevent the
        subscriber from being activated and keyed."""
        fake_client = _FakePaddleClient({})  # customer_id will 404
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store, paddle_client=fake_client)
        body = _body("subscription.activated", _real_shape_activated("sub_1", "ctm_ghost"))
        res = handler.handle(body=body, signature_header="")
        assert res["handled"] is True
        entry = store.get("sub_1")
        assert entry["active"] is True
        assert entry["api_key"] is not None
        assert entry["email"] is None


# ── TASK 2 / 3: active_until grace window + expires_at hard ceiling ────────


class TestCancelHonorsGraceUntilPeriodEnd:
    """BUG (B): the unfixed code collapsed canceled/past_due/paused/expired
    into instant active=False, contradicting refunds.html/terms.html's
    promise of "access until the end of the current billing period". These
    tests FAIL on the unfixed code — is_active_key would be False
    immediately after cancellation even though `now` is still before the
    paid-through date.
    """

    def test_key_stays_active_after_cancel_until_period_end(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        future_end = time.time() + 3600  # 1 hour from now
        # Activate, capturing the paid-through date via current_billing_period.
        from datetime import datetime

        iso_end = datetime.fromtimestamp(future_end, tz=UTC).isoformat().replace("+00:00", "Z")
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1", period_ends_at=iso_end)),
            signature_header="",
        )
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is True

        # Real Paddle cancel payloads carry a NULL current_billing_period —
        # confirm the grace still works even though this event supplies none.
        cancel_data = {"id": "sub_1", "status": "canceled", "customer_id": "ctm_1", "current_billing_period": None}
        handler.handle(body=_body("subscription.canceled", cancel_data), signature_header="")

        # `active` is now False, but the key must still work — we are still
        # before the captured paid-through date.
        assert store.get("sub_1")["active"] is False
        assert store.is_active_key(key, now=time.time()) is True
        # ... and stop working once the paid-through date has passed.
        assert store.is_active_key(key, now=future_end + 1) is False

    def test_paused_also_honors_grace(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        from datetime import datetime

        future_end = time.time() + 3600
        iso_end = datetime.fromtimestamp(future_end, tz=UTC).isoformat().replace("+00:00", "Z")
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1", period_ends_at=iso_end)),
            signature_header="",
        )
        key = store.api_key_of("sub_1")
        handler.handle(
            body=_body("subscription.paused", {"id": "sub_1", "status": "paused", "customer_id": "ctm_1"}),
            signature_header="",
        )
        assert store.is_active_key(key, now=time.time()) is True
        assert store.is_active_key(key, now=future_end + 1) is False

    def test_legacy_subscriber_with_no_active_until_is_not_locked_out(self, tmp_path):
        """Null-handling requirement: a subscriber activated before this
        feature existed (no current_billing_period captured) must still lose
        access on cancel exactly as before — no active_until means no grace,
        not a crash and not an accidental permanent grant."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_legacy", active=True, tier="data")  # no active_until
        key = store.api_key_of("sub_legacy")
        assert store.is_active_key(key) is True
        store.set_active("sub_legacy", active=False)  # simulates a cancel with nothing captured
        assert store.is_active_key(key) is False


class TestPastDueGrantsGraceNotInstantRevocation:
    """FAILS on the unfixed code: past_due collapsed straight to
    active=False with zero grace, killing access on the first failed card
    charge instead of giving the dunning process a window to recover."""

    def test_past_due_key_stays_active_within_grace_window(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIRRA_PAST_DUE_GRACE_S", "100")
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1")), signature_header=""
        )
        key = store.api_key_of("sub_1")
        now = time.time()
        handler.handle(
            body=_body("subscription.past_due", {"id": "sub_1", "status": "past_due", "customer_id": "ctm_1"}),
            signature_header="",
        )
        assert store.get("sub_1")["active"] is False
        assert store.is_active_key(key, now=now + 50) is True  # within the 100s grace
        assert store.is_active_key(key, now=now + 150) is False  # past the grace window


class TestExpiredHardRevokesEvenDuringGrace:
    """expired must revoke immediately and OVERRIDE any grace window a prior
    canceled/past_due event had already granted — dunning is exhausted, this
    is the terminal state. There is no equivalent concept in the unfixed
    code (it also just set active=False, so this specific override-a-future-
    grace-window behavior did not exist to fail — this test protects the
    NEW behavior going forward)."""

    def test_expired_overrides_a_future_grace_window(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        far_future = time.time() + 999999
        store.set_active("sub_1", active=True, tier="data", active_until=far_future)
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is True

        handler.handle(
            body=_body("subscription.expired", {"id": "sub_1", "status": "expired", "customer_id": "ctm_1"}),
            signature_header="",
        )
        assert store.is_active_key(key) is False  # immediate, despite far_future active_until


class TestExpiresAtHardCeiling:
    """TASK 3: expires_at is a general-purpose hard ceiling for any
    time-limited key, independent of Paddle's own lifecycle events —
    verified directly against SubscriberStore/is_active_key."""

    def test_expires_at_in_future_does_not_block_access(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data", expires_at=time.time() + 3600)
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is True

    def test_expires_at_in_past_blocks_access_even_if_active_true(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active("sub_1", active=True, tier="data", expires_at=time.time() - 1)
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is False

    def test_expires_at_beats_a_future_active_until(self, tmp_path):
        """expires_at is the harder ceiling — it must win even when
        active_until would otherwise still grant grace."""
        store = SubscriberStore(str(tmp_path / "subs.json"))
        store.set_active(
            "sub_1", active=False, tier="data", active_until=time.time() + 999999, expires_at=time.time() - 1
        )
        key = store.api_key_of("sub_1")
        assert store.is_active_key(key) is False


class TestKeyStableThroughFullLifecycle:
    """Cancel (within grace) → reactivate must not rotate the key, even with
    the new active_until machinery in play."""

    def test_key_unchanged_across_cancel_and_reactivate(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1")), signature_header=""
        )
        key = store.api_key_of("sub_1")
        handler.handle(
            body=_body("subscription.canceled", {"id": "sub_1", "status": "canceled", "customer_id": "ctm_1"}),
            signature_header="",
        )
        handler.handle(
            body=_body("subscription.activated", _real_shape_activated("sub_1", "ctm_1")), signature_header=""
        )
        assert store.api_key_of("sub_1") == key


# ── Idempotency: replaying subscription.created must not double-anything ──


class TestReplayIsIdempotent:
    """Replaying the same event_id-less subscription.created payload twice
    must not mint a second key or duplicate the subscriber row (the
    event_id ledger already covers event_id-bearing replays elsewhere;
    this covers the plain "same webhook applied twice" case that any
    at-least-once delivery system requires, per the money-correctness
    checklist item 2)."""

    def test_double_apply_same_created_payload_is_a_noop_on_the_key(self, tmp_path):
        store = SubscriberStore(str(tmp_path / "subs.json"))
        handler = PaddleWebhookHandler(secret=_SECRET, store=store)
        body = _body("subscription.created", _real_shape_activated("sub_1", "ctm_1"))
        handler.handle(body=body, signature_header="")
        key1 = store.api_key_of("sub_1")
        handler.handle(body=body, signature_header="")
        key2 = store.api_key_of("sub_1")
        assert key1 == key2
        assert len(store.all()) == 1


# ── TASK 4: client.get_subscription_management_urls ────────────────────────


def _make_config() -> PaddleConfig:
    return PaddleConfig(
        mode="sandbox",
        api_key="test_fake_key_not_real",  # noqa: S106 — test fixture, not a real credential
        client_token="test_fake_token",
        webhook_secret="",
        price_id="pri_fake",
        retain_id=None,
    )


class TestGetSubscriptionManagementUrls:
    """PaddleClient.get_subscription_management_urls(subscription_id) ->
    {"update_payment_method": str | None, "cancel": str}. No network call —
    get_subscription is monkeypatched at the instance's class."""

    def test_extracts_both_urls(self, monkeypatch):
        client = PaddleClient(_make_config())

        def fake_get_subscription(self, subscription_id):
            assert subscription_id == "sub_1"
            return {
                "id": "sub_1",
                "management_urls": {
                    "update_payment_method": "https://checkout.paddle.com/manage/upm/xyz",
                    "cancel": "https://checkout.paddle.com/manage/cancel/xyz",
                },
            }

        monkeypatch.setattr(PaddleClient, "get_subscription", fake_get_subscription)
        urls = client.get_subscription_management_urls("sub_1")
        assert urls == {
            "update_payment_method": "https://checkout.paddle.com/manage/upm/xyz",
            "cancel": "https://checkout.paddle.com/manage/cancel/xyz",
        }

    def test_handles_null_update_payment_method(self, monkeypatch):
        """Manually-collected subscriptions return null for
        update_payment_method per Paddle's docs — must not crash."""
        client = PaddleClient(_make_config())

        def fake_get_subscription(self, subscription_id):
            return {"id": "sub_1", "management_urls": {"update_payment_method": None, "cancel": "https://x/cancel"}}

        monkeypatch.setattr(PaddleClient, "get_subscription", fake_get_subscription)
        urls = client.get_subscription_management_urls("sub_1")
        assert urls["update_payment_method"] is None
        assert urls["cancel"] == "https://x/cancel"


# ── TASK 5: unit_price_overrides on create_price + update_price ────────────


class _CapturingTransport(httpx.BaseTransport):
    """Fake httpx transport — captures the request, returns a canned 200.
    No real network call is ever made."""

    def __init__(self, response_json: dict):
        self.captured: httpx.Request | None = None
        self._response_json = response_json

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.captured = request
        return httpx.Response(200, json=self._response_json, request=request)


class TestCreatePriceUnitPriceOverrides:
    def test_create_price_includes_unit_price_overrides_when_given(self, monkeypatch):
        client = PaddleClient(_make_config())
        transport = _CapturingTransport({"data": {"id": "pri_new"}})
        monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw))

        overrides = [{"country_codes": ["IN"], "unit_price": {"amount": "41500", "currency_code": "INR"}}]
        result = client.create_price(
            product_id="pro_1",
            description="Test price",
            amount="50000",
            unit_price_overrides=overrides,
        )
        assert result == {"id": "pri_new"}
        sent = json.loads(transport.captured.content)
        assert sent["unit_price_overrides"] == overrides

    def test_create_price_omits_unit_price_overrides_when_not_given(self, monkeypatch):
        client = PaddleClient(_make_config())
        transport = _CapturingTransport({"data": {"id": "pri_new"}})
        monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw))

        client.create_price(product_id="pro_1", description="Test price", amount="50000")
        sent = json.loads(transport.captured.content)
        assert "unit_price_overrides" not in sent


class TestUpdatePrice:
    def test_update_price_sends_patch_with_unit_price_overrides(self, monkeypatch):
        client = PaddleClient(_make_config())
        transport = _CapturingTransport({"data": {"id": "pri_1"}})
        monkeypatch.setattr(httpx, "patch", lambda url, **kw: httpx.Client(transport=transport).patch(url, **kw))

        overrides = [{"country_codes": ["IN"], "unit_price": {"amount": "41500", "currency_code": "INR"}}]
        client.update_price("pri_1", unit_price_overrides=overrides)

        assert transport.captured.method == "PATCH"
        assert transport.captured.url.path.endswith("/prices/pri_1")
        sent = json.loads(transport.captured.content)
        assert sent == {"unit_price_overrides": overrides}

    def test_update_price_requires_amount_and_currency_code_together(self):
        client = PaddleClient(_make_config())
        with pytest.raises(ValueError):
            client.update_price("pri_1", amount="1000")
        with pytest.raises(ValueError):
            client.update_price("pri_1", currency_code="USD")

    def test_update_price_requires_something_to_update(self):
        client = PaddleClient(_make_config())
        with pytest.raises(ValueError):
            client.update_price("pri_1")

    def test_update_price_sends_only_provided_fields(self, monkeypatch):
        """Confirms omitted fields are truly absent from the payload —
        Paddle treats an omitted field as 'unchanged', so accidentally
        sending e.g. unit_price=None would be a live-pricing bug."""
        client = PaddleClient(_make_config())
        transport = _CapturingTransport({"data": {"id": "pri_1"}})
        monkeypatch.setattr(httpx, "patch", lambda url, **kw: httpx.Client(transport=transport).patch(url, **kw))

        client.update_price("pri_1", description="New description only")
        sent = json.loads(transport.captured.content)
        assert sent == {"description": "New description only"}
