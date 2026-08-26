"""Paddle webhook handler — verify + process subscription lifecycle events.

Routes Paddle events that matter to the product:
    subscription.created / subscription.updated / subscription.activated
        → mark subscriber active, grant a brief-access key
    subscription.canceled / subscription.past_due / subscription.paused
        → deactivate subscriber access

Design:
  - Verification is MANDATORY when a webhook secret is configured (live). It is
    skipped only when no secret is set (sandbox/local dev).
  - Processing is additive and idempotent: it writes subscriber state, never
    deletes live Paddle entities.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from agent.payments.webhook import (
    WebhookVerificationError,
    is_verified,
    verify_webhook_signature,
)

logger = logging.getLogger(__name__)

# Default state file mapping subscriber_id → {active, customer_id, email, tier}
_DEFAULT_STATE_PATH = ".tirra_opportunities/subscribers.json"

# Fallback tier when a webhook's price can't be mapped (or no map is configured).
_DEFAULT_TIER = "brief"


def _resolve_tier(price_id: str | None) -> str:
    """Map a Paddle price ID to a product tier via TIRRA_TIER_PRICE_MAP.

    Format: "pri_data123:data,pri_entity456:entity,pri_sched789:scheduler".
    Unmapped or missing price → the base "brief" tier (backward compatible
    with subscriptions created before tiers existed).
    """
    mapping_raw = os.getenv("TIRRA_TIER_PRICE_MAP", "").strip()
    if not price_id or not mapping_raw:
        return _DEFAULT_TIER
    for pair in mapping_raw.split(","):
        pid, _, tier = pair.strip().partition(":")
        if pid.strip() == price_id:
            return tier.strip() or _DEFAULT_TIER
    return _DEFAULT_TIER


def _generate_api_key() -> str:
    """A customer-facing API key, distinct from the internal Paddle subscription_id."""
    return "tirra_" + secrets.token_urlsafe(24)


# Event types that grant/keep access.
_ACTIVE_EVENTS = {
    "subscription.created",
    "subscription.updated",
    "subscription.activated",
    "subscription.revived",
    "subscription.trialing",
}
# Event types that revoke access.
_INACTIVE_EVENTS = {
    "subscription.canceled",
    "subscription.past_due",
    "subscription.paused",
    "subscription.expired",
}


class SubscriberStore:
    """Persistent, append-friendly record of subscriber access state."""

    def __init__(self, path: str = _DEFAULT_STATE_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_active(
        self,
        subscription_id: str,
        *,
        active: bool,
        customer_id: str | None = None,
        email: str | None = None,
        tier: str | None = None,
    ) -> dict[str, Any]:
        entry = self._data.get(subscription_id, {})
        entry["subscription_id"] = subscription_id
        entry["active"] = bool(active)
        entry["customer_id"] = customer_id or entry.get("customer_id")
        entry["email"] = email or entry.get("email")
        entry["tier"] = tier or entry.get("tier", _DEFAULT_TIER)
        if active and not entry.get("api_key"):
            entry["api_key"] = _generate_api_key()
        entry["updated_at"] = time.time()
        self._data[subscription_id] = entry
        self._save()
        return entry

    def is_active(self, subscription_id: str) -> bool:
        return bool(self._data.get(subscription_id, {}).get("active"))

    def tier_of(self, subscription_id: str) -> str | None:
        """The product tier this subscriber is on (e.g. 'brief', 'data', 'entity', 'scheduler')."""
        return self._data.get(subscription_id, {}).get("tier")

    def api_key_of(self, subscription_id: str) -> str | None:
        """The opaque customer-facing API key for this subscription, if one has been minted."""
        return self._data.get(subscription_id, {}).get("api_key")

    def _by_api_key(self, api_key: str) -> dict[str, Any] | None:
        for entry in self._data.values():
            if entry.get("api_key") == api_key:
                return entry
        return None

    def is_active_key(self, api_key: str) -> bool:
        """True if `api_key` is a currently-active subscriber's opaque API key.

        This is the customer-facing lookup: subscribers authenticate with the
        opaque `tirra_...` key minted on activation, never with the raw Paddle
        subscription_id.
        """
        entry = self._by_api_key(api_key)
        return bool(entry and entry.get("active"))

    def tier_of_key(self, api_key: str) -> str | None:
        """The product tier for the subscriber owning `api_key`."""
        entry = self._by_api_key(api_key)
        return entry.get("tier") if entry else None

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)

    def active_keys(self) -> list[str]:
        """Opaque API keys (not subscription IDs) for currently-active subscribers."""
        return [e["api_key"] for e in self._data.values() if e.get("active") and e.get("api_key")]


class PaddleWebhookHandler:
    """Verify + apply one Paddle webhook payload."""

    def __init__(self, secret: str, store: SubscriberStore | None = None) -> None:
        self.secret = secret
        self.store = store or SubscriberStore()

    def handle(self, *, body: bytes, signature_header: str) -> dict[str, Any]:
        """Verify and process a webhook. Returns a summary.

        Raises WebhookVerificationError on invalid signature (if verification
        is enabled).
        """
        if is_verified(self.secret):
            verify_webhook_signature(body=body, signature_header=signature_header, secret=self.secret)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WebhookVerificationError("webhook body is not valid JSON") from exc

        event_type = payload.get("event_type", "")
        data = payload.get("data", {})
        subscription_id = (
            data.get("id") or data.get("subscription_id") or (data.get("subscription", {}) or {}).get("id")
        )
        customer = data.get("customer", {}) or {}
        customer_id = customer.get("id")
        email = customer.get("email")
        items = data.get("items") or []
        price_id = (items[0].get("price") or {}).get("id") if items else None
        tier = _resolve_tier(price_id)

        if not subscription_id:
            return {"event_type": event_type, "handled": False, "reason": "no subscription_id"}

        if event_type in _ACTIVE_EVENTS:
            entry = self.store.set_active(subscription_id, active=True, customer_id=customer_id, email=email, tier=tier)
        elif event_type in _INACTIVE_EVENTS:
            entry = self.store.set_active(
                subscription_id, active=False, customer_id=customer_id, email=email, tier=tier
            )
        else:
            return {"event_type": event_type, "handled": False, "reason": "unhandled event"}

        logger.info(
            "[paddle] %s → subscription %s active=%s", event_type, subscription_id, event_type in _ACTIVE_EVENTS
        )
        return {
            "event_type": event_type,
            "handled": True,
            "subscription_id": subscription_id,
            "active": event_type in _ACTIVE_EVENTS,
            "api_key": entry.get("api_key"),
        }


__all__ = ["PaddleWebhookHandler", "SubscriberStore"]
