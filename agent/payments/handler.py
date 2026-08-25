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
import time
from pathlib import Path
from typing import Any

from agent.payments.webhook import (
    WebhookVerificationError,
    is_verified,
    verify_webhook_signature,
)

logger = logging.getLogger(__name__)

# Default state file mapping subscriber_id → {active, customer_id, email}
_DEFAULT_STATE_PATH = ".tirra_opportunities/subscribers.json"

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

    def set_active(self, subscription_id: str, *, active: bool, customer_id: str | None = None, email: str | None = None) -> dict[str, Any]:
        entry = self._data.get(subscription_id, {})
        entry["subscription_id"] = subscription_id
        entry["active"] = bool(active)
        entry["customer_id"] = customer_id or entry.get("customer_id")
        entry["email"] = email or entry.get("email")
        entry["updated_at"] = time.time()
        self._data[subscription_id] = entry
        self._save()
        return entry

    def is_active(self, subscription_id: str) -> bool:
        return bool(self._data.get(subscription_id, {}).get("active"))

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)

    def active_keys(self) -> list[str]:
        """Subscriber keys allowed to access the brief (subscription IDs)."""
        return [sid for sid, e in self._data.items() if e.get("active")]


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
            data.get("id")
            or data.get("subscription_id")
            or (data.get("subscription", {}) or {}).get("id")
        )
        customer = data.get("customer", {}) or {}
        customer_id = customer.get("id")
        email = customer.get("email")

        if not subscription_id:
            return {"event_type": event_type, "handled": False, "reason": "no subscription_id"}

        if event_type in _ACTIVE_EVENTS:
            self.store.set_active(subscription_id, active=True, customer_id=customer_id, email=email)
        elif event_type in _INACTIVE_EVENTS:
            self.store.set_active(subscription_id, active=False, customer_id=customer_id, email=email)
        else:
            return {"event_type": event_type, "handled": False, "reason": "unhandled event"}

        logger.info("[paddle] %s → subscription %s active=%s", event_type, subscription_id, event_type in _ACTIVE_EVENTS)
        return {"event_type": event_type, "handled": True, "subscription_id": subscription_id, "active": event_type in _ACTIVE_EVENTS}


__all__ = ["PaddleWebhookHandler", "SubscriberStore"]
