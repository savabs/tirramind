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

import copy
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from agent.payments.delivery import attempt_deliver_key_email
from agent.payments.event_ledger import ProcessedEventLedger
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

# ── Read cache for SubscriberStore ──────────────────────────────────────────
#
# brief_server.py re-instantiates SubscriberStore() on every gated request
# (_authorized_for AND _log_usage each do it separately) — that's two full
# disk reads + JSON parses + O(n) linear scans per request, purely to answer
# "is this key active" / "what tier". This module-level cache lets fresh
# instances reuse a recently-loaded parse instead of hitting disk again.
#
# TTL: 2 seconds. Chosen because the write path below (`_save`) updates this
# same cache immediately and unconditionally — a subscriber activated or
# canceled by a webhook is visible to the very next SubscriberStore() in this
# process instantly, regardless of TTL. The TTL only bounds staleness for the
# case NOT covered by write-through invalidation: something else changing
# subscribers.json on disk without going through this class (e.g. a manual
# edit, or — the one real risk — a second process writing the same file).
# 2s keeps that exposure window small without meaningfully reducing the read
# savings under normal request rates (a burst of requests within the same 2s
# is exactly the case this exists to help).
#
# Caveat (ground truth: single-process deployment): if this ever runs as
# multiple worker processes, this cache is per-process and a write in one
# worker is invisible to another until its own local TTL expires. That is a
# real staleness bug waiting to happen the day this becomes a fleet — flagged
# here, not silently assumed away.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


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

    def __init__(self, path: str = _DEFAULT_STATE_PATH, *, cache_ttl_s: float | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_key = str(self._path)
        self._cache_ttl_s = _CACHE_TTL_S if cache_ttl_s is None else cache_ttl_s
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._cache_ttl_s > 0:
            with _cache_lock:
                cached = _cache.get(self._cache_key)
                if cached is not None and (time.time() - cached[0]) < self._cache_ttl_s:
                    # Deep copy: entries are mutated in place by set_active/
                    # rotate_key/revoke_key. Sharing nested dicts across
                    # instances would let one instance's in-progress mutation
                    # leak into another's view before either has saved.
                    self._data = copy.deepcopy(cached[1])
                    return
        if not self._path.exists():
            self._data = {}
        else:
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._update_cache()

    def _update_cache(self) -> None:
        if self._cache_ttl_s <= 0:
            return
        with _cache_lock:
            _cache[self._cache_key] = (time.time(), copy.deepcopy(self._data))

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Write-through: make this write visible to every other SubscriberStore
        # instance in this process immediately, not after the TTL expires. This
        # is what makes "just activated" / "just canceled" correct despite the
        # cache above — see the module-level comment.
        self._update_cache()

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

    def get(self, subscription_id: str) -> dict[str, Any] | None:
        """Full record for a subscription_id, or None if unknown.

        This is the one lookup the claim contract needs from this store
        (agent/payments/claim.py's `claim_transaction`, once it has resolved
        a Paddle transaction_id to a subscription_id via
        `PaddleClient.get_transaction`) — there is no separate "by
        transaction" lookup here because this store has never been keyed by
        transaction_id; the transaction → subscription_id resolution is
        Paddle's own API, not something this store tracks.
        """
        entry = self._data.get(subscription_id)
        return dict(entry) if entry else None

    def rotate_key(self, subscription_id: str) -> str | None:
        """Mint a new opaque API key for `subscription_id`, invalidating the
        old one (the old key stops matching any entry the instant this
        saves — `_by_api_key` only ever matches the current `api_key` field,
        there is no separate revocation list to consult).

        Returns the new key, or None if `subscription_id` is unknown. Intended
        caller: an authenticated ops/support action (rotate a specific
        customer's key on request, e.g. after a suspected leak) or the
        self-service path via `rotate_key_for_api_key` below.
        """
        entry = self._data.get(subscription_id)
        if entry is None:
            return None
        new_key = _generate_api_key()
        entry["api_key"] = new_key
        entry["updated_at"] = time.time()
        self._data[subscription_id] = entry
        self._save()
        return new_key

    def rotate_key_for_api_key(self, old_api_key: str) -> str | None:
        """Self-service rotation: a customer proves ownership of their
        CURRENT key by presenting it, and gets a new one back; the old key
        stops working immediately (same-process; see write-through cache
        note above).

        Returns None (does nothing) if `old_api_key` doesn't resolve to a
        subscriber, or that subscriber is not currently active — an inactive
        subscriber has no standing to mint a fresh credential. Returns the
        new key on success.

        Expected HTTP route (owned by agent/brief_server.py, not this file):
            POST /api/v1/rotate-key
            Header: X-Brief-Key: <current tirra_... key>
            200 {"ok": true, "api_key": "tirra_<new>"}
            403 {"ok": false, "error": "invalid or inactive key"}  if this
                returns None — same failure shape _authorized_for already
                uses elsewhere, so the route stays a thin translation layer.
        """
        entry = self._by_api_key(old_api_key)
        if entry is None or not entry.get("active"):
            return None
        subscription_id = entry.get("subscription_id")
        if not subscription_id:
            return None
        return self.rotate_key(subscription_id)

    def revoke_key(self, subscription_id: str) -> bool:
        """Invalidate a subscription's current API key WITHOUT minting a
        replacement (e.g. an abuse response where re-provisioning should be
        a separate, deliberate step rather than automatic). The subscriber
        stays `active`; they simply have no working key until `rotate_key`
        is called for them. Returns False if `subscription_id` is unknown.
        """
        entry = self._data.get(subscription_id)
        if entry is None:
            return False
        entry["api_key"] = None
        entry["updated_at"] = time.time()
        self._data[subscription_id] = entry
        self._save()
        return True

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

    def __init__(
        self,
        secret: str,
        store: SubscriberStore | None = None,
        event_ledger: ProcessedEventLedger | None = None,
    ) -> None:
        self.secret = secret
        self.store = store or SubscriberStore()
        # Default-constructed (disk-backed, bounded, survives restart) unless
        # a caller injects one — tests inject a tmp_path-backed ledger so they
        # never touch the real project directory.
        self.event_ledger = event_ledger if event_ledger is not None else ProcessedEventLedger()

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

        # Replay/duplicate guard: Paddle includes a stable `event_id` on every
        # delivery (retries re-send the SAME event_id, they don't mint a new
        # one). This is the actual defense against a captured-and-replayed
        # webhook within the signature's timestamp window, and it's also what
        # makes a benign Paddle retry a true no-op instead of a double-write.
        # A payload with no event_id (malformed, or a hand-built test fixture)
        # can't be deduped — fall through and process it rather than silently
        # dropping something we can't identify.
        event_id = payload.get("event_id")
        if event_id and not self.event_ledger.check_and_mark(event_id):
            logger.info("[paddle] duplicate event_id=%s (%s) ignored — already processed", event_id, event_type)
            return {
                "event_type": event_type,
                "handled": False,
                "reason": "duplicate event (already processed)",
                "event_id": event_id,
            }

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
            # Capture pre-write state so we only ever attempt an email on the
            # ONE event that actually mints a fresh key — not on every
            # subsequent subscription.updated/renewal webhook for the same
            # subscriber. set_active() itself only mints when no api_key
            # exists yet (see its body above); mirror that same condition
            # here so "did we just create a key" is decided the same way.
            prior_entry = self.store.get(subscription_id)
            had_key_before = bool(prior_entry and prior_entry.get("api_key"))
            entry = self.store.set_active(subscription_id, active=True, customer_id=customer_id, email=email, tier=tier)
            if not had_key_before and entry.get("api_key"):
                # Best-effort, never fatal: see agent/payments/delivery.py.
                # With no SMTP env configured (current state) this always
                # returns "unconfigured" and logs — it does not send.
                attempt_deliver_key_email(
                    entry.get("email"), api_key=entry.get("api_key"), tier=entry.get("tier") or tier
                )
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
