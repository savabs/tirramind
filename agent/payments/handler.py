"""Paddle webhook handler — verify + process subscription lifecycle events.

Routes Paddle events that matter to the product:
    subscription.created / subscription.updated / subscription.activated /
    subscription.revived / subscription.trialing
        → mark subscriber active, grant a brief-access key, capture the
          current paid-through date (`active_until`, from
          `current_billing_period.ends_at`)
    subscription.canceled / subscription.paused
        → `active` flips False, but access continues until `active_until`
          (the paid-through date captured above) — refunds.html and
          terms.html both promise this; access is NOT revoked instantly.
    subscription.past_due
        → a grace window (`TIRRA_PAST_DUE_GRACE_S`, default 3 days), not
          instant revocation — a failed card is not a cancellation.
    subscription.expired
        → revoke immediately, overriding any prior grace window.

All of the above resolve through ONE chokepoint,
`SubscriberStore`'s `_effective_active()` — see its docstring for the exact
precedence of `expires_at` (hard ceiling) vs `active_until` (grace) vs
`active` (fallback for pre-existing records with neither field set).

Design:
  - Verification is MANDATORY when a webhook secret is configured (live). It is
    skipped only when no secret is set (sandbox/local dev).
  - Processing is additive and idempotent: it writes subscriber state, never
    deletes live Paddle entities.
  - customer_id comes from the FLAT `data.customer_id` every real Paddle
    subscription/transaction webhook carries (verified against
    developer.paddle.com, 2026-08-28) — not a nested `data.customer.id`,
    which does not exist on any real delivery and was silently None on
    every webhook ever received before this fix. No webhook payload carries
    an email address at all; `PaddleWebhookHandler._fetch_customer_email`
    resolves it via `GET /customers/{id}` the first time a subscriber
    activates with none on file.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime
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


def _parse_paddle_timestamp(value: str | None) -> float | None:
    """Parse a Paddle RFC 3339 datetime string (e.g.
    "2024-04-12T11:24:54.868Z") into Unix epoch seconds.

    Returns None for missing/unparseable input — callers MUST treat that as
    "unknown", never as epoch-0, which would look like a timestamp far in
    the past and incorrectly revoke access.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


def _extract_active_until(data: dict[str, Any]) -> float | None:
    """Best available "access valid until" timestamp from a subscription
    webhook payload's `data`.

    Verified against developer.paddle.com (2026-08-28): every subscription
    event carries `current_billing_period.{starts_at,ends_at}` while the
    subscription is active/trialing — this is the authoritative paid-through
    date. It is `null` once a subscription has actually transitioned to
    `canceled`/`paused` (by which point there IS no current period), so this
    is only ever populated by an _ACTIVE_EVENTS webhook (created / updated /
    activated / revived / trialing), including the `subscription.updated`
    Paddle fires when a customer schedules a cancel-at-period-end (still
    `status: active` at that point). Falls back to
    `scheduled_change.effective_at` for the same scenario in case
    `current_billing_period` is ever absent from a given payload variant.
    Returns None (never a stale/derived guess) if neither is present —
    callers must then preserve whatever was last captured, not overwrite it.
    """
    period = data.get("current_billing_period") or {}
    ends_at = _parse_paddle_timestamp(period.get("ends_at"))
    if ends_at is not None:
        return ends_at
    scheduled = data.get("scheduled_change") or {}
    return _parse_paddle_timestamp(scheduled.get("effective_at"))


# ── past_due grace window ────────────────────────────────────────────────────
#
# A failed card must not be treated the same as a deliberate cancellation.
# Paddle keeps retrying (dunning) for a period before it either recovers
# (fires an _ACTIVE_EVENTS update) or gives up (fires subscription.canceled/
# .expired) — this grace window is what keeps the customer's key working
# while that plays out, instead of killing access on the first failed charge.
_PAST_DUE_GRACE_ENV = "TIRRA_PAST_DUE_GRACE_S"
_DEFAULT_PAST_DUE_GRACE_S = 3 * 24 * 3600.0  # 3 days


def _past_due_grace_s() -> float:
    raw = os.getenv(_PAST_DUE_GRACE_ENV, "").strip()
    if not raw:
        return _DEFAULT_PAST_DUE_GRACE_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_PAST_DUE_GRACE_S


# Event types that grant/keep access.
_ACTIVE_EVENTS = {
    "subscription.created",
    "subscription.updated",
    "subscription.activated",
    "subscription.revived",
    "subscription.trialing",
}
# canceled / paused: access continues until the end of the already-paid
# period (refunds.html / terms.html both promise this) — `active` flips to
# False here, but `active_until` (captured on a PRIOR _ACTIVE_EVENTS webhook,
# since canceled/paused payloads themselves carry a null current_billing_period
# — see `_extract_active_until`) keeps the key working until that instant.
_GRACE_UNTIL_PERIOD_END_EVENTS = {
    "subscription.canceled",
    "subscription.paused",
}
# past_due: a failed card gets a grace window, not instant revocation.
_PAST_DUE_EVENTS = {
    "subscription.past_due",
}
# expired: Paddle has given up (dunning exhausted) — revoke immediately,
# overriding any previously-granted grace window.
_HARD_REVOKE_EVENTS = {
    "subscription.expired",
}
# Every event type that revokes access, of any flavor — kept for anything
# that only needs "is this an inactive-flavored event" without caring which
# one (e.g. logging). The three sets above are what `handle()` actually
# branches on.
_INACTIVE_EVENTS = _GRACE_UNTIL_PERIOD_END_EVENTS | _PAST_DUE_EVENTS | _HARD_REVOKE_EVENTS


def _effective_active(entry: dict[str, Any] | None, *, now: float) -> bool:
    """THE chokepoint for "is this subscriber's access currently valid".

    Every route-facing check (`is_active_key`) and every internal one
    (`is_active`, `rotate_key_for_api_key`) resolves through this single
    function — a second ad-hoc `entry.get("active")` check anywhere else
    would silently diverge from this policy the next time it changes.

    Precedence (money-correctness order, most restrictive first):
      1. `expires_at` is a HARD ceiling. Once past it, access is gone no
         matter what `active_until` says — this is both the general
         "any time-limited key" mechanism (TASK 3) and how
         `subscription.expired` overrides a stale grace window in one write
         (set `expires_at`/`active_until` to "now" — see `handle()` below).
      2. `active_until` in the future grants access even when `active` is
         False. This is what makes cancel/pause honor "access continues
         until the end of the current billing period" (refunds.html /
         terms.html already promise this) and gives past_due a grace
         window, instead of collapsing every non-"active" status into
         instant revocation.
      3. Otherwise fall back to the plain `active` flag. This is the null-
         handling guarantee: a pre-existing subscriber record written
         before this policy existed has `active_until`/`expires_at` both
         None, both steps above no-op, and behavior is byte-identical to
         before — never a lockout from a field that didn't exist yet.
    """
    if not entry:
        return False
    expires_at = entry.get("expires_at")
    if expires_at is not None and now >= expires_at:
        return False
    active_until = entry.get("active_until")
    if active_until is not None and now < active_until:
        return True
    return bool(entry.get("active"))


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
        active_until: float | None = None,
        expires_at: float | None = None,
    ) -> dict[str, Any]:
        """Write (or update) a subscriber record.

        `active_until` / `expires_at` follow "only overwrite when the caller
        actually supplies a value" semantics — same as `customer_id`/`email`
        above — because most webhook events (e.g. a bare
        `subscription.canceled`, whose payload carries no period-end info at
        all; see `_extract_active_until`) have nothing new to say about
        either field and must not clobber whatever was captured earlier.
        Passing `None` (the default) is a no-op for that field, NOT "clear
        it" — callers that genuinely need to clear a field pass an explicit
        sentinel handled at the call site, not here.
        """
        entry = self._data.get(subscription_id, {})
        entry["subscription_id"] = subscription_id
        entry["active"] = bool(active)
        entry["customer_id"] = customer_id or entry.get("customer_id")
        entry["email"] = email or entry.get("email")
        entry["tier"] = tier or entry.get("tier", _DEFAULT_TIER)
        if active_until is not None:
            entry["active_until"] = active_until
        else:
            entry.setdefault("active_until", None)
        if expires_at is not None:
            entry["expires_at"] = expires_at
        else:
            entry.setdefault("expires_at", None)
        if active and not entry.get("api_key"):
            entry["api_key"] = _generate_api_key()
        entry["updated_at"] = time.time()
        self._data[subscription_id] = entry
        self._save()
        return entry

    def is_active(self, subscription_id: str, *, now: float | None = None) -> bool:
        now_f = time.time() if now is None else now
        return _effective_active(self._data.get(subscription_id), now=now_f)

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

    def is_active_key(self, api_key: str, *, now: float | None = None) -> bool:
        """True if `api_key` is a currently-active subscriber's opaque API key.

        This is the customer-facing lookup: subscribers authenticate with the
        opaque `tirra_...` key minted on activation, never with the raw Paddle
        subscription_id. Resolves through `_effective_active` — see that
        function's docstring for why a canceled/past_due subscriber can still
        return True here (grace window) and why an expired one cannot
        (hard ceiling).
        """
        now_f = time.time() if now is None else now
        return _effective_active(self._by_api_key(api_key), now=now_f)

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
        if entry is None or not _effective_active(entry, now=time.time()):
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
        paddle_client: Any = None,
    ) -> None:
        self.secret = secret
        self.store = store or SubscriberStore()
        # Default-constructed (disk-backed, bounded, survives restart) unless
        # a caller injects one — tests inject a tmp_path-backed ledger so they
        # never touch the real project directory.
        self.event_ledger = event_ledger if event_ledger is not None else ProcessedEventLedger()
        # Used ONLY to fetch a customer's email via GET /customers/{id} — no
        # Paddle subscription/transaction webhook payload carries an email
        # field (verified against developer.paddle.com, 2026-08-28); this is
        # the one real source. `None` (the default) means "no client was
        # injected"; `_get_paddle_client()` then lazily builds one from env
        # config the first time it's actually needed, so tests that never
        # exercise that path never need TIRRA_PADDLE_* env vars set.
        self._paddle_client = paddle_client
        self._paddle_client_init_attempted = paddle_client is not None

    def _get_paddle_client(self) -> Any:
        if self._paddle_client is not None:
            return self._paddle_client
        if self._paddle_client_init_attempted:
            return None
        self._paddle_client_init_attempted = True
        try:
            from agent.payments.client import PaddleClient
            from agent.payments.config import PaddleConfig

            cfg = PaddleConfig.from_env()
            if not cfg.api_key:
                return None
            self._paddle_client = PaddleClient(cfg)
        except Exception as exc:  # noqa: BLE001 — env/config trouble must never block webhook processing
            logger.warning(
                "[paddle] could not build a PaddleClient for customer email lookup: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None
        return self._paddle_client

    def _fetch_customer_email(self, customer_id: str | None) -> str | None:
        """Best-effort GET /customers/{id} to resolve an email address.

        Never raises — a Paddle API hiccup (network error, 404, rate limit)
        must not block subscription activation any more than the delivery
        email attempt does (see agent/payments/delivery.py's own guard).
        """
        if not customer_id:
            return None
        client = self._get_paddle_client()
        if client is None:
            return None
        try:
            customer = client.get_customer(customer_id)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning(
                "[paddle] failed to fetch customer %s for email: %s: %s", customer_id, type(exc).__name__, exc
            )
            return None
        email = customer.get("email") if isinstance(customer, dict) else None
        return email.strip() if isinstance(email, str) and email.strip() else None

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
        # customer_id: real Paddle subscription/transaction webhooks carry a
        # FLAT `data.customer_id` (verified against developer.paddle.com,
        # 2026-08-28 — subscription.activated / subscription.canceled /
        # transaction.completed example payloads all show `"customer_id":
        # "ctm_..."` at the top level of `data`, never a nested `customer`
        # object). BUG (A) was reading `data.customer.id`, which is None on
        # every real delivery. The nested-`customer` fallback below exists
        # ONLY for defensive back-compat with hand-built test fixtures that
        # predate this fix — never observed in a real Paddle payload.
        customer_id = data.get("customer_id") or (data.get("customer", {}) or {}).get("id")
        # email: NO subscription or transaction webhook carries one at all
        # (same verification pass) — Paddle's Customer object is the only
        # place it lives. `_fetch_customer_email` below is the real source;
        # this nested read is the same defensive back-compat shim as above.
        email = (data.get("customer", {}) or {}).get("email")
        items = data.get("items") or []
        price_id = (items[0].get("price") or {}).get("id") if items else None
        tier = _resolve_tier(price_id)

        if not subscription_id:
            return {"event_type": event_type, "handled": False, "reason": "no subscription_id"}

        if event_type in _ACTIVE_EVENTS:
            # Capture pre-write state so we only ever attempt an email fetch
            # / delivery on the ONE event that actually mints a fresh key —
            # not on every subsequent subscription.updated/renewal webhook
            # for the same subscriber. set_active() itself only mints when
            # no api_key exists yet (see its body above); mirror that same
            # condition here so "did we just create a key" is decided the
            # same way.
            prior_entry = self.store.get(subscription_id)
            had_key_before = bool(prior_entry and prior_entry.get("api_key"))
            had_email_before = bool(prior_entry and prior_entry.get("email"))
            resolved_email = email
            if not resolved_email and not had_email_before and customer_id:
                resolved_email = self._fetch_customer_email(customer_id)
            active_until = _extract_active_until(data)
            entry = self.store.set_active(
                subscription_id,
                active=True,
                customer_id=customer_id,
                email=resolved_email,
                tier=tier,
                active_until=active_until,
            )
            if not had_key_before and entry.get("api_key"):
                # Best-effort, never fatal: see agent/payments/delivery.py.
                # With no SMTP env configured (current state) this always
                # returns "unconfigured" and logs — it does not send.
                attempt_deliver_key_email(
                    entry.get("email"), api_key=entry.get("api_key"), tier=entry.get("tier") or tier
                )
        elif event_type in _GRACE_UNTIL_PERIOD_END_EVENTS:
            # canceled / paused: `active` flips False, but `active_until`
            # (captured on a prior _ACTIVE_EVENTS webhook — this payload's
            # own current_billing_period is null per Paddle's docs) is left
            # untouched by passing None, so the grace window already on file
            # keeps working through `_effective_active`.
            active_until = _extract_active_until(data)  # usually None; see docstring
            entry = self.store.set_active(
                subscription_id,
                active=False,
                customer_id=customer_id,
                email=email,
                tier=tier,
                active_until=active_until,
            )
        elif event_type in _PAST_DUE_EVENTS:
            # A failed card gets a grace window, not instant revocation.
            # `max(...)` never SHRINKS a later, already-known paid-through
            # date (e.g. this fires mid-cycle while current_billing_period
            # still has weeks left) — it only ever extends access, never
            # cuts a legitimately-longer window short.
            prior_entry = self.store.get(subscription_id)
            prior_until = prior_entry.get("active_until") if prior_entry else None
            grace_until = time.time() + _past_due_grace_s()
            new_until = max(prior_until, grace_until) if prior_until is not None else grace_until
            entry = self.store.set_active(
                subscription_id,
                active=False,
                customer_id=customer_id,
                email=email,
                tier=tier,
                active_until=new_until,
            )
        elif event_type in _HARD_REVOKE_EVENTS:
            # expired: dunning exhausted, Paddle has given up. Revoke NOW,
            # overriding any previously-granted grace window (an explicit
            # `active_until=now` beats a lingering future value already on
            # file — see `set_active`'s "overwrite only when supplied"
            # contract above).
            entry = self.store.set_active(
                subscription_id,
                active=False,
                customer_id=customer_id,
                email=email,
                tier=tier,
                active_until=time.time(),
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
