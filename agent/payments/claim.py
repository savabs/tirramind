"""Key-claim support — the store/verification logic behind `GET /api/v1/claim`.

Context (why this exists): the API key is minted on webhook activation, the
webhook ack deliberately never echoes it (see handler.py), and nothing else
in the repo emails it to the customer. `claim_transaction()` is the
server-side half of the fix: the checkout page's `Paddle.js` `eventCallback`
hands the browser a real `transaction_id`; this module cross-verifies that id
against Paddle's own API (never trusts the browser's say-so) before ever
returning a subscriber's key, and makes repeat calls for the same
transaction safe (idempotent) without turning a leaked/bookmarked claim URL
into a way to re-extract the plaintext key indefinitely.

Ownership split (see the corrected key-delivery design, 2026-08-27):
  - THIS file (agent/payments/): `ClaimStore`, `claim_transaction`,
    `PaddleClient.get_transaction` (client.py). Owns Paddle-verification and
    claim/idempotency state.
  - agent/brief_server.py (api-backend-engineer): the actual
    `GET /api/v1/claim` HTTP route — parses `txn`, enforces rate limits, sets
    CORS, maps `ClaimResult.status` to the documented HTTP status/JSON shape.
    This module raises no HTTP concerns and knows nothing about status codes.
  - products/brief_subscription/ + /welcome (frontend-engineer): the
    checkout eventCallback and the polling claim UI.

No changes were needed to `SubscriberStore`'s schema for this — the existing
subscription_id-keyed entry (`active` / `api_key` / `tier`) is exactly what
`claim_transaction` reads via `SubscriberStore.get()`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent.payments.client import PaddleClient
from agent.payments.handler import SubscriberStore

_DEFAULT_CLAIM_PATH = ".tirra_opportunities/claims.json"

# Idempotency window: repeat claims for the same txn_id within this window
# return the SAME key again (covers double-click, page reload, duplicate
# tab, a retried fetch after a network blip — the honest browser round-trip
# after checkout redirect must not be punished for happening twice).
_CLAIM_WINDOW_S = 15 * 60

# Even inside the window, cap how many times one txn_id can re-claim before
# we start refusing to repeat the key — this is the actual "single-use"
# security property: it stops a leaked/logged/bookmark-synced claim URL from
# being replayed to re-extract the plaintext key long after the legitimate
# customer received it once.
_MAX_CLAIMS_IN_WINDOW = 5


class ClaimStore:
    """Persistent, one-row-per-checkout record of which transactions have
    been claimed (JSON-file pattern, same as SubscriberStore — checkout
    volume is small, no need for sqlite)."""

    def __init__(self, path: str = _DEFAULT_CLAIM_PATH) -> None:
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

    def get(self, txn_id: str) -> dict[str, Any] | None:
        entry = self._data.get(txn_id)
        return dict(entry) if entry else None

    def record_claim(self, txn_id: str, *, subscription_id: str, now: float | None = None) -> dict[str, Any]:
        """Record (or bump) a claim for `txn_id`. First call creates the
        entry with claim_count=1; subsequent calls increment claim_count and
        keep the original `first_claimed_at` (that timestamp anchors the
        idempotency window, it must not slide forward on repeat claims)."""
        now_f = time.time() if now is None else now
        entry = self._data.get(txn_id)
        if entry is None:
            entry = {
                "txn_id": txn_id,
                "subscription_id": subscription_id,
                "first_claimed_at": now_f,
                "claim_count": 1,
            }
        else:
            entry["subscription_id"] = subscription_id or entry.get("subscription_id")
            entry["claim_count"] = int(entry.get("claim_count", 0)) + 1
        self._data[txn_id] = entry
        self._save()
        return dict(entry)


@dataclass
class ClaimResult:
    """Result of `claim_transaction`. `.status` is one of: claimed,
    already_claimed, pending, not_completed, subscriber_inactive,
    unknown_transaction, upstream_error. HTTP-status mapping is
    brief_server.py's job, not this module's."""

    status: str
    api_key: str | None = None
    tier: str | None = None
    subscription_id: str | None = None
    transaction_status: str | None = None


def claim_transaction(
    txn_id: str,
    *,
    paddle_client: PaddleClient,
    subscriber_store: SubscriberStore,
    claim_store: ClaimStore,
    now: float | None = None,
) -> ClaimResult:
    """Resolve a Paddle `transaction_id` to a customer's API key, safely.

    Never trusts the caller's word that a transaction completed — always
    calls Paddle's own `GET /transactions/{id}` first. Returns a terminal
    `ClaimResult` for every case; retry-vs-terminal semantics belong to the
    HTTP layer (see the docstring status list above and the design doc for
    which statuses the client should poll on vs. treat as final).
    """
    now_f = time.time() if now is None else now

    try:
        transaction = paddle_client.get_transaction(txn_id)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return ClaimResult(status="unknown_transaction")
        return ClaimResult(status="upstream_error")
    except Exception:
        # Network error, timeout, malformed response, etc. — all treated as
        # retryable upstream failure, never as "this transaction doesn't
        # exist" (that would be a false negative with real money on the line).
        return ClaimResult(status="upstream_error")

    txn_status = transaction.get("status")
    if txn_status != "completed":
        return ClaimResult(status="not_completed", transaction_status=txn_status)

    subscription_id = transaction.get("subscription_id")
    if not subscription_id:
        # Completed payment, but Paddle hasn't attached a subscription_id to
        # the transaction yet — the async webhook may simply not have landed.
        return ClaimResult(status="pending")

    subscriber = subscriber_store.get(subscription_id)
    if subscriber is None:
        # The webhook race this whole design exists to handle: browser can
        # reach here before subscription.activated has been processed.
        return ClaimResult(status="pending", subscription_id=subscription_id)

    if not subscriber.get("active"):
        return ClaimResult(status="subscriber_inactive", subscription_id=subscription_id)

    existing = claim_store.get(txn_id)
    if existing is not None:
        age_s = now_f - float(existing.get("first_claimed_at", now_f))
        claim_count = int(existing.get("claim_count", 0))
        if age_s > _CLAIM_WINDOW_S or claim_count >= _MAX_CLAIMS_IN_WINDOW:
            return ClaimResult(status="already_claimed", subscription_id=subscription_id)

    claim_store.record_claim(txn_id, subscription_id=subscription_id, now=now_f)
    return ClaimResult(
        status="claimed",
        api_key=subscriber.get("api_key"),
        tier=subscriber.get("tier"),
        subscription_id=subscription_id,
    )


__all__ = ["ClaimStore", "ClaimResult", "claim_transaction"]
