"""Paddle server-side API client.

Thin, typed wrapper around the Paddle Billing API for the operations the
product needs. Env-driven base URL (sandbox vs live) via PaddleConfig.

Reference: https://developer.paddle.com/api-reference/billing
"""

from __future__ import annotations

from typing import Any

import httpx

from agent.payments.config import PaddleConfig


class PaddleClient:
    """Minimal Paddle Billing API client (read + simple helpers)."""

    def __init__(self, config: PaddleConfig) -> None:
        self.config = config
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    # ── Prices / products ───────────────────────────────────────────────────
    def get_price(self, price_id: str) -> dict[str, Any]:
        """Fetch a price by ID (e.g. to confirm amount in a checkout)."""
        url = f"{self.config.api_base}/prices/{price_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    def list_prices(self, product_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List prices, optionally filtered by product."""
        params: dict[str, Any] = {"limit": limit}
        if product_id:
            params["product_id"] = product_id
        url = f"{self.config.api_base}/prices"
        r = httpx.get(url, headers=self._headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def list_products(self, limit: int = 50) -> list[dict[str, Any]]:
        """List products in the catalog (for audit / mapping)."""
        url = f"{self.config.api_base}/products"
        r = httpx.get(url, headers=self._headers, params={"limit": limit}, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def create_product(
        self,
        *,
        name: str,
        tax_category: str = "standard",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a product in the catalog. Returns the created product (id prefixed 'pro_')."""
        url = f"{self.config.api_base}/products"
        payload: dict[str, Any] = {"name": name, "tax_category": tax_category}
        if description:
            payload["description"] = description
        r = httpx.post(url, headers=self._headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    def create_price(
        self,
        *,
        product_id: str,
        description: str,
        amount: str,
        currency_code: str = "USD",
        interval: str = "month",
        frequency: int = 1,
        name: str | None = None,
        unit_price_overrides: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a recurring price for a product. `amount` is a string in the
        lowest currency denomination (e.g. "50000" = $500.00 USD).

        `unit_price_overrides` (verified against developer.paddle.com,
        2026-08-28 — POST /prices request schema) lets a single price carry
        explicit country-specific amounts instead of relying on Paddle's
        automatic currency conversion from the base `unit_price`. Each entry:
            {"country_codes": ["IN"], "unit_price": {"amount": "41500",
             "currency_code": "INR"}}
        `country_codes` is a list of ISO 3166-1 alpha-2 codes (>=1 entry,
        unique across the whole list — Paddle rejects a country appearing in
        more than one override). This is what lets an explicit INR price be
        set via the API instead of waiting on a dashboard toggle (see
        `update_price` below for changing an EXISTING price's overrides
        in-place, which is the actual mechanism for that — this parameter
        only matters for a brand-new price).

        Returns the created price (id prefixed 'pri_')."""
        url = f"{self.config.api_base}/prices"
        payload: dict[str, Any] = {
            "product_id": product_id,
            "description": description,
            "unit_price": {"amount": amount, "currency_code": currency_code},
            "billing_cycle": {"interval": interval, "frequency": frequency},
        }
        if name:
            payload["name"] = name
        if unit_price_overrides:
            payload["unit_price_overrides"] = unit_price_overrides
        r = httpx.post(url, headers=self._headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    def update_price(
        self,
        price_id: str,
        *,
        amount: str | None = None,
        currency_code: str | None = None,
        unit_price_overrides: list[dict[str, Any]] | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """PATCH an existing price (verified against developer.paddle.com,
        2026-08-28 — PATCH /prices/{price_id}).

        Only the fields explicitly passed are sent — Paddle treats an
        omitted field as "leave unchanged", not "clear it", so this method
        must never send a field the caller didn't ask to change. `amount`
        and `currency_code` are a pair: Paddle's `unit_price` is a single
        object, so both or neither must be given (raises `ValueError`
        otherwise — better a loud local error than silently sending a
        malformed partial `unit_price`).

        This is the actual mechanism for "set an explicit INR price via the
        API instead of waiting on a dashboard toggle" (see this module's
        `create_price` docstring): call this against the LIVE price_id with
        `unit_price_overrides=[{"country_codes": ["IN"], "unit_price":
        {"amount": "<paise>", "currency_code": "INR"}}]`.

        NEVER call this against a live price_id from this codebase without
        the architect's explicit go-ahead — this agent does not create,
        modify, or delete live Paddle products/prices; it proposes the
        exact call and the architect executes it (see CLAUDE.md hard rules).

        Raises `ValueError` if called with nothing to update, or with only
        one of `amount`/`currency_code`. Raises `httpx.HTTPStatusError` on
        an unknown price_id or a rejected payload (e.g. a country_code
        collision across override entries).
        """
        payload: dict[str, Any] = {}
        if amount is not None or currency_code is not None:
            if amount is None or currency_code is None:
                raise ValueError("amount and currency_code must both be provided together")
            payload["unit_price"] = {"amount": amount, "currency_code": currency_code}
        if unit_price_overrides is not None:
            payload["unit_price_overrides"] = unit_price_overrides
        if description is not None:
            payload["description"] = description
        if name is not None:
            payload["name"] = name
        if not payload:
            raise ValueError("update_price called with nothing to update")
        url = f"{self.config.api_base}/prices/{price_id}"
        r = httpx.patch(url, headers=self._headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ── Subscriptions ──────────────────────────────────────────────────────
    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Fetch a subscription by ID (status, customer, items)."""
        url = f"{self.config.api_base}/subscriptions/{subscription_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    def get_subscription_management_urls(self, subscription_id: str) -> dict[str, str | None]:
        """Clean accessor for the customer-portal links Paddle attaches to a
        subscription, for an account page: `{"update_payment_method": str |
        None, "cancel": str}`.

        Verified against developer.paddle.com (2026-08-28, GET /subscriptions
        response schema): `management_urls` is a required object on every
        subscription with `cancel` (always a string — the hosted portal's
        cancellation page) and `update_payment_method` (nullable — Paddle
        returns null for a manually-collected/invoiced subscription, since
        there is no self-service payment method to update).

        Intended caller: the account-page HTTP route (owned by
        api-backend-engineer, not this file) — it needs exactly these two
        links and nothing else out of `get_subscription()`'s much larger
        payload. Raises `httpx.HTTPStatusError` on an unknown/inaccessible
        subscription_id, same as `get_subscription()` itself; the route
        layer decides how to surface that (e.g. 404 vs hide the section).
        """
        subscription = self.get_subscription(subscription_id)
        urls = subscription.get("management_urls") or {}
        return {
            "update_payment_method": urls.get("update_payment_method"),
            "cancel": urls.get("cancel"),
        }

    # ── Transactions ────────────────────────────────────────────────────────
    def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """Fetch a transaction by ID (status, subscription_id, customer).

        Used by the key-claim flow (agent/payments/claim.py) to independently
        confirm — server-side, against Paddle's own API — that a `txn_id` the
        browser hands back after checkout is real and completed, before ever
        looking up or returning a subscriber's API key. Raises
        `httpx.HTTPStatusError` (404) for an unknown transaction id; callers
        distinguish "unknown" from "not yet completed" by checking the raised
        status code vs. the `status` field of a successfully fetched one.
        """
        url = f"{self.config.api_base}/transactions/{transaction_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ── Customers ──────────────────────────────────────────────────────────
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Fetch a customer by ID (for Retain pwCustomer)."""
        url = f"{self.config.api_base}/customers/{customer_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ── IP allowlist source of truth ────────────────────────────────────────
    @staticmethod
    def fetch_webhook_ips() -> list[str]:
        """Paddle's current webhook IPs (data.ipv4_cidrs, /32 CIDRs).

        This is the source of truth for allowlisting webhook traffic — never
        hard-code the list; it can change.
        Reference: https://api.paddle.com/ips
        """
        r = httpx.get("https://api.paddle.com/ips", timeout=30)
        r.raise_for_status()
        return r.json().get("data", {}).get("ipv4_cidrs", [])


__all__ = ["PaddleClient"]
