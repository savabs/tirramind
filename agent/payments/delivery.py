"""API key email delivery — groundwork only, disabled until credentials exist.

Context (2026-08-27 ground truth): `tirramind.com` has no MX records and
`support@` bounces. There is no mailbox. The key-claim flow
(`agent/payments/claim.py` + `/welcome`) already solves *first* delivery via
the checkout redirect + poll, but a customer who closes that tab before
claiming, or whose browser flow fails, still has no recovery path — email is
the second channel for exactly that case, the moment one exists.

This module ships that capability now, wired in but INERT: with no SMTP env
vars set it no-ops loudly (never crashes, never silently swallows) so that
the day the owner's Cloudflare Email Routing SMTP credentials land as env
vars, delivery starts working with a config change, not a code change — the
same pattern as `PaddleConfig` (sandbox → live is env-only).

Stdlib only (`smtplib`/`email.message`) — CLAUDE.md §7 forbids adding a new
dependency without approval, and none is needed here.

Required environment variables (ALL must be set for delivery to be attempted;
any missing → `unconfigured`, not a crash):

    TIRRA_SMTP_HOST        e.g. "smtp.<region>.mailchannels.net" or the
                           Cloudflare Email Routing SMTP relay host.
    TIRRA_SMTP_PORT        default 587 (STARTTLS). Optional.
    TIRRA_SMTP_USERNAME    SMTP auth username.
    TIRRA_SMTP_PASSWORD    SMTP auth password/API token. NEVER log this.
    TIRRA_SMTP_FROM        the From: address, e.g. "no-reply@tirramind.com".
    TIRRA_SMTP_USE_TLS     "true" (default) to STARTTLS; "false" to send
                           plaintext (only ever appropriate for a local test
                           relay, never production — default stays on).

Nothing in this module is called with real credentials anywhere in this
repo today. Do not add any.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

logger = logging.getLogger(__name__)

# The exact env vars required for delivery to be attempted at all.
_REQUIRED_ENV_VARS = (
    "TIRRA_SMTP_HOST",
    "TIRRA_SMTP_USERNAME",
    "TIRRA_SMTP_PASSWORD",
    "TIRRA_SMTP_FROM",
)
_DEFAULT_PORT = 587
_DEFAULT_TIMEOUT_S = 10

# Every failure mode this module can report. `unconfigured` and `error` are
# BOTH non-fatal to the caller — the webhook path must proceed identically
# either way; this status exists for logging/observability, not control flow
# the caller is required to branch on.
_STATUS_SENT = "sent"
_STATUS_UNCONFIGURED = "unconfigured"
_STATUS_ERROR = "error"


@dataclass(frozen=True)
class DeliveryResult:
    """Outcome of one `send_api_key_email` attempt. Never raises to the caller."""

    status: str  # "sent" | "unconfigured" | "error"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == _STATUS_SENT


def _missing_env_vars(env: dict[str, str]) -> list[str]:
    return [name for name in _REQUIRED_ENV_VARS if not (env.get(name) or "").strip()]


def is_configured(env: dict[str, str] | None = None) -> bool:
    """True only if every required SMTP env var is set (non-empty)."""
    e = os.environ if env is None else env
    return not _missing_env_vars(e)


def _build_message(*, to_email: str, api_key: str, tier: str, from_addr: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Your TirraMind API key"
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(
        "Thanks for subscribing to TirraMind.\n\n"
        f"Tier: {tier}\n"
        f"API key: {api_key}\n\n"
        "Keep this key secret — it authenticates every request to your "
        "subscription. If you believe it has been exposed, you can rotate "
        "it via the self-service rotation endpoint.\n"
    )
    return msg


def send_api_key_email(
    to_email: str,
    *,
    api_key: str,
    tier: str,
    env: dict[str, str] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> DeliveryResult:
    """Best-effort delivery of a subscriber's API key by email.

    Contract: NEVER raises. NEVER silently no-ops — every unconfigured or
    failed attempt is logged at a level that shows up by default (warning/
    error), so "why didn't the customer get their email" is always
    grep-able, but the return value alone is enough for a caller that
    chooses not to branch on it (fire-and-forget from the webhook path).
    """
    e = os.environ if env is None else env

    if not to_email or not to_email.strip():
        logger.warning("[delivery] no recipient email on file — cannot send API key email (tier=%s)", tier)
        return DeliveryResult(status=_STATUS_UNCONFIGURED, detail="no recipient email")

    missing = _missing_env_vars(e)
    if missing:
        logger.warning(
            "[delivery] email delivery is UNCONFIGURED — missing env var(s) %s. "
            "The API key for %s was minted but NOT emailed; the customer must "
            "use the /welcome claim flow to retrieve it.",
            ", ".join(missing),
            to_email,
        )
        return DeliveryResult(status=_STATUS_UNCONFIGURED, detail=f"missing: {', '.join(missing)}")

    host = e["TIRRA_SMTP_HOST"].strip()
    try:
        port = int((e.get("TIRRA_SMTP_PORT") or "").strip() or _DEFAULT_PORT)
    except ValueError:
        logger.error("[delivery] TIRRA_SMTP_PORT is not a valid integer: %r", e.get("TIRRA_SMTP_PORT"))
        return DeliveryResult(status=_STATUS_ERROR, detail="invalid TIRRA_SMTP_PORT")
    username = e["TIRRA_SMTP_USERNAME"].strip()
    password = e["TIRRA_SMTP_PASSWORD"]  # not stripped — a trailing space could be a valid token char
    from_addr = e["TIRRA_SMTP_FROM"].strip()
    use_tls = (e.get("TIRRA_SMTP_USE_TLS") or "true").strip().lower() not in ("false", "0", "no")

    message = _build_message(to_email=to_email, api_key=api_key, tier=tier, from_addr=from_addr)

    try:
        with smtplib.SMTP(host, port, timeout=timeout_s) as client:
            if use_tls:
                client.starttls()
            client.login(username, password)
            client.send_message(message)
    except Exception as exc:  # noqa: BLE001 — any SMTP/network failure must not crash the caller
        # Never log `password`; `exc` from smtplib does not include it.
        logger.error("[delivery] failed to send API key email to %s: %s: %s", to_email, type(exc).__name__, exc)
        return DeliveryResult(status=_STATUS_ERROR, detail=f"{type(exc).__name__}: {exc}")

    logger.info("[delivery] API key email sent to %s (tier=%s)", to_email, tier)
    return DeliveryResult(status=_STATUS_SENT)


def attempt_deliver_key_email(
    to_email: str | None,
    *,
    api_key: str | None,
    tier: str,
    env: dict[str, str] | None = None,
) -> DeliveryResult:
    """Guarded wrapper for the webhook path: swallows EVERYTHING beyond what
    `send_api_key_email` already handles, because a subscription activation
    must succeed regardless of what this function does. This is defense in
    depth on top of `send_api_key_email`'s own try/except, not a substitute
    for it — belt and suspenders on the one call that must never take down
    subscription activation.
    """
    if not api_key:
        return DeliveryResult(status=_STATUS_UNCONFIGURED, detail="no api_key to deliver")
    try:
        return send_api_key_email(to_email or "", api_key=api_key, tier=tier, env=env)
    except Exception as exc:  # noqa: BLE001 — absolute last-resort guard
        logger.error("[delivery] unexpected error attempting key email delivery: %s: %s", type(exc).__name__, exc)
        return DeliveryResult(status=_STATUS_ERROR, detail=f"unexpected: {type(exc).__name__}: {exc}")


__all__ = [
    "DeliveryResult",
    "is_configured",
    "send_api_key_email",
    "attempt_deliver_key_email",
]
