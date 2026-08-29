"""Propose/apply a 14-day CARD-REQUIRED trial_period on selected TirraMind
tier prices.

THIS SCRIPT DOES NOT MUTATE LIVE PADDLE BY ITSELF — same contract as
set_paddle_inr_pricing.py. Default (no flags, or --dry-run) prints the exact
PATCH payload per tier and makes no calls. --execute (+ the confirmation
flag) actually calls Paddle; this agent must never pass --execute against a
live catalog itself (see CLAUDE.md hard rules) — the architect runs it.

── Field shape (verified against developer.paddle.com, 2026-08-28) ─────────
`trial_period` lives on the PRICE object (both POST /prices and
PATCH /prices/{id}):

    "trial_period": {
        "interval": "day" | "week" | "month" | "year",
        "frequency": <int, minimum 1>,
        "requires_payment_method": <bool, default true>
    }

`requires_payment_method` is nested INSIDE trial_period, not a sibling
top-level field — confirmed by fetching both the create-price and
update-price schema pages independently and cross-checking they agree.
There is also an optional `unit_price` nested inside trial_period for a
"paid trial" (charge a reduced amount during the trial itself); we don't
use it here, because a card-required *free* trial just needs
requires_payment_method: true with no nested unit_price — Paddle's own
error page (errors/prices/trial_is_either_paid_or_cardless) confirms the
only invalid combination is a nonzero nested unit_price together with
requires_payment_method: false, which does not apply here.

requires_payment_method defaults to true, so a bare
{"interval": "day", "frequency": 14} would already be card-required. It is
passed explicitly below anyway — the owner was explicit that card-required
is a deliberate choice, not a default that happened to be there, so the
payload should say so rather than rely on an implicit default silently
doing the right thing.

Card-required is also the ONLY option Paddle Checkout (Paddle.js overlay,
what pricing.html actually uses) can support for a genuinely cardless flow
anyway — a true cardless trial needs a server-created transaction + Paddle's
one-page (non-overlay) checkout, which is a different integration this repo
does not have. So "card-required" isn't just the owner's preference here;
it's the only trial shape the current checkout integration can drive.

── Which tiers get a trial (see the accompanying report for full reasoning) ──
Recommended: scheduler, entity. NOT brief, NOT data.

  - brief ($19/mo, weekly deliverable): a 14-day trial spans exactly ~2
    weekly issues. A no-cost trial-and-cancel gets two real briefs delivered
    for $0 before the first charge would ever fire — the product's own
    cadence turns the standard 14-day window into a built-in abuse vector
    that doesn't exist for the other tiers. The $19 price point is already
    low-friction; a trial defends against a risk (abuse) that isn't the
    real barrier to purchase (price) for this tier.
  - scheduler ($50/mo): a real integration decision (does the schedule/API
    fit an existing workflow), not a content-consumption decision. Getting
    that answer costs the trialing customer real setup effort, which is a
    natural brake on the same abuse pattern that hurts `brief`. Recommended.
  - entity ($300/mo): a considered purchase for a professional/institutional
    buyer evaluating data quality and entity coverage before committing.
    Directly matches "prove it's worth $300 before I commit" — the textbook
    case for a trial. Recommended.
  - data ($500/mo): currently GATED — not purchasable at all (TIER_AVAILABILITY
    .data = false in pricing.html, backed by only ~296 rows per the live
    state notes). Configuring a trial for a tier nobody can buy yet is
    premature, and when it IS turned on, evaluating a still-thin dataset
    during a no-commitment trial window is a worse first impression than
    the current committed-from-day-one purchase. Revisit once the data
    platform is actually enabled and its depth has been reassessed.

Usage:
    # Preview only (default)
    .venv/bin/python scripts/set_paddle_trial_periods.py

    # Apply for real — NEVER run this from this agent session
    .venv/bin/python scripts/set_paddle_trial_periods.py --execute --i-understand-this-hits-paddle-now
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.payments.config import PaddleConfig  # noqa: E402

# NOTE ON WHY THIS SCRIPT DOES NOT USE agent/payments/client.py's
# `PaddleClient.update_price` for the --execute path: that method (as of
# 2026-08-28) only accepts amount/currency_code/unit_price_overrides/
# description/name — it has no `trial_period` parameter. Extending it is out
# of scope here: agent/payments/client.py is explicitly NOT owned by this
# task (foundation agent owns agent/payments/*), so rather than either (a)
# editing a file outside this task's ownership, or (b) shipping a script
# whose --execute path silently can't do what its dry-run preview promises,
# this script makes the PATCH call directly via httpx, matching
# PaddleClient's own request shape exactly (same headers, same
# raise_for_status-then-.json()["data"] pattern). Flagged in this agent's
# report as a real gap: PaddleClient.update_price should eventually grow a
# `trial_period` parameter so trial config doesn't need this workaround.

_PRICING_HTML = Path(__file__).resolve().parents[1] / "products/brief_subscription/pricing.html"

# Tiers recommended for a trial. See docstring above for the "why" per tier.
_TRIAL_TIERS = ["scheduler", "entity"]

_TRIAL_PERIOD = {"interval": "day", "frequency": 14, "requires_payment_method": True}


def _load_tier_price_ids() -> dict[str, str]:
    html = _PRICING_HTML.read_text(encoding="utf-8")
    match = re.search(r"const TIER_PRICE_IDS = \{(.*?)\n\s*\};", html, re.S)
    if not match:
        raise RuntimeError("could not find TIER_PRICE_IDS block in pricing.html")
    ids: dict[str, str] = {}
    for tier, price_id in re.findall(r'(\w+):\s*"([^"]+)"', match.group(1)):
        ids[tier] = price_id
    missing = set(_TRIAL_TIERS) - set(ids)
    if missing:
        raise RuntimeError(f"pricing.html TIER_PRICE_IDS is missing tier(s): {sorted(missing)}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Actually call PATCH /prices/{id} against Paddle")
    parser.add_argument(
        "--i-understand-this-hits-paddle-now",
        action="store_true",
        dest="confirm",
        help="Required alongside --execute as an explicit second confirmation",
    )
    args = parser.parse_args()

    tier_price_ids = _load_tier_price_ids()
    cfg = PaddleConfig.from_env()
    print(f"[trial-config] mode={cfg.mode} api_base={cfg.api_base}")
    print(f"[trial-config] price IDs sourced from {_PRICING_HTML}")
    print(
        f"[trial-config] tiers getting a trial: {_TRIAL_TIERS} (brief and data deliberately excluded — see docstring)\n"
    )

    for tier in _TRIAL_TIERS:
        price_id = tier_price_ids[tier]
        url = f"{cfg.api_base}/prices/{price_id}"
        payload = {"trial_period": _TRIAL_PERIOD}
        print(f"── {tier} ({price_id}) — 14-day card-required trial ──────────────────")
        print(f"PATCH {url}")
        print(json.dumps(payload, indent=2))
        print(
            f"curl -sS -X PATCH '{url}' "
            "-H 'Authorization: Bearer $TIRRA_PADDLE_API_KEY' "
            "-H 'Content-Type: application/json' "
            f"-d '{json.dumps(payload)}'"
        )
        print()

    if not args.execute:
        print("[trial-config] dry run complete — no calls were made. Pass --execute (+ confirmation flag) to apply.")
        return

    if not args.confirm:
        print("[trial-config] --execute given without --i-understand-this-hits-paddle-now — refusing to call Paddle.")
        raise SystemExit(2)

    print(
        f"[trial-config] EXECUTING against {cfg.api_base} — this will really PATCH "
        f"{'LIVE' if cfg.is_live else 'sandbox'} prices.\n"
    )
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    for tier in _TRIAL_TIERS:
        price_id = tier_price_ids[tier]
        url = f"{cfg.api_base}/prices/{price_id}"
        r = httpx.patch(url, headers=headers, json={"trial_period": _TRIAL_PERIOD}, timeout=30)
        r.raise_for_status()
        result = r.json().get("data", {})
        print(f"[trial-config] updated {tier} ({price_id}): trial_period={result.get('trial_period')}")


if __name__ == "__main__":
    main()
