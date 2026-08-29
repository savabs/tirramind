"""Propose/apply explicit INR `unit_price_overrides` on the four TirraMind
tier prices, so an India-billing-address customer sees a real rupee price at
checkout instead of Paddle's automatic FX conversion (currently returns USD —
see docs/research or the payments-and-billing report dated 2026-08-28).

THIS SCRIPT DOES NOT MUTATE LIVE PADDLE BY ITSELF. Per this agent's hard
rules it may not create/modify/delete live Paddle products or prices — it
only *proposes* the exact call. Default behavior (no flags, or --dry-run) is
read-only: it prints the exact PATCH call(s) it would make, one per tier,
and the exact equivalent curl. Passing --execute additionally requires
--i-understand-this-hits-paddle-now, and even then this agent itself must
never invoke it against a live catalog — see CLAUDE.md hard rules. The
architect runs `--execute` by hand once they're satisfied with the payload
below.

── Where the INR amounts came from (do not treat as a placeholder) ─────────
Spot rate checked 2026-08-28: USD/INR ~= 95.5 (multiple sources, e.g.
tradingeconomics.com, xe.com — see the payments report for citations).
Naive FX conversion of the four live USD prices at that spot rate:

    Brief          $19  -> raw ~ INR 1,814.50
    Scheduler      $50  -> raw ~ INR 4,775.00
    Entity Graph  $300  -> raw ~ INR 28,650.00
    Data Platform $500  -> raw ~ INR 47,750.00

Two reasons NOT to just round those to the nearest rupee and call it done:

1. Ugly numbers erode trust. "INR 1,814.50" reads like an afterthought FX
   conversion (because it is one), not a price TirraMind chose. Indian SaaS
   and consumer software overwhelmingly uses a charm-pricing convention of
   ending just under a round barrier (INR 1,899 / 4,999 / 29,999 / 49,999),
   the same psychological trick as USD's own $19/$300 (not $18.62).

2. A raw FX conversion has zero margin for INR volatility. The rupee moved
   ~8% against the dollar in the last 12 months alone (per the same rate
   check). A price set at exactly today's spot is stale within weeks and
   would need another manual PATCH to defend margin. Building in a small
   (~+1% to +5%) buffer above spot, then rounding UP to the nearest
   just-under-a-round-number charm price, absorbs realistic near-term
   drift without another repricing pass, while staying materially FX-fair
   to the customer (no tier here is priced more than 5% over raw FX).

This does NOT apply PPP-style deep discounting (e.g. pricing at 40-60% of
FX like consumer/SMB SaaS often does in India). Rationale: TirraMind is not
selling a mass-market consumer tool — the buyer for Entity Graph ($300) and
Data Platform ($500) is a professional/institutional desk evaluating a data
product on its merits ("the integration intelligence" per the owner's own
positioning), not a price-sensitive individual. Discounting a $300/mo
professional data product to ~$150-equivalent for one geography undermines
that positioning and invites confusion about why the same API costs
different amounts for the same access. The buffer above is currency-risk
management, not a market-segmentation discount.

    Brief          INR 1,899  (raw 1,814.50, +4.65%)  ends just under 1,900
    Scheduler      INR 4,999  (raw 4,775.00, +4.69%)  ends just under 5,000
    Entity Graph  INR 29,999  (raw 28,650.00, +4.71%) ends just under 30,000
    Data Platform INR 49,999  (raw 47,750.00, +4.71%) ends just under 50,000
                                                       (gated tier, not yet
                                                       purchasable — priced
                                                       now so nothing needs
                                                       revisiting at launch)

All four sit within a tight, consistent +4.6%-4.7% band above spot FX — the
buffer size itself is deliberately uniform across tiers rather than picked
per-tier, so there's one policy to defend ("we price ~5% above spot, rounded
to a charm number"), not four different justifications.

Paddle's existing account tax_mode is "location" (confirmed live via a real
sandbox pricing-preview call, 2026-08-28 — see the payments report), which
means unit_price is treated as tax-inclusive: a $300 USD price nets exactly
$300.00 at checkout with 18% GST carved out of that total, not added on top.
The INR figures above follow the same convention — INR 29,999 is what an
Indian customer pays in total, not a pre-tax figure needing GST added.

Usage:
    # Preview only (default) — prints the exact PATCH payload per tier, makes no calls
    .venv/bin/python scripts/set_paddle_inr_pricing.py

    # Apply for real — NEVER run this from this agent session
    .venv/bin/python scripts/set_paddle_inr_pricing.py --execute --i-understand-this-hits-paddle-now
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.payments.client import PaddleClient  # noqa: E402
from agent.payments.config import PaddleConfig  # noqa: E402

_PRICING_HTML = Path(__file__).resolve().parents[1] / "products/brief_subscription/pricing.html"

# amount is in paise (INR minor unit), as a string — same convention as
# create_price's `amount` for USD cents.
_INR_OVERRIDES = {
    "brief": {"rupees": 1_899, "amount": "189900"},
    "scheduler": {"rupees": 4_999, "amount": "499900"},
    "entity": {"rupees": 29_999, "amount": "2999900"},
    "data": {"rupees": 49_999, "amount": "4999900"},
}


def _load_tier_price_ids() -> dict[str, str]:
    """Parse TIER_PRICE_IDS straight out of pricing.html so this script always
    targets whatever price IDs are actually wired into checkout right now —
    never a hand-copied snapshot that can drift after the next
    setup_paddle_products.py run."""
    html = _PRICING_HTML.read_text(encoding="utf-8")
    match = re.search(r"const TIER_PRICE_IDS = \{(.*?)\n\s*\};", html, re.S)
    if not match:
        raise RuntimeError("could not find TIER_PRICE_IDS block in pricing.html")
    ids: dict[str, str] = {}
    for tier, price_id in re.findall(r'(\w+):\s*"([^"]+)"', match.group(1)):
        ids[tier] = price_id
    missing = set(_INR_OVERRIDES) - set(ids)
    if missing:
        raise RuntimeError(f"pricing.html TIER_PRICE_IDS is missing tier(s): {sorted(missing)}")
    return ids


def _override_payload(tier: str) -> list[dict]:
    spec = _INR_OVERRIDES[tier]
    return [{"country_codes": ["IN"], "unit_price": {"amount": spec["amount"], "currency_code": "INR"}}]


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
    print(f"[inr-pricing] mode={cfg.mode} api_base={cfg.api_base}")
    print(f"[inr-pricing] price IDs sourced from {_PRICING_HTML}\n")

    for tier, price_id in tier_price_ids.items():
        if tier not in _INR_OVERRIDES:
            continue
        overrides = _override_payload(tier)
        spec = _INR_OVERRIDES[tier]
        url = f"{cfg.api_base}/prices/{price_id}"
        print(f"── {tier} ({price_id}) — INR {spec['rupees']:,} ──────────────────")
        print(f"PATCH {url}")
        print(json.dumps({"unit_price_overrides": overrides}, indent=2))
        print(
            f"curl -sS -X PATCH '{url}' "
            "-H 'Authorization: Bearer $TIRRA_PADDLE_API_KEY' "
            "-H 'Content-Type: application/json' "
            f"-d '{json.dumps({'unit_price_overrides': overrides})}'"
        )
        print()

    if not args.execute:
        print("[inr-pricing] dry run complete — no calls were made. Pass --execute (+ confirmation flag) to apply.")
        return

    if not args.confirm:
        print("[inr-pricing] --execute given without --i-understand-this-hits-paddle-now — refusing to call Paddle.")
        raise SystemExit(2)

    print(
        f"[inr-pricing] EXECUTING against {cfg.api_base} — this will really PATCH "
        f"{'LIVE' if cfg.is_live else 'sandbox'} prices.\n"
    )
    client = PaddleClient(cfg)
    for tier, price_id in tier_price_ids.items():
        if tier not in _INR_OVERRIDES:
            continue
        result = client.update_price(price_id, unit_price_overrides=_override_payload(tier))
        print(f"[inr-pricing] updated {tier} ({price_id}): unit_price_overrides={result.get('unit_price_overrides')}")


if __name__ == "__main__":
    main()
