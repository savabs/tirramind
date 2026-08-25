---
title: "Signal Primer — What TirraMind Actually Sees"
tags:
  - doc/research
  - topic/signals
  - phase/1
  - status/active
---

# Signal Primer — What TirraMind Actually Sees

**Status:** active
**Purpose:** plain-language guide to the signals TirraMind produces, so we can decide which to trust and which power a product. Written from real code + real fired output, not marketing.

## Ground truth first

- TirraMind has **63 data tools**, all hitting **free, no-auth public APIs**.
- Foundation verified: `10,461 tests pass` (fast suite), core imports.
- **Live-fired signals** (real API calls, no auth, confirmed working on 2026-08-23):
  | Signal family | Tool | Live proof |
  |---|---|---|
  | Government/procurement | `gov_contracts` (US) | ✅ VALHALLA ENGINEERING / Dept of Veterans Affairs / "RENOVATE PHARMACY CACHE" award + 3 more (USASpending) |
  | Market positioning | `cftc` | ✅ NAT GAS + WTI: open interest, managed-money net, Δ (CFTC COT 2026-08-18) |
  | Physical/shipping | `ais_vessel` | ✅ 309 Finnish port calls across 28 ports, vessel routes (cargo) |

  So "does it work?" — the 3 signal families that matter most for a market/contract-intelligence product all **actually fire** with real data, free.

## The signal families

Signals group into 5 families. Each is: *what it is* → *what data* → *what it'd mean for a product*.

### 1. Market positioning & instrument signals
| Tool | What it sees | Meaning for a product |
|---|---|---|
| `cftc` | Managed-money futures positioning (CFTC) | Extreme positioning → possible top/bottom |
| `options_chain` | Options chain EOD | Vol/flow context |
| `instrument_universe` | returns, volume, volatility per instrument | Which assets move |
| `finra_short_volume` | short interest | Crowding / squeeze risk |
| `polymarket` / `polymarket_whales` | prediction-market probabilities, whale trades | Aggregated likelihood of events |
| `defi_flows` | protocol TVL change | Capital rotation |
| `whale_alert` | BTC large transfers | Whale behavior |

### 2. Corporate / insider / company signals
| Tool | What it sees | Meaning |
|---|---|---|
| `insider_filings` | insider trades | Insider confidence |
| `form144` | planned insider sales | Pre-sale signal |
| `lobbying` | lobbying spend | Policy exposure |
| `bankruptcy_court` / `creditor_filings` | bankruptcy status, creditor filings | Distress early-warning |
| `patent_filings` | patent activity | Innovation/moat signal |
| `drug_regulatory` | drug approvals | Biotech catalysts |
| `academic_preprints` | research velocity | Science → industry lead |
| `cert_transparency` | TLS cert issuance | New infrastructure deployment |

### 3. Macro / sovereign / geo signals
| Tool | What it sees | Meaning |
|---|---|---|
| `central_bank_balance` | CB balance sheets + policy rates | Monetary stance |
| `sovereign_debt` | sovereign yields | Country stress |
| `global_pmi` | economic activity | Growth regime |
| `comtrade` | trade flows | Trade/goods cycles |
| `capital_flows` | cross-border capital | Capital rotation |
| `gdelt` | geopolitical events | Event-driven risk |
| `political_risk` | campaign finance | Policy direction |
| `migration_flows` | migration pressure | Demographic pressure |

### 4. Physical / real-economy signals
| Tool | What it sees | Meaning |
|---|---|---|
| `ais_vessel` | vessel positions + port calls | Shipping/trade flow |
| `energy_supply` | petroleum inventories, rig counts | Energy tightness |
| `electricity_monitor` | grid demand | Industrial activity |
| `transport_throughput` | border throughput, interconnections | Logistics load |
| `weather_alerts` / `earthquake_proximity` | weather/hazard events | Event risk windows |
| `satellite_activity` | satellite observation (needs key) | Real-economy visual state |
| `nightlight_activity` | nighttime light patterns | Economic activity proxy |
| `disease_surveillance` | pathogen levels | Health crises |
| `food_security` | food pressure | Insecurity risk |
| `labor_disruptions` | labor action | Disruption risk |

> ⚠️ several of these (satellite_activity, fire modes, FOIA modes) need an API key or specific key; the fire-free subset is the "always works" subset.

> 💡 The physical layer is the genuinely rare one. Most consumers/competitors use layers 1–3. Very few products combine positioning + entity graph + physical.

### 5. Government / procurement signals (the Tender Alpha base — most important for us)
| Tool | What it sees | Meaning |
|---|---|---|
| `gov_contracts` | **`contract awards US (USASpending) + UK (Contracts Finder)** — verified live | recipient, agency, award ID, amount, date, description (tested above: VALHALLA ENGINEERING / VA / RENOVATE PHARMACY CACHE) |
| `sanctions_monitor` | sanctions listings | Policy restriction |
| `foia_requests` | investigation signals (needs key) | Legal escalation |
| `regulatory_gazette` | regulatory velocity | Reg pressure |
| `building_permits` | construction permits | Construction / housing activity |

**This family is directly reusable for the government-contract intelligence product we discussed.** The award data is the raw material for P(win) — you can score "which small contracts are worth a small business's time."

---

## What "include whatever signal that works" means operationally

You said: include whatever signal works, and you need to understand it first. So the rule is **trial-then-trust, not hype-then-assume**:

1. Each signal candidate runs against its free API **for real** (done for gov_contracts ✅).
2. We record what it would take to make it product-useful for a contract alert product (done above for contracts).
3. Signals that produce actual edge are kept; ones requiring keys are flagged "needs key."

Only signals that (a) run free and (b) add information a competitor doesn't have should power a paid product.

---

## Next

- [x] Verify TirraMind runs: venv + deps installed, core imports OK, `10,461` fast tests pass
- [x] Live-fire signal families: `gov_contracts` (US awards), `cftc` (commodity positioning), `ais_vessel` (shipping) — all real data, free
- [x] Cross-domain proof: 12,271 links, US cross-domain footprint (5 macro domains), contract-recipient companies linked via `awarded_by`/`operates_in` — see [[cross_domain_signal_proof]]
- [ ] Close the company-level overlap gap (contract recipient × shipping/filings/GDELT in one window) — the actual Tender Alpha build
- [ ] Decide which 1–3 signal families to pair with the contract-alert product (Tender Alpha)
- [ ] Lock the Tender Alpha engine scope from that decision

## Related
- [[revenue_plan_2026-05-08]]
- [[quant_training_ground]]
- [[ghost_pattern_income_task]]
- [[gov_contracts]]