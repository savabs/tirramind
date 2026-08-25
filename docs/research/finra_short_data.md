---
title: "Feature: FINRA Short Selling Data (Dark Pool + Short Volume + Short Interest)"
tags:
  - doc/research
  - topic/finra
---

# Feature: FINRA Short Selling Data (Dark Pool + Short Volume + Short Interest)

## Current Architecture

- Existing insider flow tools: `insider_filings.py` (Form 4), `form144.py` (sell intent)
- Existing positioning tool: `cftc.py` (futures)
- Missing: equity-level short selling data — the other side of informed flow
- Base class: `Tool(ABC)` in `agent/tools/base.py`
- Cache: `DataCache` in `agent/data/cache.py`

## API Discovery Results

### What's Free (No Auth)

**1. Reg SHO Daily Short Volume** — PRIMARY SIGNAL
- Endpoint: `POST https://api.finra.org/data/group/otcMarket/name/regShoDaily`
- Content: daily short volume by ticker, per reporting facility
- Partition keys: `tradeReportDate` (required for sorting)
- Fields: `securitiesInformationProcessorSymbolIdentifier`, `totalParQuantity`, `shortParQuantity`, `shortExemptParQuantity`, `reportingFacilityCode`, `marketCode`
- Facilities: NYTRF (NYSE TRF), NQTRF (NASDAQ TRF), NCTRF (CBOE TRF)
- Volume: ~27,042 records/day (3 facilities × ~9,000 tickers)
- Freshness: T+0 or T+1 (today's date returns data)
- Rate limit: no documented limit, but API returns max 5000 per request
- Pagination: `offset` + `limit` params work correctly
- No API key required

**2. Consolidated Short Interest** — SUPPLEMENTARY
- Endpoint: `POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`
- Content: accumulated short positions, bi-monthly (mid-month + end-of-month)
- Partition keys: `settlementDate` (required)
- Fields: `symbolCode`, `settlementDate`, `currentShortPositionQuantity`, `previousShortPositionQuantity`, `changePercent`, `daysToCoverQuantity`, `averageDailyVolumeQuantity`, `marketClassCode`, `issueName`
- Freshness: ~2 month lag (latest as of Mar 25, 2026 = Jan 30, 2026)
- No API key required

### What's Stale or Gated

**3. ATS Weekly Summary (Dark Pool per-venue)** — STALE
- Endpoint: `GET/POST https://api.finra.org/data/group/otcMarket/name/weeklySummary`
- Only returns data for week of 2023-11-06 (last available)
- Fields: per-ATS venue breakdown (JPBX, UBSA, VIRTU, etc.) — shows which dark pool is executing
- Status: appears discontinued or moved behind auth
- **Verdict: unusable for current data**

**4. TRACE Bond Data (Corporate + Treasury)** — AUTH GATED
- Endpoints: `fixedIncomeMarket/*` — all return 401
- Would have shown: corporate bond spread changes, treasury activity
- **Verdict: requires FINRA registration/auth — violates $0 principle for now**

## Signal Theory

### Reg SHO Short Volume Ratio — Why This Matters

The short volume ratio = shortParQuantity / totalParQuantity for a given ticker on a given day.

**Baseline behavior:** Market makers routinely sell short to provide liquidity (fill incoming buy orders). This creates a "normal" short volume ratio of ~40-55% for liquid names. The ratio itself is NOT the signal — the *deviation from the stock's own baseline* is the signal.

**Observed baselines (2026-03-24):**
| Ticker | Total Vol | Short Vol | Ratio | Notes |
|--------|-----------|-----------|-------|-------|
| NVDA | 60.7M | 26.1M | 43.0% | Typical mega-cap |
| TSLA | 32.1M | 15.1M | 46.9% | Elevated |
| AMZN | 12.2M | 2.9M | 23.6% | Low — net buying pressure |
| SPY | 23.3M | 12.2M | 52.5% | ETF — high is normal |
| QQQ | 16.0M | 9.5M | 59.8% | ETF — very high |
| TLT | 9.1M | 5.2M | 57.5% | Bond ETF |
| HYG | 30.3M | 18.4M | 60.9% | High-yield bond ETF — bearish signal |

**Signal extraction patterns:**
1. **Ratio spike**: When a stock's short ratio exceeds its 5-day or 20-day moving average by >1.5σ, directional shorting is increasing. Bearish.
2. **Ratio collapse**: When ratio drops >1.5σ below average, short covering is occurring. Bullish (squeeze potential).
3. **Cross-asset divergence**: HYG at 60.9% while SPY at 52.5% = credit market more bearish than equity. Credit leads equity by 2-4 weeks (academic evidence: Acharya & Johnson 2007).
4. **Anomaly scan**: Fetch all tickers on a day, rank by ratio deviation from sector baseline → find the tickers where someone is aggressively shorting.

### Short Interest — Complement to Daily Volume

- `daysToCoverQuantity` > 5 days = high squeeze risk
- `changePercent` > +20% in one reporting period = new bearish positioning building
- Cross-reference: rising short interest + Form 144 sell-intent filing = bearish confirmation
- Cross-reference: rising short interest + Form 4 insider BUYING = potential squeeze setup

### Short Exempt Volume — Hidden Signal

`shortExemptParQuantity` = short sales exempt from the uptick rule (under Reg SHO Rule 201). These are typically market makers exercising their bona fide market-making exemption. Spikes in exempt volume can indicate forced selling or unusual market-making activity.

## Cross-Signal Opportunities

| FINRA Signal | Cross With | Combined Signal |
|-------------|-----------|-----------------|
| High short ratio spike | Form 144 sell-intent cluster | Double bearish confirmation |
| Short ratio collapse + high DTC | Form 4 insider buying cluster | Squeeze setup |
| HYG short ratio > SPY short ratio | CFTC Treasury positioning | Credit stress leading equity |
| Anomalous short volume on ticker | GDELT events for sector | Event-driven short selling |

## Risks

1. **Market maker noise**: 40-55% short volume is NORMAL for market makers. Must distinguish directional from liquidity shorts — use deviation from baseline, not absolute ratio.
2. **ETF creation/redemption**: ETF short volume includes authorized participant activity. ETF ratios are structurally higher.
3. **Partial data**: Only covers off-exchange (TRF) volume, not lit exchange volume. TRF is ~40-50% of total volume for most NMS stocks.
4. **Pagination cost for scan mode**: 27,042 records per day requires 6 API calls to fetch all. For scan mode, consider capping or using ticker-specific queries.
5. **Short interest lag**: 2-month delay means it's confirmation, not leading signal.
6. **API stability**: FINRA already deprecated the ATS weekly data (only 2023 available). No guarantees on regShoDaily longevity.

## Data Requirements

- **No API key** — both endpoints are free/public
- **Rate limiting**: not documented, but respect 0.1s delay between requests
- **Cache TTL**: regShoDaily = 24hr (daily data), consolidatedShortInterest = 7 days (bi-monthly)

## Design Decision

**Tool name:** `finra_short_volume` (not `finra_data` — scope to what actually works)

**Two modes:**
1. `short_volume` — Reg SHO daily short volume for specific tickers or scan
2. `short_interest` — Consolidated short interest for specific tickers

**Why not "dark pool" in the name:** The ATS per-venue dark pool data is stale (2023 only). Reg SHO covers TRF (off-exchange) volume which includes dark pools, but we can't attribute to specific venues. Honest naming.

**Bandit arm:** `institutional_flow` — covers the short-selling side of institutional activity. Complements `insider_flow` (buying side) and `futures_positioning` (derivatives side).

## Related

- [[finra_short_volume_spec|Spec: FINRA Short Volume]]
