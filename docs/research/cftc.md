---
title: "Feature: CFTC Commitments of Traders (COT) Tool"
tags:
  - doc/research
  - topic/cftc
---

# Feature: CFTC Commitments of Traders (COT) Tool

## Current Architecture
- Follows Tool ABC pattern (base.py): name, description, parameters, execute()
- Registration in cli.py → ToolRegistry
- Bandit arm in bandit.py for autonomous goal selection
- DataCache for caching (agent/data/cache.py)
- Schema validation via jsonschema (base.py validate_args)

## Data Source

### Primary: Disaggregated Futures-Only Report
- **Weekly file** (latest week only, no headers): `https://www.cftc.gov/dea/newcot/f_disagg.txt`
- **Historical ZIPs** (yearly, WITH headers): `https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip`
  - Contains `f_year.txt` — full year CSV with 191 columns
  - Available 2006–2026
- **Size**: Weekly ~430KB, yearly ZIP ~530KB
- **Rate limiting**: None observed. Public data.
- **Schedule**: Snapshot Tuesday, released Friday 3:30 PM ET

### Verified Column Layout (191 columns, from historical ZIP headers)
Key columns for quant signals (index → name):
- [0] Market_and_Exchange_Names
- [2] Report_Date_as_YYYY-MM-DD
- [3] CFTC_Contract_Market_Code (e.g. "001602" = Wheat-SRW)
- [6] CFTC_Commodity_Code (e.g. "001")
- [7] Open_Interest_All
- [8-9] Prod_Merc_Positions_Long/Short_All (producers/merchants = commercials)
- [10-12] Swap_Positions_Long/Short/Spread_All
- [13-15] M_Money_Positions_Long/Short/Spread_All (managed money = speculators)
- [16-18] Other_Rept_Positions_Long/Short/Spread_All
- [55-70] Change_in_* (weekly deltas for all categories)
- [71-86] Pct_of_OI_* (percentage of open interest)
- [119-132] Traders_* (number of traders)
- [161-168] Conc_Gross/Net_LE_4/8_TDR_* (concentration of top 4/8 traders)
- [185] Contract_Units (e.g. "(CONTRACTS OF 5,000 BUSHELS)")

### TFF Report (Financial Futures)
- URL: `https://www.cftc.gov/dea/newcot/FinFutWk.txt`
- 87 columns, different trader categories: Dealer, Asset Manager, Leveraged Funds, Other
- Historical: `https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip`
- Covers: equity indices, bonds, currencies, rates

## Observations
- Weekly file has NO headers (must apply column layout from historical format)
- Historical ZIP has headers — use as authoritative reference
- 266 contracts in weekly disagg, 84 in TFF
- Data uses "." for missing values (e.g., trader counts when <4 traders in category)
- Field [190] `FutOnly_or_Combined` — always "FutOnly" in futures-only reports
- Contract names include exchange: "WHEAT-SRW - CHICAGO BOARD OF TRADE"

## Key Quant Signals

1. **Managed Money Net** = M_Money_Long - M_Money_Short → speculator positioning
2. **Producer/Merchant Net** = Prod_Merc_Long - Prod_Merc_Short → commercial hedging pressure
3. **COT Index** = (Current_Net - Min_52w) / (Max_52w - Min_52w) × 100 → positioning extremes
4. **Weekly Flow** = Change_in_M_Money_Long + Change_in_M_Money_Short → momentum/capitulation
5. **Concentration** = Conc_Net_LE_4_TDR → top-4 trader dominance (crowding risk)
6. **Open Interest Change** = Change_in_Open_Interest → market participation signal

## Contract Code → Ticker Mapping (key contracts)
| CFTC Code | Name | Approx Ticker |
|-----------|------|---------------|
| 001602 | Wheat-SRW | ZW |
| 002602 | Corn | ZC |
| 005602 | Soybeans | ZS |
| 006765 | WTI Crude | CL |
| 023651 | Natural Gas | NG |
| 084691 | Gold | GC |
| 085692 | Silver | SI |
| 090741 | Canadian Dollar | 6C |
| 092741 | Swiss Franc | 6S |
| 096742 | British Pound | 6B |
| 097741 | Japanese Yen | 6J |
| 098662 | US Dollar Index | DX |
| 099741 | Euro FX | 6E |
| 13874A | Bitcoin | BTC |
| 13874P | Ether | ETH |

## Risks
- Weekly data has a 3-day lag (Tuesday snapshot → Friday release). Not real-time.
- The weekly flat file has no headers, must apply header mapping from historical format.
- Missing values as "." need safe parsing.
- Some contracts appear/disappear across weeks.
- File format could change without notice (unlikely, been stable for years).

## Implementation Approach
- **Two modes**: `latest` (weekly file, fast) and `historical` (yearly ZIPs, for signal computation)
- **Historical needed for COT Index**: Need 52 weeks of history to compute positioning extremes
- Use historical ZIP for bootstrap, then weekly for updates
- Cache both: yearly file aggressively (changes weekly at most), weekly file for 6 hours
- Parse with csv module, apply header mapping from historical format
- Filter by contract name substring (user provides "wheat" or "crude" etc.)
- Return: positioning summary, signals, raw data in ToolResult.data

---

## Related

- [[cftc_spec|Spec: Cftc]]
