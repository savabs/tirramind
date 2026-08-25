---
title: "Feature: 7b-AA — Global PMI / Leading Indicators"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/global-pmi
---

# Feature: 7b-AA — Global PMI / Leading Indicators

## Current Architecture
- **Layer:** 1 (Surveillance Surface) — `agent/tools/`
- **Pattern:** Tool inherits from `Tool` base class, uses `httpx` for HTTP, `DataCache` for caching
- **Existing overlap:** `macro_data.py` covers FRED/ECB/World Bank general macro series. This tool focuses specifically on OECD Composite Leading Indicators (CLI) and related leading indicator composites — a distinct, more targeted data surface.
- **Registration:** `agent/cli.py` (import + `registry.register()`), `agent/learning/bandit.py` (GoalArm)

## Data Sources

### Primary: OECD SDMX API — Composite Leading Indicators (CLI)
- **API:** `https://sdmx.oecd.org/public/rest/data/`
- **Auth:** None required. Free, open API.
- **License:** OECD Terms and Conditions — data is free to use, subject to ToC. Generally permissive for derived analysis.
- **Format:** JSON (`format=jsondata`), CSV (`format=csvfilewithlabels`), XML (default SDMX)
- **Rate limits:** Rate limiting introduced, best practices recommended. No hard numbers published.

#### Key Dataflows

1. **CLI — Composite Leading Indicators:** `OECD.SDD.STES,DSD_STES@DF_CLI`
   - Monthly, 40+ countries + OECD aggregate
   - Dimension path: `{country}.M.LI...AA...H` for amplitude-adjusted CLI
   - Signal: CLI turning points precede GDP turning points by 6-9 months. Cross-country CLI divergence = relative growth momentum signal.
   - Countries available: USA, GBR, DEU, FRA, JPN, CHN, KOR, AUS, CAN, ITA, ESP, BRA, IND, MEX, TUR, ZAF, IDN, RUS, OECD, G7, EA19, etc.

2. **BCI — Business Confidence Indicators:** `OECD.SDD.STES,DSD_STES@DF_BCI`
   - Monthly, survey-based business confidence
   - Signal: Divergence between BCI and CLI = manufacturing vs services stress

3. **CCI — Consumer Confidence Indicators:** `OECD.SDD.STES,DSD_STES@DF_CCI`
   - Monthly, consumer confidence surveys
   - Signal: Consumer confidence leads consumer spending by 2-3 months

### OECD SDMX Query Structure
```
{host}/{agency},{dataflow},{version}/{data_selection}?{params}

Example (CLI for USA, monthly, amplitude-adjusted):
https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI/USA.M.LI...AA...H?startPeriod=2024-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels
```

### Why Not FRED for PMI?
- ISM Manufacturing PMI (NAPM) was **removed from FRED** in June 2016 per ISM licensing restrictions
- ISM data requires a paid ISM subscription for programmatic access
- FRED still has: INDPRO (Industrial Production), UMCSENT (Consumer Sentiment) — but these are US-only
- OECD CLI is the best free alternative: covers 40+ countries, monthly, no auth, well-maintained API

### Secondary: FRED Series (via existing macro_data tool)
The existing `macro_data` tool already provides FRED access. For US-specific leading indicators that are on FRED (INDPRO, UMCSENT, T10Y2Y, etc.), users should use `macro_data`. This tool focuses on the **global** dimension via OECD.

## Signal Theory

Leading indicators are the **backbone of macro regime detection**:
- **OECD CLI** is specifically designed to detect turning points 6-9 months ahead of GDP. A CLI reading below 100 and declining = contraction signal.
- **Cross-country CLI comparison** reveals relative growth momentum — a divergence between US CLI and China CLI historically precedes FX moves and commodity flows.
- **BCI-CLI divergence** signals manufacturing stress before it shows in hard data.
- **Simultaneous CLI decline across G7** = synchronized global slowdown (rare but high-impact).
- **CLI rate of change** (monthly momentum) is more actionable than the level itself.

The key insight: CLI is a composite of components (stock prices, building permits, money supply, yield curve, etc.) already filtered by the OECD's statistical office. It's a pre-processed leading signal.

## Observations
- OECD SDMX API is well-documented and confirmed working (live XML response received during research)
- CSV format with labels is the most convenient for parsing
- The Python example in OECD docs shows: `requests.get(url)` → `pd.read_csv(StringIO(response.text))`
- CLI data goes back to 1960s for major economies
- Monthly frequency is lower than daily tools, but the signal is higher-quality (composite, seasonally adjusted)

## Risks
- OECD API has undocumented rate limits — need conservative request patterns
- SDMX dimension strings are arcane (`USA.M.LI...AA...H`) — need careful mapping
- Data revisions: OECD can revise historical CLI values (use `dimensionAtObservation=AllDimensions` to detect)
- Some countries have sparse/discontinued series
- API response format (SDMX XML) is complex; CSV format is much simpler to parse
- Large multi-country queries can timeout — query per-country or use small country groups

## Data Requirements
- CLI values by country (monthly, amplitude-adjusted)
- BCI values by country (monthly)
- CCI values by country (monthly)
- Historical depth: at least 5 years for regime detection, ideally 10+
- Country coverage: G7 + BRICS + OECD aggregate at minimum

## Math/Algorithm Survey
- CLI turning point detection: zero-crossing of month-over-month change
- Cross-country momentum spread: CLI(US) - CLI(CN), CLI(US) - CLI(DE)
- Rate of change: (CLI_t - CLI_{t-6}) / CLI_{t-6} for 6-month momentum
- Regime classification: CLI > 100 expanding, < 100 contracting, + direction
- Hurst exponent on CLI series for persistence/mean-reversion detection
- All computed internally; no external libs needed

## OSS/External Research
- `pandasdmx` Python library exists for SDMX parsing, but adds dependency — CSV format avoids this
- OECD provides official Python example using `requests` + `pandas` for CSV
- No license conflicts — OECD ToC allows derived analysis
- Search terms used: "OECD CLI API", "OECD SDMX REST API", "composite leading indicators API python", "ISM PMI alternative data"

---

## Related

- [[7b-AA_global_pmi_spec|Spec: 7B-Aa Global Pmi]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
