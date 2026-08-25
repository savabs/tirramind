---
title: "Research: Ghost Pattern Graph Audit — What We Have vs What We Need"
tags:
  - doc/research
  - phase/42
  - topic/world-model
  - topic/convergence
  - layer/surveillance
  - layer/feature-engineering
---

# Research: Ghost Pattern Graph Audit

**Date:** 2026-05-02
**Context:** Phase 41b revealed IC=-0.033. Before more GNN training, understand whether the
entity graph actually contains the cross-domain signals the vision requires.

---

## 1. What the Vision Requires

> "We are secret journalists. Our job is to find data that has patterns — things that lead to
> things that not many people can see."

The instrument universe is **89 macro assets**: energy futures, agricultural futures, metals,
FX pairs, country ETFs, sector ETFs, bond ETFs, equity index futures, crypto.

For this universe, the "ghost patterns" live in:

| Signal cluster | Ghost pattern | Leads | Lag |
|---|---|---|---|
| **CFTC positioning** | Managed money net flips long before price | commodity futures | 1–4 weeks |
| **AIS vessel routing** | Oil tanker destination anomaly before supply shock | WTI, Brent, NG | 2–8 weeks |
| **EIA petroleum stocks** | Inventory draw before futures backwardation | WTI, Brent, NG | 1–2 weeks |
| **Weather/NOAA drought** | Drought index before crop failure | Wheat, Corn, Soy, Coffee | 4–12 weeks |
| **Satellite fire detection** | Fire hotspot in growing region before crop shortfall | agricultural futures | 2–8 weeks |
| **USDA crop condition** | Poor crop rating before harvest downgrade | Wheat, Corn, Soy | 2–6 weeks |
| **GDELT geopolitical** | Conflict event in producer country before supply disruption | energy, metals, ag | 1–6 weeks |
| **Capital flows** | Cross-border flow reversal before FX regime break | FX pairs | 2–8 weeks |
| **Disease wastewater** | Pathogen surge before labor/retail disruption | sector ETFs | 2–4 weeks |
| **Central bank balance** | Balance sheet expansion before FX weakness | FX pairs | 4–12 weeks |
| **Polymarket positioning** | Prediction market odds shift before news | all | days–weeks |

None of these require company-level data. This is a **macro instrument universe**. The company
layer is largely irrelevant here. The signal path is:

```
Physical world (Layer 0) → Country/commodity events (Layer 1) → Futures/FX/index prices (Layer 2)
```

---

## 2. Current Graph State vs What's Needed

### 2.1 What exists (honest count)

| Source tool | Obs count | Entity type | Quality | Is it the ghost? |
|---|---|---|---|---|
| gdelt | 901,773 | country | Daily, global | Partially — country events exist, but coarse |
| instrument_universe | 69,424 | instrument | Daily prices | This is the TARGET, not the signal |
| polymarket | 5,356 | topic | Prediction odds | ✅ YES — forward-looking, rare signal |
| form144 | 493 | person/company | SEC pre-sale | Equity-focused, less relevant here |
| cftc | 300 | cftc_contract | Pro positioning | ✅ YES — but only 20 contracts, 300 obs |
| insider_filings | 237 | company/person | SEC Form 4 | Equity-focused, less relevant here |
| whale_alert | 122 | wallet | Crypto moves | ✅ Relevant for BTC/ETH |
| defi_flows | 60 | protocol | TVL/flows | ✅ Relevant for crypto |
| drug_regulatory | 50 | company | FDA approvals | Healthcare ETF signal |
| global_pmi | 32 | country | PMI index | ✅ Macro signal |
| sovereign_debt | 26 | country | Credit spreads | ✅ Country ETF signal |
| academic_preprints | 15 | company/org | Research papers | Weak |
| capital_flows | 6 | country | Cross-border flows | ✅ FX signal — nearly empty |
| central_bank_balance | 5 | country | Balance sheet | ✅ FX signal — nearly empty |
| political_risk | 1 | country | Political stability | ✅ Country ETF signal — nearly empty |

### 2.2 The critical gaps (tools built but near-zero data)

These tools ARE wired in `daily_collection.py` and DO have `_persist_entities`, but have
been run either never or only a handful of times:

| Tool | Expected obs/run | Why it matters | Status |
|---|---|---|---|
| `ais_vessel` | 500+ | Oil tanker routing → energy supply | Built, wired, 0 obs |
| `energy_supply` | 50+ | EIA weekly petroleum stocks | Built, wired, ~0 obs |
| `supply_chain_monitor` | 100+ | BLS PPI cross-sector | Built, wired, ~0 obs |
| `weather_alerts` | 50+ | NOAA drought → crops | Built, wired, ~0 obs |
| `food_security` | 30+ | FAO food production | Built, wired, ~0 obs |
| `satellite_activity` | 50+ | NASA fire hotspots | Built, wired, ~0 obs |
| `earthquake_proximity` | 20+ | Mine/port disruption | Built, wired, ~0 obs |
| `disease_surveillance` | 50+ | CDC wastewater | Built, wired, ~0 obs |
| `comtrade` | 50+ | Bilateral trade flows | Built, wired, ~0 obs |
| `capital_flows` | 10+ | Cross-border flows | Built, wired, 6 obs total |
| `central_bank_balance` | 5+ | Central bank balance sheet | Built, wired, 5 obs total |
| `electricity_monitor` | 100+ | Regional power demand | Built, wired, ~0 obs |

**Root cause: the pipeline scheduler is not running continuously.** The `daily_collection.py`
DAG is correctly built and wired. It runs when triggered manually or when the CLI is started.
But there is no always-on process — the machine is not running the scheduler 24/7.

### 2.3 CFTC — the biggest quick win

Current state: **20 contracts, 300 obs, only 5 `cftc_tracks` links to instruments**.

CFTC disaggregated data covers agriculture + petroleum + natural gas + metals + financials.
The weekly flat file at `https://www.cftc.gov/dea/newcot/f_disagg.txt` contains ALL contracts
in one CSV. Historical ZIPs go back to 2006 at `https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip`.

**The opportunity: backfill 3 years of CFTC history for all 89 instruments.**

Managed money net position is one of the most documented free alpha signals in commodities:
- Mou (2010): MM net predicts returns 1–4 weeks ahead in crude, grains, metals
- De Roon, Nijman, Veld (2000): hedger pressure in futures predicts excess returns
- Gorton & Rouwenhorst (2006): futures returns correlated with basis, hedging pressure

The tool already exists (`cftc.py`). Only 5 instruments have a `cftc_tracks` link. Need:
1. Map all 89 instruments to their CFTC contract codes
2. Backfill 3yr history (156 weekly observations per contract)
3. This gives ~89 × 156 ≈ 13,800 new observations immediately

### 2.4 The IC target problem

The Phase 40/41b exit condition is: **mean IC > 0.03, t-stat > 2.0**.

This is mathematically wrong for our fold count.

With 3yr data, MIN_TRAIN=252, TEST_SIZE=21, STEP_SIZE=21:
- N_folds ≈ (765 - 252) / 21 ≈ **24 folds**
- IC t-statistic formula: $t = \bar{IC} \cdot \sqrt{N} / \sigma_{IC}$
- For IC = 0.033, N = 24, $\sigma_{IC}$ ≈ 0.10: **t ≈ 0.16** — far below 2.0

To get t > 2.0 with N = 24 folds requires mean IC ≈ **0.04–0.06** (assuming $\sigma_{IC}$ = 0.10).

**Corrected target:** IC > 0.04 AND t-stat > 2.0 (or equivalently ICIR > 0.40)
where ICIR = mean_IC / std_IC across folds.

Alternatively: accept t > 1.5 as "directional signal" until we have 5yr of data (48 folds).

---

## 3. The Real Architecture Problem: Graph Sparsity

The GNN can only learn cross-entity patterns if cross-entity edges exist **with observations
on both sides**.

Current edge map (what has real data on both sides):

```
country --[event_involves]--> country          ✅ 9484 links, 901K obs per side
topic   --[topic_relates]--> instrument        ✅ 1693 links (polymarket topics)
wallet  --[trades]--> instrument               ⚠️  33 links, 122 obs (BTC/ETH only)
cftc_contract --[cftc_tracks]--> instrument    ❌  5 links, 300 obs (20 of 89 instruments)
instrument --[tracks_issuer]--> company        ❌  45 links, but company has 50 obs total
instrument --[located_in]--> country           ✅  66 links (instrument → producer country)
```

**The instrument→country edge is the most important live cross-entity edge.**
WTI→US, Brent→GB/EU, Wheat→US/Ukraine, Copper→Chile — these exist.
If country events (GDELT) flow through these edges, the GNN CAN learn physical→price.

**But GDELT is all at country granularity.** The question is: are the GDELT country events
informative about which specific commodity will move? Only if the events are coded to the
right commodity-producing countries AND the GNN can learn that routing.

This IS plausible — GDELT codes ~1M events/week by country. Ukraine conflict events should
propagate through Ukraine→country to Wheat/Corn futures via the `located_in` reverse edge.

The problem: there are no `located_in` links for agricultural commodities by producer country.
All 66 `located_in` links appear to be exchange country (e.g., CME = US). Not production country.

---

## 4. Priority Stack: What Will Actually Move the Needle

Ordered by: (signal quality × data availability × engineering cost)

### Priority 1: CFTC full backfill + complete instrument mapping ⭐⭐⭐⭐⭐
- **Why:** CFTC managed money net is the most documented free commodity alpha signal
- **Data:** Free, weekly, 3yr history available instantly
- **Cost:** Low — tool exists, just need CFTC code→instrument mapping + backfill call
- **Expected output:** ~13,800 obs, 89 `cftc_tracks` links
- **References:** CFTC disaggregated futures: `https://www.cftc.gov/dea/newcot/f_disagg.txt`

### Priority 2: Fix instrument→country links to include PRODUCER country ⭐⭐⭐⭐
- **Why:** This activates the GDELT→instrument signal path that already has 901K obs
- **Data:** Known mapping: Wheat→Ukraine/US/Russia, Copper→Chile/Peru, Oil→Saudi/US
- **Cost:** Very low — just add links, no new tool needed
- **Expected output:** 20–30 new instrument→country edges for commodities
- **Reference:** USDA FAS, World Bank commodity production by country

### Priority 3: EIA weekly petroleum stocks ⭐⭐⭐⭐
- **Why:** Inventory builds/draws directly cause oil price moves; free weekly EIA data
- **Data:** `https://www.eia.gov/opendata/` — free API key, petroleum weekly series
- **Cost:** Low — `energy_supply.py` tool exists
- **Expected output:** Weekly obs on energy instrument nodes directly
- **Key series:** STEO, PET (EIA-914 crude production, weekly stocks)

### Priority 4: USDA crop condition + WASDE ⭐⭐⭐
- **Why:** Crop condition ratings (poor/fair/good/excellent) lead ag futures by 2–6 weeks
- **Data:** USDA NASS API: `https://quickstats.nass.usda.gov/api/` (free, no auth beyond key)
- **Cost:** Medium — needs new tool
- **Expected output:** Weekly obs linking to Wheat/Corn/Soy/Cotton/Coffee instruments
- **Reference:** USDA NASS Quick Stats API documentation

### Priority 5: NOAA drought monitor ⭐⭐⭐
- **Why:** Palmer Drought Index leads agricultural commodity prices by 4–12 weeks
- **Data:** `https://droughtmonitor.unl.edu/DmData/DataDownload/USDM_all.zip` (free, no auth)
- **Cost:** Low — simple CSV download, map to country→instrument via producer links
- **Expected output:** Weekly drought severity obs on country nodes

### Priority 6: Run the pipeline continuously for 30 days ⭐⭐⭐⭐⭐
- **Why:** All the tools are built and wired. Disease surveillance, AIS, satellite, capital flows,
  electricity, food security — they ALL have zero obs because nobody is running the scheduler.
- **What to do:** Start the TirraMind CLI with `blocking=True` on a machine that stays up 24/7,
  OR schedule it in a free always-on environment (Kaggle scheduled notebooks, Railway.app free tier,
  Render.com background worker free tier).
- **Expected output:** After 30 days, ~50 tools × ~50 obs/day = ~75,000 new observations across
  all entity types — without writing a single new line of code.

---

## 5. What NOT to Do Next

| Temptation | Why to resist |
|---|---|
| More GNN training epochs | Return head has nothing cross-entity to learn yet |
| Better ListNet / loss weighting | Optimizing a signal that doesn't exist yet |
| More complex GNN architecture | Architecture is fine; data is the bottleneck |
| Company layer for macro instruments | Company layer is equity-focused; irrelevant for futures/FX/ETF universe |
| IC > 0.03 exit condition | Target is miscalibrated — fix the target, not the model |

---

## 6. Revised Exit Conditions for Phase 41b

Old (broken): IC > 0.03 AND t-stat > 2.0

New (correct):
- **Short-term (achievable with current data):** ICIR > 0.40 across ≥ 20 folds
  - where ICIR = mean_IC / std_IC
  - ICIR > 0.40 is "has real signal" threshold in quant literature (Lo 2002, Grinold & Kahn)
- **Medium-term (after CFTC backfill + producer-country links):** IC > 0.04 AND t > 1.5
- **Long-term (after 6mo continuous pipeline):** IC > 0.05 AND t > 2.0

---

## 7. The Real Moat Diagnostic

After CFTC backfill + producer-country links + 30 days of live pipeline, run this test:

**Does removing CFTC positioning from the GNN drop IC by >20%?**
- YES → CFTC net is a genuine GNN-learned signal → we have cross-entity alpha
- NO → GNN is using only price autocorrelation → need more cross-entity data

**Does removing GDELT (country events) from the GNN drop IC by >10%?**
- YES → geopolitical→price path is learned → we are actually seeing ghost patterns
- NO → more cross-entity edges needed → focus on producer-country links

---

## 8. Sources Verified

- CFTC disaggregated futures (weekly flat file): `https://www.cftc.gov/dea/newcot/f_disagg.txt` ✅
- CFTC historical ZIPs: `https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip` ✅
- EIA open data API: `https://www.eia.gov/opendata/` — free, requires key registration ✅
- USDA NASS Quick Stats API: `https://quickstats.nass.usda.gov/api/` — free ✅
- NOAA Drought Monitor: `https://droughtmonitor.unl.edu/DmData/DataDownload/` ✅
- FAO FAOSTAT: `https://www.fao.org/faostat/en/#data` — bulk downloads, free ✅
- OpenSky (aircraft): research/non-commercial only — note in apidoc explicitly asks LLMs not to use ✅ (noted)
- Finnish Digitraffic AIS (marine): `https://www.digitraffic.fi/en/marine/` — appears temporarily down ⚠️

## Related

- [[phase41b_gnn_signal_extraction]] — current task
- [[phase41b_gnn_signal_extraction_spec]] — current spec
- [[tirramind_structure]] — canonical metrics
