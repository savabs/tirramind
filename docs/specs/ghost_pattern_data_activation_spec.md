---
title: "Spec: Ghost Pattern Data Activation"
tags:
  - doc/spec
  - phase/42
  - topic/world-model
  - layer/surveillance
  - status/active
---

# Spec: Ghost Pattern Data Activation

**Research:** [[ghost_pattern_graph_audit]]
**Task:** [[phase42_ghost_pattern_activation]]

---

## Goal

Convert the current sparse entity graph (dominated by GDELT + prices) into a graph where
cross-entity cross-domain signal paths exist with sufficient observation density for the GNN
to learn "physical world → price" routing.

**Target state:**
- CFTC managed money net: all 89 instruments mapped, 3yr history backfilled
- Producer-country edges: commodity instruments linked to producer nations (not just exchange)
- Pipeline running continuously: all wired tools accumulating daily observations
- Corrected IC target: ICIR > 0.40 instead of broken "IC > 0.03, t > 2.0"

---

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/cftc.py` | Add historical backfill mode, expand contract→instrument mapping |
| `agent/pipeline/seed_links.py` | Add producer-country instrument→country link seeds |
| `scripts/backfill_cftc.py` | New script: download 3yr CFTC history into pipeline.db |
| `agent/quant/backtest/walk_forward.py` | Fix IC exit target: add ICIR metric |
| `[[ghost_pattern_graph_audit]]` | Already written ✅ |

---

## Implementation Steps

### Step 42.1 — Map all 89 instruments to CFTC contract codes

CFTC `f_disagg.txt` uses Market_and_Exchange_Names strings, not standard tickers.
Build a mapping table: our ticker → CFTC `Market_and_Exchange_Names` substring.

Verification: `python -c "import pandas as pd; df = pd.read_csv('..f_disagg.txt'); print(df.Market_and_Exchange_Names.unique()[:50])"`
Exit: mapping table covers all energy, metals, agriculture instruments (≥40 of 89)

### Step 42.2 — Add `backfill` mode to CFTC tool

Extend `agent/tools/cftc.py` with `mode='backfill'`:
- Download `fut_disagg_txt_{year}.zip` for last 3 years (2022, 2023, 2024)
- Parse, filter by mapped contracts, persist via `_persist_entities`
- Rate-limit: 1 request/second (polite crawl)

Verification: query `SELECT count(*) FROM entity_observations WHERE source_tool='cftc'` → >10,000
Exit: ≥13,000 new CFTC observations in pipeline.db

### Step 42.3 — Add producer-country edges for commodity instruments

Add to `agent/pipeline/seed_links.py` (or create it if missing):
Known production links for commodity futures in the 89-instrument universe:
```
CL → US, SA, RU, AE     (WTI crude: US, Saudi, Russia, UAE)
BZ → GB, NG             (Brent: North Sea, West Africa)
NG → US, RU, QA         (Natural Gas: US, Russia, Qatar)
GC → US, AU, CN, ZA     (Gold: US, Australia, China, S.Africa)
HG → CL, PE, CN         (Copper: Chile, Peru, China)
W  → US, UA, RU         (Wheat: US, Ukraine, Russia)
C  → US, BR, AR         (Corn: US, Brazil, Argentina)
S  → US, BR, AR         (Soybeans: US, Brazil, Argentina)
KC → BR, CO, VN         (Coffee: Brazil, Colombia, Vietnam)
CC → CI, GH             (Cocoa: Cote d'Ivoire, Ghana)
```
Persist via `entity_links` table with `link_type='produced_in'`, `confidence=0.95`

Verification: `SELECT count(*) FROM entity_links WHERE link_type='produced_in'` → >30 rows
Exit: all 20 commodity instruments have at least 1 `produced_in` link to a country entity

### Step 42.4 — Fix IC exit condition: add ICIR metric

In `agent/quant/backtest/walk_forward.py`, add:
```python
ic_series = [fold["spearman_ic"] for fold in fold_results]
ic_ir = np.mean(ic_series) / (np.std(ic_series) + 1e-8)
```
Report both IC mean, t-stat, AND ICIR in the backtest summary.
Update exit condition doc in task file.

Verification: `python -m pytest tests/test_walk_forward.py -v` passes
Exit: ICIR field appears in backtest JSON output

### Step 42.5 — Run pipeline for 7 days, verify accumulation

Start CLI or run `scripts/run_daily_collection.py` each day for 7 consecutive days.
After 7 days, run entity count audit:
```sql
SELECT source_tool, count(*) FROM entity_observations
WHERE ingested_at > datetime('now', '-7 days')
GROUP BY source_tool ORDER BY count(*) DESC
```
Exit: ≥5 tools each contribute >50 new observations in the 7-day window

### Step 42.6 — Retrain GNN on enriched graph

After steps 42.1–42.5:
1. Update Kaggle notebook: add `"--return-log-var-max", "0.0"` to cmd
2. Rebuild Kaggle zip (preserve `checkpoints/` subfolder, no `-j`)
3. Upload dataset, retrain epochs 21–40 from checkpoint 20
4. Run IC backtest with new ICIR metric

Exit: ICIR > 0.25 (directional signal exists), even if IC t-stat not yet > 2.0

---

## Edge Cases

- CFTC file may lag by 1 week — use latest available, store `report_week` in metadata_json
- Some instruments have no CFTC equivalent (crypto, country ETFs) — skip gracefully
- Producer-country links are approximate (mining company ≠ country, but country is the
  proxy for geopolitical disruption risk in the macro instrument universe)
- ICIR can be negative if signals are consistently wrong — that is ALSO information

---

## Testing Plan

- Step 42.1: Unit test CFTC code→ticker mapping, at least 40 entries
- Step 42.2: Test backfill mode downloads, parses, and persists without duplicates (idempotent)
- Step 42.3: Test seed_links inserts all commodity→country pairs, deduplicates on conflict
- Step 42.4: Test ICIR formula with known IC series (constant IC = ICIR = IC/0 = inf → cap at 10)
- Step 42.6: Smoke test retrain runs 1 epoch without error

---

## Related

- [[ghost_pattern_graph_audit]] — research backing this spec
- [[phase41b_gnn_signal_extraction]] — predecessor task
- [[tirramind_structure]] — canonical metrics
