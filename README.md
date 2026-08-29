# TirraMind Collector

**A free-data collector for alternative financial signals.** 34 working
collectors pull from free public APIs — 27 of them needing no credential at
all — covering CFTC positioning, SEC insider filings, DeFi TVL, GDELT events,
AIS vessel tracking, federal contracts, bankruptcy dockets, sanctions, drug
approvals, prediction markets and more, normalising everything into one SQLite
table you can query.

It is a **data collection layer, and only that.** The modelling layers built on
top of it do not work. That is stated plainly below, before anything else,
because it is the single most important thing a stranger needs to know.

---

## Read this first: what does not work

This repository was an attempt to build a predictive trading system on
alternative data. **The predictive thesis failed four independent tests.** The
collector survived. Everything else should be treated as non-functional.

**1. There is no measurable predictive edge.** A forward-return event study on
the CFTC positioning anomalies found **0 of 51 hypotheses surviving
Benjamini–Hochberg correction at α=0.05**. Best uncorrected p was 0.002
(`mm_net_pct_oi`, |z|≥2, 20-day horizon), which becomes p_adj = 0.102.
Method was deliberately hostile to a positive result: unconditional-return
null rather than zero, CFTC publication lag honoured (Tuesday as-of / Friday
release), circular block bootstrap CIs, and pseudo-replication quantified
(123 events across only 116 distinct weeks).
→ `docs/research/cftc_forward_return_event_study.md`, `scripts/cftc_event_study.py`

*Stated honestly:* the specification pooled two-sided |z|, which cancels a
signed effect by construction, and had roughly 10% power. The defensible claim
is **"no detectable effect at low power on this specification"** — not "no
effect exists". Re-running BH on the published p-values of Sanders, Irwin &
Merrin (2009) gives 0 of 30 survivors (min adjusted p 0.105 vs our 0.102), so
this null replicates the literature.

**2. The entity graph does not join across sources.** This was the finding that
ended the project. Cross-source synthesis was the entire premise, and it is not
possible with this data. Of 561 possible source pairs, 43 share at least one
entity — but 40 of those are *same-namespace* joins (ISO country codes, equity
tickers, CFTC contract codes) that join a source to another view of itself.
**Genuine cross-domain joins: 3 pairs, 4 shared entities** (`cert_transparency`↔
`dns_monitor`, 2 domains; `form144`↔`insider_filings`, 1 person;
`drug_regulatory`↔`gov_contracts`, 1 company). Every identifier namespace is an
island: `btc_address` appears only in `whale_alert`, `mmsi` only in
`ais_vessel`, `protocol_name` only in `defi_flows`. Several sources carry no
resolvable identifier at all. Verify it yourself with the query in
[docs/COLLECTOR.md](docs/COLLECTOR.md#6-verifying-the-join-claim).

**3. The model layers emit degenerate constants.** Treat
`agent/models/`, `agent/fusion/`, `agent/learning/` and `agent/adversarial/`
as non-functional. Measured on the shipped database:
- `convergence_clusters`: all 42 rows score between 0.9977 and 0.9999 — a
  0.002-wide band presented as a discriminative score — and
  `contributing_tools_json` is empty (`[]`) on all 42 rows.
- `entity_alerts.composite_surprise` ranges 6.4×10⁷ to 3.3×10⁸. It is
  dominated by `neighborhood_surprise` (r = 0.88), and `temporal_surprise`
  reaches 1.788×10⁹ — a raw Unix timestamp. The ranking is sorting by clock
  time wearing the costume of a surprise score.

**4. There is no data moat, and the original product was mispriced.** The CFTC
serves all 194 disaggregated fields free and keyless. `pip install openbb-cftc`
maps 355 fields. TradingView ships a free built-in COT indicator. A paid
subscription product built on this was retired; this repo is what remains and
is worth keeping.

**What this means for you:** use this repo as a *collector*. It is a decent
one. Do not use it as a signal generator, and do not trust any score any layer
above `agent/tools/` and `agent/pipeline/` produces without re-deriving it.

---

## What does work

- **34 collectors producing rows** into a single normalised schema.
- **Free public sources only.** No exchange redistribution fees and no vendor
  contracts — a genuine structural property, not a marketing line. 27 of 34
  need no credential whatsoever.
- **Freshness:** 33 of 34 collectors last ingested within 4 days. Only
  `cftc_derived` is stale (118 days).
- **Honest failure reporting.** A collector that cannot reach its source
  returns `success=False` with the reason rather than writing a silent zero.
- **A real engineering failure log.** `LESSONS.md` documents F-01 through F-13,
  each an actual production failure with symptom, root cause, fix, and a
  prevention rule. It is the most transferable artifact here.

---

## Quickstart (5 minutes)

The collector needs **five packages**. It does not need `chromadb`, `torch`,
`openai`, or any of the ML stack in `pyproject.toml`'s optional extras — those
belong to the layers that do not work.

```bash
git clone <repo-url> tirramind && cd tirramind

python3 -m venv .venv && source .venv/bin/activate
pip install httpx jsonschema defusedxml numpy yfinance
```

Collect from three live keyless sources:

```bash
PYTHONPATH=. python - <<'PY'
from agent.data.cache import DataCache
from agent.pipeline.store import PipelineStore
from agent.tools.cftc import CFTCTool
from agent.tools.defi_flows import DefiFlowsTool
from agent.tools.gdelt import GDELTTool

store = PipelineStore(db_path="demo.db")
cache = DataCache()

for name, tool, params in [
    ("defi_flows", DefiFlowsTool(cache=cache, pipeline_store=store), {"mode": "tvl", "limit": 20}),
    ("cftc",       CFTCTool(cache=cache, pipeline_store=store),      {"mode": "latest"}),
    ("gdelt",      GDELTTool(cache=cache, pipeline_store=store),     {"mode": "events", "limit": 20}),
]:
    r = tool.execute(**params)
    print(f"{name:12} success={r.success}  {str(r.output)[:80]}")
PY
```

Expected output (row counts vary with live data):

```
defi_flows   success=True  Top 20 DeFi protocols by TVL. Total TVL: $586,124,297,565
cftc         success=True  CFTC COT — 34 contracts (of 278 total)
gdelt        success=True  GDELT — 20 events (from 2840 total, 4 batches, 1h lookback)
```

Now query what landed:

```bash
python - <<'PY'
import sqlite3
c = sqlite3.connect("demo.db")
for r in c.execute("""
    SELECT source_tool, COUNT(*) rows, COUNT(DISTINCT entity_id) entities
    FROM entity_observations GROUP BY 1 ORDER BY 2 DESC
"""):
    print(f"  {r[0]:14} rows={r[1]:5d} entities={r[2]:5d}")
PY
```

```
  cftc           rows=   34 entities=   34
  defi_flows     rows=   20 entities=   20
  gdelt          rows=   16 entities=    5
```

> This quickstart was executed verbatim in a clean Python 3.12 virtualenv on
> 2026-08-29 against live upstream APIs, producing 70 observations across the
> three sources. See [docs/COLLECTOR.md](docs/COLLECTOR.md) for the full
> verification record.

---

## What it collects

34 collectors, 375,657 observation rows, 6,172 distinct entities in the
reference database as of 2026-08-29. Full table with per-collector row counts,
date ranges, freshness and credential requirements:
**[docs/COLLECTOR.md](docs/COLLECTOR.md)**.

By domain:

| Domain | Collectors |
|---|---|
| Positioning & markets | `cftc`, `cftc_derived`, `finra_short_volume`, `options_chain`, `instrument_universe`, `dividend_data` |
| Corporate & legal | `form144`, `insider_filings`, `bankruptcy_court`, `creditor_filings`, `gov_contracts` |
| Crypto & DeFi | `defi_flows`, `whale_alert` |
| Geopolitics & news | `gdelt`, `political_risk`, `sanctions_monitor`, `regulatory_gazette` |
| Physical & logistics | `ais_vessel`, `transport_throughput`, `supply_chain_monitor`, `energy_supply` |
| Macro | `sovereign_debt`, `global_pmi`, `capital_flows`, `central_bank_balance`, `comtrade`, `consumer_sentiment` |
| Science & health | `drug_regulatory`, `disease_surveillance`, `academic_preprints` |
| Internet infrastructure | `dns_monitor`, `cert_transparency` |
| Attention | `wikipedia_pageviews`, `polymarket` |

**Credentials.** 27 of 34 need none. Four require a free API key and skip
cleanly without one: `capital_flows`, `central_bank_balance`,
`consumer_sentiment` (all [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))
and `creditor_filings` (UK Companies House). Two fall back to a heavily
rate-limited `DEMO_KEY`: `energy_supply` ([EIA](https://www.eia.gov/opendata/register.php))
and `political_risk` (FEC). `comtrade` takes an optional key for higher limits.
Collectors wired but not currently producing rows — `satellite_activity` and
`nightlight_activity` — need a free
[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/) key.

---

## Known data-quality issues

These are real and you will hit them. They are listed here rather than
discovered by you later.

- **Observations are not deduplicated.** `store_entity_observation` is a plain
  `INSERT` with no unique constraint; the index on
  `(entity_id, source_tool, observed_at)` is *not* unique. Re-running a
  collector re-appends. In the reference database **52.8% of rows are
  duplicates** on `(entity_id, source_tool, observed_at, observation_type)`,
  and **32.4% are byte-identical** including `value_json`. **Always
  `SELECT DISTINCT` or `GROUP BY` at read time.**
- **`observed_at` is source-reported and not always sane.** `gov_contracts`
  carries timestamps up to 2030-01-30 (contract *end* dates used as observation
  time); `gdelt` has rows dated 1920-01-01. Filter to a plausible window.
- **Collector freshness and *source* freshness are different things.** 33 of 34
  collectors last ingested within 4 days, but a collector that ran successfully
  may still be returning old data because the upstream source has not published:
  `sanctions_monitor` ran 2 days ago and its newest datum is 39 days old;
  `cftc` ran 2 days ago against a report 12 days old (weekly cadence plus
  publication lag). `cftc_derived` is the one genuinely dead collector — last
  ingest 118 days ago. `docs/COLLECTOR.md` reports both ages per collector;
  check the one you actually care about.
- **Sparse tails.** Several collectors have produced fewer than 100 rows;
  `consumer_sentiment` has produced exactly one. Breadth here is wider than
  depth.
- **`daily_collection` mostly reports failure.** 11 failed runs to 1 completed
  in the shipped history. The DAG marks the whole run failed when any node
  fails, which is most runs — individual collectors still write their rows.

---

## Repository layout

Only the first two directories are trustworthy.

```
agent/tools/        # Layer 1 — the collectors. WORKS.
agent/pipeline/     # SQLite store, DAG scheduler, operators. WORKS.
agent/quant/        # Layer 2 — feature math. Partially exercised; unvalidated.
agent/models/       # Layer 3 — Bayesian/graph world model. DEGENERATE.
agent/fusion/       # Layer 4 — Kalman/particle fusion. DEGENERATE.
agent/learning/     # Layer 5 — RL policy, bandits. DEGENERATE.
agent/adversarial/  # Layer 6 — edge decay, robustness. DEGENERATE.
agent/reasoning/    # Layer 7 — LLM narration only. Never decides.

docs/research/      # Research notes, incl. the negative result
LESSONS.md          # F-01..F-13 production failure log
scripts/            # run_collection.py, cftc_event_study.py
tests/
```

---

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).

**Why permissive rather than copyleft.** The usual argument for copyleft is
protection: AGPL would stop a vendor wrapping this in a SaaS and giving nothing
back. That argument is weak *here specifically*, because finding #4 above
measured the moat and found none — every upstream source is free and keyless,
so a vendor who wanted to repackage this could rebuild it from the same public
APIs without touching our code, and copyleft would buy protection against a
threat that has no reason to exist while taxing the adoption that is now the
only remaining value. Apache-2.0 over MIT because it adds an explicit patent
grant and trademark clarity at no cost to adopters. **Recommendation:
Apache-2.0. The owner decides** — swapping `LICENSE` for AGPL-3.0 is a
one-file change if protection is valued over reach.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: new collectors are
welcome, claims about predictive performance are not — any such claim needs a
multiple-testing correction and a pre-registered specification, and will
otherwise be declined.
