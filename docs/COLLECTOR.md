---
title: "Collector reference — sources, schema, and known data-quality issues"
tags:
  - doc/reference
  - topic/pipeline
  - status/active
date: 2026-08-29
---

# The Collector — reference

Everything in this document is a measurement, not a claim. Each section ends
with the command that reproduces it. All figures are from the reference
database `.tirra_pipeline/pipeline.db` as of **2026-08-29**; your numbers will
differ once you collect your own data.

> Read [the honesty section of the README](../README.md#read-this-first-what-does-not-work)
> first. The collector works. The modelling layers above it do not.

---

## 1. Schema

One table holds everything. There is no per-source schema to learn.

```sql
CREATE TABLE entity_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id        TEXT NOT NULL REFERENCES entities(entity_id),
    source_tool      TEXT NOT NULL,   -- which collector wrote this
    observed_at      REAL NOT NULL,   -- source-reported event time (unreliable, see §5)
    ingested_at      REAL NOT NULL,   -- wall-clock time we wrote it (reliable)
    observation_type TEXT NOT NULL,   -- e.g. tvl_change, futures_positioning
    depth_level      INTEGER NOT NULL DEFAULT 1,
    value_json       TEXT NOT NULL,   -- the payload
    metadata_json    TEXT
);

CREATE TABLE entities (
    entity_id      TEXT PRIMARY KEY,  -- opaque hash
    entity_type    TEXT NOT NULL,     -- company, person, country, wallet, vessel, ...
    canonical_name TEXT NOT NULL,
    created_at     REAL NOT NULL,
    metadata_json  TEXT
);
```

`observed_at` and `ingested_at` mean different things and you want different
ones for different questions. Use `ingested_at` to ask "is this collector
alive"; use `observed_at` to ask "how recent is the underlying fact".

Entity types present: `topic` (1,517), `wallet` (1,384), `person` (1,156),
`company` (1,066), `vessel` (502), `country` (279), `instrument` (93),
`organization` (60), `protocol` (54), `cftc_contract` (40), `domain` (20),
`maritime_area` (1).

---

## 2. The 34 collectors

| collector | rows | entities | earliest obs | latest obs | obs age (d) | last ingest (d) | credential |
|---|---:|---:|---|---|---:|---:|---|
| `defi_flows` | 162,251 | 52 | 2021-05-12 | 2026-08-27 | 2 | 2 | none |
| `gdelt` | 92,211 | 200 | 1920-01-01 | 2026-08-27 | 2 | 2 | none |
| `instrument_universe` | 81,268 | 89 | 2023-04-18 | 2026-08-26 | 3 | 2 | none |
| `polymarket` | 8,730 | 1,493 | 2024-07-24 | 2026-08-27 | 2 | 2 | none |
| `cftc` | 5,488 | 40 | 2015-01-06 | 2026-08-18 | 12 | 2 | none |
| `cftc_derived` | 5,080 | 40 | 2015-01-06 | 2026-04-14 | 138 | 118 | none |
| `dividend_data` | 5,020 | 21 | 2000-08-24 | 2026-08-21 | 8 | 2 | none |
| `whale_alert` | 4,612 | 1,384 | 2026-04-21 | 2026-08-27 | 2 | 2 | none |
| `form144` | 3,872 | 950 | 2026-04-14 | 2026-08-26 | 3 | 2 | none |
| `ais_vessel` | 1,634 | 503 | 2026-03-11 | 2026-08-27 | 2 | 2 | none |
| `sovereign_debt` | 1,448 | 13 | 2023-07-03 | 2026-08-27 | 2 | 2 | none |
| `gov_contracts` | 1,060 | 105 | 1978-09-14 | 2030-01-30 | -1250 | 2 | none |
| `insider_filings` | 901 | 178 | 2023-06-06 | 2026-08-25 | 4 | 2 | none |
| `energy_supply` | 640 | 1 | 2025-06-06 | 2026-08-21 | 9 | 2 | EIA (DEMO_KEY fallback) |
| `drug_regulatory` | 333 | 141 | 1972-03-24 | 2026-08-27 | 2 | 2 | none |
| `regulatory_gazette` | 242 | 44 | 2026-04-24 | 2026-08-27 | 2 | 2 | none |
| `dns_monitor` | 160 | 20 | 2026-08-25 | 2026-08-27 | 2 | 2 | none |
| `bankruptcy_court` | 100 | 42 | 2026-08-25 | 2026-08-27 | 2 | 2 | none |
| `global_pmi` | 96 | 8 | 2026-04-21 | 2026-08-27 | 2 | 2 | none |
| `political_risk` | 81 | 21 | 2026-04-24 | 2026-08-27 | 2 | 2 | FEC (DEMO_KEY fallback) |
| `options_chain` | 80 | 8 | 2026-06-04 | 2026-08-27 | 2 | 2 | none |
| `finra_short_volume` | 80 | 20 | 2026-08-27 | 2026-08-27 | 2 | 2 | none |
| `academic_preprints` | 75 | 9 | 2026-04-24 | 2026-08-27 | 2 | 2 | none |
| `sanctions_monitor` | 55 | 11 | 2025-06-15 | 2026-07-21 | 39 | 2 | none |
| `wikipedia_pageviews` | 43 | 9 | 2026-04-20 | 2026-08-26 | 4 | 2 | none |
| `creditor_filings` | 20 | 16 | 2026-07-29 | 2026-08-25 | 5 | 4 | Companies House (required) |
| `cert_transparency` | 19 | 2 | 2026-07-31 | 2026-08-17 | 12 | 2 | none |
| `transport_throughput` | 15 | 3 | 2026-08-25 | 2026-08-27 | 2 | 2 | none |
| `supply_chain_monitor` | 12 | 6 | 2026-08-25 | 2026-08-27 | 2 | 2 | none |
| `central_bank_balance` | 9 | 3 | 2026-04-21 | 2026-08-25 | 4 | 4 | FRED (required) |
| `disease_surveillance` | 8 | 1 | 2026-08-25 | 2026-08-27 | 2 | 2 | none |
| `capital_flows` | 8 | 5 | 2026-04-21 | 2026-08-25 | 4 | 4 | FRED (required) |
| `comtrade` | 5 | 1 | 2026-08-26 | 2026-08-27 | 2 | 2 | UN Comtrade (optional) |
| `consumer_sentiment` | 1 | 1 | 2026-08-25 | 2026-08-25 | 4 | 4 | FRED (required) |
| **34 collectors** | **375,657** | **6,172 distinct** | | | | | **27 keyless** |

Reproduce:

```bash
python - <<'PY'
import sqlite3, datetime as dt
c = sqlite3.connect("file:.tirra_pipeline/pipeline.db?mode=ro", uri=True)
now = dt.datetime.now(dt.timezone.utc).timestamp()
q = """SELECT source_tool, COUNT(*), COUNT(DISTINCT entity_id),
              MIN(observed_at), MAX(observed_at), MAX(ingested_at)
       FROM entity_observations GROUP BY 1 ORDER BY 2 DESC"""
for s, n, e, mn, mx, ing in c.execute(q):
    print(f"{s:24}{n:8d}{e:7d}  obs_age={(now-mx)/86400:7.1f}d  ingest_age={(now-ing)/86400:7.1f}d")
PY
```

### Credentials

27 of 34 collectors need no credential at all.

| Collector | Key | Behaviour without it | Register |
|---|---|---|---|
| `capital_flows` | `TIRRA_FRED_API_KEY` | skips cleanly | [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `central_bank_balance` | `TIRRA_FRED_API_KEY` | skips cleanly | FRED |
| `consumer_sentiment` | `TIRRA_FRED_API_KEY` | skips cleanly | FRED |
| `creditor_filings` | `TIRRA_COMPANIES_HOUSE_KEY` | skips cleanly | UK Companies House |
| `energy_supply` | `TIRRA_EIA_API_KEY` | falls back to `DEMO_KEY`, heavily rate-limited | [EIA](https://www.eia.gov/opendata/register.php) |
| `political_risk` | `TIRRA_FEC_API_KEY` | falls back to `DEMO_KEY`, heavily rate-limited | [FEC](https://api.open.fec.gov/developers/) |
| `comtrade` | `TIRRA_UN_COMTRADE_KEY` | works without; key raises limits | UN Comtrade |

Wired but not currently producing rows, and key-gated:
`satellite_activity` and `nightlight_activity` need a free
[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/) key
(note they read *different* env vars — `TIRRA_NASA_FIRMS_KEY` and
`FIRMS_API_KEY` respectively; set both to the same value).
`electricity_monitor` and `interconnection_queue` need the EIA key.

Copy `.env.example` to `.env` and fill in what you want. Never commit `.env` —
it is gitignored, keep it that way.

---

## 3. Wired vs. producing

`agent/pipeline/dags/daily_collection.py` wires **50 collector nodes**. **34**
have produced rows into the reference database. The gap is collectors that are
key-gated, that have been failing, or that have never been run to success.
`agent/tools/` contains 65 modules in total, but that count includes
non-collectors (`web_search`, `shell_runner`, `code_executor`, `file_manager`,
`backtest`, …) and is not a collector count. **34 is the honest number.**

The DAG's own run history is mostly red:

| DAG | completed | failed |
|---|---:|---:|
| `daily_collection` | 1 | 11 |

The DAG marks a run failed when *any* node fails, and with 50 nodes against
live third-party APIs at least one nearly always does. Individual collectors
still write their rows on a "failed" run — which is why 34 sources have data
despite 11 of 12 runs being marked failed. Do not read the run status as "no
data was collected", and do not read it as "everything is fine" either.

---

## 4. Running collectors

### One collector

```python
from agent.data.cache import DataCache
from agent.pipeline.store import PipelineStore
from agent.tools.defi_flows import DefiFlowsTool

store = PipelineStore(db_path="demo.db")
tool = DefiFlowsTool(cache=DataCache(), pipeline_store=store)
result = tool.execute(mode="tvl", limit=20)
print(result.success, result.output)
```

Every collector follows the same contract: construct with
`(cache=..., pipeline_store=...)`, call `.execute(**params)`, get back a
`ToolResult` with `.success`, `.output` and `.data`. Parameters differ per
collector — a collector called with no arguments will usually tell you what it
wanted:

```
success=False  Invalid mode ''. Must be one of: ['chain', 'dex_volume', 'history', 'stablecoins', 'tvl']
```

This is the "honest failure reporting" property: a collector that cannot do its
job returns `success=False` and a reason. It does not write a zero row and
report success. The per-node parameters used in production are in
`agent/pipeline/dags/daily_collection.py`.

### The whole DAG

```bash
python scripts/run_collection.py --db-path .tirra_pipeline/pipeline.db --workers 4
```

Slow — 50 nodes against live APIs. Intended to be run once a day, not
interactively. Expect it to report failure; see §3.

---

## 5. Data-quality issues you will hit

### 5.1 Observations are not deduplicated

`PipelineStore.store_entity_observation` is a plain `INSERT`:

```python
conn.execute(
    "INSERT INTO entity_observations "
    "(entity_id, source_tool, observed_at, ingested_at, "
    " observation_type, depth_level, value_json, metadata_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ...)
```

There is **no unique constraint**. The only index on the natural key,
`idx_entity_obs_lookup ON entity_observations(entity_id, source_tool,
observed_at)`, is a lookup index and is *not* `UNIQUE`. Re-running a collector
re-appends its rows.

In the reference database:

| measure | rows | share |
|---|---:|---:|
| total observations | 375,657 | 100% |
| distinct on `(entity_id, source_tool, observed_at, observation_type)` | 177,290 | 47.2% |
| **duplicates on that key** | **198,367** | **52.8%** |
| distinct including identical `value_json` | 254,060 | 67.6% |
| **byte-identical duplicate rows** | **121,597** | **32.4%** |

Worst offenders by duplicate share: `energy_supply` 90.0%, `dividend_data`
89.6%, `gdelt` 81.1%, `gov_contracts` 73.0%, `insider_filings` 72.1%,
`form144` 69.1%, `defi_flows` 66.0%, `whale_alert` 64.7%.

**Always deduplicate at read time.** For example:

```sql
SELECT DISTINCT entity_id, observed_at, observation_type, value_json
FROM entity_observations
WHERE source_tool = 'defi_flows';
```

Reproduce:

```bash
python - <<'PY'
import sqlite3
c = sqlite3.connect("file:.tirra_pipeline/pipeline.db?mode=ro", uri=True)
tot  = c.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
key  = c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT entity_id,source_tool,observed_at,observation_type FROM entity_observations)").fetchone()[0]
full = c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT entity_id,source_tool,observed_at,observation_type,value_json FROM entity_observations)").fetchone()[0]
print(f"total {tot}  distinct-on-key {key} ({100*(tot-key)/tot:.1f}% dup)  byte-identical dup {tot-full} ({100*(tot-full)/tot:.1f}%)")
PY
```

### 5.2 `observed_at` is source-reported and sometimes nonsense

It is whatever the upstream source said, parsed as best we could. Known bad
ranges in the reference database:

- `gov_contracts` has `observed_at` up to **2030-01-30** — contract *end* dates
  used as observation time.
- `gdelt` has rows dated **1920-01-01** — unparseable dates falling back to an
  epoch-ish default.
- `drug_regulatory` reaches back to **1972-03-24**, which is genuine (old
  approvals), not a bug.

Filter to a plausible window before doing anything time-sensitive, and prefer
`ingested_at` when you mean "recently collected".

### 5.3 Sparse tails

Breadth is wider than depth. Seven collectors have produced fewer than 20 rows
(`transport_throughput` 15, `supply_chain_monitor` 12, `central_bank_balance`
9, `disease_surveillance` 8, `capital_flows` 8, `comtrade` 5,
`consumer_sentiment` 1). Two collectors — `defi_flows` and `gdelt` — are 68% of
all rows. Any analysis pooling across sources is dominated by those two unless
you weight deliberately. This is the mechanism behind LESSONS `F-07`/`F-10`
(GDELT flooding the graph).

---

## 6. Verifying the join claim

The project's terminal finding was that **the entity graph does not join across
sources**, which kills cross-source synthesis as a premise. Here is the exact
measurement so you can check it rather than trust it.

Of `C(34,2) = 561` possible source pairs, **43 share at least one entity**. But
40 of those 43 are *same-namespace* joins — a source joined to another view of
the same identifier system, which is not cross-domain synthesis:

- **ISO country codes** (36 pairs): `capital_flows`, `central_bank_balance`,
  `comtrade`, `consumer_sentiment`, `disease_surveillance`, `energy_supply`,
  `global_pmi`, `sovereign_debt`, `transport_throughput` all pairwise share a
  handful of country entities. Joining "US GDP" to "US energy supply" on the
  string `US` is not a discovered link.
- **Equity tickers** (3 pairs): `instrument_universe` ↔ `dividend_data` ↔
  `options_chain`.
- **CFTC contract codes** (1 pair): `cftc` ↔ `cftc_derived`, 40 entities — a
  source joined to its own derivative.

**Genuine cross-domain joins: 3 pairs, 4 shared entities.**

| pair | shared | entity type |
|---|---:|---|
| `cert_transparency` ↔ `dns_monitor` | 2 | domain |
| `form144` ↔ `insider_filings` | 1 | person |
| `drug_regulatory` ↔ `gov_contracts` | 1 | company |

Three joins across 561 possible pairs, on 375,657 observations. Every
identifier namespace is an island: `btc_address` appears only in `whale_alert`
(1,384 entities), `mmsi` only in `ais_vessel` (502), `protocol_name` only in
`defi_flows` (52), `sanctions_un` only in `sanctions_monitor` (11). Several
sources carry no resolvable cross-source identifier at all.

Reproduce:

```bash
python - <<'PY'
import sqlite3, itertools
c = sqlite3.connect("file:.tirra_pipeline/pipeline.db?mode=ro", uri=True)
etype = dict(c.execute("SELECT entity_id, entity_type FROM entities"))
tools = [r[0] for r in c.execute("SELECT DISTINCT source_tool FROM entity_observations ORDER BY 1")]
ents = {t: {r[0] for r in c.execute(
            "SELECT DISTINCT entity_id FROM entity_observations WHERE source_tool=?", (t,))}
        for t in tools}
pairs = list(itertools.combinations(tools, 2))
shared = [(a, b, ents[a] & ents[b]) for a, b in pairs if ents[a] & ents[b]]
cross = [(a, b, s) for a, b, s in shared
         if len({etype.get(e) for e in s}) != 1
         or next(iter({etype.get(e) for e in s})) not in ("country", "instrument", "cftc_contract")]
print(f"possible pairs      : {len(pairs)}")
print(f"share >=1 entity    : {len(shared)}")
print(f"same-namespace only : {len(shared) - len(cross)}")
print(f"genuine cross-domain: {len(cross)}")
for a, b, s in cross:
    print(f"   {a} <-> {b}: {len(s)} ({sorted({etype.get(e) for e in s})})")
PY
```

> **Note on a previously circulated figure.** Earlier internal write-ups state
> "2 joinable pairs, 3 shared entities". That count was taken before
> `drug_regulatory` ↔ `gov_contracts` acquired its single shared company, and
> the database keeps growing. The measurement above is the current one; re-run
> it rather than quoting either number. The conclusion is unchanged and is not
> sensitive to which of these you use: three joins out of 561 is
> indistinguishable from zero for the purpose of cross-source synthesis.

---

## 7. What the layers above the collector do

Nothing you should rely on. Recorded here so nobody has to rediscover it.

**`convergence_clusters`** — 42 rows. `correlated_surprise_score` spans
0.99765 to 0.99986: a 0.0022-wide band presented as a discriminative ranking.
`contributing_tools_json` is `[]` on **all 42 rows**, so no cluster can name a
single tool that contributed to it.

**`entity_alerts`** — 4,852 rows. `composite_surprise` ranges
6.42×10⁷ to 3.34×10⁸. A surprise score should be O(1)–O(10). Component
correlations against the composite:

| component | range | corr with composite |
|---|---|---:|
| `neighborhood_surprise` | 0 – 3.28×10⁸ | **0.88** |
| `temporal_surprise` | 7.02×10⁷ – **1.788×10⁹** | 0.39 |
| `value_surprise` | 1.26 – 8.50×10⁵ | −0.43 |
| `hawkes_intensity` | 0.1 – 160.1 | 0.08 |
| `obs_type_surprise` | 1.85 – 23.03 | −0.06 |

`temporal_surprise` reaching 1.788×10⁹ is a raw Unix timestamp. The two
timestamp-scale components swamp the two that are genuinely O(1)–O(100), so the
"alert ranking" is substantially a sort by clock time.

Reproduce:

```bash
python - <<'PY'
import sqlite3, statistics
c = sqlite3.connect("file:.tirra_pipeline/pipeline.db?mode=ro", uri=True)
print("convergence:", c.execute(
    "SELECT MIN(correlated_surprise_score), MAX(correlated_surprise_score), COUNT(*) "
    "FROM convergence_clusters").fetchone())
print("empty contributing_tools_json:", c.execute(
    "SELECT COUNT(*) FROM convergence_clusters "
    "WHERE contributing_tools_json IN ('','[]') OR contributing_tools_json IS NULL").fetchone()[0])
print("composite_surprise range:", c.execute(
    "SELECT MIN(composite_surprise), MAX(composite_surprise) FROM entity_alerts").fetchone())
print("max temporal_surprise:", c.execute(
    "SELECT MAX(temporal_surprise) FROM entity_alerts").fetchone()[0])
PY
```

The through-line, worth naming because it is the transferable lesson: this
codebase evolved from *healthy-looking and empty* to *healthy-looking and full
of constants*. Every table had rows. Every job reported success. None of the
numbers meant anything. Row counts and green runs are not evidence — see
`LESSONS.md` F-12 and F-13.

---

## 8. Adding a collector

1. Subclass `Tool` in `agent/tools/<name>.py` with `name`, `description`,
   `parameters` (JSON Schema) and `execute(**kwargs) -> ToolResult`.
2. Accept `pipeline_store` in `__init__` and call
   `store.register_entity(...)` then `store.store_entity_observation(...)`.
3. On failure return `ToolResult(success=False, output="<why>")`. **Never**
   return `success=True` with zero rows — that is the failure mode this
   codebase spent a year learning about.
4. Set `observed_at` from the *source's* timestamp where one exists. If the
   source gives you nothing, use collection time and say so in `metadata_json`.
5. Free and keyless sources only, or key-gated with a clean skip.
6. Register it in `agent/cli.py::build_tool_registry` and add a node to
   `agent/pipeline/dags/daily_collection.py`.

See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## 9. Verification record

The quickstart in the README was executed on **2026-08-29** in a clean
virtualenv (Python 3.12.14, macOS) created fresh, with only
`httpx jsonschema defusedxml numpy yfinance` installed — no `chromadb`, no
`torch`, no `openai`. Against live upstream APIs it produced:

```
defi_flows   success=True  Top 20 DeFi protocols by TVL. Total TVL: $586,124,297,565
cftc         success=True  CFTC COT — 34 contracts (of 278 total), report date 2026-08-25
gdelt        success=True  GDELT — 20 events (from 2840 total, 4 batches, 1h lookback)
```

```
  cftc           rows=   34 entities=   34
  defi_flows     rows=   20 entities=   20
  gdelt          rows=   16 entities=    5
```

70 observations, 59 entities, from three keyless sources. The minimal
dependency set was established by importing all 64 tool modules in that
clean environment and recording the failures: everything imports on
`httpx + jsonschema + defusedxml`, except `backtest`, `dividend_data`,
`instrument_universe`, `liquidity_regime`, `m15_universe`, `options_chain`
(need `numpy`) and `market_data` (needs `yfinance`).
