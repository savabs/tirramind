---
title: "Spec: Historical Backfill Runner (Phase 47)"
tags:
  - doc/spec
  - phase/47
  - topic/backfill
  - topic/training-data
  - layer/surveillance
  - status/active
---

# Spec: Historical Backfill Runner (Phase 47)

## Goal

Populate PipelineStore with 5 years of historical observations by calling all
51 tools with maximum date ranges. Unlocks Phase 40 immediately — no live
accumulation wait. The GNN needs to have seen COVID shock (2020), supply chain
crisis (2021-22), rate cycle (2022-23), and multiple sanctions waves to
generalise across regimes.

**Mandatory exit gate:** `python scripts/density_audit.py` must exit 0.
Phase 47 is not done until it does.

---

## Design Principles

1. **No new infrastructure.** Every tool already writes to `entity_observations`
   via `_persist_entities`. The backfill runner just calls existing `execute()`.
2. **Idempotent by checkpoint.** A JSON checkpoint records which tool calls
   completed. Re-running skips completed entries. Prevents duplicate writes.
3. **Surgical `_backfill` bypass for 6 capped tools.** Six tools enforce a
   Python-level `max(1, min(days_back, N))` clamp. These are agent-facing
   guards, not API constraints — USGS, EDGAR, CDC, OONI all support arbitrary
   historical date ranges natively. Adding `_backfill: bool = False` to each
   tool's `execute()` signature removes the clamp when True. 2-line change per
   tool; existing behaviour and tests unchanged because default is False.
4. **Timestamp correctness audit before any writes.** Verify each tool writes
   `observed_at` from the actual data record's date, not `time.time()`. The
   GNN's temporal ordering breaks if all backfilled observations carry today's
   timestamp. Fix before backfilling.
5. **Rate-limit politeness.** 1.5s sleep between tool invocations. No API has
   a limit that threatens a sequential 51-tool backfill at this cadence.

---

## Files Affected

### Create
| File | Purpose |
|---|---|
| `scripts/backfill.py` | Main orchestrated runner: tool dispatch, chunking, checkpoint, rate-limiting, progress |
| `scripts/density_audit.py` | Post-backfill density report: obs/entity type, temporal span, sparse flags |

### Modify (surgical — `_backfill` bypass only)
| File | Clamp removed | Change size |
|---|---|---|
| `agent/tools/earthquake_proximity.py` | `min(days_back, 30)` | +2 LOC |
| `agent/tools/disease_surveillance.py` | `min(int(days_back), 180)` | +2 LOC |
| `agent/tools/insider_filings.py` | `min(days_back, 90)` | +2 LOC |
| `agent/tools/form144.py` | `min(days_back, 60)` | +2 LOC |
| `agent/tools/sanctions_monitor.py` | `min(days_back, 365)` | +2 LOC |
| `agent/tools/internet_infrastructure.py` | `min(_safe_int(days_back,30), 90)` | +2 LOC |

---

## Tool Classification

### Group A — Call once (uncapped or clamp bypassed via `_backfill=True`)

| Tool | Backfill kwargs | Notes |
|---|---|---|
| `academic_preprints` | `days_back=1825` | arXiv/bioRxiv — no cap |
| `ais_vessel` | `days_back=730` | OpenSky free: ~2yr historical max |
| `bankruptcy_court` | `days_back=1825` | PACER/EDGAR — no cap |
| `building_permits` | `days_back=1825` | Census.gov — no cap |
| `capital_flows` | `days_back=1825` | IMF BOP — no cap |
| `central_bank_balance` | `days_back=1825` | Fed/ECB/BOJ — no cap |
| `cftc` | loop: `mode="historical", year=Y` for Y in 2020–2026 | Year-based mode; 7 calls |
| `comtrade` | `days_back=1825` | UN Comtrade — no cap |
| `consumer_sentiment` | `days_back=1825` | BLS/Eurostat — no cap |
| `creditor_filings` | `days_back=1825` | PACER — no cap (Python clamp is advisory) |
| `disease_surveillance` | `days_back=1825, _backfill=True` | CDC NWSS historical; code cap=180 → bypass |
| `earthquake_proximity` | `days_back=1825, _backfill=True` | USGS time-unlimited; code cap=30 → bypass |
| `food_security` | `days_back=1825` | FAO/FEWS — no cap |
| `form144` | `days_back=1825, _backfill=True` | EDGAR EFTS date range; code cap=60 → bypass |
| `gdelt` | chunked — see Step 47.2 | GDELT is 15-min file-based; needs `start_datetime` param added first |
| `gov_contracts` | `days_back=1825` | USASpending.gov — no cap |
| `insider_filings` | `days_back=1825, _backfill=True` | EDGAR — code cap=90 → bypass |
| `job_postings` | `days_back=1095` | BLS JOLTS — ~3yr available |
| `labor_disruptions` | `days_back=1825` | BLS — no cap |
| `macro_data` | per-series loop (see Step 47.2) | FRED: 20yr; ECB; WorldBank |
| `market_data` | per-ticker `period="5y"` | Run via `backfill_instruments.py` (already exists) |
| `migration_flows` | `days_back=1825` | UN/UNHCR — no cap |
| `satellite_activity` | `days_back=1095` | Copernicus/NASA free: ~3yr |
| `supply_chain_monitor` | `days_back=1825` | BLS/FRED — no cap |
| `transport_throughput` | `days_back=1825` | BTS/Eurostat — no cap |

### Group B — Verify then attempt (probe with `days_back=30` first)

Run `python scripts/backfill.py --verify` before full backfill. This probes
each Group B tool; tools that return data proceed. Tools that return empty or
error are auto-downgraded to skip with a note in the checkpoint.

| Tool | Expected depth | Notes |
|---|---|---|
| `defi_flows` | 2yr on-chain | The Graph historical — verify endpoint |
| `drug_regulatory` | 5yr FDA | FDA Drugs@FDA date params |
| `electricity_monitor` | 2yr EIA | EIA/Entso-E — verify |
| `energy_supply` | 5yr EIA | Verify date range |
| `finra_short_volume` | 20d ONLY | FINRA files exist only ~20 trading days; call `days_back=20` once |
| `foia_requests` | 3yr MuckRock | Verify date range |
| `global_pmi` | 5yr Markit/ISM | Verify endpoint |
| `interconnection_queue` | 3yr FERC | Verify |
| `internet_infrastructure` | 2yr OONI | Code cap=90 → `_backfill=True`; OONI supports historical |
| `lobbying` | 5yr OpenSecrets | Verify |
| `patent_filings` | 5yr USPTO | Verify |
| `political_risk` | 5yr FEC | Verify |
| `polymarket` | 2yr Gamma API | Verify market history depth |
| `polymarket_whales` | 2yr Gamma API | Verify |
| `power_grid` | 2yr EIA/RTO | Verify date param |
| `regulatory_gazette` | 5yr Fed Register | Code cap=365 → `_backfill=True` |
| `sanctions_monitor` | recent additions | OFAC/UN point-in-time; `days_back=1825, _backfill=True` |
| `sovereign_debt` | 5yr IMF/OECD | Verify |
| `treasury_receipts` | 5yr Treasury.gov | Verify |
| `weather_alerts` | 5yr NOAA archive | NOAA historical archive — verify |
| `whale_alert` | 1–2yr | blockchain.com confirmed block history — verify depth |
| `wikipedia_pageviews` | 5yr Wikimedia | Code cap=365 per call; loop 5× |

### Group C — Skip (live-only or utility)

| Tool | Reason |
|---|---|
| `cert_transparency` | crt.sh searches current cert state only |
| `dns_monitor` | Live bulk-resolve only |
| `internet_outages` | RIPE/Cloudflare real-time only |
| `backtest` | Internal engine |
| `code_executor` | Utility |
| `file_manager` | Utility |
| `instrument_universe` | Reference data — use `backfill_instruments.py` |
| `liquidity_regime` | Computed signal — runs after market_data backfill |
| `pipeline_query` | Internal query |
| `shell_runner` | Utility |
| `web_browse` | Live only |
| `web_search` | Live only |

---

## Implementation Steps

### Step 47.1 — Timestamp audit + `_backfill` bypass in 6 tools

**Timestamp audit first (no code changes yet).** For each of the 6 capped tools,
find the line where `observed_at` is set inside `_persist_entities`. It must
come from the data record's date string, not from `time.time()`. If any tool
uses `time.time()` universally, fix that before adding the bypass — otherwise
5 years of backfilled observations all land at today's timestamp and the GNN's
temporal ordering is destroyed.

**Bypass pattern (identical for all 6 tools):**

```python
# Before (existing execute() body):
days_back = max(1, min(days_back, 30))   # example: earthquake_proximity

# After (new signature + body):
def execute(self, *, mode: str = "recent", days_back: int = 7,
            ..., _backfill: bool = False, **_: Any) -> ToolResult:
    if not _backfill:
        days_back = max(1, min(days_back, 30))   # agent safety guard still active
```

The leading underscore on `_backfill` marks it as a non-public parameter. It
will not appear in the tool's JSON schema, so the agent orchestrator never sees it.

Tools to modify:
- `earthquake_proximity.py` — clamp: `max(1, min(days_back, 30))`
- `disease_surveillance.py` — clamp: `max(1, min(int(days_back), 180))`
- `insider_filings.py`      — clamp: `max(1, min(days_back, 90))`
- `form144.py`              — clamp: `max(1, min(days_back, 60))`
- `sanctions_monitor.py`    — clamp: `max(1, min(days_back, 365))`
- `internet_infrastructure.py` — clamp: `max(1, min(_safe_int(days_back, 30), 90))`

**Tests for Step 47.1 (2 per tool, 12 total):**

```python
# Example — same pattern for all 6 tools
def test_backfill_bypass_removes_clamp():
    tool = EarthquakeProximityTool()
    with patch.object(tool, "_execute_recent") as mock:
        mock.return_value = ToolResult(success=True, output="ok")
        tool.execute(days_back=1825, _backfill=True)
        assert mock.call_args.kwargs["days_back"] == 1825   # not clamped

def test_clamp_still_applies_without_flag():
    tool = EarthquakeProximityTool()
    with patch.object(tool, "_execute_recent") as mock:
        mock.return_value = ToolResult(success=True, output="ok")
        tool.execute(days_back=1825)
        assert mock.call_args.kwargs["days_back"] == 30    # clamped to 30
```

---

### Step 47.2 — Implement `scripts/backfill.py`

**Module structure:**

```
scripts/backfill.py
  BACKFILL_PLAN: list[dict]    — all entries, single source of truth
  BackfillCheckpoint           — load/save JSON; flush after each entry
  _build_registry(db_path)     — PipelineStore + ToolRegistry
  _run_one(registry, entry, dry_run)
  _generate_calls(entry)       — expand chunk configs into individual calls
  _count_obs(store, label)     — obs delta before/after
  main()                       — argparse → loop → summary
```

**Key BACKFILL_PLAN entries:**

```python
MACRO_SERIES = [
    "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10Y2Y", "DTWEXBGS",
    "M2SL", "DGS10", "DGS2", "VIXCLS", "BAMLH0A0HYM2",
]

BACKFILL_PLAN = [
    # Group A
    {"label": "academic_preprints",   "tool": "academic_preprints",   "kwargs": {"days_back": 1825}},
    {"label": "insider_filings",      "tool": "insider_filings",      "kwargs": {"days_back": 1825, "_backfill": True}},
    {"label": "earthquake_proximity", "tool": "earthquake_proximity", "kwargs": {"days_back": 1825, "_backfill": True}},
    # ... (all Group A entries per classification table above)

    # CFTC year loop
    # {"label": f"cftc_{y}", "tool": "cftc", "kwargs": {"mode": "historical", "year": y}}
    # for y in range(2020, 2027)

    # macro_data per series
    # {"label": f"macro_{sid}", "tool": "macro_data",
    #  "kwargs": {"source": "fred", "series_id": sid, "start_date": START_5Y, "end_date": TODAY}}
    # for sid in MACRO_SERIES

    # GDELT chunked (168h window × 104 calls = 2yr; requires start_datetime param)
    {"label": "gdelt", "tool": "gdelt",
     "kwargs": {"mode": "events", "quad_class": "conflict"},
     "chunk": {"mode": "hours_back_loop", "window": 168, "count": 104}},

    # wikipedia chunked (365d window × 5 calls = 5yr)
    {"label": "wikipedia_pageviews", "tool": "wikipedia_pageviews",
     "kwargs": {},
     "chunk": {"mode": "days_back_loop", "window": 365, "count": 5}},

    # FINRA: accept 20d API limit
    {"label": "finra_short_volume",   "tool": "finra_short_volume", "kwargs": {"mode": "short_volume", "days_back": 20}},
    {"label": "finra_short_interest", "tool": "finra_short_volume", "kwargs": {"mode": "short_interest"}},

    # Group B entries (all with "group": "B")
    {"label": "defi_flows",         "tool": "defi_flows",    "kwargs": {"days_back": 1825}, "group": "B"},
    # ... etc

    # Group C: skip entries
    {"label": "cert_transparency",  "skip": True, "reason": "live-only: crt.sh current state only"},
    # ... etc
]
```

**Checkpoint file:** `.tirra_pipeline/backfill_checkpoint.json`

```json
{
  "started_at": "2026-04-24T10:00:00Z",
  "days_back": 1825,
  "completed": ["academic_preprints", "cftc_2020"],
  "failed":    {"earthquake_proximity": "ConnectionError: timeout"},
  "skipped":   ["cert_transparency"]
}
```

Flush to disk **immediately** after each label completes or fails. A crash
loses only the currently-running tool's writes.

**Chunk expansion:**

```python
def _generate_calls(entry: dict) -> list[dict]:
    chunk = entry.get("chunk")
    base = entry.get("kwargs", {})
    if chunk is None:
        return [base]
    if chunk["mode"] == "hours_back_loop":
        return [{**base, "hours_back": chunk["window"]} for _ in range(chunk["count"])]
    if chunk["mode"] == "days_back_loop":
        return [{**base, "days_back": chunk["window"]} for _ in range(chunk["count"])]
    return [base]
```

**Note on GDELT chunking:** The GDELT tool fetches the N most-recent hours.
Without a date-offset parameter, 104 calls with `hours_back=168` fetch the
same recent data 104 times. Before implementing GDELT chunking, add a
`start_datetime: str | None = None` param to the GDELT tool that downloads
batch files starting from a specific UTC timestamp. If deferred, mark the
GDELT label as `"skip": True, "reason": "deferred — needs start_datetime param"`.

**CLI:**

```
python scripts/backfill.py
  [--db-path PATH]      default: .tirra_pipeline/pipeline.db
  [--days-back N]       override default 1825
  [--dry-run]           print plan; no API requests
  [--tool LABEL]        run only this label
  [--verify]            probe each Group B tool with days_back=30
  [--no-retry]          skip tools in checkpoint.failed
  [--delay SECS]        sleep between tools (default: 1.5)
  [--skip-group-b]      Group A only
  [--group-b-only]      Group B only
```

**Progress output:**

```
Phase 47 Backfill — .tirra_pipeline/pipeline.db  (1825 days)
──────────────────────────────────────────────────────────────────────────
[  1/54] academic_preprints          ✓  4,821 obs  (+4,821)  12.4s
[  3/54] insider_filings   [BYPASS]  ✓ 31,204 obs  (+31,204) 22.7s
[  4/54] defi_flows        [GRP-B]   ✗  empty response — skipping
──────────────────────────────────────────────────────────────────────────
Done: 51 completed, 1 failed, 14 skipped
Total new observations: 1,247,830  |  Wall time: 3h 12m
```

**Error handling:**
- ALL exceptions caught per tool. Log; mark failed; continue.
- `sqlite3.OperationalError`: retry once after 5s; mark failed if still locked.
- HTTP 429: sleep 60s; retry once; mark failed.
- `success=False`: warning log; NOT marked failed.

---

### Step 47.3 — Implement `scripts/density_audit.py`

**This is the mandatory Phase 47 exit gate.** Exit code 0 = Phase 40 ready.

**Queries:**

```sql
-- Observations per entity_type
SELECT e.entity_type,
       COUNT(DISTINCT e.entity_id)  AS entity_count,
       COUNT(eo.id)                  AS obs_count,
       MIN(eo.observed_at)           AS earliest_obs,
       MAX(eo.observed_at)           AS latest_obs
FROM entity_observations eo
JOIN entities e ON eo.entity_id = e.entity_id
GROUP BY e.entity_type
ORDER BY obs_count DESC;

-- Observations per source_tool
SELECT source_tool, COUNT(*) AS obs_count
FROM entity_observations
GROUP BY source_tool
ORDER BY obs_count DESC;
```

**Sparse flag:** entity type is `SPARSE` if obs_count < `--min-obs` (default 100)
OR `(latest_obs - earliest_obs)` < `--min-days` × 86400 seconds (default 180d)
OR entity_count < `--min-entities` (default 5).

**Shannon entropy:**

```python
import math
def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log(c/total) for c in counts if c > 0)
```

**Output format:**

```
================================================================================
Phase 47 Density Audit — .tirra_pipeline/pipeline.db — 2026-04-25
================================================================================
Total observations: 1,247,830   Total entities: 14,821

By entity_type:
  TYPE              ENTITIES        OBS   OBS/ENT   SPAN(d)   STATUS
  company              4,821    847,203     175.7      1827    OK
  country                198    183,402     926.3      1825    OK
  instrument              90     89,441     993.8      1826    OK
  vessel               8,430     51,200       6.1       730    OK
  domain                  44        120       2.7        14    SPARSE (span<180d)
  wallet                  12         43       3.6        14    SPARSE (obs<100)

Entity-type Shannon entropy: 2.14 nats

By source_tool (any zero shown):
  insider_filings         241,403
  gdelt                   198,402
  ais_vessel                    0  WARNING: zero observations

================================================================================
SPARSE TYPES:
  domain — 120 obs, 44 entities, span=14d
    Recommendation: live-only source. Accept with justification or skip.
  wallet — 43 obs, 12 entities, span=14d
    Recommendation: extend whale_alert days_back or wire additional crypto tool.

VERDICT: FAIL (2 sparse types)
Document override in quant_training_ground.md before starting Phase 40.
================================================================================
```

**Exit codes:** `0` = all types pass. `1` = any sparse type.

**Justification override:** Sparse types that are structurally live-only (e.g.,
`domain`) may be documented as accepted overrides in `quant_training_ground.md`.
Phase 40 may proceed with that documentation. The audit continues to show FAIL.

**CLI:**

```
python scripts/density_audit.py [--db-path PATH] [--min-obs N] [--min-days N] [--min-entities N]
```

---

### Step 47.4 — Tests (33 total)

**12 bypass tests** (2 per capped tool, added to each tool's existing test file).
Pattern: bypass removes clamp + no-bypass still clamps. See Step 47.1 example.

**12 backfill runner tests (`tests/scripts/test_backfill.py`):**

```
test_checkpoint_load_empty            — missing file → empty checkpoint struct
test_checkpoint_load_existing         — JSON on disk → correct completed set
test_checkpoint_round_trip            — write → reload equals original
test_completed_tool_skipped           — label in completed → execute() not called
test_failed_tool_retried              — label in failed → execute() called
test_failed_tool_skipped_no_retry     — --no-retry → failed label also skipped
test_dry_run_no_api_calls             — dry_run=True → zero execute() calls
test_single_tool_flag                 — --tool macro_GDP → only that label runs
test_obs_delta_counted                — obs before/after correctly computed
test_exception_isolation              — one tool raises → run continues; label in failed
test_rate_limit_retry                 — HTTP 429 → sleep 60s, retry once
test_checkpoint_flushed_immediately   — file updated after each tool, not at end
```

**9 density audit tests (`tests/scripts/test_density_audit.py`):**

```
test_empty_db_verdict_fail            — no observations → FAIL (exit 1)
test_all_above_threshold_pass         — all types above thresholds → exit 0
test_obs_below_100_sparse             — one type < 100 obs → SPARSE, exit 1
test_span_below_180d_sparse           — short temporal span → SPARSE, exit 1
test_entity_count_below_5_sparse      — < 5 entities → SPARSE
test_entropy_computation              — known distribution → verify exact nats
test_source_tool_zero_warning         — tool with 0 obs in output
test_exit_code_zero_pass              — clean audit → sys.exit(0)
test_exit_code_one_fail               — sparse type → sys.exit(1)
```

All tests use in-memory SQLite (`:memory:`). No real API calls.

---

### Step 47.5 — Smoke test (before full run)

```bash
# Dry-run: confirm full plan, zero API calls
python scripts/backfill.py --dry-run

# Single-tool smoke tests (fast + reliable tools first)
python scripts/backfill.py --tool macro_FEDFUNDS
python scripts/backfill.py --tool cftc_2023
python scripts/backfill.py --tool earthquake_proximity

# Spot-check: obs timestamps should span years, not hours
python scripts/density_audit.py
```

---

### Step 47.6 — Full backfill (operation, not code)

```bash
# 1. Group A first (most reliable)
python scripts/backfill.py --skip-group-b --days-back 1825

# 2. Instrument price backfill (existing script)
python scripts/backfill_instruments.py --years 5

# 3. Verify Group B, then run
python scripts/backfill.py --verify
python scripts/backfill.py --group-b-only

# 4. Density audit
python scripts/density_audit.py

# 5. Extend sparse types if fixable
# python scripts/backfill.py --tool <sparse_label> --days-back 3650

# 6. When density audit exits 0 → Phase 40 can start
```

---

## Edge Cases

| Scenario | Handling |
|---|---|
| Tool raises `ConnectionError` | Caught; label marked failed; run continues |
| Tool returns `success=False` | Warning log; NOT marked failed (tool executed fine) |
| API returns only 90d when 1825d requested | Accept; density audit surfaces the temporal gap |
| DB locked | Retry once after 5s; mark failed if still locked |
| GDELT without `start_datetime` param | Skip GDELT chunk; mark deferred in checkpoint |
| Checkpoint file corrupted (bad JSON) | Reset to empty; restart from first un-completed label |
| Duplicate observations from partial re-run | Checkpoint prevents re-running same label |
| `_backfill=True` sent to non-modified tool | Absorbed by `**_: Any`; no error |
| `observed_at = time.time()` found in a tool | Timestamp audit in 47.1 catches this; fix before backfilling |
| Sparse type is structurally live-only | Document justification in task file; Phase 40 may proceed |

---

## Testing Plan

1. Run bypass tests per tool: `pytest tests/test_<tool>.py -k backfill` — 2 pass each
2. `pytest tests/scripts/test_backfill.py -v` — 12 pass
3. `pytest tests/scripts/test_density_audit.py -v` — 9 pass
4. Full regression: `pytest` — 9676+ pass, 0 fail

---

## Exit Conditions

Phase 47 is **complete** when ALL of the following are checked:

- [ ] Timestamp audit done: all 6 bypassed tools write `observed_at` from data date
- [ ] `_backfill` bypass in all 6 capped tools (12 bypass tests pass)
- [ ] `scripts/backfill.py` implemented (12 runner tests pass)
- [ ] `scripts/density_audit.py` implemented (9 audit tests pass)
- [ ] Full regression clean (9676+ pass, 0 fail)
- [ ] `python scripts/backfill.py` completes for Group A tools without crash
- [ ] `python scripts/density_audit.py` exits 0 OR sparse types explicitly justified in `quant_training_ground.md`
- [ ] Task file updated; checkpoint written to `docs/memory/`

---

## Related

- [[historical_backfill]] — research: tool classification, rate limits, API references
- [[quant_training_ground]] — task file; Phase 47 entry point
- [[living_system_online_gnn]] — Phase 46 (completed); EWC layer live
- [[phase40_real_data_model_refresh]] — immediate next after Phase 47 exits 0
- [[convergence_as_control]] — Phase 49b; parallelisable during Phase 47 window
- [[data_strategy_doctrine]] — governing doctrine; depth targets and exit conditions
