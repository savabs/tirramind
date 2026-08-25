---
title: "Spec: Phase 7c — Convergence Detection Layer"
tags:
  - doc/spec
  - layer/feature-engineering
  - layer/surveillance
  - layer/world-model
  - phase/7c
  - phase/9
  - topic/convergence
---

# Spec: Phase 7c — Convergence Detection Layer

## Goal

Transform TirraMind's 60 independent data tools into an integrated intelligence system that detects when normally-uncorrelated signals begin moving together — the mathematical signature of a hidden cause propagating through observable reality. The system must:

1. Extract structured evidence from heterogeneous tool outputs into a uniform protocol
2. Align multi-frequency signals onto common temporal grids without future leakage
3. Compute rolling anomaly scores per signal stream
4. Score pairwise coincidences via multiple statistical methods
5. Detect multi-source convergence events via graph-based clique analysis
6. Match events against a causal chain template library
7. Control false discovery rate across thousands of simultaneous tests
8. Emit convergence signals to the pipeline store for downstream phases (8–12)

**Layer placement:** This sits at the Layer 2 (Feature Engineering) / Layer 3 (World Model) boundary. It reads Layer 1 (tool) outputs and produces Layer 2 signals. No LLM. No RL. Pure math and rules.

**Research:** [[convergence_detection|convergence_detection]]

---

## Files Affected

### Create

| File | Purpose | Layer |
|------|---------|-------|
| `agent/convergence/__init__.py` | Package exports | — |
| `agent/convergence/evidence.py` | Evidence dataclass + EvidenceBus collection | L1→L2 bridge |
| `agent/convergence/taxonomy.py` | Signal taxonomy categories + SignalMeta registry (frequency, direction, TTL) | L2 config |
| `agent/convergence/extractors.py` | Per-tool evidence extraction functions (60 thin extractors) | L1→L2 bridge |
| `agent/convergence/alignment.py` | Temporal alignment: multi-resolution grids, LOCF, staleness detection | L2 |
| `agent/convergence/atomic_signals.py` | Rolling z-score, empirical percentile, anomaly flagging, direction normalization | L2 |
| `agent/convergence/coincidence.py` | Pairwise coincidence scoring: rolling correlation deviation, joint exceedance, concordance | L2→L3 |
| `agent/convergence/graph.py` | Coincidence graph: networkx construction, edge pruning, clique/component detection | L3 |
| `agent/convergence/templates.py` | Causal chain template library + declarative pattern matcher | L3 |
| `agent/convergence/fdr.py` | FDR control: BH procedure, Fisher's combined test, persistence filter, cross-category check | L3 |
| `agent/convergence/detector.py` | Top-level ConvergenceDetector: orchestrates full pipeline per cycle | L3 |
| `agent/convergence/signals.py` | ConvergenceSignal dataclass + emission to pipeline store | L3→store |
| `agent/pipeline/dags/convergence_detection.py` | DAG definition: depends on daily_collection outputs, runs detector | pipeline |
| `tests/test_convergence_evidence.py` | Evidence dataclass + bus tests | — |
| `tests/test_convergence_taxonomy.py` | Taxonomy + signal registry tests | — |
| `tests/test_convergence_extractors.py` | Evidence extractor tests for all 60 tools (fixtures) | — |
| `tests/test_convergence_alignment.py` | Temporal alignment tests (multi-freq, LOCF, staleness, no future leak) | — |
| `tests/test_convergence_atomic.py` | Z-score, percentile, anomaly, direction normalization tests | — |
| `tests/test_convergence_coincidence.py` | Pairwise scoring tests (all 3 methods + p-values) | — |
| `tests/test_convergence_graph.py` | Graph construction, pruning, clique detection tests | — |
| `tests/test_convergence_templates.py` | Template library + matcher tests | — |
| `tests/test_convergence_fdr.py` | BH, Fisher's, persistence, cross-category tests | — |
| `tests/test_convergence_detector.py` | End-to-end detector integration tests (synthetic scenarios) | — |
| `tests/test_convergence_signals.py` | Signal emission + pipeline store integration tests | — |
| `tests/test_convergence_dag.py` | Convergence DAG wiring + executor integration | — |

### Modify

| File | Change |
|------|--------|
| `agent/pipeline/dags/__init__.py` | Register `convergence_detection` DAG |
| `agent/cli.py` | Register convergence DAGs in pipeline startup |
| `pyproject.toml` | Add `networkx>=3.0`, `ruptures>=1.1` dependencies (statsmodels already present) |

---

## Data Structures

### Evidence (core protocol)

```python
@dataclass(frozen=True)
class Evidence:
    source: str           # Tool name (e.g., "cftc", "weather_alerts")
    signal_id: str        # Unique ID (e.g., "cftc.crude_oil.mm_net_long")
    timestamp: float      # Unix epoch of observation
    value: float          # Numeric value (NaN for missing/categorical-encoded)
    direction: int        # +1 stress/expansion, -1 relief/contraction, 0 neutral
    confidence: float     # 0.0–1.0, source-quality weighted
    category: str         # Taxonomy bucket (see taxonomy.py)
    tags: tuple[str, ...] # Immutable metadata (country, sector, entity)
    ttl: int              # Seconds until stale
```

### SignalMeta (registry entry)

```python
@dataclass(frozen=True)
class SignalMeta:
    signal_id: str            # Matches Evidence.signal_id
    source: str               # Tool name
    category: str             # Taxonomy category
    frequency: str            # "intraday" | "daily" | "weekly" | "monthly" | "event"
    direction_semantics: str  # Human description: "higher = more stress"
    flip_sign: bool           # True if raw value convention is opposite ours
    default_ttl: int          # Seconds
    min_observations: int     # Minimum history needed for z-score (default 30)
```

### ConvergenceSignal (output)

```python
@dataclass
class ConvergenceSignal:
    signal_name: str           # "convergence.<event_type>.<iso_date>"
    computed_at: float         # Unix timestamp
    value: float               # ConvergenceScore ∈ [0, 1]
    event_type: str            # Template name or "unknown_pattern"
    signals_involved: list[str]  # Signal IDs in the clique
    categories_involved: list[str]
    cross_category_count: int
    p_value: float             # Fisher's combined p-value
    persistence_days: int      # Consecutive active periods
    template_match: float      # 0 = no match, 1 = perfect match
    direction: int             # +1 stress, -1 relief
    lead_signal: str           # Earliest-activating signal
    lag_signals: list[str]     # Later-activating signals
```

---

## Signal Taxonomy Categories (11)

| Category | Description | Example tools |
|----------|-------------|---------------|
| `physical_flow` | Movement of goods/vessels/energy | ais_vessel, transport_throughput, energy_supply |
| `physical_disruption` | Natural/infrastructure disruption | weather_alerts, earthquake, internet_infra |
| `financial_stress` | Credit/debt/default indicators | sovereign_debt, creditor_filings, bankruptcy, defi_flows |
| `monetary_policy` | Central bank actions + flows | central_bank_balance, rate_monitor, capital_flows |
| `regulatory_action` | Government/agency actions | sanctions, drug_regulatory, regulatory_gazette, foia |
| `behavioral_intent` | Forward-looking human actions | patent_filings, lobbying, job_postings, wikipedia_pageviews, cert_transparency |
| `positioning` | Financial positioning data | cftc, finra_short_volume, polymarket_whales, insider_filings |
| `macro_momentum` | Economic momentum indicators | global_pmi, consumer_sentiment, building_permits, treasury_receipts |
| `biological` | Health/bio surveillance | disease_surveillance, food_security |
| `geopolitical` | Political/conflict signals | political_risk, gdelt, migration_flows |
| `supply_chain` | Production/logistics bottlenecks | supply_chain_monitor, interconnection_queue, gov_contracts |

---

## Implementation Steps

### Sub-phase 7c-A: Evidence Protocol + Taxonomy + Extractors

This is the foundation layer. Every subsequent sub-phase reads Evidence objects. Nothing else works until this is solid.

#### Step 7c-A.1: Create package skeleton + Evidence dataclass

**File:** `agent/convergence/__init__.py`, `agent/convergence/evidence.py`

Create `agent/convergence/` package. Implement:

- `Evidence` frozen dataclass (fields per spec above)
- `EvidenceBus` — simple list-based collector:
  - `submit(evidence: Evidence) -> None` — validate and append
  - `flush() -> list[Evidence]` — return collected evidence and clear
  - `snapshot() -> list[Evidence]` — return copy without clearing
- Validation: `confidence` ∈ [0,1], `direction` ∈ {-1, 0, +1}, `category` must be in taxonomy, `signal_id` must be non-empty, `timestamp` > 0, `ttl` > 0
- `__init__.py` exports: `Evidence`, `EvidenceBus`

**Test:** `tests/test_convergence_evidence.py`
- Valid construction
- Invalid confidence/direction/category raise ValueError
- Bus submit/flush/snapshot semantics
- Frozen immutability

**Verification:** `pytest tests/test_convergence_evidence.py -v` — all pass.

---

#### Step 7c-A.2: Signal taxonomy + SignalMeta registry

**File:** `agent/convergence/taxonomy.py`

Implement:

- `CATEGORIES: frozenset[str]` — the 11 category strings
- `SignalMeta` frozen dataclass (fields per spec above)
- `SignalRegistry` class:
  - `register(meta: SignalMeta) -> None` — add to registry, error on duplicate signal_id
  - `get(signal_id: str) -> SignalMeta | None`
  - `by_source(source: str) -> list[SignalMeta]`
  - `by_category(category: str) -> list[SignalMeta]`
  - `all_ids() -> list[str]`
  - `frequencies() -> dict[str, list[str]]` — signal_ids grouped by frequency
- `VALID_FREQUENCIES: frozenset = {"intraday", "daily", "weekly", "monthly", "event"}`

**Test:** `tests/test_convergence_taxonomy.py`
- Category set completeness (all 11)
- SignalMeta construction + validation (invalid category/frequency rejected)
- Registry CRUD: register, get, by_source, by_category, duplicate rejection
- Frequencies grouping

**Verification:** `pytest tests/test_convergence_taxonomy.py -v` — all pass.

---

#### Step 7c-A.3: Evidence extractors — first 10 tools (high-signal sources)

**File:** `agent/convergence/extractors.py`

Implement the extractor framework + the first 10 highest-signal tools:

- `ExtractorFn = Callable[[str, Any], list[Evidence]]` — type alias
- `EXTRACTOR_REGISTRY: dict[str, ExtractorFn]` — maps tool name → function
- `register_extractor(tool_name: str, fn: ExtractorFn) -> None`
- `extract_evidence(tool_name: str, tool_data: Any) -> list[Evidence]` — looks up extractor, calls it, returns evidence list. Unknown tools → empty list + log warning.
- Each extractor:
  - Takes `(tool_name: str, data: Any)` where `data` is `ToolResult.data`
  - Returns `list[Evidence]` (may be empty if data invalid/missing)
  - Defensive: try/except around all field access, log + skip on error, never crash
  - Sets `confidence` based on source quality (hardcoded per tool, adjustable later)

**First 10 extractors** (chosen for coverage across categories):

| # | Tool | Category | Primary signal_id pattern | Key fields extracted |
|---|------|----------|--------------------------|---------------------|
| 1 | `cftc` | positioning | `cftc.{commodity}.mm_net_long` | net positions, z-scores |
| 2 | `weather_alerts` | physical_disruption | `weather.{country}.alert_count` | alert count, severity |
| 3 | `sanctions` | regulatory_action | `sanctions.{program}.additions` | additions/removals count |
| 4 | `ais_vessel` | physical_flow | `ais.{region}.vessel_count` | vessel count changes |
| 5 | `finra_short_volume` | positioning | `finra.{ticker}.short_ratio` | short volume ratio |
| 6 | `disease_surveillance` | biological | `disease.{pathogen}.detection_rate` | wastewater/ILI levels |
| 7 | `earthquake` | physical_disruption | `earthquake.{region}.magnitude_max` | max magnitude, count |
| 8 | `global_pmi` | macro_momentum | `pmi.{country}.manufacturing` | PMI values |
| 9 | `treasury_receipts` | macro_momentum | `treasury.{category}.daily_receipts` | receipt amounts |
| 10 | `job_postings` | behavioral_intent | `jobs.{sector}.posting_change` | posting count delta |

Each extractor:
1. Validates the expected keys exist in `data` (defensive get with defaults)
2. Constructs Evidence with appropriate `direction` (+1/−1/0), `confidence` (0.3–0.9 depending on source), `category`, `tags`, `ttl`
3. Returns list (one tool may produce multiple Evidence — e.g., CFTC has multiple commodities)

**Test:** `tests/test_convergence_extractors.py`
- For each of the 10 tools: fixture with representative `ToolResult.data` → extract → verify Evidence fields
- Missing/malformed data → empty list, no exception
- Unknown tool name → empty list + warning
- None data → empty list

**Verification:** `pytest tests/test_convergence_extractors.py -v` — all pass.

---

#### Step 7c-A.4: Evidence extractors — remaining 50 tools

**File:** `agent/convergence/extractors.py` (extend)

Add extractors for the remaining ~50 tools using the same pattern. Group by category:

**physical_flow (5 more):** `transport_throughput`, `energy_supply`, `capital_flows`, `supply_chain_monitor`, `interconnection_queue`

**physical_disruption (2 more):** `internet_infra`, `electricity_monitor`

**financial_stress (5):** `sovereign_debt`, `creditor_filings`, `bankruptcy_court`, `defi_flows`, `liquidity_regime`

**monetary_policy (3):** `central_bank_balance`, `rate_monitor`, (`capital_flows` already counted above)

**regulatory_action (5):** `drug_regulatory`, `regulatory_gazette`, `foia_requests`, `gov_contracts`, `building_permits`

**behavioral_intent (6):** `patent_filings`, `lobbying`, `wikipedia_pageviews`, `cert_transparency`, `dns_changes`, `academic_papers`

**positioning (3 more):** `polymarket`, `polymarket_whales`, `insider_filings`, `form144`

**macro_momentum (3 more):** `consumer_sentiment`, `supply_chain_monitor` (already counted), (`building_permits` already counted)

**biological (1 more):** `food_security`

**geopolitical (3):** `political_risk`, `gdelt`, `migration_flows`

**supply_chain (as counted above)**

Each extractor follows the same defensive pattern established in Step 7c-A.3. Many will be very simple (2-5 lines) because the tool data is already structured.

Not all 60 tools need extractors immediately — any tool not yet producing pipeline data (unbuilt 7b items) gets a placeholder that returns `[]`.

**Test:** `tests/test_convergence_extractors.py` (extend)
- At minimum: 1 positive fixture + 1 negative (empty/malformed) per extractor
- Bulk test: all registered extractors handle `None` gracefully

**Verification:** `pytest tests/test_convergence_extractors.py -v` — all pass.

---

#### Step 7c-A.5: Sub-phase A edge-case test suite

Comprehensive edge-case tests covering the entire Evidence Protocol layer:

- Evidence with NaN value (valid — categorical signals)
- Evidence with confidence exactly 0.0 and exactly 1.0 (boundary)
- Evidence with `ttl=1` (minimum staleness)
- Evidence with very large timestamp (year 2100) — should still work
- Evidence with empty tags tuple
- EvidenceBus with 10,000 items (performance sanity)
- EvidenceBus flush while iterating (safety)
- SignalRegistry with 500 signals (performance sanity)
- Extractor receiving deeply nested/unexpected data structures
- Extractor receiving list instead of dict (wrong type)
- Extractor receiving extra unexpected keys (should ignore)
- Extractor receiving numeric strings instead of numbers (should handle or skip)
- Taxonomy category string with wrong case (should reject — case-sensitive)

**Verification:** `pytest tests/test_convergence_evidence.py tests/test_convergence_taxonomy.py tests/test_convergence_extractors.py -v` — all pass.

---

### Sub-phase 7c-B: Temporal Alignment + Atomic Signals

Takes Evidence streams and produces normalized, aligned, anomaly-scored time series ready for coincidence detection.

#### Step 7c-B.1: Temporal alignment engine

**File:** `agent/convergence/alignment.py`

Implement:

- `TimeGrid` enum: `INTRADAY`, `DAILY`, `WEEKLY`, `MONTHLY`
  - `period_seconds() -> int` — canonical period length
  - `coarser(a: TimeGrid, b: TimeGrid) -> TimeGrid` — return the coarser of two grids
- `FREQUENCY_TO_GRID: dict[str, TimeGrid]` mapping from SignalMeta.frequency
- `align_pair(series_a: list[Evidence], series_b: list[Evidence], meta_a: SignalMeta, meta_b: SignalMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Returns `(timestamps, values_a, values_b)` aligned to the coarser grid
  - Uses LOCF (last observation carried forward) for downsampling — never interpolation
  - Marks stale observations as NaN (when observation age > ttl)
  - Event-driven signals → binary flag (1 if event occurred in period, 0 otherwise) or count-in-window
- `align_to_grid(series: list[Evidence], grid: TimeGrid, start: float, end: float) -> tuple[np.ndarray, np.ndarray]`
  - Returns `(timestamps, values)` on the specified grid
  - LOCF fill, staleness-aware
- `is_stale(evidence: Evidence, as_of: float) -> bool`
  - `as_of - evidence.timestamp > evidence.ttl`

**Critical constraint:** NEVER interpolate. NEVER upsample. Always align to the COARSER frequency. Forward-fill only (LOCF). This prevents future information leakage.

**Test:** `tests/test_convergence_alignment.py`
- Daily + weekly → weekly grid
- Hourly + monthly → monthly grid
- LOCF fill correctness (last value carried, not interpolated)
- Stale values → NaN (verify with ttl boundary)
- Event-driven (irregular timestamps) → daily binary flag
- Empty series handling → all-NaN output
- Single observation series
- Non-overlapping time ranges → all NaN

**Verification:** `pytest tests/test_convergence_alignment.py -v` — all pass.

---

#### Step 7c-B.2: Atomic signal computation

**File:** `agent/convergence/atomic_signals.py`

Implement:

- `RollingStats` class:
  - `__init__(window: int = 52)` — window size (default 52 for ~1 year of weekly data)
  - `update(values: np.ndarray) -> None` — ingest new aligned values
  - `z_score(value: float) -> float` — `(x - μ) / σ`, returns 0 if σ < ε (1e-10)
  - `percentile(value: float) -> float` — empirical rank / N
  - `mean -> float`, `std -> float` — current rolling stats
  - `n_observations -> int` — count of non-NaN values ingested
- `compute_anomaly(z: float, pct: float, z_threshold: float = 2.0, pct_lo: float = 0.05, pct_hi: float = 0.95) -> bool`
  - Returns True if `|z| > z_threshold` or `pct < pct_lo` or `pct > pct_hi`
- `normalize_direction(value: float, flip_sign: bool) -> float`
  - If `flip_sign`: return `−value`, else return `value`
  - Applied BEFORE z-score so that positive always means "stress/expansion"
- `SignalStream` class — per-signal state container:
  - `__init__(signal_id: str, meta: SignalMeta)`
  - `ingest(evidence_list: list[Evidence]) -> None` — accumulate observations, sorted by time
  - `compute(as_of: float) -> AtomicSignalResult | None`
    - Returns None if insufficient observations (< meta.min_observations)
    - Returns `AtomicSignalResult(signal_id, timestamp, raw_value, z_score, percentile, is_anomaly, direction)`
  - `history() -> np.ndarray` — raw values array

```python
@dataclass
class AtomicSignalResult:
    signal_id: str
    timestamp: float
    raw_value: float
    z_score: float
    percentile: float
    is_anomaly: bool
    direction: int   # +1/-1 after direction normalization
```

**Test:** `tests/test_convergence_atomic.py`
- Z-score with known values (hand-computed)
- Percentile with known ranks
- σ ≈ 0 → z_score = 0 (not NaN/Inf)
- Direction flip: confirm negative becomes positive when flip_sign=True
- Anomaly thresholds: exactly at boundary, just above, just below
- Insufficient observations → None (cold start)
- NaN values in input → excluded from rolling stats
- Window rollover: verify old values drop out

**Verification:** `pytest tests/test_convergence_atomic.py -v` — all pass.

---

#### Step 7c-B.3: Sub-phase B edge-case test suite

- Alignment with series containing only NaN values
- Alignment with all-identical timestamps (duplicates)
- Alignment where one series has 1 observation and the other has 1000
- Z-score on a series of zeros (σ=0)
- Z-score on a series of all-identical nonzero values
- Percentile with ties (multiple identical values)
- RollingStats with window=1
- RollingStats with window larger than data (N < window)
- SignalStream.ingest with out-of-order timestamps (should sort)
- SignalStream.ingest with duplicate timestamps (should deduplicate, keep latest)
- Direction normalization with flip_sign on zero value
- Anomaly detection with all values anomalous (entire series extreme)
- Staleness edge: evidence.timestamp + evidence.ttl exactly equals as_of

**Verification:** `pytest tests/test_convergence_alignment.py tests/test_convergence_atomic.py -v` — all pass.

---

### Sub-phase 7c-C: Coincidence Detection + Graph + FDR

The mathematical core. Detects unusual pairwise co-movement, builds a sparse coincidence graph, finds cliques, and controls false discovery rate.

#### Step 7c-C.1: Pairwise coincidence scoring

**File:** `agent/convergence/coincidence.py`

Implement three scoring methods (the computationally lighter ones — transfer entropy and MI deferred to later sub-phase per research note):

- **Method 1 — Rolling Correlation Deviation:**
  ```python
  def rolling_correlation_score(
      a: np.ndarray, b: np.ndarray,
      corr_window: int = 20,
      baseline_window: int = 100
  ) -> CoincidenceResult:
  ```
  - Compute ρ_t (rolling Pearson corr over `corr_window`)
  - Compute μ_ρ, σ_ρ (mean/std of ρ over `baseline_window`)
  - Score = |ρ_t − μ_ρ| / σ_ρ (z-score of correlation itself)
  - p-value from normal distribution tail
  - Detects both convergence (uncorrelated → correlated) and divergence (correlated → decorrelated)

- **Method 2 — Joint Exceedance:**
  ```python
  def joint_exceedance_score(
      z_a: np.ndarray, z_b: np.ndarray,
      z_threshold: float = 2.0,
      window: int = 20
  ) -> CoincidenceResult:
  ```
  - Count joint exceedances: both |z_a| > threshold AND |z_b| > threshold simultaneously
  - Expected under independence: P(|Z|>2)² × window ≈ 0.023² × 20 ≈ 0.011
  - Binomial test p-value: observed exceedances vs expected rate
  - Captures tail co-movement

- **Method 3 — Concordance Index:**
  ```python
  def concordance_score(
      a: np.ndarray, b: np.ndarray,
      window: int = 20
  ) -> CoincidenceResult:
  ```
  - Binary: sign(Δa_t) == sign(Δb_t) for each period
  - Hit rate over window (proportion of concordant moves)
  - Binomial test against H₀: p = 0.5
  - Simplest directional agreement measure

- **Combined score:**
  ```python
  def combined_coincidence_score(
      a: np.ndarray, b: np.ndarray,
      z_a: np.ndarray, z_b: np.ndarray,
      weights: dict[str, float] | None = None
  ) -> CoincidenceResult:
  ```
  - Weighted average of all three methods (default: equal weights)
  - Combined p-value via Fisher's method

```python
@dataclass
class CoincidenceResult:
    method: str           # "rolling_corr" | "joint_exceedance" | "concordance" | "combined"
    score: float          # 0–∞ (z-score scale — higher = more unusual)
    p_value: float        # significance
    direction: int        # +1 converging, -1 diverging
    detail: dict          # method-specific metadata (window, observed_count, expected, etc.)
```

**Test:** `tests/test_convergence_coincidence.py`
- Two perfectly correlated signals → low p-value, high score
- Two independent random signals → p-value near uniform [0,1]
- Signal anticorrelated → captures divergence
- Known synthetic case: inject correlation change at t=50, verify detection near t=50
- Arrays with NaN (should skip NaN pairs)
- Short arrays (< window) → graceful return (score=0, p=1.0)
- Constant array (σ=0) → handle without division error

**Verification:** `pytest tests/test_convergence_coincidence.py -v` — all pass.

---

#### Step 7c-C.2: Coincidence graph + clique detection

**File:** `agent/convergence/graph.py`

Implement:

- `build_coincidence_graph(scores: dict[tuple[str, str], CoincidenceResult], p_threshold: float = 0.05) -> nx.Graph`
  - Nodes = signal_ids
  - Edges where p_value < p_threshold, weight = score
  - Node attributes: category (from taxonomy)
  - Returns networkx Graph

- `detect_convergence_cliques(G: nx.Graph, min_size: int = 3, min_categories: int = 2) -> list[ConvergenceClique]`
  - Find all maximal cliques of size ≥ min_size (Bron-Kerbosch via `nx.find_cliques`)
  - Filter: must include signals from ≥ min_categories distinct taxonomy categories
  - Cross-category priority: rank by number of distinct categories, then by total edge weight

- `score_clique(clique: ConvergenceClique) -> float`
  - Formula from research:
    $$\text{score} = \frac{1}{|C|} \sum_{(i,j) \in C} w_{ij} \times \frac{\text{cross\_cat\_count}}{|C|} \times \log_2(|C|)$$
  - Normalized to [0, 1] via sigmoid or empirical rescaling

```python
@dataclass
class ConvergenceClique:
    signals: list[str]          # Signal IDs in this clique
    categories: list[str]       # Distinct categories
    edges: list[tuple[str, str, float]]  # (sig_a, sig_b, weight)
    score: float                # Aggregate convergence score
    p_values: list[float]       # Individual edge p-values (for Fisher's)
```

**Test:** `tests/test_convergence_graph.py`
- Build graph from known scores → verify node/edge counts
- Clique detection on handcrafted graph (4 mutually connected nodes across 3 categories → detected)
- 3 connected, same category → filtered out (min_categories=2)
- Isolated node → not in any clique
- Empty graph → empty clique list
- Scoring: hand-compute expected score for a known clique

**Verification:** `pytest tests/test_convergence_graph.py -v` — all pass.

---

#### Step 7c-C.3: False discovery rate control

**File:** `agent/convergence/fdr.py`

Implement:

- `apply_bh_correction(p_values: dict[tuple[str, str], float], q: float = 0.05) -> dict[tuple[str, str], bool]`
  - Benjamini-Hochberg procedure on all pairwise p-values
  - Returns dict mapping pair → True if significant at FDR level q
  - Uses `statsmodels.stats.multitest.multipletests(method='fdr_bh')`

- `fisher_combined_test(p_values: list[float]) -> float`
  - Fisher's method: χ² = −2 Σ ln(pᵢ), df = 2k
  - Returns combined p-value from chi-squared survival function
  - Guard: p=0 → clip to 1e-300 (avoid log(0))

- `persistence_filter(events: list[ConvergenceClique], history: dict[str, int], min_periods: int = 2) -> list[ConvergenceClique]`
  - A convergence must have been detected for ≥ min_periods consecutive cycles
  - `history` maps a clique fingerprint → number of consecutive detections
  - First detection: record in history but don't emit
  - Second+ detection: emit and increment counter
  - If a clique disappears: reset its counter to 0

- `cross_category_filter(events: list[ConvergenceClique], min_categories: int = 2) -> list[ConvergenceClique]`
  - Redundant safety check (also enforced in graph.py)

- `apply_all_controls(pairs_p: dict, cliques: list, history: dict, q: float = 0.05, min_persist: int = 2, min_cats: int = 2) -> list[ConvergenceClique]`
  - Pipeline: BH → filter pairs → rebuild graph → redetect cliques → Fisher per clique → persistence → cross-category → return survivors

**Test:** `tests/test_convergence_fdr.py`
- BH with known p-values (hand-verified rejection set)
- BH with all p-values = 0.5 → none rejected
- BH with all p-values = 0.001 → all rejected
- Fisher's combined test: two p-values of 0.03 each → combined p < 0.01
- Fisher's with a single p-value → returns that p-value
- Persistence: first cycle → not emitted, second cycle → emitted
- Persistence: disappearance → counter reset → requires 2 more cycles
- Cross-category: exactly 2 categories → passes, 1 category → filtered
- Full pipeline with synthetic data: inject 1 real convergence + 10 spurious → verify only real survives

**Verification:** `pytest tests/test_convergence_fdr.py -v` — all pass.

---

#### Step 7c-C.4: Sub-phase C edge-case test suite

- Coincidence on arrays of length 0 and 1
- Coincidence where one signal is constant (σ=0)
- Coincidence with all-NaN arrays
- Graph with 1000 nodes, sparse edges → performance under 1 second
- Clique detection with a single fully-connected component of 60 nodes (worst case for Bron-Kerbosch)
- BH correction with 0 p-values (empty input)
- BH correction with 1 p-value
- BH correction with 10,000 p-values (performance)
- Fisher's test with p-value = 0.0 (clipped)
- Fisher's test with p-value = 1.0 (no evidence)
- Persistence filter with min_periods=1 (emit on first detection)
- Combined scoring with NaN weights
- Graph where every pair is significant → massive single clique → verify score scales

**Verification:** All tests in `tests/test_convergence_coincidence.py tests/test_convergence_graph.py tests/test_convergence_fdr.py` pass.

---

### Sub-phase 7c-D: Templates + Detector + DAG Integration

Wires everything together into a runnable pipeline.

#### Step 7c-D.1: Causal chain template library + matcher

**File:** `agent/convergence/templates.py`

Implement:

- `CausalTemplate` dataclass:
  ```python
  @dataclass(frozen=True)
  class TemplateStep:
      category_pattern: str   # e.g., "physical_disruption" or "physical_disruption|regulatory_action"
      signal_pattern: str     # regex on signal_id (e.g., "sanctions\\..*\\.additions")
      within_days: int        # max days after trigger step
      direction: int | None   # expected direction (+1/-1) or None (any)

  @dataclass(frozen=True)
  class CausalTemplate:
      name: str               # e.g., "supply_chain_disruption"
      description: str
      steps: tuple[TemplateStep, ...]   # Ordered trigger → response sequence
      min_match: int          # Minimum number of steps that must match (default: len(steps)-1)
  ```

- `TEMPLATE_LIBRARY: list[CausalTemplate]` — the 12 core templates from research note:
  1. Supply Chain Disruption
  2. Monetary Policy Shift
  3. Geopolitical Escalation
  4. Pandemic/Health Crisis
  5. Agricultural Shock
  6. Energy Crisis
  7. Credit Stress Cascade
  8. Tech/Innovation Disruption
  9. Labor Market Shift
  10. Trade War Escalation
  11. Real Estate / Construction Cycle
  12. Digital Infrastructure Crisis

- `match_template(clique: ConvergenceClique, evidence_timeline: list[Evidence], template: CausalTemplate) -> TemplateMatchResult`
  - Check if the signals in the clique match the template's step sequence
  - Verify temporal ordering (trigger before response) and direction constraints
  - Return match score: `matched_steps / total_steps` (0.0 – 1.0)
  - Identify lead signal (matches trigger step) and lag signals

- `match_all_templates(clique: ConvergenceClique, evidence_timeline: list[Evidence]) -> list[TemplateMatchResult]`
  - Try all templates, return sorted by match score (descending)
  - Top match with score ≥ 0.5 → assign as event_type
  - No match ≥ 0.5 → event_type = "unknown_pattern"

```python
@dataclass
class TemplateMatchResult:
    template_name: str
    match_score: float         # 0.0 – 1.0
    matched_steps: int
    total_steps: int
    lead_signal: str | None
    lag_signals: list[str]
    temporal_order_valid: bool
```

**Test:** `tests/test_convergence_templates.py`
- Verify all 12 templates parse correctly
- Known supply-chain scenario (sanctions → shipping → CFTC → PMI) → matches template with score > 0.8
- Reverse temporal order → match_score lower or temporal_order_valid=False
- Partial match (2 of 4 steps) → score ≈ 0.5
- Clique that matches no template → "unknown_pattern"
- Empty clique → no match
- Template with direction constraint → violated direction → lower score

**Verification:** `pytest tests/test_convergence_templates.py -v` — all pass.

---

#### Step 7c-D.2: ConvergenceDetector — top-level orchestrator

**File:** `agent/convergence/detector.py`

Implement:

- `ConvergenceDetectorConfig` dataclass:
  ```python
  @dataclass
  class ConvergenceDetectorConfig:
      z_threshold: float = 2.0        # Anomaly z-score threshold
      p_threshold: float = 0.05       # Pairwise significance level
      fdr_q: float = 0.05             # BH FDR target
      min_clique_size: int = 3        # Minimum signals per convergence
      min_categories: int = 2         # Minimum taxonomy categories
      min_persistence: int = 2        # Consecutive periods before emission
      corr_window: int = 20           # Rolling correlation window
      baseline_window: int = 100      # Baseline for correlation z-score
      template_boost: float = 0.5     # α in score × (1 + α × template_match)
      lookback_days: int = 365        # How far back to read pipeline data
  ```

- `ConvergenceDetector` class:
  ```python
  class ConvergenceDetector:
      def __init__(
          self,
          store: PipelineStore,
          signal_registry: SignalRegistry,
          config: ConvergenceDetectorConfig | None = None
      ): ...
      
      def detect(self, as_of: float | None = None) -> list[ConvergenceSignal]:
          """Run one full detection cycle. Returns emittable convergence signals."""
          # 1. Load recent pipeline_data from store (lookback_days)
          # 2. Extract evidence via extractors
          # 3. Build SignalStreams, compute AtomicSignalResults
          # 4. Align pairs, compute pairwise coincidence scores
          #    (Only score pairs across taxonomy categories + within same category if historical correlation is low)
          # 5. Build coincidence graph
          # 6. Detect cliques
          # 7. Apply FDR controls (BH, Fisher, persistence)
          # 8. Match templates
          # 9. Emit ConvergenceSignals
          # 10. Store signals to pipeline store
          return signals
      
      @property
      def persistence_history(self) -> dict[str, int]:
          """Current persistence state (for inspection/testing)."""
  ```

- **Smart pair selection** (avoid O(n²)):
  1. Always score cross-category pairs: for each category pair (A, B), pick the top signal from each category (highest current z-score) → ~55 category pairs
  2. Within-category: only score if their historical rolling correlation < 0.3 (normally uncorrelated)
  3. This reduces from 1,770 to ~100-300 active pairs

**Test:** `tests/test_convergence_detector.py`
- **Synthetic null scenario:** 10 signals, all independent random → 0 convergences emitted (FDR controls)
- **Synthetic positive scenario:** 5 independent + 3 injected-correlated (simulating supply chain disruption) → 1 convergence detected matching the correct template
- **Cold start:** Only 5 observations per signal → returns empty (min_observations not met)
- **Partial data:** 3 of 10 tools failed/missing → detector runs with available data, does not crash
- **Config override:** Custom thresholds → verify they propagate to all sub-components
- Historical data spanning 1 year → detect runs under 5 seconds

**Verification:** `pytest tests/test_convergence_detector.py -v` — all pass.

---

#### Step 7c-D.3: Signal emission + pipeline store integration

**File:** `agent/convergence/signals.py`

Implement:

- `ConvergenceSignal` dataclass (as specified in Data Structures section above)

- `emit_signals(signals: list[ConvergenceSignal], store: PipelineStore) -> int`
  - For each signal: call `store.store_signal(signal_name=signal.signal_name, value=signal.value, metadata=signal.to_metadata_dict())`
  - Returns count of stored signals

- `ConvergenceSignal.to_metadata_dict() -> dict`
  - Serialize all non-core fields into a dict suitable for JSON storage

- `format_signal_name(event_type: str, date: str) -> str`
  - Returns `f"convergence.{event_type}.{date}"`
  - Validates event_type is safe (alphanumeric + underscore only)

**Test:** `tests/test_convergence_signals.py`
- Emit to in-memory PipelineStore → query back → verify fields match
- Signal name formatting
- Empty signals list → 0 stored, no error
- Metadata round-trip: store → query → deserialize → compare

**Verification:** `pytest tests/test_convergence_signals.py -v` — all pass.

---

#### Step 7c-D.4: Convergence DAG definition

**File:** `agent/pipeline/dags/convergence_detection.py`

Implement:

```python
def build_convergence_detection_dag() -> DAG:
    """
    Convergence detection DAG.
    Depends on daily_collection having run (reads its outputs from pipeline store).
    Runs after daily_collection completes (or on independent schedule).
    """
    dag = DAG(
        name="convergence_detection",
        schedule="30 18 * * 1-5",  # 30 min after daily_collection
        description="Multi-source convergence detection"
    )
    
    dag.add(
        "run_detection",
        operator=_run_convergence_detection,  # FunctionOperator
        params={},
        timeout=300,  # 5 min generous timeout
        retries=1,
        store_result=True,
        table_name="convergence_results"
    )
    
    return dag

def _run_convergence_detection(params: dict, upstream_results: dict) -> dict:
    """FunctionOperator callback. Instantiates detector, runs, returns summary."""
    # 1. Create PipelineStore (from config path)
    # 2. Build SignalRegistry (full catalog)
    # 3. Create ConvergenceDetector
    # 4. Call detector.detect()
    # 5. Return summary dict with count, events, etc.
```

- Register in `agent/pipeline/dags/__init__.py`

**Test:** `tests/test_convergence_dag.py`
- DAG validates (no cycles, no missing deps)
- DAG topo_sort returns correct layers
- Can execute DAG with mocked store (no real data needed)
- Node result stored as `convergence_results` in pipeline_data

**Verification:** `pytest tests/test_convergence_dag.py -v` — all pass.

---

#### Step 7c-D.5: CLI integration + dependency registration

**Modify:** `agent/cli.py`, `agent/pipeline/dags/__init__.py`, `pyproject.toml`

- Add `convergence_detection` DAG to the pipeline startup registry
- Add `networkx>=3.0` and `ruptures>=1.1` to `pyproject.toml` dependencies
- Verify imports work and DAG is registered at startup

**Test:** Smoke test — `python -c "from agent.convergence.detector import ConvergenceDetector; print('OK')"`

**Verification:** Import succeeds, no circular imports, DAG appears in registry.

---

#### Step 7c-D.6: Sub-phase D edge-case test suite + full integration test

Comprehensive edge-case tests for the full integration:

- **Detector with empty store** (no pipeline_data rows at all) → returns []
- **Detector with data from only 1 tool** → returns [] (can't detect cross-source convergence)
- **Detector with 60 tools, all returning noise** → returns [] (FDR filters everything)
- **Detector with 60 tools, 5 showing genuine convergence** → returns 1+ events
- **Template matcher with signals that arrive in wrong temporal order** → lower score
- **Template matcher with perfect match** → score ≈ 1.0
- **DAG execution with store that raises SQLite error** → node status = "failed", not crash
- **ConvergenceSignal serialization round-trip** → no data loss
- **Signal name with special characters** → rejected
- **Persistence across multiple detect() calls** → state maintained correctly
- **Two overlapping cliques** → both detected independently
- **Performance test:** 60 sources × 365 days history → detect() < 5 seconds

**Verification:** Full test suite: `pytest tests/test_convergence_*.py -v` — all pass.

---

## Edge Cases (Cross-Cutting)

| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 1 | Tool returns `data=None` | Extractor returns `[]`, no error |
| 2 | Tool returns unexpected schema | Extractor catches KeyError, returns `[]`, logs warning |
| 3 | Only 2 signals have data (too few for clique) | No convergence emitted |
| 4 | All signals are anomalous simultaneously | Large clique detected, high score, but FDR still applies |
| 5 | Pipeline data has gaps (missing days) | LOCF fills, stale values → NaN |
| 6 | Signal flips direction mid-series | Rolling stats adapt, no special handling needed |
| 7 | SQLite database locked (concurrent access) | WAL mode handles reads; writes retry |
| 8 | New tool added (no extractor registered) | Unknown tool → empty evidence, logged |
| 9 | σ = 0 in rolling window (constant signal) | z-score → 0, not NaN/Inf |
| 10 | Clique fingerprint changes (one signal drops out)| Old fingerprint persistence counter resets |
| 11 | detect() called twice in 1 second | Second call uses fresh pipeline data, persistence state shared |
| 12 | 10,000 p-values in BH correction | statsmodels handles efficiently |

---

## Testing Plan Summary

| Test file | Coverage | Key technique |
|-----------|----------|---------------|
| `test_convergence_evidence.py` | Evidence, EvidenceBus | Unit + validation |
| `test_convergence_taxonomy.py` | Categories, SignalMeta, SignalRegistry | Unit + validation |
| `test_convergence_extractors.py` | All 60 extractors | Fixture-driven, defensive |
| `test_convergence_alignment.py` | TimeGrid, align_pair, LOCF, staleness | Synthetic time series |
| `test_convergence_atomic.py` | RollingStats, z-score, percentile, anomaly | Hand-computed values |
| `test_convergence_coincidence.py` | 3 scoring methods + combined | Synthetic correlated/independent |
| `test_convergence_graph.py` | Graph construction, clique detection, scoring | Handcrafted adjacency |
| `test_convergence_templates.py` | 12 templates, matcher, temporal ordering | Scenario-based |
| `test_convergence_fdr.py` | BH, Fisher, persistence, cross-category | Known p-value sets |
| `test_convergence_detector.py` | Full pipeline: data → evidence → convergence | Synthetic scenarios |
| `test_convergence_signals.py` | Signal emission, store integration | Round-trip verification |
| `test_convergence_dag.py` | DAG wiring, execution, registration | Integration w/ pipeline |

**Total estimated tests:** ~200-300, covering happy paths, error paths, boundary conditions, performance.

---

## Dependencies Added

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| `networkx` | ≥ 3.0 | BSD-3 | Graph construction, clique detection (Bron-Kerbosch) |
| `ruptures` | ≥ 1.1 | BSD-2 | Offline batch changepoint (PELT) — deferred to 7c+ |
| `statsmodels` | (already present) | BSD-3 | BH FDR correction, Fisher's combined test |
| `numpy` | (already present) | BSD-3 | Array operations, rolling statistics |
| `scipy` | (already present) | BSD-3 | Statistical tests (binomial, chi-squared) |

---

## Not In Scope (Deferred)

| Feature | Deferred to | Reason |
|---------|-------------|--------|
| Transfer entropy (directed causality) | 7c+ (after 30+ days of data) | Computationally heavier, needs more history |
| Mutual information (nonlinear detection) | 7c+ | Same as TE |
| pgmpy Bayesian network integration | Phase 9 | World Model phase |
| Offline batch changepoint (ruptures/PELT) | 7c+ | Needs accumulated history |
| Real-time streaming (Kafka/Flink) | Never (batch is sufficient) | Over-engineering |
| LLM-based convergence reasoning | Phase 12+ | Violates pipeline determinism |
| Copula tail dependence | Likely never | Data requirements unrealistic |
| Dempster-Shafer evidence theory | Likely never | Confidence scores sufficient |

---

## Related

- [[convergence_detection|Research: Convergence Detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
