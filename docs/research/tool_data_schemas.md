---
title: "Tool Data Schemas — ToolResult `data=` Catalog"
tags:
  - doc/research
---

# Tool Data Schemas — ToolResult `data=` Catalog

Extracted from source code in `agent/tools/`. Each entry shows the tool name,
file, modes, and the keys/structure of the `data` dict returned in `ToolResult`.

Tools marked **(output-only)** return `ToolResult(success=True, output=str)`
without a structured `data` dict.

---

## 1. transport_throughput

**FILE:** `transport_throughput.py`
**MODES:** `recent`, `trend`, `port`, `compare`

| Mode | DATA KEYS |
|------|-----------|
| `recent` | `records: list[{border, measure, total, period}]`, `count: int`, `period: str` |
| `trend` | `series: list[{date, border, total}]`, `count: int`, `measure: str` |
| `port` | `ports: list[{port, state, border, value}]`, `count: int`, `period: str` |
| `compare` | `comparison: list[{date, canada, mexico, ratio}]`, `count: int`, `measure: str` |

---

## 2. capital_flows

**FILE:** `capital_flows.py`
**MODES:** `holdings`, `flows`, `reserves`

| Mode | DATA KEYS |
|------|-----------|
| `holdings` | `mode: "holdings"`, `holdings: list[{country, key, latest_value_billions, latest_date, mom_change_pct, observations}]`, `coordination: {coordinated_selling, coordinated_buying, sellers, buyers}`, `errors: list` |
| `flows` | `mode: "flows"`, `flows: list[{series, key, description, latest_value, latest_date, period_average, flow_reversal, observations}]`, `errors: list` |
| `reserves` | `mode: "reserves"`, `reserves: list[{series, key, latest_value, latest_date, observations, stress: {stress, drawdown_pct, latest_value, comparison_value, ...}}]`, `stress_alerts: list`, `errors: list` |

---

## 3. creditor_filings

**FILE:** `creditor_filings.py`
**MODES:** `search`, `uk_charges`, `stress_scan`

| Mode | DATA KEYS |
|------|-----------|
| `search` | `mode: "search"`, `query: str`, `sec_count: int`, `sec_total: int`, `sec_entries: list[{company_name, cik, file_date, form, items}]`, `ch_charges: list\|None`, `ch_red_flags: int` |
| `uk_charges` | `mode: "uk_charges"`, `company: dict`, `charges: list[{charge_number, status, created_on, delivered_on, satisfied_on, classification, persons_entitled, particulars}]`, `red_flags: int` |
| `stress_scan` | `mode: "stress_scan"`, `sec_count: int`, `clusters: list[{entity, filing_count, date_range, cik}]`, `sec_entries: list` |

---

## 4. bankruptcy_court

**FILE:** `bankruptcy_court.py`
**MODES:** `us_bankruptcy`, `sec_enforcement`, `sec_bankruptcy`, `uk_insolvency`

| Mode | DATA KEYS |
|------|-----------|
| `us_bankruptcy` | `mode: "us_bankruptcy"`, `count: int`, `chapter_breakdown: dict`, `court_breakdown: dict`, `entries: list[{case_number, debtor_name, chapter, court, court_name, link, pub_date, description}]` |
| `sec_enforcement` | `mode: "sec_enforcement"`, `count: int`, `type_breakdown: dict`, `entries: list[{title, type, link, pub_date, description}]` |
| `sec_bankruptcy` | `mode: "sec_bankruptcy"`, `count: int`, `total: int`, `days_back: int`, `entries: list[{company_name, cik, file_date, form, items}]` |
| `uk_insolvency` | `mode: "uk_insolvency"`, `count: int`, `source_breakdown: dict`, `entries: list[{title, source, link, pub_date, description}]` |

---

## 5. liquidity_regime

**FILE:** `liquidity_regime.py`
**MODES:** *(single mode)* — params: `lookback_years`, `global_`

**DATA KEYS:** `current_regime: str("contraction"|"neutral"|"expansion")`, `current_state: int`, `composite_zscore: float`, `regime_means: list[float]`, `regime_variances: list[float]`, `transition_matrix: list[list[float]]`, `last_changepoint: str|None`, `n_changepoints: int`, `n_weeks: int`

---

## 6. central_bank_balance

**FILE:** `central_bank_balance.py`
**MODES:** `balance_sheets`, `liquidity_index`, `policy_divergence`, `rate_monitor`

| Mode | DATA KEYS |
|------|-----------|
| `balance_sheets` | `banks: list[{bank, code, currency, latest_date, native_trillions, usd_trillions, wow_pct, mom_pct, yoy_pct}]`, `errors: list` |
| `liquidity_index` | `gross_usd: float`, `rrp_usd: float`, `tga_usd: float`, `net_usd: float`, `components: dict`, `errors: list` |
| `policy_divergence` | `assessments: list`, `rates: dict`, `divergences: list`, `synchronized: bool`, `errors: list` |
| `rate_monitor` | `rates: list`, `errors: list` |

---

## 7. drug_regulatory

**FILE:** `drug_regulatory.py`
**MODES:** `approvals`, `adverse_events`, `labels` (also supports `count_field` for faceted counts)

| Mode | DATA KEYS |
|------|-----------|
| `approvals` | `mode: "approvals"`, `results: list[{application_number, sponsor, brands, latest_submission_type, latest_submission_date, review_priority}]`, `total: int` |
| `adverse_events` | `mode: "adverse_events"`, `results: list[{date, serious, drugs, reactions}]`, `total: int`, `signals: {seriousness_ratio, serious_count, total_in_page}` |
| `labels` | `mode: "labels"`, `results: list[{brand, generic, has_boxed_warning, warnings_text, indications_text}]`, `total: int` |
| *(any w/ count_field)* | `mode: str`, `count_field: str`, `counts: list[{term, count}]`, `total: int` |

---

## 8. regulatory_gazette

**FILE:** `regulatory_gazette.py`
**MODES:** `recent`, `search`, `agency`, `upcoming`

| Mode | DATA KEYS |
|------|-----------|
| `recent/search/upcoming` | `documents: list[{title, type, document_number, publication_date, agencies, abstract, action, comments_close_on, effective_on, topics, significant, docket_ids, page_length, url}]`, `count: int` (some modes add `total: int`) |
| `agency` *(list)* | `agencies: dict` (the `MARKET_AGENCIES` constant) |

---

## 9. building_permits

**FILE:** `building_permits.py`
**MODES:** `permits`, `regional`, `housing_starts`

| Mode | DATA KEYS |
|------|-----------|
| `permits` | `mode: "permits"`, `summary: dict[series_id → {label, latest_date, latest_value, mom_pct, yoy_pct, trend, consecutive_declines}]`, `series: dict[series_id → list[{date, value}]]` |
| `regional` | `mode: "regional"`, `summary: dict`, `series: dict` |
| `housing_starts` | `mode: "housing_starts"`, `summary: dict`, `series: dict` |

---

## 10. patent_filings

**FILE:** `patent_filings.py`
**MODES:** `search`, `trends`, `assignee`

| Mode | DATA KEYS |
|------|-----------|
| `search` | `mode: "search"`, `patents: list`, `total_count: int`, `returned: int` |
| `trends` | `mode: "trends"`, `cpc_class: str`, `yearly_counts: dict[year→count]`, `total_count: int`, `sample_size: int` — OR — `mode: "trends"`, `signal_classes: dict` (SIGNAL_CPC) |
| `assignee` | `mode: "assignee"`, *(similar structure — not fully extracted)* |

---

## 11. lobbying

**FILE:** `lobbying.py`
**MODES:** `search`, `spending`, `issues`

| Mode | DATA KEYS |
|------|-----------|
| `search` | `mode: "search"`, `filings: list[{filing_uuid, filing_type, filing_year, filing_period, dt_posted, registrant_name, registrant_id, client_name, client_id, amount, issue_codes, issue_descriptions}]`, `total_count: int`, `returned: int` |
| `spending` | `mode: "spending"`, `target: str`, `yearly_totals: dict[year→float]`, `anomaly: {anomaly: bool, ratio, latest, historical_avg}`, `total_filings: int`, `errors: list` |
| `issues` | `mode: "issues"`, *(issue area filings — similar structure)* |

---

## 12. wikipedia_pageviews

**FILE:** `wikipedia_pageviews.py`
**MODES:** `spike`, `top`, `series`

| Mode | DATA KEYS |
|------|-----------|
| `spike` | `spikes: list[{article, project, latest_views, mean_views, std_views, z_score, spike_ratio, days_analyzed, date}]`, `errors: list`, `watchlist_size: int` |
| `top` | `articles: list`, `project: str`, `date: str` |
| `series` | `article: str`, `project: str`, `views: list[{date, views}]`, `stats: {mean, std, min, max, days}` |

---

## 13. cert_transparency

**FILE:** `cert_transparency.py`
**MODES:** `search`, `subdomains`, `recent`

| Mode | DATA KEYS |
|------|-----------|
| `search` | `domain: str`, `certs: list[{id, common_name, name_value, issuer, not_before, not_after, entry_timestamp, serial_number, is_expired, days_remaining}]`, `count: int`, `active: int`, `expired: int` |
| `subdomains` | `domain: str`, `subdomains: list[{subdomain, cert_count, latest_entry}]`, `concrete: list`, `wildcards: list`, `count: int` |
| `recent` | *(similar structure to search)* |

---

## 14. dns_monitor

**FILE:** `dns_monitor.py`
**MODES:** `resolve`, `diff`, `bulk_resolve`

| Mode | DATA KEYS |
|------|-----------|
| `resolve` | `domain: str`, `records: dict`, `analysis: dict`, `record_count: int` |
| `diff` *(new baseline)* | `domain: str`, `baseline_established: True`, `records: dict`, `changes: []`, `analysis: dict` |
| `diff` *(change detect)* | `domain: str`, `baseline_established: False`, `records: dict`, `changes: list`, `analysis: dict` |
| `bulk_resolve` | `results: list[{domain, records, analysis, record_count}]`, `errors: list`, `domain_count: int`, `total_records: int` |

---

## 15. polymarket

**FILE:** `polymarket.py`
**MODES:** *(single mode)* — params: `category`, `limit`, `search`

**DATA KEYS:** `markets: list[{question, slug, yes_price, no_price, volume_total, volume_24h, liquidity, spread, price_change_24h, price_change_1wk, end_date, category}]`, `total: int`

---

## 16. polymarket_whales

**FILE:** `polymarket_whales.py`
**MODES:** `top_wallets`, `wallet_detail`, `market_whales`, `recent_signals`

| Mode | DATA KEYS |
|------|-----------|
| `top_wallets` | `wallets: list[{wallet, composite, accuracy, total_resolved, total_volume, markets}]` |
| `wallet_detail` | `score: dict\|None`, `trades: list` |
| `market_whales` | `market: str`, `whales: list[{wallet, score, accuracy, trades, total_usdc}]` |
| `recent_signals` | *(not fully extracted)* |

---

## 17. insider_filings

**FILE:** `insider_filings.py`
**MODES:** *(single mode)* — params: `days_back`, `ticker`, `min_cluster_size`

**DATA KEYS:** `clusters: list[{ticker, company, insider_count, total_value, cluster_start, cluster_end, conviction, insiders: list[{name, role, shares, price, date}]}]`, `total_filings: int`, `total_purchases: int`, `scan_range: {start, end}`

---

## 18. form144

**FILE:** `form144.py`
**MODES:** *(single mode)* — params: `days_back`, `ticker`, `min_cluster_size`

**DATA KEYS:** `clusters: list[{ticker, company, insider_count, total_value, pct_of_outstanding, cluster_start, cluster_end, urgency, conviction, filings: list[{insider_name, relationship, shares_to_sell, dollar_value, filing_date, acquisition_type}]}]`, `total_filings: int`, `total_parsed: int`, `scan_range: {start, end}`

---

## 19. gdelt

**FILE:** `gdelt.py`
**MODES:** `events`, `articles`

| Mode | DATA KEYS |
|------|-----------|
| `events` | `events: list[{actor1: {name, code, country}, actor2: {...}, event_code, event_root, event_description, quad_class, quad_label, goldstein, num_mentions, num_sources, location: {name, country, lat, lon}}]`, `summary: dict` |
| `articles` | `articles: list[{title, url, domain, seendate, tone, ...}]` |

---

## 20. whale_alert

**FILE:** `whale_alert.py`
**MODES:** `mempool`, `confirmed`

**DATA KEYS:** `transactions: list[dict]`, `summary: dict`, `mode: str`

Each transaction (parsed from blockchain.info): BTC value, input/output addresses, confirmation status.

---

## 21. comtrade

**FILE:** `comtrade.py`
**MODES:** `flows`, `commodity`, `partners`

| Mode | DATA KEYS |
|------|-----------|
| `flows` | `mode: "flows"`, `reporter: str`, `partner: str`, `flow: str`, `record_count: int`, `records: list[{period, reporter, reporter_code, partner, partner_code, flow, flow_code, commodity_code, commodity, trade_value_usd, quantity, quantity_unit}]` |
| `commodity` *(with code)* | `mode: "commodity"`, `commodity_code: str`, `commodity_name: str`, `flow: str`, `record_count: int`, `records: list[...]` (same record structure) |
| `commodity` *(no code)* | `mode: "commodity"`, `commodities: dict` (STRATEGIC_COMMODITIES map) |
| `partners` | `mode: "partners"`, `reporter: str`, `flow: str`, `record_count: int`, `records: list[...]` (same record structure) |

---

## 22. sovereign_debt

**FILE:** `sovereign_debt.py`
**MODES:** `us_yields`, `eu_yields`, `jp_yields`, `uk_gilts`, `spreads`

| Mode | DATA KEYS |
|------|-----------|
| `us_yields` | `month: str`, `entries: int`, `records: list[{date, yields: {1m, 2m, 3m, ...30y}, curve_2s10s, curve_3m10y}]` |
| `eu_yields` | `countries: dict[cc → list[{period, yield_pct}]]`, `errors: list\|None`, `period_start: str` |
| `jp_yields` | `entries: int`, `records: list[{date, yields: {1y, 2y, ...40y}}]` |
| `uk_gilts` | `total_auctions: int`, `records: list[{date, ...auction fields}]` |
| `spreads` | `de_benchmark_yield: float`, `spreads: list[{country, yield_pct, de_yield_pct, spread_vs_de, period}]`, `us_curve: {date, curve_2s10s, curve_3m10y}\|None` |

---

## 23. macro_data

**FILE:** `macro_data.py`
**SOURCES:** `fred`, `ecb`, `world_bank`

| Source | DATA KEYS |
|--------|-----------|
| `fred` | `dict[series_id → list[{date: str, value: str}]]` |
| `ecb` | `dict[series_id → list[{date: str, value: str}]]` |
| `world_bank` | `dict[country_code → list[{date, value, country_code, ...}]]` |

---

## 24. market_data

**FILE:** `market_data.py`
**MODES:** *(single mode)* — params: `tickers`, `period`, `interval`

**DATA KEYS:** `dict[ticker → list[{Date/Datetime, Open, High, Low, Close, Volume}]]`

(yfinance DataFrame converted to records via `to_dict(orient="records")`)

---

## 25. energy_supply

**FILE:** `energy_supply.py`
**MODES:** `petroleum_stocks`, `petroleum_supply`, `rig_count`

| Mode | DATA KEYS |
|------|-----------|
| `petroleum_stocks` | `series: dict[name → list[{period, value, ...}]]`, `signals: dict[name → dict]`, `label: "petroleum_stocks"`, `weeks: int` |
| `petroleum_supply` | `series: dict[name → list[{period, value, ...}]]`, `signals: dict[name → dict]`, `label: "petroleum_supply"`, `weeks: int` |
| `rig_count` | `records: list[dict]`, `count: int`, `months: int`, `signals: dict` |

---

## 26. supply_chain_prices

**FILE:** `supply_chain_monitor.py`
**MODES:** `producer_prices`, `import_prices`, `pressure_index`

| Mode | DATA KEYS |
|------|-----------|
| `producer_prices` | `mode: "producer_prices"`, `months: int`, `series: dict`, `signals: dict` |
| `import_prices` | `mode: "import_prices"`, `months: int`, `series: dict`, `signals: dict` |
| `pressure_index` | `mode: "pressure_index"`, `months: int`, `ppi_signals: dict`, `import_signals: dict`, `pressure: dict` |

---

## OUTPUT-ONLY TOOLS (no structured `data` dict)

These tools return `ToolResult(success=True, output=str)` without a `data=` parameter.
They use formatted text output, often cached as strings.

| # | Tool | File | Modes |
|---|------|------|-------|
| 27 | `satellite_activity` | `satellite_activity.py` | `fire`, `vegetation`, `events` |
| 28 | `foia_requests` | `foia_requests.py` | `search`, `agency_activity`, `entity_cluster` |
| 29 | `interconnection_queue` | `interconnection_queue.py` | `queue`, `summary`, `datacenter` |
| 30 | `internet_infrastructure` | `internet_infrastructure.py` | `outages`, `censorship`, `signals`, `incidents` |
| 31 | `electricity_monitor` | `electricity_monitor.py` | `demand`, `generation`, `interchange` |

---

*Generated from source code scan. Some nested structures (signals, summary dicts)
vary by data content and are described at the top-level keys only.*

## Related

- [[project_memory]]
