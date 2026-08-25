---
title: "Feature: 7b-W — Drug/Medical Regulatory (OpenFDA)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/drug-regulatory
---

# Feature: 7b-W — Drug/Medical Regulatory (OpenFDA)

## Current Architecture
- **Layer:** 1 (Surveillance Surface) — `agent/tools/`
- **Pattern:** Tool inherits from `Tool` base class, uses `httpx` for HTTP, `DataCache` for caching
- **Existing overlap:** `disease_surveillance.py` covers WHO/CDC disease data; this tool targets pharmaceutical regulatory data (FDA approvals, adverse events, labeling)
- **Registration:** `agent/cli.py` (import + `registry.register()`), `agent/learning/bandit.py` (GoalArm)

## Data Source: OpenFDA API

- **API:** `https://api.fda.gov/`
- **Auth:** Optional API key (free). Without key: 240 req/min, 1000 req/day per IP. With key: 240 req/min, 120K req/day.
- **License:** US government public data. Terms of service at open.fda.gov/terms/. Standard government disclaimer.
- **Format:** JSON (Elasticsearch-based query syntax)
- **Update frequency:** Varies by endpoint — FAERS quarterly, Drugs@FDA daily (M-F)

### Key Endpoints

1. **Drugs@FDA (Approvals):** `https://api.fda.gov/drug/drugsfda.json`
   - Coverage: 1939 to present, daily updates (M-F)
   - Fields: `application_number`, `sponsor_name`, `products[].brand_name`, `products[].dosage_form`, `submissions[].submission_type` (ORIG/SUPPL), `submissions[].submission_status_date`, `submissions[].review_priority` (STANDARD/PRIORITY)
   - Signal: New drug approvals (NDA/BLA) = direct impact on pharma company stock. Priority reviews = accelerated timeline. Supplemental approvals = label expansions (new indications = revenue growth).

2. **Drug Adverse Events (FAERS):** `https://api.fda.gov/drug/event.json`
   - Coverage: 2004 to present, quarterly updates
   - Fields: `patient.drug[].medicinalproduct`, `patient.reaction[].reactionmeddrapt`, `receivedate`, `serious` (1=yes), `seriousnessdeath`, `companynumb`
   - Signal: Adverse event spikes = safety signal → potential FDA action (black box warning, REMS, recall). Pharma stock crash catalyst. Class-action litigation leading indicator.

3. **Drug Labeling:** `https://api.fda.gov/drug/label.json`
   - Coverage: SPL label data, updated daily
   - Fields: `openfda.brand_name`, `openfda.generic_name`, `warnings`, `boxed_warning`, `indications_and_usage`
   - Signal: Label changes (new warnings, boxed warning additions) = regulatory risk for manufacturer.

### Query Syntax (Elasticsearch)
- `search=field:value` — basic term search
- `search=field:value+AND+field2:value2` — boolean AND
- `search=receivedate:[20260101+TO+20260331]` — date ranges
- `count=field` — return faceted counts (e.g., top drugs by adverse events)
- `limit=100` — max results per request (max 1000 for count)
- `skip=0` — pagination offset

### Harmonized Fields
OpenFDA harmonizes across datasets using: `manufacturer_name`, `brand_name`, `generic_name`, `product_type`, `route`, `substance_name`, `application_number`, `product_ndc`

## Signal Theory

Drug regulatory data is **asymmetric information gold** for pharma-sector signals:
- **New approvals** are binary events with massive stock impact (5-30% moves). FDA approval calendars are public, but real-time monitoring of the API catches supplemental approvals and label expansions that slip under the radar.
- **Adverse event surges** are leading indicators for: safety recalls, clinical holds, black-box warnings, lawsuits. The lag between FAERS signal and market reaction can be weeks to months.
- **Label changes** (especially new warnings/contraindications) directly affect prescribing and revenue. Monitoring label.json catches these before press releases.
- **Priority review designations** signal which drugs the FDA considers addressing unmet medical needs — affects competitive landscape.

Pharma stocks are uniquely event-driven. A single FDA decision can move a $100B company by 10%. This tool monitors the full lifecycle: approval → labeling → adverse events.

## Observations
- Elasticsearch query syntax is well-documented
- No auth required for basic use (240/min is generous)
- FAERS data uses ICH E2B/M2 standard — structured, machine-readable
- Drugs@FDA harmonization enables cross-referencing events ↔ approvals ↔ labels via `application_number`
- Historical depth: approvals since 1939, events since 2004

## Risks
- FAERS data has known quality issues: duplicate reports, lag in reporting, inconsistent drug name spelling
- Rate limiting without API key (1000/day) could be hit in heavy use
- Quarterly FAERS updates mean stale data between releases
- MedDRA term mapping is complex (preferred terms vs lower-level terms)
- API occasionally returns 500s during heavy load

## Data Requirements
- Recent drug approvals (Drugs@FDA) with submission type, priority, sponsor
- Adverse event counts by drug, reaction type, seriousness
- Label change detection (boxed warnings, new contraindications)
- Date-range queries for trend analysis

## Math/Algorithm Survey
- Adverse event count anomaly detection: Z-score vs historical baseline per drug
- Approval rate momentum: rolling count of approvals per sponsor/therapeutic area
- Seriousness ratio: (serious events / total events) per drug — spike = safety signal
- New molecular entity (NME) tracking vs supplemental approvals
- All computed internally; no external math libs needed beyond basic stats

## OSS/External Research
- `openfda` Python package exists but is minimal and not maintained
- Direct API access is simpler and more reliable than any wrapper
- No license conflicts — US government data, public domain
- Search terms used: "openfda python", "FDA FAERS API", "drug adverse events API"

---

## Related

- [[7b-W_drug_regulatory_spec|Spec: 7B-W Drug Regulatory]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
