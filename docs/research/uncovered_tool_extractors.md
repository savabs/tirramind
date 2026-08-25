---
title: "Research: Uncovered Tool Extractors"
tags:
  - doc/research
---

# Research: Uncovered Tool Extractors

## Goal
Wire 3 existing data tools into the convergence engine as extractors. These tools are fully implemented with free APIs but produce no convergence signals today — hidden edge sitting unused.

## Current State
- 46 registered extractors covering 46 tools
- 5 data tools with NO extractor: labor_disruptions, gov_contracts, academic_preprints, internet_outages, migration_flows
- supply_chain_monitor confirmed as alias for supply_chain_prices (already covered)

## Selected Tools (ranked by signal value × uniqueness)

### 1. labor_disruptions (BLS Work Stoppages)
**Source:** BLS Public Data API v2 (free, no auth)
**Data:** Monthly workers involved (thousands) + days idle (thousands) in major stoppages (1,000+ workers)
**Tool output shape:**
```
{
  "signals": {
    "latest_value": float,        # thousands of workers / idle days
    "period_average": float,
    "period_peak": float,
    "active_months": int,
    "active_pct": float,
    "trend": str,                 # ESCALATING / RISING / DECLINING / STABLE / QUIET
    "trend_ratio": float | None,  # recent-6-mo / prior-6-mo average
    "alert": str | None,          # CRITICAL / WARNING / NOTICE
  }
}
# Overview mode adds:
{
  "signals": {
    "workers": { ... single signals ... },
    "idle_days": { ... single signals ... },
    "intensity_ratio": float | None,
    "consecutive_active_months": int,
    "combined_alert": str | None,
  }
}
```
**Signal uniqueness:** T0 physical-world observable — workers physically stopping. Nobody in quant finance monitors BLS strike data for convergence signal.
**Template impact:** 9 templates already reference `jobs\.` signals in behavioral_intent. labor_disruptions adds a DIFFERENT signal — not job openings, but strike/stoppage activity. Creates new `strike.*` signal prefix.
**Category mapping:** behavioral_intent (workers acting on grievance) + macro_momentum (production disruption)

### 2. gov_contracts (USASpending.gov + UK Contracts Finder)
**Source:** USASpending.gov POST API + UK Contracts Finder OCDS (both free, no auth)
**Data:** Federal/public contract awards with amounts, agencies, recipients
**Tool output shape:**
```
{
  "awards": [
    {
      "award_id": str,
      "recipient": str,
      "amount_usd": float,
      "agency": str,
      "sub_agency": str,
      "award_type": str,
      "start_date": str,
      "end_date": str,
      "description": str,
    }, ...
  ],
  "total": int,
  "count": int,
  "region": "us" | "uk",    # UK only
}
```
**Signal uniqueness:** Government procurement data leaks fiscal intent 3-12 months ahead. Defense contract surges precede geopolitical posturing. New agency spending patterns signal policy direction before announcements.
**Template impact:** 6 templates reference lobbying signals (behavioral_intent). gov_contracts adds award-level fiscal signals: total spend, concentration, defense share, agency patterns.
**Category mapping:** regulatory_action (government spending = policy action)

### 3. academic_preprints (arXiv + ClinicalTrials.gov)
**Source:** arXiv Atom XML API + ClinicalTrials.gov v2 API (both free, no auth)
**Data:** Research papers (arXiv) + clinical trial registrations/completions (ClinicalTrials.gov)
**Tool output shape:**
```
# papers/trending mode:
{
  "papers": [
    {"arxiv_id": str, "title": str, "authors": [...], "categories": [...], "published": str, "summary": str},
    ...
  ],
  "total_results": int,
  "count": int,
}
# trials mode:
{
  "studies": [
    {"nct_id": str, "title": str, "status": str, "conditions": [...], "interventions": [...], "sponsor": str, "sponsor_class": str},
    ...
  ],
  "total_count": int,
  "count": int,
}
```
**Signal uniqueness:** ClinicalTrials.gov Phase III completions precede FDA filings. ArXiv category surges in cs.AI or q-fin signal paradigm shifts. Patent → paper → product chain.
**Template impact:** 3 templates reference `fda\.` or `drug_regulatory\.` (drug_safety_crisis, pharma_pipeline_collapse, stealth_accumulation). Clinical trials data provides LEADING signal ahead of FDA actions.
**Category mapping:** behavioral_intent (research = intent) + biological (clinical trials) + regulatory_action (trial status)

## Rejected Tools (lower priority for now)
- **internet_outages** — Overlaps substantially with internet_infrastructure extractor (same OONI data). Defer.
- **migration_flows** — UNHCR data is very slow-moving (annual). Low-frequency signal doesn't fit 30-45 day template windows well. Defer.

## Signal Design

### labor_disruptions → Extractor signals
| signal_id | category | direction logic | confidence |
|-----------|----------|-----------------|------------|
| strike.us.workers_involved | behavioral_intent | +1 if ESCALATING/RISING, -1 if DECLINING, 0 if STABLE/QUIET | 0.75 |
| strike.us.idle_days | macro_momentum | +1 if ESCALATING/RISING (more disruption), -1 if DECLINING | 0.70 |
| strike.us.intensity | macro_momentum | +1 if intensity_ratio > 1 (longer disputes), -1 if < 1, 0 if None | 0.65 |
| strike.us.consecutive_months | behavioral_intent | +1 if consecutive_active_months >= 3, 0 otherwise | 0.70 |

### gov_contracts → Extractor signals
| signal_id | category | direction logic | confidence |
|-----------|----------|-----------------|------------|
| gov_contract.us.total_value | regulatory_action | +1 if total_value > threshold, -1 if below | 0.70 |
| gov_contract.us.award_count | regulatory_action | +1 if growing, -1 if shrinking | 0.65 |
| gov_contract.us.defense_share | geopolitical | +1 if defense > 50% of awards by value, 0 otherwise | 0.75 |
| gov_contract.uk.award_count | regulatory_action | +1 if growing, -1 if shrinking | 0.60 |

### academic_preprints → Extractor signals
| signal_id | category | direction logic | confidence |
|-----------|----------|-----------------|------------|
| trials.total_active | biological | +1 if many active trials, value = count | 0.65 |
| trials.phase3_completions | regulatory_action | +1 if completions detected (FDA filing imminent) | 0.80 |
| trials.sponsor_concentration | behavioral_intent | +1 if industry-sponsored > academic (commercial intent) | 0.60 |
| arxiv.market_relevant.volume | behavioral_intent | +1 if trending up, value = total results | 0.50 |

## Risks
- BLS work stoppage data is monthly — may be stale. TTL should be long (30 days).
- USASpending API has slow response times. Must handle timeouts gracefully.
- ClinicalTrials.gov doesn't reliably report phase. Must parse from title/study_type.
- ArXiv signal is very noisy — low confidence score appropriate.

## References
- BLS Work Stoppages: https://www.bls.gov/wsp/
- USASpending.gov API: https://api.usaspending.gov/
- ClinicalTrials.gov API v2: https://clinicaltrials.gov/data-api/api
- arXiv API: https://info.arxiv.org/help/api/index.html

---

## Related

- [[uncovered_tool_extractors_spec|Spec: Uncovered Tool Extractors]]
