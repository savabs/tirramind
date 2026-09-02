---
title: "Spec: Open Prediction Ledger"
tags:
  - doc/spec
  - topic/interconnection
  - topic/site
  - status/active
---

# Spec: Open Prediction Ledger

Research: [[interconnection_queue_attrition]]

## Purpose

A forward record of probabilistic calls that cannot be edited after the fact.
Implemented in `savabs/queue_attrition` as `src/registry.py`, surfaced publicly
at <https://tirramind.com/predictions>.

## Schema

One row per call, thirteen fields:

| field | written | meaning |
|---|---|---|
| `predicted_at` | at call | UTC timestamp, ISO-8601 |
| `domain` | at call | prediction family, e.g. `interconnection` |
| `entity_id` | at call | composite identity; `Queue ID` alone is not unique |
| `entity_name` | at call | human label, blank where the ISO publishes none |
| `p` | at call | probability of the positive outcome |
| `model` | at call | model ID, e.g. `logistic-v1` |
| `features_json` | at call | the inputs the probability was computed from |
| `thesis` | at call | free text, optional |
| `resolves_by` | at call | ISO-8601 deadline; never earlier than the call |
| `outcome_source` | at call | the named referee, fixed before the outcome |
| `outcome` | at resolution | `open`, `1` (hit) or `0` (miss) |
| `resolved_at` | at resolution | UTC timestamp |
| `resolution_note` | at resolution | free text |

## Enforced guarantees

These are refusals in code, not conventions:

1. **Append-only.** A second call on an existing `(domain, entity_id)` is
   rejected. Probabilities cannot be revised as odds change.
2. **Write-once outcomes.** A resolved row cannot be re-resolved.
3. **Mandatory falsifiability.** `resolves_by` and `outcome_source` are
   required at call time. `resolves_by` is floored at
   `predicted_at + MIN_HORIZON_YEARS` (2), because a deadline that has already
   elapsed makes the call unfalsifiable.
4. **Named referee.** `outcome_source` is fixed before the outcome exists, so
   a friendly source cannot be chosen afterwards.

## Scoring

`score()` reports Brier against the **base rate**, never against zero, plus
per-bucket calibration with Wilson intervals. `report()` warns explicitly below
n=20 rather than printing a number that looks like a result.

## Resolution mechanism

Interconnection calls resolve when the originating ISO reports a terminal
status, observed in `data/snapshots/`. Operators overwrite queue state in
place, so the daily snapshot archive is the referee — a source that can be
rewritten would make the record worthless.

## Exceptions

Regenerating the ledger is permitted on its creation day only, with nothing
resolved and nothing published, and the superseded file retained. Thereafter a
modelling change gets a new `model` ID, not a rewritten file.
