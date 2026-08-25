---
title: "Spec: Bloomberg Non-Commodity Extraction"
tags:
  - doc/spec
---

# Spec: Bloomberg Non-Commodity Extraction

## Goal

Translate the useful, non-commodity lessons from institutional Bloomberg workflows into implementation-ready TirraMind artifacts without drifting into terminal-clone work.

This spec does not propose rebuilding Bloomberg functions. It defines the narrow capabilities worth extracting:
- a compact daily context object
- execution-quality features for thin / event-driven markets
- a rule for evaluating Bloomberg-inspired ideas against TirraMind's moat strategy

## Research

See:
- `[[bloomberg_workflow_noncommodity]]`
- `[[daily_context_schema]]`
- `[[execution_quality_feature_schema]]`

## Files Affected

### New files
- `[[bloomberg_workflow_noncommodity]]`
- `[[daily_context_schema]]`
- `[[execution_quality_feature_schema]]`
- `[[bloomberg_noncommodity_extraction_spec]]`

### Future implementation files (not part of this documentation-only step)
- `agent/pipeline/dags/` daily context DAG definition
- `agent/quant/` context compression / execution modeling modules
- `agent/tools/` venue-specific collection or query tools as needed

## Implementation Steps

### Step 1: Define daily context schema
- Create a machine-first schema for global state compression.
- Include cross-asset moves, rates/liquidity state, event clusters, and regime flags.
- Ensure the object is deterministic, versioned, and suitable for pipeline storage.
- Test plan: synthetic payload validates against schema and supports missing / partial source coverage.

### Step 2: Define execution-quality feature schema
- Create a venue-agnostic feature contract for execution quality in thin or event-driven markets.
- Cover spread, slippage, fill probability, book imbalance, depth sufficiency, impact proxy, and implementation shortfall.
- Make the schema usable for prediction-market and low-liquidity execution analysis.
- Test plan: synthetic examples for liquid, thin, and stressed books produce valid feature rows.

### Step 3: Add non-commodity evaluation rule to future planning
- Use a decision filter for any Bloomberg-inspired capability:
  - Does it improve unique-signal extraction?
  - Does it improve world-model evidence quality?
  - Does it improve execution quality?
  - If not, it is probably commoditized.
- Apply this rule before new research or implementation proposals are added to active tasks.
- Test plan: classify a sample set of Bloomberg-like ideas into build vs avoid categories.

### Step 4: Design a daily context pipeline stage
- Add a future DAG stage that collects and compresses a daily macro / market / event context object.
- The output should be a stored evidence artifact for downstream world-model consumption, not a UI dashboard.
- Test plan: one daily run produces a versioned context object with source lineage and timestamps.

### Step 5: Design execution-quality modeling for target venues
- Add venue-specific mappings from raw market microstructure data into the execution-quality feature schema.
- Prioritize prediction markets and thin books where execution friction can erase modeled edge.
- Test plan: replay known thin-book scenarios and verify that estimated execution cost rises under stress.

## Edge Cases
- Missing source coverage on a given day
- Delayed or stale macro inputs
- Venue feeds with partial book depth
- Event-driven gaps where pre-trade reference price is unstable
- Markets with zero quoted spread but negligible real size
- Schema evolution over time requiring version bumps

## Testing Plan
- Schema validation tests for the daily context object
- Schema validation tests for execution-quality feature rows
- Replay tests on synthetic stressed books and event spikes
- Regression tests for schema version compatibility
- Planning-rule tests that reject clearly commoditized feature proposals

## Non-Goals
- Rebuilding Bloomberg terminal screens
- Creating analyst-facing dashboard clones
- Building ranked generic-news terminals on common feeds
- Reproducing Bloomberg messaging / network identity

## Related

- [[bloomberg_workflow_noncommodity|Research: Bloomberg Workflow Noncommodity]]
