---
title: "Task: 7b-AI_internet_infrastructure"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/internet-infrastructure
---

# Task: 7b-AI_internet_infrastructure

Status: completed
Research: [[7b-AI_internet_infrastructure]]
Spec: [[7b-AI_internet_infrastructure_spec]]

## Goal

Establish a commercially safe implementation path for global internet-infrastructure anomaly monitoring.

## Scope Notes

- Layer: Layer 1 — Surveillance Surface
- Main files expected to change:
  - `agent/tools/internet_infrastructure.py`
  - `agent/cli.py`
  - `agent/learning/bandit.py`
  - `tests/test_internet_infrastructure_edge.py`
- Non-goals:
  - No implementation against non-commercial or unclear-license providers
  - No Phase 7c coincidence engine work in this task

## Steps

- [x] 1.1: Confirm provider legality and source viability
  Verification: IODA (public API, NSF-funded) ✅ + OONI (CC BY 4.0) ✅. Cloudflare Radar (CC BY-NC) ❌. RIPEstat (commercial unclear) ❌. Two approved providers.
- [x] 1.2: Define normalized evidence schema and tool modes
  Verification: spec rewritten with 4 concrete modes (outages, censorship, signals, incidents), normalized schema, cache TTLs.
- [x] 1.3: Implement `InternetInfrastructureTool` skeleton with validated params
  Verification: tool created with full parameter validation, mode routing, cache integration.
- [x] 1.4: Implement IODA + OONI provider adapters
  Verification: all 4 modes implemented with IODA alerts/events/signals + OONI aggregation/incidents.
- [x] 1.5: Implement anomaly scoring for visibility / routing changes
  Verification: gtr-norm severity thresholds (WARNING<0.80, CRITICAL<0.50), trend analysis, HEAVY BLOCKING detection.
- [x] 1.6: Register tool and add focused bandit arm
  Verification: registered in cli.py, bandit arm updated to include both internet_infrastructure and internet_outages tools.
- [x] 1.7: Add exhaustive edge-case suite and run live smoke validation
  Verification: 117 tests pass. Live smoke test confirms all 4 modes return valid data (8 IODA alerts, IR censorship 13.6% RISING, US signals NORMAL, 62 OONI incidents).

## Completion Checklist

- [x] Research note exists and is current
- [x] Spec matches the actual implementation plan
- [x] Each completed step has a verification result
- [x] Edge-case testing was added and run for code changes
- [x] Checkpoint written at the end of the session or sub-phase

## Notes

- The provider/legal gate is mandatory. Do not start runtime code until at least one provider is commercially safe.
- Treat Cloudflare Radar and RIPEstat as concept-only unless explicit commercial clearance is documented.

---

## Related

- [[7b-AI_internet_infrastructure|Research: 7B-Ai Internet Infrastructure]]
- [[7b-AI_internet_infrastructure_spec|Spec: 7B-Ai Internet Infrastructure]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
