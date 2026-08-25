---
title: "Spec: M1 Continuous-Time Heterogeneous World Model"
tags:
  - doc/spec
  - phase/moonshot
  - topic/world-model
  - topic/neural-sde
  - topic/continuous-time
  - layer/learning
  - status/research
---

> Primary doc: [[m1_continuous_world_model.html]]

# Spec: M1 — Continuous-Time Heterogeneous World Model

Neural SDE on a Heterogeneous Event Graph.
Formula: `dX_i(t) = f_theta(X_i, {X_j}_{N(i)}, Z(t)) dZ(t) + g_phi(X_i) dW_i(t)`

**Status:** Research/spec phase. No implementation until spec reviewed.

## Related
- [[world_model_spec]] — Phase 9 Bayesian world model (DAG + Kalman); M1 complements not replaces
- [[world_model_bridge_spec]] — GNN to world model bridge
- [[temporal_het_gnn_spec]] — HetTGN foundation
- [[learning_stack_spec]] — Full learning stack architecture
