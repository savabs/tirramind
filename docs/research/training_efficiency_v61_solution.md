---
title: "V61 Training Efficiency Solution"
tags:
  - doc/research
  - phase/50
  - topic/gnn-training
  - topic/embedding-collapse
  - topic/multi-task-learning
  - status/active
---

# V61 Training Efficiency Solution

**Primary content:** [[training_efficiency_v61_solution.html]]

## Problem
V60 showed increased embedding collapse (48.4% vs 23.2%) and flat return loss despite VICReg. Negative IC (-0.0440) far below V52 baseline (+0.0474).

## Solution Components
1. **ContraNorm layer** — Architectural anti-collapse (Guo et al. PKU 2023)
2. **Log loss transformation** — Multi-task balancing (Lin et al. HKUST 2026)
3. **PCGrad** — Gradient surgery for conflicting tasks (Yu et al. NeurIPS 2020)
4. **Reduced value_weight** — 0.3 → 0.1 to prevent dominance

## Related
- [[LESSONS]]
- [[quant_training_ground]]
- [[research_gnn_training_efficiency]]
- [[V60_analysis]]
