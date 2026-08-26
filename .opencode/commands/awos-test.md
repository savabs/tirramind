---
title: "AWOS Test Command"
tags:
  - tool/opencode
description: Run AWOS test suite with coverage
agent: awos
---

Run the AWOS test suite:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && python -m pytest tests/ -v --tb=short -m "not integration" 2>&1 | tail -60`

Analyze the results. If there are failures:
1. Identify the root cause
2. Suggest targeted fixes
3. Do NOT make changes unless explicitly asked
