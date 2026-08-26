---
title: "AWOS Stats Command"
tags:
  - tool/opencode
description: Show AWOS token usage and cost statistics
agent: awos
---

Show AWOS self-learning observability stats:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && python awos.py stats 2>&1`

Also show budget status and savings:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && python awos.py stats --savings 2>&1 && python awos.py budget 2>&1`

Summarize: MTD spending, savings from model routing, session count, and any anomalies.
