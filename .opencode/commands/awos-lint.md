---
description: Run ruff linting on the scaffold codebase
agent: awos
---

Run ruff lint checks on the AWOS codebase:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && ruff check scaffold/ 2>&1 | tail -40`

If there are issues, suggest fixes. Do NOT make changes unless explicitly asked.
