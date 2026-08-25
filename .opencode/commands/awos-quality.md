---
description: Run quality gate checks before task completion
agent: awos
---

Run the AWOS quality gate for task completion:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && python scripts/quality_gate.py --task $ARGUMENTS 2>&1`

Also run:
- Ruff lint: `ruff check scaffold/`
- Unit tests: `python -m pytest tests/ -v --tb=short -m "not integration"`
- Obsidian lint: `python scripts/obsidian_lint.py`

Report all findings. Only mark the task complete if the quality gate passes.
