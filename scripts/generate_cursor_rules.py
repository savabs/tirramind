#!/usr/bin/env python3
"""Generate .cursor/rules/*.mdc from module .instructions.md files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT / ".cursor" / "rules"

MODULE_RULES: list[tuple[str, str, str, str]] = [
    (
        "agent/core/.instructions.md",
        "agent-core.mdc",
        "agent/core/**",
        "Core module conventions (Agent Layer orchestrator)",
    ),
    (
        "agent/config/.instructions.md",
        "agent-config.mdc",
        "agent/config/**",
        "Config module conventions (TIRRA_ env vars)",
    ),
    (
        "agent/pipeline/.instructions.md",
        "agent-pipeline.mdc",
        "agent/pipeline/**",
        "Pipeline module conventions (DAG executor)",
    ),
    (
        "agent/tools/.instructions.md",
        "agent-tools.mdc",
        "agent/tools/**",
        "Tools module conventions (Layer 1 surveillance)",
    ),
    (
        "agent/quant/.instructions.md",
        "agent-quant.mdc",
        "agent/quant/**",
        "Quant module conventions (Layer 2 feature engineering)",
    ),
    (
        "tests/.instructions.md",
        "tests-conventions.mdc",
        "tests/**",
        "Test conventions (pytest edge-case coverage)",
    ),
]

WORKFLOW_RULE = """---
description: TirraMind workflow enforcement and automation entry points
alwaysApply: true
---

# TirraMind Workflow (enforced)

## Pipeline order
Non-trivial work: research → spec → task → implement. Preflight scope before gate:
`docs/research/`, `docs/specs/`, `tasks/active/`, `docs/memory/` only.

## Artifacts
New research/spec/task/checkpoint/ADR content: `.html` canonical + thin `.md` stub with YAML frontmatter and `[[wiki links]]`.

## Automation (prefer over re-reading long policy docs)
- `make vault-lint` — obsidian + fact drift (also pre-commit)
- `make quality-gate-fast` — vault lint + active task step checks
- `make quality-gate` — full gate including pytest
- Session context injected via `.cursor/hooks/session_context.py`

## Agent skills (invoke with `/skill-name` in chat)
- `/full-pipeline` — research → spec → implement → review → checkpoint
- `/next-step` — one atomic task step
- `/sprint` — all remaining task steps
- `/research` — research-only, no code
- `/debug` — structured debug protocol
- `/session-start` — manual cold-start (hook injects basics automatically)
- `/spec-to-task` — spec → tasks/active file
- `/brainstorm-to-spec` — fuzzy idea → bounded plan
- `/review-quant` — quant module review
- `/extract-learnings` — mine done tasks → project memory
- `/post-mortem` — retrospective after hard failures

Regenerate: `make cursor-skills` (from `.github/prompts/`)

## MCP tools (external verification — Phase 4)
Bootstrap: `make mcp-setup` → edit `.cursor/mcp.json` with API keys → restart Cursor.
- **tavily** — web search (zero-hallucination rule; see `copilot-instructions.md` Internet Research Protocol)
- **wolfram-alpha** — symbolic math verification (`WOLFRAM_APP_ID` in `.env`, synced via `make mcp-setup`)
- **context7** — library docs before coding unfamiliar APIs
- **sequential-thinking** — multi-step math derivations
- **git** — structured repo queries (`pip install mcp-server-git`)
- **playwright** — browser automation for data tools
- **memory** — persistent graph in `.tirra_memory/`

Template (no secrets): `.cursor/mcp.json.example`

## Canonical policy index
- `AGENTS.md` — agent roles and doctrine
- `.github/copilot-instructions.md` — full workflow spec
"""


def _mdc_body(source: Path, description: str, globs: str) -> str:
    content = source.read_text(encoding="utf-8").strip()
    return (
        "---\n"
        f"description: {description}\n"
        f"globs: {globs}\n"
        "alwaysApply: false\n"
        "---\n\n"
        f"{content}\n"
    )


def main() -> None:
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    for rel_src, out_name, globs, description in MODULE_RULES:
        source = ROOT / rel_src
        if not source.exists():
            raise FileNotFoundError(source)
        (RULES_DIR / out_name).write_text(
            _mdc_body(source, description, globs),
            encoding="utf-8",
        )

    (RULES_DIR / "tirramind-workflow.mdc").write_text(WORKFLOW_RULE.strip() + "\n", encoding="utf-8")
    print(f"Generated {len(MODULE_RULES) + 1} rules in {RULES_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
