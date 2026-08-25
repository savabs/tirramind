#!/usr/bin/env python3
"""Migrate .github/prompts/*.prompt.md to .cursor/skills/*/SKILL.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / ".github" / "prompts"
SKILLS_DIR = ROOT / ".cursor" / "skills"

# Extra trigger hints appended to descriptions for skill discovery.
TRIGGER_HINTS: dict[str, str] = {
    "brainstorm-to-spec": "Use when the idea is fuzzy and needs decomposition before research or spec.",
    "debug": "Use when tests fail, fixes loop, or root cause is unclear.",
    "extract-learnings": "Use after completing features to mine tasks/done and update project memory.",
    "full-pipeline": "Use for a new non-trivial feature end-to-end: research, spec, implement, review, checkpoint.",
    "next-step": "Use to execute exactly one unchecked step from the active task file.",
    "post-mortem": "Use after a hard bug, failed approach, or unproductive session.",
    "research": "Use for research-only work with no code changes.",
    "review-quant": "Use when reviewing agent/quant modules for numerical stability and tests.",
    "session-start": "Use for manual cold-start beyond the automatic sessionStart hook context.",
    "spec-to-task": "Use when a spec is complete and needs an atomic tasks/active file.",
    "sprint": "Use to execute all remaining task steps in one run without pausing.",
}

LEGACY_STUB = "> Legacy reference. Use Cursor skill `/{name}` — canonical: `.cursor/skills/{name}/SKILL.md`\n\n"


def _parse_prompt(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text.strip()
    fm_block = text[3:end].strip()
    body = text[end + 4 :].strip()
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def _skill_md(name: str, description: str, body: str) -> str:
    hint = TRIGGER_HINTS.get(name, "")
    full_description = description if not hint else f"{description} {hint}"
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {full_description}\n"
        "disable-model-invocation: true\n"
        "---\n\n"
        f"{body}\n"
    )


def migrate() -> list[str]:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    for prompt_path in sorted(PROMPTS_DIR.glob("*.prompt.md")):
        name = prompt_path.stem.replace(".prompt", "")
        meta, body = _parse_prompt(prompt_path)
        description = meta.get("description", f"TirraMind workflow skill: {name}")

        skill_dir = SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            _skill_md(name, description, body),
            encoding="utf-8",
        )
        created.append(name)

        # Legacy stub on the old prompt file (idempotent)
        original = prompt_path.read_text(encoding="utf-8")
        if not original.startswith("> Legacy reference"):
            prompt_path.write_text(LEGACY_STUB.format(name=name) + original, encoding="utf-8")

    return created


def main() -> None:
    names = migrate()
    print(f"Migrated {len(names)} prompts to .cursor/skills/")
    for name in names:
        print(f"  /{name}")


if __name__ == "__main__":
    main()
