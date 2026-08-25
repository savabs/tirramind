"""Shared helpers for TirraMind Cursor hooks."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "tasks" / "active"
MEMORY_DIR = ROOT / "docs" / "memory"
CHECKPOINT_GLOB = "chat_checkpoint_*.md"


@dataclass
class TaskSummary:
    stem: str
    path: Path
    unchecked: int
    checked: int
    research: str
    spec: str

    @property
    def primary_doc(self) -> str:
        html = self.path.with_suffix(".html")
        return str(html.relative_to(ROOT)) if html.exists() else str(self.path.relative_to(ROOT))


def list_active_task_files() -> list[Path]:
    if not TASKS_DIR.exists():
        return []
    return sorted(
        p for p in TASKS_DIR.glob("*.md") if p.name != ".gitkeep"
    )


def summarize_task(path: Path) -> TaskSummary:
    content = path.read_text(encoding="utf-8", errors="replace")
    unchecked = len(re.findall(r"^- \[ \]", content, re.MULTILINE))
    checked = len(re.findall(r"^- \[x\]", content, re.MULTILINE))

    research_m = re.search(r"^Research:\s*\[\[([^\]]+)\]\]", content, re.MULTILINE)
    spec_m = re.search(r"^Spec:\s*\[\[([^\]]+)\]\]", content, re.MULTILINE)

    return TaskSummary(
        stem=path.stem,
        path=path,
        unchecked=unchecked,
        checked=checked,
        research=research_m.group(1) if research_m else "",
        spec=spec_m.group(1) if spec_m else "",
    )


def latest_checkpoint() -> Path | None:
    if not MEMORY_DIR.exists():
        return None
    candidates = list(MEMORY_DIR.glob(CHECKPOINT_GLOB))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def checkpoint_excerpt(path: Path, max_chars: int = 1200) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def run_obsidian_strict() -> tuple[bool, str]:
    script = ROOT / "scripts" / "obsidian_lint.py"
    result = subprocess.run(
        [sys.executable, str(script), "--strict", "--no-stale"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def build_session_context() -> str:
    tasks = [summarize_task(p) for p in list_active_task_files()]
    in_progress = [t for t in tasks if t.unchecked > 0]
    checkpoint = latest_checkpoint()

    lines = [
        "# TirraMind Session Context (auto-injected)",
        "",
        "## Cold-start",
        "- Canonical agent policy: `AGENTS.md`",
        "- Workflow rules: `.github/copilot-instructions.md`",
        "- Vault lint: `make vault-lint` | Quality gate: `make quality-gate-fast`",
        "",
    ]

    if checkpoint:
        rel = checkpoint.relative_to(ROOT)
        lines.extend(
            [
                f"## Latest checkpoint: `{rel}`",
                checkpoint_excerpt(checkpoint),
                "",
            ]
        )

    lines.append(f"## Active tasks ({len(tasks)} total)")
    if not tasks:
        lines.append("- (none in tasks/active/)")
    else:
        show = in_progress or tasks[:8]
        for task in show:
            status = f"{task.checked} done, {task.unchecked} remaining"
            triad = []
            if task.research:
                triad.append(f"research=[[{task.research}]]")
            if task.spec:
                triad.append(f"spec=[[{task.spec}]]")
            triad_txt = f" ({', '.join(triad)})" if triad else ""
            lines.append(f"- `{task.primary_doc}` — {status}{triad_txt}")
        if len(tasks) > len(show):
            lines.append(f"- …and {len(tasks) - len(show)} more in `tasks/active/`")

    if in_progress:
        lines.extend(
            [
                "",
                "## Suggested focus",
                "Pick one active task and execute the next unchecked step.",
                "Confirm research/spec/task triad exists before non-trivial implementation.",
            ]
        )

    return "\n".join(lines)


def collect_quality_nudges() -> list[str]:
    """Return actionable stop-time nudges (not a full task backlog dump)."""
    nudges: list[str] = []

    for path in list_active_task_files():
        task = summarize_task(path)
        if task.checked > 0 and task.unchecked == 0:
            nudges.append(
                f"- `{task.stem}`: all steps checked — run `make quality-gate-fast`, "
                f"update `status/done`, move html+md to `tasks/done/`"
            )

    lint_ok, _ = run_obsidian_strict()
    if not lint_ok:
        nudges.append(
            "- Vault lint blocking errors (FM01/FM02) — run `make vault-lint` and fix frontmatter"
        )

    return nudges
