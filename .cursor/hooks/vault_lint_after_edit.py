#!/usr/bin/env python3
"""Cursor hook: run vault lint when the agent edits markdown in docs/, tasks/, or wiki/."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAULT_PARTS = frozenset({"docs", "tasks", "wiki"})
LOG_DIR = ROOT / ".cursor" / "logs"


def _is_vault_markdown(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    return bool(VAULT_PARTS & set(Path(path).parts))


def _edited_path(payload: dict) -> str:
    file_path = payload.get("file_path") or ""
    if file_path:
        return file_path

    tool_input = payload.get("tool_input") or {}
    if isinstance(tool_input, dict):
        return str(tool_input.get("path") or tool_input.get("file_path") or "")
    return ""


def _run_obsidian_lint() -> tuple[int, str]:
    script = ROOT / "scripts" / "obsidian_lint.py"
    result = subprocess.run(
        [sys.executable, str(script), "--strict", "--no-stale"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("{}")
        return

    edited = _edited_path(payload)
    if not _is_vault_markdown(edited):
        print("{}")
        return

    code, output = _run_obsidian_lint()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "vault-lint.log"
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n--- after edit: {edited} (exit {code}) ---\n{output}\n")

    hook_event = payload.get("hook_event_name", "")
    if hook_event == "postToolUse" and code != 0:
        tail = "\n".join(output.splitlines()[-20:])
        msg = (
            "Vault lint reported issues after editing a markdown file. "
            "Fix FM01/FM02 (and LK01 when promoted) before continuing:\n"
            f"{tail}"
        )
        print(json.dumps({"additional_context": msg}))
        return

    print("{}")


if __name__ == "__main__":
    main()
