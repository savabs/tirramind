"""
Tool: Shell Runner

Execute shell commands with timeout, output capture, and filtered environment.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from typing import Any

from agent.tools.base import Tool, ToolResult
from agent.tools.code_executor import _safe_env

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_LEN = 8_000

# Commands that are never allowed
_BLOCKED_PATTERNS = frozenset({
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "shutdown", "reboot", "halt", "poweroff",
})


class ShellRunnerTool(Tool):
    name = "run_shell"
    description = (
        "Execute a shell command and return stdout/stderr. "
        "Use this for file operations, system queries, git commands, package management, etc. "
        "Dangerous commands (rm -rf /, mkfs, etc.) are blocked. "
        "Runs with filtered environment (API keys not exposed)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory (optional, defaults to temp dir).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def execute(self, *, command: str, working_dir: str | None = None, **_: Any) -> ToolResult:
        # Safety check
        cmd_lower = command.lower().strip()
        for pattern in _BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return ToolResult(
                    success=False,
                    output=f"Blocked: command matches dangerous pattern '{pattern}'",
                )

        # Default to temp dir if no working_dir specified
        cwd = working_dir or tempfile.gettempdir()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=cwd,
                env=_safe_env(),
            )
            stdout = result.stdout[:_MAX_OUTPUT_LEN] if result.stdout else ""
            stderr = result.stderr[:_MAX_OUTPUT_LEN] if result.stderr else ""

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"STDERR: {stderr}")
            output_parts.append(f"Exit code: {result.returncode}")

            return ToolResult(
                success=result.returncode == 0,
                output="\n".join(output_parts),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=f"Command timed out after {self._timeout}s")
        except Exception as exc:
            log.exception("Shell command failed")
            return ToolResult(success=False, output=f"Shell error: {exc}")
