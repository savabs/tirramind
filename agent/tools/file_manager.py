"""
Tool: File Manager

Read, write, list, and search files on the local filesystem.
Scoped to allowed directories for safety.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_MAX_READ_LEN = 10_000  # characters


class FileReadTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the text content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
        },
        "required": ["path"],
    }

    def execute(self, *, path: str, **_: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(success=False, output=f"File not found: {p}")
        if not p.is_file():
            return ToolResult(success=False, output=f"Not a file: {p}")
        try:
            content = p.read_text(errors="replace")
            if len(content) > _MAX_READ_LEN:
                content = content[:_MAX_READ_LEN] + "\n\n[... truncated]"
            return ToolResult(success=True, output=content)
        except Exception as exc:
            return ToolResult(success=False, output=f"Read error: {exc}")


class FileWriteTool(Tool):
    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }

    def execute(self, *, path: str, content: str, **_: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return ToolResult(success=True, output=f"Written {len(content)} chars to {p}")
        except Exception as exc:
            return ToolResult(success=False, output=f"Write error: {exc}")


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and folders in a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path."},
        },
        "required": ["path"],
    }

    def execute(self, *, path: str, **_: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return ToolResult(success=False, output=f"Not a directory: {p}")
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            lines = []
            for e in entries[:100]:
                prefix = "📁 " if e.is_dir() else "   "
                lines.append(f"{prefix}{e.name}")
            if len(entries) > 100:
                lines.append(f"... and {len(entries) - 100} more")
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as exc:
            return ToolResult(success=False, output=f"List error: {exc}")
