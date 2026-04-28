"""Atomic-write and markdown section helpers."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SELF_MARKER = "<!-- awos:self -->"


def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Uses a same-directory temp file + ``os.replace``. Preserves encoding.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


# --- markdown sections ---------------------------------------------------
# A section is demarcated by an ATX header ("# ...", "## ...", ...) at
# column 0 (or after blank/YAML frontmatter). We operate on sections by
# exact title match (case-sensitive, trimmed).
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def find_section(body: str, title: str) -> tuple[int, int] | None:
    """Return (start_line_idx, end_line_idx_exclusive) for a section.

    ``start_line_idx`` is the header line; ``end_line_idx_exclusive`` is
    the first line belonging to the *next* section of the same level or
    shallower. If not found returns None.
    """
    lines = body.splitlines()
    start: int | None = None
    header_level = 0
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        this_title = m.group(2).strip()
        if start is None and this_title == title.strip():
            start = i
            header_level = level
            continue
        if start is not None and level <= header_level:
            return (start, i)
    if start is not None:
        return (start, len(lines))
    return None


def append_to_section(
    body: str, title: str, content_to_add: str, *, dedup: bool = True
) -> str:
    """Append ``content_to_add`` to the given section.

    If the section doesn't exist, append a new ``## title`` at the end.
    If ``dedup`` is True and ``content_to_add`` already appears verbatim
    inside the section, the body is returned unchanged.
    """
    loc = find_section(body, title)
    content_to_add = content_to_add.rstrip() + "\n"
    if loc is None:
        tail = body.rstrip() + f"\n\n## {title}\n\n{content_to_add}\n"
        return tail
    start, end = loc
    lines = body.splitlines(keepends=True)
    section = "".join(lines[start:end])
    if dedup and content_to_add.strip() and content_to_add.strip() in section:
        return body
    # strip trailing blanks in section, then append, then keep one blank
    section_stripped = section.rstrip() + "\n\n"
    new_section = section_stripped + content_to_add + "\n"
    return "".join(lines[:start]) + new_section + "".join(lines[end:])


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_LAST_UPDATED_RE = re.compile(
    r"(Last updated:\s*)[\d\-T:Z UTC]+.*",
    re.IGNORECASE,
)


def bump_last_updated(body: str) -> str:
    """Replace the 'Last updated:' line with the current UTC timestamp.

    Preserves the version suffix (e.g. '— v1.2') if present.
    If the pattern isn't found, inserts a line after the first H1.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _replacer(m: re.Match) -> str:  # type: ignore[type-arg]
        return m.group(1) + now

    new_body, n = _LAST_UPDATED_RE.subn(_replacer, body)
    if n:
        return new_body

    # No existing line — insert after first H1
    lines = body.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines.insert(i + 1, f"\n> Last updated: {now}\n")
            return "".join(lines)
    return body


def append_changelog(body: str, line: str) -> str:
    """Append a dated entry to the AWOS '11. Changelog' section.

    Ensures the marker is present so self-written lines don't loop back
    through the classifier.
    """
    ts = now_iso()
    entry = f"- {ts} {SELF_MARKER} {line.strip()}"
    return append_to_section(body, "11. Changelog", entry, dedup=False)


__all__ = [
    "SELF_MARKER",
    "atomic_write",
    "append_changelog",
    "append_to_section",
    "bump_last_updated",
    "find_section",
    "now_iso",
]
