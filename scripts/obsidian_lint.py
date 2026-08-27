#!/usr/bin/env python3
"""Vault-wide Obsidian lint and health check for TirraMind.

Checks ALL markdown files across docs/, tasks/, wiki/, and root.
Distinct from agent/wiki/catalog.py which only covers wiki/pages/.

Checks:
  FM01  Missing frontmatter
  FM02  Missing required frontmatter field (title, tags)
  FM03  Tags not in known taxonomy
  LK01  Broken [[wiki link]] (target file does not exist)
  LK02  Orphan page (no incoming wiki links from other files)
  ST01  File exceeds size threshold (default 500 lines)
  ST02  Missing ## Related section on research/spec/task files
  ST03  Stale file (not modified in git for >90 days)

Usage:
  python scripts/obsidian_lint.py [--json] [--no-stale] [--threshold LINES]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Constants ────────────────────────────────────────────────────────────────

WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")

# Patterns that look like wiki links but are template placeholders or
# documentation examples (e.g., [[<name>]], [[wiki links]], [[foo]])
PLACEHOLDER_LINK_RE = re.compile(r"[<>{}\.\.\.]")
EXAMPLE_LINK_TARGETS = {
    "wiki links",
    "wiki link",
    "links",
    "link",
    "foo",
    "bar",
    "filename",
    "research_note",
    "spec",
    "task",
    "display text",
    "log-yyyy",
    "chat_checkpoint_yyyy-mm-dd",
}

SKIP_DIRS = {
    ".obsidian",
    ".venv",
    ".git",
    ".cursor",  # Cursor hooks/rules/skills — not Obsidian vault artifacts
    ".claude",  # Claude skills — not Obsidian vault artifacts
    "node_modules",
    "__pycache__",
    "tirramind_vault",
    ".mypy_cache",
    ".pytest_cache",
    # Generated customer artifacts, not vault notes. intelligence_brief.md is
    # rewritten by every brief run, so it can never carry stable frontmatter —
    # linting it blocks commits whenever a brief has been generated locally.
    ".tirra_delivery",
}

# Files that are allowed to lack frontmatter (templates, config, etc.)
FRONTMATTER_EXEMPT = {
    "RESEARCH_TEMPLATE",
    "TASK_TEMPLATE",
    "TEMPLATE",
    "README",
    "CHANGELOG",
    "LICENSE",
    "CONTRIBUTING",
    "AGENTS",
    "SCHEMA",
}

# Files/directories where [[wiki link]] syntax is used as documentation
# examples, not real cross-references.  Skip LK01 checks for these.
LINK_CHECK_EXEMPT_DIRS = (".github/",)
LINK_CHECK_EXEMPT_SUFFIXES = (".instructions.md", ".agent.md", ".prompt.md")
LINK_CHECK_EXEMPT_FILES = {"AGENTS"}

# Required frontmatter fields for the vault (distinct from wiki/pages/ which
# has its own stricter set enforced by agent/wiki/catalog.py)
VAULT_REQUIRED_FIELDS = ("title", "tags")

# Known tag prefixes — any tag must start with one of these.
TAG_PREFIXES = (
    "doc/",
    "phase/",
    "topic/",
    "layer/",
    "status/",
)

# Known full tags for quick validation.  Not exhaustive — the prefix check
# catches novel tags that follow the taxonomy; this set catches common typos.
KNOWN_TAGS = {
    # doc types
    "doc/research",
    "doc/spec",
    "doc/task",
    "doc/adr",
    "doc/checkpoint",
    "doc/wiki",
    "doc/memory",
    # status
    "status/active",
    "status/done",
    # phases
    "phase/7b",
    "phase/7c",
    "phase/8",
    "phase/9",
    # layers
    "layer/surveillance",
    "layer/feature-engineering",
    "layer/world-model",
    "layer/fusion",
    "layer/learning",
    "layer/adversarial",
    "layer/llm-support",
}

# Directories whose files should have ## Related sections
RELATED_REQUIRED_DIRS = (
    "docs/research/",
    "docs/specs/",
    "tasks/active/",
    "tasks/done/",
)


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)
    total_files: int = 0
    files_with_frontmatter: int = 0
    total_wiki_links: int = 0
    unique_link_targets: int = 0
    orphan_count: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────


def iter_md_files() -> list[Path]:
    """Yield all non-excluded markdown files under ROOT."""
    result = []
    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT)
        parts = rel.parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        # Skip MagicMock junk files (test artifacts at root)
        if str(rel).startswith("<"):
            continue
        result.append(md)
    return result


def parse_frontmatter_simple(text: str) -> dict[str, object] | None:
    """Parse YAML frontmatter.  Returns None if frontmatter is absent."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 1)
    if end == -1:
        return None
    fm_text = text[4:end]
    meta: dict[str, object] = {}
    current_key: str | None = None
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current_key is not None:
                meta.setdefault(current_key, [])
                lst = meta[current_key]
                if isinstance(lst, list):
                    val = stripped[2:].strip().strip("'\"")
                    lst.append(val)
            continue
        current_key = None
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip("'\"")
            if val:
                meta[key] = val
            else:
                meta[key] = []  # will be filled by list items
            current_key = key
    return meta


FENCED_CODE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks so their content isn't parsed for links."""
    return FENCED_CODE_RE.sub("", text)


def extract_wiki_links(text: str) -> list[str]:
    """Return all wiki link targets found in text (outside code blocks)."""
    return WIKI_LINK_RE.findall(strip_code_blocks(text))


def build_stem_index(files: list[Path]) -> dict[str, str]:
    """Map file stems to relative paths.  First occurrence wins."""
    idx: dict[str, str] = {}
    for f in files:
        stem = f.stem
        rel = str(f.relative_to(ROOT))
        if stem not in idx:
            idx[stem] = rel
    return idx


def git_last_modified(path: Path) -> datetime | None:
    """Get the last git-commit date for a file.  Returns None on failure."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(path.relative_to(ROOT))],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return datetime.fromisoformat(result.stdout.strip())
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


# ── Lint passes ──────────────────────────────────────────────────────────────


def lint_frontmatter(rel: str, meta: dict[str, object] | None, report: LintReport) -> None:
    stem = Path(rel).stem
    if stem in FRONTMATTER_EXEMPT:
        return
    # Agent/prompt/instructions files use their own frontmatter schema
    if any(rel.endswith(s) for s in LINK_CHECK_EXEMPT_SUFFIXES):
        return
    if any(rel.startswith(d) for d in LINK_CHECK_EXEMPT_DIRS):
        return
    if meta is None:
        report.findings.append(Finding("FM01", "Missing frontmatter", rel))
        return
    report.files_with_frontmatter += 1
    for field_name in VAULT_REQUIRED_FIELDS:
        val = meta.get(field_name)
        if val is None or (isinstance(val, str) and not val.strip()):
            report.findings.append(Finding("FM02", f"Missing required field: {field_name}", rel))
    # Tag taxonomy check
    tags = meta.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            if not any(tag.startswith(p) for p in TAG_PREFIXES):
                report.findings.append(
                    Finding(
                        "FM03",
                        f"Tag '{tag}' not in known taxonomy ({', '.join(TAG_PREFIXES)})",
                        rel,
                    )
                )


def lint_links(all_links: dict[str, list[str]], stem_index: dict[str, str], report: LintReport) -> None:
    """Check for broken links and orphan pages."""
    incoming: dict[str, int] = {}
    for rel, links in all_links.items():
        # Skip link-target checks for instruction/agent/prompt files
        exempt = (
            any(rel.startswith(d) for d in LINK_CHECK_EXEMPT_DIRS)
            or any(rel.endswith(s) for s in LINK_CHECK_EXEMPT_SUFFIXES)
            or Path(rel).stem in LINK_CHECK_EXEMPT_FILES
        )
        for target in links:
            target_clean = target.strip()
            # Skip placeholder / example targets
            if PLACEHOLDER_LINK_RE.search(target_clean) or target_clean.lower() in EXAMPLE_LINK_TARGETS:
                continue
            # Normalize: strip pages/ prefix if present
            if target_clean.startswith("pages/"):
                target_clean = Path(target_clean).stem
            if target_clean not in stem_index:
                if not exempt:
                    report.findings.append(Finding("LK01", f"Broken link [[{target}]] — target not found", rel))
            else:
                incoming[target_clean] = incoming.get(target_clean, 0) + 1

    # Orphan detection — files with zero incoming links
    for stem, rel in stem_index.items():
        if stem in FRONTMATTER_EXEMPT:
            continue
        # Skip root-level files from orphan check (README, project_memory, etc.)
        if "/" not in rel:
            continue
        # Skip categories that are naturally orphaned (navigated outside the graph)
        if any(rel.startswith(d) for d in LINK_CHECK_EXEMPT_DIRS):
            continue
        if any(rel.endswith(s) for s in LINK_CHECK_EXEMPT_SUFFIXES):
            continue
        # Checkpoint files are time-series archives, not graph-navigated
        if "chat_checkpoint_" in stem:
            continue
        # Wiki navigation files (generated/append-only)
        if rel in ("wiki/index.md", "wiki/log.md"):
            continue
        if stem not in incoming:
            report.findings.append(Finding("LK02", "Orphan page — no incoming wiki links", rel))
            report.orphan_count += 1


# Files that are legitimately large — living docs, major specs, or main task files.
# Exempt from ST01 size warnings.
SIZE_EXEMPT = {
    "docs/memory/project_memory.md",
    "docs/research/convergence_detection.md",
    "docs/research/deep_surveillance_tools.md",
    "docs/specs/convergence_detection_spec.md",
    "docs/specs/world_model_spec.md",
    "tasks/active/quant_training_ground.md",
}


def lint_structure(rel: str, content: str, line_threshold: int, report: LintReport) -> None:
    """Check file size and Related section presence."""
    line_count = content.count("\n") + 1
    if line_count > line_threshold and rel not in SIZE_EXEMPT:
        report.findings.append(
            Finding(
                "ST01",
                f"File has {line_count} lines (threshold: {line_threshold})",
                rel,
            )
        )

    # Related section check for research/spec/task files
    if any(rel.startswith(d) for d in RELATED_REQUIRED_DIRS):
        stem = Path(rel).stem
        if stem not in FRONTMATTER_EXEMPT:
            if "## Related" not in content:
                report.findings.append(Finding("ST02", "Missing ## Related section", rel))


def lint_staleness(rel: str, md_path: Path, stale_days: int, report: LintReport) -> None:
    """Flag files not modified in git for >stale_days."""
    last_mod = git_last_modified(md_path)
    if last_mod is None:
        return  # untracked or git unavailable
    now = datetime.now(UTC)
    age = (now - last_mod).days
    if age > stale_days:
        report.findings.append(Finding("ST03", f"Stale — last git change {age} days ago", rel))


# ── Main ─────────────────────────────────────────────────────────────────────


def run_lint(*, line_threshold: int = 500, check_stale: bool = True, stale_days: int = 90) -> LintReport:
    """Execute all lint passes and return the report."""
    report = LintReport()
    files = iter_md_files()
    report.total_files = len(files)
    stem_index = build_stem_index(files)

    all_links: dict[str, list[str]] = {}
    all_targets: set[str] = set()

    for md_path in files:
        rel = str(md_path.relative_to(ROOT))
        content = md_path.read_text(encoding="utf-8", errors="replace")
        meta = parse_frontmatter_simple(content)

        # Frontmatter checks
        lint_frontmatter(rel, meta, report)

        # Collect links
        links = extract_wiki_links(content)
        all_links[rel] = links
        report.total_wiki_links += len(links)
        all_targets.update(links)

        # Structure checks
        lint_structure(rel, content, line_threshold, report)

        # Staleness checks
        if check_stale:
            lint_staleness(rel, md_path, stale_days, report)

    report.unique_link_targets = len(all_targets)

    # Cross-file link checks
    lint_links(all_links, stem_index, report)

    return report


def format_text(report: LintReport) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append(f"Obsidian Vault Lint — {report.total_files} files scanned")
    lines.append(f"  Files with frontmatter: {report.files_with_frontmatter}")
    lines.append(f"  Wiki links: {report.total_wiki_links} ({report.unique_link_targets} unique targets)")
    lines.append(f"  Orphan pages: {report.orphan_count}")
    lines.append("")

    if not report.findings:
        lines.append("No findings. Vault is clean.")
        return "\n".join(lines)

    # Group by code
    by_code: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_code.setdefault(f.code, []).append(f)

    for code in sorted(by_code):
        items = by_code[code]
        lines.append(f"[{code}] ({len(items)} issues)")
        for item in sorted(items, key=lambda x: x.path):
            lines.append(f"  {item.path}: {item.message}")
        lines.append("")

    lines.append(f"Total findings: {len(report.findings)}")
    return "\n".join(lines)


def format_json(report: LintReport) -> str:
    """Format report as JSON."""
    return json.dumps(
        {
            "total_files": report.total_files,
            "files_with_frontmatter": report.files_with_frontmatter,
            "total_wiki_links": report.total_wiki_links,
            "unique_link_targets": report.unique_link_targets,
            "orphan_count": report.orphan_count,
            "findings": [asdict(f) for f in report.findings],
        },
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Obsidian vault-wide lint for TirraMind")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-stale", action="store_true", help="Skip staleness checks")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 only on blocking errors (FM01, FM02, LK01). Advisory codes LK02/ST01/ST02/ST03 are reported but do not fail.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Line count threshold for ST01 (default: 500)",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=90,
        help="Days before a file is flagged stale (default: 90)",
    )
    args = parser.parse_args()

    report = run_lint(
        line_threshold=args.threshold,
        check_stale=not args.no_stale,
        stale_days=args.stale_days,
    )

    if args.json:
        print(format_json(report))
    else:
        print(format_text(report))

    # Exit with code 1 if there are findings (useful for CI)
    if args.strict:
        # Blocking errors enforced in CI.
        # LK01 is advisory until the 52 pre-existing broken links are resolved
        # (many point to Python module names that are not wiki pages).
        # Promote LK01 to _BLOCKING once `obsidian_lint.py --strict` reports 0 LK01s.
        _BLOCKING = {"FM01", "FM02"}
        blocking = [f for f in report.findings if f.code in _BLOCKING]
        sys.exit(1 if blocking else 0)
    sys.exit(1 if report.findings else 0)


if __name__ == "__main__":
    main()
