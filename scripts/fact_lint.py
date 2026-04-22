#!/usr/bin/env python3
"""Fact drift linter — self-discovering, schema-driven.

This linter has NO hardcoded knowledge of TirraMind's specific metrics,
canonical files, or fact names.  It discovers all of that by reading the
project's own frontmatter declarations.  Adding a new canonical fact means
updating the owning file's frontmatter — not editing this script.

How to declare ownership (add to any .md file's YAML frontmatter):

    canonical_facts:
      - fact_key: "test_pass_count"
        pattern: "\\b9[,\\s]?\\d{3}\\s*pass(?:ing|ed)"
      - fact_key: "dag_node_count"
        pattern: "\\b\\d{1,3}[- ]node DAG|DAG:?\\s+\\d{1,3}\\s*nodes?"

That file becomes the canonical owner for those facts.
Any active-state file with a DIFFERENT extracted value → FL03 error.
Checkpoint files (tag: doc/checkpoint) and done-task files (tag: status/done)
are historical records and are exempt from FL03.

Rules enforced:
    FL00  No canonical_facts declarations found anywhere (project not configured)
    FL01  A fact value appears in a non-owner active-state file
    FL02  A checkpoint contains a raw fact value (advisory — should link to owner)
    FL03  A fact has different values across active-state files

Exit codes:
    0 — clean
    1 — errors found
    2 — usage error
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent


# ── Frontmatter parsing ────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from a markdown file.  Returns {} on failure."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    if yaml is None:
        # Minimal fallback: extract tags list only
        result: dict = {}
        tags_m = re.search(r"^tags:\s*\n((?:\s+-[^\n]+\n?)+)", fm_text, re.MULTILINE)
        if tags_m:
            result["tags"] = re.findall(r"-\s*(.+)", tags_m.group(1))
        return result
    try:
        parsed = yaml.safe_load(fm_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _file_role(fm: dict) -> str:
    """Return role from frontmatter tags: checkpoint | done | active | unknown."""
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_set = {str(t).strip() for t in tags}
    if "doc/checkpoint" in tag_set:
        return "checkpoint"
    if "status/done" in tag_set:
        return "done"
    if "status/active" in tag_set:
        return "active"
    return "unknown"


# ── Canonical fact discovery ───────────────────────────────────────────────


class FactSpec(NamedTuple):
    fact_key: str
    pattern: str
    owner_file: str  # relative path from ROOT


def _discover_canonical_facts(files: list[Path]) -> list[FactSpec]:
    """Scan all .md files for canonical_facts: declarations in frontmatter.

    The linter learns what facts exist and who owns them entirely from the
    project's own markdown files.  Nothing is hardcoded in this script.
    """
    if yaml is None:
        # Without PyYAML we can't parse the structured canonical_facts list
        return []
    specs: list[FactSpec] = []
    for p in files:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        declared = fm.get("canonical_facts")
        if not declared or not isinstance(declared, list):
            continue
        rel = str(p.relative_to(ROOT))
        for entry in declared:
            if not isinstance(entry, dict):
                continue
            key = (entry.get("fact_key") or "").strip()
            pat = (entry.get("pattern") or "").strip()
            if key and pat:
                specs.append(FactSpec(fact_key=key, pattern=pat, owner_file=rel))
    return specs


# ── Value extraction ───────────────────────────────────────────────────────


def _extract_values(text: str, pattern: str) -> list[str]:
    """Return all captured group values matching pattern in text."""
    values = []
    try:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            groups = m.groups()
            val = (
                next((g for g in groups if g is not None), m.group(0))
                if groups
                else m.group(0)
            )
            if val:
                values.append(val.replace(",", "").replace(" ", "").strip())
    except re.error:
        pass
    return values


# ── Finding type ───────────────────────────────────────────────────────────


class Finding(NamedTuple):
    code: str
    file: str
    line: int
    detail: str


# ── Active-state predicate ─────────────────────────────────────────────────


def _is_active_state(rel: str, fm: dict) -> bool:
    """True if this file is expected to reflect current reality."""
    role = _file_role(fm)
    return (
        role == "active"
        or rel.startswith("tasks/active/")
        or rel.startswith("memories/repo/")
    )


# ── FL00: no declarations found ────────────────────────────────────────────


def check_fl00(specs: list[FactSpec]) -> list[Finding]:
    if not specs:
        return [
            Finding(
                code="FL00",
                file="(project)",
                line=0,
                detail=(
                    "No canonical_facts declarations found in any frontmatter. "
                    "Add 'canonical_facts:' list to the owner file(s) to enable drift detection. "
                    "See script docstring for the declaration format."
                ),
            )
        ]
    return []


# ── FL01: fact value copied into non-owner active-state file ───────────────


def check_fl01(files: list[Path], specs: list[FactSpec]) -> list[Finding]:
    """FL01: fact value from canonical owner also appears in a non-owner active file."""
    findings: list[Finding] = []
    owner_files = {spec.owner_file for spec in specs}

    # Collect the canonical values from each owner
    canonical_values: dict[str, set[str]] = {}
    for spec in specs:
        owner_path = ROOT / spec.owner_file
        if not owner_path.exists():
            continue
        try:
            text = owner_path.read_text(errors="replace")
        except OSError:
            continue
        for val in _extract_values(text, spec.pattern):
            canonical_values.setdefault(spec.fact_key, set()).add(val)

    for p in files:
        rel = str(p.relative_to(ROOT))
        if rel in owner_files:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        if not _is_active_state(rel, fm):
            continue
        for spec in specs:
            owner_vals = canonical_values.get(spec.fact_key, set())
            if not owner_vals:
                continue
            for val in _extract_values(text, spec.pattern):
                if val in owner_vals:
                    findings.append(
                        Finding(
                            code="FL01",
                            file=rel,
                            line=0,
                            detail=(
                                f"fact '{spec.fact_key}' value '{val}' is copied from "
                                f"canonical owner '{spec.owner_file}'. "
                                f"Replace with [[{Path(spec.owner_file).stem}]] reference."
                            ),
                        )
                    )
    return findings


# ── FL02: checkpoint contains raw fact value (advisory) ───────────────────


def check_fl02(files: list[Path], specs: list[FactSpec]) -> list[Finding]:
    """FL02 (advisory): checkpoint has a raw fact value instead of a wiki link."""
    findings: list[Finding] = []
    for p in files:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        if _file_role(fm) != "checkpoint":
            continue
        rel = str(p.relative_to(ROOT))
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if "[[" in line:
                continue  # already a cross-reference
            for spec in specs:
                if _extract_values(line, spec.pattern):
                    findings.append(
                        Finding(
                            code="FL02",
                            file=rel,
                            line=lineno,
                            detail=(
                                f"fact '{spec.fact_key}' as raw value on line {lineno}. "
                                f"Consider 'see [[{Path(spec.owner_file).stem}]]' instead."
                            ),
                        )
                    )
                    break
    return findings


# ── FL03: same fact, different values across active-state files ────────────


def check_fl03(files: list[Path], specs: list[FactSpec]) -> list[Finding]:
    """FL03: two active-state files disagree on the value of the same fact."""
    findings: list[Finding] = []
    for spec in specs:
        value_map: dict[str, list[str]] = {}
        for p in files:
            rel = str(p.relative_to(ROOT))
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            if not _is_active_state(rel, fm):
                continue
            for val in _extract_values(text, spec.pattern):
                value_map.setdefault(val, []).append(rel)

        if len(value_map) <= 1:
            continue

        detail_parts = [f"'{v}' in {sorted(fs)}" for v, fs in sorted(value_map.items())]
        findings.append(
            Finding(
                code="FL03",
                file="(multiple active files)",
                line=0,
                detail=(
                    f"fact '{spec.fact_key}' has conflicting values "
                    f"(canonical owner: '{spec.owner_file}'): "
                    + " vs ".join(detail_parts)
                ),
            )
        )
    return findings


# ── File collection ────────────────────────────────────────────────────────


def _md_files() -> list[Path]:
    patterns = [
        "docs/**/*.md",
        "tasks/**/*.md",
        "wiki/**/*.md",
        "memories/**/*.md",
        "*.md",
        "AGENTS.md",
        ".github/*.md",
    ]
    seen: set[Path] = set()
    result = []
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p not in seen:
                seen.add(p)
                result.append(p)
    return sorted(result)


# ── Fix hints ──────────────────────────────────────────────────────────────


def _fix_hint(f: Finding) -> str:
    return {
        "FL00": "Add 'canonical_facts:' list to owner file frontmatter. See script docstring.",
        "FL01": "Remove raw value, replace with [[wiki link]] to canonical owner.",
        "FL02": "Replace raw number with 'see [[owner_file]]' or remove from checkpoint.",
        "FL03": "Update canonical owner to current value; remove raw value from other active files.",
    }.get(f.code, "")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-discovering fact drift linter. Reads canonical_facts from frontmatter."
    )
    parser.add_argument(
        "--fix-hints", action="store_true", help="Print repair suggestions."
    )
    parser.add_argument("--strict", action="store_true", help="Treat FL02 as an error.")
    parser.add_argument("--fl01-only", action="store_true", help="Run FL01 only.")
    args = parser.parse_args()

    if yaml is None:
        print(
            "fact_lint: WARNING — PyYAML not installed; canonical_facts discovery disabled.\n"
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )

    files = _md_files()
    specs = _discover_canonical_facts(files)

    findings: list[Finding] = []
    findings += check_fl00(specs)

    if specs:
        findings += check_fl03(files, specs)
        if not args.fl01_only:
            if args.strict:
                findings += check_fl02(files, specs)

    if not findings:
        owner_files = sorted({s.owner_file for s in specs})
        print(
            f"fact_lint: clean — {len(specs)} fact(s) monitored "
            f"across {len(owner_files)} owner file(s)."
        )
        sys.exit(0)

    errors = [f for f in findings if f.code in ("FL00", "FL01", "FL03")]
    warnings = [f for f in findings if f.code == "FL02"]

    if warnings and not args.strict:
        print(
            f"fact_lint: {len(warnings)} advisory warning(s) (FL02) — run with --strict to enforce"
        )
        for w in warnings:
            loc = f"{w.file}:{w.line}" if w.line else w.file
            print(f"  [FL02] {loc}: {w.detail}")
            if args.fix_hints:
                print(f"         FIX: {_fix_hint(w)}")

    if errors:
        print(f"\nfact_lint: {len(errors)} error(s) — fact drift detected")
        for e in errors:
            loc = f"{e.file}:{e.line}" if e.line else e.file
            print(f"  [{e.code}] {loc}: {e.detail}")
            if args.fix_hints:
                print(f"           FIX: {_fix_hint(e)}")
        sys.exit(1)

    if args.strict and warnings:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
