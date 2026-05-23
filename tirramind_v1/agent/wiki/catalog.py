"""Deterministic catalog and lint tooling for the TirraMind wiki."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = (
    "title",
    "type",
    "summary",
    "status",
    "source_docs",
    "updated_on",
)
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass(frozen=True)
class LintFinding:
    """A single lint issue discovered while scanning the wiki."""

    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class WikiPage:
    """Parsed wiki page metadata and link references."""

    title: str
    page_type: str
    summary: str
    status: str
    source_docs: tuple[str, ...]
    updated_on: str
    rel_path: str
    wiki_link_path: str
    links: tuple[str, ...]


@dataclass(frozen=True)
class CatalogResult:
    """Final catalog render plus findings."""

    pages: tuple[WikiPage, ...]
    findings: tuple[LintFinding, ...]
    index_markdown: str


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a constrained YAML frontmatter block from a markdown file."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter start")

    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError("missing frontmatter end")

    metadata: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in lines[1:closing_index]:
        if not raw_line.strip():
            continue

        stripped = raw_line.lstrip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError("list item without a parent key")
            item = _strip_quotes(stripped[2:].strip())
            metadata.setdefault(current_list_key, [])
            cast_list = metadata[current_list_key]
            if not isinstance(cast_list, list):
                raise ValueError(f"field {current_list_key} mixes scalar and list values")
            cast_list.append(item)
            continue

        current_list_key = None
        if ":" not in raw_line:
            raise ValueError(f"invalid frontmatter line: {raw_line.strip()}")

        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("frontmatter key cannot be empty")

        if value == "":
            metadata[key] = []
            current_list_key = key
            continue

        metadata[key] = _strip_quotes(value)

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def _normalize_link_target(target: str) -> str:
    clean = target.strip().replace("\\", "/")
    if clean.endswith(".md"):
        clean = clean[:-3]
    return clean.lstrip("./")


def extract_wiki_links(body: str) -> tuple[str, ...]:
    """Return normalized wiki-link targets from markdown body content."""
    matches = {_normalize_link_target(match) for match in WIKI_LINK_RE.findall(body)}
    return tuple(sorted(target for target in matches if target))


def _page_link_path(page_path: Path, pages_root: Path, wiki_root: Path) -> str:
    rel_to_wiki = page_path.relative_to(wiki_root).as_posix()
    if not rel_to_wiki.startswith("pages/"):
        rel_to_wiki = (Path("pages") / page_path.relative_to(pages_root)).as_posix()
    return rel_to_wiki[:-3] if rel_to_wiki.endswith(".md") else rel_to_wiki


def scan_pages(repo_root: Path) -> tuple[tuple[WikiPage, ...], tuple[LintFinding, ...]]:
    """Scan wiki pages and collect parse-time findings."""
    wiki_root = repo_root / "wiki"
    pages_root = wiki_root / "pages"
    if not pages_root.exists():
        return (), ()

    pages: list[WikiPage] = []
    findings: list[LintFinding] = []

    for page_path in sorted(pages_root.rglob("*.md")):
        rel_path = page_path.relative_to(repo_root).as_posix()
        try:
            raw_text = page_path.read_text(encoding="utf-8")
            metadata, body = parse_frontmatter(raw_text)
        except ValueError as exc:
            findings.append(
                LintFinding(
                    code="frontmatter_error",
                    message=str(exc),
                    path=rel_path,
                )
            )
            continue

        missing_fields = [field for field in REQUIRED_FIELDS if field not in metadata]
        for field in missing_fields:
            findings.append(
                LintFinding(
                    code="missing_field",
                    message=f"missing required field: {field}",
                    path=rel_path,
                )
            )

        source_docs = metadata.get("source_docs", [])
        if not isinstance(source_docs, list):
            findings.append(
                LintFinding(
                    code="invalid_field_type",
                    message="source_docs must be a YAML list",
                    path=rel_path,
                )
            )
            source_docs = []

        if missing_fields:
            continue

        pages.append(
            WikiPage(
                title=str(metadata["title"]),
                page_type=str(metadata["type"]),
                summary=str(metadata["summary"]),
                status=str(metadata["status"]),
                source_docs=tuple(str(item) for item in source_docs),
                updated_on=str(metadata["updated_on"]),
                rel_path=rel_path,
                wiki_link_path=_page_link_path(page_path, pages_root, wiki_root),
                links=extract_wiki_links(body),
            )
        )

    return tuple(pages), tuple(findings)


def lint_pages(pages: tuple[WikiPage, ...], findings: tuple[LintFinding, ...]) -> tuple[LintFinding, ...]:
    """Run structural lint checks on parsed pages."""
    all_findings = list(findings)

    by_title: dict[str, list[WikiPage]] = {}
    by_link_path = {page.wiki_link_path: page for page in pages}
    inbound_counts = {page.wiki_link_path: 0 for page in pages}

    for page in pages:
        by_title.setdefault(page.title.casefold(), []).append(page)

    for duplicates in by_title.values():
        if len(duplicates) <= 1:
            continue
        for page in duplicates:
            all_findings.append(
                LintFinding(
                    code="duplicate_title",
                    message=f"duplicate page title: {page.title}",
                    path=page.rel_path,
                )
            )

    for page in pages:
        for target in page.links:
            if target not in by_link_path:
                all_findings.append(
                    LintFinding(
                        code="broken_link",
                        message=f"broken wiki link: [[{target}]]",
                        path=page.rel_path,
                    )
                )
                continue
            if target != page.wiki_link_path:
                inbound_counts[target] += 1

    for page in pages:
        if inbound_counts[page.wiki_link_path] == 0:
            all_findings.append(
                LintFinding(
                    code="orphan_page",
                    message="page has no inbound links",
                    path=page.rel_path,
                )
            )

    return tuple(all_findings)


def render_index(pages: tuple[WikiPage, ...]) -> str:
    """Render the generated wiki index markdown."""
    header = [
        "# TirraMind Wiki Index",
        "",
        "This file is generated by `tirra-wiki-catalog`. Do not edit it manually.",
        "",
    ]
    if not pages:
        header.append("No pages indexed yet.")
        return "\n".join(header).rstrip() + "\n"

    grouped: dict[str, list[WikiPage]] = {}
    for page in sorted(pages, key=lambda item: (item.page_type, item.title.casefold())):
        grouped.setdefault(page.page_type, []).append(page)

    body: list[str] = []
    for page_type in sorted(grouped):
        body.append(f"## {page_type.title()}")
        body.append("")
        for page in grouped[page_type]:
            body.append(
                f"- [[{page.wiki_link_path}|{page.title}]] — {page.summary} "
                f"(status: {page.status}, updated: {page.updated_on})"
            )
        body.append("")

    return "\n".join(header + body).rstrip() + "\n"


def build_catalog(repo_root: Path, *, write_index: bool = True) -> CatalogResult:
    """Build the wiki catalog, optionally writing the generated index file."""
    pages, parse_findings = scan_pages(repo_root)
    findings = lint_pages(pages, parse_findings)
    index_markdown = render_index(pages)

    if write_index:
        index_path = repo_root / "wiki" / "index.md"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index_markdown, encoding="utf-8")

    return CatalogResult(pages=pages, findings=findings, index_markdown=index_markdown)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the wiki catalog command."""
    parser = argparse.ArgumentParser(description="Build and lint the TirraMind wiki catalog")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing the wiki/ directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check and lint without rewriting wiki/index.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    result = build_catalog(repo_root, write_index=not args.check)

    if result.findings:
        for finding in result.findings:
            location = f" [{finding.path}]" if finding.path else ""
            print(
                f"wiki catalog: {finding.code}{location}: {finding.message}",
                file=sys.stderr,
            )
        return 1

    print(f"wiki catalog: indexed {len(result.pages)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
