"""Tests for the deterministic TirraMind wiki catalog tool."""

from __future__ import annotations

from pathlib import Path

from agent.wiki.catalog import build_catalog, main, parse_frontmatter


def _write_page(repo_root: Path, rel_path: str, body: str) -> None:
    page_path = repo_root / rel_path
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(body, encoding="utf-8")


def _page(
    title: str,
    *,
    page_type: str = "architecture",
    summary: str = "Summary",
    status: str = "active",
    source_docs: tuple[str, ...] = ("README.md",),
    updated_on: str = "2026-04-05",
    body: str = "",
) -> str:
    source_lines = "\n".join(f"  - {item}" for item in source_docs)
    return (
        "---\n"
        f"title: {title}\n"
        f"type: {page_type}\n"
        f"summary: {summary}\n"
        f"status: {status}\n"
        "source_docs:\n"
        f"{source_lines}\n"
        f"updated_on: {updated_on}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


class TestParseFrontmatter:
    """Frontmatter parsing should stay strict and predictable."""

    def test_parse_valid_frontmatter_with_list(self):
        text = _page("Example", body="Body text")

        metadata, body = parse_frontmatter(text)

        assert metadata["title"] == "Example"
        assert metadata["source_docs"] == ["README.md"]
        assert body == "# Example\n\nBody text"

    def test_missing_frontmatter_start_raises(self):
        try:
            parse_frontmatter("# No frontmatter")
        except ValueError as exc:
            assert "missing frontmatter start" in str(exc)
        else:
            raise AssertionError("expected parse_frontmatter to reject missing frontmatter")

    def test_list_item_without_parent_key_raises(self):
        text = "---\n- bad\n---\n"

        try:
            parse_frontmatter(text)
        except ValueError as exc:
            assert "list item without a parent key" in str(exc)
        else:
            raise AssertionError("expected malformed list to fail")


class TestBuildCatalog:
    """Catalog generation and linting behavior."""

    def test_healthy_wiki_builds_index_without_findings(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/b]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/b.md",
            _page("B", body="[[pages/architecture/a]]"),
        )

        result = build_catalog(tmp_path)

        assert len(result.pages) == 2
        assert result.findings == ()
        assert "[[pages/architecture/a|A]]" in result.index_markdown
        assert (tmp_path / "wiki" / "index.md").exists()

    def test_empty_pages_directory_is_valid(self, tmp_path: Path):
        (tmp_path / "wiki" / "pages").mkdir(parents=True)

        result = build_catalog(tmp_path)

        assert result.pages == ()
        assert result.findings == ()
        assert "No pages indexed yet." in result.index_markdown

    def test_missing_required_field_is_reported(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/missing.md",
            "---\n"
            "title: Missing\n"
            "type: architecture\n"
            "summary: Missing field test\n"
            "status: active\n"
            "updated_on: 2026-04-05\n"
            "---\n",
        )

        result = build_catalog(tmp_path)

        assert any(f.code == "missing_field" for f in result.findings)

    def test_malformed_frontmatter_is_reported(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/bad.md",
            "---\ntitle Bad\n---\n",
        )

        result = build_catalog(tmp_path)

        assert any(f.code == "frontmatter_error" for f in result.findings)

    def test_duplicate_titles_are_reported(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("Duplicate", body="[[pages/architecture/b]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/b.md",
            _page("Duplicate", body="[[pages/architecture/a]]"),
        )

        result = build_catalog(tmp_path)

        duplicate_findings = [f for f in result.findings if f.code == "duplicate_title"]
        assert len(duplicate_findings) == 2

    def test_broken_link_is_reported(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/missing]]"),
        )

        result = build_catalog(tmp_path)

        assert any(f.code == "broken_link" for f in result.findings)

    def test_orphan_page_is_reported(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/b]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/b.md",
            _page("B", body="[[pages/architecture/a]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/c.md",
            _page("C"),
        )

        result = build_catalog(tmp_path)

        orphan_findings = [f for f in result.findings if f.code == "orphan_page"]
        assert len(orphan_findings) == 1
        assert orphan_findings[0].path == "wiki/pages/architecture/c.md"

    def test_non_page_markdown_files_are_ignored(self, tmp_path: Path):
        (tmp_path / "wiki").mkdir(parents=True)
        (tmp_path / "wiki" / "SCHEMA.md").write_text("# Schema\n", encoding="utf-8")
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/b]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/b.md",
            _page("B", body="[[pages/architecture/a]]"),
        )

        result = build_catalog(tmp_path)

        assert len(result.pages) == 2
        assert result.findings == ()


class TestMain:
    """CLI return codes should reflect lint health."""

    def test_main_returns_zero_for_healthy_wiki(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/b]]"),
        )
        _write_page(
            tmp_path,
            "wiki/pages/architecture/b.md",
            _page("B", body="[[pages/architecture/a]]"),
        )

        exit_code = main(["--repo-root", str(tmp_path)])

        assert exit_code == 0

    def test_main_returns_one_when_findings_exist(self, tmp_path: Path):
        _write_page(
            tmp_path,
            "wiki/pages/architecture/a.md",
            _page("A", body="[[pages/architecture/missing]]"),
        )

        exit_code = main(["--repo-root", str(tmp_path), "--check"])

        assert exit_code == 1
