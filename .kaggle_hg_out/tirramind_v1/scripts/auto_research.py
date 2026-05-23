#!/usr/bin/env python3
"""
auto_research.py — Automated ML problem researcher.

Searches Semantic Scholar (free, no key) and optionally GitHub for papers/issues
relevant to a TirraMind training problem. Outputs a triage report for the
Claude skill `research-training-issue` to deep-read.

Usage:
    python scripts/auto_research.py --problem "GNN return loss flat IC=0"
    python scripts/auto_research.py --problem "..." --github-search --max-papers 7
    python scripts/auto_research.py --from-trigger knowledge/trigger_2026_05_08.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
GITHUB_SEARCH_URL = "https://api.github.com/search/issues"
ARXIV_ABS_URL = "https://arxiv.org/abs/{}"
ARXIV_SRC_URL = "https://arxiv.org/src/{}"

# GNN/finance repos worth searching for related issues
GITHUB_REPOS = [
    "pyg-team/pytorch_geometric",
    "dmlc/dgl",
    "snap-stanford/ogb",
    "microsoft/qlib",
    "AI4Finance-Foundation/FinRL",
]

STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would could should may might shall can need dare ought used to at in "
    "on of for with by from to up about as into through during and or but "
    "not no nor so yet both either neither once whether while since until "
    "before after than that this these those i we you he she it they me "
    "him her us them my our your his its their what which who when where "
    "why how all each every both few more most other some such only own "
    "same so than too very just because if though although because".split()
)

REQUEST_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_json(
    url: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Fetch JSON from url with optional query params. Retries on 429 (rate limit)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "TirraMind-AutoResearch/1.0 (contact: tirramind)")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 5 * (attempt + 1)
                print(
                    f"  [WARN] Rate limited (429). Waiting {wait}s...", file=sys.stderr
                )
                time.sleep(wait)
                continue
            print(f"  [WARN] HTTP {exc.code}: {url[:80]}...", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Request failed: {url[:80]}... — {exc}", file=sys.stderr)
            return None
    print(f"  [WARN] Gave up after 3 retries: {url[:80]}...", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def extract_keywords(problem: str) -> list[str]:
    """Return non-stopword tokens from problem description, lowercase."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", problem.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def build_semantic_scholar_query(problem: str) -> str:
    """Build a search query enriched with GNN/finance context."""
    keywords = extract_keywords(problem)
    # Limit to top-8 keywords + domain terms to avoid over-specification
    core = " ".join(keywords[:8])
    return f"{core} heterogeneous graph neural network temporal prediction"


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------


def search_semantic_scholar(problem: str, max_papers: int = 5) -> list[dict]:
    """Return up to max_papers papers from Semantic Scholar."""
    query = build_semantic_scholar_query(problem)
    print(f"  Semantic Scholar query: {query!r}", file=sys.stderr)

    params = {
        "query": query,
        "fields": "title,abstract,year,citationCount,externalIds,openAccessPdf",
        "limit": str(max_papers * 3),  # fetch extra to filter
    }
    data = _get_json(SEMANTIC_SCHOLAR_URL, params=params)
    if not data or "data" not in data:
        return []

    results = []
    for paper in data["data"]:
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        year = paper.get("year") or 0
        if year < 2020:
            continue  # skip old papers
        results.append(
            {
                "title": paper.get("title", "Untitled"),
                "year": year,
                "citations": paper.get("citationCount", 0),
                "abstract": (paper.get("abstract") or "")[:300],
                "arxiv_id": arxiv_id,
                "has_source": bool(arxiv_id),  # TeX source available if arXiv
                "arxiv_abs": ARXIV_ABS_URL.format(arxiv_id) if arxiv_id else None,
                "arxiv_src": ARXIV_SRC_URL.format(arxiv_id) if arxiv_id else None,
            }
        )

    # Sort by citations descending, then by year
    results.sort(key=lambda p: (-p["citations"], -p["year"]))
    return results[:max_papers]


# ---------------------------------------------------------------------------
# GitHub issue search
# ---------------------------------------------------------------------------


def search_github(problem: str, max_issues: int = 5) -> list[dict]:
    """Search GitHub issues/discussions for related problems."""
    keywords = extract_keywords(problem)
    query_terms = " ".join(keywords[:6])
    repo_filter = " ".join(f"repo:{r}" for r in GITHUB_REPOS)
    query = f"{query_terms} {repo_filter} is:issue"

    print(f"  GitHub query: {query!r}", file=sys.stderr)

    data = _get_json(
        GITHUB_SEARCH_URL, params={"q": query, "per_page": str(max_issues)}
    )
    if not data or "items" not in data:
        return []

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("html_url", ""),
            "repo": item.get("repository_url", "").replace(
                "https://api.github.com/repos/", ""
            ),
            "state": item.get("state", ""),
            "created_at": item.get("created_at", "")[:10],
        }
        for item in data["items"][:max_issues]
    ]


# ---------------------------------------------------------------------------
# Read trigger file
# ---------------------------------------------------------------------------


def read_trigger_file(path: Path) -> str:
    """Extract the problem description from a trigger file."""
    text = path.read_text()
    # Look for '## Problem' or 'Problem:' section
    for pattern in [r"## Problem\s*\n(.+?)(?:\n##|\Z)", r"Problem:\s*(.+)"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()[:500]
    # Fall back to entire file content (first 500 chars)
    return text[:500].strip()


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert to snake_case slug, max 40 chars."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return "_".join(words[:5])[:40]


def render_report(problem: str, papers: list[dict], issues: list[dict]) -> str:
    """Render a Markdown triage report."""
    slug = slugify(problem)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"---",
        f'title: "Triage: {problem[:60]}"',
        f"tags:",
        f"  - doc/research",
        f"  - topic/training",
        f"  - topic/auto-research",
        f"---",
        f"",
        f"# Auto Research Triage: `{slug}`",
        f"",
        f"**Problem:** {problem}",
        f"**Generated:** {ts}",
        f"",
        f"---",
        f"",
        f"## Top Papers (Semantic Scholar)",
        f"",
    ]

    if papers:
        for i, p in enumerate(papers, 1):
            lines += [
                f"### {i}. {p['title']} ({p['year']}, {p['citations']} citations)",
                f"",
                f"{p['abstract']}{'...' if len(p['abstract']) == 300 else ''}",
                f"",
            ]
            if p["arxiv_abs"]:
                lines += [f"- arXiv abstract: {p['arxiv_abs']}"]
            if p["arxiv_src"]:
                lines += [f"- **TeX source** (for skill deep-read): `{p['arxiv_src']}`"]
            lines.append("")
    else:
        lines += ["*No papers found. Try broader keywords.*", ""]

    if issues:
        lines += [
            f"## Related GitHub Issues",
            f"",
        ]
        for i, iss in enumerate(issues, 1):
            state_tag = "✅ closed" if iss["state"] == "closed" else "🟡 open"
            lines += [
                f"### {i}. [{iss['repo']}] {iss['title']} ({state_tag}, {iss['created_at']})",
                f"",
                f"URL: {iss['url']}",
                f"",
            ]

    lines += [
        f"---",
        f"",
        f"## Next Step — Run the Claude Skill",
        f"",
        f"Invoke the `research-training-issue` skill in Copilot chat:",
        f"",
        f"```",
        f'research this training issue: "{problem}"',
        f"```",
        f"",
        f"Or with explicit paper IDs:",
        f"",
        f"```",
    ]
    arxiv_ids = [p["arxiv_id"] for p in papers if p.get("arxiv_id")]
    if arxiv_ids:
        ids_str = " ".join(f"arXiv:{aid}" for aid in arxiv_ids[:3])
        lines.append(f'research this training issue: "{problem}" papers: {ids_str}')
    lines += [
        f"```",
        f"",
        f"The skill will download and read the TeX source of each paper, then write",
        f"`knowledge/diag_{slug}.md` with codebase-grounded solution recommendations.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TirraMind automated ML problem researcher (triage layer)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", type=str, help="Problem description string.")
    group.add_argument(
        "--from-trigger",
        type=Path,
        metavar="FILE",
        help="Read problem description from a trigger file (knowledge/trigger_*.md).",
    )
    parser.add_argument(
        "--max-papers", type=int, default=5, help="Max papers to fetch (default: 5)."
    )
    parser.add_argument(
        "--github-search",
        action="store_true",
        help="Also search GitHub issues (rate-limited: 10 req/min unauthenticated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this file (default: knowledge/triage_{slug}.md).",
    )
    args = parser.parse_args()

    # Resolve problem description
    if args.from_trigger:
        trigger_path = Path(args.from_trigger)
        if not trigger_path.exists():
            print(f"ERROR: trigger file not found: {trigger_path}", file=sys.stderr)
            sys.exit(1)
        problem = read_trigger_file(trigger_path)
        print(f"Problem (from trigger): {problem[:120]}...", file=sys.stderr)
    else:
        problem = args.problem

    print("Searching Semantic Scholar...", file=sys.stderr)
    papers = search_semantic_scholar(problem, max_papers=args.max_papers)
    print(f"  Found {len(papers)} papers.", file=sys.stderr)

    issues: list[dict] = []
    if args.github_search:
        print("Searching GitHub issues...", file=sys.stderr)
        time.sleep(1)  # avoid hammering the API
        issues = search_github(problem)
        print(f"  Found {len(issues)} issues.", file=sys.stderr)

    report = render_report(problem, papers, issues)

    # Determine output path
    out_path = args.output
    if out_path is None:
        slug = slugify(problem)
        out_path = Path("knowledge") / f"triage_{slug}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nTriage report written to: {out_path}", file=sys.stderr)

    # Also print to stdout for easy piping
    print(report)


if __name__ == "__main__":
    main()
