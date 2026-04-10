#!/usr/bin/env python3
"""Add Obsidian YAML frontmatter and [[wiki links]] to TirraMind docs.

This script:
1. Adds YAML frontmatter (tags, aliases) to docs that lack it
2. Converts plain-text file references to [[wiki links]]
3. Adds a "Related" section to research ↔ spec ↔ task triads
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Tag taxonomy ─────────────────────────────────────────────────────────────

def infer_tags(rel_path: str, content: str) -> list[str]:
    """Infer Obsidian tags from file path and content."""
    tags = []
    p = rel_path.lower()

    # Doc type tags
    if p.startswith("docs/research/"):
        tags.append("doc/research")
    elif p.startswith("docs/specs/"):
        tags.append("doc/spec")
    elif p.startswith("tasks/active/"):
        tags.append("doc/task")
        tags.append("status/active")
    elif p.startswith("tasks/done/"):
        tags.append("doc/task")
        tags.append("status/done")
    elif p.startswith("docs/adr/"):
        tags.append("doc/adr")
    elif p.startswith("docs/memory/chat_checkpoint"):
        tags.append("doc/checkpoint")
    elif p.startswith("docs/memory/project_memory"):
        tags.append("doc/memory")
    elif p.startswith("wiki/"):
        tags.append("doc/wiki")

    # Phase tags
    if "7b" in p or "batch7" in p or "batch8" in p:
        tags.append("phase/7b")
    if "7c" in p or "convergence" in p.replace("convergence_pre_wm", ""):
        tags.append("phase/7c")
    if "phase 9" in content.lower() or "world_model" in p or "world model" in p:
        tags.append("phase/9")

    # Topic tags from filename
    topic_map = {
        "convergence": "topic/convergence",
        "backtest": "topic/backtest",
        "world_model": "topic/world-model",
        "polymarket": "topic/polymarket",
        "cftc": "topic/cftc",
        "gdelt": "topic/gdelt",
        "pipeline": "topic/pipeline",
        "workflow": "topic/workflow",
        "liquidity": "topic/liquidity",
        "regime": "topic/regime",
        "scoring": "topic/scoring",
        "whale": "topic/whale-tracking",
        "insider": "topic/insider-filings",
        "macro": "topic/macro",
        "market_data": "topic/market-data",
        "power_grid": "topic/power-grid",
        "adsb": "topic/adsb-flight",
        "vessel": "topic/vessel-tracking",
        "weather": "topic/weather",
        "earthquake": "topic/earthquake",
        "satellite": "topic/satellite",
        "dns": "topic/dns",
        "cert_transparency": "topic/cert-transparency",
        "bankruptcy": "topic/bankruptcy",
        "foia": "topic/foia",
        "sovereign_debt": "topic/sovereign-debt",
        "treasury": "topic/treasury",
        "sanctions": "topic/sanctions",
        "disease": "topic/disease-surveillance",
        "pmi": "topic/global-pmi",
        "sentiment": "topic/consumer-sentiment",
        "supply_chain": "topic/supply-chain",
        "drug_regulatory": "topic/drug-regulatory",
        "central_bank": "topic/central-bank",
        "defi": "topic/defi",
        "internet_infrastructure": "topic/internet-infrastructure",
        "form144": "topic/form144",
        "finra": "topic/finra",
        "quant": "topic/quant",
        "rl_layer": "topic/reinforcement-learning",
        "signal_protocol": "topic/signal-protocol",
        "prompt_injection": "topic/security",
        "senior_eng": "topic/engineering",
        "wiki": "topic/wiki",
    }
    fname = Path(rel_path).stem.lower()
    for key, tag in topic_map.items():
        if key in fname:
            tags.append(tag)

    # Layer tags from content
    layer_kw = {
        "layer 1": "layer/surveillance",
        "surveillance surface": "layer/surveillance",
        "layer 2": "layer/feature-engineering",
        "feature engineering": "layer/feature-engineering",
        "layer 3": "layer/world-model",
        "world model": "layer/world-model",
        "layer 4": "layer/fusion",
        "signal fusion": "layer/fusion",
        "layer 5": "layer/learning",
        "rl policy": "layer/learning",
        "layer 6": "layer/adversarial",
        "layer 7": "layer/llm-support",
    }
    lower_content = content[:3000].lower()
    for kw, tag in layer_kw.items():
        if kw in lower_content and tag not in tags:
            tags.append(tag)

    # Surveillance tool research/spec files → layer/surveillance
    if any(x in p for x in ["7b-", "batch7", "batch8"]) and "layer/surveillance" not in tags:
        tags.append("layer/surveillance")

    return sorted(set(tags))


def infer_title(rel_path: str, content: str) -> str:
    """Extract title from first H1 or derive from filename."""
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    stem = Path(rel_path).stem
    return stem.replace("_", " ").replace("-", " ").title()


def has_frontmatter(content: str) -> bool:
    return content.startswith("---\n")


def build_frontmatter(title: str, tags: list[str], aliases: list[str] | None = None) -> str:
    lines = ["---"]
    # Escape title if it has special chars
    if any(c in title for c in ":[]{}#&*!|>'\"%@`"):
        lines.append(f'title: "{title}"')
    else:
        lines.append(f"title: {title}")
    if tags:
        lines.append("tags:")
        for t in tags:
            lines.append(f"  - {t}")
    if aliases:
        lines.append("aliases:")
        for a in aliases:
            lines.append(f"  - {a}")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ── Wiki link conversion ────────────────────────────────────────────────────

# Mapping of all .md files in the repo for resolving links
ALL_MD_FILES: dict[str, str] = {}  # stem -> relative path


def index_md_files():
    """Build index of all markdown files."""
    for md in ROOT.rglob("*.md"):
        if ".obsidian" in str(md) or ".venv" in str(md) or ".git/" in str(md.relative_to(ROOT)):
            continue
        rel = str(md.relative_to(ROOT))
        stem = md.stem
        ALL_MD_FILES[stem] = rel


def convert_markdown_links_to_wiki(content: str) -> str:
    """Convert [text](docs/research/foo.md) style links to [[wiki links]]."""
    # Match markdown links pointing to local .md files
    # Pattern: [display text](relative/path/to/file.md) or [text](../path/file.md)
    def replace_md_link(m):
        display = m.group(1)
        target = m.group(2)
        # Skip external URLs
        if target.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        # Skip non-md files
        if not target.endswith(".md"):
            return m.group(0)
        # Extract stem from target
        target_stem = Path(target.split("#")[0]).stem
        # Check for heading anchor
        heading = ""
        if "#" in target:
            heading = "#" + target.split("#", 1)[1]
        # If display text is a file path, use the stem as display
        display_clean = display.strip()
        if "/" in display_clean and display_clean.endswith(".md"):
            display_clean = Path(display_clean).stem
        # If display text matches the target, simple link
        if display_clean == target_stem:
            return f"[[{target_stem}{heading}]]"
        else:
            return f"[[{target_stem}{heading}|{display_clean}]]"

    content = re.sub(r"\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)", replace_md_link, content)
    return content


def convert_plain_path_refs_to_wiki(content: str) -> str:
    """Convert bare path references like docs/research/foo.md to [[wiki links]]."""
    def replace_path_ref(m):
        path = m.group(0)
        stem = Path(path).stem
        return f"[[{stem}]]"

    # Match docs/research/*.md, docs/specs/*.md, tasks/*/*.md patterns
    # Only match when not already inside a markdown link, wiki link, or after |
    patterns = [
        r"(?<!\[)(?<!\()(?<!\|)docs/research/\w[\w\-]*\.md",
        r"(?<!\[)(?<!\()(?<!\|)docs/specs/\w[\w\-]*\.md",
        r"(?<!\[)(?<!\()(?<!\|)docs/memory/\w[\w\-]*\.md",
        r"(?<!\[)(?<!\()(?<!\|)docs/adr/\w[\w\-]*\.md",
        r"(?<!\[)(?<!\()(?<!\|)tasks/active/\w[\w\-]*\.md",
        r"(?<!\[)(?<!\()(?<!\|)tasks/done/\w[\w\-]*\.md",
    ]
    for pat in patterns:
        content = re.sub(pat, replace_path_ref, content)

    # Clean up any nested wiki links: [[foo|[[bar]]]] → [[bar|foo]]
    content = re.sub(r"\[\[([^\]|]+)\|\[\[([^\]]+)\]\]\]\]", r"[[\2|\1]]", content)
    return content


# ── Related section builder ──────────────────────────────────────────────────

def find_related(rel_path: str) -> list[str]:
    """Find related documents based on naming conventions."""
    stem = Path(rel_path).stem
    related = []

    # Research ↔ Spec pair
    if rel_path.startswith("docs/research/") and stem != "RESEARCH_TEMPLATE":
        spec_stem = stem + "_spec"
        if spec_stem in ALL_MD_FILES:
            related.append(f"[[{spec_stem}|Spec: {stem.replace('_', ' ').title()}]]")
        # Check for matching task
        if stem in ALL_MD_FILES and ALL_MD_FILES[stem].startswith("tasks/"):
            related.append(f"[[{stem}|Task: {stem.replace('_', ' ').title()}]]")

    elif rel_path.startswith("docs/specs/") and stem.endswith("_spec"):
        research_stem = stem.replace("_spec", "")
        if research_stem in ALL_MD_FILES and ALL_MD_FILES[research_stem].startswith("docs/research/"):
            related.append(f"[[{research_stem}|Research: {research_stem.replace('_', ' ').title()}]]")
        # Check for matching task
        if research_stem in ALL_MD_FILES and ALL_MD_FILES[research_stem].startswith("tasks/"):
            related.append(f"[[{research_stem}|Task: {research_stem.replace('_', ' ').title()}]]")

    elif rel_path.startswith("tasks/"):
        # Task → Research + Spec
        task_stem = stem
        if task_stem in ALL_MD_FILES and ALL_MD_FILES[task_stem].startswith("docs/research/"):
            related.append(f"[[{task_stem}|Research: {task_stem.replace('_', ' ').title()}]]")
        spec_stem = task_stem + "_spec"
        if spec_stem in ALL_MD_FILES:
            related.append(f"[[{spec_stem}|Spec: {task_stem.replace('_', ' ').title()}]]")

    # Cross-topic relationships for convergence family
    convergence_family = [
        "convergence_detection", "convergence_backtest",
        "convergence_template_expansion", "convergence_signal_expansion",
        "convergence_napm_refresh", "convergence_backtest_fast_mode",
        "convergence_backtest_score_cache", "convergence_template_batch2",
        "convergence_audit_pre_worldmodel",
    ]
    if any(s in stem for s in ["convergence"]):
        for fam in convergence_family:
            if fam != stem and fam in ALL_MD_FILES:
                related.append(f"[[{fam}]]")

    # World model relates to convergence and signal protocol
    if "world_model" in stem:
        for r in ["convergence_detection", "signal_protocol_feature_engineering", "rl_layer"]:
            if r in ALL_MD_FILES and r != stem:
                related.append(f"[[{r}]]")

    # Backtest family
    if "backtest" in stem and "convergence" not in stem:
        for r in ["convergence_backtest", "walkforward_backtest", "scoring_validation"]:
            if r in ALL_MD_FILES and r != stem:
                related.append(f"[[{r}]]")

    # Signal/quant family
    if any(x in stem for x in ["signal_protocol", "scoring", "liquidity_regime"]):
        for r in ["convergence_detection", "world_model", "backtest_performance"]:
            if r in ALL_MD_FILES and r != stem:
                related.append(f"[[{r}]]")

    # Pipeline
    if "pipeline" in stem:
        for r in ["convergence_detection", "world_model"]:
            if r in ALL_MD_FILES and r != stem:
                related.append(f"[[{r}]]")

    # Surveillance tools (7b-*) relate to convergence and signal expansion
    if stem.startswith("7b-"):
        for r in ["convergence_detection", "tier1_signal_expansion", "tier2_signal_expansion", "observational_surface"]:
            if r in ALL_MD_FILES:
                related.append(f"[[{r}]]")

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for r in related:
        key = r.split("|")[0].replace("[[", "").replace("]]", "")
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def add_related_section(content: str, related: list[str]) -> str:
    """Append or update Related section at the end of document."""
    if not related:
        return content

    # Check if Related section already exists
    if re.search(r"^##\s+Related\b", content, re.MULTILINE):
        return content  # Don't duplicate

    related_block = "\n\n---\n\n## Related\n\n"
    for r in related:
        related_block += f"- {r}\n"

    return content.rstrip() + related_block


# ── Convert Research/Spec "Research:" and "Spec:" header refs ────────────────

def convert_header_refs(content: str) -> str:
    """Convert Research: path/to/file.md and Spec: path/to/file.md to wiki links."""
    # Research: docs/research/foo.md → Research: [[foo]]
    content = re.sub(
        r"^(Research:\s*)docs/research/([\w\-]+)\.md\s*$",
        lambda m: f"{m.group(1)}[[{m.group(2)}]]",
        content, flags=re.MULTILINE
    )
    # Spec: docs/specs/foo_spec.md → Spec: [[foo_spec]]
    content = re.sub(
        r"^(Spec:\s*)docs/specs/([\w\-]+)\.md\s*$",
        lambda m: f"{m.group(1)}[[{m.group(2)}]]",
        content, flags=re.MULTILINE
    )
    return content


# ── Main processing ─────────────────────────────────────────────────────────

SKIP_FILES = {
    "RESEARCH_TEMPLATE.md", "TASK_TEMPLATE.md", "TEMPLATE.md",
    ".gitkeep", "SCHEMA.md",
}

SKIP_DIRS = {".obsidian", ".venv", ".git", "__pycache__", "node_modules",
             "tirramind_vault", ".tirra_cache", ".tirra_pipeline",
             "tirramind_agent.egg-info", ".pytest_cache"}


def process_file(filepath: Path, dry_run: bool = False) -> dict:
    """Process a single markdown file. Returns change summary."""
    rel = str(filepath.relative_to(ROOT))
    changes = {"path": rel, "frontmatter_added": False, "links_converted": 0, "related_added": 0}

    if filepath.name in SKIP_FILES:
        return changes

    content = filepath.read_text(encoding="utf-8", errors="replace")
    original = content

    # 1. Add frontmatter if missing
    if not has_frontmatter(content):
        title = infer_title(rel, content)
        tags = infer_tags(rel, content)
        if tags:  # Only add frontmatter if we have tags
            fm = build_frontmatter(title, tags)
            content = fm + "\n" + content
            changes["frontmatter_added"] = True

    # 2. Convert markdown links to wiki links
    before = content
    content = convert_markdown_links_to_wiki(content)
    content = convert_plain_path_refs_to_wiki(content)
    content = convert_header_refs(content)
    link_changes = sum(1 for a, b in zip(before.split("\n"), content.split("\n")) if a != b)
    changes["links_converted"] = link_changes

    # 3. Add Related section
    related = find_related(rel)
    if related:
        before_len = len(content)
        content = add_related_section(content, related)
        if len(content) > before_len:
            changes["related_added"] = len(related)

    # Write if changed
    if content != original and not dry_run:
        filepath.write_text(content, encoding="utf-8")

    return changes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add Obsidian links to TirraMind docs")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    print("Indexing markdown files...")
    index_md_files()
    print(f"  Found {len(ALL_MD_FILES)} markdown files")

    # Directories to process
    scan_dirs = ["docs", "tasks", "wiki"]
    root_mds = list(ROOT.glob("*.md"))

    all_files = []
    for d in scan_dirs:
        p = ROOT / d
        if p.exists():
            for md in p.rglob("*.md"):
                if not any(skip in str(md) for skip in SKIP_DIRS):
                    all_files.append(md)
    all_files.extend(root_mds)

    print(f"Processing {len(all_files)} files...")
    total_fm = 0
    total_links = 0
    total_related = 0

    for f in sorted(all_files):
        result = process_file(f, dry_run=args.dry_run)
        if result["frontmatter_added"] or result["links_converted"] or result["related_added"]:
            print(f"  {'[DRY] ' if args.dry_run else ''}✓ {result['path']}"
                  f" (fm={result['frontmatter_added']}, links={result['links_converted']}, related={result['related_added']})")
            total_fm += result["frontmatter_added"]
            total_links += result["links_converted"]
            total_related += result["related_added"]

    print(f"\nDone. Frontmatter added: {total_fm}, Links converted: {total_links}, Related sections: {total_related}")


if __name__ == "__main__":
    main()
