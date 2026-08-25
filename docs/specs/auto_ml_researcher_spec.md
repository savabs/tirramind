---
title: "Spec: Automated ML Problem Researcher"
tags:
  - doc/spec
  - phase/auto-research
  - topic/tooling
  - topic/training
  - status/active
---

# Spec: Automated ML Problem Researcher

## Goal

Build a two-component automated research system that, given a TirraMind training problem,
produces a ranked, codebase-grounded solution report:

1. **`.claude/skills/research-training-issue/SKILL.md`** — Claude skill invoked by the agent
   (GitHub Copilot or Claude Code). Takes a problem description, searches arXiv + GitHub,
   reads the TeX source of the top 3 papers, and writes `knowledge/diag_{slug}.md` connecting
   findings to the relevant TirraMind source files.

2. **`scripts/auto_research.py`** — CLI tool that performs the search + triage layer (finding
   paper IDs and GitHub issue URLs). Outputs a list of URLs for the skill to deep-read.

Research doc: [[auto_ml_researcher]]

---

## Files Affected

| File | Action |
|---|---|
| `.claude/skills/research-training-issue/SKILL.md` | Create |
| `scripts/auto_research.py` | Create |
| `knowledge/` | Create directory (gitignore with exception for `knowledge/*.md`) |
| `.gitignore` | Add `knowledge/` entry (keep `.md` files, ignore `.tar.gz` cache) |

---

## Implementation Steps

### Step AR.1 — Create `knowledge/` directory and `.gitignore` rules

Create `knowledge/` directory. Update `.gitignore` to:
- Ignore `knowledge/*.tar.gz` (arXiv source cache, too large for git)
- Track `knowledge/*.md` (summaries should be in git for team visibility)

### Step AR.2 — Create `scripts/auto_research.py`

A standalone CLI tool that:
1. Accepts `--problem "description of the issue"` as input
2. Calls Semantic Scholar API (free, no key) to find top 5 papers by relevance
3. Optionally calls GitHub search API for related issues/discussions (unauthenticated, 10 req/min)
4. Outputs a Markdown-formatted triage report to stdout and writes it to
   `knowledge/triage_{slug}.md`
5. Prints the arXiv IDs + GitHub URLs for the Claude skill to read

**CLI interface:**
```bash
python scripts/auto_research.py \
  --problem "GNN return loss flat despite IC target, 92% GDELT imbalance" \
  --codebase-context "agent/models/gnn/trainer.py" \
  [--max-papers 5] \
  [--github-search] \
  [--output knowledge/triage_{slug}.md]
```

**Semantic Scholar query construction:**
- Extract keywords from problem description (keyword extractor = split + filter stopwords)
- Build query: `{keywords} GNN heterogeneous temporal finance prediction`
- Fields requested: `title,abstract,year,citationCount,openAccessPdf.url,externalIds`
- Sort: by citationCount descending (most-cited = most relevant for foundational issues)
- Filter: year >= 2020, has openAccess PDF (so we can fetch TeX source)

**GitHub search:**
- Query: `{keywords} in:issues,discussions is:issue` restricted to top repos:
  `pytorch-geometric/pytorch_geometric`, `dmlc/dgl`, `rusty1s/pytorch_geometric`
- Returns top 5 issue URLs

**Output format:**
```markdown
# Auto Research Triage: {problem_slug}

**Problem:** {problem description}
**Generated:** {timestamp}

## Top Papers (Semantic Scholar)

1. **{title}** ({year}, {citationCount} citations)
   Abstract: {first 200 chars}...
   arXiv: https://arxiv.org/abs/{arxiv_id}

## Related GitHub Issues

1. {repo}/{issue_number}: {title}
   URL: {url}

## Next Step

Run the Claude skill with these URLs:
`/research-training-issue arXiv:{id1} arXiv:{id2} problem:"{problem}"`
```

### Step AR.3 — Create `.claude/skills/research-training-issue/SKILL.md`

A Claude/Copilot agent skill that:
1. Accepts: problem description + list of arXiv IDs or URLs (from `auto_research.py` output)
2. For each arXiv ID:
   a. Fetches TeX source from `https://arxiv.org/src/{id}` → saves to `~/.cache/tirramind/papers/{id}.tar.gz`
   b. Extracts → finds `main.tex` entrypoint
   c. Reads the full paper (recursing through `\input{}` includes)
3. Reads the relevant TirraMind source file(s) mentioned in `--codebase-context`
4. Synthesizes a report in `knowledge/diag_{slug}.md`:
   - Problem statement
   - What each paper says that's relevant
   - How it maps to TirraMind's code (specific file:line references)
   - Ranked solution options (most promising first)
   - Implementation notes for each solution

**Skill trigger phrases** (how the user invokes it in chat):
- "research this training issue: [description]"
- "auto-research [problem]"
- "find papers for [issue]"

### Step AR.4 — Add loss-component ratio check to trainer

In `agent/models/gnn/trainer.py`, after each epoch's loss computation:
- If `dt_loss / (return_loss + 1e-8) > 50` for 3+ consecutive epochs → log WARNING:
  `"WARNING: return head receiving <2% of gradient budget (dt/ret ratio={ratio:.0f}x). Consider increasing --return-weight or re-checking data balance."`
- This is the "optimization target validator" the user asked for

**Files affected:** `agent/models/gnn/trainer.py` only.

---

## Edge Cases

- Semantic Scholar API returns no results → fall back to `arxiv.org/search/` URL with keywords
- arXiv TeX source fetch fails (some older papers don't have TeX) → fall back to PDF abstract
- GitHub rate limit hit (10 req/min unauthenticated) → skip GitHub, log message
- `knowledge/` already has a report for this problem slug → warn and overwrite (add `--force` flag)
- Problem description is too vague → skill asks a clarifying question before proceeding

---

## Testing Plan

**AR.1 (directory + gitignore):** `ls knowledge/` exists; `git status` shows `*.md` tracked, `*.tar.gz` ignored.

**AR.2 (auto_research.py):**
```bash
# Happy path
python scripts/auto_research.py --problem "GNN flat IC heterogeneous graph finance"
# Verify: outputs triage file with ≥3 papers, all have valid arXiv URLs

# Edge case: very obscure query with no results
python scripts/auto_research.py --problem "xyzzy nonexistent training error type 42"
# Verify: outputs "No papers found, try broader terms" — doesn't crash

# Edge case: no network
REQUESTS_TIMEOUT=0.001 python scripts/auto_research.py --problem "test"
# Verify: exits with clear error message, not stack trace
```

**AR.3 (skill):** Manually invoke via Copilot chat with a known problem → verify `knowledge/diag_*.md` is created with paper summaries and TirraMind code references.

**AR.4 (trainer check):** Unit test: mock epoch with `dt_loss=100`, `return_loss=1.0`, run for 3 epochs → verify WARNING emitted. Confirm no WARNING when ratio < 50.

---

## Related

- [[auto_ml_researcher]] — research doc
- [[auto_ml_researcher_task]] — task file
- [[phase41b_gnn_signal_extraction]] — the problem that triggered this feature
