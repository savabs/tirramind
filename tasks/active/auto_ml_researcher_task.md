---
title: "Task: Automated ML Problem Researcher"
tags:
  - doc/task
  - phase/auto-research
  - topic/tooling
  - topic/training
  - status/active
---

# Task: Automated ML Problem Researcher

Status: active
Research: [[auto_ml_researcher]]
Spec: [[auto_ml_researcher_spec]]

## Exit Condition

Given a training problem description, `python scripts/auto_research.py --problem "..."` returns
a triage report with ≥3 ranked papers within 10 seconds. The Claude skill
`research-training-issue` reads the top papers and writes `knowledge/diag_{slug}.md`
with codebase-grounded solution recommendations.

## Steps

- [x] AR.0 — Research complete (`[[auto_ml_researcher]]`)
- [x] AR.0b — Spec written (`[[auto_ml_researcher_spec]]`)
- [x] AR.1 — Create `knowledge/` directory + update `.gitignore`
- [x] AR.2 — Create `scripts/auto_research.py` (Semantic Scholar search + GitHub search + triage report)
- [x] AR.3 — Create `.claude/skills/research-training-issue/SKILL.md`
- [x] AR.4 — Add loss-component ratio warning to `agent/models/gnn/trainer.py`
- [ ] AR.5 — Run edge case tests (no results, no network, duplicate slug)
- [ ] AR.6 — Manual smoke test: invoke skill via Copilot chat with real problem

## Notes

- The Semantic Scholar API is free, no API key required — use it as the primary search backend
- GitHub search: 10 req/min unauthenticated — scope to pytorch-geometric, dgl, and stock-prediction repos
- arXiv TeX source URL pattern: `https://arxiv.org/src/{arxiv_id}` (not the PDF URL)
- Cache arXiv source tarballs at `~/.cache/tirramind/papers/` (not committed to git)
- The "optimization target validator" (AR.4) is the trainer-side early warning: warns when
  `dt_loss / return_loss > 50` for 3+ consecutive epochs
- This whole feature is Layer 7 (LLM Support) — it helps diagnose problems, it doesn't train

## Related

- [[auto_ml_researcher]] — research
- [[auto_ml_researcher_spec]] — spec
- [[phase41b_gnn_signal_extraction]] — predecessor task (triggered this feature need)
