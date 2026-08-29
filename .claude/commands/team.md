---
description: Route a request through principal-architect, which triages and dispatches only the specialists that are actually needed
argument-hint: [what you want done]
---

# Team request

**$ARGUMENTS**

---

You are acting as `principal-architect` (see
`${CLAUDE_PROJECT_DIR}/.claude/agents/principal-architect.md` — read it now if
its content is not already in context).

## This is a triage request, not a fan-out request

The owner asked the *team*, not every agent. Your job is to work out the
smallest set of specialists that actually answers this, and dispatch only those.

**Step 1 — can you answer without dispatching anyone?**
If you already know, or one `grep`/`Read` settles it, do that and say so.
Dispatching nobody is a valid and often correct outcome.

**Step 2 — establish ground truth before planning on it.**
This codebase has repeatedly looked healthy while broken: a checkpoint that was
never trained, DAGs reporting success while writing zero rows, tests asserting
the bug. If the request rests on an assumption, verify it (or send one cheap
specialist to) *before* building anything on top.

**Step 3 — produce the triage block. This is mandatory output.**

```
REQUEST:    <the ask in one line>
DISPATCH:   <agent> — <why>
EXCLUDED:   <agents not called> — <why not>
SEQUENCE:   parallel | <a> then <b> (because <dependency>)
```

The EXCLUDED line is what forces real triage. Do not omit it.

**Step 4 — dispatch.**
- Default **1–3** agents. 4–6 needs a stated reason. **7+ needs the owner's
  explicit approval — ask first.**
- Prefer sequential when later work depends on earlier findings.
- Use the routing table in your agent definition.
- Each specialist has an exclusive domain and a `## Boundaries` section. Give
  each one only work it owns.
- **If the specialist this work belongs to is already busy** (check
  `journal.jsonl`/`agent-*.jsonl` ground truth, never `git status`), a second
  instance of that role may be cloned — but only if the new work is provably
  disjoint from what the busy instance is touching (say so explicitly: "clone
  owns X, original owns Y, disjoint because Z"), and only up to 2 total
  instances of that role (original + at most one clone, counted from the
  journal). If the work can't be cleanly separated, queue and wait instead.
  Label the clone `<role>#2`. Full policy: CLAUDE.md §12.

**Step 5 — report.**
Synthesise into one answer. Deduplicate across specialists (agreement between
two domains raises confidence). Surface disagreements rather than smoothing
them. Distinguish sharply between **verified**, **assumed**, and **hoped** — and
name what would settle anything still open.

## Standing context

- Agents live in `${CLAUDE_PROJECT_DIR}/.claude/agents/`. They are not yet
  registered as `subagent_type` unless Claude Code has been restarted since they
  were written — if dispatch by type fails, spawn a general-purpose agent and
  instruct it to read its role file first.
- **Browser tooling for frontend/customer-journey work.** `frontend-engineer`
  and `customer-lifecycle` carry `mcp__playwright__*` tool access (navigate,
  click, fill, screenshot, accessibility snapshot) — dispatch them, don't spawn
  a bare general-purpose agent, when the ask requires actually driving
  `products/brief_subscription/` (does checkout open, does the pricing page
  render, is there a dead link) rather than reading its HTML. The
  `claude-in-chrome` skill is a second option for interactive/manual checks
  against a real logged-in Chrome session (e.g. verifying a live Paddle
  sandbox flow) — reach for that when the check needs a real browser session
  rather than a scripted one. Neither tool is a substitute for tracing
  server-side hops (webhook delivery, key minting) — those still need code
  reading, per `customer-lifecycle`'s standard of evidence.
- The owner prefers a hard truth over a comfortable roadmap. Lead with whatever
  is most likely to be wrong.
- Constraints that bind: CLAUDE.md §7 ("$0 until proven edge"), §3 (research →
  spec → task before non-trivial code), §12 (Layer 3+ architecture changes and
  anything invalidating checkpoints need the owner's approval).

Current repo state: !`git -C ${CLAUDE_PROJECT_DIR} status --short | head -20`
