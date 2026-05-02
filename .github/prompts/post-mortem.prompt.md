---
description: "Post-mortem after a hard bug, failed approach, or wasted session. Capture what happened and what we learned."
---

# Post-Mortem

Use this after:
- A bug that took more than 30 minutes to diagnose
- An approach that was abandoned after significant work
- A session that ended without meaningful progress
- A design decision that turned out to be wrong

## Instructions

1. Write the post-mortem to `docs/memory/postmortem_<date>_<topic>.md`.
2. Fill in ALL sections below — skip none.

## Template

```markdown
# Post-Mortem: <topic>
Date: <YYYY-MM-DD>

## What Happened
<2-3 sentences: the symptom, the timeline, the impact>

## Root Cause
<1-2 sentences: the actual underlying problem, not the symptom>

## What We Tried (chronological)
1. <first thing tried — result>
2. <second thing tried — result>
3. <what finally worked or why we stopped>

## What We Learned
- <insight 1>
- <insight 2>

## What Changes
- [ ] <concrete action: new test, doc update, code guard, process change>
- [ ] <concrete action>

## Could We Have Caught This Earlier?
<yes/no — and what would have caught it: better test, better spec, better research?>
```

## Rules
- Be honest. The value is in what went wrong, not in defending decisions.
- Keep it short. This is a learning artifact, not a report.
- If the post-mortem reveals a gap in tests or specs, create the missing artifact NOW.
- If it reveals a pattern that will recur, update `docs/memory/project_memory.md`.
