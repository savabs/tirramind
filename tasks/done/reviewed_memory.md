---
title: "Task: Reviewed Memory — Lesson Promotion Pipeline + Episodic Decay"
tags:
  - doc/task
  - status/done
  - phase/29
  - topic/memory
  - layer/learning
---

# Task: Reviewed Memory

Status: completed
Research: [[reviewed_memory]]
Spec: [[reviewed_memory_spec]]

## Steps

- [x] 29.1: Add `validated` + `run_id` fields to LearningEntry, add get_validated_learnings()
- [x] 29.2: Add EpisodicMemory.decay() with archival
- [x] 29.3: Add config params (TIRRA_LESSON_MIN_SUPPORT, TIRRA_LESSON_MIN_RUNS, TIRRA_EPISODE_TTL_DAYS)
- [x] 29.4: Create CandidateStore (agent/memory/candidates.py) — cluster, stage, promote, reject
- [x] 29.5: Wire CandidateStore into autonomous loop + filter reflection inputs
- [x] 29.6: Edge case test suite (14 cases minimum)

## Notes

- Inspired by agentic-stack candidate lifecycle concept, adapted for autonomous statistical operation
- No CLI tools, no markdown rendering, no cron — pure inline post-run processing
- Backward compatible with existing learnings.jsonl files

## Related

- [[reviewed_memory]]
- [[reviewed_memory_spec]]
- [[store]]
- [[autonomous]]
- [[reflection]]