---
name: debug
description: Structured debug workflow: reproduce → instrument → capture → hypothesis → fix → regress. Use when tests fail, fixes loop, or root cause is unclear.
disable-model-invocation: true
---

# Debug Mode

You are in **debug mode**. The goal is to diagnose and fix a problem systematically, not guess.

## When To Use

Use this when:
- A test is failing and the cause is not obvious
- The agent is looping on the same fix
- A feature works in isolation but fails in integration
- An error message is misleading or incomplete

## Protocol

Follow these steps in exact order. Do NOT skip steps.

### Step 1: Reproduce

- Run the failing test or command. Capture the exact error output.
- If the failure is intermittent, run it 3 times and note the pattern.
- Write down: what was expected vs. what actually happened.

### Step 2: Isolate

- Find the smallest input that triggers the failure.
- Remove unrelated code/config from the reproduction path.
- Identify: is this a unit failure, integration failure, or environment issue?

### Step 3: Instrument

- Add targeted logging or print statements at the failure boundary.
- Inspect: input values, intermediate state, return values, exception details.
- Do NOT add logging everywhere — only at the decision points around the failure.

### Step 4: Capture

Document in this format:

```
## Bug Report
- **Symptom**: What the user/test sees
- **Expected**: What should happen
- **Actual**: What does happen
- **Smallest reproduction**: Minimal code/command
- **Key observation**: The ONE thing that's surprising from instrumentation
```

### Step 5: Hypothesize

- Form exactly ONE hypothesis about the root cause.
- State what evidence would confirm or refute it.
- If the evidence is ambiguous, add more instrumentation (go back to Step 3).

### Step 6: Fix

- Make the minimal change that addresses the root cause.
- Do not fix symptoms. Fix the cause.
- If the fix requires changing more than one file, explain why.

### Step 7: Regress

- Run the original failing test. It must pass.
- Run the full test suite. Nothing else should break.
- If new failures appear, they become the next bug — go back to Step 1.

### Step 8: Record

- Remove debug instrumentation.
- If the bug was subtle or architectural, write a note in the research doc or project memory.
- If it revealed a gap in test coverage, write the missing test.

## Rules

- Do NOT guess-and-check. If you don't know the cause, instrument more.
- Do NOT change multiple things at once. One hypothesis, one change, one validation.
- If stuck after 3 cycles, escalate: "I've tried X, Y, Z. The evidence points to [area]. I need human guidance on [specific question]."
- Never silence an error. Understand it, then fix or handle it.
