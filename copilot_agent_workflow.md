---
title: Copilot Agent Workflow System
tags:
  - topic/workflow
---

# Copilot Agent Workflow System

This document defines a structured workflow for AI coding agents using
GitHub Copilot in Visual Studio Code. The goal is to minimize context
window usage and improve reasoning quality by moving planning, research,
and long-term knowledge into files instead of chat history.

------------------------------------------------------------------------

# Core Concept

Instead of performing reasoning, planning, and coding in the chat
context, the workflow separates them into stages and persists the
outputs as files.

Pipeline:

User Request → Research Phase → Specification Phase → Implementation
Phase

Each phase produces documents that the next phase consumes.

------------------------------------------------------------------------

# Recommended Project Structure

project/ │ ├ .github/ │ └ copilot-instructions.md │ ├ docs/ │ ├
research/ │ ├ specs/ │ └ memory/ │ ├ tasks/ │ └ active/ │ └ src/

Folder purposes:

docs/research/ Stores analysis and exploration of the project.

docs/specs/ Stores structured implementation plans for features.

docs/memory/ Stores long‑term architectural knowledge of the project.

tasks/active/ Tracks current tasks being worked on.

------------------------------------------------------------------------

# Agent Operating Rules

The agent must follow these rules:

1.  Never immediately implement a feature.
2.  Always perform research first.
3.  Convert research into a structured specification.
4.  Implement code only after a spec exists.
5.  Store reusable knowledge in project memory files.

------------------------------------------------------------------------

# Phase 1: Research

Goal: Understand the current architecture before implementing changes.

Steps:

1.  Read only the files relevant to the requested feature.
2.  Analyze project structure and dependencies.
3.  Identify the correct insertion points for new code.
4.  Record findings.

Output file:

docs/research/`<feature_name>`{=html}.md

Example structure:

Feature: image_upload

Current architecture: - backend: express - routes folder handles API

Observations: - no upload endpoint exists - file storage system not
implemented

Risks: - file size validation required

No code should be edited during this phase.

------------------------------------------------------------------------

# Phase 2: Specification

Goal: Transform research into a precise implementation plan.

Output file:

docs/specs/`<feature_name>`{=html}\_spec.md

Specification structure:

Goal What the feature must accomplish.

Files Affected List of files that must be created or modified.

Implementation Steps Ordered steps for implementation.

Edge Cases Possible failure scenarios.

Testing Plan How the feature should be validated.

Example:

Goal: Allow users to upload images.

Files: src/routes/upload.ts src/services/storage.ts

Steps: 1 create upload endpoint 2 validate file size 3 store image 4
return uploaded image URL

Edge Cases: large file uploads invalid file types

------------------------------------------------------------------------

# Phase 3: Implementation

Goal: Implement the feature exactly according to the specification.

Rules:

1.  Follow the spec strictly.
2.  Modify only files listed in the spec.
3.  Do not re‑analyze architecture during coding.
4.  If issues arise, update the spec before continuing.

After implementation:

Mark task as completed.

------------------------------------------------------------------------

# Memory System

Important architectural information should be stored in:

the latest maintained checkpoint in `docs/memory/` and the active research/spec/task triad.

If a dedicated project memory file exists in `docs/memory/`, it may be used as persistent context, but it is not a required dependency.

------------------------------------------------------------------------

# Task Management

Each active feature should have a task file:

tasks/active/`<task_name>`{=html}.md

Example:

```
Task: image_upload

Status: active

Research: [[image_upload]]

Spec: [[image_upload_spec]]
```

------------------------------------------------------------------------

# Context Efficiency Rules

To avoid context window overflow:

1.  Move reasoning into files instead of chat.
2.  Reference spec documents instead of repeating analysis.
3.  Read only necessary files.
4.  Avoid long explanations in chat.
5.  Start new chat sessions after completing a feature.

------------------------------------------------------------------------

# Daily Workflow

1.  User requests feature.
2.  Agent performs research.
3.  Agent writes research document.
4.  Agent creates specification document.
5.  Agent implements code using the spec.
6.  Agent updates task file.
7.  Agent writes a checkpoint at natural breakpoints.

------------------------------------------------------------------------

# Benefits

This workflow provides:

-   Smaller chat context
-   Better reasoning separation
-   Reproducible implementation plans
-   Improved code quality
-   Scalable development workflow

------------------------------------------------------------------------

End of Workflow Instructions

## Related

- [[copilot_pro_optimization_guide|Copilot Pro+ Optimization Guide]]
- [[quant_training_ground]]
