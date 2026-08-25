---
description: Generate an AWOS session checkpoint
agent: awos
---

Generate a session checkpoint following AWOS.md protocol:

!`cd /home/becmachlean/2024/projects/AWOS_coding_agent && python scripts/session_checkpoint.py -m "$ARGUMENTS" 2>&1`

If the checkpoint script is not available, manually summarize:
1. What was accomplished this session
2. What files changed
3. What decisions were made
4. What's blocked / next steps
5. Bump the version in the checkpoint file
