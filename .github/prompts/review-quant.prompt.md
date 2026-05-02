---
description: "Review a quant module for numerical stability, correctness, edge cases, and test coverage."
---

## Instructions

Review the referenced quant module for:

### Numerical Stability
- Division by zero guards
- NaN/Inf propagation checks
- Floating point precision issues (use relative comparisons, not exact equality)
- Log of zero or negative values
- Empty array handling

### Correctness
- Algorithm matches the documented paper/reference
- Return types match docstring
- Edge cases: single data point, constant series, all-NaN

### Test Coverage
- Check if `tests/test_<module>_edge.py` exists
- Verify edge cases are covered: empty input, single element, NaN, Inf, large arrays, all-zero
- If tests are missing, list what needs to be added

### Style
- Follows conventions in `agent/quant/.instructions.md`
- Type hints present and accurate
- Docstrings with Parameters/Returns sections

Report findings as: PASS / WARN (non-critical) / FAIL (must fix), with specific line references.
