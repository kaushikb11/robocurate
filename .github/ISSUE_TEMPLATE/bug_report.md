---
name: Bug report
about: Report something that isn't working as expected
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## Environment

- **RoboCurate version:** (e.g. output of `uv run python -c "import robocurate; print(robocurate.__version__)"`)
- **OS:** (e.g. macOS 14.5, Ubuntu 22.04)
- **Python version:** (e.g. 3.11.9)
- **Optional extras installed:** (e.g. none / `demo-score` / `rlds` / `maniskill-demos` / `all`)

## Minimal reproduction

The smallest snippet or command that triggers the bug. The
`examples/make_demo_dataset.py` demo builds a tiny dataset you can curate, which
usually makes a self-contained repro easy:

```bash
# e.g.
uv run python examples/make_demo_dataset.py /tmp/demo
uv run robocurate curate /tmp/demo ...
```

```python
# or a minimal Python snippet
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full traceback or error output if there is one.
