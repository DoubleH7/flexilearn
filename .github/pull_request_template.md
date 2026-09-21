## What does this change?

<!-- A sentence or two. Link the issue it closes, if there is one. -->

## Why?

<!-- The problem this solves. For a bug fix, what went wrong and why. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run pytest` passes
- [ ] `uv run python smoke_test.py` passes
- [ ] A bug fix includes a regression test that fails without the fix
- [ ] New or changed behavior is reflected in the README / docstrings
- [ ] No new hard dependency (optional ones go behind an extra in `pyproject.toml`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
