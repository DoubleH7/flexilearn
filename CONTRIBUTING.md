# Contributing to Flexilearn

Thanks for your interest. Bug reports, documentation fixes and pull requests are all
welcome.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency and environment
management. Always run through `uv run` rather than activating the venv manually.

```bash
git clone https://github.com/DoubleH7/flexilearn.git
cd flexilearn
uv sync --extra video          # the video extra lets the full test suite run
```

## The checks CI runs

Run these before opening a pull request — they are exactly what
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python smoke_test.py
uv build
```

Without the `video` extra, one `write_mp4` test is skipped rather than failed.

## How the codebase is organized

`flexilearn/` is the framework and the only thing shipped in the wheel. It should stay
free of experiment-specific code: anything tied to a particular model, dataset or research
problem belongs in a `definitions/` directory in *your* project, not here.

The framework is deliberately simulator- and dataset-agnostic. `flexilearn/sim.py`
imports no simulator; envs are duck-typed against the Gymnasium 5-tuple API. Please keep
it that way — a new hard dependency needs a strong justification, and anything optional
belongs behind an extra in `pyproject.toml` with a clear `ImportError` message (see
`write_mp4` for the pattern).

MLflow access is encapsulated in a single `Tracker` (`flexilearn/tracking.py`). Aside from
the standalone `flexilearn-registry` CLI, no other module should import `mlflow`, and
tracking failures must never abort training.

## Adding a component

New component types are registered with a decorator and referenced from config by key:

```python
@register_model("my_model")
class MyModel(nn.Module): ...
```

| Component       | Decorator                   |
| --------------- | --------------------------- |
| Model           | `@register_model("key")`    |
| Task            | `@register_task("key")`     |
| Dataset         | `@register_dataset("key")`  |
| Data prep stage | `@register_dataprep("key")` |
| Loss            | `@register_loss("key")`     |
| Optimizer       | `@register_optimizer("key")`|

If you are adding something to the framework itself rather than to your own project, it
needs a docstring explaining *why* the design is the way it is, and a test.

## Tests

Tests live in `tests/` and run against a real `Engine` on CPU using the tiny synthetic
components in [`tests/conftest.py`](tests/conftest.py) — reuse `make_config()` and the
autouse `_isolate_outputs` fixture rather than writing to the repository root.

A bug fix should come with a regression test that fails before the fix.

## Style

`ruff` handles both linting and formatting; the configuration lives in `pyproject.toml`.
Source files carry an SPDX header:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
```

## Licensing of contributions

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same terms that cover the project.
