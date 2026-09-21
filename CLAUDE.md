# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Flexilearn** is a modular PyTorch training framework for rapid ML experimentation. All
components (models, data pipelines, tasks, optimizers, losses) are registered via
decorators and wired together entirely through a config YAML — no code changes are needed
to swap components.

The repository contains the framework (`flexilearn/`), its test suite (`tests/`), a
runnable example (`examples/minimal/`), and an empty `definitions/` tree that is the
extension point for user components.

## Running things

```bash
uv sync --extra video        # video extra lets the full test suite run

# Run the bundled example
uv run flexilearn examples/minimal/config.yml --definitions examples/minimal/definitions

# Run a config against ./definitions in the working directory
uv run flexilearn config.yml

# Tests, end-to-end check, lint
uv run pytest
uv run python smoke_test.py
uv run ruff check . && uv run ruff format --check .
```

The venv is managed by `uv`. Always use `uv run` rather than activating the venv manually.

## Architecture

### The Registry Pattern (critical to understand)

Every component type has a global registry in `flexilearn/registry.py`. Components
self-register at import time using decorators:

```python
@register_model("my_model")
class MyModel(nn.Module): ...


@register_task("my_task")
class MyTask(Task): ...
```

**Registration is automatic.** `flexilearn/discovery.py` provides `discover_definitions()`,
which walks a `definitions/` tree and imports every module so the decorators run. The
entry point (`flexilearn/entry.py`) calls it at the start of `main()` before the engine is
constructed. Dropping a new file into `definitions/` is enough — no manual import list.

The directory is resolved in order: the `--definitions` flag, `$FLEXILEARN_DEFINITIONS`,
`./definitions` in the working directory, then `definitions/` beside the installed
package. A missing directory warns rather than raising, so an installed wheel works with
components registered directly in Python.

### Core Abstractions

**`RuntimeState`** (`flexilearn/runtime_state.py`) — owns all live objects. It lazily
builds `Group` and `DataSource` instances on first access and manages checkpointing.
Checkpoint identity is a SHA-256 hash of the config, so changing config creates a new
checkpoint directory under `outputs/<hash>/`.

**`Group`** (`flexilearn/group.py`) — owns one `nn.ModuleDict` + at most one optimizer.
Call `group.run(x)` for the forward pass (chains models sequentially), `group.update(loss)`
for backward + optimizer step with gradient accumulation. A group declared without an
`optimizer:` block is an inference-only unit whose `update()` is a no-op. Groups are
declared in config under `groups:` and accessed in tasks via `self.get_group("name")`.

**`Task`** (`flexilearn/task.py`) — abstract base; implement `loop()` to define one
training iteration. The engine calls `loop()` repeatedly per the curriculum. Access groups
and data sources through `self.get_group()` / `self.get_source()`. Must be registered with
`@register_task("name")`.

**`DataSource`** (`flexilearn/data.py`) — wraps a dataset + DataLoader + a pipeline of
`Stage` transforms. Stages are chained as Python generators; `data_source.get_next()`
pulls one item through the full pipeline. Declared under `data_sources:` in config.

**`EnvFactory`** (`flexilearn/sim.py`) — simulations as data sources. A sim enters config
as a `data_sources:` block whose dataset subclasses `EnvFactory` (implement `make()`;
register with `@register_dataset`); tasks pull live envs via `get_next()`. Envs are
duck-typed (Gymnasium 5-tuple API) — the framework imports no simulator. `persistent: true`
in config reuses one env across pulls (expensive sims). Declare with
`loader: {batch_size: 0, num_workers: 0}` (enforced at build). Companions: `borrow_env`
(lifecycle), `rollout` → `Episode` (episode driver; policies return action chunks, raise
`PolicyFailure("reason")` into the failure taxonomy; envs report `info["success"]` /
`info["failure_mode"]`), `write_mp4` (rollout videos; needs the `video` extra).

**`Engine`** (`flexilearn/engine.py`) — iterates over `curriculum` tasks in order, calls
`task._step()` (which calls `loop()`) up to `iterations` times per epoch. Handles
`StopIteration` (dataset exhaustion) by advancing epochs, applies per-task failure
policies, and triggers checkpointing when `task.checkpoint_flag` is true.

**`Tracker`** (`flexilearn/tracking.py`) — the **only** module in the training path that
imports `mlflow`. Nothing under `definitions/` should ever import it; models declare
intent through a plain `export(path)` hook and core does the logging. Tracking failures
must never abort training.

### Config Structure

```yaml
experiment:      # name, seed, attempt_resume
curriculum:      # ordered dict of named tasks; each has type, iterations, epochs, checkpointing_interval
groups:          # named groups; each has models dict, optimizer, settings (grad_accumulation, device, is_trainable)
data_sources:    # named sources; each has dataset, loader, preparations (Stage pipeline)
logging:         # MLflow: scalar/gradient/histogram intervals, tracking_uri, registry
checkpointing:   # inclusion dictionary; per-group weights policy (full/trainable/none)
```

The `type` field in any config block maps to the registry key. Extra fields under `type`
are passed as `**kwargs` to the constructor (via Pydantic's `model_extra`).

### Extending the Framework

To add a new component, create a file in the appropriate `definitions/` subdirectory and
register it:

| Component | Decorator | Location |
|---|---|---|
| Model | `@register_model("key")` | `definitions/models/` |
| Task | `@register_task("key")` | `definitions/tasks/` |
| Dataset | `@register_dataset("key")` | `definitions/data/` |
| Data prep stage | `@register_dataprep("key")` | `definitions/data/` |
| Loss | `@register_loss("key")` | `definitions/losses/` |

`discover_definitions()` imports the file automatically — just reference the key in config.

### Checkpointing

Checkpoints are saved under `outputs/<config-hash>/checkpoint-full-<timestamp>.fl` as
torch serialized dicts containing groups, datasources, global RNG state, and task
counters. Tensors are written on CPU and loaded with `map_location="cpu"`, so a checkpoint
written on a CUDA host is loadable on a CPU-only one. Set `attempt_resume: true` in config
to auto-resume from the latest checkpoint for the same config hash.

## Conventions

* Source files carry an SPDX header (`# SPDX-License-Identifier: Apache-2.0`).
* `ruff` handles lint and formatting; config lives in `pyproject.toml`.
* Keep `flexilearn/` free of experiment-specific code, and free of new hard dependencies —
  optional ones go behind an extra with a clear `ImportError` message (see `write_mp4`).
* A bug fix comes with a regression test that fails before the fix.

## Known Issues / In-Progress

* `Group.run()` passes outputs sequentially through all models in the dict — intentional
  for pipelines, but it means model order in config matters.
* See `docs/dev/KNOWN_ISSUES.md` for tracked open issues and `docs/dev/ROADMAP.md` for
  planned work.
