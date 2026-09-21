# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-09-14

First public release.

### Added

- **Registry + discovery pattern** — six decorators (`register_model`, `register_task`,
  `register_dataset`, `register_dataprep`, `register_loss`, `register_optimizer`) and
  `discover_definitions()`, which imports everything under `definitions/` at startup.
- **Engine / Task / Group / DataSource** — a curriculum of tasks, each iteration driven by
  a user-implemented `loop()`; groups own one `nn.ModuleDict` plus one optimizer with
  gradient accumulation, freezing, and per-task overrides.
- **Config-hash run identity** — checkpoints under `outputs/<sha256-prefix>/`, with
  `attempt_resume` restoring groups, data sources, RNG state and task counters.
- **Selective checkpointing** — a `checkpointing:` inclusion dictionary with per-group
  `weights: full | trainable | none` policies, plus an `export()` hook for adapters.
- **Failure policies** — per-task `upon_failure: exit | retry | continue | until`, with
  evaluation-driven termination and parameter restore on retry.
- **MLflow tracking** — encapsulated in a single `Tracker`; scalars, gradients,
  histograms, system metrics, dataset lineage, provenance tags, nested per-task runs, and
  run reattachment on resume. A no-op when `logging.enabled` is false.
- **Model registry CLI** — `flexilearn-registry list | promote | promote-best`.
- **Simulations as data sources** — `EnvFactory`, `borrow_env`, `rollout`, `Episode`,
  `PolicyFailure` and `write_mp4`, with no simulator dependency in the framework.
- **`examples/minimal/`** — a runnable synthetic-regression experiment, and `smoke_test.py`
  for an end-to-end train → checkpoint → resume check.
- Optional extras: `gpu` (MLflow GPU system metrics) and `video` (`write_mp4` encoding).
- `py.typed` marker, so the package's annotations reach downstream type checkers.

### Fixed

- A group declared without an `optimizer:` block now builds as an inference-only unit
  instead of raising `AttributeError` — which is what `Group`'s own zero-trainable-params
  error message had been advising all along.
- Checkpoints are written with CPU tensors and loaded with `map_location="cpu"`, so a
  checkpoint saved on a CUDA host can be resumed or inspected on a CPU-only one.
- `Task.checkpoint_flag` no longer raises `ValueError` from `max([])` for a task that
  registers no groups when `cumulative_checkpointing_counter` is false.
- An explicit `num_workers: null` in a data source's loader no longer raises `TypeError`;
  null is treated as 0, matching the schema's documented meaning.
- `discover_definitions()` resolves `definitions/` from an explicit path,
  `$FLEXILEARN_DEFINITIONS`, or the working directory before falling back to the package
  directory, and warns instead of raising when none exists — an installed wheel was
  previously unusable because the CLI always raised `FileNotFoundError`.

[Unreleased]: https://github.com/DoubleH7/flexilearn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DoubleH7/flexilearn/releases/tag/v0.1.0
