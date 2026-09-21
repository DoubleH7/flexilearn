# Flexilearn

[![CI](https://github.com/DoubleH7/flexilearn/actions/workflows/ci.yml/badge.svg)](https://github.com/DoubleH7/flexilearn/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Flexilearn is a modular, **configuration-driven** training framework built on PyTorch,
designed for rapid experimentation and reproducible research workflows.

Every component — models, optimizers, datasets, preprocessing stages, losses, and
training tasks — is **registered with a decorator** and **wired together entirely from a
single `config.yml`**. Swapping a model, changing an optimizer, or adding a
preprocessing step needs no code changes: you reference a registry key in the config.

## Quickstart

```bash
git clone https://github.com/DoubleH7/flexilearn.git
cd flexilearn
uv sync

# Train the bundled example: a linear fit on synthetic data, CPU-only, ~1 second.
uv run flexilearn examples/minimal/config.yml --definitions examples/minimal/definitions
```

That writes checkpoints to `outputs/<config-hash>/` and MLflow metrics to `./mlruns`
(view them with `mlflow ui`). Read
[`examples/minimal/config.yml`](examples/minimal/config.yml) alongside
[`examples/minimal/definitions/synthetic.py`](examples/minimal/definitions/synthetic.py) —
together they are the whole mental model: registered components on one side, the config
that wires them on the other.

With no `--definitions`, the CLI looks for `$FLEXILEARN_DEFINITIONS`, then `./definitions`
in the working directory — so in your own project, `uv run flexilearn config.yml` is enough.

```bash
uv run pytest                    # the test suite
uv run python smoke_test.py      # end-to-end check (train → checkpoint → resume)
uv run ruff check . && uv run ruff format --check .
```

The virtualenv is managed by `uv` — always use `uv run` rather than activating it
manually.

### Installing as a dependency

```bash
pip install flexilearn            # or: uv add flexilearn
pip install flexilearn[gpu]       # + GPU system metrics for MLflow
pip install flexilearn[video]     # + rollout video encoding (flexilearn.sim.write_mp4)
```

---

## Design goals

* Modular experimentation with minimal boilerplate
* Clean separation between data, models, and training logic
* Reproducible, configuration-driven execution (config-hash run identity)
* Stability-oriented training (gradient accumulation, selective checkpointing)
* Automated checkpointing, experiment tracking, system metrics, and a model registry (MLflow)

---

## The registry + discovery pattern

This is the framework's defining idea. Each component type has a global registry in
`flexilearn/registry.py`. Components self-register at import time via decorators:

```python
@register_model("my_model")
class MyModel(nn.Module): ...


@register_task("my_task")
class MyTask(Task): ...
```

Registration is **automatic**: `flexilearn/discovery.py` walks the `definitions/` tree
and imports every module so the decorators run. The entry point calls
`discover_definitions()` before the engine starts. **Dropping a new file into
`definitions/` is enough** — there is no manual import list to maintain.

| Component       | Decorator                  | Location                |
| --------------- | -------------------------- | ----------------------- |
| Model           | `@register_model("key")`   | `definitions/models/`   |
| Task            | `@register_task("key")`    | `definitions/tasks/`    |
| Dataset         | `@register_dataset("key")` | `definitions/data/`     |
| Data prep stage | `@register_dataprep("key")`| `definitions/data/`     |
| Loss            | `@register_loss("key")`    | `definitions/losses/`   |

The `type:` field in any config block maps to a registry key; every extra field under it
is passed verbatim as `**kwargs` to the constructor (via Pydantic `extra='allow'`).

---

## Architecture

### RuntimeState (`flexilearn/runtime_state.py`)

The singleton that owns all live objects. It lazily builds `Group` and `DataSource`
instances on first access, manages checkpointing, and owns the experiment `Tracker`.
**Run identity is a SHA-256 hash of the config** — checkpoints live under
`outputs/<hash>/`, and the same hash reattaches an MLflow run on resume.

### Group (`flexilearn/group.py`)

A learning unit owning **one `nn.ModuleDict` + exactly one optimizer**.

* `group.run(x)` — forward pass. With no `model=` argument it **chains every model in
  config order** (the output of each feeds the next); `group.run(x, model="name")` runs a
  single named model.
* `group.update(loss)` — backward + optimizer step with gradient accumulation.
* Freeze / unfreeze via `set_trainable()`; the optimizer is built **only over
  `requires_grad=True` parameters**, so a frozen-backbone / PEFT (LoRA) setup optimizes
  just the adapter. A group with zero trainable parameters raises a clear error at
  construction, naming the group.

Groups are declared under `groups:` and accessed in tasks via `self.get_group("name")`.

### Task (`flexilearn/task.py`)

Abstract base; implement `loop()` to define one training iteration. The engine calls
`loop()` repeatedly per the curriculum. Access groups and data through `self.get_group()`
/ `self.get_source()`, and log metrics with `self.log_scalar("loss", value)`. Register
with `@register_task("name")`.

### DataSource + Stage (`flexilearn/data.py`)

A `DataSource` wraps a dataset + DataLoader + a pipeline of **`Stage`** transforms chained
as Python generators. `data_source.get_next()` pulls one item through the full pipeline.
Stages (tokenizer, windowing, shuffle buffer, batching, …) keep preprocessing modular and
declared in config. Declared under `data_sources:`.

### Engine (`flexilearn/engine.py`)

Iterates the `curriculum` tasks in order, calling `task._step()` up to `iterations` times
per epoch. Handles dataset exhaustion (advances epochs), per-task failure policy
(`exit`/`retry`/`continue`), and triggers checkpointing when a task's
`checkpoint_flag` trips. Brackets the whole run in an MLflow run
(`FINISHED`/`FAILED`).

---

## Configuration (`config.yml`)

```yaml
experiment:    # name, seed, attempt_resume, allow_fresh_on_resume_failure
curriculum:    # ordered dict of named tasks; type, iterations, epochs, checkpointing_interval, ...
groups:        # named groups; models dict, optimizer, settings (grad_accumulation, device, is_trainable)
data_sources:  # named sources; dataset, loader, preparations (Stage pipeline)
logging:       # MLflow tracking — see below
checkpointing: # what each checkpoint must contain (optional; defaults to full)
```

---

## Monitoring with MLflow

Training metrics, parameters, and artifacts are tracked with **MLflow**, configured
under `logging:`:

```yaml
logging:
  enabled: true              # set false to disable tracking entirely (no MLflow import)
  tracking_uri: ./mlruns     # local file store; or a server URI / sqlite:///mlflow.db
  experiment: my_experiment  # defaults to experiment.name
  scalar_interval: 10        # log loss / LR every N steps
  gradient_interval: 50      # log per-group global grad norm every N steps
  histogram_interval: 0      # grad histograms every N steps (0 = off)

  # --- richer tracking (all best-effort; never abort training) ---
  system_metrics: false      # CPU/RAM/disk/net + per-GPU metrics (GPU needs nvidia-ml-py)
  log_learning_rate: true    # per-group optimizer LR (lr/<group>)
  log_throughput: true       # training speed (perf/steps_per_sec, perf/sec_per_step)
  log_dataset_lineage: true  # each data source logged as an MLflow input
  log_git_info: true         # commit/branch/dirty tags + uv.lock artifact (see note below)

  # --- model registry (see "Model registry" below) ---
  register_models: false     # register trained models on each checkpoint
  registered_model_name:     # defaults to experiment.name
  registry_uri:              # defaults to tracking_uri
```

What gets logged:

* **Params** — the resolved config (flattened), logged once per run.
* **Scalars** — tasks call `self.log_scalar(name, value)`, gated by `scalar_interval`;
  plus per-group learning rate (`lr/<group>`) and throughput (`perf/*`).
* **Gradients** — per-group global grad norm (`grad_norm/<group>`) every
  `gradient_interval`; optional grad histograms every `histogram_interval`.
* **System metrics** — `system/*` (CPU, memory, disk, network, and per-GPU
  utilization/memory/power when `nvidia-ml-py` is installed), sampled on a background
  thread when `system_metrics: true`.
* **Dataset lineage** — each data source is logged as an MLflow input (a first-class
  dataset for HF datasets, descriptor tags otherwise).
* **Provenance tags** — `git_sha` / `git_branch` / `git_dirty`, plus `model_types` /
  `task_types` for filtering, and the `uv.lock` archived as an artifact.
* **Artifacts** — every `.fl` checkpoint and any model `export()` output (e.g. LoRA
  adapters) are logged alongside their metrics.

**Run identity & resume.** Each run is tagged `config_hash=<artifact id>`. On startup the
`Tracker` searches the experiment for that tag and **reattaches to the same run** if it
exists — so a resumed training run *continues* its MLflow run rather than forking a new
one. This mirrors the config-hash identity that keys `outputs/<hash>/`. (The hash excludes
the `logging:` block, so tweaking log settings appends to the same run instead of forking.)

View runs with `mlflow ui` (or `mlflow ui --backend-store-uri ./mlruns`). MLflow's local
file store is functional but deprecated (recent versions emit a `FutureWarning` on every
run); for heavier use — and for the **model registry**, which needs a database backend —
point `tracking_uri` at `sqlite:///mlflow.db` or a tracking server.

> **A note on what leaves your machine.** `log_git_info` is **on by default**: every fresh
> run tags `git_sha` / `git_branch` / `git_dirty` and uploads `uv.lock` as an artifact.
> Against the default local `./mlruns` store nothing leaves the machine. If you point
> `tracking_uri` at a shared or hosted server, that means your branch names and resolved
> dependency set go upstream too — set `log_git_info: false` if that matters to you.

### Model registry

With `register_models: true`, every checkpoint also registers the trained model as a new
MLflow **model version**: plain `nn.Module` models via the pytorch flavor; models with an
`export()` hook (e.g. LoRA adapters) as artifact-sourced versions, so the frozen base
weights are not re-serialized. Manage the lifecycle with the bundled CLI:

```bash
flexilearn-registry list --config config.yml
flexilearn-registry promote <name> <version> production       # alias-based; --legacy-stage for old servers
flexilearn-registry promote-best <name> --metric eval/mean_return
```

> **Implementation note.** All MLflow access in the training path is encapsulated in a
> single `Tracker` (`flexilearn/tracking.py`); the standalone `flexilearn-registry` CLI is
> the only other module that imports `mlflow`. Crucially, **nothing under `definitions/`
> ever imports `mlflow`** — user models declare intent through the plain `export(path)`
> hook, and core does the logging. Tracking failures never abort training (they disable
> tracking and warn), and `enabled: false` makes the Tracker a complete no-op.

---

## Checkpointing

Checkpoints are saved under `outputs/<config-hash>/checkpoint-full-<timestamp>.fl` as
torch-serialized dicts. The `checkpointing:` block is an **inclusion dictionary**
declaring what each checkpoint must contain; omitting it yields a full, everything-included
checkpoint (RNG state, task counters, datasource counters, group weights + optimizer).

Per-group **weight policy** scopes the heavy part:

| `weights:`  | Persists                                          | Use for                                   |
| ----------- | ------------------------------------------------- | ----------------------------------------- |
| `full`      | every parameter + buffer (default)                | from-scratch models (weights are the only copy) |
| `trainable` | only `requires_grad=True` params (e.g. a LoRA adapter) | PEFT / frozen-backbone — base is rebuilt by the constructor on resume |
| `none`      | no weights                                        | a group reconstructed entirely at build time (frozen reference model) |

For a large frozen-backbone model fine-tuned with LoRA, `weights: trainable` is the
difference between serializing the entire base model on every save and serializing just
the adapter, optimizer state, and counters — often two orders of magnitude smaller, since
the frozen base is reproduced by the model's own constructor (`from_pretrained`) on
resume. Set `attempt_resume: true` to auto-resume from the latest checkpoint for the same
config hash.

Checkpoint tensors are written on CPU and loaded with `map_location="cpu"`, so a
checkpoint saved on a CUDA host can be resumed or inspected on a CPU-only one.

### Export hook

If a model defines `def export(self, path: pathlib.Path) -> None:`, it is called on every
checkpoint save with a fresh directory `outputs/<hash>/export-<ts>/<group>/<model>/` —
write adapters/config into it (e.g. `self.model.save_pretrained(path)`). Models without
`export` are skipped; an export failure is logged but never aborts checkpointing. Exports
are also logged to MLflow as artifacts.

---

## Extending the framework — contracts worth knowing

* **Tasks own the device move.** The data pipeline never calls `.to(device)`.
  `Group.run`'s chaining path (`model=None`) assumes a **Tensor** and moves it; the
  single-model path (`model="name"`) moves nothing. For a non-tensor batch (e.g. a VLM
  processor dict), move tensors in the task (`group.device` gives the target) and call the
  model directly via `group.models["name"]`.
* **Nested config dicts arrive whole.** A nested block under `type:` reaches the
  constructor as a plain `dict` (with nested lists intact) — e.g.
  `lora: { r: 8, target_modules: [q_proj, v_proj] }` → `__init__(..., lora={...})`.
* **Raw module access.** `group.models` is a public `nn.ModuleDict`; use
  `group.models["name"]` for custom methods like `.generate()`.
* **A group literally named `default` is required** (config validation).

---

## Examples

* **[`examples/minimal/`](examples/minimal/)** — a linear fit on synthetic data: one
  registered dataset, one registered task, and the config that wires them to the built-in
  `linear` model and `mse` loss. CPU-only, no downloads, about a second to run. This is
  the intended starting point — copy the directory and replace its two components.
  ```bash
  uv run flexilearn examples/minimal/config.yml --definitions examples/minimal/definitions
  ```
* **[`smoke_test.py`](smoke_test.py)** — the same synthetic regression driven from Python
  instead of YAML, asserting that Engine + RuntimeState + Task wiring, counter
  persistence, and checkpoint resume all hold. Useful both as a health check and as a
  worked example of building a `Config` in code.
  ```bash
  uv run python smoke_test.py
  ```

`definitions/` in the repository root is deliberately empty — it is the extension point
for your own components, and `discover_definitions()` imports everything under it at
startup.

---

## Known limitations

* Prep stages run in the main process, not in DataLoader workers — a heavy stage is
  not parallelised by raising `num_workers`. (`shuffle` and `num_workers: 0` are both
  supported: `shuffle` is seeded per-epoch in `DataSource.initialize()`, and every
  `EnvFactory` source is required to declare `num_workers: 0`.)
* DataLoader-level batching uses PyTorch's default collate, which can't batch multimodal
  dicts / PIL images — set `loader.batch_size: 0` and batch inside a custom stage.
* Prep stages are re-instantiated each epoch (`reset_state()` re-runs their `__init__`), so
  a stage that loads a heavy processor reloads it per epoch.
* **Mid-epoch resume restarts the epoch's data.** A `DataSource` checkpoints its reset
  counter and constructor kwargs, not its iterator position — so resuming mid-epoch
  replays the epoch from the top while the task's restored counters say otherwise.
  Checkpoint on epoch boundaries if exact sample coverage matters.
* **Persistent envs are not closed at epoch boundaries.** `reset_datasources()` rebuilds a
  source's dataset without calling `close()` on the previous `EnvFactory`, so a
  `persistent: true` env is dropped un-closed once per epoch.

See [`docs/dev/KNOWN_ISSUES.md`](docs/dev/KNOWN_ISSUES.md) for the full tracked list.

---

## Roadmap

* **System** — parameter-precision control, acceleration, `torch.compile`, thread/memory ceilings.
* **Curriculum** — task success/failure status driving `upon_failure`.
* **Group** — parameterization abstraction, EMA, per-group learning history.
* **Model** — a "builder" type for automatic ensembling from config.
* **Environment** — a first-class abstraction for simulations / APIs / terminals.
  *Partially realized* today via the env-as-datasource pattern: environments live in the
  data layer (`EnvFactory`) and tasks drive the closed loop.
* **Monitoring** — ✓ system-resource monitoring (CPU/GPU/mem via MLflow system metrics);
  still planned: a warning system (e.g. email alerts).
* **Experimenting** — full MLflow data versioning, building on the dataset-lineage inputs
  now logged per run.
* **Export** — automatic export to hardware-friendly formats (e.g. gguf).

See [`docs/dev/ROADMAP.md`](docs/dev/ROADMAP.md) for the detailed work queue.

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development
setup, the checks CI runs, and how the registry pattern shapes a good pull request.
Security issues: please follow [`SECURITY.md`](SECURITY.md) rather than opening a public
issue.

---

## License

Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

```
Copyright 2026 Hesam Haddad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Flexilearn does not bundle or redistribute third-party source code; its runtime
dependencies remain under their own licenses, listed in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md).

## Citation

If you use Flexilearn in academic work, please cite it. The author's ORCID iD is
[0009-0006-5468-6484](https://orcid.org/0009-0006-5468-6484). GitHub renders
[`CITATION.cff`](CITATION.cff) into a ready-made citation via the *Cite this repository*
button, or use:

```bibtex
@software{haddad_flexilearn,
  author  = {Haddad, Hesam},
  title   = {Flexilearn: a configuration-driven PyTorch training framework},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/DoubleH7/flexilearn}
}
```
