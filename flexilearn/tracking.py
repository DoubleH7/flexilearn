# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Experiment tracking — the single, encapsulated home for MLflow.

``Tracker`` is the **only** object in the framework that imports or calls
``mlflow``. Every other module (``Engine``, ``RuntimeState``, ``Task``,
``Group``) talks to this small API and never touches MLflow directly. That
keeps the integration swappable and testable, and lets the whole thing become a
no-op when tracking is disabled — without requiring MLflow to be importable.

Design notes:
  * **Lazy import.** ``mlflow`` is imported inside ``start()``, only when
    ``logging.enabled`` is true, so disabled / offline runs pay nothing.
  * **Reattach on resume.** Each run is tagged ``config_hash=<artifact id>``.
    ``start()`` searches the experiment for an existing run with that tag and
    *continues* it (so a resumed training run appends to the same MLflow run)
    rather than forking a new one. This mirrors the config-hash identity that
    already keys the ``outputs/<hash>/`` checkpoint directories.
  * **Failure-tolerant.** A tracking error never aborts training: ``start()``
    disables tracking and warns; per-metric logging swallows and debug-logs.
"""

import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config_models import LoggingConfig
from .monitor import debug, info, warn

# MLflow's param values are length-limited; keep well under the floor.
_MAX_PARAM_LEN = 250


class Tracker:
    """Encapsulates all MLflow calls behind a small, no-op-safe API."""

    def __init__(self, cfg: LoggingConfig, default_experiment: str) -> None:
        self._cfg = cfg
        self._enabled = bool(cfg.enabled)
        self._default_experiment = default_experiment
        self.scalar_interval = cfg.scalar_interval
        self.gradient_interval = cfg.gradient_interval
        self.histogram_interval = cfg.histogram_interval

        # Richer-tracking switches (see LoggingConfig).
        self._system_metrics = bool(cfg.system_metrics)
        self._log_lr = bool(cfg.log_learning_rate)
        self._log_throughput = bool(cfg.log_throughput)
        self._log_dataset_lineage = bool(cfg.log_dataset_lineage)
        self._log_git = bool(cfg.log_git_info)

        # Registry switches, read by RuntimeState when packaging models.
        self.register_models = bool(cfg.register_models)
        self.registered_model_name = cfg.registered_model_name or default_experiment

        self._mlflow = None  # the imported module, once started
        self._active = False  # True only between start() and end() (parent run)
        self._run_id: str | None = None  # active run (task child while one is open)
        self._parent_run_id: str | None = None  # the experiment-level run
        self._config_hash: str | None = None
        self._task_active = False  # True while a per-task child run is open

    @property
    def run_id(self) -> str | None:
        """The active MLflow run id, or None when tracking is off."""
        return self._run_id

    @property
    def log_throughput(self) -> bool:
        """Whether the Engine should compute & log step throughput."""
        return self._enabled and self._log_throughput

    # ------------------
    # Lifecycle
    # ------------------
    def start(self, config_hash: str, params: dict[str, Any] | None = None) -> None:
        """Open (or reattach to) the MLflow run for this experiment.

        A tracking failure here disables tracking and continues training.
        """
        if not self._enabled:
            debug("Tracking disabled (logging.enabled=false) — Tracker is a no-op.")
            return
        self._config_hash = config_hash
        try:
            import mlflow

            self._mlflow = mlflow
            if self._cfg.tracking_uri:
                mlflow.set_tracking_uri(self._cfg.tracking_uri)
            # Registry can live on a different backend than the tracking store
            # (e.g. a shared model registry); default to the tracking URI.
            mlflow.set_registry_uri(self._cfg.registry_uri or self._cfg.tracking_uri)
            experiment = self._cfg.experiment or self._default_experiment
            mlflow.set_experiment(experiment)

            # System metrics (GPU/CPU/RAM) are sampled on a background thread for
            # the whole process once enabled; best-effort, needs psutil/pynvml.
            if self._system_metrics:
                try:
                    mlflow.enable_system_metrics_logging()
                    info("MLflow: system metrics logging enabled")
                except Exception as exc:  # noqa: BLE001
                    warn(f"MLflow system metrics unavailable ({exc}) — skipping.")

            existing = self._find_run(config_hash, experiment)
            if existing is not None:
                mlflow.start_run(run_id=existing)
                info(f"MLflow: reattached to run {existing[:8]} (config_hash={config_hash})")
            else:
                mlflow.start_run(run_name=f"{experiment}-{config_hash[:8]}")
                mlflow.set_tag("config_hash", config_hash)
                info(f"MLflow: started run for config_hash={config_hash}")
                # Params and provenance tags are immutable/identifying per run, so
                # set them only on a fresh run (a resume reattaches and appends).
                self._log_params(params or {})
                self._set_provenance_tags(params or {})

            self._active = True
            self._run_id = mlflow.active_run().info.run_id
            self._parent_run_id = self._run_id
        except Exception as exc:  # noqa: BLE001 — tracking must never abort training
            warn(f"MLflow start failed ({exc}) — continuing without tracking.")
            self._enabled = False
            self._active = False

    def end(self, status: str = "FINISHED") -> None:
        if not self._active:
            return
        # Close any still-open task child run first (defensive; the engine
        # normally pairs start/end_task_run itself).
        if self._task_active:
            self.end_task_run(status=status)
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow end_run failed: {exc}")
        finally:
            self._active = False

    # ------------------
    # Per-task (nested) runs — one MLflow run per curriculum task
    # ------------------
    def start_task_run(self, task_name: str, params: dict[str, Any] | None = None) -> None:
        """Open (or reattach to) a nested child run for one curriculum task.

        Metrics/artifacts logged while it is open land on the **task's** run
        rather than the experiment-level parent. Reattaches (resume) when a child
        run already exists for this ``config_hash`` + ``task`` pair, so a resumed
        task continues its own run. Failure-tolerant: on error, task metrics fall
        back to the parent run rather than aborting training.
        """
        if not self._active:
            return
        try:
            experiment = self._cfg.experiment or self._default_experiment
            existing = self._find_task_run(self._config_hash, task_name, experiment)
            if existing is not None:
                self._mlflow.start_run(run_id=existing, nested=True)
                info(f"MLflow: reattached task run '{task_name}' ({existing[:8]})")
            else:
                self._mlflow.start_run(
                    run_name=f"{experiment}-{self._config_hash[:8]}-{task_name}", nested=True
                )
                self._mlflow.set_tags({"config_hash": self._config_hash, "task": task_name})
                info(f"MLflow: started task run '{task_name}'")
                if params:
                    self._log_params({"task": params})
            self._task_active = True
            self._run_id = self._mlflow.active_run().info.run_id
        except Exception as exc:  # noqa: BLE001 — never abort training
            warn(f"MLflow start_task_run('{task_name}') failed ({exc}) — using parent run.")
            self._task_active = False

    def end_task_run(self, status: str = "FINISHED") -> None:
        """Close the current task child run; the parent run becomes active again."""
        if not self._task_active:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow end task run failed: {exc}")
        finally:
            self._task_active = False
            self._run_id = self._parent_run_id  # mlflow pops back to the parent

    def _find_task_run(self, config_hash: str, task_name: str, experiment: str) -> str | None:
        """run_id of an existing child run for this config_hash + task, if any."""
        exp = self._mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return None
        runs = self._mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.config_hash = '{config_hash}' and tags.task = '{task_name}'",
            max_results=1,
            output_format="list",
        )
        return runs[0].info.run_id if runs else None

    def _find_run(self, config_hash: str, experiment: str) -> str | None:
        """Return the run_id of an existing run carrying this config_hash tag."""
        exp = self._mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return None
        runs = self._mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.config_hash = '{config_hash}'",
            max_results=1,
            output_format="list",
        )
        return runs[0].info.run_id if runs else None

    # ------------------
    # Scalars
    # ------------------
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log one metric unconditionally (caller has already decided to)."""
        if not self._active:
            return
        try:
            self._mlflow.log_metric(key, float(value), step=int(step))
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow log_metric('{key}') failed: {exc}")

    def maybe_log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a metric only on the configured ``scalar_interval``."""
        if _fires(step, self.scalar_interval):
            self.log_scalar(key, value, step)

    def maybe_log_lr(self, group_name: str, optimizer, step: int) -> None:
        """Log the group's optimizer learning rate(s) on ``scalar_interval``.

        Called from ``Group.update`` beside the gradient hook. With a single
        param group it logs ``lr/<group>``; with several (e.g. per-layer decay)
        it logs one series per group as ``lr/<group>/pg{i}``.
        """
        if not self._active or not self._log_lr:
            return
        if not _fires(step, self.scalar_interval):
            return
        try:
            param_groups = optimizer.param_groups
            if len(param_groups) == 1:
                self.log_scalar(f"lr/{group_name}", float(param_groups[0]["lr"]), step)
            else:
                for i, pg in enumerate(param_groups):
                    self.log_scalar(f"lr/{group_name}/pg{i}", float(pg["lr"]), step)
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow log lr for '{group_name}' failed: {exc}")

    # ------------------
    # Gradients / histograms (called from Group.update with live grads)
    # ------------------
    def maybe_log_gradients(
        self, group_name: str, named_params: Iterable[tuple[str, nn.Parameter]], step: int
    ) -> None:
        """Log the group's global grad norm (and, optionally, a grad histogram).

        ``gradient_interval`` gates the scalar norm; ``histogram_interval`` gates
        the (best-effort, matplotlib) histogram independently. Called from
        ``Group.update`` right before the optimizer zeroes the gradients.
        """
        if not self._active:
            return
        want_norm = _fires(step, self.gradient_interval)
        want_hist = _fires(step, self.histogram_interval)
        if not (want_norm or want_hist):
            return

        grads = [p.grad.detach() for _, p in named_params if p.grad is not None]
        if not grads:
            return

        if want_norm:
            total = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
            self.log_scalar(f"grad_norm/{group_name}", total.item(), step)
        if want_hist:
            self._log_grad_histogram(group_name, grads, step)

    def _log_grad_histogram(self, group_name: str, grads, step: int) -> None:
        """Best-effort gradient histogram as an MLflow figure (no native API)."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            flat = torch.cat([g.float().flatten() for g in grads]).cpu().numpy()
            fig, ax = plt.subplots()
            ax.hist(flat, bins=64)
            ax.set_title(f"grad histogram — {group_name} (step {step})")
            self._mlflow.log_figure(fig, f"grad_hist/{group_name}/step_{step}.png")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 — matplotlib optional / best-effort
            debug(f"grad histogram skipped for '{group_name}': {exc}")

    # ------------------
    # Artifacts
    # ------------------
    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
            debug(f"MLflow: logged artifact {path}")
        except Exception as exc:  # noqa: BLE001
            warn(f"MLflow log_artifact('{path}') failed: {exc}")

    def log_artifacts(self, local_dir, artifact_path: str | None = None) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
            debug(f"MLflow: logged artifacts from {local_dir}")
        except Exception as exc:  # noqa: BLE001
            warn(f"MLflow log_artifacts('{local_dir}') failed: {exc}")

    # ------------------
    # Datasets (lineage)
    # ------------------
    def log_dataset(self, name: str, dataset: Any, context: str = "train") -> None:
        """Record a data source as an MLflow input for lineage.

        Best-effort: a genuine ``datasets.Dataset`` is logged via
        ``mlflow.data.from_huggingface`` (so the UI shows a first-class dataset);
        anything else falls back to descriptor tags (class, size, source path).
        Called once per source from ``DataSource.initialize`` (core only — the
        dataset object itself never imports mlflow).
        """
        if not self._active or not self._log_dataset_lineage:
            return
        try:
            hf = _as_hf_dataset(dataset)
            if hf is not None:
                ds = self._mlflow.data.from_huggingface(hf, name=name)
                self._mlflow.log_input(ds, context=context)
                debug(f"MLflow: logged HF dataset input '{name}'")
                return
            for key, val in _describe_dataset(dataset).items():
                self._mlflow.set_tag(f"dataset.{name}.{key}", val)
            debug(f"MLflow: tagged dataset descriptor for '{name}'")
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow log_dataset('{name}') failed: {exc}")

    # ------------------
    # Model packaging & registry
    # ------------------
    def log_pytorch_model(
        self, model: nn.Module, name: str, registered_name: str | None = None
    ) -> str | None:
        """Log a plain ``nn.Module`` as an MLflow pytorch model; return its URI.

        Produces a proper ``MLmodel`` (loadable with ``mlflow.pytorch.load_model``)
        and, when ``registered_name`` is given, creates a registry version. ``name``
        is the per-run artifact name and must be flat (MLflow 3.x forbids ``/ : . %``
        in model names).
        """
        if not self._active:
            return None
        try:
            try:
                # MLflow 3.x: the param is `name`; older releases use `artifact_path`.
                info_ = self._mlflow.pytorch.log_model(model, name=name)
            except TypeError:
                info_ = self._mlflow.pytorch.log_model(model, artifact_path=name)
            model_uri = info_.model_uri
            debug(f"MLflow: logged pytorch model at {model_uri}")
            if registered_name:
                self.register_model(model_uri, registered_name)
            return model_uri
        except Exception as exc:  # noqa: BLE001
            warn(f"MLflow log_pytorch_model('{name}') failed: {exc}")
            return None

    def register_model(self, model_uri: str, name: str) -> None:
        """Register ``model_uri`` (a ``runs:/…`` or model URI) as a new version.

        Works for both flavored models and raw artifact dirs (e.g. a logged LoRA
        adapter): ``register_model`` records the source without requiring an
        ``MLmodel`` file, so exported adapters become versioned without bundling
        the frozen base weights.
        """
        if not self._active:
            return
        try:
            mv = self._mlflow.register_model(model_uri, name)
            info(f"MLflow: registered '{name}' as version {mv.version}")
        except Exception as exc:  # noqa: BLE001
            warn(f"MLflow register_model('{name}') failed: {exc}")

    # ------------------
    # Params / provenance
    # ------------------
    def _log_params(self, params: dict[str, Any]) -> None:
        flat = _flatten(params)
        if not flat:
            return
        try:
            self._mlflow.log_params(flat)
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow log_params failed: {exc}")

    def _set_provenance_tags(self, params: dict[str, Any]) -> None:
        """Tag a fresh run with git state and human-readable model/task types.

        The git tags anchor reproducibility (which commit produced this run); the
        type tags make the Experiments table filterable without unpacking params.
        ``uv.lock`` is logged as an artifact so the exact env can be rebuilt.
        """
        try:
            tags = _model_task_tags(params)
            if self._log_git:
                tags.update(_git_tags())
            if tags:
                self._mlflow.set_tags(tags)
            if self._log_git:
                lockfile = Path("uv.lock")
                if lockfile.exists():
                    self._mlflow.log_artifact(str(lockfile), artifact_path="repro")
        except Exception as exc:  # noqa: BLE001
            debug(f"MLflow provenance tagging failed: {exc}")


def _fires(step: int, interval: int) -> bool:
    """True when ``interval`` is enabled (>0) and ``step`` lands on it."""
    return interval > 0 and step % interval == 0


def _as_hf_dataset(dataset: Any):
    """Return an underlying ``datasets.Dataset`` if one is reachable, else None.

    A registered dataset may *be* an HF dataset or may *wrap* one on a common
    attribute (``hf_dataset`` / ``dataset`` / ``_dataset``). Detected by
    duck-typing (``features`` + ``__len__``) to avoid importing ``datasets``.
    """
    candidates = [dataset] + [
        getattr(dataset, a, None) for a in ("hf_dataset", "dataset", "_dataset", "ds")
    ]
    for cand in candidates:
        if cand is None:
            continue
        if hasattr(cand, "features") and hasattr(cand, "__len__") and hasattr(cand, "column_names"):
            return cand
    return None


def _describe_dataset(dataset: Any) -> dict[str, str]:
    """A small, tag-safe descriptor of a dataset (class, size, source path)."""
    out: dict[str, str] = {"class": type(dataset).__name__}
    try:
        out["size"] = str(len(dataset))
    except Exception:  # noqa: BLE001 — datasets need not be sized
        pass
    for attr in ("path", "repo_id", "name", "source", "dataset_path", "root"):
        val = getattr(dataset, attr, None)
        if isinstance(val, (str, int)):
            out[attr] = str(val)[:_MAX_PARAM_LEN]
    return out


def _model_task_tags(params: dict[str, Any]) -> dict[str, str]:
    """Derive filterable ``model_types`` / ``task_types`` tags from the config."""
    tags: dict[str, str] = {}
    curriculum = params.get("curriculum") or {}
    task_types = sorted(
        {t.get("type") for t in curriculum.values() if isinstance(t, dict) and t.get("type")}
    )
    if task_types:
        tags["task_types"] = ",".join(task_types)[:_MAX_PARAM_LEN]

    model_types = set()
    for group in (params.get("groups") or {}).values():
        if not isinstance(group, dict):
            continue
        for model in (group.get("models") or {}).values():
            if isinstance(model, dict) and model.get("type"):
                model_types.add(model["type"])
    if model_types:
        tags["model_types"] = ",".join(sorted(model_types))[:_MAX_PARAM_LEN]
    return tags


def _git_tags() -> dict[str, str]:
    """Best-effort git provenance: commit sha, branch, and dirty flag."""

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:  # noqa: BLE001 — not a repo / git absent
            return None

    tags: dict[str, str] = {}
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return tags  # not a git checkout
    tags["git_sha"] = sha
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        tags["git_branch"] = branch
    status = _git("status", "--porcelain")
    tags["git_dirty"] = "true" if status else "false"
    return tags


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested config dict into dotted ``key: str`` MLflow params."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = str(obj)[:_MAX_PARAM_LEN]
    elif obj is not None:
        out[prefix] = str(obj)[:_MAX_PARAM_LEN]
    return out


__all__ = ["Tracker"]
