# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
import datetime
import hashlib
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from .config_models import Config, DataSourceSpec, GroupSpec, TaskSpec
from .data import DataSource
from .group import Group
from .monitor import debug, error, info, success, warn
from .tracking import Tracker

if TYPE_CHECKING:
    from .task import Task

OUTPUT_DIRECTORY = Path("./outputs")
CHECKPOINT_PREFIX = "checkpoint"
CheckpointComponent = Literal["groups", "sources", "runtime", "full"]
TIME_FORMAT = "%Y-%m-%d-%H-%M"
CHECKPOINT_FORMAT_VERSION = 1


class RuntimeState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._sources: dict[str, DataSource] = {}
        self._source_specs: dict[str, DataSourceSpec] = self.config.data_sources
        self._groups: dict[str, Group] = {}
        self._group_specs: dict[str, GroupSpec] = self.config.groups
        self._tasks: dict[str, Task] = {}
        self._task_specs: dict[str, TaskSpec] = self.config.curriculum
        self._task_states = {}
        self.global_seed = self.config.experiment.seed
        # Inclusion dictionary: what each checkpoint must contain (defaults to
        # a full, everything-included checkpoint when omitted from config).
        self._ckpt_cfg = self.config.checkpointing
        # Experiment tracking — the only object that touches MLflow. A no-op
        # when logging.enabled is false. Groups receive a reference on build so
        # they can log gradients where the live grads exist (see _attach_tracker).
        self.tracker = Tracker(self.config.logging, default_experiment=self.config.experiment.name)

    def initialize(self):
        resumed = False
        if self.config.experiment.attempt_resume:
            resumed = self.attempt_checkpoint_load()
        if not resumed:
            info("Starting fresh experiment")
            debug(f"Seeding RNG with seed={self.global_seed}")
            torch.manual_seed(self.global_seed)
            np.random.seed(self.global_seed)
            random.seed(self.global_seed)
        # Open (or reattach to) the MLflow run for this config hash. Done here,
        # after any resume, so a resumed run continues the same MLflow run.
        self.tracker.start(
            config_hash=self.artifact_identifiers,
            params=self.config.model_dump(
                include={"experiment", "curriculum", "groups", "data_sources", "logging"},
                mode="json",
            ),
        )

    # ------------------
    # Global State Management
    # ------------------
    @property
    def _global_state(self) -> dict:
        """Capture Global state for resumability."""
        # RNG states
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

        return {"rng_states": rng_states, "global_seed": self.global_seed}

    def _load_global_state(self, state: dict):
        """Restore states from checkpoint_state_dict."""
        # RNG states
        rng_states = state["rng_states"]
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in rng_states:
            torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
        self.global_seed = state["global_seed"]

    # ------------------
    # Datasource Management
    # ------------------
    @property
    def _datasource_states(self) -> dict:
        """Collect state dicts from all datasources."""
        return {name: source.source_state_dict for name, source in self._sources.items()}

    def _load_datasource_states(self, s_states: dict):
        """Restore datasources from checkpoint state dicts."""
        for s_name, s_state in s_states.items():
            # Pass the spec to from_checkpoint so it can reconstruct properly
            self._sources[s_name] = DataSource.from_checkpoint(s_state, self._source_specs[s_name])

    def reset_datasources(self):
        """Reset all datasources to their initial state."""
        debug("Resetting all data sources for new epoch")
        for source in self._sources.values():
            source.reset_state()

    # ------------------
    # Group Management
    # ------------------
    @property
    def _group_states(self) -> dict:
        """Collect checkpoint state from all groups, each under its policy.

        A group uses its per-group override if one is declared, otherwise the
        default group policy from ``checkpointing.groups``.
        """
        states = {}
        for name, group in self._groups.items():
            policy = self._ckpt_cfg.groups.overrides.get(name, self._ckpt_cfg.groups)
            states[name] = group.checkpoint_state(policy)
        return states

    def _load_group_states(self, g_states: dict):
        """Restore groups from checkpoint state dicts."""
        for g_name, g_state in g_states.items():
            # Pass the spec to from_checkpoint so it can reconstruct properly
            group = Group.from_checkpoint(g_state, self._group_specs[g_name], g_name)
            group.set_tracker(self.tracker)
            self._groups[g_name] = group

    @property
    def groups_can_checkpoint(self) -> bool:
        return not any([g.unapplied_steps for g in self._groups.values()])

    # ------------------
    # Task Management
    # ------------------

    def get_task_states(self) -> dict:
        """Collect state dicts from all tasks."""
        if self._tasks:
            self._task_states = {name: task.task_state_dict for name, task in self._tasks.items()}
        return self._task_states

    def _load_task_states(self, states):
        for t_name, t_state in states.items():
            self._task_states[t_name] = t_state

    # ------------------
    # Lazy Builders
    # ------------------
    def get_source(self, s_name: str) -> DataSource:
        """Lazily builds sources."""
        assert s_name in self._source_specs, f"source {s_name} is not defined. check configuration."
        if s_name not in self._sources:
            debug(f"Building data source: '{s_name}'")
            spec = self._source_specs[s_name]
            source = self._build_source(spec)
            self._sources[s_name] = source
            # Record dataset lineage once, on first build (no-op when tracking or
            # lineage logging is off, or before the run is active on resume).
            self.tracker.log_dataset(s_name, source.dataset)
        return self._sources[s_name]

    def _build_source(self, spec: DataSourceSpec) -> DataSource:
        data_source = DataSource(datasource_spec=spec, seed=self.config.experiment.seed)
        data_source.initialize()
        return data_source

    def get_group(self, g_name: str) -> Group:
        """Lazily builds groups."""
        assert g_name in self._group_specs, f"group {g_name} is not defined. check configuration."
        if g_name not in self._groups:
            debug(f"Building group: '{g_name}'")
            spec = self._group_specs[g_name]
            self._groups[g_name] = self._build_group(spec, g_name)
        return self._groups[g_name]

    def _build_group(self, spec: GroupSpec, name: str) -> Group:
        group = Group(group_spec=spec, name=name)
        group.set_tracker(self.tracker)
        return group

    # ------------------
    # Checkpointing
    # ------------------
    @property
    def artifact_identifiers(self):
        """Generate a stable unique ID for experiment artifacts based on config."""
        definitive_params = self.config.model_dump(
            include={"groups", "data_sources", "experiment", "curriculum"},
            mode="json",
        )
        encoded = json.dumps(definitive_params, sort_keys=True).encode()
        unique_id = hashlib.sha256(encoded).hexdigest()[:8]
        return unique_id

    @property
    def checkpoint_path(self) -> Path:
        """Base path for saving checkpoints."""
        unique_id = self.artifact_identifiers
        output_dir = OUTPUT_DIRECTORY / f"{unique_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def find_checkpoints(self) -> list[Path] | None:
        """Find the checkpoints for this run in the output directory."""
        output_dir = self.checkpoint_path
        checkpoint_files = list(output_dir.glob(f"{CHECKPOINT_PREFIX}-*.fl"))
        if not checkpoint_files:
            return None
        return sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)

    def checkpoint_all(self):
        """Save system state, including only the components the checkpointing
        config asks for.

        ``groups`` is always written — its per-group counters (optimizer steps,
        window count, ...) are required for a correct resume, and the
        ``checkpointing.groups.weights`` policy already scopes the heavy part.
        ``globals`` / ``tasks`` / ``data_sources`` are tiny and toggled purely
        as a control / reproducibility choice.
        """
        checkpoint_data = {"groups": self._group_states}
        if self._ckpt_cfg.data_sources:
            checkpoint_data["data_sources"] = self._datasource_states
        if self._ckpt_cfg.globals:
            checkpoint_data["globals"] = self._global_state
        if self._ckpt_cfg.tasks:
            checkpoint_data["tasks"] = self.get_task_states()
        current_datetime = datetime.datetime.now()
        formatted_time = current_datetime.strftime(TIME_FORMAT)
        component: CheckpointComponent = "full"
        save_path = self.checkpoint_path / Path(
            CHECKPOINT_PREFIX + "-" + component + "-" + formatted_time
        )
        tmp_path = save_path.with_suffix(".tmp")
        fin_path = save_path.with_suffix(".fl")

        debug(f"Writing checkpoint to {tmp_path}")
        torch.save(checkpoint_data, tmp_path)
        os.rename(tmp_path, fin_path)
        success(f"Checkpoint saved: {fin_path}")

        # Track the snapshot alongside its metrics (no-op when tracking is off).
        self.tracker.log_artifact(fin_path, artifact_path="checkpoints")

        # Export when configured, or whenever registry is on (registration needs
        # the model artifacts logged first).
        if self._ckpt_cfg.export or self.tracker.register_models:
            self._export_models(formatted_time)

        for group in self._groups.values():
            group.reset_window()

    def _export_models(self, timestamp: str) -> None:
        """Run each model's optional ``export(path)`` hook alongside the .fl snapshot.

        If a model defines an ``export`` method it is called with a fresh
        directory ``outputs/<hash>/export-<timestamp>/<group>/<model>/``. This
        lets HF / PEFT models persist adapters (e.g. via ``save_pretrained``)
        without coupling the core to those libraries. Models that do not define
        ``export`` are skipped, so existing checkpoints are unaffected. An
        export failure is logged but does not abort checkpointing — the ``.fl``
        snapshot is already durable on disk.
        """
        export_root = self.checkpoint_path / f"export-{timestamp}"
        register = self.tracker.register_models
        run_id = self.tracker.run_id
        total_models = sum(len(group.models) for group in self._groups.values())
        for group_name, group in self._groups.items():
            for model_name, model in group.models.items():
                if hasattr(model, "export") and callable(model.export):
                    target = export_root / group_name / model_name
                    target.mkdir(parents=True, exist_ok=True)
                    try:
                        model.export(target)
                        success(f"Exported '{group_name}/{model_name}' to {target}")
                        artifact_path = f"exports/{timestamp}/{group_name}/{model_name}"
                        self.tracker.log_artifacts(target, artifact_path=artifact_path)
                        if register and run_id is not None:
                            self.tracker.register_model(
                                f"runs:/{run_id}/{artifact_path}",
                                self._registered_name(group_name, model_name, total_models),
                            )
                    except Exception as e:  # noqa: BLE001 — a failed export must not lose the checkpoint
                        error(f"export() failed for '{group_name}/{model_name}': {e}")
                elif register and isinstance(model, torch.nn.Module):
                    # No export() hook: package the live module as a flavored
                    # MLflow pytorch model so it gets a proper, registrable MLmodel.
                    # The artifact name must be flat (MLflow 3.x forbids '/ : .').
                    self.tracker.log_pytorch_model(
                        model,
                        name=f"model-{group_name}-{model_name}-{timestamp}",
                        registered_name=self._registered_name(group_name, model_name, total_models),
                    )

    def _registered_name(self, group_name: str, model_name: str, total_models: int) -> str:
        """Registry name for a model — the bare base when there's only one model,
        else suffixed by ``group/model`` so versions don't collide."""
        base = self.tracker.registered_model_name
        if total_models <= 1:
            return base
        return f"{base}-{group_name}-{model_name}"

    def attempt_checkpoint_load(self) -> bool:
        """Attempt to load the most recent checkpoint from output directories.

        Searches experiment output directories in reverse modification order,
        and loads the latest checkpoint found. Selectively restores components
        based on what was saved in the checkpoint.

        Returns:
            True if a checkpoint was successfully loaded, False otherwise.
        """
        info("Attempting to load from checkpoint")

        checkpoints = self.find_checkpoints()
        if checkpoints is None:
            info("No checkpoints found — starting fresh.")
            return False
        checkpoint = checkpoints[-1]
        try:
            info(f"Resuming from checkpoint: {checkpoint.name}")
            # map_location="cpu" keeps a checkpoint written on a CUDA host
            # loadable on a CPU-only one; Group._move_optimizer_to_device and
            # the Group's own .to(device) move the tensors back afterwards.
            checkpoint_data = torch.load(checkpoint, weights_only=False, map_location="cpu")
            # A checkpoint only carries the components the checkpointing config
            # included, so each restore is guarded — a partial checkpoint loads
            # cleanly instead of raising a KeyError.
            if "groups" in checkpoint_data:
                debug("Restoring groups...")
                self._load_group_states(checkpoint_data["groups"])
            if "data_sources" in checkpoint_data:
                debug("Restoring data sources...")
                self._load_datasource_states(checkpoint_data["data_sources"])
            if "globals" in checkpoint_data:
                debug("Restoring global RNG state...")
                self._load_global_state(checkpoint_data["globals"])
            if "tasks" in checkpoint_data:
                debug("Restoring task states...")
                self._load_task_states(checkpoint_data["tasks"])
            success("Checkpoint successfully loaded.")
            return True

        except Exception as e:
            error(f"Failed to load checkpoint '{checkpoint.name}': {e}")
            if self.config.experiment.allow_fresh_on_resume_failure:
                warn("allow_fresh_on_resume_failure=true — falling back to fresh training.")
                return False
            raise


__all__ = ["RuntimeState"]
