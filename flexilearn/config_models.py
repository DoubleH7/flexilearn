# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from .registry import (
    DATA_PREPARATION_REGISTRY,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    TASK_REGISTRY,
)

DEFAULT_GROUP_NAME = "default"
REGISTRY = dict[str, type]


class TypedConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


# ============================================================================
# Per-task Group Overrides
# ============================================================================
class GroupOverride(BaseModel):
    """A light, reversible modification a task applies to a shared group.

    Applied when the task starts and restored when it ends (see
    ``Engine._run_task``), so one group can be reused across curriculum tasks
    with per-task tweaks. ``None`` means "leave as the group declared it".
    """

    trainable: bool | None = None  # freeze/unfreeze the whole group
    learning_rate: float | None = None  # override every optimizer param group's lr


# ============================================================================
# Experiment Config
# ============================================================================
class ExperimentConfig(BaseModel):
    name: str
    attempt_resume: bool | None = False
    allow_fresh_on_resume_failure: bool | None = False
    # Seed is de-facto required (RuntimeState.initialize feeds it straight into
    # torch/numpy/random seeding, where None raises), so give it a deterministic
    # default rather than crashing when omitted. A *random* default would also
    # change the config hash (and thus the checkpoint directory) on every run.
    seed: int = 0


# ============================================================================
# Curriculum Config
# ============================================================================
class TaskSpec(TypedConfig):
    iterations: PositiveInt
    # How often (in group update windows) to checkpoint. 0 disables
    # checkpointing for this task (Task.checkpoint_flag treats 0 as "never").
    checkpointing_interval: NonNegativeInt
    cumulative_checkpointing_counter: bool | None = True
    # Failure handling. `exit` re-raises, `continue` skips the failing iteration,
    # `retry` restores the pre-task parameter snapshot and restarts (without
    # rewinding RNG/data, so the attempt diverges), and `until` keeps training
    # through statistically-failing evaluations until one passes (then advances).
    upon_failure: Literal["exit", "retry", "continue", "until"] | None = "exit"
    # How often to run Task.evaluate() for statistical-failure detection,
    # measured in the task's `global_step` (cumulative across epochs and, for
    # tasks that override global_step to a group counter, across resumes — NOT
    # the per-epoch iteration counter). 0 disables evaluation entirely.
    evaluation_interval: NonNegativeInt = 0
    # Also evaluate once at task entry, before any training step — an untrained
    # /pre-task baseline (e.g. a step-0 rollout). Only meaningful with
    # evaluation_interval > 0; under `until`, a passing baseline completes the
    # task immediately.
    evaluate_at_start: bool | None = False
    # Cap on `retry` task-restarts before giving up.
    max_retries: PositiveInt = 5
    # Number of epochs. -1 means run indefinitely (intended for `upon_failure:
    # until`, where the passing eval is the real termination condition); any
    # positive value is a hard upper bound.
    epochs: int | None = 1
    mode: Literal["train", "eval"] | None = "train"
    # Per-task group modifications, keyed by group name (validated against the
    # declared groups in Config.validate_override_groups).
    overrides: dict[str, GroupOverride] | None = None
    verbose: PositiveInt | None = 10

    @property
    def ctor(self):
        return TASK_REGISTRY[self.type]

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v):
        if v is not None and v != -1 and v < 1:
            raise ValueError(
                f"epochs must be -1 (run indefinitely) or a positive integer, got {v}."
            )
        return v

    @model_validator(mode="after")
    def validate_until_needs_eval(self):
        """`until` terminates only when an evaluation passes, so it requires one."""
        if self.upon_failure == "until" and self.evaluation_interval <= 0:
            raise ValueError(
                "upon_failure='until' requires evaluation_interval > 0 — otherwise "
                "the task can never terminate (no eval can ever pass)."
            )
        return self


# ============================================================================
# Model Config
# ============================================================================
class ModelSpec(TypedConfig):
    @property
    def instanciation_args(
        self,
    ):
        return MODEL_REGISTRY[self.type], self.model_extra or {}


# ============================================================================
# Optimizer Config
# ============================================================================
class OptimizerSpec(TypedConfig):
    @property
    def instanciation_args(
        self,
    ):
        return OPTIMIZER_REGISTRY[self.type], self.model_extra or {}


class GroupSettings(BaseModel):
    # Optimizer steps are applied every `grad_accumulation` update() calls, and
    # the loss is scaled by 1/grad_accumulation. 1 = no accumulation (every
    # update() steps immediately); 0 is invalid — it would divide the loss by
    # zero in Group.update.
    grad_accumulation: PositiveInt | None = 1
    is_trainable: bool | None = True
    device: Literal["cpu", "cuda", "mps", "tpu"] = "cuda"


# ============================================================================
# Group Config
# ============================================================================
class GroupSpec(BaseModel):
    models: dict[str, ModelSpec]
    optimizer: OptimizerSpec | None = None
    settings: GroupSettings | None = GroupSettings()

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        """Ensure at least one model is defined"""
        if not v:
            raise ValueError(f"at least one model must be defined. got {v}")
        return v


# ============================================================================
# Data Config
# ============================================================================
class PreparationSpec(TypedConfig):
    @property
    def instanciation_args(
        self,
    ):
        return DATA_PREPARATION_REGISTRY[self.type], self.model_extra or {}


class DatasetConfig(TypedConfig):
    @property
    def instanciation_args(self):
        return DATASET_REGISTRY[self.type], self.model_extra or {}


class DataLoaderConfig(BaseModel):
    batch_size: int | None = None
    num_workers: NonNegativeInt | None = 1  # 0 = load in the main process
    prefetch_factor: int | None = None
    shuffle: bool | None = False  # seeded per-epoch in DataSource.initialize()
    drop_last: bool | None = False  # drop the final partial batch


class DataSourceSpec(BaseModel):
    loader: DataLoaderConfig | None = DataLoaderConfig()
    dataset: DatasetConfig
    preparations: list[PreparationSpec] = Field(default_factory=list)


# ============================================================================
# Monitoring Config
# ============================================================================
class LoggingConfig(BaseModel):
    """Experiment-tracking config, consumed by ``flexilearn.tracking.Tracker``.

    The three ``*_interval`` fields gate how often metrics are logged, in global
    steps. ``0`` disables that channel (the default for histograms, which are
    expensive). When ``enabled`` is false the Tracker is a complete no-op and
    MLflow is never imported.
    """

    enabled: bool = True
    tracking_uri: str | None = "./mlruns"  # local file store by default
    experiment: str | None = None  # falls back to experiment.name
    scalar_interval: NonNegativeInt = 10  # loss / lr scalars
    gradient_interval: NonNegativeInt = 50  # per-group global grad norm
    histogram_interval: NonNegativeInt = 0  # grad histograms (0 = off)

    # --- Richer experiment tracking (all best-effort; never abort training) ---
    system_metrics: bool = False  # GPU/CPU/RAM via psutil/pynvml
    log_learning_rate: bool = True  # per-group LR at scalar_interval
    log_throughput: bool = True  # steps/sec + sec/step at scalar_interval
    log_dataset_lineage: bool = True  # log each data source as an MLflow input
    log_git_info: bool = True  # commit/branch/dirty tags + lockfile

    # --- Model registry (packaging trained models into versioned registry) ---
    register_models: bool = False  # master switch for registration
    registered_model_name: str | None = None  # falls back to experiment.name
    registry_uri: str | None = None  # falls back to tracking_uri


# ============================================================================
# Checkpointing Config
# ============================================================================
class GroupCheckpointPolicy(BaseModel):
    """How much of a single group's state to persist in a checkpoint.

    ``weights`` scopes the serialized model weights:
      * ``full``      — every parameter and buffer (the only safe option for a
                        from-scratch model whose weights are the sole copy).
      * ``trainable`` — only ``requires_grad=True`` parameters. For a PEFT /
                        frozen-backbone model this is the adapter alone; the
                        base is reproduced by the model constructor on resume.
      * ``none``      — no weights; for a group rebuilt entirely at construction
                        time (e.g. a frozen pretrained reference model).
    """

    weights: Literal["full", "trainable", "none"] = "full"
    optimizer: bool | None = True


class GroupsCheckpointConfig(GroupCheckpointPolicy):
    """Default group policy plus optional per-group overrides (by group name)."""

    overrides: dict[str, GroupCheckpointPolicy] = Field(default_factory=dict)


class CheckpointingConfig(BaseModel):
    """Inclusion dictionary — declares what a checkpoint must contain.

    ``groups`` is always written (group counters are required for resume); its
    ``weights`` policy scopes the heavy part. The remaining components are tiny;
    their toggles exist for control / reproducibility choices, not size.

    Omitting the ``checkpointing:`` block entirely yields this default: a full,
    everything-included checkpoint — identical to the framework's prior behavior.
    """

    globals: bool | None = True  # RNG state
    tasks: bool | None = True  # task counters
    data_sources: bool | None = True  # datasource counters / kwargs
    export: bool | None = True  # run each model's export() hook
    groups: GroupsCheckpointConfig = Field(default_factory=GroupsCheckpointConfig)


# ============================================================================
# Main Config Model
# ============================================================================
class Config(BaseModel):
    """Main configuration model that holds all configuration sections."""

    experiment: ExperimentConfig
    curriculum: dict[str, TaskSpec]
    groups: dict[str, GroupSpec]
    data_sources: dict[str, DataSourceSpec]
    logging: LoggingConfig
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)

    @field_validator("groups")
    @classmethod
    def validate_default_group_exists(cls, v: dict):
        """
        Ensure the required default group is declared.
        """
        if DEFAULT_GROUP_NAME not in v:
            raise ValueError(f'config must define the default group"{DEFAULT_GROUP_NAME}"')
        return v

    @model_validator(mode="after")
    def validate_checkpoint_overrides(self):
        """Every checkpointing override must name a group that actually exists."""
        unknown = set(self.checkpointing.groups.overrides) - set(self.groups)
        if unknown:
            raise ValueError(
                f"checkpointing.groups.overrides names unknown group(s): "
                f"{sorted(unknown)}. Defined groups: {sorted(self.groups)}."
            )
        return self

    @model_validator(mode="after")
    def validate_override_groups(self):
        """Every per-task override must name a group that actually exists."""
        defined = set(self.groups)
        for t_name, task in self.curriculum.items():
            unknown = set(task.overrides or {}) - defined
            if unknown:
                raise ValueError(
                    f"task '{t_name}' overrides names unknown group(s): "
                    f"{sorted(unknown)}. Defined groups: {sorted(defined)}."
                )
        return self
