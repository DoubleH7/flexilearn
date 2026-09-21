# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .config_models import TaskSpec
from .data import DataSource
from .group import Group
from .monitor import debug

if TYPE_CHECKING:
    from .runtime_state import RuntimeState

TASK_STATE_DICT_VERSION = 1


@dataclass
class EvalResult:
    """Outcome of a Task.evaluate() call.

    ``success=False`` signals a *statistical* failure (the run is technically
    alive but under-performing / diverging), which the engine routes through the
    task's ``upon_failure`` policy — distinct from a crash (an uncaught
    Exception). ``metrics`` carries the numbers the verdict was based on, for
    logging.
    """

    success: bool
    metrics: dict = field(default_factory=dict)


class Task(ABC):
    def __init__(self, task_spec: TaskSpec, runtime: "RuntimeState") -> None:
        # spec variables
        self.iterations = task_spec.iterations
        self.epochs = task_spec.epochs
        self.checkpointing_interval = task_spec.checkpointing_interval
        self.upon_failure = task_spec.upon_failure
        self.evaluation_interval = task_spec.evaluation_interval
        self.evaluate_at_start = task_spec.evaluate_at_start
        self.max_retries = task_spec.max_retries
        self.mode = task_spec.mode
        self.overrides = task_spec.overrides
        self.verbose = task_spec.verbose
        if task_spec.cumulative_checkpointing_counter:
            self._count_filter_func = sum
        else:
            self._count_filter_func = max

        self._runtime = runtime

        # state variables
        self._group_names: set[str] = set()
        self._iter_count = 0
        self._epoch_count = 0
        # True while inside an eval pass (_step with mode=eval, or
        # _evaluate_step). Lets get_group() put a group registered *mid-eval*
        # straight into eval mode — _set_groups_mode("eval") ran before the
        # group was known (e.g. the very first evaluate_at_start baseline).
        self._in_eval = False
        debug(
            f"Task '{self.__class__.__name__}' initialized: {self.iterations} iterations × {self.epochs} epochs"
        )

    # ------------------
    # State Dict Handling
    # ------------------

    @property
    def task_state_dict(self):
        return {
            "format_version": TASK_STATE_DICT_VERSION,
            "_iter_count": self._iter_count,
            "_epoch_count": self._epoch_count,
            "_group_names": self._group_names,
        }

    def _load_task_state(self, state_dict: dict):
        if state_dict["format_version"] != TASK_STATE_DICT_VERSION:
            raise RuntimeError(
                f"Task Checkpoint version mismatch. "
                f"Expected {TASK_STATE_DICT_VERSION}, "
                f"got {state_dict['format_version']}."
            )
        self._iter_count = state_dict["_iter_count"]
        self._epoch_count = state_dict["_epoch_count"]
        self._group_names = state_dict["_group_names"]

    @classmethod
    def from_checkpoint(cls, state: dict, spec: TaskSpec, runtime: "RuntimeState") -> "Task":
        """Create task from checkpoint state dict."""
        ctor = spec.ctor
        task = ctor(spec, runtime)
        task._load_task_state(state)
        debug(
            f"Task '{task.__class__.__name__}' restored: epoch {task._epoch_count}, iteration {task._iter_count}"
        )
        return task

    @property
    def checkpoint_flag(self) -> bool:
        if not self.checkpointing_interval:
            return False
        # A task that registers no groups (a pure collection or eval task) has
        # nothing to count, so it never triggers a checkpoint on its own. The
        # empty list is checked explicitly because `max([])` raises.
        if not self._group_names:
            return False
        counts = [self._runtime.get_group(g_name).window_count for g_name in self._group_names]
        return self._count_filter_func(counts) >= self.checkpointing_interval

    # ------------------
    # Under the hood functionality
    # ------------------

    def _increment_epoch(self):
        """Called at the end of an epoch"""
        self._iter_count = 0
        self._epoch_count += 1

    def _set_groups_mode(self, mode: str) -> None:
        for name in self._group_names:
            self._runtime.get_group(name).set_mode(mode)

    def _step(self):
        if self.mode == "eval":
            self._in_eval = True
            self._set_groups_mode("eval")
            try:
                with torch.no_grad():
                    self.loop()
            finally:
                self._in_eval = False
                self._set_groups_mode("train")
        else:
            self.loop()
        self._iter_count += 1

    def _evaluate_step(self) -> "EvalResult":
        """Run evaluate() under eval mode + no_grad, restoring train mode after.

        Mirrors the eval handling in _step(); used by the engine for periodic
        statistical-failure checks (gated by ``evaluation_interval``).
        """
        self._in_eval = True
        self._set_groups_mode("eval")
        try:
            with torch.no_grad():
                return self.evaluate()
        finally:
            self._in_eval = False
            self._set_groups_mode("train")

    # ------------------
    # User Methods
    # ------------------

    def get_group(self, g_name: str) -> Group:
        group = self._runtime.get_group(g_name)
        if g_name not in self._group_names:
            self._group_names.add(g_name)
            if self._in_eval:
                # Registered mid-eval: the eval-mode sweep already ran without
                # this group, so switch it here (train mode is restored for all
                # registered groups when the eval pass ends).
                group.set_mode("eval")
            debug(f"Task '{self.__class__.__name__}' registered group '{g_name}'")
        return group

    def get_source(self, s_name: str) -> DataSource:
        return self._runtime.get_source(s_name)

    @property
    def global_step(self) -> int:
        """Monotonic step for metric logging across this task's run."""
        return self._epoch_count * self.iterations + self._iter_count

    def log_scalar(self, name: str, value: float) -> None:
        """Log a scalar metric, gated by ``logging.scalar_interval``.

        Routes to the experiment tracker; a no-op when tracking is disabled.
        """
        self._runtime.tracker.maybe_log_scalar(name, value, self.global_step)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric immediately at an explicit step (no interval gate)."""
        self._runtime.tracker.log_scalar(name, value, step)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        """Log a file to the experiment tracker (e.g. an eval rollout video)."""
        self._runtime.tracker.log_artifact(path, artifact_path=artifact_path)

    @abstractmethod
    def loop(self):
        """
        Defines one logical training iteration.
        Responsible for:
        - calling group.run(data)
        - computing losses
        - calling group.update(loss)
        """
        pass

    def evaluate(self) -> EvalResult:
        """Periodic evaluation hook for statistical-failure detection.

        Override to run a validation pass and return ``EvalResult(success=...)``
        based on a metric threshold (read thresholds from config kwargs via
        ``task_spec.model_extra``). The default never flags a failure, so tasks
        that don't evaluate keep working unchanged. Runs under eval mode +
        ``torch.no_grad()`` (see ``_evaluate_step``).
        """
        return EvalResult(success=True)
