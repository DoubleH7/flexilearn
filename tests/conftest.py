# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Shared fixtures for the engine-level tests.

Registers tiny synthetic components (a model whose parameter actually moves
under SGD, plus instrumented tasks) so the failure-policy and override tests
can drive a real ``Engine`` end-to-end on CPU without any data sources,
tracking, or definitions/ discovery.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flexilearn import EvalResult, Task, register_model, register_task
from flexilearn.config_loader import parse_config  # noqa: F401  (import sanity)
from flexilearn.config_models import Config


@register_model("drift_model")
class DriftModel(nn.Module):
    """One trainable scalar with a non-zero gradient, so SGD visibly moves it."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs):
        return ((self.w - 3.0) ** 2).sum()


@register_task("instrumented")
class InstrumentedTask(Task):
    """Trains drift_model and records everything the tests need to assert on.

    Config kwargs:
      * ``crash_at``       — raise at this per-epoch iteration (default: never)
      * ``crash_once``     — only crash on the first attempt (for `retry`)
      * ``eval_pass_after``— evaluate() succeeds once global_step >= this value
                             (default 0 = always passes); None-like -1 = never
    """

    GROUP = "default"

    def __init__(self, task_spec, runtime) -> None:
        super().__init__(task_spec, runtime)
        extra = task_spec.model_extra or {}
        self.crash_at = extra.get("crash_at")
        self.crash_once = bool(extra.get("crash_once", False))
        self.eval_pass_after = int(extra.get("eval_pass_after", 0))

        self.events: list[str] = []  # interleaving of "loop" / "eval"
        self.attempt_start_w: list[float] = []  # w at iter 0 of each attempt
        self.attempt_start_rng: list[float] = []  # a torch draw at iter 0
        self.loops_run = 0
        self.eval_group_training_flags: list[bool] = []
        self._crashed_already = False

    def loop(self):
        group = self.get_group(self.GROUP)
        if self._iter_count == 0:
            self.attempt_start_w.append(float(group.models["m"].w.detach()))
            self.attempt_start_rng.append(float(torch.rand(())))
        if (
            self.crash_at is not None
            and self._iter_count == self.crash_at
            and not (self.crash_once and self._crashed_already)
        ):
            self._crashed_already = True
            raise RuntimeError("synthetic crash")
        loss = group.models["m"]()
        group.update(loss)
        self.events.append("loop")
        self.loops_run += 1

    def evaluate(self) -> EvalResult:
        group = self.get_group(self.GROUP)
        self.events.append("eval")
        self.eval_group_training_flags.append(group.training)
        if self.eval_pass_after < 0:
            return EvalResult(success=False, metrics={"step": self.global_step})
        return EvalResult(
            success=self.global_step >= self.eval_pass_after,
            metrics={"step": self.global_step},
        )


def make_config(curriculum: dict, lr: float = 0.1) -> Config:
    """Minimal CPU config: one drift_model group, no data sources, no tracking."""
    return Config(
        experiment={"name": "test", "seed": 0},
        curriculum=curriculum,
        groups={
            "default": {
                "models": {"m": {"type": "drift_model"}},
                "optimizer": {"type": "sgd", "lr": lr},
                "settings": {"grad_accumulation": 1, "device": "cpu"},
            }
        },
        data_sources={},
        logging={"enabled": False},
    )


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    """Engine.execute checkpoints into ./outputs — keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
