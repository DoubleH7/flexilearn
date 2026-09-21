# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Tests for per-task group overrides (plans/OVERRIDES_PLAN.md).

Unit level: Group.apply_override / restore_override round-trip. Engine level: a
2-task curriculum where task B freezes the shared group — its weights must not
move during B, and the baseline trainability must be restored afterwards.
"""

from __future__ import annotations

from conftest import make_config
from flexilearn import Engine
from flexilearn.config_models import GroupOverride, GroupSpec
from flexilearn.group import Group


def build_group(lr: float = 0.1) -> Group:
    spec = GroupSpec(
        models={"m": {"type": "drift_model"}},
        optimizer={"type": "sgd", "lr": lr},
        settings={"grad_accumulation": 1, "device": "cpu"},
    )
    return Group(spec)


# ---------------------------------------------------------------------------
# Unit: apply / restore round-trip
# ---------------------------------------------------------------------------


def test_apply_and_restore_override():
    group = build_group(lr=0.1)
    snap = group.override_snapshot()

    group.apply_override(GroupOverride(trainable=False, learning_rate=0.5))
    assert group.trainable is False
    assert all(not p.requires_grad for p in group.parameters())
    assert all(pg["lr"] == 0.5 for pg in group.optimizer.param_groups)

    group.restore_override(snap)
    assert group.trainable is True
    # Trainability is restored to what each model declared at construction.
    assert all(
        p.requires_grad == group._init_requires_grad[name] for name, p in group.named_parameters()
    )
    assert all(pg["lr"] == 0.1 for pg in group.optimizer.param_groups)


def test_partial_override_leaves_other_knob_alone():
    group = build_group(lr=0.1)
    group.apply_override(GroupOverride(learning_rate=0.01))  # trainable=None
    assert group.trainable is True
    assert all(pg["lr"] == 0.01 for pg in group.optimizer.param_groups)


# ---------------------------------------------------------------------------
# Engine: freeze the shared group for one curriculum task
# ---------------------------------------------------------------------------


def test_frozen_task_does_not_move_weights_and_baseline_is_restored():
    config = make_config(
        {
            "a": {
                "type": "instrumented",
                "iterations": 3,
                "epochs": 1,
                "checkpointing_interval": 0,
            },
            "b": {
                "type": "instrumented",
                "iterations": 3,
                "epochs": 1,
                "checkpointing_interval": 0,
                "overrides": {"default": {"trainable": False}},
            },
        }
    )
    engine = Engine(config)
    engine.execute()

    group = engine.runtime._groups["default"]
    task_a = engine.runtime._tasks["a"]
    task_b = engine.runtime._tasks["b"]

    # Task A actually trained: w moved away from its init (1.0).
    w_after_a = task_b.attempt_start_w[0]  # w at B's first iteration
    assert w_after_a != task_a.attempt_start_w[0]

    # Task B ran its loops, but the frozen group ignored update():
    assert task_b.loops_run == 3
    assert float(group.models["m"].w.detach()) == w_after_a

    # After B ended, the override was restored to the baseline.
    assert group.trainable is True
