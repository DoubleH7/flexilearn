# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Tests for checkpoint portability and the group-less checkpoint counter.

Two regressions live here:

* Checkpoints were written with tensors still on their training device and
  loaded without ``map_location``, so a CUDA-trained run could not be resumed
  or even inspected on a CPU-only host.
* ``Task.checkpoint_flag`` reduced an empty list with ``max()`` when a task
  registered no groups and ``cumulative_checkpointing_counter`` was false,
  raising ``ValueError`` instead of simply not checkpointing.
"""

from __future__ import annotations

import torch

from conftest import make_config
from flexilearn import Engine, Task, register_task
from flexilearn.config_models import GroupCheckpointPolicy, GroupSpec
from flexilearn.group import Group


def build_group(device: str = "cpu") -> Group:
    spec = GroupSpec(
        models={"m": {"type": "drift_model"}},
        optimizer={"type": "sgd", "lr": 0.1},
        settings={"grad_accumulation": 1, "device": device},
    )
    return Group(spec)


# ---------------------------------------------------------------------------
# Device portability
# ---------------------------------------------------------------------------


def _assert_all_cpu(state: dict) -> None:
    for name, tensor in state["model"].items():
        assert tensor.device.type == "cpu", f"{name} left on {tensor.device}"


def test_full_checkpoint_weights_are_cpu_tensors():
    group = build_group()
    state = group.checkpoint_state(GroupCheckpointPolicy(weights="full"))
    _assert_all_cpu(state)


def test_trainable_checkpoint_weights_are_cpu_tensors():
    group = build_group()
    state = group.checkpoint_state(GroupCheckpointPolicy(weights="trainable"))
    assert state["model"], "expected at least one trainable parameter"
    _assert_all_cpu(state)


def test_checkpoint_survives_a_save_load_roundtrip_through_disk(tmp_path):
    """torch.save/torch.load with map_location is what RuntimeState does."""
    group = build_group()
    group.update(group.models["m"]())
    state = group.checkpoint_state(GroupCheckpointPolicy(weights="full"))

    path = tmp_path / "group.fl"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=False, map_location="cpu")

    spec = GroupSpec(
        models={"m": {"type": "drift_model"}},
        optimizer={"type": "sgd", "lr": 0.1},
        settings={"grad_accumulation": 1, "device": "cpu"},
    )
    restored = Group.from_checkpoint(loaded, spec)
    assert restored.optimizer_steps == group.optimizer_steps
    assert float(restored.models["m"].w.detach()) == float(group.models["m"].w.detach())


# ---------------------------------------------------------------------------
# Group-less tasks
# ---------------------------------------------------------------------------


@register_task("groupless")
class GrouplessTask(Task):
    """A collection-style task: it never calls get_group(), so registers none."""

    def loop(self):
        self._noop = torch.rand(())


def test_groupless_task_with_max_counter_does_not_crash():
    """`cumulative_checkpointing_counter: false` selects max(), and max([]) raises."""
    config = make_config(
        {
            "collect": {
                "type": "groupless",
                "iterations": 3,
                "epochs": 1,
                "checkpointing_interval": 1,
                "cumulative_checkpointing_counter": False,
            },
        }
    )
    engine = Engine(config)
    engine.execute()

    task = engine.runtime._tasks["collect"]
    assert task._group_names == set()
    assert task.checkpoint_flag is False


def test_groupless_task_with_sum_counter_also_reports_false():
    config = make_config(
        {
            "collect": {
                "type": "groupless",
                "iterations": 2,
                "epochs": 1,
                "checkpointing_interval": 1,
                "cumulative_checkpointing_counter": True,
            },
        }
    )
    engine = Engine(config)
    engine.execute()
    assert engine.runtime._tasks["collect"].checkpoint_flag is False
