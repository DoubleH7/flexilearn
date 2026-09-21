# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Tests for Group construction, notably the optional ``optimizer:`` block.

A group declared without an optimizer is an inference-only unit (a reference
policy, a reward model, a frozen teacher). Before this was supported, omitting
the block raised ``AttributeError: 'NoneType'`` — while Group's own
zero-trainable-params error told you to do exactly that.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flexilearn import register_model
from flexilearn.config_models import GroupOverride, GroupSpec
from flexilearn.group import Group


@register_model("all_frozen_model")
class AllFrozenModel(nn.Module):
    """Every parameter frozen at construction — a backbone with no trainable head."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.w.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return ((self.w - 3.0) ** 2).sum()


def build_optimizerless_group(is_trainable: bool = False) -> Group:
    spec = GroupSpec(
        models={"m": {"type": "all_frozen_model"}},
        settings={
            "grad_accumulation": 1,
            "device": "cpu",
            "is_trainable": is_trainable,
        },
    )
    return Group(spec)


# ---------------------------------------------------------------------------
# The group builds at all
# ---------------------------------------------------------------------------


def test_group_without_optimizer_builds():
    group = build_optimizerless_group()
    assert group.optimizer is None
    assert group.trainable is False
    assert "m" in group.models


def test_frozen_group_error_message_advice_actually_works():
    """Group's zero-trainable error says to drop `optimizer:` — that must work.

    The advice used to crash, which is what made this bug worth fixing: the
    framework told you to do something it could not do.
    """
    with_optimizer = GroupSpec(
        models={"m": {"type": "all_frozen_model"}},
        optimizer={"type": "sgd", "lr": 0.1},
        settings={"grad_accumulation": 1, "device": "cpu"},
    )
    with pytest.raises(ValueError, match="zero trainable parameters"):
        Group(with_optimizer)

    # Now follow the advice the error just gave.
    group = build_optimizerless_group()
    assert group.optimizer is None


# ---------------------------------------------------------------------------
# It behaves as a no-op rather than crashing
# ---------------------------------------------------------------------------


def test_update_is_a_noop_and_weights_do_not_move():
    group = build_optimizerless_group()
    before = float(group.models["m"].w.detach())

    group.update(group.models["m"]())

    assert float(group.models["m"].w.detach()) == before
    assert group.optimizer_steps == 0
    assert group.window_count == 0


def test_is_trainable_true_is_forced_false():
    group = build_optimizerless_group(is_trainable=True)
    assert group.trainable is False
    # And it cannot be talked back into trainability later.
    group.set_trainable(True)
    assert group.trainable is False


def test_learning_rate_and_override_roundtrip_do_not_crash():
    group = build_optimizerless_group()
    snap = group.override_snapshot()
    assert snap["lrs"] == []

    group.set_learning_rate(0.5)  # no optimizer to set it on; must not raise
    group.apply_override(GroupOverride(trainable=False, learning_rate=0.5))
    group.restore_override(snap)

    assert group.trainable is False


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_without_optimizer():
    from flexilearn.config_models import GroupCheckpointPolicy

    group = build_optimizerless_group()
    state = group.checkpoint_state(GroupCheckpointPolicy(weights="full", optimizer=True))

    # Nothing to serialize, so the key is absent rather than None.
    assert "optimizer" not in state

    spec = GroupSpec(
        models={"m": {"type": "all_frozen_model"}},
        settings={"grad_accumulation": 1, "device": "cpu"},
    )
    restored = Group.from_checkpoint(state, spec)
    assert restored.optimizer is None
    assert float(restored.models["m"].w.detach()) == float(group.models["m"].w.detach())
