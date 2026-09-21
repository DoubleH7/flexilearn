# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Regression tests for the config-schema fixes (bugs.md items 1-3) and the
failure-policy / override validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import make_config
from flexilearn.config_models import ExperimentConfig, GroupSettings, TaskSpec

# --- bugs.md F1: grad_accumulation — 1 disables, 0 is invalid -----------------


def test_grad_accumulation_one_is_valid_zero_is_not():
    assert GroupSettings(grad_accumulation=1).grad_accumulation == 1
    with pytest.raises(ValidationError):
        GroupSettings(grad_accumulation=0)


# --- bugs.md F2: checkpointing_interval 0 disables checkpointing --------------


def test_checkpointing_interval_zero_is_valid():
    spec = TaskSpec(type="instrumented", iterations=1, checkpointing_interval=0)
    assert spec.checkpointing_interval == 0
    with pytest.raises(ValidationError):
        TaskSpec(type="instrumented", iterations=1, checkpointing_interval=-1)


# --- bugs.md F3: omitting seed must not crash a fresh run ---------------------


def test_seed_has_deterministic_default():
    assert ExperimentConfig(name="x").seed == 0


# --- failure-policy validators -------------------------------------------------


def test_until_requires_evaluation_interval():
    with pytest.raises(ValidationError, match="until"):
        TaskSpec(
            type="instrumented",
            iterations=1,
            checkpointing_interval=0,
            upon_failure="until",
        )


def test_epochs_allows_minus_one_rejects_zero():
    spec = TaskSpec(
        type="instrumented",
        iterations=1,
        checkpointing_interval=0,
        epochs=-1,
    )
    assert spec.epochs == -1
    with pytest.raises(ValidationError):
        TaskSpec(
            type="instrumented",
            iterations=1,
            checkpointing_interval=0,
            epochs=0,
        )


# --- override validator ---------------------------------------------------------


def test_override_naming_unknown_group_is_rejected():
    with pytest.raises(ValidationError, match="unknown group"):
        make_config(
            {
                "t": {
                    "type": "instrumented",
                    "iterations": 1,
                    "checkpointing_interval": 0,
                    "overrides": {"ghost": {"trainable": False}},
                }
            }
        )
