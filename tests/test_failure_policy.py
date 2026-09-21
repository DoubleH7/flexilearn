# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Engine-level tests for the failure policies (plans/FAILURE_POLICY_PLAN.md).

Each test drives a real Engine on CPU with the instrumented synthetic task from
conftest.py: crash failures at a chosen iteration, statistical failures via an
evaluate() that passes only after N global steps, and the four policies
(exit / continue / retry / until).
"""

from __future__ import annotations

import pytest

from conftest import make_config
from flexilearn import Engine


def run_engine(task_spec: dict):
    engine = Engine(make_config({"t": task_spec}))
    engine.execute()
    return engine.runtime._tasks["t"]


# ---------------------------------------------------------------------------
# Crash failures
# ---------------------------------------------------------------------------


def test_exit_propagates_crash():
    engine = Engine(
        make_config(
            {
                "t": {
                    "type": "instrumented",
                    "iterations": 5,
                    "epochs": 1,
                    "checkpointing_interval": 0,
                    "upon_failure": "exit",
                    "crash_at": 2,
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        engine.execute()


def test_continue_skips_failing_iteration():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 5,
            "epochs": 1,
            "checkpointing_interval": 0,
            "upon_failure": "continue",
            "crash_at": 2,
        }
    )
    # Iteration 2 crashed and was skipped; the other 4 ran; the epoch completed.
    assert task.loops_run == 4
    assert task._epoch_count == 1


def test_retry_restores_params_but_diverges():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 5,
            "epochs": 1,
            "checkpointing_interval": 0,
            "upon_failure": "retry",
            "crash_at": 3,
            "crash_once": True,
        }
    )
    # Two attempts: the first crashed at iteration 3, the second completed.
    assert len(task.attempt_start_w) == 2
    # Parameters were restored to the pre-task snapshot for the restart...
    assert task.attempt_start_w[0] == task.attempt_start_w[1]
    # ...but RNG was deliberately NOT rewound, so the attempt diverges.
    assert task.attempt_start_rng[0] != task.attempt_start_rng[1]
    # The successful attempt ran the full epoch.
    assert task._epoch_count == 1
    assert task.loops_run == 3 + 5  # 3 before the crash, 5 in the clean attempt


def test_retry_exhausts_max_retries():
    engine = Engine(
        make_config(
            {
                "t": {
                    "type": "instrumented",
                    "iterations": 5,
                    "epochs": 1,
                    "checkpointing_interval": 0,
                    "upon_failure": "retry",
                    "crash_at": 0,
                    "max_retries": 2,
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match="max_retries"):
        engine.execute()


# ---------------------------------------------------------------------------
# Statistical failures (evaluate() verdicts)
# ---------------------------------------------------------------------------


def test_until_terminates_on_first_passing_eval():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 100,
            "epochs": -1,
            "checkpointing_interval": 0,
            "upon_failure": "until",
            "evaluation_interval": 1,
            "eval_pass_after": 7,
        }
    )
    # Passed as soon as global_step reached 7 — well before the (infinite)
    # epoch bound; the passing eval is what completed the task.
    assert task.loops_run == 7


def test_until_positive_epochs_is_a_hard_cap():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 5,
            "epochs": 1,
            "checkpointing_interval": 0,
            "upon_failure": "until",
            "evaluation_interval": 1,
            "eval_pass_after": -1,  # never passes
        }
    )
    # Eval never passes, but the positive epoch bound still ends the task.
    assert task.loops_run == 5


def test_exit_on_statistical_failure():
    engine = Engine(
        make_config(
            {
                "t": {
                    "type": "instrumented",
                    "iterations": 5,
                    "epochs": 1,
                    "checkpointing_interval": 0,
                    "upon_failure": "exit",
                    "evaluation_interval": 1,
                    "eval_pass_after": -1,
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match="failed evaluation"):
        engine.execute()


def test_continue_ignores_statistical_failure():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 5,
            "epochs": 1,
            "checkpointing_interval": 0,
            "upon_failure": "continue",
            "evaluation_interval": 1,
            "eval_pass_after": -1,
        }
    )
    assert task.loops_run == 5
    assert task.events.count("eval") == 5


# ---------------------------------------------------------------------------
# evaluate_at_start (pre-training baseline)
# ---------------------------------------------------------------------------


def test_evaluate_at_start_runs_before_training():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 4,
            "epochs": 1,
            "checkpointing_interval": 0,
            "evaluation_interval": 100,
            "evaluate_at_start": True,
        }
    )
    assert task.events[0] == "eval"
    # Regression: the group is first *built* inside that baseline eval, after
    # the eval-mode sweep already ran — get_group must still put it in eval.
    assert task.eval_group_training_flags[0] is False
    # Training proceeded normally afterwards, back in train mode.
    assert task.loops_run == 4


def test_until_passing_baseline_completes_without_training():
    task = run_engine(
        {
            "type": "instrumented",
            "iterations": 5,
            "epochs": -1,
            "checkpointing_interval": 0,
            "upon_failure": "until",
            "evaluation_interval": 1,
            "evaluate_at_start": True,
            "eval_pass_after": 0,  # passes immediately
        }
    )
    assert task.loops_run == 0
    assert task.events == ["eval"]
