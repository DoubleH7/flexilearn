# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Tests for the env-as-datasource machinery (flexilearn/sim.py).

A dependency-free fake env (duck-typed Gymnasium 5-tuple API) exercises the
factory hand-out policies, the DataSource loader guard, borrow_env lifetimes,
and the rollout driver's success / failure-taxonomy / chunk / render paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexilearn import (
    DataSource,
    EnvFactory,
    Episode,  # noqa: F401  (public API sanity)
    PolicyFailure,
    borrow_env,
    register_dataset,
    rollout,
)
from flexilearn.config_models import DataSourceSpec


class FakeEnv:
    """Reaches success after ``solve_at`` +1 steps; truncates at ``max_steps``."""

    def __init__(self, solve_at: int = 3, max_steps: int = 10, fail_wrong: bool = False):
        self.solve_at = solve_at
        self.max_steps = max_steps
        self.fail_wrong = fail_wrong
        self.pos = 0
        self.closed = False
        self.seen_seed = None

    def reset(self, *, seed=None, options=None):
        self.pos = 0
        self.seen_seed = seed
        return {"pos": self.pos}, {}

    def step(self, action):
        self.pos += int(action)
        terminated, truncated = False, False
        reward = 0.0
        info: dict = {}
        if self.fail_wrong and self.pos < 0:
            terminated, info = True, {"success": False, "failure_mode": "wrong_target"}
        elif self.pos >= self.solve_at:
            terminated, reward, info = True, 1.0, {"success": True, "failure_mode": None}
        elif self.pos <= -self.max_steps:
            truncated, info = True, {"success": False, "failure_mode": "timeout"}
        return {"pos": self.pos}, reward, terminated, truncated, info

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


@register_dataset("fake_env")
class FakeEnvFactory(EnvFactory):
    def __init__(self, persistent: bool = False, **env_kwargs):
        super().__init__(persistent=persistent)
        self.env_kwargs = env_kwargs

    def make(self):
        return FakeEnv(**self.env_kwargs)


def make_source(loader: dict, persistent: bool = False) -> DataSource:
    spec = DataSourceSpec(
        dataset={"type": "fake_env", "persistent": persistent},
        loader=loader,
    )
    source = DataSource(spec, seed=0)
    source.initialize()
    return source


# ---------------------------------------------------------------------------
# EnvFactory hand-out policies + DataSource integration
# ---------------------------------------------------------------------------


def test_fresh_factory_yields_new_envs():
    source = make_source({"batch_size": 0, "num_workers": 0})
    a, b = source.get_next(), source.get_next()
    assert isinstance(a, FakeEnv) and a is not b


def test_persistent_factory_yields_same_env():
    source = make_source({"batch_size": 0, "num_workers": 0}, persistent=True)
    a, b = source.get_next(), source.get_next()
    assert a is b
    source.dataset.close()
    assert a.closed


@pytest.mark.parametrize(
    "loader",
    [
        {"batch_size": 4, "num_workers": 0},
        {"batch_size": 0, "num_workers": 2},
        {"batch_size": 0, "num_workers": 0, "shuffle": True},
    ],
)
def test_env_source_rejects_batching_workers_shuffle(loader):
    with pytest.raises(ValueError, match="batch_size: 0"):
        make_source(loader)


def test_borrow_env_closes_fresh_but_not_persistent():
    source = make_source({"batch_size": 0, "num_workers": 0})
    with borrow_env(source) as env:
        pass
    assert env.closed

    source = make_source({"batch_size": 0, "num_workers": 0}, persistent=True)
    with borrow_env(source) as env:
        pass
    assert not env.closed


# ---------------------------------------------------------------------------
# rollout driver
# ---------------------------------------------------------------------------


def test_rollout_success_and_seed():
    env = FakeEnv(solve_at=3)
    ep = rollout(env, lambda obs: [1], seed=7)
    assert ep.success is True and ep.failure_mode is None
    assert ep.terminated and not ep.truncated
    assert ep.length == 3 and ep.reward_sum == 1.0
    assert env.seen_seed == 7


def test_rollout_executes_action_chunks():
    env = FakeEnv(solve_at=4)
    calls = []

    def chunked_policy(obs):
        calls.append(obs["pos"])
        return [1, 1]  # chunk of 2, executed open-loop

    ep = rollout(env, chunked_policy)
    assert ep.success is True and ep.length == 4
    assert calls == [0, 2]  # re-planned once per chunk, not per step


def test_rollout_policy_failure_enters_taxonomy():
    env = FakeEnv()

    def failing_policy(obs):
        raise PolicyFailure("decode_error")

    ep = rollout(env, failing_policy)
    assert ep.success is None and ep.failure_mode == "decode_error"
    assert ep.length == 0


def test_rollout_env_failure_mode_passthrough():
    env = FakeEnv(fail_wrong=True)
    ep = rollout(env, lambda obs: [-1])
    assert ep.success is False and ep.failure_mode == "wrong_target"


def test_rollout_max_steps_cap_marks_timeout():
    env = FakeEnv(solve_at=10_000)
    ep = rollout(env, lambda obs: [0], max_steps=5)
    assert ep.truncated and ep.failure_mode == "timeout" and ep.length == 5


def test_rollout_render_collects_frames():
    env = FakeEnv(solve_at=2)
    ep = rollout(env, lambda obs: [1], render=True)
    assert len(ep.frames) == ep.length + 1  # initial frame included
    assert ep.frames[0].shape == (4, 4, 3)


def test_rollout_env_without_success_convention():
    """Envs that never set info['success'] yield success=None (unknown)."""

    class PlainEnv(FakeEnv):
        def step(self, action):
            obs, r, term, trunc, _ = super().step(action)
            return obs, r, term, trunc, {}

    ep = rollout(PlainEnv(solve_at=1), lambda obs: [1])
    assert ep.success is None and ep.terminated


# ---------------------------------------------------------------------------
# write_mp4 (optional dependency)
# ---------------------------------------------------------------------------


def test_write_mp4_roundtrip(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    from flexilearn import write_mp4

    frames = [np.full((32, 32, 3), i * 10, dtype=np.uint8) for i in range(8)]
    out = write_mp4(frames, tmp_path / "clip.mp4", fps=4)
    assert out.exists() and out.stat().st_size > 0
