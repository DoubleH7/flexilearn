# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Simulations as data sources: the framework's env-factory machinery.

Flexilearn models environments in the *data layer*: a simulation enters the
config as an ordinary ``data_sources:`` block whose dataset subclasses
:class:`EnvFactory`, and a task pulls live environments through
``self.get_source(name).get_next()`` exactly like it pulls batches. The policy
lives in a ``Group`` the data source can't reach — the source only *provides*
the env. This keeps eval stratification pure config: one data-source block per
condition, every constructor kwarg a condition axis.

The framework is simulator-agnostic: an "environment" is anything duck-typing
the Gymnasium 5-tuple API (``reset(seed=...) -> (obs, info)``,
``step(a) -> (obs, reward, terminated, truncated, info)``, ``close()``, and
``render()`` when videos are wanted). Nothing here imports gymnasium — a
gym.Env satisfies the protocol, but so does any hand-rolled or vendor sim.

Conventions carried in ``info`` (all optional):
  * ``info["success"]``      — bool; the episode's true verdict (rewards alone
                               can't always express it).
  * ``info["failure_mode"]`` — short label for the failure taxonomy
                               (e.g. ``"wrong_target"``, ``"timeout"``).

Typical task-side usage::

    with borrow_env(self.get_source("env_nominal")) as env:
        ep = rollout(env, policy, seed=i, render=True)
    # ep.success / ep.failure_mode / ep.rewards / ep.frames

where ``policy(obs)`` returns a *sequence* of actions (an action chunk; single-
action policies return a length-1 list) and may raise
``PolicyFailure("reason")`` to abort the episode into the failure taxonomy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset

from .monitor import debug


class EnvFactory(IterableDataset):
    """Base class for environment data sources.

    Subclass and implement :meth:`make`; every kwarg your subclass accepts is a
    config-declarable condition axis. Declare the source with
    ``loader: {batch_size: 0, num_workers: 0}`` — ``DataSource`` enforces this
    at build time (envs are live objects: they can't be collated into batches
    or pickled across worker processes).

    ``persistent`` (config-settable on any subclass that forwards ``**kwargs``
    or declares it) controls the hand-out policy:
      * ``False`` (default) — a *fresh* env per ``get_next()`` pull; the
        consumer owns it and should close it (``borrow_env`` does).
      * ``True`` — one env is built on first pull and every pull returns the
        same live instance. For simulators with expensive construction
        (renderer/physics startup). Consumers must treat it as *borrowed* —
        reset it, don't close it (``borrow_env`` handles both cases).
    """

    def __init__(self, persistent: bool = False) -> None:
        self.persistent = persistent
        self._env = None

    def make(self):
        """Build and return one environment instance."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the cached persistent env, if one was built."""
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def __iter__(self) -> Iterator:
        while True:
            if self.persistent:
                if self._env is None:
                    self._env = self.make()
                    debug(f"{type(self).__name__}: built persistent env")
                yield self._env
            else:
                yield self.make()


@contextmanager
def borrow_env(source):
    """Pull one env from a source and manage its lifetime by factory policy.

    Fresh-per-pull envs are closed on exit; a persistent factory's env is left
    alive (the factory owns it). Task code stays identical either way.
    """
    env = source.get_next()
    try:
        yield env
    finally:
        if not getattr(source.dataset, "persistent", False):
            env.close()


class PolicyFailure(Exception):
    """Raised by a policy inside :func:`rollout` to abort the episode.

    ``reason`` becomes the episode's ``failure_mode`` — the policy's own
    failure taxonomy (e.g. ``"no_action_tokens"``, ``"decode_error"``) lands
    next to the env's (``"wrong_target"``, ``"timeout"``).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Episode:
    """Outcome of one :func:`rollout`.

    ``success`` is taken from ``info["success"]`` when the env provides it,
    else ``None`` (unknown — derive a verdict from ``rewards`` in the task).
    """

    rewards: list[float] = field(default_factory=list)
    length: int = 0
    success: bool | None = None
    failure_mode: str | None = None
    terminated: bool = False
    truncated: bool = False
    frames: list[Any] | None = None
    final_info: dict = field(default_factory=dict)

    @property
    def reward_sum(self) -> float:
        return float(sum(self.rewards))

    @property
    def max_reward(self) -> float:
        return float(max(self.rewards)) if self.rewards else 0.0


def rollout(
    env,
    policy: Callable[[Any], Sequence],
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    render: bool = False,
    reset_options: dict | None = None,
) -> Episode:
    """Drive one episode: reset, then ``policy(obs) -> action chunk`` until done.

    ``policy`` returns a **sequence** of actions executed open-loop before the
    next call (chunked policies return their chunk; single-action policies a
    length-1 list), or raises :class:`PolicyFailure` to abort. ``max_steps`` is
    a safety cap on env steps for envs without their own truncation — hitting
    it marks the episode ``truncated`` with ``failure_mode="timeout"`` (unless
    the env already set one). ``render=True`` collects ``env.render()`` frames
    (initial frame included) into ``Episode.frames``.

    Deterministic evaluation: pass a fixed ``seed`` per episode index so
    conditions stay comparable across checkpoints and runs.
    """
    ep = Episode()
    obs, _ = (
        env.reset(seed=seed, options=reset_options)
        if reset_options is not None
        else env.reset(seed=seed)
    )
    if render:
        ep.frames = [env.render()]

    done = False
    while not done:
        try:
            actions = policy(obs)
        except PolicyFailure as failure:
            ep.failure_mode = failure.reason
            break
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            ep.rewards.append(float(reward))
            ep.length += 1
            if render:
                ep.frames.append(env.render())
            if max_steps is not None and ep.length >= max_steps and not (terminated or truncated):
                truncated = True
                info = dict(info)
                info.setdefault("failure_mode", "timeout")
            if terminated or truncated:
                ep.terminated, ep.truncated = terminated, truncated
                ep.final_info = dict(info)
                success = info.get("success")
                ep.success = bool(success) if success is not None else None
                ep.failure_mode = info.get("failure_mode")
                done = True
                break
    return ep


def write_mp4(frames: Sequence, path, fps: int = 8) -> Path:
    """Encode frames (HxWx3 uint8 arrays) to a browser-friendly mp4.

    yuv420p + "-movflags +faststart" put the moov atom up front — without it
    many players (including the MLflow UI's in-browser preview) report the
    file as corrupt. Requires the optional ``video`` extra; raises ImportError
    with the fix if absent.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "write_mp4 requires imageio + imageio-ffmpeg. Install the optional "
            "extra: `uv sync --extra video` (or `pip install flexilearn[video]`)."
        ) from e
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        macro_block_size=16,
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame, dtype=np.uint8))
    return path


__all__ = [
    "EnvFactory",
    "borrow_env",
    "PolicyFailure",
    "Episode",
    "rollout",
    "write_mp4",
]
