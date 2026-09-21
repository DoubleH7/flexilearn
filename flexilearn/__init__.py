# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Flexilearn — a configuration-driven PyTorch training framework.

Components are registered with decorators and wired together from a single
``config.yml``; swapping a model or an optimizer needs no code change, only a
different registry key in the config.

The extension surface is six decorators::

    from flexilearn import register_model, register_task, register_dataset

    @register_model("my_model")
    class MyModel(nn.Module): ...

    @register_task("my_task")
    class MyTask(Task):
        def loop(self):
            group = self.get_group("default")
            batch = self.get_source("main").get_next()
            group.update(loss_fn(group.run(batch)))

Drop that file under ``definitions/`` and reference ``my_task`` from a config —
:func:`discover_definitions` imports it at startup so the decorators run.

The core objects:

* :class:`Engine` — walks the curriculum, calling each task's ``loop()``.
* :class:`Task` — one training iteration; the unit you implement.
* :class:`Group` — one ``nn.ModuleDict`` plus one optimizer.
* :class:`DataSource` — a dataset, a DataLoader, and a pipeline of stages.
* :class:`EnvFactory` — simulations modeled as data sources.

See the README for the config schema and the full architecture.
"""

from .data import DataSource, Stage, wrap_function
from .discovery import discover_definitions
from .engine import Engine
from .group import Group
from .registry import (
    loss_function,
    register_dataprep,
    register_dataset,
    register_loss,
    register_model,
    register_optimizer,
    register_task,
)
from .sim import EnvFactory, Episode, PolicyFailure, borrow_env, rollout, write_mp4
from .task import EvalResult, Task

__all__ = [
    "register_model",
    "register_optimizer",
    "register_loss",
    "register_task",
    "register_dataset",
    "register_dataprep",
    "loss_function",
    "Group",
    "Task",
    "EvalResult",
    "DataSource",
    "Stage",
    "wrap_function",
    "EnvFactory",
    "borrow_env",
    "PolicyFailure",
    "Episode",
    "rollout",
    "write_mp4",
    "Engine",
    "discover_definitions",
]
