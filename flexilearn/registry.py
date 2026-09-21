# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""The component registries — how config keys become Python objects.

Each component type has one process-global dict mapping a **registry key** (the
string you write as ``type:`` in ``config.yml``) to the class that implements
it. Components register themselves at import time via the decorators here, and
:func:`flexilearn.discovery.discover_definitions` imports every module under
``definitions/`` at startup so those decorators run.

A config block like::

    models:
      main:
        type: linear
        in_features: 8

resolves ``linear`` through ``MODEL_REGISTRY`` and passes every remaining field
to the constructor as keyword arguments.

Registration is a plain dict assignment, so re-registering a key overwrites it
and importing a module twice is harmless. The PyTorch optimizers, ``linear``,
and the two common losses at the bottom of this file are pre-registered, so a
config can use them with no definitions of its own.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports at runtime
    from .task import Task

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}
OPTIMIZER_REGISTRY: dict[str, type[optim.Optimizer]] = {}
TASK_REGISTRY: dict[str, type["Task"]] = {}
LOSS_REGISTRY: dict[str, type[nn.Module]] = {}
DATASET_REGISTRY: dict[str, type[Dataset]] = {}
DATA_PREPARATION_REGISTRY: dict[str, type[Callable]] = {}
_LOSS_INSTANCES: dict[str, nn.Module] = {}

__all__ = [
    "MODEL_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "TASK_REGISTRY",
    "LOSS_REGISTRY",
    "DATASET_REGISTRY",
    "DATA_PREPARATION_REGISTRY",
]


def register_model(name: str):
    """Register an ``nn.Module`` under ``name``, usable as a ``models:`` type.

    Config fields beside ``type`` become constructor kwargs::

        @register_model("mlp")
        class MLP(nn.Module):
            def __init__(self, hidden: int = 64): ...
    """

    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def register_optimizer(name: str):
    """Register a ``torch.optim.Optimizer`` under ``name``.

    Built in already: ``adam``, ``adamw``, ``sgd``. The optimizer is constructed
    over the group's ``requires_grad`` parameters only.
    """

    def wrapper(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return wrapper


def register_task(name: str):
    """Register a :class:`~flexilearn.task.Task` subclass under ``name``.

    The class must implement ``loop()`` (one training iteration); the engine
    calls it per the curriculum. Referenced as a ``curriculum:`` entry's type.
    """

    def wrapper(cls):
        TASK_REGISTRY[name] = cls
        return cls

    return wrapper


def register_loss(name: str):
    """Register a loss module under ``name``, retrieved via :func:`loss_function`.

    Built in already: ``cross_entropy``, ``mse``.
    """

    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls

    return wrapper


def loss_function(name: str) -> nn.Module:
    """Return a cached loss instance. Stateless losses go here; learnable ones belong in a Group."""
    if name not in _LOSS_INSTANCES:
        _LOSS_INSTANCES[name] = LOSS_REGISTRY[name]()
    return _LOSS_INSTANCES[name]


def register_dataset(name: str):
    """Register a ``Dataset`` (or :class:`~flexilearn.sim.EnvFactory`) under ``name``.

    Used as a ``data_sources:`` entry's ``dataset.type``. Map-style and
    iterable-style datasets both work; an ``EnvFactory`` turns a simulator into
    a data source.
    """

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def register_dataprep(name: str):
    """Register a pipeline stage under ``name``, for a source's ``preparations:``.

    A stage is a callable class taking an iterator and yielding transformed
    items; stages are chained as generators after the DataLoader. Wrap a plain
    function with :func:`~flexilearn.data.wrap_function`.
    """

    def wrapper(cls):
        DATA_PREPARATION_REGISTRY[name] = cls
        return cls

    return wrapper


# =======================
# pytorch optimizers
# =======================
register_optimizer("adam")(optim.Adam)
register_optimizer("adamw")(optim.AdamW)
register_optimizer("sgd")(optim.SGD)

# =======================
# pytorch models
# =======================
register_model("linear")(nn.Linear)

# =======================
# pytorch losses
# =======================
register_loss("cross_entropy")(nn.CrossEntropyLoss)
register_loss("mse")(nn.MSELoss)
