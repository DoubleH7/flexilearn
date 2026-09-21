# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""The smallest useful flexilearn experiment: a linear fit on synthetic data.

Two components, each registered with a decorator and referenced from
``config.yml`` by its key — this is the whole extension mechanism:

* ``synthetic_regression`` (a dataset) yields ``(x, y = Wx + b + noise)`` pairs.
* ``supervised_regression`` (a task) defines one training iteration.

The model (``linear``) and the loss (``mse``) are built in, so nothing else
needs registering. Run it with::

    flexilearn examples/minimal/config.yml --definitions examples/minimal/definitions
"""

import torch
from torch.utils.data import IterableDataset

from flexilearn import Task, loss_function, register_dataset, register_task


@register_dataset("synthetic_regression")
class SyntheticRegression(IterableDataset):
    """Yields (x, y) pairs from a fixed random linear map, forever."""

    def __init__(self, in_features: int = 8, out_features: int = 4, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.in_features = in_features
        self.out_features = out_features
        self._W = torch.randn(in_features, out_features, generator=generator)
        self._b = torch.randn(out_features, generator=generator)
        self._generator = generator

    def __iter__(self):
        while True:
            x = torch.randn(self.in_features, generator=self._generator)
            noise = 0.01 * torch.randn(self.out_features, generator=self._generator)
            yield x, x @ self._W + self._b + noise


@register_task("supervised_regression")
class SupervisedRegression(Task):
    """One iteration: pull a batch, forward it through the group, step."""

    def loop(self):
        group = self.get_group("default")
        source = self.get_source("main")

        x, y = source.get_next()
        prediction = group.run(x)
        loss = loss_function("mse")(prediction, y.to(prediction.device))
        group.update(loss)

        # Recorded to MLflow at `logging.scalar_interval`, and printed by the
        # engine's progress display.
        self.log_scalar("train/loss", loss.item())
