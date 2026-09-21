# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
import random
from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config_models import DataSourceSpec
from .monitor import debug

DATASOURCE_STATE_DICT_VERSION = 1

Stage = Callable[[Iterator], Iterator]


def wrap_function(func):
    class FunctionStage:
        def __init__(self):
            if not callable(func):
                raise TypeError("func must be callable")

        def __call__(self, iterator):
            for item in iterator:
                yield func(item)

    return FunctionStage


class DataSource:
    def __init__(self, datasource_spec: DataSourceSpec, seed: int = None):
        dataset_ctor, dataset_kwargs = datasource_spec.dataset.instanciation_args
        prep_ctor_list = []
        prep_kwargs_list = []
        for prep in datasource_spec.preparations:
            prep_ctor, prep_kwargs = prep.instanciation_args
            prep_ctor_list.append(prep_ctor)
            prep_kwargs_list.append(prep_kwargs)

        loader_kwargs = datasource_spec.loader

        self.loader_kwargs = loader_kwargs or {}
        self.dataset_ctor = dataset_ctor
        self.dataset_kwargs = dataset_kwargs
        self.prep_ctors_list = prep_ctor_list
        self.prep_kwargs_list = prep_kwargs_list
        self.seed = seed
        self._reset_counter = 0
        stage_names = [p.type for p in datasource_spec.preparations]
        debug(
            f"DataSource configured: dataset={datasource_spec.dataset.type}, stages=[{', '.join(stage_names)}]"
        )

    def initialize(self):
        self._dataset = self.dataset_ctor(**self.dataset_kwargs)
        self._validate_env_factory_loader()
        applied_loader_kwargs = self.loader_kwargs.model_dump()
        presumed_batch_size = applied_loader_kwargs.get("batch_size")
        if (
            presumed_batch_size is None or presumed_batch_size == 0
        ):  # batching is disabled by default
            applied_loader_kwargs["batch_size"] = 1
            applied_loader_kwargs["collate_fn"] = lambda x: x[0]
        # `num_workers` is Optional in the schema, so an explicit `null` in
        # config reaches here as None. DataLoader rejects that, and null means
        # the same thing as 0: load in the main process.
        if applied_loader_kwargs.get("num_workers") is None:
            applied_loader_kwargs["num_workers"] = 0
        if self.seed is not None:
            epoch_seed = self.seed + self._reset_counter
            if applied_loader_kwargs.get("shuffle"):
                generator = torch.Generator()
                generator.manual_seed(epoch_seed)
                applied_loader_kwargs["shuffle"] = True
                applied_loader_kwargs["generator"] = generator
            if applied_loader_kwargs["num_workers"] > 0:

                def worker_init_fn(worker_id):
                    worker_seed = epoch_seed + worker_id
                    # Runs inside each freshly forked worker process, so these
                    # seed that worker's interpreter only — the parent's global
                    # RNG state is untouched.
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    torch.manual_seed(worker_seed)

                applied_loader_kwargs["worker_init_fn"] = worker_init_fn
        self._iterator = iter(DataLoader(self._dataset, **applied_loader_kwargs))
        for ctor, kwargs in zip(self.prep_ctors_list, self.prep_kwargs_list, strict=True):
            self._iterator = ctor(**kwargs)(self._iterator)
        seed_info = (
            f"seed={self.seed + self._reset_counter}" if self.seed is not None else "no seed"
        )
        debug(f"DataSource pipeline ready (epoch {self._reset_counter}, {seed_info})")

    def _validate_env_factory_loader(self):
        """Fail loudly when an EnvFactory source is mis-declared in config.

        Environments are live objects: they cannot be collated into batches and
        (usually) cannot be pickled across DataLoader worker processes — the
        source must hand them through untouched.
        """
        from .sim import EnvFactory

        if not isinstance(self._dataset, EnvFactory):
            return
        kwargs = self.loader_kwargs.model_dump()
        bad = []
        if kwargs.get("batch_size"):
            bad.append(f"batch_size={kwargs['batch_size']}")
        if kwargs.get("num_workers"):
            bad.append(f"num_workers={kwargs['num_workers']}")
        if kwargs.get("shuffle"):
            bad.append("shuffle=true")
        if bad:
            raise ValueError(
                f"Environment data source ({type(self._dataset).__name__}) has "
                f"incompatible loader settings: {', '.join(bad)}. Envs are live "
                f"objects — declare the source with "
                f"`loader: {{batch_size: 0, num_workers: 0}}` so get_next() "
                f"returns one env through the pipeline untouched."
            )

    def get_next(self):
        return next(self._iterator)

    @property
    def dataset(self):
        """The built dataset instance (available after initialize()).

        Lets a task reach dataset-owned state — e.g. a fitted normalizer on an
        offline demo dataset, needed by an evaluation rollout.
        """
        return self._dataset

    def reset_state(self):
        debug(f"DataSource resetting for epoch {self._reset_counter + 1}")
        self._reset_counter += 1
        self.initialize()

    # ------------------
    # Checkpointing
    # ------------------
    @property
    def source_state_dict(self) -> dict:
        """Get state dict for checkpoint saving."""
        return {
            "format_version": DATASOURCE_STATE_DICT_VERSION,
            "dataset_kwargs": self.dataset_kwargs,
            "prep_kwargs_list": self.prep_kwargs_list,
            "loader_kwargs": self.loader_kwargs,
            "seed": self.seed,
            "reset_counter": self._reset_counter,
        }

    def load_source_state(self, state: dict):
        """Load state from checkpoint dict."""
        if state["format_version"] != DATASOURCE_STATE_DICT_VERSION:
            raise RuntimeError(
                f"DataSource Checkpoint version mismatch. "
                f"Expected {DATASOURCE_STATE_DICT_VERSION}, "
                f"got {state['format_version']}."
            )
        self.dataset_kwargs = state.get("dataset_kwargs", self.dataset_kwargs)
        self.prep_kwargs_list = state.get("prep_kwargs_list", self.prep_kwargs_list)
        self.loader_kwargs = state.get("loader_kwargs", self.loader_kwargs)
        self.seed = state.get("seed", self.seed)
        self._reset_counter = state.get("reset_counter", 0)
        self.initialize()
        debug(f"DataSource state restored: reset_counter={self._reset_counter}")

    @classmethod
    def from_checkpoint(cls, state: dict, spec: DataSourceSpec) -> "DataSource":
        """Create DataSource from checkpoint state."""
        # Create a new DataSource with the spec and restored state
        source = cls(spec, seed=state.get("seed"))
        source.load_source_state(state)
        return source
