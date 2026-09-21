# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Tests for DataSource loader construction.

``DataLoaderConfig.num_workers`` is ``Optional[NonNegativeInt]``, so an explicit
``num_workers: null`` is schema-valid — but the seeding branch compared it to 0
directly and raised ``TypeError: '>' not supported between 'NoneType' and 'int'``
before any data was pulled.
"""

from __future__ import annotations

from torch.utils.data import IterableDataset

from flexilearn import DataSource, register_dataset
from flexilearn.config_models import DataSourceSpec


@register_dataset("counting_dataset")
class CountingDataset(IterableDataset):
    """Yields 0, 1, 2, ... up to ``n``."""

    def __init__(self, n: int = 8):
        self.n = n

    def __iter__(self):
        yield from range(self.n)


def build_source(seed=0, **loader) -> DataSource:
    spec = DataSourceSpec(
        dataset={"type": "counting_dataset", "n": 4},
        loader=loader,
    )
    source = DataSource(datasource_spec=spec, seed=seed)
    source.initialize()
    return source


def test_explicit_null_num_workers_builds_and_pulls():
    """`num_workers: null` in YAML arrives as None and must not crash."""
    source = build_source(num_workers=None, batch_size=0)
    assert source.get_next() == 0
    assert source.get_next() == 1


def test_null_num_workers_without_a_seed_also_builds():
    """The seeding branch is skipped entirely when seed is None."""
    source = build_source(seed=None, num_workers=None, batch_size=0)
    assert source.get_next() == 0


def test_zero_num_workers_builds():
    source = build_source(num_workers=0, batch_size=0)
    assert source.get_next() == 0


def test_batch_size_zero_disables_batching():
    """batch_size 0/None hands items through unbatched (the collate override)."""
    source = build_source(num_workers=0, batch_size=0)
    assert source.get_next() == 0

    batched = build_source(num_workers=0, batch_size=2)
    first = batched.get_next()
    assert len(first) == 2
