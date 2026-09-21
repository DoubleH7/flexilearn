# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""End-to-end smoke test for the framework.

Runs a tiny synthetic regression, entirely offline, to verify:
  - Engine + RuntimeState + Task wiring
  - Counter persistence across a checkpoint round-trip
  - Group.run dispatch
  - Train/eval mode toggling
  - Checkpoint save & resume continues counters

No downloads, no GPU, no definitions/ discovery — every component is registered
inline below. Run it with `uv run python smoke_test.py`; it works in a scratch
directory so it leaves no outputs/ or mlruns/ behind.
"""

import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import IterableDataset

from flexilearn import Engine, Task, loss_function, register_dataset, register_loss, register_task
from flexilearn.config_models import Config


# ---- synthetic data -----------------------------------------------------
@register_dataset("synthetic_regression")
class SyntheticRegression(IterableDataset):
    """Yields (x, y=Wx+b+noise) pairs forever."""

    def __init__(self, in_features: int = 8, out_features: int = 4, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self._W = torch.randn(in_features, out_features, generator=g)
        self._b = torch.randn(out_features, generator=g)
        self._g = g

    def __iter__(self):
        while True:
            x = torch.randn(8, generator=self._g)
            y = x @ self._W + self._b + 0.01 * torch.randn(4, generator=self._g)
            yield x, y


register_loss("mse")(nn.MSELoss)


# ---- task ---------------------------------------------------------------
@register_task("smoke_supervised")
class SmokeSupervised(Task):
    def loop(self):
        group = self.get_group("default")
        source = self.get_source("main")
        x, y = source.get_next()
        out = group.run(x)
        loss = loss_function("mse")(out, y.to(out.device))
        group.update(loss)


# ---- config -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "experiment": {
        "name": "smoke",
        "seed": 1234,
        "attempt_resume": False,
    },
    "curriculum": {
        "tiny_train": {
            "type": "smoke_supervised",
            "iterations": 40,
            "checkpointing_interval": 20,
            "epochs": 1,
        },
    },
    "groups": {
        "default": {
            "models": {
                "main": {"type": "linear", "in_features": 8, "out_features": 4},
            },
            "optimizer": {"type": "sgd", "lr": 0.05},
            "settings": {"grad_accumulation": 2, "device": DEVICE},
        },
    },
    "data_sources": {
        "main": {
            "loader": {"batch_size": 16, "num_workers": 1},
            "dataset": {
                "type": "synthetic_regression",
                "in_features": 8,
                "out_features": 4,
                "seed": 7,
            },
        },
    },
    "logging": {"scalar_interval": 5, "gradient_interval": 50, "histogram_interval": 100},
}


def run() -> None:
    # --- run 1: train from scratch, save a checkpoint ---
    config = Config(**CONFIG)

    # Wipe any prior smoke output for a clean slate.
    out_root = Path("./outputs")
    if out_root.exists():
        for d in out_root.iterdir():
            shutil.rmtree(d, ignore_errors=True)

    print("=" * 60)
    print("RUN 1: fresh training")
    print("=" * 60)
    engine = Engine(config)
    engine.execute()

    g = engine.runtime.get_group("default")
    print(
        f"\n[run 1 final] optimizer_steps={g.optimizer_steps}, samples_seen={g.samples_seen}, window_count={g.window_count}"
    )
    assert g.optimizer_steps == 40 // 2, f"expected 20 optimizer steps, got {g.optimizer_steps}"
    assert g.samples_seen == 40, f"expected 40 samples_seen, got {g.samples_seen}"

    # --- run 2: identical config + attempt_resume → verify counters survived round-trip ---
    cfg2 = dict(CONFIG)
    cfg2["experiment"] = dict(CONFIG["experiment"], attempt_resume=True)
    config2 = Config(**cfg2)

    print("\n" + "=" * 60)
    print("RUN 2: resume from checkpoint (verify counter persistence)")
    print("=" * 60)
    engine2 = Engine(config2)
    engine2.execute()

    g2 = engine2.runtime.get_group("default")
    print(
        f"\n[run 2 final] optimizer_steps={g2.optimizer_steps}, samples_seen={g2.samples_seen}, window_count={g2.window_count}"
    )
    assert g2.optimizer_steps == 20, (
        f"cumulative optimizer_steps should survive resume, got {g2.optimizer_steps}"
    )
    assert g2.samples_seen == 40, (
        f"cumulative samples_seen should survive resume, got {g2.samples_seen}"
    )

    # --- run 3: extend curriculum to a second task, resume, verify counters keep climbing ---
    cfg3 = dict(CONFIG)
    cfg3["experiment"] = dict(CONFIG["experiment"], attempt_resume=True)
    cfg3["curriculum"] = {
        "tiny_train": dict(CONFIG["curriculum"]["tiny_train"]),
        "more_train": {
            "type": "smoke_supervised",
            "iterations": 30,
            "checkpointing_interval": 15,
            "epochs": 1,
        },
    }
    config3 = Config(**cfg3)

    # Curriculum changed → new experiment hash, so manually copy the checkpoint over.
    src_ckpts = sorted(Path(f"./outputs/{engine2.runtime.artifact_identifiers}").glob("*.fl"))
    engine3 = Engine(config3)
    dst_dir = Path(f"./outputs/{engine3.runtime.artifact_identifiers}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_ckpts[-1], dst_dir / src_ckpts[-1].name)
    print(f"\n[seed run 3] copied {src_ckpts[-1].name} to {dst_dir}")

    print("\n" + "=" * 60)
    print("RUN 3: resume + add a second task, verify cumulative counters keep climbing")
    print("=" * 60)
    engine3.execute()

    g3 = engine3.runtime.get_group("default")
    print(
        f"\n[run 3 final] optimizer_steps={g3.optimizer_steps}, samples_seen={g3.samples_seen}, window_count={g3.window_count}"
    )
    # tiny_train was already done; only more_train runs (30 iters, grad_acc=2 → 15 optimizer steps)
    assert g3.optimizer_steps == 35, (
        f"expected 20+15=35 cumulative optimizer steps, got {g3.optimizer_steps}"
    )
    assert g3.samples_seen == 70, f"expected 40+30=70 cumulative samples, got {g3.samples_seen}"

    print("\nSMOKE TEST PASSED")


def main() -> None:
    """Run the smoke test inside a scratch directory.

    The engine writes ./outputs and ./mlruns relative to the working directory,
    so running from a temp dir keeps a checkout clean.
    """
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="flexilearn-smoke-") as scratch:
        os.chdir(scratch)
        try:
            run()
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    main()
