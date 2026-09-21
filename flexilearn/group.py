# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config_models import DEFAULT_GROUP_NAME, GroupCheckpointPolicy, GroupSpec
from .monitor import debug, warn

GROUP_STATE_DICT_VERSION = 3


class Group(nn.Module):
    """One ``nn.ModuleDict`` plus (at most) one optimizer — the unit of learning.

    A group is what a task trains. It bundles one or more models with the single
    optimizer that steps them, and owns everything that decides *how* they
    learn: gradient accumulation, freezing and unfreezing, the learning rate,
    the device, and what goes into a checkpoint.

    Tasks reach one via ``self.get_group("name")`` and drive it with two calls::

        group = self.get_group("default")
        output = group.run(batch)     # forward
        group.update(loss)            # backward + (accumulated) optimizer step

    ``run()`` with no ``model=`` argument chains the models in config order,
    threading each output into the next, so declaration order matters. Pass
    ``model="name"`` to run exactly one, or reach the raw module through the
    public ``group.models`` dict for custom entry points like ``.generate()``.

    ``update()`` scales the loss by ``1/grad_accumulation`` and steps only once
    the accumulation window fills. A group that is not trainable — or that
    declares no ``optimizer:`` block — accepts ``update()`` and does nothing,
    which is what makes a frozen reference model or reward model expressible in
    config alone.

    A group named ``default`` is required by config validation.
    """

    def __init__(
        self,
        group_spec: GroupSpec,
        name: str = DEFAULT_GROUP_NAME,
    ):
        """Build every model in ``group_spec`` and, unless omitted, its optimizer."""
        super().__init__()
        self._name = name

        # Model instanciation. Each model's __init__ runs arbitrary setup —
        # freezing a backbone, attaching LoRA adapters, enabling gradient
        # checkpointing / enable_input_require_grads(). Group never re-wraps or
        # replaces the module object after this, so that setup is preserved.
        self.models = nn.ModuleDict()
        for model_name, model_spec in group_spec.models.items():
            model_ctor, model_kwargs = model_spec.instanciation_args
            self.models[model_name] = model_ctor(**model_kwargs)

        # Snapshot the model-declared trainability of every parameter so that
        # set_trainable() can restore a frozen backbone instead of blindly
        # unfreezing it (see set_trainable()).
        self._init_requires_grad = {
            p_name: p.requires_grad for p_name, p in self.named_parameters()
        }

        # Optimizer instanciation — built only over parameters that require
        # grad, so a frozen backbone / PEFT setup (LoRA, adapters, linear
        # probes) optimizes just the unfrozen weights.
        #
        # `optimizer:` is optional. A group declared without one is an
        # inference-only unit — a reference policy, a reward model, a frozen
        # teacher — so no optimizer is built and `update()` becomes a no-op.
        self.optimizer: Optimizer | None = None
        if group_spec.optimizer is not None:
            optim_ctor, optim_kwargs = group_spec.optimizer.instanciation_args
            trainable_params = [p for p in self.models.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError(
                    f"Group '{name}': optimizer '{group_spec.optimizer.type}' received "
                    f"zero trainable parameters — every model parameter has "
                    f"requires_grad=False. Unfreeze the adapter / LoRA / head "
                    f"parameters you intend to train (typically in the model's "
                    f"__init__), or remove this group's `optimizer:` block if the "
                    f"group is meant to stay fully frozen."
                )
            self.optimizer = optim_ctor(trainable_params, **optim_kwargs)

        # Other settings
        self._grad_acc_steps = group_spec.settings.grad_accumulation
        self._device = torch.device(group_spec.settings.device)
        self._trainable = group_spec.settings.is_trainable

        # An optimizer-less group cannot take a step, so it is never trainable
        # regardless of what `is_trainable` asked for.
        if self.optimizer is None and self._trainable:
            warn(
                f"Group '{name}' declares no `optimizer:` block, so it cannot be "
                f"trained — forcing is_trainable=False. Add an optimizer if this "
                f"group is meant to learn."
            )
            self._trainable = False

        # behavior control variables
        self._optimizer_steps = 0  # cumulative optimizer.step() calls, never reset
        self._samples_seen = 0  # cumulative forward/backward passes, never reset
        self._window_count = 0  # passes since last checkpoint save, reset after each save
        self._acc_step_count = 0  # accumulation position within current optimizer step

        # Experiment tracker, injected by RuntimeState after construction. None
        # until then (and for groups built outside a runtime, e.g. in tests).
        self._tracker = None

        # initialization
        self.to(self._device)
        model_summary = ", ".join(f"{n}({s.type})" for n, s in group_spec.models.items())
        debug(
            f"Group built: [{model_summary}] on {self._device}, trainable={self._trainable}, grad_acc={self._grad_acc_steps}"
        )

    # ------------------
    # training/interface
    # ------------------
    def set_tracker(self, tracker) -> None:
        """Attach the experiment tracker used for gradient logging in update()."""
        self._tracker = tracker

    @property
    def device(self):
        return self._device.type

    def run(self, *args, model: str | None = None, **kwargs) -> Tensor:
        """Forward pass through the group's models.

        If `model` is given, runs only that named model with all passed args/kwargs.
        Otherwise, chains all models in config order, threading the output of each into the next.
        """
        if model is not None:
            return self.models[model](*args, **kwargs)
        x = args[0] if args else kwargs.get("x")
        x = x.to(self._device)
        for _, m in self.models.items():
            x = m(x)
        return x

    def update(self, loss: Tensor) -> None:
        if not self._trainable or self.optimizer is None:
            return

        (loss / self._grad_acc_steps).backward()
        self._acc_step_count += 1
        self._window_count += 1
        self._samples_seen += 1

        if self._acc_step_count >= self._grad_acc_steps:
            # Log gradients while they are still live (before zero_grad), keyed
            # by the group's optimizer-step counter. No-op when tracking is off.
            if self._tracker is not None:
                self._tracker.maybe_log_gradients(
                    self._name, self.named_parameters(), self._optimizer_steps
                )
                self._tracker.maybe_log_lr(self._name, self.optimizer, self._optimizer_steps)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._optimizer_steps += 1
            self._acc_step_count = 0

    @property
    def window_count(self) -> int:
        return self._window_count

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    @property
    def unapplied_steps(self) -> bool:
        return bool(self._acc_step_count)

    def reset_window(self) -> None:
        self._window_count = 0

    # ------------------
    # Mode Control
    # ------------------
    def set_mode(self, mode: str) -> None:
        """Switch between 'train' and 'eval' mode (affects dropout, batchnorm, etc.)."""
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'train' or 'eval'.")

    # ------------------
    # Trainability Control
    # ------------------
    def set_trainable(self, flag: bool) -> None:
        """Freeze or unfreeze the group.

        When disabling, every parameter is hard-frozen. When enabling, each
        parameter is restored to the trainability its model declared at
        construction — so a frozen backbone (PEFT / linear probe) stays frozen
        across a checkpoint resume rather than being silently unfrozen.
        """
        if flag and self.optimizer is None:
            debug(f"Group '{self._name}' has no optimizer; ignoring set_trainable(True)")
            return
        self._trainable = flag
        for p_name, p in self.named_parameters():
            p.requires_grad_(flag and self._init_requires_grad.get(p_name, True))
        if not flag and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        debug(f"Group '{self._name}' trainability set to {flag}")

    @property
    def trainable(self):
        return self._trainable

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate on every optimizer param group."""
        if self.optimizer is None:
            debug(f"Group '{self._name}' has no optimizer; ignoring learning rate")
            return
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        debug(f"Group '{self._name}' learning rate set to {lr}")

    # ------------------
    # Per-task Overrides (apply on task enter, restore on task exit)
    # ------------------
    def override_snapshot(self) -> dict:
        """Capture the override-able state so it can be restored after a task."""
        return {
            "trainable": self._trainable,
            "lrs": (
                [] if self.optimizer is None else [pg["lr"] for pg in self.optimizer.param_groups]
            ),
        }

    def apply_override(self, override) -> None:
        """Apply a GroupOverride; only its non-None fields take effect."""
        if override.trainable is not None:
            self.set_trainable(override.trainable)
        if override.learning_rate is not None:
            self.set_learning_rate(override.learning_rate)

    def restore_override(self, snapshot: dict) -> None:
        """Restore the state captured by ``override_snapshot``."""
        self.set_trainable(snapshot["trainable"])
        if self.optimizer is None:
            return
        for pg, lr in zip(self.optimizer.param_groups, snapshot["lrs"], strict=True):
            pg["lr"] = lr

    # ------------------
    # State Dict Handling
    # ------------------
    def checkpoint_state(self, policy: GroupCheckpointPolicy) -> dict:
        """Build this group's checkpoint state under an inclusion policy.

        ``policy.weights`` scopes the serialized model weights — ``full`` keeps
        every parameter and buffer, ``trainable`` keeps only requires_grad
        parameters (a PEFT adapter alone), ``none`` keeps no weights. See
        ``GroupCheckpointPolicy``.

        Note: ``trainable`` persists trainable *parameters* only, not buffers
        (e.g. BatchNorm running stats). A model with non-reproducible buffers
        must use ``full``. The counters are always included — they are tiny and
        required for a correct resume.

        Tensors are moved to CPU on the way out so that a checkpoint written on
        a CUDA host stays loadable (and inspectable) on a CPU-only one.
        """
        if policy.weights == "full":
            model_state = {name: t.cpu() for name, t in super().state_dict().items()}
        elif policy.weights == "trainable":
            model_state = {
                name: param.detach().cpu()
                for name, param in self.named_parameters()
                if param.requires_grad
            }
        else:  # "none"
            model_state = {}

        state = {
            "format_version": GROUP_STATE_DICT_VERSION,
            "weights_scope": policy.weights,
            "model": model_state,
            "optimizer_steps": self._optimizer_steps,
            "samples_seen": self._samples_seen,
            "window_count": self._window_count,
            "acc_step_count": self._acc_step_count,
            "trainable": self._trainable,
        }
        if policy.optimizer and self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def _load_group_state(self, state: dict):
        version = state.get("format_version")
        if version != GROUP_STATE_DICT_VERSION:
            raise RuntimeError(
                f"Group checkpoint version mismatch: expected v{GROUP_STATE_DICT_VERSION}, "
                f"got v{version}. Older checkpoints are not compatible with the current "
                f"checkpoint schema — delete the outputs/ directory to start fresh."
            )

        # A "full" snapshot holds every key and loads strictly; "trainable" and
        # "none" snapshots are partial by design (the base is rebuilt by the
        # model constructor before this runs), so they load non-strict.
        weights_scope = state.get("weights_scope", "full")
        self.load_state_dict(state.get("model", {}), strict=(weights_scope == "full"))
        debug(f"Model weights restored (scope={weights_scope})")

        optimizer_state = state.get("optimizer")
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_to_device()
            debug("Optimizer state restored")

        self.set_trainable(state.get("trainable"))
        self._optimizer_steps = state.get("optimizer_steps", 0)
        self._samples_seen = state.get("samples_seen", 0)
        self._window_count = state.get("window_count", 0)
        self._acc_step_count = state.get("acc_step_count", 0)
        debug(
            f"Group restored: optimizer_steps={self._optimizer_steps}, samples_seen={self._samples_seen}"
        )

    def _move_optimizer_to_device(self):
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    @classmethod
    def from_checkpoint(
        cls, state: dict, spec: GroupSpec, name: str = DEFAULT_GROUP_NAME
    ) -> "Group":
        """Create Group from checkpoint state dict."""
        group = cls(spec, name=name)
        group._load_group_state(state)
        return group
