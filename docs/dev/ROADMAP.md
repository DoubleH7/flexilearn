# Roadmap — framework (`flexilearn/`)

Planned work on the framework itself, with the reasoning that motivates each item. The
user-facing summary lives in the README's "Roadmap" section; this file is the detail.

Two items from earlier revisions of this document have since shipped: the checkpoint
portability fix with a `tests/test_checkpointing.py` suite, and optimizer-less groups.
Both are recorded in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) as F7–F9.

---

## 1. Rewrite `Group.run()`

**Problem.** With no `model=` argument, `run()` takes `args[0]`, moves it to the device,
and chains every model in `ModuleDict` order. Three consequences:

* **A silent-data-loss shape.** A call like `group.run(a, b)` on the chaining path binds
  `args[0]` and **silently discards `b`** — a two-input loss computed on one input, no
  error raised.
* **Real work routes around it.** Any task with a non-tensor batch (a multimodal
  processor dict, say) cannot use the chaining path at all and reaches into
  `group.models[...]` directly.
* **Inconsistent device handling.** The `model=` branch returns before the
  `.to(self._device)`, so the convenience exists only on the path that chaining users
  don't take.

**Fix.** Drop chaining; dispatch to the sole model when the group has one; move tensor
args on *both* paths:

```python
def run(self, *args, model: str | None = None, **kwargs) -> Tensor:
    """Forward through a named model, or the only model when the group has one."""
    if model is None:
        if len(self.models) != 1:
            raise ValueError(
                f"Group '{self._name}' has {len(self.models)} models "
                f"({sorted(self.models)}) — pass model='<name>' to pick one."
            )
        model = next(iter(self.models))
    to_dev = lambda v: v.to(self._device) if isinstance(v, Tensor) else v
    return self.models[model](*map(to_dev, args), **{k: to_dev(v) for k, v in kwargs.items()})
```

Single-model groups relying on the device move keep working unchanged. Because kwargs
would be moved too, tasks could use `run(model="x", **batch)` instead of reaching into
`group.models`.

**Then** delete the `Group.run()` entry under "Known Issues" in `CLAUDE.md` and the
"model order in config matters" caveat — it stops being true.

---

## 2. Small-fixes batch

* **`_samples_seen` measures the wrong thing.** It is incremented once per `update()`
  call, so it counts update calls, not samples — off by the batch factor for anything
  reading it as throughput. Rename to `update_calls`. It is a checkpoint key, so bump
  `GROUP_STATE_DICT_VERSION`; the version-mismatch path already tells users to delete
  `outputs/`.
* **The `default` group name is a fossil.** `Config.validate_default_group_exists` forces
  every config to declare a group named literally `default`, even when the natural name is
  something else. Replace with a "groups is non-empty" check and drop the name
  requirement — the test suite names its group `default` by choice, so tests are
  unaffected.
* **Gradients survive the epoch boundary.** `Engine.run_loop` treats `StopIteration` as a
  clean epoch end and breaks out — but if `_acc_step_count > 0`, the accumulated gradients
  sit in `.grad` and are added to the next epoch's first `update()`, producing one
  oversized effective step spanning the boundary. `groups_can_checkpoint` already knows
  about `unapplied_steps`; the epoch boundary should too. Decide explicitly whether to
  flush (step) or discard (`zero_grad`), and document which.

---

## 3. Datasource resume — fix the contract, not the code

Tracked as open issue 1 in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

**Do not attempt exact mid-epoch resume.** Generator-based `Stage` pipelines have no seek,
and a reservoir shuffle draws from restored *global* RNG, so even a perfect cursor would
not reproduce the order. The achievable contract is epoch-granular resumption; the problem
is that `attempt_resume: true` presents it as general.

**Do instead:**

* Document the real guarantee on `DataSource.source_state_dict`.
* `warn()` on resume when a task restores with `_iter_count > 0`: the data source restarts
  at the epoch head, so roughly N samples will be re-seen and the epoch tail skipped.

**If exactness is ever needed**, add it as an opt-in protocol rather than a core feature:
if a dataset defines `state_dict` / `load_state_dict`, `DataSource` delegates to it;
otherwise the documented fallback applies.

---

## 4. LR scheduler

**Problem.** There is no scheduling abstraction — only `Group.set_learning_rate` and the
per-task `learning_rate` override. For warmup-sensitive fine-tuning this is the most
conspicuous missing feature; today the only way to get a warmup is to hand-roll it inside
a task's `loop()`.

**Design — follow the existing pattern exactly.** A `SCHEDULER_REGISTRY` with torch's
built-ins pre-registered the way optimizers already are, plus an optional per-group block:

```yaml
groups:
  main:
    optimizer: {type: adamw, lr: 1e-4}
    scheduler: {type: cosine_warmup, warmup_steps: 500, total_steps: 20000}
```

**Two things to get right:**

* Step it in `Group.update()` immediately after `optimizer.step()` — per *optimizer* step,
  never per `update()` call. Under `grad_accumulation: 4` those differ by 4×.
* Persist scheduler state alongside the optimizer's in `checkpoint_state` (another
  `GROUP_STATE_DICT_VERSION` bump — batch it with item 2's if both land together).

**One conflict to resolve explicitly:** a per-task `learning_rate` override and a
scheduler fight, since the scheduler overwrites the override on its next step. Either
reject the combination in validation, or have `apply_override` rebase the scheduler's
`base_lrs`. Pick one and write it down.

---

## 5. Scope discipline (standing rule)

Config-as-code has a known failure mode: the YAML slowly becomes a programming language.
This framework is not there, but the first symptoms exist.

**Rule to adopt:** a new config field needs two independent experiments that want it.
Anything expressible inside a task's `loop()` stays in the task.

**Reconsider the eval-phase idiom.** `upon_failure: until` + `epochs: -1` +
`evaluate_at_start` yields a pure-eval curriculum phase with no engine changes — genuinely
elegant, and evidence that the primitives are orthogonal. It is also "run one eval pass"
expressed as an edge case of the failure-policy state machine, which is obvious to whoever
wrote it and opaque six months later. An explicit `mode: eval, epochs: 1` phase would say
what it means. Weigh the clarity against the extra engine branch before adding more of
these.

The `checkpointing:` inclusion dictionary with per-group override policies is arguably the
second instance. Each addition is individually justified; the aggregate is what to watch.
