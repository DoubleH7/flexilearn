# Known issues

Open findings in the framework, plus a changelog of fixed ones kept for their root
causes. Add new open items above the `## Fixed` heading, continuing the numbering.

Items here are tracked in the file rather than only in the issue tracker so that the
reasoning survives alongside the code. User-facing consequences are also summarized in
the README's "Known limitations".

---

## Open

### 1. Datasource mid-epoch position is not checkpointed

`DataSource.source_state_dict` persists only `reset_counter` + constructor kwargs, not the
iterator position. On a mid-epoch resume the task's restored `_iter_count` says "iteration
300" while the rebuilt source streams from the top of the epoch again — the first samples
get re-trained on and the epoch's tail is never seen (shuffle order also differs, since a
reservoir shuffle draws from restored global RNG). Not silent-corruption level, but worth
either documenting as a hard limitation or saving a sample cursor.

**Workaround:** checkpoint on epoch boundaries when exact sample coverage matters.

### 2. Repeated evals when a group-counter `global_step` stalls

`Engine.run_loop` fires evaluation on `task.global_step % evaluation_interval == 0`. A
task that overrides `global_step` to a group counter can stall that counter — for example
by skipping `group.update()` on a batch with nothing to learn from — making the same step
value hit the modulo on consecutive iterations, and producing duplicate evaluations.

**Fix direction:** track the last evaluated step and skip a repeat, rather than relying on
the modulo alone.

### 3. `reset_datasources` leaks persistent envs

`DataSource.reset_state()` → `initialize()` rebuilds `self._dataset` from the constructor,
discarding the previous `EnvFactory` instance without calling its `close()`. A cached
`persistent: true` env — the setting that exists precisely for expensive sims — is dropped
un-closed at every epoch boundary and rebuilt from scratch.

**Fix direction:** call `close()` on the outgoing dataset in `reset_state()` when it is an
`EnvFactory`.

---

## Fixed

Kept as a changelog with root causes.

### F1. `grad_accumulation` can never be 0 despite its own "disable" message

`GroupSettings.grad_accumulation` was `PositiveInt` yet its validator message said "(to
disable, set to zero)" — dead code, since `PositiveInt` already forbids <= 0, and 0 would
divide the loss by zero in `Group.update`. Resolution: 1 *is* the "disabled" state (an
optimizer step every update). The dead validator and the misleading message were removed.

### F2. `checkpointing_interval` 0 is unreachable but the consumer expects it

`TaskSpec.checkpointing_interval` was `PositiveInt` (rejecting 0) while
`Task.checkpoint_flag` treats 0 as "checkpointing disabled". Now `NonNegativeInt`, so 0
actually disables checkpointing; the misnamed dead validator was removed.

### F3. Omitting `seed` crashes a fresh run

`ExperimentConfig.seed` was `Optional[int] = None`, but `RuntimeState.initialize` calls
`torch.manual_seed(seed)`, which raises on `None`. Now `seed: int = 0` — a deterministic
default (a random one would churn the config hash and thus the checkpoint directory).

### F4. `overrides` was dead config

`Task` stored `self.overrides` but nothing read it. Fixed by the per-task group overrides
feature; covered by `tests/test_overrides.py`.

### F5. `retry` budget never reset

`Engine.run_loop` initialised `retries` once per task and never reset it, so
`upon_failure: retry` was a global budget of `max_retries` failures rather than a
per-iteration one, and the log message "on same iteration" was inaccurate. Fixed by the
failure-policy rework; covered by `tests/test_failure_policy.py`.

### F6. `retry` restored nothing for lazily-built groups

The pre-task snapshot in `Engine._run_task` only captured groups already built at task
entry — but in the common case the group is first built lazily inside the task's first
`loop()`, so the snapshot was empty and "restoring pre-task parameters" silently kept the
failed attempt's trained weights while logging that it had restored them. Fix: on retry,
groups first built during the failed attempt are dropped from the runtime and lazily
rebuilt from spec — fresh construction *is* their pre-task state.

### F7. A frozen, optimizer-less group could not be built

`GroupSpec.optimizer` was `Optional` but `Group.__init__` dereferenced it
unconditionally, raising `AttributeError: 'NoneType'` when the block was omitted. Worse,
`Group`'s own zero-trainable-params error advised "remove this group's `optimizer:` block
if the group is meant to stay fully frozen" — advice that crashed if followed. Now an
optimizer-less group builds as an inference-only unit: no optimizer, `update()` is a
no-op, `is_trainable` is forced false, and the optimizer is omitted from checkpoints.
Covered by `tests/test_group.py`.

### F8. Checkpoints were not portable across devices

`Group.checkpoint_state` stored `param.detach()` still on device and
`RuntimeState.attempt_checkpoint_load` called `torch.load(..., weights_only=False)` with
no `map_location`, so a checkpoint written on a CUDA host could not be resumed or even
inspected on a CPU-only one. Now tensors are moved to CPU on save and loaded with
`map_location="cpu"`; the existing `_move_optimizer_to_device` moves optimizer state back.
Covered by `tests/test_checkpointing.py`.

### F9. `Task.checkpoint_flag` crashed for group-less tasks

With `cumulative_checkpointing_counter: false` the counter reducer is `max`, and `max([])`
raises `ValueError` when the task registers no groups (a collection- or eval-style task
never calls `get_group()`). The default `sum` path returned 0 and was safe. Now the empty
case is checked explicitly and reports "no checkpoint due". Covered by
`tests/test_checkpointing.py`.

### F10. `num_workers: null` crashed `DataSource.initialize`

`DataLoaderConfig.num_workers` is `Optional[NonNegativeInt]`, so an explicit `null` was
schema-valid, but it was compared with `> 0` (a `TypeError` on `None`) and then passed
through to `DataLoader`, which rejects it too. Null now normalizes to 0, matching the
field's documented meaning of "load in the main process". Covered by `tests/test_data.py`.

### F11. An installed wheel could not run at all

`discover_definitions()` resolved `definitions/` from `Path(__file__).parent.parent`,
which is `site-packages/` in an installed wheel — so the CLI always raised
`FileNotFoundError` and the published package was unusable outside a source checkout.
Resolution is now explicit path → `$FLEXILEARN_DEFINITIONS` → `./definitions` → package
parent, and a missing directory warns instead of raising. Guarded by a CI job that
installs the built wheel into a clean environment and runs its CLI.
