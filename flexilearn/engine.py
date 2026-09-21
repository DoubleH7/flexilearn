# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
import time

from .config_models import Config, GroupCheckpointPolicy
from .monitor import create_progress, debug, error, info, success, warn
from .runtime_state import RuntimeState
from .task import Task


class _RestartTask(Exception):
    """Internal control-flow signal: unwind to the task level and restart it.

    Raised by the failure handlers when ``upon_failure='retry'`` decides a task
    should be restarted from scratch. Caught only by ``Engine._run_task``; never
    escapes the engine.
    """


class Engine:
    """Drives a curriculum: the top-level object a run is built around.

    The engine walks ``curriculum:`` in declaration order and, for each task,
    calls ``task._step()`` (and through it your ``loop()``) up to ``iterations``
    times per epoch. Around that inner loop it owns the concerns a task should
    not have to: advancing epochs when a data source is exhausted, applying and
    restoring per-task group overrides, running periodic evaluations, enforcing
    the task's ``upon_failure`` policy, triggering checkpoints, and recording
    throughput.

    Construction is cheap — groups and data sources are built lazily by
    :class:`~flexilearn.runtime_state.RuntimeState` on first access, and a
    resume is attempted when ``execute()`` opens the run::

        config = parse_config("config.yml")
        Engine(config).execute()
    """

    def __init__(self, config: Config):
        """Build the runtime for ``config``. Nothing is constructed until execute()."""
        self.runtime = RuntimeState(config)
        # Throughput accounting: a global step counter plus a rolling window of
        # wall-clock spent in task._step(), flushed to MLflow on scalar_interval.
        self._global_step = 0
        self._perf_elapsed = 0.0
        self._perf_count = 0

    def execute(self):
        """Run the whole curriculum start to finish.

        Initializes the runtime (which opens or reattaches the tracking run and
        restores a checkpoint when ``attempt_resume`` is set), runs each task in
        order — skipping any the checkpoint records as already complete — and
        finalizes with a last checkpoint. Exceptions escape only when the
        failing task's ``upon_failure`` policy is ``exit``.
        """
        self.runtime.initialize()  # opens the MLflow run (reattaches on resume)
        try:
            curriculum = self.runtime._task_specs
            info("Starting experiment execution")
            for t_name, taskspec in curriculum.items():
                task_states = self.runtime.get_task_states()
                if t_name in task_states:
                    debug(f"Task '{t_name}' resumed from checkpoint")
                    task = Task.from_checkpoint(
                        task_states[t_name], self.runtime._task_specs[t_name], self.runtime
                    )
                else:
                    task_ctor = taskspec.ctor
                    task = task_ctor(taskspec, self.runtime)
                self.runtime._task_specs[t_name] = taskspec
                self.runtime._tasks[t_name] = task
                info(f"Executing task: '{t_name}'")
                # Each curriculum task gets its own (nested) MLflow run, so its
                # metrics/checkpoints are isolated under the experiment parent run.
                self.runtime.tracker.start_task_run(t_name, taskspec.model_dump(mode="json"))
                try:
                    self._run_task(t_name, task)
                except BaseException:
                    self.runtime.tracker.end_task_run(status="FAILED")
                    raise
                else:
                    self.runtime.tracker.end_task_run(status="FINISHED")
            info("Finalizing experiment...")
            self.runtime.checkpoint_all()
        except BaseException:
            # Close the tracking run as failed, then re-raise (covers exceptions
            # and KeyboardInterrupt). FINISHED is only reached on a clean run.
            self.runtime.tracker.end(status="FAILED")
            raise
        else:
            self.runtime.tracker.end(status="FINISHED")
            success("Experiment completed successfully!")

    def _run_task(self, task_name, task: Task):
        """Run one curriculum task, applying per-task overrides and `retry`.

        Force-builds any group named by the task's overrides, snapshots the
        baseline (pre-override) state for `retry` restarts, applies the overrides,
        and restores them when the task ends so the shared group is clean for the
        next task.
        """
        overrides = task.overrides or {}
        # Build override targets first so the retry snapshot captures their
        # baseline (pre-override) state.
        for gname in overrides:
            self.runtime.get_group(gname)
        snapshot = self._snapshot_groups()

        override_snaps = {}
        for gname, ov in overrides.items():
            group = self.runtime.get_group(gname)
            override_snaps[gname] = group.override_snapshot()
            group.apply_override(ov)

        try:
            retries = 0
            while True:
                try:
                    self.run_loop(task_name, task)
                    return
                except _RestartTask:
                    retries += 1
                    if retries > task.max_retries:
                        error(f"Task '{task_name}' exhausted {task.max_retries} retries — exiting.")
                        # `from None`: _RestartTask is an internal control
                        # signal, not the underlying cause worth chaining.
                        raise RuntimeError(
                            f"Task '{task_name}' exceeded max_retries={task.max_retries}."
                        ) from None
                    warn(
                        f"Task '{task_name}': retry {retries}/{task.max_retries} — restoring "
                        f"pre-task parameters and restarting (RNG/data not rewound, so the "
                        f"attempt diverges)."
                    )
                    self._restore_groups(snapshot)
                    # Groups first built *during* the failed attempt have no
                    # baseline in the pre-task snapshot — their pre-task state
                    # is "not built yet". Drop them so the restarted attempt
                    # lazily rebuilds them from spec (fresh construction);
                    # otherwise the retry would silently keep the failed
                    # attempt's trained weights.
                    for gname in task._group_names - set(snapshot):
                        self.runtime._groups.pop(gname, None)
                    task._iter_count = 0
                    task._epoch_count = 0
                    # Re-apply overrides on top of the restored baseline.
                    for gname, ov in overrides.items():
                        self.runtime.get_group(gname).apply_override(ov)
        finally:
            # Restore the override knobs (trainable / lr) to baseline; trained
            # weights are kept. Runs even on crash/exit so the group stays clean.
            for gname, snap in override_snaps.items():
                self.runtime.get_group(gname).restore_override(snap)

    def run_loop(self, task_name, task: Task):
        """Run one task's epoch/iteration loop until it completes.

        Each iteration calls ``task._step()``; ``StopIteration`` from an
        exhausted data source ends the epoch early and resets the sources.
        Along the way this fires evaluations on ``evaluation_interval``, saves a
        checkpoint whenever ``task.checkpoint_flag`` goes true, and routes
        crashes through the task's failure policy.
        """
        # epochs == -1 runs indefinitely (the passing eval is the terminator
        # under `until`); any positive value is a hard upper bound.
        total_epochs = "∞" if task.epochs < 0 else task.epochs

        # Optional pre-training baseline evaluation (e.g. the untrained
        # policy's rollout). Only on a truly fresh task — a resumed or
        # mid-epoch task already has its baseline. Under `until`, a passing
        # baseline completes the task without training.
        if (
            task.evaluation_interval
            and task.evaluate_at_start
            and task._epoch_count == 0
            and task._iter_count == 0
        ):
            if self._handle_evaluation(task_name, task):
                return
        while task.epochs < 0 or task._epoch_count < task.epochs:
            info(f"Task '{task_name}': epoch {task._epoch_count + 1}/{total_epochs}")
            with create_progress() as progress:
                epoch_bar = progress.add_task(f"[cyan]{task_name}", total=task.iterations)
                if task._iter_count > 0:
                    progress.update(epoch_bar, completed=task._iter_count)

                while task._iter_count < task.iterations:
                    try:
                        step_start = time.perf_counter()
                        task._step()
                        self._record_step_time(time.perf_counter() - step_start)
                        progress.update(epoch_bar, advance=1)
                    except StopIteration:
                        # A finite (map-style) data source signals the end of an
                        # epoch by exhausting — the normal epoch boundary, not an
                        # error. (Streaming sources are bounded by `iterations`.)
                        info(
                            f"Task '{task_name}': data source exhausted after "
                            f"{task._iter_count} iterations — completing epoch "
                            f"{task._epoch_count + 1}/{total_epochs}."
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 — the failure policy decides what a crash means
                        # Crash failure. On `continue` the handler skips the
                        # iteration and returns; every other policy raises.
                        self._handle_crash(task_name, task, exc)
                        continue

                    # Statistical-failure check on the configured cadence,
                    # measured in the task's global_step — cumulative across
                    # epochs (and resumes, for tasks that key it to a group
                    # counter), unlike the per-epoch _iter_count, so intervals
                    # larger than one epoch still fire. A passing eval under
                    # `until` completes the task.
                    if (
                        task.evaluation_interval
                        and task.global_step % task.evaluation_interval == 0
                    ):
                        if self._handle_evaluation(task_name, task):
                            return

                    if task.checkpoint_flag:
                        if self.runtime.groups_can_checkpoint:
                            self.runtime.checkpoint_all()
                        else:
                            debug("Checkpoint deferred — groups have unapplied gradient steps")

            task._increment_epoch()
            if task.epochs < 0 or task._epoch_count < task.epochs:
                self.runtime.reset_datasources()

    # ------------------
    # Failure handling
    # ------------------
    def _handle_crash(self, task_name, task: Task, exc: Exception) -> None:
        """Apply `upon_failure` to a crash (an uncaught Exception in _step()).

        `continue` skips the failing iteration (incrementing the counter, since
        the raising `_step()` did not). `retry` raises `_RestartTask`. `exit` and
        `until` re-raise — a crash is never "by design".
        """
        policy = task.upon_failure
        if policy == "continue":
            warn(f"Task '{task_name}' iter {task._iter_count} crashed ({exc}) — skipping.")
            task._iter_count += 1
            return
        if policy == "retry":
            warn(f"Task '{task_name}' crashed ({exc}) — restarting task.")
            raise _RestartTask() from exc
        error(f"Task '{task_name}' failed at iteration {task._iter_count}: {exc}")
        raise

    def _handle_evaluation(self, task_name, task: Task) -> bool:
        """Run the periodic evaluation and apply the policy on a failed verdict.

        Returns True when the task is complete (a passing eval under `until`),
        else False. A failed verdict under `retry` raises `_RestartTask`; under
        `exit` raises; under `continue`/`until` it logs and keeps training.
        """
        result = task._evaluate_step()
        if result.success:
            if task.upon_failure == "until":
                info(
                    f"Task '{task_name}': evaluation passed at iter {task._iter_count} "
                    f"({result.metrics}) — task complete."
                )
                return True
            return False

        policy = task.upon_failure
        if policy in ("until", "continue"):
            debug(
                f"Task '{task_name}': statistical failure at iter {task._iter_count} "
                f"({result.metrics}) — continuing ({policy})."
            )
            return False
        if policy == "retry":
            warn(f"Task '{task_name}': statistical failure ({result.metrics}) — restarting task.")
            raise _RestartTask()
        error(
            f"Task '{task_name}': statistical failure at iter {task._iter_count} "
            f"({result.metrics}) — exiting."
        )
        raise RuntimeError(f"Task '{task_name}' failed evaluation: {result.metrics}")

    # ------------------
    # Pre-task group snapshot (for `retry`)
    # ------------------
    def _snapshot_groups(self) -> dict:
        """In-memory snapshot of every built group, for a `retry` restart.

        Reuses ``Group.checkpoint_state`` with a full-weights + optimizer policy.
        Only groups already built at task entry are captured; a group first built
        lazily during the task carries no baseline and is left as-is on restart.
        """
        policy = GroupCheckpointPolicy(weights="full", optimizer=True)
        return {
            name: group.checkpoint_state(policy) for name, group in self.runtime._groups.items()
        }

    def _restore_groups(self, snapshot: dict) -> None:
        """Restore group parameters/optimizer/counters from a pre-task snapshot.

        Deliberately does NOT touch global RNG or datasource cursors, so the
        restarted attempt diverges from the one that failed.
        """
        for name, state in snapshot.items():
            group = self.runtime._groups.get(name)
            if group is not None:
                group._load_group_state(state)

    def _record_step_time(self, dt: float) -> None:
        """Accumulate per-step wall-clock and flush throughput on the interval.

        Logs ``perf/steps_per_sec`` and ``perf/sec_per_step`` averaged over the
        window, then resets it — a no-op when throughput logging is disabled.
        """
        tracker = self.runtime.tracker
        if not tracker.log_throughput:
            return
        self._global_step += 1
        self._perf_elapsed += dt
        self._perf_count += 1
        interval = tracker.scalar_interval
        if interval > 0 and self._global_step % interval == 0 and self._perf_elapsed > 0:
            tracker.log_scalar(
                "perf/steps_per_sec", self._perf_count / self._perf_elapsed, self._global_step
            )
            tracker.log_scalar(
                "perf/sec_per_step", self._perf_elapsed / self._perf_count, self._global_step
            )
            self._perf_elapsed = 0.0
            self._perf_count = 0


__all__ = ["Engine"]
