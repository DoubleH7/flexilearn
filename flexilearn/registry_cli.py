# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""``flexilearn-registry`` — manage MLflow registered models from the CLI.

This is an *operational* tool, separate from the training loop: it lists
registered models and promotes versions through their lifecycle. It is core
(not under ``definitions/``), so importing ``mlflow`` here is fine. (Named
``registry_cli`` to avoid clashing with ``flexilearn.registry``, the component
registry.)

Promotion uses **aliases** by default (``set_registered_model_alias``) because
the classic *stages* API (``transition_model_version_stage``) is deprecated in
modern MLflow; pass ``--legacy-stage`` to use stages on an older server.

The tracking/registry URI is resolved from (in order): explicit
``--tracking-uri`` / ``--registry-uri``, a ``--config`` file's ``logging:``
block, or MLflow's own default (``MLFLOW_TRACKING_URI`` env / ``./mlruns``).

Examples::

    flexilearn-registry list
    flexilearn-registry list --config config.yml
    flexilearn-registry promote my_policy 3 production
    flexilearn-registry promote my_policy 3 Production --legacy-stage
    flexilearn-registry promote-best my_policy --metric eval/mean_reward
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def _resolve_uris(args) -> None:
    """Point MLflow at the right tracking + registry backends before any call."""
    tracking_uri = args.tracking_uri
    registry_uri = args.registry_uri
    if args.config and (tracking_uri is None or registry_uri is None):
        # Reuse the experiment's own logging config so the CLI and the trainer
        # always agree on which backend holds the runs/registry.
        from .config_loader import parse_config

        log_cfg = parse_config(args.config).logging
        tracking_uri = tracking_uri or log_cfg.tracking_uri
        registry_uri = registry_uri or log_cfg.registry_uri or log_cfg.tracking_uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)


def _cmd_list(args) -> None:
    client = MlflowClient()
    models = client.search_registered_models()
    if not models:
        print("No registered models found.")
        return
    for model in models:
        # aliases is a dict {alias: version} in MLflow 3.x, a list of objects on
        # older releases; normalize to {alias: str(version)} either way.
        raw_aliases = getattr(model, "aliases", None) or {}
        if isinstance(raw_aliases, dict):
            aliases = {a: str(v) for a, v in raw_aliases.items()}
        else:
            aliases = {a.alias: str(a.version) for a in raw_aliases}
        print(f"\n{model.name}")
        versions = client.search_model_versions(f"name = '{model.name}'")
        for mv in sorted(versions, key=lambda v: int(v.version)):
            alias_str = ", ".join(f"@{a}" for a, v in aliases.items() if v == str(mv.version))
            stage = mv.current_stage if mv.current_stage and mv.current_stage != "None" else ""
            tail = "  ".join(x for x in (stage, alias_str) if x)
            print(f"  v{mv.version:<4} run={mv.run_id or '-':<34} {tail}")


def _cmd_promote(args) -> None:
    client = MlflowClient()
    if args.legacy_stage:
        client.transition_model_version_stage(args.name, args.version, args.stage)
        print(f"{args.name} v{args.version} → stage '{args.stage}' (legacy)")
    else:
        alias = args.stage.lower()
        client.set_registered_model_alias(args.name, alias, args.version)
        print(f"{args.name} v{args.version} → alias '@{alias}'")


def _cmd_promote_best(args) -> None:
    client = MlflowClient()
    experiment = args.experiment
    if experiment is None and args.config:
        from .config_loader import parse_config

        cfg = parse_config(args.config)
        experiment = cfg.logging.experiment or cfg.experiment.name
    if experiment is None:
        raise SystemExit("promote-best needs --experiment or --config to locate runs.")

    order = "ASC" if args.mode == "min" else "DESC"
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        order_by=[f"metrics.`{args.metric}` {order}"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No runs with metric '{args.metric}' in experiment '{experiment}'.")
    best_run_id = runs.iloc[0]["run_id"]
    best_value = runs.iloc[0].get(f"metrics.{args.metric}")

    versions = [
        mv
        for mv in client.search_model_versions(f"name = '{args.name}'")
        if mv.run_id == best_run_id
    ]
    if not versions:
        raise SystemExit(
            f"Best run {best_run_id[:8]} has no version registered under '{args.name}'."
        )
    best_version = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias(args.name, args.alias, best_version)
    print(
        f"{args.name} v{best_version} → alias '@{args.alias}' "
        f"(best {args.metric}={best_value} from run {best_run_id[:8]})"
    )


def _add_uri_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracking-uri", default=None, help="MLflow tracking URI override.")
    p.add_argument("--registry-uri", default=None, help="MLflow registry URI override.")
    p.add_argument("--config", default=None, help="Read URIs/experiment from a flexilearn config.")


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flexilearn-registry", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered models and versions.")
    _add_uri_args(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_promote = sub.add_parser("promote", help="Promote a version via alias (or legacy stage).")
    p_promote.add_argument("name")
    p_promote.add_argument("version")
    p_promote.add_argument("stage", help="Alias/stage, e.g. production, staging, archived.")
    p_promote.add_argument(
        "--legacy-stage",
        action="store_true",
        help="Use the deprecated transition_model_version_stage API instead of an alias.",
    )
    _add_uri_args(p_promote)
    p_promote.set_defaults(func=_cmd_promote)

    p_best = sub.add_parser("promote-best", help="Alias the version from the best run by a metric.")
    p_best.add_argument("name")
    p_best.add_argument(
        "--metric", required=True, help="Metric to rank runs by, e.g. eval/mean_reward."
    )
    p_best.add_argument("--mode", choices=("max", "min"), default="max", help="Optimize direction.")
    p_best.add_argument(
        "--alias", default="production", help="Alias to assign (default: production)."
    )
    p_best.add_argument("--experiment", default=None, help="Experiment name to search.")
    _add_uri_args(p_best)
    p_best.set_defaults(func=_cmd_promote_best)

    args = parser.parse_args(argv)
    _resolve_uris(args)
    args.func(args)


if __name__ == "__main__":
    main()
