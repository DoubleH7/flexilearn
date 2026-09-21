# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Package entry point invoked by `uv run flexilearn`."""

import argparse

from dotenv import load_dotenv

from flexilearn import Engine
from flexilearn.config_loader import parse_config
from flexilearn.discovery import discover_definitions


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="flexilearn",
        description="Run a flexilearn training experiment from a config file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Path to the experiment config YAML (default: config.yml). "
        "e.g. `flexilearn examples/minimal/config.yml`.",
    )
    parser.add_argument(
        "--definitions",
        default=None,
        metavar="PATH",
        help="Directory of component definitions to import before running. "
        "Defaults to $FLEXILEARN_DEFINITIONS, then ./definitions.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Load a repo-root .env (if present) into the process environment before
    # anything else runs, so secrets like HF_TOKEN / HUGGING_FACE_HUB_TOKEN are
    # picked up by huggingface_hub on the first model/tokenizer download.
    # `.env` is gitignored; copy `.env.example` and fill in your token.
    load_dotenv()

    # Import every definitions/ module so component decorators register.
    discover_definitions(args.definitions)
    config = parse_config(args.config)
    engine = Engine(config)
    engine.execute()


if __name__ == "__main__":
    main()
