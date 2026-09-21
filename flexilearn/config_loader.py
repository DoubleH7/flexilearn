# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad

from pathlib import Path

import yaml

from .config_models import Config


def parse_config(config_path: str | Path = "config.yml") -> Config:
    """Parse a YAML configuration file and return a validated Config object.

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If config doesn't match schema
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
