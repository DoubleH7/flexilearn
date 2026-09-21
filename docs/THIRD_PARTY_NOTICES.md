# Third-party notices

Flexilearn is licensed under the Apache License 2.0 (see [`LICENSE`](../LICENSE)).

**Flexilearn does not bundle, vendor, or redistribute any third-party source
code.** Everything in `flexilearn/` is original work. The packages below are
ordinary runtime dependencies: they are resolved and installed by `uv`/`pip`
from PyPI, remain under their own licenses, and are listed here for the
convenience of anyone auditing the dependency tree.

License identifiers were read from the installed package metadata
(`<package>-<version>.dist-info/METADATA`) rather than transcribed by hand.

## Runtime dependencies

| Package | License |
|---|---|
| [torch](https://pypi.org/project/torch/) | BSD-3-Clause |
| [numpy](https://pypi.org/project/numpy/) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [rich](https://pypi.org/project/rich/) | MIT |
| [pydantic](https://pypi.org/project/pydantic/) | MIT |
| [mlflow](https://pypi.org/project/mlflow/) | Apache-2.0 |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | BSD-3-Clause |
| [PyYAML](https://pypi.org/project/PyYAML/) | MIT |
| [psutil](https://pypi.org/project/psutil/) | BSD-3-Clause |

## Optional dependencies

| Package | Extra | License |
|---|---|---|
| [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) | `gpu` | BSD |

## Development dependencies

| Package | License |
|---|---|
| [pytest](https://pypi.org/project/pytest/) | MIT |
| [ruff](https://pypi.org/project/ruff/) | MIT |

Each dependency pulls in its own transitive tree; consult the resolved
`uv.lock` or run `uv pip list` for the complete set installed in a given
environment.
