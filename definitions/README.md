# definitions/

The framework's extension point — intentionally empty.

This is where **your** components live. Drop a Python file into one of the
subdirectories below, register its class with a decorator from
`flexilearn/registry.py`, and reference the key from your config:

| Component | Decorator | Location |
|---|---|---|
| Model | `@register_model("key")` | `definitions/models/` |
| Task | `@register_task("key")` | `definitions/tasks/` |
| Dataset / env factory | `@register_dataset("key")` | `definitions/data/` |
| Data prep stage | `@register_dataprep("key")` | `definitions/data/` |
| Loss | `@register_loss("key")` | `definitions/losses/` |

`discover_definitions()` imports every `*.py` under this directory at startup, so
dropping a file in and referencing its key from a config is all it takes — there
is no import list to maintain. Files whose names start with `_` are skipped.

The subdirectories are a convention, not a requirement: discovery walks the whole
tree, so organize it however suits your project.

## Where this directory is looked for

In order, first hit wins:

1. the `--definitions PATH` flag,
2. `$FLEXILEARN_DEFINITIONS`,
3. `./definitions` relative to the working directory,
4. `definitions/` beside the installed `flexilearn` package.

So from a source checkout, `uv run flexilearn config.yml` finds this directory
automatically. Using flexilearn as an installed dependency, point it at your own
project's directory with the flag or the environment variable.

## A worked example

[`examples/minimal/`](../examples/minimal/) is a complete, runnable version of
exactly this layout — a registered dataset and task, plus the config that wires
them together:

```bash
uv run flexilearn examples/minimal/config.yml --definitions examples/minimal/definitions
```
