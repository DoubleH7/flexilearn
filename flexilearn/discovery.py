# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
"""Automatic discovery of component definitions.

Every module under a ``definitions/`` directory registers components (models,
tasks, datasets, losses, data-prep stages) with the global registries in
``registry.py`` via decorators that run at import time. Those modules must
therefore be imported before the engine runs.

``discover_definitions`` walks that tree and imports every module, so adding a
new file there is enough — there is no manual import list to keep in sync (this
previously lived, by hand, in ``entry.py``).

The directory is resolved in this order, first hit wins:

1. the ``root`` argument (``flexilearn --definitions PATH``),
2. ``$FLEXILEARN_DEFINITIONS``,
3. ``./definitions`` relative to the current working directory,
4. ``definitions/`` beside the installed ``flexilearn`` package.

Rule 3 is what makes an installed wheel usable: the package may live in
site-packages while the experiment's components sit in the working directory.
"""

import importlib
import os
import sys
from pathlib import Path

from .monitor import debug, warn

# The directory holding the flexilearn package — in a source checkout this is
# the repo root, so `definitions/` beside it is the final fallback.
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent

ENV_VAR = "FLEXILEARN_DEFINITIONS"


def resolve_definitions_root(root=None):
    """Return the definitions directory to import from, or None if there is none.

    An explicit ``root`` that does not exist is an error — the caller asked for
    a specific directory by name and typos should not silently degrade into
    "no components registered". The implicit candidates are allowed to miss.
    """
    if root is not None:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"definitions directory not found: {path}")
        return path

    candidates = []
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        candidates.append(Path(from_env).expanduser())
    candidates.append(Path.cwd() / "definitions")
    candidates.append(_PACKAGE_PARENT / "definitions")

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    return None


def discover_definitions(root=None) -> list[str]:
    """Import every module under ``definitions/`` so its decorators register.

    Returns the dotted names of the imported modules, in import order. Safe to
    call more than once: already-imported modules are served from
    ``sys.modules``, and the registries are plain dict assignments.

    A missing directory is a warning, not an error — flexilearn is usable as a
    library whose components are registered directly in Python. A config that
    then names an unregistered key fails with a clear registry error.

    Import errors inside a definition module are deliberately left to propagate
    — a broken module should fail loudly rather than silently drop components.
    """
    definitions = resolve_definitions_root(root)
    if definitions is None:
        warn(
            f"No definitions/ directory found (looked at ${ENV_VAR}, "
            f"./definitions, and {_PACKAGE_PARENT / 'definitions'}). "
            f"No components were auto-registered."
        )
        return []

    # definitions/ is imported as a namespace package rooted at its parent, so
    # modules inside it can import each other as `definitions.sub.module`.
    search_root = definitions.parent
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

    imported: list[str] = []
    for path in sorted(definitions.rglob("*.py")):
        if path.name.startswith("_"):  # __init__.py and other dunder files
            continue
        dotted = ".".join(path.relative_to(search_root).with_suffix("").parts)
        importlib.import_module(dotted)
        imported.append(dotted)

    debug(f"Discovered {len(imported)} definition module(s) under {definitions}")
    return imported


__all__ = ["discover_definitions", "resolve_definitions_root"]
