# Releasing

## Cutting a release

1. Update `CHANGELOG.md`: move `[Unreleased]` entries under a new version heading with
   today's date, and update the link definitions at the bottom.
2. Bump `version` in `pyproject.toml` and `version` / `date-released` in `CITATION.cff`.
   These three must agree.
3. Verify the tree:
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run pytest
   uv run python smoke_test.py
   uv build
   scripts/verify-public-tree.sh
   ```
4. Commit, tag, push:
   ```bash
   git tag -a v0.1.0 -m "Flexilearn v0.1.0"
   git push origin main --tags
   ```
5. Cut a GitHub Release from the tag, with notes drawn from `CHANGELOG.md`.

## The leak gate

`scripts/verify-public-tree.sh` fails if a private-experiment identifier appears in a
tracked file, a path, or **any commit message in history** — the case a working-tree grep
would miss. Run it before pushing anywhere public:

```bash
scripts/verify-public-tree.sh            # this repo
scripts/verify-public-tree.sh /path/to/other-clone
```

Extend its `PATTERN` as private work grows. A false positive costs a minute; a leak is
permanent once pushed.

## Keeping a private fork in sync

This repository is the canonical home of the framework. A private repository holding
experiment branches should treat it as a read-only upstream:

```bash
git remote add upstream https://github.com/DoubleH7/flexilearn.git
git remote set-url --push upstream DISABLED    # hard stop against pushing private work up
git fetch upstream
git merge upstream/main --allow-unrelated-histories   # one-time: the public root is an orphan commit
```

The `--allow-unrelated-histories` flag is needed only for the first merge, because the
public repository begins from a fresh root commit rather than sharing the private
repository's history. Afterwards the two share a merge base and later syncs are ordinary
merges. Framework fixes should be made here and merged **down** into the private
repository, then outward into its experiment branches — never the other direction.

`git remote set-url --push upstream DISABLED` is the important line: it makes
`git push upstream` fail instead of publishing private branches.
