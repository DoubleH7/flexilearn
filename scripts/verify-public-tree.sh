#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
#
# Leak gate for the public repository.
#
# Run this against a git repository (default: the current one) before pushing it
# anywhere public. It fails if any private-experiment identifier appears in a
# tracked file, a path, or a commit message — including in history, which is the
# case a working-tree grep alone would miss.
#
#   scripts/verify-public-tree.sh [path-to-repo]
#
# Exit status is 0 when the tree is clean, 1 otherwise.

set -euo pipefail

REPO="${1:-.}"
cd "$REPO"

# Identifiers that must never appear in the public repository. Extend this list
# as private work grows; a false positive is cheap, a leak is not.
#
# Note: the handle "TheDoubleH" is deliberately NOT listed — it is public, and
# appears in NOTICE and CITATION.cff by design. What must not leak is the
# personal email (thedoubleh<n>@...) and local paths (/home/...), both covered.
PATTERN='qwen|molora|more_vla|oxe|pusht|reach_|barlow|exp/vla|exp/more|exp/ssl|config_(vla|qwen|pusht|rl|more|bench|probe)|/home/|thedoubleh[0-9]*@'

fail=0
report() { printf '  %s\n' "$1"; fail=1; }

echo "Leak gate: $(cd "$REPO" && pwd)"

echo "[1/5] commit messages and paths across all history"
if hits=$(git log --all --name-only --format='%s%n%b' | grep -inE "$PATTERN" | head -20) && [ -n "$hits" ]; then
    report "FAIL — private identifiers found in history:"
    printf '    %s\n' "$hits"
else
    echo "  ok"
fi

echo "[2/5] tracked file contents"
if hits=$(git grep -inE "$PATTERN" -- . ':(exclude)scripts/verify-public-tree.sh' | head -20) && [ -n "$hits" ]; then
    report "FAIL — private identifiers found in tracked files:"
    printf '    %s\n' "$hits"
else
    echo "  ok"
fi

echo "[3/5] secrets"
if hits=$(git grep -inE 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY' -- . \
          | grep -viE '\.env\.example|x{10,}' | head -10) && [ -n "$hits" ]; then
    report "FAIL — possible secret:"
    printf '    %s\n' "$hits"
else
    echo "  ok"
fi

echo "[4/5] files that must not be tracked"
step_fail=0
for f in .env mlflow.db; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        report "FAIL — $f is tracked"; step_fail=1
    fi
done
if git ls-files | grep -qE '^(outputs|mlruns|logs|data)/'; then
    report "FAIL — runtime artifacts are tracked"; step_fail=1
fi
[ "$step_fail" -eq 0 ] && echo "  ok"

echo "[5/5] files that must be present"
step_fail=0
for f in LICENSE NOTICE README.md CITATION.cff CONTRIBUTING.md SECURITY.md pyproject.toml; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { report "FAIL — $f is missing"; step_fail=1; }
done
[ "$step_fail" -eq 0 ] && echo "  ok"

echo
if [ "$fail" -eq 0 ]; then
    echo "PASS — $(git rev-list --count --all) commit(s), $(git ls-files | wc -l) tracked files."
else
    echo "FAILED — do not push."
fi
exit "$fail"
