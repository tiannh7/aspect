#!/usr/bin/env bash

set -euo pipefail

patch_file="doc/review/degree_one_com_constraint_followup_f7_v2.patch"
required_base="f7fdd6eac0cbdbd498e2485c1ccc16b5ac885a40"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: run this script from an ASPECT git worktree" >&2
  exit 1
fi

if [[ ! -f "$patch_file" ]]; then
  echo "error: missing $patch_file" >&2
  exit 1
fi

if ! git merge-base --is-ancestor "$required_base" HEAD; then
  echo "error: HEAD does not contain required base $required_base" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree or index contains changes; refusing to mix patches" >&2
  git status --short
  exit 1
fi

echo "Checking $patch_file against $(git rev-parse --short HEAD)..."
git apply --check --3way "$patch_file"

echo "Applying $patch_file..."
git apply --3way "$patch_file"

echo
echo "Patch applied but not committed. Review with:"
echo "  git diff --check"
echo "  git diff --stat"
echo "  git diff"
echo
echo "Then build and run the focused tests before committing."
