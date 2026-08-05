#!/usr/bin/env bash
# Shared setup for the dev scripts.
#
# The mirrored trees are not a src/ layout — production puts
# third_party/object_search on PYTHONPATH so `prepare`, `inference` and `indexing`
# import as top-level names. Every script needs that same path, plus the repo root
# for `toolbox.*`.
#
# PYTHON can be overridden; it is no longer hardcoded to one developer's conda env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

PYTHON="${PYTHON:-python3}"
export PYTHON

# third_party/object_search is the wemap-vision-object-search submodule, mounted at
# the path the backend itself uses — which is why this PYTHONPATH is unchanged from
# before the split. Fail here rather than three imports deep in a test run.
if [[ ! -d "$REPO_ROOT/third_party/object_search/prepare" ]]; then
  echo "error: the pipeline submodule is not checked out at" >&2
  echo "       $REPO_ROOT/third_party/object_search" >&2
  echo "       run: git submodule update --init" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/object_search${PYTHONPATH:+:$PYTHONPATH}"

# Load .env if present, without clobbering anything already exported.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    key="${line%%=*}"
    key="${key//[[:space:]]/}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key?}=${line#*=}"
    fi
  done < "$REPO_ROOT/.env"
fi
