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
