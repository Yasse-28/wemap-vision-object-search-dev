#!/usr/bin/env bash
# Every gate, in order, across both repos.
#
# Exists because the split makes `pytest` two invocations rather than one: the
# submodule owns its 20 mirrored tests and runs them with its own config. Re-coupling
# them into one run would put the annotation_service and object_search_online roots
# back on this repo's sys.path, where their flat `app.py` modules collide by name.
#
# usage: scripts/check-all.sh [path/to/wemap-vision-backend]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$REPO_ROOT"
failed=()

run() {
  local label="$1"; shift
  echo
  echo "== $label"
  if "$@"; then
    echo "   ok"
  else
    echo "   FAILED: $label" >&2
    failed+=("$label")
  fi
}

run "mirror has not drifted"   "$REPO_ROOT/scripts/check-mirror.sh" "$@"
run "ruff"                     ruff check .
run "black"                    black --check .
run "mypy"                     "$REPO_ROOT/scripts/check-types.sh"
run "pytest (toolbox)"         "$PYTHON" -m pytest -q
run "pytest (mirror)"          bash -c "cd '$REPO_ROOT/third_party/object_search' && '$PYTHON' -m pytest -q"
run "tsc"                      bash -c "cd '$REPO_ROOT/toolbox' && npm run --silent type-check"
run "node --test"              bash -c "cd '$REPO_ROOT/toolbox' && npm test --silent -w backend"

echo
if (( ${#failed[@]} )); then
  printf 'FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "All gates passed."
