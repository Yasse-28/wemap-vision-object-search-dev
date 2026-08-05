#!/usr/bin/env bash
# Type-check our own code: toolbox/.
#
# The mirrored trees (third_party/object_search, services/object_search_online) are
# deliberately NOT checked. They are byte-for-byte copies of wemap-vision-backend, so
# an error there cannot be fixed here without breaking the mirror — checking them
# would only produce permanent suppressions. Their typing is the backend's job.
#
# They stay on MYPYPATH so our annotations resolve against the real modules
# (`from indexing.grid import filter_by_distance`, `from prepare.convention import …`).
#
# A single pass over the whole repo would also fail outright: the mirror has flat
# script directories whose modules collide by name — `annotation_service/app.py` and
# `services/object_search_online/app.py` are both top-level `app`. Production never
# sees the collision because each piece runs with its own directory as CWD.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MYPY="${MYPY:-mypy}"

cd "$REPO_ROOT"

# Empty PYTHONPATH so lib.sh's export cannot reintroduce the name collisions.
MYPYPATH="$REPO_ROOT/third_party/object_search" PYTHONPATH="" \
  "$MYPY" toolbox
