#!/usr/bin/env bash
# Start the bricks service: the dev-only stand-in for Django's object-search API.
#
# Serves POST /{map_id}/object-search/localize — enrichment + clustering + ranking
# on top of the mirrored service's ANN hits. This is what the toolbox UI and the
# HTTP benchmark talk to.
#
# Needs the mirrored online service running first (scripts/run-online-service.sh).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONFIG="${1:-${OBJECT_SEARCH_WORKBENCH_CONFIG:-}}"
if [[ -z "$CONFIG" ]]; then
  echo "usage: $(basename "$0") <config.json>" >&2
  echo "       (or set OBJECT_SEARCH_WORKBENCH_CONFIG)" >&2
  exit 2
fi

exec "$PYTHON" -m toolbox.bricks.service \
  --config "$CONFIG" \
  --host "${BRICKS_HOST:-127.0.0.1}" \
  --port "${BRICKS_PORT:-45679}" \
  --ann-base-url "${OBJECT_SEARCH_ANN_URL:-http://127.0.0.1:8000}" \
  --cors
