#!/usr/bin/env bash
# Start the MIRRORED online service (embed + HNSW), the production GPU service.
#
# It answers POST /object-search/by-text | by-image with a flat [{id, similarity}]
# list — no coordinates, no clusters. Geometry and clustering live in the bricks
# service (scripts/run-bricks-service.sh), which is what Django does upstream.
#
# Must run with services/object_search_online as CWD: app.py imports bare
# `service_state` and `v1_legacy`, exactly as the production container does.
# Do not try `python -m object_search_online.app` — there is no package there.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PORT="${OBJECT_SEARCH_ONLINE_PORT:-8000}"
HOST="${OBJECT_SEARCH_ONLINE_HOST:-127.0.0.1}"

if [[ "${ENVIRONMENT_NAME:-}" != "onprem" ]]; then
  echo "warning: ENVIRONMENT_NAME is '${ENVIRONMENT_NAME:-<unset>}', not 'onprem'." >&2
  echo "         The service will try AWS Secrets Manager instead of DATABASE_*." >&2
fi

# The service tree lives in the submodule. CWD, not just PYTHONPATH: app.py imports
# bare `service_state` and `v1_legacy`, exactly as the production container does.
cd "$REPO_ROOT/third_party/object_search/services/object_search_online"
exec "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT"
