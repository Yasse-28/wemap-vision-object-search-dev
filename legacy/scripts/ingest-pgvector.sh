#!/bin/bash
# Ingest object-search.db embeddings into pgvector.
# Requires PGVECTOR_PASSWORD (and optionally PGVECTOR_HOST/PORT) in the environment.
# Usage: ./scripts/ingest-pgvector.sh
# Example:
#   $PYTHON -m pipeline.offline.ingest_pgvector \
#     --map_path "$VPS_DATA_DIR/maps/sncf-paris-gare-du-nord" \
#     --map-id sncf-paris-gare-du-nord \
#     --db-location local \
#     --unlogged

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

PYTHON=/home/yacine/anaconda3/envs/wemap-vision/bin/python

$PYTHON -m pipeline.offline.ingest_pgvector \
    --map_path "$VPS_DATA_DIR/maps/bbhotel-choisy" \
    --db-location local \
    --unlogged
