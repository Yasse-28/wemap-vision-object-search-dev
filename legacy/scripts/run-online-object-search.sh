#! /bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

PYTHON=/home/yacine/anaconda3/envs/wemap-vision/bin/python

$PYTHON -m pipeline.online.app \
    --config_file_path $VPS_DATA_DIR/config.json \
    --host 0.0.0.0 \
    --port 45678 \
    --cors