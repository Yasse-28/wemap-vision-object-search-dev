#!/usr/bin/env bash
# Build an object-search index for one map, end to end.
#
#   prepare  →  prepare_postprocess  →  ingest  →  partial HNSW
#
# Replaces the standalone `build_index.py` (now in legacy/): detection, cutouts and
# embedding are the mirrored production `prepare` job; the 3D lifting and pgvector
# ingest are the ported bricks.
#
# The postprocess step is NOT optional — prepare emits no `depth` column, and
# without it every object_position is NULL and localize silently returns nothing.
#
# The venue and the georef id both come from the map's manifest
# ({map_id}_{version}_{date}_{time}.json). Neither is a flag: overriding the venue
# would index a different candidate set than production, and overriding the georef
# id would index under a key the online service never queries — zero hits, no error.
#
# On a small GPU, cutout extraction is the step that runs out of memory. Either
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, or pass --cutout-batch 4;
# both are output-neutral. See toolbox/bricks/vendored/proposal_cutouts.py.
#
# usage: scripts/build-index.sh <map_path> [--cutout-batch 4]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <map_path> [--cutout-batch <n>]" >&2
  exit 2
fi

MAP_PATH="$(cd "$1" && pwd)"; shift
# Empty means "keep the default".
CUTOUT_BATCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cutout-batch) CUTOUT_BATCH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OUTPUT_DIR="$MAP_PATH/object-search"
# The map directory mirrors the S3 layout: images/ beside depths/.
if [[ ! -d "$MAP_PATH/images" ]]; then
  echo "error: no ERP images directory at '$MAP_PATH/images'." >&2
  exit 1
fi
# Keyframe ids, poses, venue and georef id all come from the manifest.
shopt -s nullglob
manifests=("$MAP_PATH"/*_*_????????_??????.json)
shopt -u nullglob
if (( ${#manifests[@]} == 0 )); then
  echo "error: '$MAP_PATH' has no v2 manifest" \
       "('{map_id}_{version}_{date}_{time}.json')." >&2
  exit 1
fi

# Deliberately NOT `python -m prepare`: that CLI numbers keyframes positionally
# (`enumerate`), so ingest would attach candidates to the wrong poses, and it never
# writes thumbnails. prepare_runner resolves real keyframe ids from the map's pose
# source, exactly as the Django command does in production. See its module docstring.
echo "== 1/3 prepare (detect + cutouts + embed) =="
prepare_args=()
[[ -n "$CUTOUT_BATCH" ]] && prepare_args+=(--cutout-batch "$CUTOUT_BATCH")
"$PYTHON" -m toolbox.bricks.prepare_runner "$MAP_PATH" \
  --output-dir "$OUTPUT_DIR" "${prepare_args[@]}"

echo "== 2/3 postprocess (thumbnail_key + depth) =="
"$PYTHON" -m toolbox.bricks.prepare_postprocess "$MAP_PATH" "$OUTPUT_DIR/metadata.parquet"

echo "== 3/3 ingest (3D lift + COPY + HNSW) =="
"$PYTHON" -m toolbox.bricks.ingest_cli "$MAP_PATH"

echo "Done. Serve it with scripts/run-online-service.sh + scripts/run-bricks-service.sh."
