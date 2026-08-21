#!/usr/bin/env bash
# Dump prod object-search prepare outputs (metadata.parquet + embeddings.npy) for
# one map, from prod S3, so a local index can be built without the GPU hours.
#
#   usage: scripts/dump-object-search.sh [flags] <map_path | map_id> [dest_dir]
#
#   map_path: the map directory, or its pivot-format export JSON directly. Reads
#             `map.id`, `map.uuid`, `map.georef_version` and `videos[]` from it, and
#             dumps into <map_dir>/object-search/.
#   map_id:   the integer Map.id — dumps to ./object_search_dump/<map_id>. Degraded
#             mode: without the pivot file neither the version nor the capture list
#             is known, so trajectory.json cannot be fetched and every capture S3
#             holds is downloaded, including those of other map versions.
#   dest_dir: overrides the destination in both cases.
#
#   --thumbnails:      also sync the cutout thumbnails (BIG — ~250 MB per capture).
#   --thumbnails-only: skip the prepare outputs, fetch only the thumbnails.
#   --force:           re-download prepare outputs that are already on disk.
#   --dry-run:         list what would be fetched, download nothing.
#
# Prepare outputs already present are SKIPPED unless --force, because a fresh
# metadata.parquet silently undoes `prod_dump_remap` — the ids revert to prod's and
# nothing downstream complains.
#
# ## Why the pivot file decides which captures to fetch
#
# S3 holds every capture ever prepared for the map, across georef versions. Only the
# captures the *current* version was built from have keyframes in the manifest, so
# only those can be positioned here: the others remap to -1 and ingest drops them —
# silently, after the download. The pivot export now carries that list (`videos[]`),
# so the loop iterates it instead of the S3 listing, and reports both gaps by name:
# a listed capture with no prepare outputs in prod, and a capture in S3 that this
# version does not contain.
#
# Deliberately parses the pivot JSON with stdlib python3 rather than
# `toolbox.bricks.map_manifest`: this script must run before anything is installed,
# and the four fields it needs are top-level. The reader is the authority for
# everything downstream — see its `_map_version` / `_video_capture_ids`.
#
# Requires active AWS creds (`aws sts get-caller-identity`).
#
# After dumping, both steps are mandatory before ingest:
#   $PYTHON -m toolbox.bricks.prod_dump_remap <map_path>
#   $PYTHON -m toolbox.bricks.prod_dump_merge <map_path>
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BUCKET="wemap-vision-storage-prod"
PUBLIC_BUCKET="wemap-vision-storage-public-prod"

WANT_THUMBNAILS=0
WANT_OUTPUTS=1
FORCE=0
DRY_RUN=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --thumbnails) WANT_THUMBNAILS=1 ;;
    --thumbnails-only) WANT_THUMBNAILS=1; WANT_OUTPUTS=0 ;;
    --force) FORCE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) ARGS+=("$arg") ;;
  esac
done
set -- ${ARGS+"${ARGS[@]}"}

INPUT="${1:?usage: dump-object-search.sh [flags] <map_path | map_id> [dest_dir]}"

# `aws s3 cp`, or an echo of it under --dry-run.
run_aws() {
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "DRY-RUN: aws $*"
  else
    aws "$@"
  fi
}

MAP_DIR=""
MAP_UUID=""
MAP_VERSION=""
MANIFEST_VIDEOS=""
if [[ "$INPUT" =~ ^[0-9]+$ ]]; then
  MAP_ID="$INPUT"
  echo "WARNING: called with a map_id, so videos[] and georef_version are unknown." >&2
  echo "         Every capture in S3 will be fetched and trajectory.json skipped —" >&2
  echo "         pass the map directory instead to dump only this version." >&2
else
  JSON_FILE="$INPUT"
  if [[ -d "$INPUT" ]]; then
    MAP_DIR="$INPUT"
    JSON_FILE=$(find "$INPUT" -maxdepth 1 -name '*.json' -print0 | xargs -0 ls -t | head -n1)
    if [[ -z "$JSON_FILE" ]]; then
      echo "No pivot JSON found in directory: $INPUT" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$JSON_FILE" ]]; then
    echo "Not a valid map_id nor an existing path: $INPUT" >&2
    exit 1
  fi
  [[ -z "$MAP_DIR" ]] && MAP_DIR=$(dirname "$JSON_FILE")

  # One read, four fields. `georef_version` is the exported value and wins over the
  # {map}_{version}_{date}_{time}.json filename, which a rename can make lie; the
  # filename stays the fallback for exports predating the field. Same rule as
  # map_manifest._map_version, and a mismatch is reported the same way.
  # "|"-separated, not whitespace: an empty version would otherwise shift videos[]
  # into MAP_VERSION and the whole run would target the wrong S3 paths.
  IFS="|" read -r MAP_ID MAP_UUID MAP_VERSION MANIFEST_VIDEOS < <(python3 - "$JSON_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
block = data.get("map") or {}

match = re.match(r"^.+_(?P<version>\d+)_\d{8}_\d{6}\.json$", path.name)
from_name = int(match.group("version")) if match else None
raw = block.get("georef_version")
version = from_name if raw is None else int(raw)
if raw is not None and from_name is not None and from_name != int(raw):
    print(
        f"WARNING: filename says version {from_name}, map.georef_version says "
        f"{int(raw)} — trusting the field.",
        file=sys.stderr,
    )

videos = sorted({int(v) for v in (data.get("videos") or [])})
print(
    "|".join(
        (
            str(block.get("id", "")),
            str(block.get("uuid", "")),
            "" if version is None else str(version),
            ",".join(str(v) for v in videos),
        )
    )
)
PY
  )
  if [[ -z "$MAP_ID" ]]; then
    echo "'${JSON_FILE}' has no map.id — not a pivot-format export." >&2
    exit 1
  fi
  echo "Resolved map_id=${MAP_ID} uuid=${MAP_UUID} version=${MAP_VERSION:-?} from ${JSON_FILE}"
fi

# Default dest: next to the pivot export when a map_path was given, else
# ./object_search_dump/<map_id>. An explicit dest_dir always wins.
if [[ -n "${2:-}" ]]; then
  DEST="$2"
elif [[ -n "$MAP_DIR" ]]; then
  DEST="${MAP_DIR}/object-search"
else
  DEST="./object_search_dump/${MAP_ID}"
fi

mkdir -p "$DEST"

# A merged parquet at the root of the destination hides everything fetched below it:
# `discover_capture_dirs` returns the root alone as soon as it holds a
# metadata.parquet, so remap and ingest would silently keep reading the old merged
# file and never see the new captures. Refuse instead of dumping into that.
# --thumbnails-only writes no parquet, so it is exempt.
if [[ "$WANT_OUTPUTS" = "1" && -f "${DEST}/metadata.parquet" ]]; then
  echo "error: '${DEST}/metadata.parquet' exists — that is prod_dump_merge's output," >&2
  echo "       and discover_capture_dirs stops at it, so per-capture downloads placed" >&2
  echo "       beside it would be invisible to remap and ingest." >&2
  echo "       Move it aside first (e.g. mv '${DEST}' '${DEST}.merged-<date>')," >&2
  echo "       then re-run; prod_dump_merge rebuilds it from the new captures." >&2
  exit 1
fi

echo "Listing video_capture ids for map ${MAP_ID} in s3://${BUCKET}/${MAP_ID}/ ..."
S3_CAPTURE_IDS=$(aws s3 ls "s3://${BUCKET}/${MAP_ID}/" | awk '{print $2}' | tr -d '/')

if [[ -z "$S3_CAPTURE_IDS" ]]; then
  echo "No video_capture prefixes found under s3://${BUCKET}/${MAP_ID}/ — check the map id." >&2
  exit 1
fi

# The manifest's videos[] is the authority when we have it: iterate it, not the
# listing. Captures S3 holds that this version does not contain are named and
# skipped — fetching them costs bandwidth and buys nothing, since none of their
# keyframes is in the manifest.
if [[ -n "$MANIFEST_VIDEOS" ]]; then
  IFS=',' read -r -a WANTED <<< "$MANIFEST_VIDEOS"
  EXTRANEOUS=()
  for vc_id in $S3_CAPTURE_IDS; do
    if [[ ! ",${MANIFEST_VIDEOS}," == *",${vc_id},"* ]]; then
      EXTRANEOUS+=("$vc_id")
    fi
  done
  echo "Manifest version ${MAP_VERSION} lists ${#WANTED[@]} video_capture(s)."
  if (( ${#EXTRANEOUS[@]} )); then
    echo "Not in this map version, skipped: ${EXTRANEOUS[*]}"
  fi
else
  WANTED=($S3_CAPTURE_IDS)
fi

DUMPED_IDS=()
MISSING_IDS=()
for vc_id in ${WANTED+"${WANTED[@]}"}; do
  key_prefix="${MAP_ID}/${vc_id}/object_search"
  if ! aws s3 ls "s3://${BUCKET}/${key_prefix}/metadata.parquet" >/dev/null 2>&1; then
    MISSING_IDS+=("$vc_id")
    continue
  fi
  DUMPED_IDS+=("$vc_id")
  if [[ "$WANT_OUTPUTS" = "0" ]]; then
    continue
  fi
  if [[ -f "${DEST}/${vc_id}/metadata.parquet" && "$FORCE" = "0" ]]; then
    echo "video_capture ${vc_id}: already on disk, skipping (--force to re-download," \
         "which reverts any prod_dump_remap)"
    continue
  fi
  echo "video_capture ${vc_id}: downloading metadata.parquet + embeddings.npy"
  [[ "$DRY_RUN" = "1" ]] || mkdir -p "${DEST}/${vc_id}"
  run_aws s3 cp "s3://${BUCKET}/${key_prefix}/metadata.parquet" "${DEST}/${vc_id}/metadata.parquet"
  run_aws s3 cp "s3://${BUCKET}/${key_prefix}/embeddings.npy" "${DEST}/${vc_id}/embeddings.npy"
done

# Coverage, stated up front rather than left to surface as a hole in the index:
# prepare has not run in prod for these captures, so their part of the map has no
# candidates at all — which looks exactly like "the model found nothing there".
if (( ${#MISSING_IDS[@]} )); then
  echo "No object_search outputs in prod (prepare not run / failed): ${MISSING_IDS[*]}"
fi
echo "Coverage: ${#DUMPED_IDS[@]}/${#WANTED[@]} video_capture(s) of this map version have outputs."

# --- trajectory.json: the prod keyframe ids, needed to remap before ingest -----
TRAJECTORY="${DEST}/trajectory.json"
if [[ -z "$MAP_UUID" ]]; then
  echo "WARNING: the map uuid is unknown — trajectory.json was NOT downloaded and" >&2
  echo "         the dump cannot be ingested as is. Re-run with the map directory." >&2
elif [[ -z "$MAP_VERSION" ]]; then
  echo "WARNING: neither map.georef_version nor the filename gives a version, so" >&2
  echo "         the 360-viewer path is unknown. Download the matching" >&2
  echo "         versions/<n>/360-viewer/trajectory.json by hand." >&2
elif [[ -f "$TRAJECTORY" && "$FORCE" = "0" ]]; then
  echo "trajectory.json already on disk, keeping it"
else
  traj_key="${MAP_UUID}/versions/${MAP_VERSION}/360-viewer/trajectory.json"
  echo "Downloading trajectory.json (map version ${MAP_VERSION}) — the keyframe id map"
  run_aws s3 cp "s3://${PUBLIC_BUCKET}/${traj_key}" "$TRAJECTORY"
fi

# --- thumbnails: opt-in, one prefix per capture, keyed exactly as thumbnail_key --
if [[ "$WANT_THUMBNAILS" = "1" ]]; then
  if [[ ! -f "$TRAJECTORY" ]]; then
    echo "WARNING: no trajectory.json, so video_capture uuids are unknown —" >&2
    echo "         thumbnails skipped." >&2
  else
    # thumbnail_key is "<map_uuid>/video-captures/<vc_uuid>/object-search/thumbnails/…"
    # and the toolbox resolves it relative to the map directory, so mirror the key
    # path verbatim under the map dir (DEST when there is no map dir).
    THUMB_ROOT="${MAP_DIR:-$DEST}"
    for vc_id in ${DUMPED_IDS+"${DUMPED_IDS[@]}"}; do
      vc_uuid=$(python3 -c "
import json,sys
caps=json.load(open(sys.argv[1]))['captures']
print(next((c['video_capture_uuid'] for c in caps if str(c['video_capture_id'])==sys.argv[2]), ''))
" "$TRAJECTORY" "$vc_id")
      if [[ -z "$vc_uuid" ]]; then
        echo "video_capture ${vc_id}: not in this map version, no thumbnails to fetch"
        continue
      fi
      prefix="${MAP_UUID}/video-captures/${vc_uuid}/object-search/thumbnails"
      echo "video_capture ${vc_id}: syncing thumbnails → ${THUMB_ROOT}/${prefix}"
      run_aws s3 sync --only-show-errors \
        "s3://${PUBLIC_BUCKET}/${prefix}/" "${THUMB_ROOT}/${prefix}/"
    done
  fi
fi

echo "Done. Dumped to ${DEST}"
if [[ -n "$MAP_DIR" && -f "$TRAJECTORY" ]]; then
  echo "Next: $PYTHON -m toolbox.bricks.prod_dump_remap '${MAP_DIR}'"
  echo "      $PYTHON -m toolbox.bricks.prod_dump_merge '${MAP_DIR}'"
  echo "      $PYTHON -m toolbox.bricks.ingest_cli '${MAP_DIR}'"
fi
