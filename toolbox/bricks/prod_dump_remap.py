"""Re-key a **prod** object-search dump to this repo's local keyframe ids.

`scripts/dump-object-search.sh` pulls prepare outputs straight from prod S3, which
saves
the GPU hours a local `prepare` run costs. But the two worlds number keyframes
differently, and nothing downstream notices:

- prod writes the real `VideoKeyframe.id` into `metadata.parquet` (43875–108564 for
  VINCI Saint Domingue);
- here the id **is the `geo_keyframes` array index** (0–74987) — see
  `georef_source`, `map_manifest`.

The ranges overlap, so ingesting a prod dump unremapped attaches roughly a quarter
of the candidates to *unrelated* keyframes — wrong positions, no error anywhere —
and drops the rest. This module closes that gap, in place, before ingest.

## Where the mapping comes from

The map manifest carries no keyframe id, only image URLs. The 360-viewer
`trajectory.json` (public bucket, `<map_uuid>/versions/<n>/360-viewer/`) carries
the ids and per-keyframe WGS84 coords for the same keyframe set — same count, same
points. So the join is geometric: manifest EUS → WGS84 (`GeoTransform`), matched to
trajectory coords rounded to `_COORD_DECIMALS`.

Verified on VINCI Saint Domingue v8: 74988/74988 manifest keyframes matched, and of
the 20364 ids independently known from the same version's `graph.json`
(`images[].id` + `filename`), 20360 agreed and **none disagreed** (the remaining 4
fall in coincident-coordinate groups, which this module refuses to guess at).

## What it writes

Only `video_keyframe_id` changes. Rows whose id has no local counterpart get
`UNMAPPED_KEYFRAME_ID` (-1) — never a valid array index, so `ingest_cli` drops them
in its `kept_vk_ids` filter. Rows are neither reordered nor deleted, so
`embeddings.npy` stays aligned and `thumbnail_key` (which encodes the original
row index) stays valid.

A marker is written into the parquet's schema metadata under `REMAP_METADATA_KEY`.
Re-running is a no-op rather than a second remap, which would be catastrophic:
already-local ids are all valid indices, so a second pass would silently shuffle
every candidate onto a different keyframe.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolbox.bricks.ingest_cli import DEFAULT_OUTPUTS_DIRNAME, discover_capture_dirs
from toolbox.bricks.map_manifest import MapManifest, load_map_manifest
from toolbox.bricks.prepare_postprocess import write_parquet_atomically
from toolbox.bricks.vendored.geo_transform import Vector3Batch
from toolbox.logging import logger

# The outputs dirname is `ingest_cli`'s, deliberately not a second default: a dump
# that lands anywhere else needs a flag at ingest too, and the two drifting apart is
# how a remapped dump gets ingested from the wrong directory.
TRAJECTORY_FILENAME = "trajectory.json"
REMAP_METADATA_KEY = b"object_search_keyframe_id_space"
REMAP_METADATA_VALUE = b"manifest-index"
UNMAPPED_KEYFRAME_ID = -1
# Rounding of the (lng, lat, alt) join key: ~1 cm horizontally, 1 mm vertically.
# Both sides are the same export's numbers, so this only absorbs float printing.
_COORD_DECIMALS = (7, 7, 3)
# Below this share of manifest keyframes matched, assume the trajectory belongs to a
# different map version rather than remapping a fraction of the dump.
_MIN_MATCH_RATIO = 0.9


@dataclass(frozen=True)
class KeyframeIdMap:
    """Prod `VideoKeyframe.id` → local `geo_keyframes` index."""

    video_keyframe_ids: np.ndarray  # (N,) int64, sorted
    local_keyframe_ids: np.ndarray  # (N,) int64, aligned with the above
    ambiguous: int  # keyframes skipped: coincident coordinates, no unique match
    unmatched: int  # manifest keyframes absent from the trajectory

    def __len__(self) -> int:
        return int(self.video_keyframe_ids.size)

    def apply(self, ids: np.ndarray) -> np.ndarray:
        """Map `ids`, with `UNMAPPED_KEYFRAME_ID` where there is no counterpart."""
        out = np.full(ids.shape, UNMAPPED_KEYFRAME_ID, dtype=np.int64)
        if len(self) == 0 or ids.size == 0:
            return out
        pos = np.searchsorted(self.video_keyframe_ids, ids)
        pos = np.clip(pos, 0, len(self) - 1)
        hit = self.video_keyframe_ids[pos] == ids
        out[hit] = self.local_keyframe_ids[pos[hit]]
        return out


def find_trajectory(map_path: Path, outputs_dirname: str) -> Path | None:
    """The dump's `trajectory.json`, in the outputs directory or the map root."""
    map_path = Path(map_path)
    for candidate in (
        map_path / outputs_dirname / TRAJECTORY_FILENAME,
        map_path / TRAJECTORY_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def load_trajectory_keyframes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Keyframe ids and their `(lng, lat, alt)` from a 360-viewer `trajectory.json`.

    Returns:
        `(ids (N,) int64, lnglatalt (N, 3) float64)`, flattened over every capture.

    Raises:
        ValueError: if the file holds no keyframe.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids: list[int] = []
    coords: list[list[float]] = []
    for capture in data.get("captures") or []:
        for keyframe in capture.get("keyframes") or []:
            coord = keyframe.get("coords") or {}
            ids.append(int(keyframe["id"]))
            coords.append(
                [float(coord["lng"]), float(coord["lat"]), float(coord["alt"])]
            )
    if not ids:
        raise ValueError(f"{path}: no keyframes in any capture.")
    return np.asarray(ids, dtype=np.int64), np.asarray(coords, dtype=np.float64)


def _coordinate_keys(lnglatalt: np.ndarray) -> list[tuple[float, float, float]]:
    lng, lat, alt = _COORD_DECIMALS
    return [
        (round(float(a), lng), round(float(b), lat), round(float(c), alt))
        for a, b, c in lnglatalt
    ]


def build_keyframe_id_map(
    manifest: MapManifest, trajectory_path: Path
) -> KeyframeIdMap:
    """Join manifest keyframes to trajectory ids on WGS84 position.

    Only keys that are unique on *both* sides are mapped: a handful of keyframes
    share a position to the millimetre, and guessing between them would put
    candidates on the wrong keyframe just as silently as not remapping at all.

    Raises:
        ValueError: if too few manifest keyframes match, which means the
            trajectory comes from a different map version than the manifest.
    """
    trajectory_ids, trajectory_coords = load_trajectory_keyframes(trajectory_path)
    trajectory_by_key: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(_coordinate_keys(trajectory_coords)):
        trajectory_by_key[key].append(index)

    eus = np.asarray([kf.position_eus for kf in manifest.keyframes], dtype=np.float64)
    wgs84 = np.asarray(
        manifest.geo_transform().local_positions_to_wgs84(Vector3Batch(eus))
    )
    manifest_keys = _coordinate_keys(wgs84)
    manifest_key_counts: dict[tuple, int] = defaultdict(int)
    for key in manifest_keys:
        manifest_key_counts[key] += 1

    pairs: list[tuple[int, int]] = []
    ambiguous = unmatched = 0
    for local_id, key in enumerate(manifest_keys):
        group = trajectory_by_key.get(key)
        if group is None:
            unmatched += 1
        elif len(group) > 1 or manifest_key_counts[key] > 1:
            ambiguous += 1
        else:
            pairs.append((int(trajectory_ids[group[0]]), local_id))

    matched = len(manifest.keyframes) - unmatched
    if matched < _MIN_MATCH_RATIO * len(manifest.keyframes):
        raise ValueError(
            f"{trajectory_path}: only {matched}/{len(manifest.keyframes)} manifest "
            "keyframes have a matching position — this trajectory is from another "
            f"map version. Download the one for version {manifest.map_version}: "
            f"s3://wemap-vision-storage-public-prod/{manifest.map_uuid}/versions/"
            f"{manifest.map_version}/360-viewer/{TRAJECTORY_FILENAME}"
        )

    pairs.sort()
    logger.info(
        "Keyframe id map: %d of %d manifest keyframes (%d ambiguous position(s), "
        "%d absent from the trajectory).",
        len(pairs),
        len(manifest.keyframes),
        ambiguous,
        unmatched,
    )
    return KeyframeIdMap(
        video_keyframe_ids=np.asarray([p[0] for p in pairs], dtype=np.int64),
        local_keyframe_ids=np.asarray([p[1] for p in pairs], dtype=np.int64),
        ambiguous=ambiguous,
        unmatched=unmatched,
    )


def is_remapped(table: pa.Table) -> bool:
    """Whether this parquet's `video_keyframe_id` is already in manifest-index space."""
    metadata = table.schema.metadata or {}
    return metadata.get(REMAP_METADATA_KEY) == REMAP_METADATA_VALUE


def remap_capture(metadata_path: Path, id_map: KeyframeIdMap) -> tuple[int, int]:
    """Rewrite one capture's `video_keyframe_id` column in place.

    Returns:
        `(rows, mapped)` — total rows and how many got a local keyframe id.
        `(rows, rows)` with no write when the file is already remapped.
    """
    metadata_path = Path(metadata_path)
    table = pq.read_table(metadata_path)
    if is_remapped(table):
        logger.info("%s: already remapped, left alone.", metadata_path)
        return table.num_rows, table.num_rows

    field_index = table.schema.get_field_index("video_keyframe_id")
    if field_index < 0:
        raise ValueError(f"{metadata_path}: no video_keyframe_id column.")
    field = table.schema.field(field_index)

    remapped = id_map.apply(table.column(field_index).to_numpy().astype(np.int64))
    table = table.set_column(
        field_index, field, pa.array(remapped, type=field.type)
    ).replace_schema_metadata(
        {**(table.schema.metadata or {}), REMAP_METADATA_KEY: REMAP_METADATA_VALUE}
    )
    write_parquet_atomically(table, metadata_path)

    mapped = int((remapped != UNMAPPED_KEYFRAME_ID).sum())
    return table.num_rows, mapped


def report_out_of_version_captures(
    manifest: MapManifest, capture_dirs: list[Path]
) -> list[Path]:
    """Name the capture dirs this map version does not contain.

    A dump taken before `videos[]` existed holds every capture S3 had for the map,
    including other versions'. Those have no keyframe in the manifest, so they remap
    to nothing — visible today only as a 0 % line per capture, which reads like a
    coordinate-join failure. With the list, the reason is stated instead of inferred.
    Returns them so a caller can report; nothing is skipped, because a remap of rows
    that are already unmappable is a no-op either way.
    """
    if not manifest.video_capture_ids:
        return []
    out_of_version = [
        path
        for path in capture_dirs
        if path.name.isdigit() and int(path.name) not in manifest.video_capture_ids
    ]
    if out_of_version:
        logger.warning(
            "%d capture(s) are not in map version %d (manifest videos[]): %s — no "
            "keyframe of theirs is in the manifest, so every row will map to %d and "
            "ingest will drop them. scripts/dump-object-search.sh no longer fetches "
            "these; the directories can be deleted.",
            len(out_of_version),
            manifest.map_version,
            ", ".join(path.name for path in out_of_version),
            UNMAPPED_KEYFRAME_ID,
        )
    return out_of_version


def remap_dump(
    map_path: Path,
    *,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    trajectory_path: Path | None = None,
) -> tuple[int, int]:
    """Remap every capture of a prod dump under `{map}/{outputs_dirname}/`.

    Returns:
        `(rows, mapped)` summed over the captures.

    Raises:
        FileNotFoundError: if the trajectory or the dump directory is missing.
    """
    map_path = Path(map_path)
    trajectory = Path(trajectory_path) if trajectory_path else None
    if trajectory is None:
        trajectory = find_trajectory(map_path, outputs_dirname)
    if trajectory is None or not trajectory.is_file():
        raise FileNotFoundError(
            f"No {TRAJECTORY_FILENAME} for '{map_path}'. It carries the prod keyframe "
            "ids; scripts/dump-object-search.sh downloads it beside the captures."
        )

    manifest = load_map_manifest(map_path)
    id_map = build_keyframe_id_map(manifest, trajectory)

    # Same discovery rule as ingest: a directory holding a metadata.parquet.
    capture_dirs = discover_capture_dirs(map_path, outputs_dirname)
    if not capture_dirs:
        raise FileNotFoundError(
            f"No metadata.parquet under '{map_path / outputs_dirname}'."
        )

    report_out_of_version_captures(manifest, capture_dirs)

    total_rows = total_mapped = 0
    for capture_dir in capture_dirs:
        rows, mapped = remap_capture(capture_dir / "metadata.parquet", id_map)
        total_rows += rows
        total_mapped += mapped
        share = mapped / rows if rows else 0.0
        log = logger.info if mapped else logger.warning
        log(
            "%s: %d/%d rows mapped (%.1f%%)%s",
            capture_dir.name,
            mapped,
            rows,
            share * 100.0,
            (
                " — no keyframe of this capture is in the manifest, so none of its "
                "candidates can be positioned; ingest will skip it"
                if not mapped
                else ""
            ),
        )
    logger.info(
        "Remapped %d/%d candidate rows to manifest keyframe ids.",
        total_mapped,
        total_rows,
    )
    return total_rows, total_mapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-key a prod object-search dump's video_keyframe_id to the local "
            "geo_keyframes indices, using the 360-viewer trajectory.json."
        )
    )
    parser.add_argument(
        "map_path", type=Path, help="Map directory (holds the v2 manifest)."
    )
    parser.add_argument(
        "--outputs-dirname",
        default=DEFAULT_OUTPUTS_DIRNAME,
        help=f"Dump directory under the map (default: {DEFAULT_OUTPUTS_DIRNAME}).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help=f"Path to {TRAJECTORY_FILENAME}. Defaults to the dump dir, then the map "
        "root.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.map_path.is_dir():
        parser.error(f"No such map directory: '{args.map_path}'.")

    remap_dump(
        args.map_path,
        outputs_dirname=args.outputs_dirname,
        trajectory_path=args.trajectory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
