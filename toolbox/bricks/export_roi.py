"""Export the keyframes inside a set of drawn regions as a new map directory.

This is the first module in the repo that **writes** a map directory. Everything
else consumes one: the manifest is a dump of production's Django objects and the
pixels are fetched by the sibling `../retrieve-map-data`. Writing one means taking
on two contracts that were previously somebody else's problem.

## Keyframe ids are array indices

The manifest carries no integer keyframe id — `keyframe_id` is the index in
`geo_keyframes` (see `map_manifest`). A subset therefore **renumbers every
keyframe**, densely, from 0. That is not a choice we can avoid; the alternative,
leaving holes, breaks the `id == index` assumption that `prepare_runner` and
`ingest_cli` both rely on. The old→new table is written to
`export-provenance.json` so the renumbering stays auditable, and the optional
`object-search/` artifacts are remapped through that same table.

## `geo_ref_id` must be new

There is no table per map. `object_search_candidate` and `geokeyframe` are single
tables partitioned by the `geo_ref_id` column, and `ingest_cli` starts by running
`DELETE ... WHERE geo_ref_id = %s`. An exported map that kept its source's
`geo_ref_id` would therefore *erase the source* from pgvector on its next ingest.
`allocate_geo_ref_id` takes the next free value instead; `service.load_map_entries`
grew a guard against two maps sharing one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolbox.bricks.db import build_dsn, connect
from toolbox.bricks.ingest_cli import DEFAULT_OUTPUTS_DIRNAME, EMBEDDING_DIM
from toolbox.bricks.map_manifest import MapManifest, find_manifest, load_manifest
from toolbox.bricks.vendored.geo_transform import Vector3Batch, _point_in_ring
from toolbox.logging import logger

MANIFEST_STAMP_FORMAT = "%Y%m%d_%H%M%S"
PROVENANCE_FILENAME = "export-provenance.json"
THUMBNAILS_DIRNAME = "thumbnails"

# A v1-converted index has no crops on disk and carries this virtual key instead of
# a thumbnail path (see `AI_CONTEXT/toolbox.md`). It embeds the row index, so it has
# to be rewritten alongside the real ones.
VIRTUAL_ROW_PREVIEW_PREFIX = "object-search/rows/"


@dataclass(frozen=True)
class Roi:
    """One drawn region: an open lng/lat ring, on one level.

    `level` is the level *value* as the manifest spells it (`geo_levels[].value`),
    or None for a region drawn with no level resolved. A None level matches only
    keyframes whose own level is unresolved — it is not a wildcard, because
    "the SDK reported no floor" and "this region covers every floor" are different
    statements and only the first one can happen by accident.
    """

    ring: tuple[tuple[float, float], ...]  # (lng, lat)
    level: float | None


@dataclass
class Selection:
    keyframe_ids: list[int]
    per_level: dict[str, int]

    @property
    def total(self) -> int:
        return len(self.keyframe_ids)


@dataclass
class ExportStats:
    keyframes: int = 0
    images_linked: int = 0
    images_copied: int = 0
    depths_linked: int = 0
    depths_copied: int = 0
    missing_images: list[str] = field(default_factory=list)
    missing_depths: list[str] = field(default_factory=list)
    artifact_rows: int = 0
    thumbnails: int = 0


def _level_key(value: float | None) -> str:
    """Stable dict key for a level value, so per-level counts survive JSON."""
    return "null" if value is None else f"{value:g}"


def select_keyframes(manifest: MapManifest, rois: Sequence[Roi]) -> Selection:
    """Keyframes falling inside at least one ROI, matched on level.

    The level comes from the manifest's own altitude bands, fed the **EUS up
    coordinate** — never the WGS84 altitude, which is the third silent failure
    mode listed in `AI_CONTEXT/overview.md`.
    """
    if not rois:
        return Selection(keyframe_ids=[], per_level={})

    positions = np.stack([kf.position_eus for kf in manifest.keyframes])
    transform = manifest.geo_transform()
    wgs84 = np.asarray(transform.local_positions_to_wgs84(Vector3Batch(positions)))
    levels = transform.levels_for_altitudes(
        positions[:, 1], lats=wgs84[:, 1], lngs=wgs84[:, 0]
    )

    # Rings are converted once, not per keyframe: `_point_in_ring` indexes them
    # positionally and a tuple of tuples would be rebuilt for every point.
    by_level: dict[str, list[list[list[float]]]] = {}
    for roi in rois:
        by_level.setdefault(_level_key(roi.level), []).append(
            [[point[0], point[1]] for point in roi.ring]
        )

    keyframe_ids: list[int] = []
    per_level: dict[str, int] = {}
    for index, keyframe in enumerate(manifest.keyframes):
        level = None if np.isnan(levels[index]) else float(levels[index])
        candidates = by_level.get(_level_key(level))
        if not candidates:
            continue
        lng, lat = float(wgs84[index, 0]), float(wgs84[index, 1])
        if not any(_point_in_ring(lng, lat, ring) for ring in candidates):
            continue
        keyframe_ids.append(keyframe.keyframe_id)
        per_level[_level_key(level)] = per_level.get(_level_key(level), 0) + 1

    return Selection(keyframe_ids=keyframe_ids, per_level=per_level)


def allocate_geo_ref_id(
    map_paths: Iterable[Path], dsn: str | None = None
) -> tuple[int, bool]:
    """The next free `geo_ref_id`, and whether the database could be consulted.

    Scans every configured map's manifest, and — when the database answers — the
    two tables partitioned by the column. A missing database is not fatal: the
    manifests alone give a safe value for a toolbox that has never ingested. The
    flag is recorded in the provenance so a later collision is explicable.
    """
    highest = 0
    for map_path in map_paths:
        manifest_path = find_manifest(Path(map_path))
        if manifest_path is None:
            continue
        try:
            manifest = load_manifest(manifest_path)
        except ValueError as error:
            logger.warning("Skipping %s while allocating: %s", manifest_path, error)
            continue
        if manifest.geo_ref_id is not None:
            highest = max(highest, int(manifest.geo_ref_id))

    database_consulted = False
    try:
        with connect(dsn or build_dsn()) as conn, conn.cursor() as cursor:
            for table in ("object_search_candidate", "geokeyframe"):
                cursor.execute(f"SELECT max(geo_ref_id) FROM {table}")
                row = cursor.fetchone()
                if row and row[0] is not None:
                    highest = max(highest, int(row[0]))
            database_consulted = True
    except Exception as error:  # noqa: BLE001 - any driver/connection error is fine
        logger.warning(
            "Could not read the database while allocating a geo_ref_id (%s); "
            "falling back to the configured manifests alone.",
            error,
        )

    # The online service validates `geo_ref_id > 0`, so 0 is never a valid answer.
    return max(highest + 1, 1), database_consulted


def _link_or_copy(source: Path, destination: Path) -> bool:
    """Hardlink `source` to `destination`, copying instead if the link fails.

    Returns True when the hardlink worked. A cross-device destination raises
    `EXDEV`, and so does any filesystem without hardlink support — both are
    ordinary, not errors.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return True
    except OSError:
        shutil.copy2(source, destination)
        return False


def _copy_assets(
    source_map: Path,
    staging: Path,
    keyframes: Sequence[Any],
    stats: ExportStats,
) -> None:
    """Link `images/` and `depths/` for the selected keyframes.

    Basenames are preserved, so the exported manifest's `image_url` / `depth_url`
    keep resolving with no rewriting.
    """
    for keyframe in keyframes:
        for dirname, filename, linked_attr, copied_attr, missing_attr in (
            (
                "images",
                keyframe.image_filename,
                "images_linked",
                "images_copied",
                "missing_images",
            ),
            (
                "depths",
                keyframe.depth_filename,
                "depths_linked",
                "depths_copied",
                "missing_depths",
            ),
        ):
            if not filename:
                continue
            source = source_map / dirname / filename
            if not source.is_file():
                getattr(stats, missing_attr).append(filename)
                continue
            linked = _link_or_copy(source, staging / dirname / filename)
            attr = linked_attr if linked else copied_attr
            setattr(stats, attr, getattr(stats, attr) + 1)


def _export_artifacts(
    source_map: Path,
    staging: Path,
    id_map: dict[int, int],
    stats: ExportStats,
) -> None:
    """Subset `object-search/` — parquet, embeddings and thumbnails.

    Only the single-directory layout is exported. `discover_capture_dirs` also
    accepts `object-search/{capture}/`, but an export that merged several captures
    into one would have to renumber `row_index` across them, and no map in use here
    has that layout; a per-capture source is refused rather than silently flattened.
    """
    outputs = source_map / DEFAULT_OUTPUTS_DIRNAME
    metadata_path = outputs / "metadata.parquet"
    embeddings_path = outputs / "embeddings.npy"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{outputs}: no metadata.parquet. Either run prepare on the source map "
            "or export without the object-search artifacts."
        )
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"{outputs}: no embeddings.npy beside the parquet.")

    metadata = pq.read_table(metadata_path).to_pandas()
    # Headerless raw float16 dump despite the .npy name — `np.load` rejects it.
    # See `ingest_cli._read_capture_outputs`.
    embeddings = np.fromfile(embeddings_path, dtype=np.float16).reshape(
        -1, EMBEDDING_DIM
    )
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"{outputs}: metadata/embeddings length mismatch "
            f"({len(metadata)} vs {embeddings.shape[0]})."
        )

    keep = metadata["video_keyframe_id"].isin(id_map.keys()).to_numpy()
    metadata = metadata.loc[keep].reset_index(drop=True)
    embeddings = embeddings[keep]

    metadata["video_keyframe_id"] = [
        id_map[int(value)] for value in metadata["video_keyframe_id"]
    ]
    if "geokeyframe_id" in metadata.columns:
        metadata["geokeyframe_id"] = metadata["video_keyframe_id"]

    destination = staging / DEFAULT_OUTPUTS_DIRNAME
    destination.mkdir(parents=True, exist_ok=True)

    old_row_indices = (
        metadata["row_index"].to_numpy().copy()
        if "row_index" in metadata.columns
        else np.arange(len(metadata), dtype=np.int64)
    )
    if "row_index" in metadata.columns:
        metadata["row_index"] = np.arange(len(metadata), dtype=np.int64)

    if "thumbnail_key" in metadata.columns:
        metadata["thumbnail_key"] = [
            _rewrite_thumbnail_key(
                key, int(old_row_indices[index]), index, outputs, destination, stats
            )
            for index, key in enumerate(metadata["thumbnail_key"])
        ]

    pq.write_table(
        pa.Table.from_pandas(metadata, preserve_index=False),
        destination / "metadata.parquet",
    )
    embeddings.tofile(destination / "embeddings.npy")
    stats.artifact_rows = len(metadata)


def _rewrite_thumbnail_key(
    key: Any,
    old_row_index: int,
    new_row_index: int,
    source_outputs: Path,
    destination_outputs: Path,
    stats: ExportStats,
) -> Any:
    """Move one thumbnail to its new row index, rewriting the key.

    Two shapes exist. A real key points at a file under
    `object-search/thumbnails/`; the virtual `object-search/rows/{row}.png` of a
    v1-converted index has no file behind it and only needs its number changed.
    """
    if not isinstance(key, str) or not key:
        return key

    if key.startswith(VIRTUAL_ROW_PREVIEW_PREFIX):
        return f"{VIRTUAL_ROW_PREVIEW_PREFIX}{new_row_index}.png"

    source = source_outputs.parent / key
    if not source.is_file():
        # Also try the key relative to the outputs dir, which is how a per-capture
        # layout would spell it.
        source = source_outputs / Path(key).name
    if not source.is_file():
        return key

    new_name = f"{new_row_index:06d}{source.suffix}"
    _link_or_copy(source, destination_outputs / THUMBNAILS_DIRNAME / new_name)
    stats.thumbnails += 1
    return f"{DEFAULT_OUTPUTS_DIRNAME}/{THUMBNAILS_DIRNAME}/{new_name}"


def build_manifest_document(
    source_document: dict[str, Any],
    keyframe_ids: Sequence[int],
    map_id: str,
    geo_ref_id: int,
) -> dict[str, Any]:
    """The exported manifest, as a plain document.

    Built from the *raw* source JSON rather than from `MapManifest`, so fields this
    repo does not read survive the round trip. Only four things change: the
    keyframe list, the `geo_ref` on every level, the map identity, and the version,
    which restarts at 1 because this is a new lineage rather than a new dump of the
    source.
    """
    document: dict[str, Any] = json.loads(json.dumps(source_document))
    source_keyframes = document.get("geo_keyframes") or []
    document["geo_keyframes"] = [source_keyframes[i] for i in keyframe_ids]

    for level in document.get("geo_levels") or []:
        if "geo_ref" in level:
            level["geo_ref"] = geo_ref_id

    map_block = dict(document.get("map") or {})
    map_block["uuid"] = str(uuid.uuid4())
    map_block["name"] = map_id
    document["map"] = map_block
    return document


def export_roi(
    source_map: Path,
    destination: Path,
    map_id: str,
    rois: Sequence[Roi],
    geo_ref_id: int,
    include_artifacts: bool = False,
    parent_map: str | None = None,
    database_consulted: bool = True,
) -> dict[str, Any]:
    """Write the subset of `source_map` inside `rois` as a new map at `destination`.

    Everything is staged in a sibling `.partial-*` directory and moved into place
    at the end. A half-written map directory would otherwise still satisfy
    `find_manifest`, which is exactly the kind of plausible-but-wrong state this
    repo's gotchas are made of.
    """
    source_map = Path(source_map)
    destination = Path(destination)
    manifest_path = find_manifest(source_map)
    if manifest_path is None:
        raise FileNotFoundError(f"{source_map}: no v2 manifest.")
    manifest = load_manifest(manifest_path)

    selection = select_keyframes(manifest, rois)
    if not selection.keyframe_ids:
        raise ValueError("No keyframe falls inside the drawn regions.")

    _progress("select", f"{selection.total} keyframe(s) selected")

    id_map = {old: new for new, old in enumerate(selection.keyframe_ids)}
    source_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document = build_manifest_document(
        source_document, selection.keyframe_ids, map_id, geo_ref_id
    )

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} exists and is not empty.")

    stamp = datetime.now(timezone.utc).strftime(MANIFEST_STAMP_FORMAT)
    staging = destination.parent / f".{destination.name}.partial-{stamp}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    stats = ExportStats(keyframes=selection.total)
    try:
        (staging / f"{map_id}_1_{stamp}.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _progress("manifest", f"{map_id}_1_{stamp}.json")

        keyframes = [manifest.keyframes[i] for i in selection.keyframe_ids]
        _copy_assets(source_map, staging, keyframes, stats)
        _progress(
            "assets",
            f"{stats.images_linked + stats.images_copied} image(s), "
            f"{stats.depths_linked + stats.depths_copied} depth(s)",
        )

        if include_artifacts:
            _export_artifacts(source_map, staging, id_map, stats)
            _progress("artifacts", f"{stats.artifact_rows} row(s)")

        provenance = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "map_id": manifest.map_id,
                "path": str(source_map),
                "manifest": manifest_path.name,
                "geo_ref_id": manifest.geo_ref_id,
            },
            "map_id": map_id,
            "parent_map": parent_map,
            "geo_ref_id": geo_ref_id,
            "geo_ref_id_allocated_against_database": database_consulted,
            "rois": [
                {"level": roi.level, "ring": [list(point) for point in roi.ring]}
                for roi in rois
            ],
            "keyframe_count": selection.total,
            "keyframes_per_level": selection.per_level,
            "keyframe_id_map": {str(old): new for old, new in id_map.items()},
            "includes_object_search_artifacts": include_artifacts,
            "assets": {
                "images_linked": stats.images_linked,
                "images_copied": stats.images_copied,
                "depths_linked": stats.depths_linked,
                "depths_copied": stats.depths_copied,
                "missing_images": stats.missing_images,
                "missing_depths": stats.missing_depths,
            },
        }
        (staging / PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.rmdir()  # empty, checked above
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _progress("done", str(destination))
    return {
        "map_id": map_id,
        "path": str(destination),
        "geo_ref_id": geo_ref_id,
        "keyframe_count": selection.total,
        "keyframes_per_level": selection.per_level,
        "missing_images": stats.missing_images,
        "missing_depths": stats.missing_depths,
        "artifact_rows": stats.artifact_rows,
        "thumbnails": stats.thumbnails,
    }


def _progress(step: str, detail: str) -> None:
    """One parsable progress line, consumed by the toolbox backend's job runner."""
    print(f"PROGRESS {step} {detail}", flush=True)


def parse_rois(raw: Sequence[dict[str, Any]]) -> list[Roi]:
    rois: list[Roi] = []
    for index, entry in enumerate(raw):
        ring = entry.get("ring") or []
        if len(ring) < 3:
            raise ValueError(f"rois[{index}]: a region needs at least 3 vertices.")
        level = entry.get("level")
        rois.append(
            Roi(
                ring=tuple((float(p[0]), float(p[1])) for p in ring),
                level=None if level is None else float(level),
            )
        )
    return rois


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", required=True, help="Path to the JSON job spec, or '-' for stdin."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the selection and print its counts.",
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text("utf-8")
    spec = json.loads(text)

    source_map = Path(spec["source_map_path"])
    rois = parse_rois(spec.get("rois") or [])

    if args.dry_run:
        manifest_path = find_manifest(source_map)
        if manifest_path is None:
            raise FileNotFoundError(f"{source_map}: no v2 manifest.")
        selection = select_keyframes(load_manifest(manifest_path), rois)
        geo_ref_id, consulted = allocate_geo_ref_id(
            [Path(p) for p in spec.get("configured_map_paths") or []]
        )
        json.dump(
            {
                "total": selection.total,
                "per_level": selection.per_level,
                "keyframe_ids": selection.keyframe_ids,
                "geo_ref_id": geo_ref_id,
                "geo_ref_id_allocated_against_database": consulted,
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    geo_ref_id = spec.get("geo_ref_id")
    consulted = True
    if geo_ref_id is None:
        geo_ref_id, consulted = allocate_geo_ref_id(
            [Path(p) for p in spec.get("configured_map_paths") or []]
        )

    result = export_roi(
        source_map=source_map,
        destination=Path(spec["destination_path"]),
        map_id=str(spec["map_id"]),
        rois=rois,
        geo_ref_id=int(geo_ref_id),
        include_artifacts=bool(spec.get("include_artifacts")),
        parent_map=spec.get("parent_map"),
        database_consulted=consulted,
    )
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
