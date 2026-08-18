"""Delete a map exported by `export_roi`: its rows in pgvector, then its directory.

The toolbox's TypeScript backend has no Postgres client — it opens SQLite and
proxies everything else — so the destructive half lives here, beside the code that
wrote the map in the first place.

## What may be deleted

Only a map the toolbox produced. The gate is the config's `parent_map` key plus an
`export-provenance.json` on disk: a map fetched from production has neither, and
this repo could not rebuild its pixels (they come from `../retrieve-map-data`).
A map that still has children is refused too, so a subtree cannot be orphaned by
deleting its root.

## Why the database goes first

A failed `DELETE` must leave the directory intact, because a directory is
recoverable evidence and orphaned rows are not: once the manifest is gone, nothing
maps a `geo_ref_id` back to what it described. The reverse order would trade a
visible failure for a silent one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from toolbox.bricks.db import build_dsn, connect
from toolbox.bricks.export_roi import PROVENANCE_FILENAME
from toolbox.bricks.ingest import INDEX_NAME_TEMPLATE, drop_partial_hnsw_index
from toolbox.bricks.map_manifest import find_manifest, load_manifest
from toolbox.logging import logger


class DeletionRefused(Exception):
    """The map is not one this tool is allowed to delete."""


@dataclass
class ConfiguredMap:
    """One entry of the toolbox config, as the backend hands it over."""

    map_id: str
    path: Path
    parent_map: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConfiguredMap":
        return cls(
            map_id=str(raw["id"]),
            path=Path(str(raw["path"])),
            parent_map=(
                None if raw.get("parent_map") in (None, "") else str(raw["parent_map"])
            ),
        )


def children_of(map_id: str, configured: Sequence[ConfiguredMap]) -> list[str]:
    return [entry.map_id for entry in configured if entry.parent_map == map_id]


def _resolve_target(map_id: str, configured: Sequence[ConfiguredMap]) -> ConfiguredMap:
    for entry in configured:
        if entry.map_id == map_id:
            return entry
    raise DeletionRefused(f"'{map_id}' is not in the toolbox config.")


def plan_deletion(
    map_id: str, configured: Sequence[ConfiguredMap], dsn: str | None = None
) -> dict[str, Any]:
    """Everything the confirmation dialog needs, plus every reason to refuse.

    Raises `DeletionRefused` for the two hard gates (not a sub-map, has children).
    A missing directory or an unreadable manifest is reported, not raised: a
    half-deleted map must stay deletable.
    """
    target = _resolve_target(map_id, configured)

    if target.parent_map is None:
        raise DeletionRefused(
            f"'{map_id}' has no parent_map, so it was not produced by this toolbox. "
            "Maps received from production are not deletable here."
        )

    children = children_of(map_id, configured)
    if children:
        raise DeletionRefused(
            f"'{map_id}' still has sub-map(s): {', '.join(sorted(children))}. "
            "Delete those first."
        )

    warnings: list[str] = []
    keyframe_count: int | None = None
    geo_ref_id: int | None = None

    manifest_path = find_manifest(target.path)
    if manifest_path is None:
        warnings.append(f"No v2 manifest under {target.path}.")
    else:
        try:
            manifest = load_manifest(manifest_path)
            keyframe_count = len(manifest.keyframes)
            geo_ref_id = (
                None if manifest.geo_ref_id is None else int(manifest.geo_ref_id)
            )
        except ValueError as error:
            warnings.append(f"Manifest unreadable: {error}")

    has_provenance = (target.path / PROVENANCE_FILENAME).is_file()
    if not has_provenance:
        warnings.append(
            f"No {PROVENANCE_FILENAME}: the directory will be left on disk and only "
            "the database rows and the config entry will be removed."
        )

    # The geo_ref_id is read from the map's own manifest, never from the config —
    # the config's copy is ignored everywhere else and a stale one would aim the
    # DELETE at a neighbour.
    sharing = [
        entry.map_id
        for entry in configured
        if entry.map_id != map_id and _manifest_geo_ref_id(entry.path) == geo_ref_id
    ]
    if geo_ref_id is not None and sharing:
        raise DeletionRefused(
            f"geo_ref_id {geo_ref_id} is also used by {', '.join(sorted(sharing))}. "
            "Purging it would erase that map too."
        )

    candidates, geokeyframes = _count_rows(geo_ref_id, dsn)

    return {
        "map_id": map_id,
        "path": str(target.path),
        "parent_map": target.parent_map,
        "geo_ref_id": geo_ref_id,
        "keyframe_count": keyframe_count,
        "candidate_rows": candidates,
        "geokeyframe_rows": geokeyframes,
        "index_name": (
            None
            if geo_ref_id is None
            else INDEX_NAME_TEMPLATE.format(geo_ref_id=geo_ref_id)
        ),
        "removes_directory": has_provenance,
        "warnings": warnings,
    }


def _manifest_geo_ref_id(map_path: Path) -> int | None:
    manifest_path = find_manifest(map_path)
    if manifest_path is None:
        return None
    try:
        manifest = load_manifest(manifest_path)
    except ValueError:
        return None
    return None if manifest.geo_ref_id is None else int(manifest.geo_ref_id)


def _count_rows(
    geo_ref_id: int | None, dsn: str | None
) -> tuple[int | None, int | None]:
    """Rows the purge would remove, or (None, None) when the database is absent."""
    if geo_ref_id is None:
        return None, None
    try:
        with connect(dsn or build_dsn()) as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM object_search_candidate WHERE geo_ref_id = %s",
                [geo_ref_id],
            )
            candidates = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT count(*) FROM geokeyframe WHERE geo_ref_id = %s", [geo_ref_id]
            )
            geokeyframes = int(cursor.fetchone()[0])
            return candidates, geokeyframes
    except Exception as error:  # noqa: BLE001 - the database is optional here
        logger.warning("Could not count rows for geo_ref_id %s: %s", geo_ref_id, error)
        return None, None


def purge_database(geo_ref_id: int, dsn: str | None = None) -> dict[str, int]:
    """Drop the partial HNSW index and delete both tables' rows, in one transaction."""
    with connect(dsn or build_dsn()) as conn:
        conn.autocommit = False
        try:
            drop_partial_hnsw_index(conn, geo_ref_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM object_search_candidate WHERE geo_ref_id = %s",
                    [geo_ref_id],
                )
                candidates = cursor.rowcount
                cursor.execute(
                    "DELETE FROM geokeyframe WHERE geo_ref_id = %s", [geo_ref_id]
                )
                geokeyframes = cursor.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {"candidate_rows": candidates, "geokeyframe_rows": geokeyframes}


def delete_map(
    map_id: str, configured: Sequence[ConfiguredMap], dsn: str | None = None
) -> dict[str, Any]:
    """Purge the database, then remove the directory. Database first, deliberately."""
    plan = plan_deletion(map_id, configured, dsn=dsn)

    deleted = {"candidate_rows": 0, "geokeyframe_rows": 0}
    if plan["geo_ref_id"] is not None:
        deleted = purge_database(int(plan["geo_ref_id"]), dsn=dsn)
        _progress("database", f"{deleted['candidate_rows']} candidate row(s) removed")

    directory_removed = False
    if plan["removes_directory"]:
        shutil.rmtree(Path(plan["path"]))
        directory_removed = True
        _progress("directory", plan["path"])

    return {
        **plan,
        "deleted_candidate_rows": deleted["candidate_rows"],
        "deleted_geokeyframe_rows": deleted["geokeyframe_rows"],
        "directory_removed": directory_removed,
    }


def _progress(step: str, detail: str) -> None:
    print(f"PROGRESS {step} {detail}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", required=True, help="Path to the JSON job spec, or '-' for stdin."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without touching anything.",
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text("utf-8")
    spec = json.loads(text)
    configured = [ConfiguredMap.from_dict(item) for item in spec["configured_maps"]]

    try:
        if args.dry_run:
            result = plan_deletion(str(spec["map_id"]), configured)
        else:
            result = delete_map(str(spec["map_id"]), configured)
    except DeletionRefused as error:
        json.dump({"refused": str(error)}, sys.stdout)
        sys.stdout.write("\n")
        return 2

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
