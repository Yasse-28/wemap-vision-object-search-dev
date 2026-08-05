"""Load GeoPose-style JSON5 config listing maps (standalone object search)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import json5


@dataclass(frozen=True)
class ObjectSearchConfig:
    """Per-map object-search retrieval settings (config.json ``objectSearch``).

    Defaults match the documented default behaviour: pgvector on, AWS database,
    HNSW ``ef_search`` of 1000.
    """

    use_pgvector: bool = True
    pgvector_db_location: str = "aws"  # "local" or "aws"
    ef_search: int = 1000  # HNSW hnsw.ef_search (search breadth)


def _parse_object_search(raw: object, base: ObjectSearchConfig) -> ObjectSearchConfig:
    """Override ``base`` with any keys present in an ``objectSearch`` dict."""
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError("'objectSearch' must be an object")
    location = str(raw.get("pgvectorDbLocation", base.pgvector_db_location)).lower()
    if location not in ("local", "aws"):
        raise ValueError("objectSearch.pgvectorDbLocation must be 'local' or 'aws'")
    return ObjectSearchConfig(
        use_pgvector=bool(raw.get("usePgvector", base.use_pgvector)),
        pgvector_db_location=location,
        ef_search=int(raw.get("EFSearch", base.ef_search)),
    )


@dataclass(frozen=True)
class MapEntry:
    map_id: str
    map_path: Path
    emmid: Optional[int] = None
    object_search_index_path: Optional[str] = None
    object_search: ObjectSearchConfig = ObjectSearchConfig()


def load_map_entries(config_path: Path) -> Tuple[Path, List[MapEntry]]:
    """Parse config JSON5; return (config_parent, list of MapEntry).

    Each map must have ``id``. Optional ``path``, ``emmid`` (Livemap embed id),
    and ``object_search_index_path`` (override for ``object-search.db``).
    If ``path`` is set, ``map_path = config.parent / path``;
    else ``map_path = config.parent / maps / id`` (GeoPose-style default).
    """
    config_path = config_path.resolve()
    data = json5.loads(config_path.read_text(encoding="utf-8"))
    if "maps" not in data or not isinstance(data["maps"], list):
        raise ValueError("Config must contain a 'maps' array")

    folder = config_path.parent
    # Optional top-level objectSearch sets the default for every map; a per-map
    # objectSearch block overrides it.
    default_object_search = _parse_object_search(
        data.get("objectSearch"), ObjectSearchConfig()
    )
    seen: set[str] = set()
    entries: List[MapEntry] = []
    for i, m in enumerate(data["maps"]):
        if not isinstance(m, dict):
            raise ValueError(f"maps[{i}] must be an object")
        if "id" not in m:
            raise ValueError(f"maps[{i}] missing required 'id'")
        mid = str(m["id"])
        if mid in seen:
            raise ValueError(f"Duplicate map id: {mid!r}")
        seen.add(mid)
        if "path" in m:
            map_path = (folder / str(m["path"])).resolve()
        else:
            map_path = (folder / "maps" / mid).resolve()
        emmid: Optional[int] = None
        if "emmid" in m:
            raw_emmid = m["emmid"]
            if not isinstance(raw_emmid, int) or isinstance(raw_emmid, bool):
                raise ValueError(f"maps[{i}] emmid must be an integer")
            emmid = raw_emmid
        index_path = (
            str(m["object_search_index_path"])
            if "object_search_index_path" in m
            else None
        )
        object_search = _parse_object_search(
            m.get("objectSearch"), default_object_search
        )
        entries.append(
            MapEntry(
                map_id=mid,
                map_path=map_path,
                emmid=emmid,
                object_search_index_path=index_path,
                object_search=object_search,
            )
        )

    return folder, entries
