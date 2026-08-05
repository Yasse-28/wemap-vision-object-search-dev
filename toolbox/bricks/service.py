"""The local stand-in for Django's object-search API.

In production the request path is:

    livemap → Django (v1_5_views) → GPU service (/object-search/by-text) → Postgres
                    │
                    └── candidates.load_enriched_candidates + v1_5_logic

This service is the middle box. It calls the **mirrored** online service for
embedding + HNSW (exactly as Django's `clients/online.py` does), then runs the
ported enrichment and clustering. That keeps the mirror untouched while giving the
toolbox and the benchmark the `localize` endpoint they need.

## Endpoints

    GET  /health
    POST /{map_id}/object-search/localize   JSON {text, num_results, ...}
                                            or multipart {image, ...}
    POST /{map_id}/object-search/text       flat v2 candidate list

The `localize` path and its response shape (`{"localizations": [...]}` with
`coordinates: [lat, lng, alt]` and `match_score`) are what
`toolbox/benchmark/object_search_http_benchmark.py --api-style standalone --online`
already expects, so the benchmark needs no changes.

## Configuration

Reads the same config file as the toolbox's TS backend
(`toolbox/backend/src/config.ts`), plus one new per-map field:

    { "maps": [ { "id": "my-map", "path": "maps/my-map", "geo_ref_id": 1 } ] }

`geo_ref_id` is optional for v2 maps — it is read from the manifest, which is also
where `ingest_cli` gets it, so the two cannot disagree. v1 maps must set it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from toolbox.bricks import db
from toolbox.bricks.candidates import (
    K_INTERNAL,
    load_enriched_candidates,
    resolve_candidates_v2_response,
)
from toolbox.bricks.georef_source import load_pose_source
from toolbox.bricks.localize import LocalizationParams, build_localize_response
from toolbox.bricks.vendored.geo_transform import GeoTransform
from toolbox.logging import logger

DEFAULT_ANN_BASE_URL = "http://127.0.0.1:8000"
# 45678 on purpose: this service is a drop-in replacement for the standalone one it
# supersedes, so the toolbox, wemap-vision-tools and the benchmark keep the port and
# the `/{map_id}/object-search/…` path shape they already agree on (ADR 0001 §gaps).
DEFAULT_PORT = 45678


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class MapEntry:
    id: str
    path: Path
    geo_ref_id: int


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    """Tolerate the comments and trailing commas the toolbox config file uses.

    The TS side does the same thing in `config.ts`. Kept as a few lines here rather
    than a `json5` dependency for one file.
    """
    # Strip // and /* */ outside of strings.
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    quote = ""
    escaped = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string, quote = True, ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            out.append("\n")
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


def load_map_entries(config_path: Path) -> dict[str, MapEntry]:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    data = json.loads(
        _strip_json_comments_and_trailing_commas(
            config_path.read_text(encoding="utf-8")
        )
    )
    maps = data.get("maps")
    if not isinstance(maps, list):
        raise ValueError("Config must contain a 'maps' array.")

    entries: dict[str, MapEntry] = {}
    for index, item in enumerate(maps):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"maps[{index}] must be an object with a string id.")
        map_id = item["id"]
        if map_id in entries:
            raise ValueError(f"Duplicate map id: {map_id}")
        raw_path = item.get("path")
        map_path = (
            (config_dir / raw_path).resolve()
            if isinstance(raw_path, str) and raw_path
            else (config_dir / "maps" / map_id).resolve()
        )
        geo_ref_id = item.get("geo_ref_id")
        if geo_ref_id is None:
            # Prefer the manifest's own id over a config guess; the two disagreeing
            # returns zero hits with no error.
            try:
                geo_ref_id = load_pose_source(map_path).geo_ref_id
            except (FileNotFoundError, ValueError):
                geo_ref_id = None
            if geo_ref_id is None:
                raise ValueError(
                    f"maps[{index}] ('{map_id}') needs a geo_ref_id: none in the "
                    "config, and no v2 manifest to take it from."
                )
            logger.info(
                "Map '%s': geo_ref_id %s from its manifest.", map_id, geo_ref_id
            )
        if not isinstance(geo_ref_id, int):
            raise ValueError(f"maps[{index}].geo_ref_id must be an integer.")
        entries[map_id] = MapEntry(id=map_id, path=map_path, geo_ref_id=int(geo_ref_id))
    return entries


# ------------------------------------------------------------------------ ANN client


class OnlineServiceUnavailable(Exception):
    """The mirrored online service is unreachable or unhealthy."""


def _post(url: str, payload: dict, timeout_s: float) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OnlineServiceUnavailable(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OnlineServiceUnavailable(f"{url} unreachable: {exc}") from exc


def query_by_text(
    ann_base_url: str, geo_ref_id: int, text: str, num_results: int, timeout_s: float
) -> list[dict]:
    """Call the mirrored `/object-search/by-text`; return `[{id, similarity}, ...]`.

    Same contract as production's `object_search/clients/online.py::query_by_text`.
    """
    hits = _post(
        f"{ann_base_url.rstrip('/')}/object-search/by-text",
        {"geo_ref_id": geo_ref_id, "text": text, "num_results": num_results},
        timeout_s,
    )
    if not isinstance(hits, list):
        raise OnlineServiceUnavailable("by-text did not return a list of hits.")
    return hits


def query_by_image(
    ann_base_url: str,
    geo_ref_id: int,
    image_bytes: bytes,
    filename: str,
    num_results: int,
    timeout_s: float,
) -> list[dict]:
    """Call the mirrored `/object-search/by-image` (multipart)."""
    boundary = "----objectsearchbricks"
    parts: list[bytes] = []
    for name, value in (("geo_ref_id", geo_ref_id), ("num_results", num_results)):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(image_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    request = urllib.request.Request(
        f"{ann_base_url.rstrip('/')}/object-search/by-image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            hits = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OnlineServiceUnavailable(
            f"HTTP {exc.code} from by-image: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OnlineServiceUnavailable(f"by-image unreachable: {exc}") from exc
    if not isinstance(hits, list):
        raise OnlineServiceUnavailable("by-image did not return a list of hits.")
    return hits


# --------------------------------------------------------------------------- models


class LocalizeRequest(BaseModel):
    """Matches what the HTTP benchmark posts, plus livemap's `num_results`."""

    text: str = Field(min_length=1)
    num_results: int = Field(default=100, gt=0, le=5000)
    min_similarity: float = 0.2
    candidate_count: int = Field(default=K_INTERNAL, gt=0, le=5000)
    clustering_eps_m: float = 2.0
    min_keyframes_per_cluster: int = 2
    max_observations_per_cluster: int = 10
    # Accepted and ignored: the benchmark always sends it, the router that would
    # have consumed it was a stub and went to legacy/.
    search_type: str | None = None

    def to_params(self) -> LocalizationParams:
        return LocalizationParams(
            candidate_count=self.candidate_count,
            num_results=self.num_results,
            min_similarity=self.min_similarity,
            max_observations_per_cluster=self.max_observations_per_cluster,
            clustering_eps_m=self.clustering_eps_m,
            min_keyframes_per_cluster=self.min_keyframes_per_cluster,
        )


class TextSearchRequest(BaseModel):
    text: str = Field(min_length=1)
    num_results: int = Field(default=K_INTERNAL, gt=0, le=5000)


# ------------------------------------------------------------------------------ app


class ServiceState:
    def __init__(self) -> None:
        self.maps: dict[str, MapEntry] = {}
        self.ann_base_url: str = DEFAULT_ANN_BASE_URL
        self.timeout_s: float = 60.0
        self._geo_transforms: dict[str, GeoTransform] = {}

    def geo_transform(self, entry: MapEntry) -> GeoTransform:
        """Cached per map — re-parsing it per request would dominate latency."""
        if entry.id not in self._geo_transforms:
            try:
                self._geo_transforms[entry.id] = load_pose_source(
                    entry.path
                ).geo_transform
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(
                    status_code=503, detail=f"Map '{entry.id}': {exc}"
                ) from exc
        return self._geo_transforms[entry.id]


state = ServiceState()


def create_app() -> FastAPI:
    app = FastAPI(title="object-search bricks (dev-only)")

    def _entry(map_id: str) -> MapEntry:
        entry = state.maps.get(map_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown map '{map_id}'.")
        return entry

    @app.get("/health")
    def health() -> dict:
        """Per-map readiness, in the `{id: state}` shape the toolbox already polls.

        A map is `ready` once its pose source loads — a v2 manifest, or a legacy
        `georef.db`. That is the one input this service cannot work without. Whether
        the map has *rows* in pgvector is not checked here: that would need a DB
        round-trip per health poll, and an unindexed map surfaces plainly as an empty
        `localizations` list.
        """

        def map_state(entry: MapEntry) -> str:
            try:
                load_pose_source(entry.path)
            except (FileNotFoundError, ValueError):
                return "error"
            return "ready"

        return {
            "status": "ok",
            "maps": {entry.id: map_state(entry) for entry in state.maps.values()},
        }

    @app.post("/{map_id}/object-search/localize")
    async def localize(map_id: str, request: Request) -> dict:
        """Text (JSON) or image (multipart) → clustered, ranked map positions."""
        entry = _entry(map_id)
        content_type = request.headers.get("content-type", "")

        t0 = time.perf_counter()
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("image")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=422, detail="Missing 'image' file.")
            params = LocalizationParams(
                num_results=int(form.get("num_results", 100)),
                candidate_count=int(form.get("candidate_count", K_INTERNAL)),
            )
            try:
                hits = query_by_image(
                    state.ann_base_url,
                    entry.geo_ref_id,
                    await upload.read(),
                    getattr(upload, "filename", "query.jpg") or "query.jpg",
                    params.candidate_count,
                    state.timeout_s,
                )
            except OnlineServiceUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        else:
            body = LocalizeRequest.model_validate(await request.json())
            params = body.to_params()
            try:
                hits = query_by_text(
                    state.ann_base_url,
                    entry.geo_ref_id,
                    body.text,
                    params.candidate_count,
                    state.timeout_s,
                )
            except OnlineServiceUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        time_embedding_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        geo_transform = state.geo_transform(entry)
        with db.connect() as conn:
            candidates = load_enriched_candidates(
                conn, entry.geo_ref_id, hits, geo_transform
            )
        response = build_localize_response(
            candidates,
            geo_transform,
            params=params,
            time_embedding_ms=time_embedding_ms,
            time_retrieval_ms=int((time.perf_counter() - t1) * 1000),
        )
        if not response["localizations"] and hits:
            logger.warning(
                "%d ANN hits but zero localizations for map '%s'. Usual cause: no "
                "object_position (depth missing — run prepare_postprocess), or "
                "min_keyframes_per_cluster filtering everything out.",
                len(hits),
                map_id,
            )
        return response

    @app.post("/{map_id}/object-search/text")
    async def text_search(map_id: str, body: TextSearchRequest) -> dict:
        """Flat, unclustered candidate list.

        Returns two views of the same result set:

        - `candidates`: the enriched rows (lat/lng/alt/level/thumbnail/…). This is
          what the toolbox frontend builds its result list from
          (`enrichedFromCandidates`), so **the key names here are a wire contract** —
          `toolbox/tests/test_integration_db.py` pins them.
        - `results`: `[[cutout_id, similarity], …]` — the standalone service's shape,
          still read by the frontend's `parseTextPairs`.

        Enrichment happens server-side against the live index. The frontend used to
        re-enrich through the workbench's `enrich-text-results` route, which read the
        retired SQLite index; that route is gone.
        """
        entry = _entry(map_id)
        try:
            hits = query_by_text(
                state.ann_base_url,
                entry.geo_ref_id,
                body.text,
                body.num_results,
                state.timeout_s,
            )
        except OnlineServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        with db.connect() as conn:
            response = resolve_candidates_v2_response(
                conn, entry.geo_ref_id, hits, state.geo_transform(entry)
            )
        response["results"] = [
            [str(candidate["id"]), candidate["similarity"]]
            for candidate in response["candidates"]
        ]
        # The standalone had a Qwen query router that would set this; it was a stub
        # that always returned "cutout" and went to legacy/ with the rest.
        response["router_object_type"] = None
        return response

    @app.post("/{map_id}/object-search/localize-offline")
    async def localize_offline(map_id: str) -> dict:
        """Retired: there is no offline path any more.

        "Offline" meant exact cosine over an index held in RAM, loaded from the
        standalone `object-search.db`. Retrieval is now HNSW in pgvector, so the
        distinction is gone. Answering with a 501 and this explanation beats a 404
        from the proxy, which reads as "wrong URL".
        """
        _entry(map_id)
        raise HTTPException(
            status_code=501,
            detail=(
                "Offline localization was removed with the standalone SQLite index "
                "(ADR 0002). Retrieval now goes through pgvector HNSW — use "
                "/object-search/localize instead."
            ),
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dev-only localize service: the pure-Python stand-in for the Django "
            "object-search API. Requires the mirrored online service for embeddings."
        )
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Toolbox config file."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--ann-base-url",
        default=DEFAULT_ANN_BASE_URL,
        help=f"Mirrored online service (default: {DEFAULT_ANN_BASE_URL}).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--cors", action="store_true", help="Allow all origins (for the toolbox UI)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    state.maps = load_map_entries(args.config)
    state.ann_base_url = args.ann_base_url
    state.timeout_s = args.timeout
    logger.info(
        "Serving %d map(s): %s", len(state.maps), ", ".join(sorted(state.maps)) or "-"
    )

    app = create_app()
    if args.cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
