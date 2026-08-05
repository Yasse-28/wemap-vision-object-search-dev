"""
FastAPI entrypoint for file-backed object search (multi-map, GeoPose-style config).

Run from ``object-search`` with PYTHONPATH set:

  cd object-search && PYTHONPATH=. python -m pipeline.online.app \\
    --config_file_path /path/to/config.json --host 0.0.0.0 --port 8090

Config JSON must contain a ``maps`` array. Each map has ``id``; optional ``path``
relative to the config file's parent directory. If ``path`` is omitted,
``{config.parent}/maps/{id}`` is used.

Per map directory must contain ``object-search.db`` and uses
``object-search-history.db`` in the same folder.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

# Repo root (directory that contains `pipeline`)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from starlette.datastructures import UploadFile as StarletteUploadFile  # noqa: E402

from pipeline.core.models.metaclip import get_shared_metaclip  # noqa: E402
from pipeline.online.localize_3d import OnlineLocalizationParams  # noqa: E402
from pipeline.online.map_registry import ObjectSearchMapRegistry  # noqa: E402
from pipeline.online.request_models import (  # noqa: E402
    EncodeTextRequest,
    ObjectLocalizationRequest,
    ObjectLocalizationResponse,
    ObjectSearchRequest,
    OnlineObjectLocalizationRequest,
)
from pipeline.online.search_service import IndexNotReadyError  # noqa: E402

_SUPPORTED_IMAGE_CONTENT_TYPES = ("image/jpeg", "image/png")


def _validate_localize_json_request(body: object) -> ObjectLocalizationRequest:
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")
    if "image" in body or "file" in body:
        raise HTTPException(
            status_code=400, detail="Image queries must use multipart/form-data"
        )
    try:
        return ObjectLocalizationRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e


def _validate_image_localize_params(
    *,
    num_results: object,
    min_similarity: object,
    max_observations_per_cluster: object = None,
) -> ObjectLocalizationRequest:
    try:
        kwargs: dict[str, object] = dict(
            text="[image]",
            num_results=100 if num_results in (None, "") else num_results,
            min_similarity=0.2 if min_similarity in (None, "") else min_similarity,
        )
        if max_observations_per_cluster not in (None, ""):
            kwargs["max_observations_per_cluster"] = max_observations_per_cluster
        return ObjectLocalizationRequest(**kwargs)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e


async def _read_uploaded_image(file: UploadFile | StarletteUploadFile) -> Image.Image:
    if file.content_type not in _SUPPORTED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image type: {file.content_type}. "
                "Only JPEG and PNG are supported."
            ),
        )

    content = await file.read()
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        ) from e


_ONLINE_LOCALIZE_FORM_FIELDS = (
    "num_results",
    "min_similarity",
    "max_observations_per_cluster",
    "candidate_count",
    "depth_dir",
    "clustering_eps_m",
    "min_depth_m",
    "max_depth_m",
    "embedding_similarity_threshold",
    "min_keyframes_per_cluster",
    "face_dedup_iou",
    "clustering_method",
    "use_stored_positions",
    "robust_centroid",
)


def _validate_online_image_localize_params(
    *, form_data: dict[str, object]
) -> OnlineObjectLocalizationRequest:
    kwargs: dict[str, object] = {"text": "[image]"}
    for field in _ONLINE_LOCALIZE_FORM_FIELDS:
        value = form_data.get(field)
        if value not in (None, ""):
            kwargs[field] = value
    try:
        return OnlineObjectLocalizationRequest(**kwargs)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e


async def _parse_online_image_localize_request(
    request: Request,
) -> tuple[OnlineObjectLocalizationRequest, Image.Image, str]:
    form = await request.form()
    text = form.get("text")
    if isinstance(text, str) and text.strip():
        raise HTTPException(
            status_code=400, detail="Provide either text or image, not both"
        )

    uploads = [
        value
        for field_name in ("image", "file")
        for value in form.getlist(field_name)
        if isinstance(value, (UploadFile, StarletteUploadFile))
    ]
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="multipart/form-data requests must include an image file",
        )
    if len(uploads) > 1:
        raise HTTPException(
            status_code=400, detail="Only one image file can be localized at a time"
        )

    form_data = {key: form.get(key) for key in _ONLINE_LOCALIZE_FORM_FIELDS}
    req = _validate_online_image_localize_params(form_data=form_data)
    file = uploads[0]
    image = await _read_uploaded_image(file)
    filename = file.filename or "upload"
    return req, image, f"[image] {filename}"


def _online_localization_params_from_req(
    req: OnlineObjectLocalizationRequest,
) -> OnlineLocalizationParams:
    return OnlineLocalizationParams(
        depth_dir=req.depth_dir,
        candidate_count=req.candidate_count,
        max_observations_per_cluster=req.max_observations_per_cluster,
        clustering_eps_m=req.clustering_eps_m,
        min_depth_m=req.min_depth_m,
        max_depth_m=req.max_depth_m,
        embedding_similarity_threshold=req.embedding_similarity_threshold,
        min_similarity=req.min_similarity,
        min_keyframes_per_cluster=req.min_keyframes_per_cluster,
        face_dedup_iou=req.face_dedup_iou,
        clustering_method=req.clustering_method,
        use_stored_positions=req.use_stored_positions,
        robust_centroid=req.robust_centroid,
    )


async def _parse_image_localize_request(
    request: Request,
) -> tuple[ObjectLocalizationRequest, Image.Image, str]:
    form = await request.form()
    text = form.get("text")
    if isinstance(text, str) and text.strip():
        raise HTTPException(
            status_code=400, detail="Provide either text or image, not both"
        )

    uploads = [
        value
        for field_name in ("image", "file")
        for value in form.getlist(field_name)
        if isinstance(value, (UploadFile, StarletteUploadFile))
    ]
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="multipart/form-data requests must include an image file",
        )
    if len(uploads) > 1:
        raise HTTPException(
            status_code=400, detail="Only one image file can be localized at a time"
        )

    req = _validate_image_localize_params(
        num_results=form.get("num_results"),
        min_similarity=form.get("min_similarity"),
        max_observations_per_cluster=form.get("max_observations_per_cluster"),
    )
    file = uploads[0]
    image = await _read_uploaded_image(file)
    filename = file.filename or "upload"
    return req, image, f"[image] {filename}"


def _store_history_request(
    request: Request,
    map_id: str,
    *,
    text: str,
    search_type: str,
    enforced: bool,
    timestamp: int,
    time_router_ms: int,
    time_embedding_ms: int,
    time_retrieval_ms: int,
) -> None:
    history = request.app.state.registry.get_history(map_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Map {map_id!r} not configured")
    history.store_request(
        text,
        search_type,
        enforced,
        timestamp,
        time_router_ms,
        time_embedding_ms,
        time_retrieval_ms,
    )


def create_app(config_path: Path, *, enable_cors: bool = False) -> FastAPI:
    config_path = config_path.resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metaclip = get_shared_metaclip(device)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = ObjectSearchMapRegistry.from_config_file(config_path, device=device)
        app.state.registry = registry
        registry.start_background_loading()
        yield
        registry.close()

    app = FastAPI(title="Standalone object search", lifespan=lifespan)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        reg = getattr(request.app.state, "registry", None)
        if reg is None:
            return {"status": "starting", "maps": {}}
        maps = {map_id: reg.get_map_state(map_id) for map_id in reg.map_ids()}
        all_ready = all(s == "ready" for s in maps.values())
        any_error = any(s == "error" for s in maps.values())
        status = "ready" if all_ready else ("degraded" if any_error else "loading")
        return {"status": status, "maps": maps}

    def _get_service_or_raise(reg: ObjectSearchMapRegistry, map_id: str) -> Any:
        service = reg.get_service(map_id)
        if service is not None:
            return service
        state = reg.get_map_state(map_id)
        if state is None:
            raise HTTPException(
                status_code=404, detail=f"Map {map_id!r} not configured"
            )
        if state == "loading":
            raise HTTPException(
                status_code=503,
                detail=f"Map {map_id!r} is still loading, retry shortly",
            )
        raise HTTPException(status_code=503, detail=f"Map {map_id!r} failed to load")

    @app.post("/{map_id}/object-search/text")
    def object_search_text(
        request: Request, map_id: str, req: ObjectSearchRequest
    ) -> Any:
        reg = request.app.state.registry
        service = _get_service_or_raise(reg, map_id)

        ts = int(time.time())
        try:
            result = service.search(
                req.text, req.num_results, search_type=req.search_type
            )
        except IndexNotReadyError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        enforced = req.search_type != "auto"
        actual_type = (
            result.router_object_type
            if result.router_object_type is not None
            else req.search_type
        )
        _store_history_request(
            request,
            map_id,
            text=req.text,
            search_type=actual_type,
            enforced=enforced,
            timestamp=ts,
            time_router_ms=result.time_router_ms,
            time_embedding_ms=result.time_embedding_ms,
            time_retrieval_ms=result.time_retrieval_ms,
        )

        if result.router_object_type is not None:
            return JSONResponse(
                content={
                    "results": result.results,
                    "router_object_type": result.router_object_type,
                }
            )
        return JSONResponse(content=result.results)

    @app.post(
        "/{map_id}/object-search/text/localize-offline",
        response_model=ObjectLocalizationResponse,
        include_in_schema=False,
    )
    @app.post(
        "/{map_id}/object-search/localize-offline",
        response_model=ObjectLocalizationResponse,
    )
    async def object_search_localize(request: Request, map_id: str) -> Any:
        """
        Return 3D geographic positions of objects matching a text or image query.

        Searches object embeddings (not cutouts) and returns cluster-level
        geographic locations (lat/lon/alt) for the best-matching objects.
        Requires that the index was built with 3D localization data.
        """
        reg = request.app.state.registry
        service = _get_service_or_raise(reg, map_id)

        ts = int(time.time())
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        try:
            if content_type == "application/json":
                try:
                    req = _validate_localize_json_request(await request.json())
                except ValueError as e:
                    raise HTTPException(
                        status_code=400, detail="Invalid JSON body"
                    ) from e
                result = service.search_objects_localized(
                    req.text,
                    req.num_results,
                    min_similarity=req.min_similarity,
                    max_observations_per_cluster=req.max_observations_per_cluster,
                )
                history_text = req.text
            elif content_type == "multipart/form-data":
                req, image, history_text = await _parse_image_localize_request(request)
                result = service.search_objects_localized_by_image(
                    image,
                    req.num_results,
                    min_similarity=req.min_similarity,
                    log_query=history_text,
                    max_observations_per_cluster=req.max_observations_per_cluster,
                )
            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Use application/json for text queries or"
                        " multipart/form-data for image queries"
                    ),
                )
        except IndexNotReadyError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        _store_history_request(
            request,
            map_id,
            text=history_text,
            search_type="object",
            enforced=True,
            timestamp=ts,
            time_router_ms=0,
            time_embedding_ms=result.time_embedding_ms,
            time_retrieval_ms=result.time_retrieval_ms,
        )
        return result

    @app.post(
        "/{map_id}/object-search/text/localize",
        response_model=ObjectLocalizationResponse,
        include_in_schema=False,
    )
    @app.post(
        "/{map_id}/object-search/localize",
        response_model=ObjectLocalizationResponse,
    )
    async def object_search_localize_online(request: Request, map_id: str) -> Any:
        """
        Return request-time 3D geographic positions for objects matching the
        text **or image** query.

        Unlike `/localize-offline`, this does not require pre-baked cluster
        arrays in the index — it re-clusters stored per-object ENU positions
        (or falls back to depth-map reprojection if those are absent).
        Accepts JSON for text queries or multipart/form-data for image queries.
        """
        reg = request.app.state.registry
        service = _get_service_or_raise(reg, map_id)

        ts = int(time.time())
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()

        try:
            if content_type == "application/json":
                try:
                    body = await request.json()
                except ValueError as e:
                    raise HTTPException(
                        status_code=400, detail="Invalid JSON body"
                    ) from e
                try:
                    req = (
                        OnlineObjectLocalizationRequest(**body)
                        if isinstance(body, dict)
                        else None
                    )
                except ValidationError as e:
                    raise HTTPException(status_code=422, detail=e.errors()) from e
                if req is None:
                    raise HTTPException(
                        status_code=422, detail="JSON body must be an object"
                    )
                if req.max_depth_m <= req.min_depth_m:
                    raise HTTPException(
                        status_code=422,
                        detail="max_depth_m must be greater than min_depth_m",
                    )
                history_text = req.text
                result = service.search_objects_localized_online(
                    req.text,
                    req.num_results,
                    localization_params=_online_localization_params_from_req(req),
                    include_debug=req.include_debug,
                )
            elif content_type == "multipart/form-data":
                req, image, history_text = await _parse_online_image_localize_request(
                    request
                )
                if req.max_depth_m <= req.min_depth_m:
                    raise HTTPException(
                        status_code=422,
                        detail="max_depth_m must be greater than min_depth_m",
                    )
                result = service.search_objects_localized_online_by_image(
                    image,
                    req.num_results,
                    localization_params=_online_localization_params_from_req(req),
                    log_query=history_text,
                    include_debug=req.include_debug,
                )
            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Use application/json for text queries or"
                        " multipart/form-data for image queries"
                    ),
                )
        except IndexNotReadyError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        _store_history_request(
            request,
            map_id,
            text=history_text,
            search_type="object",
            enforced=True,
            timestamp=ts,
            time_router_ms=0,
            time_embedding_ms=result.time_embedding_ms,
            time_retrieval_ms=result.time_retrieval_ms,
        )
        return result

    @app.post("/object-search/encode-text")
    def encode_text(req: EncodeTextRequest) -> Any:
        """Encode text prompts into MetaCLIP embeddings."""
        result = {}
        for text in req.texts:
            features = metaclip.get_text_features(text)
            if features.dim() > 1:
                features = features[0]
            embedding = features.detach().float().cpu().numpy().ravel().tolist()
            result[text] = embedding
        return result

    @app.post("/object-search/encode-image")
    async def encode_image(files: list[UploadFile] = File(...)) -> Any:
        """Encode images into MetaCLIP embeddings."""
        images = []
        filenames = []
        for file in files:
            images.append(await _read_uploaded_image(file))
            filenames.append(file.filename or "upload")

        features = metaclip.get_image_features(images)
        result = {}
        for i, filename in enumerate(filenames):
            embedding = features[i].detach().float().cpu().numpy().ravel().tolist()
            result[filename] = embedding
        return result

    return app


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone object-search API (config-driven maps)"
    )
    p.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="Path to JSON config listing maps (GeoPose-style).",
    )
    p.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    p.add_argument("--port", type=int, default=45678, help="Bind port.")
    p.add_argument(
        "--cors",
        action="store_true",
        help="Enable permissive CORS (same spirit as GeoPose --cors).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    app = create_app(
        Path(args.config_file_path),
        enable_cors=args.cors,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
