from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field


class ObjectSearchRequest(BaseModel):
    """Request body aligned with GeoPose `ObjectSearchRequest` (sandbox)."""

    text: str = Field(
        ...,
        description="Text to search for",
        min_length=1,
        max_length=1000,
    )
    num_results: int = Field(
        100,
        description="Number of results to return",
        ge=1,
        le=1000,
    )
    search_type: str = Field(
        "cutout",
        description="Routing: 'cutout', 'object', or 'auto' (LLM router)",
        pattern="^(auto|cutout|object)$",
    )


class ObjectLocalizationRequest(ObjectSearchRequest):
    """Request body for localization from object detections."""

    min_similarity: float = Field(
        0.2,
        description=(
            "Minimum cluster-level text-to-object similarity before"
            " localization ranking"
        ),
        ge=-1,
        le=1,
    )
    max_observations_per_cluster: int = Field(
        10,
        description="Maximum observations returned per localized cluster",
        ge=1,
        le=10000,
    )


class OnlineObjectLocalizationRequest(ObjectLocalizationRequest):
    """Request body for request-time localization from object detections."""

    candidate_count: int = Field(
        1000,
        description="Number of best-matching object detections to localize online",
        ge=1,
        le=10000,
    )
    depth_dir: str = Field(
        "depths",
        description="Depth zarr directory under the map path",
        min_length=1,
        max_length=200,
    )
    clustering_eps_m: float = Field(
        2.0,
        description="Spatial neighbor radius in meters for online clustering",
        gt=0,
        le=100,
    )
    min_depth_m: float = Field(
        0.5,
        description="Minimum valid depth in meters",
        ge=0,
        le=1000,
    )
    max_depth_m: float = Field(
        50.0,
        description="Maximum valid depth in meters",
        gt=0,
        le=1000,
    )
    embedding_similarity_threshold: float = Field(
        0.85,
        description=(
            "Minimum object embedding cosine similarity for single_linkage"
            " clustering only; ignored when clustering_method is leader_canopy"
        ),
        ge=-1,
        le=1,
    )
    min_keyframes_per_cluster: int = Field(
        2,
        description=(
            "Minimum distinct keyframes per spatial cluster during online"
            " localization"
        ),
        ge=1,
        le=100,
    )
    face_dedup_iou: float = Field(
        0.5,
        description=(
            "Per-cutout 2D bbox IoU threshold for deduplicating overlapping proposals "
            "before candidate_count top-K (0 disables dedup)"
        ),
        ge=0,
        le=1,
    )
    clustering_method: Literal["leader_canopy", "single_linkage"] = Field(
        "leader_canopy",
        description=(
            "Online 3D clustering: leader_canopy (similarity-seeded, anti-chaining) or "
            "single_linkage (legacy union-find A/B)"
        ),
    )
    use_stored_positions: bool = Field(
        True,
        description=(
            "If True and the index contains pre-computed per-object ENU "
            "positions, re-cluster those instead of recomputing 3D positions "
            "from depth maps at request time. Much faster; depth-map / georef "
            "files are not read. Falls back to depth-based projection if the "
            "index has no stored positions."
        ),
    )
    robust_centroid: bool = Field(
        False,
        description=(
            "If True, replace each cluster centroid with the geometric (L1) "
            "median of its member positions. Outlier-robust; recommended when "
            "a cluster's reported position is being dragged off by a few bad "
            "observations even though the cluster's members are mostly right."
        ),
    )
    include_debug: bool = Field(
        False,
        description=(
            "If True, include the local-sandbox debug payload (per-keyframe "
            "camera lat/lon/alt/level for contributing keyframes). Off by "
            "default: it issues a georef.db lookup per contributing keyframe "
            "and can add seconds to the response on large maps."
        ),
    )


class ObjectObservation(BaseModel):
    """Individual object observation within a cluster."""

    object_idx: int = Field(
        ...,
        description=(
            "Object row index in the object-search index (SQLite `object` table)"
        ),
    )
    cutout_id: str = Field(
        ..., description="ID of the cutout containing this observation"
    )
    keyframe_id: str = Field(..., description="ID of the keyframe")
    coordinates: Tuple[float, float, float] | None = Field(
        None,
        description=(
            "Observation geographic coordinates [lat, lng, alt], " "None if unavailable"
        ),
    )
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="Bounding box (x1, y1, x2, y2) in cutout pixel coordinates"
    )
    similarity_score: float = Field(..., description="Text-to-embedding similarity")
    quaternion: List[float] | None = Field(
        None,
        description=(
            "Observation view orientation quaternion [w, x, y, z] in"
            " wemap-vision-tools EUS convention"
        ),
        min_length=4,
        max_length=4,
    )
    heading: float | None = Field(
        None,
        description="Object bbox-center heading relative to north, in degrees",
    )


class ObjectLocation(BaseModel):
    """Geographic location of a detected object cluster."""

    coordinates: Tuple[float, float, float] = Field(
        ...,
        description="Cluster geographic coordinates [lat, lng, alt]",
    )
    confidence: float = Field(..., description="Localization confidence (0-1)")
    observation_count: int = Field(..., description="Number of observations in cluster")
    similarity_score: float = Field(
        ..., description="Text-to-embedding similarity (0-1)"
    )
    match_score: float = Field(
        ...,
        description="Combined query/spatial cluster ranking score (0-1)",
    )
    level: int | None = Field(
        None, description="Indoor level index, None if outdoors or undetermined"
    )
    keyframe_ids: List[str] = Field(
        default_factory=list,
        description="IDs of keyframes where this object was observed",
    )
    observations: List[ObjectObservation] = Field(
        default_factory=list,
        description=(
            "Individual object observations in this cluster, sorted by similarity"
        ),
    )


class KeyframeDebugMetadata(BaseModel):
    """Local UI/debug metadata for drawing contributing keyframes."""

    lat: float | None = Field(None, description="Keyframe latitude")
    lon: float | None = Field(None, description="Keyframe longitude")
    alt: float | None = Field(None, description="Keyframe altitude")
    level: str | None = Field(None, description="Indoor level identifier")


class ObjectLocalizationDebug(BaseModel):
    """Additional local-debug payload for the sandbox UI."""

    keyframes: Dict[str, KeyframeDebugMetadata] = Field(
        default_factory=dict,
        description="Keyframe camera metadata keyed by keyframe ID",
    )


class ObjectLocalizationResponse(BaseModel):
    """Response for object localization endpoint."""

    localizations: List[ObjectLocation] = Field(
        default_factory=list,
        description="List of object locations matching the query",
    )
    time_embedding_ms: int = Field(
        ..., description="Time spent computing text embedding (ms)"
    )
    time_retrieval_ms: int = Field(
        ..., description="Time spent retrieving and clustering results (ms)"
    )
    debug: ObjectLocalizationDebug = Field(
        default_factory=ObjectLocalizationDebug,
        description="Local sandbox UI/debug payload",
    )


class EncodeTextRequest(BaseModel):
    """Request body for text encoding endpoint."""

    texts: List[str] = Field(
        ...,
        description="List of text prompts to encode",
        min_length=1,
    )
