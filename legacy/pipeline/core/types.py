from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# Geometry mode type alias (moved from pipeline.offline.build_index and
# ingest.equirect_extract)
GeometryMode = Literal["cubemap", "horizon"]

# Default cutout ID stride (moved from pipeline.offline.schema.cutout_schema)
DEFAULT_ID_STRIDE: int = 1024
UNRESOLVED_LEVEL_SENTINEL: int = np.iinfo(np.int32).min

# Default on-disk name for the SQLite object-search index under a map directory.
OBJECT_SEARCH_INDEX_DB_FILENAME = "object-search.db"

# (id, cosine_similarity score)
ResultRow = Tuple[str, float]


@dataclass
class ObjectSearchResult:
    results: List[ResultRow]
    router_object_type: Optional[str]
    time_router_ms: int
    time_embedding_ms: int
    time_retrieval_ms: int


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference throughput (batch processing, worker threads)."""

    batch_size: int = 32
    cutout_workers: int = 2
    cutout_prefetch: int = 8
    max_detections_per_image: Optional[int] = None
    object_crop_batch_size: Optional[int] = None


@dataclass(frozen=True)
class CutoutGeometryConfig:
    """Configuration for equirectangular cutout extraction geometry."""

    geometry: GeometryMode = "cubemap"
    id_stride: int = DEFAULT_ID_STRIDE
    cubemap_face_size: int = 512
    cubemap_fov_deg: float = 90.0
    horizon_nb_of_cuts: int = 12
    horizon_fov: float = 60.0
    horizon_inclination: float = 0.0
    horizon_width: int = 720
    horizon_height: int = 720


@dataclass(frozen=True)
class LocalizationConfig:
    """Configuration for 3D object localization and clustering."""

    enabled: bool = True
    level_filter: Optional[int] = None
    depth_dir: str = "depths"
    clustering_eps_m: float = 1.5
    min_depth_m: float = 0.5
    max_depth_m: float = 50.0
    embedding_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class KeyframeFilterConfig:
    """Configuration for pre-index keyframe filtering."""

    level_filter: Optional[int] = None
    polygon_geojson_path: Optional[Path] = None


@dataclass(frozen=True)
class OfflineBuildConfig:
    """Configuration loaded from the offline build YAML."""

    device: Optional[str] = None
    skip_objects: Optional[bool] = None
    inference: Optional[InferenceConfig] = None
    geometry: Optional[CutoutGeometryConfig] = None
    localization: Optional[LocalizationConfig] = None
    visual_refinement: Optional["VisualRefinementConfig"] = None
    ocr_refinement: Optional["OcrRefinementConfig"] = None


@dataclass(frozen=True)
class VisualRefinementConfig:
    """Configuration for DINO visual cluster refinement."""

    enabled: bool = False
    repo: str = "facebookresearch/dinov2"
    model: str = "dinov2_vitb14"
    image_size: int = 518
    batch_size: int = 16
    similarity_threshold: float = 0.35
    min_cluster_observations: int = 3
    min_component_observations: int = 2


@dataclass(frozen=True)
class OcrRefinementConfig:
    """Configuration for OCR-guided cluster refinement."""

    enabled: bool = False
    lightweight_first: bool = False
    lightweight_lang: str = "fr"
    lightweight_preprocess: str = "raw"
    lightweight_min_score: float = 0.7
    lightweight_single_letter_min_score: float = 0.85
    lightweight_consensus_min_support_fraction: float = 0.6
    lightweight_consensus_min_margin: int = 1
    vl_fallback: bool = True
    textness_threshold: float = 0.02
    min_anchor_observations: int = 2
    min_anchor_keyframes: int = 2
    assignment_margin_m: float = 0.5
    preprocess: str = "upscale4_autocontrast"
    max_candidates: Optional[int] = None
    max_candidates_per_cluster: Optional[int] = None
    batch_size: int = 4
    max_new_tokens: int = 32


@dataclass
class ObjectSearchIndexMetadata:
    """JSON metadata stored in the object-search index DB (params table).

    Describes offline extraction geometry, model snapshots, build parameters,
    and provenance.
    """

    schema_version: int = 3
    projection_dim: int = 0
    created_utc: str = ""
    object_detector_prompt: str = ""
    cutout_count: int = 0
    object_count: int = 0
    # Optional provenance
    source_images_dir: Optional[str] = None
    notes: str = ""
    # Offline extraction (v2+)
    source: str = "equirect360"
    geometry: str = "cubemap"
    id_stride: int = 1024
    cutout_id_encoding: str = "cutout_id = keyframe_id * id_stride + local_index"
    cubemap_face_size: Optional[int] = None
    cubemap_fov_deg: Optional[float] = None
    horizon_nb_of_cuts: Optional[int] = None
    horizon_fov: Optional[float] = None
    horizon_inclination: Optional[float] = None
    horizon_width: Optional[int] = None
    horizon_height: Optional[int] = None
    # Resolved GroundingDINO + post-process snapshot (JSON string) for reproducibility
    gdino_params_json: str = ""
    # Resolved non-model build parameters that affect checkpoint compatibility.
    build_params_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ObjectSearchIndexMetadata:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> ObjectSearchIndexMetadata:
        return cls.from_dict(json.loads(payload))


def default_created_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LoadedIndex:
    """In-memory object-search index (loaded from the SQLite object-search index DB)."""

    metadata: ObjectSearchIndexMetadata
    cutout_embeddings: np.ndarray
    cutout_ids: np.ndarray
    cutout_keyframe_ids: np.ndarray
    cutout_center_xy: np.ndarray
    cutout_rotation_cutout_to_equirect: np.ndarray
    object_embeddings: np.ndarray
    object_keyframe_ids: np.ndarray
    object_cutout_ids: np.ndarray
    object_bboxes: np.ndarray
    object_processed_cutout_ids: Optional[np.ndarray] = None
    # object_idx (PK) per row, in row order — used to map pgvector ids → row
    # positions. Empty when objects are absent. May be empty embeddings when the
    # index is loaded in pgvector mode (embeddings live in Postgres).
    object_ids: Optional[np.ndarray] = None

    # 3D localization (optional; may be absent in older index versions)
    object_positions_keyframe: Optional[np.ndarray] = (
        None  # [N, 3] XYZ in keyframe camera frame (OpenCV)
    )
    object_positions_local: Optional[np.ndarray] = (
        None  # [N, 3] XYZ in local metric map frame (ENU)
    )
    object_positions_world: Optional[np.ndarray] = (
        None  # [N, 3] geographic [lat, lon, alt] float32
    )
    object_depths: Optional[np.ndarray] = None  # [N] sampled depth values
    object_localization_valid: Optional[np.ndarray] = None  # [N] bool mask
    object_cluster_ids: Optional[np.ndarray] = None  # [N] cluster assignment
    object_detection_levels: Optional[np.ndarray] = (
        None  # [N] keyframe/cutout level used for clustering
    )
    # Bounding box spherical coordinates (optional; new in this schema version)
    object_bbox_spherical: Optional[np.ndarray] = (
        None  # [N, 4] [theta, phi, fov_x, fov_y] radians
    )
    cluster_centroids_world: Optional[np.ndarray] = None  # [K, 3] centroid XYZ
    cluster_centroids_geo: Optional[np.ndarray] = None  # [K, 3] lat/lon/alt
    cluster_observation_counts: Optional[np.ndarray] = None  # [K] int
    cluster_confidence: Optional[np.ndarray] = None  # [K] float
    cluster_levels: Optional[np.ndarray] = None  # [K] int
    cluster_cutout_cluster_ids: Optional[np.ndarray] = (
        None  # [M] cluster id per deduplicated cluster-cutout row
    )
    cluster_cutout_ids: Optional[np.ndarray] = None  # [M] unique cutout id per cluster
    cluster_cutout_keyframe_ids: Optional[np.ndarray] = (
        None  # [M] keyframe id for the cutout
    )
    cluster_cutout_levels: Optional[np.ndarray] = (
        None  # [M] resolved level for the cutout in the cluster
    )
    cluster_cutout_observation_counts: Optional[np.ndarray] = (
        None  # [M] detections from the cutout in the cluster
    )
    object_visual_similarity_scores: Optional[np.ndarray] = (
        None  # [N] DINO similarity score
    )
    object_visual_candidate_mask: Optional[np.ndarray] = (
        None  # [N] visual candidate mask
    )
    object_visual_assigned_mask: Optional[np.ndarray] = (
        None  # [N] assigned by visual refinement
    )
    object_textness_scores: Optional[np.ndarray] = (
        None  # [N] OCR textness prompt-margin score
    )
    object_ocr_texts: Optional[np.ndarray] = None  # [N] raw OCR text
    object_ocr_tokens: Optional[np.ndarray] = (
        None  # [N] normalized OCR tokens, space-separated
    )
    object_ocr_keys: Optional[np.ndarray] = None  # [N] normalized OCR identity key
    object_ocr_candidate_mask: Optional[np.ndarray] = (
        None  # [N] candidate mask sent to OCR
    )
    object_ocr_assigned_mask: Optional[np.ndarray] = (
        None  # [N] assigned by OCR refinement
    )
    object_ocr_source: Optional[np.ndarray] = None  # [N] OCR source enum
    cluster_ocr_texts: Optional[np.ndarray] = (
        None  # [K] representative OCR text per cluster
    )
    cluster_ocr_tokens: Optional[np.ndarray] = (
        None  # [K] representative normalized OCR tokens
    )
    cluster_ocr_keys: Optional[np.ndarray] = (
        None  # [K] representative normalized OCR key per cluster
    )
    cluster_ocr_observation_counts: Optional[np.ndarray] = (
        None  # [K] support count for cluster OCR key
    )
    cluster_ocr_source: Optional[np.ndarray] = None  # [K] OCR source enum
    # Hybrid-pipeline fields (from YOLO-World + GDINO combined detection)
    object_labels: Optional[np.ndarray] = (
        None  # [N] '<U256' YOLO category or 'gdino_venue'
    )
    object_sources: Optional[np.ndarray] = None  # [N] '<U16'  'yolo' or 'gdino'
