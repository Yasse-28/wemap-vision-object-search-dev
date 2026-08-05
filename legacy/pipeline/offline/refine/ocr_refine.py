"""OCR-guided refinement for localized object clusters.

This module is optional and intentionally isolated from the base localization
path because OCR models add heavy optional dependencies.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Sequence, overload

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

from pipeline.core.logging import logger
from pipeline.core.models.ocr.identity import (
    OcrIdentity,
    ocr_compatible,
    ocr_identity,
    ocr_key_string,
    ocr_token_string,
)
from pipeline.core.models.ocr.paddle_lightweight import LightweightPaddleOcrRecognizer
from pipeline.core.models.ocr.paddle_utils import default_lightweight_paddle_device
from pipeline.core.models.ocr.paddle_utils import (
    release_ocr_memory as _release_ocr_memory,
)
from pipeline.core.models.ocr.paddle_vl import PaddleOcrVlRecognizer

OCR_SOURCE_NONE = 0
OCR_SOURCE_LIGHTWEIGHT_PPOCR = 1
OCR_SOURCE_PADDLEOCR_VL = 2

TEXTNESS_POSITIVE_PROMPTS = (
    "a sign containing readable text or numbers",
    "a display board with text",
    "a label with numbers",
    "a platform sign with numbers",
)
TEXTNESS_NEGATIVE_PROMPTS = (
    "an object without text",
    "a plain object with no writing",
    "an icon or pictogram without readable text",
    "a wall or surface without text",
)


@dataclass
class OcrRefinementResult:
    cluster_ids: np.ndarray
    textness_scores: np.ndarray
    ocr_texts: np.ndarray
    ocr_tokens: np.ndarray
    ocr_keys: np.ndarray
    ocr_candidate_mask: np.ndarray
    ocr_assigned_mask: np.ndarray
    ocr_source: np.ndarray


def _compact_string_array(values: Sequence[str], *, max_chars: int) -> np.ndarray:
    cleaned = [str(value)[:max_chars] for value in values]
    width = max(1, max((len(value) for value in cleaned), default=0))
    return np.asarray(cleaned, dtype=f"<U{min(max_chars, width)}")


def _make_ocr_refinement_result(
    *,
    cluster_ids: np.ndarray,
    textness_scores: np.ndarray,
    ocr_texts: Sequence[str],
    ocr_tokens: Sequence[str],
    ocr_keys: Sequence[str],
    ocr_candidate_mask: np.ndarray,
    ocr_assigned_mask: np.ndarray,
    ocr_source: np.ndarray,
) -> OcrRefinementResult:
    return OcrRefinementResult(
        cluster_ids=cluster_ids,
        textness_scores=np.asarray(textness_scores, dtype=np.float32),
        ocr_texts=_compact_string_array(ocr_texts, max_chars=512),
        ocr_tokens=_compact_string_array(ocr_tokens, max_chars=512),
        ocr_keys=_compact_string_array(ocr_keys, max_chars=256),
        ocr_candidate_mask=np.asarray(ocr_candidate_mask, dtype=bool),
        ocr_assigned_mask=np.asarray(ocr_assigned_mask, dtype=bool),
        ocr_source=np.asarray(ocr_source, dtype=np.int16),
    )


@dataclass(frozen=True)
class OcrConsensus:
    cluster_id: int
    key: str
    support_count: int
    total_count: int
    keyframe_count: int
    margin: int
    support_fraction: float
    accepted: bool


def preprocess_for_ocr(
    image: Image.Image, name: str = "upscale4_autocontrast"
) -> Image.Image:
    image = image.convert("RGB")
    if name == "raw":
        return image
    if name == "upscale2":
        return image.resize(
            (image.width * 2, image.height * 2), Image.Resampling.BICUBIC
        )
    if name == "upscale4_autocontrast":
        return _enhance(image, scale=4)
    if name == "upscale6_autocontrast":
        return _enhance(image, scale=6)
    raise ValueError(f"Unknown OCR preprocessor: {name}")


def _enhance(image: Image.Image, *, scale: int) -> Image.Image:
    image = image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.BICUBIC
    )
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = ImageEnhance.Sharpness(image).enhance(1.6)
    return image


def compute_textness_scores(
    *,
    clip_model: Any,
    object_embeddings: np.ndarray,
) -> np.ndarray:
    prompts = [*TEXTNESS_POSITIVE_PROMPTS, *TEXTNESS_NEGATIVE_PROMPTS]
    text_features = clip_model.get_text_features(prompts).detach().float().cpu().numpy()
    object_features = np.asarray(object_embeddings, dtype=np.float32)
    scores = object_features @ text_features.T
    pos = scores[:, : len(TEXTNESS_POSITIVE_PROMPTS)].max(axis=1)
    neg = scores[:, len(TEXTNESS_POSITIVE_PROMPTS) :].max(axis=1)
    return np.asarray((pos - neg).astype(np.float32))


def extract_object_crops(
    *,
    get_cutout_image: Callable[[int], Image.Image | None],
    object_cutout_ids: np.ndarray,
    object_bboxes: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> list[Image.Image | None]:
    crops: list[Image.Image | None] = []
    mask = (
        np.asarray(candidate_mask, dtype=bool)
        if candidate_mask is not None
        else np.ones(len(object_cutout_ids), dtype=bool)
    )
    for cutout_id, bbox in zip(object_cutout_ids, object_bboxes):
        idx = len(crops)
        if idx >= len(mask) or not bool(mask[idx]):
            crops.append(None)
            continue
        image = get_cutout_image(int(cutout_id))
        if image is None:
            crops.append(None)
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
        x0 = max(0, min(image.width - 1, x0))
        y0 = max(0, min(image.height - 1, y0))
        x1 = max(x0 + 1, min(image.width, x1))
        y1 = max(y0 + 1, min(image.height, y1))
        crops.append(image.crop((x0, y0, x1, y1)))
    return crops


class LazyObjectCrops(Sequence[Image.Image | None]):
    """Crop object images on demand instead of materializing every crop in RAM."""

    def __init__(
        self,
        *,
        get_cutout_image: Callable[[int], Image.Image | None],
        object_cutout_ids: np.ndarray,
        object_bboxes: np.ndarray,
        candidate_mask: np.ndarray | None = None,
    ) -> None:
        self._get_cutout_image = get_cutout_image
        self._object_cutout_ids = np.asarray(object_cutout_ids, dtype=np.int64).ravel()
        self._object_bboxes = np.asarray(object_bboxes, dtype=np.float32).reshape(-1, 4)
        if candidate_mask is None:
            self._candidate_mask = np.ones(self._object_cutout_ids.shape[0], dtype=bool)
        else:
            self._candidate_mask = np.asarray(candidate_mask, dtype=bool).ravel()

    def __len__(self) -> int:
        return int(self._object_cutout_ids.shape[0])

    def __iter__(self) -> Iterator[Image.Image | None]:
        for idx in range(len(self)):
            yield self[idx]

    @overload
    def __getitem__(self, idx: int) -> Image.Image | None: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[Image.Image | None]: ...

    def __getitem__(
        self, idx: int | slice
    ) -> Image.Image | None | Sequence[Image.Image | None]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if idx >= self._candidate_mask.shape[0] or not bool(self._candidate_mask[idx]):
            return None
        image = self._get_cutout_image(int(self._object_cutout_ids[idx]))
        if image is None:
            return None
        x0, y0, x1, y1 = [int(round(float(v))) for v in self._object_bboxes[idx]]
        x0 = max(0, min(image.width - 1, x0))
        y0 = max(0, min(image.height - 1, y0))
        x1 = max(x0 + 1, min(image.width, x1))
        y1 = max(y0 + 1, min(image.height, y1))
        return image.crop((x0, y0, x1, y1))

    def candidate_indices(self) -> np.ndarray:
        count = min(len(self), int(self._candidate_mask.shape[0]))
        return np.where(self._candidate_mask[:count])[0].astype(np.int64, copy=False)


def _select_ocr_candidates(
    *,
    textness: np.ndarray,
    textness_threshold: float,
    cluster_ids: np.ndarray,
    valid_mask: np.ndarray,
    positions_world: np.ndarray,
    object_keyframe_ids: np.ndarray,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
    max_ocr_candidates_per_cluster: int | None,
    max_ocr_candidates: int | None,
    n_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter and cap candidates; return (candidate_indices, ocr_candidate_mask)."""
    ocr_candidate_mask = (
        valid_mask
        & (cluster_ids >= 0)
        & np.isfinite(positions_world).all(axis=1)
        & (textness >= float(textness_threshold))
    )
    candidate_indices = np.where(ocr_candidate_mask)[0]
    localized_mask = (
        valid_mask & (cluster_ids >= 0) & np.isfinite(positions_world).all(axis=1)
    )
    localized_textness = textness[localized_mask]
    if localized_textness.size > 0:
        logger.info(
            "OCR refinement candidates: %d/%d localized observations at"
            " textness_threshold=%.3f (min=%.3f mean=%.3f max=%.3f)",
            int(candidate_indices.size),
            int(localized_textness.size),
            float(textness_threshold),
            float(localized_textness.min()),
            float(localized_textness.mean()),
            float(localized_textness.max()),
        )
    else:
        logger.info("OCR refinement candidates: 0 localized observations available")
    candidate_indices = _filter_supported_ocr_candidate_clusters(
        candidate_indices=candidate_indices,
        cluster_ids=cluster_ids,
        object_keyframe_ids=object_keyframe_ids,
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
    )
    ocr_candidate_mask = np.zeros(n_objects, dtype=bool)
    ocr_candidate_mask[candidate_indices] = True
    logger.info(
        "OCR refinement retained %d candidates after unsupported-cluster filtering",
        int(candidate_indices.size),
    )
    if (
        max_ocr_candidates_per_cluster is not None
        and max_ocr_candidates_per_cluster > 0
        and candidate_indices.size > 0
    ):
        candidate_indices = _limit_ocr_candidates_per_cluster(
            candidate_indices=candidate_indices,
            cluster_ids=cluster_ids,
            object_keyframe_ids=object_keyframe_ids,
            textness=textness,
            max_candidates_per_cluster=max_ocr_candidates_per_cluster,
        )
        ocr_candidate_mask = np.zeros(n_objects, dtype=bool)
        ocr_candidate_mask[candidate_indices] = True
        logger.info(
            "OCR refinement retained %d candidates after per-cluster cap=%d",
            int(candidate_indices.size),
            int(max_ocr_candidates_per_cluster),
        )
    if (
        max_ocr_candidates is not None
        and max_ocr_candidates > 0
        and candidate_indices.size > max_ocr_candidates
    ):
        order = np.argsort(-textness[candidate_indices])[:max_ocr_candidates]
        candidate_indices = candidate_indices[order]
        ocr_candidate_mask = np.zeros(n_objects, dtype=bool)
        ocr_candidate_mask[candidate_indices] = True
        logger.warning(
            "OCR refinement candidate list capped at %d highest-textness observations",
            int(max_ocr_candidates),
        )
    return candidate_indices, ocr_candidate_mask


class _LightweightOcrPassResult(NamedTuple):
    identities: dict[int, OcrIdentity]
    vl_candidate_indices: np.ndarray
    lightweight_text_regions: dict[int, Image.Image]


def _lightweight_read_crops(
    *,
    candidate_indices: np.ndarray,
    object_crops: Sequence[Image.Image | None],
    recognizer: LightweightPaddleOcrRecognizer,
    lightweight_preprocess: str,
    n_objects: int,
    ocr_texts: list[str],
    ocr_tokens: list[str],
    ocr_keys: list[str],
    ocr_source: np.ndarray,
) -> tuple[dict[int, OcrIdentity], np.ndarray, dict[int, Image.Image]]:
    """Read each candidate crop with lightweight PP-OCR.

    Mutates ocr_texts/tokens/keys/source in-place for accepted reads.

    Returns:
        (identities, lightweight_accepted mask, text regions by object index)
    """
    identities: dict[int, OcrIdentity] = {}
    lightweight_accepted = np.zeros(n_objects, dtype=bool)
    lightweight_text_regions: dict[int, Image.Image] = {}
    attempts = successes = 0
    started_at = time.perf_counter()
    for idx in tqdm(
        candidate_indices.tolist(), desc="Lightweight OCR refinement", unit="crop"
    ):
        crop = object_crops[idx] if idx < len(object_crops) else None
        if crop is None:
            continue
        attempts += 1
        try:
            read = recognizer.read(preprocess_for_ocr(crop, lightweight_preprocess))
        except Exception as exc:
            logger.warning("Lightweight OCR failed for object idx %d: %s", idx, exc)
            continue
        if read.text_region is not None:
            lightweight_text_regions[idx] = read.text_region
        if not read.accepted:
            continue
        ocr_texts[idx] = read.text
        ocr_tokens[idx] = ocr_token_string(read.identity)
        ocr_keys[idx] = ocr_key_string(read.identity)
        identities[idx] = read.identity
        ocr_source[idx] = OCR_SOURCE_LIGHTWEIGHT_PPOCR
        lightweight_accepted[idx] = True
        successes += 1
    logger.info(
        "OCR refinement lightweight PP-OCR inference took %.2fs for %d/%d"
        " accepted reads",
        time.perf_counter() - started_at,
        successes,
        attempts,
    )
    return identities, lightweight_accepted, lightweight_text_regions


def _select_vl_fallback(
    *,
    candidate_indices: np.ndarray,
    cluster_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    ocr_keys: list[str],
    lightweight_accepted: np.ndarray,
    lightweight_text_regions: dict[int, Image.Image],
    min_anchor_observations: int,
    min_anchor_keyframes: int,
    min_support_fraction: float,
    min_margin: int,
    vl_fallback: bool,
) -> tuple[np.ndarray, dict[int, Image.Image]]:
    """Compute lightweight consensus and select the VL-fallback candidate set.

    Returns (vl_candidate_indices, text regions filtered to that set).
    """
    consensus_by_cluster = _cluster_ocr_consensus(
        candidate_indices=candidate_indices,
        cluster_ids=cluster_ids,
        object_keyframe_ids=object_keyframe_ids,
        ocr_keys=np.asarray(ocr_keys),
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
        min_support_fraction=min_support_fraction,
        min_margin=min_margin,
    )
    consensus_clusters = {
        cluster_id
        for cluster_id, consensus in consensus_by_cluster.items()
        if consensus.accepted
    }
    logger.info(
        "OCR refinement lightweight PP-OCR consensus accepted %d/%d candidate clusters",
        len(consensus_clusters),
        len(consensus_by_cluster),
    )
    if consensus_by_cluster:
        weak = [
            c
            for c in consensus_by_cluster.values()
            if not c.accepted and c.total_count > 0
        ]
        if weak:
            logger.info(
                "OCR refinement lightweight PP-OCR consensus unresolved clusters=%d"
                " (min_support=%.2f min_margin=%d)",
                len(weak),
                float(min_support_fraction),
                int(min_margin),
            )
    ppocr_positive_indices = np.asarray(
        [
            int(idx)
            for idx in candidate_indices.tolist()
            if bool(lightweight_accepted[int(idx)])
            or int(idx) in lightweight_text_regions
        ],
        dtype=np.int64,
    )
    vl_candidate_indices = (
        _fallback_candidates_without_consensus(
            candidate_indices=ppocr_positive_indices,
            cluster_ids=cluster_ids,
            consensus_clusters=consensus_clusters,
        )
        if vl_fallback
        else np.asarray([], dtype=np.int64)
    )
    fallback_index_set = set(vl_candidate_indices.tolist())
    filtered_regions = {
        idx: region
        for idx, region in lightweight_text_regions.items()
        if idx in fallback_index_set
    }
    return vl_candidate_indices, filtered_regions


def _run_lightweight_ocr_pass(
    *,
    candidate_indices: np.ndarray,
    object_crops: Sequence[Image.Image | None],
    cluster_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    device: str,
    lightweight_lang: str,
    lightweight_device: str | None,
    lightweight_preprocess: str,
    lightweight_min_score: float,
    lightweight_single_letter_min_score: float,
    lightweight_consensus_min_support_fraction: float,
    lightweight_consensus_min_margin: int,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
    vl_fallback: bool,
    ocr_texts: list[str],
    ocr_tokens: list[str],
    ocr_keys: list[str],
    ocr_source: np.ndarray,
) -> _LightweightOcrPassResult:
    """Run lightweight PP-OCR; mutates ocr_texts/tokens/keys/source in-place."""
    load_started_at = time.perf_counter()
    recognizer = LightweightPaddleOcrRecognizer(
        lang=lightweight_lang,
        device=lightweight_device or default_lightweight_paddle_device(device),
        min_score=lightweight_min_score,
        single_letter_min_score=lightweight_single_letter_min_score,
    )
    logger.info(
        "OCR refinement lightweight PP-OCR load took %.2fs",
        time.perf_counter() - load_started_at,
    )
    identities, lightweight_accepted, lw_regions = _lightweight_read_crops(
        candidate_indices=candidate_indices,
        object_crops=object_crops,
        recognizer=recognizer,
        lightweight_preprocess=lightweight_preprocess,
        n_objects=len(ocr_texts),
        ocr_texts=ocr_texts,
        ocr_tokens=ocr_tokens,
        ocr_keys=ocr_keys,
        ocr_source=ocr_source,
    )
    vl_candidate_indices, lightweight_text_regions = _select_vl_fallback(
        candidate_indices=candidate_indices,
        cluster_ids=cluster_ids,
        object_keyframe_ids=object_keyframe_ids,
        ocr_keys=ocr_keys,
        lightweight_accepted=lightweight_accepted,
        lightweight_text_regions=lw_regions,
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
        min_support_fraction=lightweight_consensus_min_support_fraction,
        min_margin=lightweight_consensus_min_margin,
        vl_fallback=vl_fallback,
    )
    LightweightPaddleOcrRecognizer.destroy_instance()
    _release_ocr_memory("lightweight PP-OCR before PaddleOCR-VL fallback")
    return _LightweightOcrPassResult(
        identities=identities,
        vl_candidate_indices=vl_candidate_indices,
        lightweight_text_regions=lightweight_text_regions,
    )


def _build_vl_batch_items(
    *,
    batch_indices: list[int],
    lightweight_text_regions: dict[int, Image.Image],
    object_crops: Sequence[Image.Image | None],
    ocr_preprocess: str,
) -> list[tuple[int, Image.Image]]:
    """Resolve and preprocess crops for a batch (preferring lightweight regions)."""
    batch_items: list[tuple[int, Image.Image]] = []
    for idx in batch_indices:
        crop = lightweight_text_regions.get(idx)
        if crop is None:
            crop = object_crops[idx] if idx < len(object_crops) else None
        if crop is None:
            continue
        batch_items.append((idx, preprocess_for_ocr(crop, ocr_preprocess)))
    return batch_items


def _recognize_vl_batch(
    recognizer: PaddleOcrVlRecognizer,
    batch_items: list[tuple[int, Image.Image]],
    use_vl_batches: bool,
) -> tuple[list[str], bool]:
    """Recognize one batch, falling back to per-crop reads on failure.

    Returns (texts aligned with batch_items, use_vl_batches) where the flag is
    cleared permanently once a batched call has failed.
    """
    try:
        if use_vl_batches:
            return recognizer.read_texts([image for _idx, image in batch_items]), True
        return [recognizer.read_text(image) for _idx, image in batch_items], False
    except Exception as exc:
        logger.warning(
            "OCR batch failed for object indices %s: %s; disabling VL batching"
            " and retrying one by one",
            [idx for idx, _image in batch_items],
            exc,
        )
        texts: list[str] = []
        for idx, image in batch_items:
            try:
                texts.append(recognizer.read_text(image))
            except Exception as single_exc:
                logger.warning("OCR failed for object idx %d: %s", idx, single_exc)
                texts.append("")
        return texts, False


def _run_vl_ocr_pass(
    *,
    vl_candidate_indices: np.ndarray,
    lightweight_text_regions: dict[int, Image.Image],
    object_crops: Sequence[Image.Image | None],
    device: str,
    ocr_preprocess: str,
    ocr_batch_size: int,
    ocr_max_new_tokens: int,
    ocr_texts: list[str],
    ocr_tokens: list[str],
    ocr_keys: list[str],
    ocr_source: np.ndarray,
    identities: dict[int, OcrIdentity],
) -> None:
    """Run PaddleOCR-VL; mutates ocr_* lists and identities in-place."""
    batch_size = max(1, int(ocr_batch_size))
    _release_ocr_memory("before PaddleOCR-VL load")
    recognizer_started_at = time.perf_counter()
    recognizer: PaddleOcrVlRecognizer | None = None
    ocr_successes = 0
    try:
        recognizer = PaddleOcrVlRecognizer(
            device=device, max_new_tokens=ocr_max_new_tokens
        )
        logger.info(
            "OCR refinement PaddleOCR-VL load took %.2fs",
            time.perf_counter() - recognizer_started_at,
        )
        ocr_started_at = time.perf_counter()
        candidate_list = vl_candidate_indices.tolist()
        use_vl_batches = batch_size > 1 and hasattr(recognizer, "read_texts")
        for offset in tqdm(
            range(0, len(candidate_list), batch_size),
            desc="OCR refinement",
            unit="batch" if batch_size > 1 else "crop",
        ):
            batch_items = _build_vl_batch_items(
                batch_indices=candidate_list[offset : offset + batch_size],
                lightweight_text_regions=lightweight_text_regions,
                object_crops=object_crops,
                ocr_preprocess=ocr_preprocess,
            )
            if not batch_items:
                continue
            texts, use_vl_batches = _recognize_vl_batch(
                recognizer, batch_items, use_vl_batches
            )
            for (idx, _image), text in zip(batch_items, texts):
                text = text.strip()
                identity = ocr_identity(text)
                ocr_texts[idx] = text
                ocr_tokens[idx] = ocr_token_string(identity)
                ocr_keys[idx] = ocr_key_string(identity)
                if not identity.is_empty:
                    identities[idx] = identity
                    ocr_source[idx] = OCR_SOURCE_PADDLEOCR_VL
                ocr_successes += 1
        logger.info(
            "OCR refinement PaddleOCR-VL inference took %.2fs for %d/%d fallback"
            " candidate crops",
            time.perf_counter() - ocr_started_at,
            ocr_successes,
            int(vl_candidate_indices.size),
        )
    except MemoryError as exc:
        logger.warning(
            "OCR refinement PaddleOCR-VL fallback skipped after GPU memory error"
            " while loading/inferring: %s",
            exc,
        )
    finally:
        if recognizer is not None:
            PaddleOcrVlRecognizer.destroy_instance()
        _release_ocr_memory("PaddleOCR-VL fallback")


def _apply_ocr_anchors(
    *,
    identities: dict[int, OcrIdentity],
    positions_world: np.ndarray,
    cluster_ids: np.ndarray,
    valid_mask: np.ndarray,
    spatial_radius_m: float,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
    object_keyframe_ids: np.ndarray,
    assignment_margin_m: float,
    n_objects: int,
    n_candidates: int,
    started_at: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build anchor groups, reassign affected clusters.

    Returns (refined, ocr_assigned_mask). Logs the summary line.
    """
    anchored_groups = _ocr_anchor_groups(
        identities=identities,
        positions_world=positions_world,
        object_keyframe_ids=object_keyframe_ids,
        spatial_radius_m=spatial_radius_m,
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
    )
    anchored_groups = _merge_compatible_anchor_groups(
        anchored_groups,
        identities=identities,
        positions_world=positions_world,
        spatial_radius_m=spatial_radius_m,
    )
    refined = np.asarray(cluster_ids, dtype=np.int32).copy()
    ocr_assigned_mask = np.zeros(n_objects, dtype=bool)
    if not anchored_groups:
        logger.info("OCR refinement skipped: no supported OCR anchor groups")
        return refined, ocr_assigned_mask

    anchor_initial_clusters = {
        int(cluster_ids[idx])
        for group in anchored_groups
        for idx in group
        if int(cluster_ids[idx]) >= 0
    }
    affected_mask = (
        valid_mask
        & np.isin(cluster_ids, list(anchor_initial_clusters))
        & np.isfinite(positions_world).all(axis=1)
    )
    affected_indices = np.where(affected_mask)[0]
    refined[affected_indices] = -1

    next_label = int(refined[refined >= 0].max(initial=-1)) + 1
    anchor_labels: list[int] = []
    anchor_centroids: list[np.ndarray] = []
    anchor_identities: list[list[OcrIdentity]] = []

    for group in anchored_groups:
        label = next_label
        next_label += 1
        group_arr = np.asarray(group, dtype=np.int64)
        refined[group_arr] = label
        ocr_assigned_mask[group_arr] = True
        anchor_labels.append(label)
        anchor_centroids.append(positions_world[group_arr].mean(axis=0))
        anchor_identities.append([identities[idx] for idx in group])

    for idx in affected_indices.tolist():
        if refined[idx] >= 0:
            continue
        assigned_label = _choose_anchor_label(
            idx=idx,
            identity=identities.get(idx),
            position=positions_world[idx],
            anchor_labels=anchor_labels,
            anchor_centroids=anchor_centroids,
            anchor_identities=anchor_identities,
            spatial_radius_m=spatial_radius_m,
            assignment_margin_m=assignment_margin_m,
        )
        if assigned_label is not None:
            refined[idx] = assigned_label
            ocr_assigned_mask[idx] = True

    refined = _relabel_compact(refined)
    logger.info(
        "OCR refinement applied: candidates=%d anchors=%d affected=%d"
        " clusters_before=%d clusters_after=%d total=%.2fs",
        n_candidates,
        len(anchored_groups),
        int(affected_indices.size),
        int(np.unique(cluster_ids[cluster_ids >= 0]).size),
        int(np.unique(refined[refined >= 0]).size),
        time.perf_counter() - started_at,
    )
    return refined, ocr_assigned_mask


def refine_clusters_with_ocr(
    *,
    cluster_ids: np.ndarray,
    valid_mask: np.ndarray,
    positions_world: np.ndarray,
    object_keyframe_ids: np.ndarray,
    object_embeddings: np.ndarray,
    object_crops: Sequence[Image.Image | None],
    device: str,
    clip_model: Any = None,
    textness_scores: np.ndarray | None = None,
    textness_threshold: float = 0.02,
    spatial_radius_m: float = 1.5,
    min_anchor_observations: int = 2,
    min_anchor_keyframes: int = 2,
    assignment_margin_m: float = 0.5,
    ocr_preprocess: str = "upscale4_autocontrast",
    max_ocr_candidates: int | None = None,
    max_ocr_candidates_per_cluster: int | None = None,
    ocr_batch_size: int = 1,
    ocr_max_new_tokens: int = 32,
    lightweight_first: bool = False,
    lightweight_lang: str = "fr",
    lightweight_device: str | None = None,
    lightweight_preprocess: str = "raw",
    lightweight_min_score: float = 0.7,
    lightweight_single_letter_min_score: float = 0.85,
    lightweight_consensus_min_support_fraction: float = 0.6,
    lightweight_consensus_min_margin: int = 1,
    vl_fallback: bool = True,
) -> OcrRefinementResult:
    started_at = time.perf_counter()
    n_objects = int(cluster_ids.shape[0])

    # --- Textness scores ---
    if textness_scores is None:
        if clip_model is None:
            raise ValueError(
                "Either clip_model or textness_scores must be provided for"
                " OCR refinement"
            )
        textness_started_at = time.perf_counter()
        textness = compute_textness_scores(
            clip_model=clip_model,
            object_embeddings=object_embeddings,
        )
        logger.info(
            "OCR refinement textness scoring took %.2fs",
            time.perf_counter() - textness_started_at,
        )
    else:
        textness = np.asarray(textness_scores, dtype=np.float32)

    ocr_texts: list[str] = [""] * n_objects
    ocr_tokens: list[str] = [""] * n_objects
    ocr_keys: list[str] = [""] * n_objects
    ocr_source = np.full(n_objects, OCR_SOURCE_NONE, dtype=np.int16)

    # --- Candidate selection (filter, per-cluster cap, global cap) ---
    candidate_indices, ocr_candidate_mask = _select_ocr_candidates(
        textness=textness,
        textness_threshold=textness_threshold,
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        positions_world=positions_world,
        object_keyframe_ids=object_keyframe_ids,
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
        max_ocr_candidates_per_cluster=max_ocr_candidates_per_cluster,
        max_ocr_candidates=max_ocr_candidates,
        n_objects=n_objects,
    )
    if candidate_indices.size == 0:
        logger.info("OCR refinement skipped: no text-like localized observations")
        return _make_ocr_refinement_result(
            cluster_ids=np.asarray(cluster_ids, dtype=np.int32).copy(),
            textness_scores=textness,
            ocr_texts=ocr_texts,
            ocr_tokens=ocr_tokens,
            ocr_keys=ocr_keys,
            ocr_candidate_mask=ocr_candidate_mask,
            ocr_assigned_mask=np.zeros(n_objects, dtype=bool),
            ocr_source=ocr_source,
        )

    # --- OCR inference passes ---
    identities: dict[int, OcrIdentity] = {}
    vl_candidate_indices = candidate_indices
    lightweight_text_regions: dict[int, Image.Image] = {}

    if lightweight_first:
        lw_result = _run_lightweight_ocr_pass(
            candidate_indices=candidate_indices,
            object_crops=object_crops,
            cluster_ids=cluster_ids,
            object_keyframe_ids=object_keyframe_ids,
            device=device,
            lightweight_lang=lightweight_lang,
            lightweight_device=lightweight_device,
            lightweight_preprocess=lightweight_preprocess,
            lightweight_min_score=lightweight_min_score,
            lightweight_single_letter_min_score=lightweight_single_letter_min_score,
            lightweight_consensus_min_support_fraction=lightweight_consensus_min_support_fraction,
            lightweight_consensus_min_margin=lightweight_consensus_min_margin,
            min_anchor_observations=min_anchor_observations,
            min_anchor_keyframes=min_anchor_keyframes,
            vl_fallback=vl_fallback,
            ocr_texts=ocr_texts,
            ocr_tokens=ocr_tokens,
            ocr_keys=ocr_keys,
            ocr_source=ocr_source,
        )
        identities = lw_result.identities
        vl_candidate_indices = lw_result.vl_candidate_indices
        lightweight_text_regions = lw_result.lightweight_text_regions

    if vl_candidate_indices.size > 0:
        _run_vl_ocr_pass(
            vl_candidate_indices=vl_candidate_indices,
            lightweight_text_regions=lightweight_text_regions,
            object_crops=object_crops,
            device=device,
            ocr_preprocess=ocr_preprocess,
            ocr_batch_size=ocr_batch_size,
            ocr_max_new_tokens=ocr_max_new_tokens,
            ocr_texts=ocr_texts,
            ocr_tokens=ocr_tokens,
            ocr_keys=ocr_keys,
            ocr_source=ocr_source,
            identities=identities,
        )
    elif lightweight_first:
        logger.info(
            "OCR refinement PaddleOCR-VL skipped: lightweight OCR resolved all"
            " candidates or fallback disabled"
        )
    else:
        logger.info("OCR refinement PaddleOCR-VL skipped: no candidate crops")

    # --- Anchor group formation + cluster reassignment ---
    refined, ocr_assigned_mask = _apply_ocr_anchors(
        identities=identities,
        positions_world=positions_world,
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        spatial_radius_m=spatial_radius_m,
        min_anchor_observations=min_anchor_observations,
        min_anchor_keyframes=min_anchor_keyframes,
        object_keyframe_ids=object_keyframe_ids,
        assignment_margin_m=assignment_margin_m,
        n_objects=n_objects,
        n_candidates=int(candidate_indices.size),
        started_at=started_at,
    )
    return _make_ocr_refinement_result(
        cluster_ids=refined,
        textness_scores=textness,
        ocr_texts=ocr_texts,
        ocr_tokens=ocr_tokens,
        ocr_keys=ocr_keys,
        ocr_candidate_mask=ocr_candidate_mask,
        ocr_assigned_mask=ocr_assigned_mask,
        ocr_source=ocr_source,
    )


def _ocr_anchor_groups(
    *,
    identities: dict[int, OcrIdentity],
    positions_world: np.ndarray,
    object_keyframe_ids: np.ndarray,
    spatial_radius_m: float,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
) -> list[list[int]]:
    by_key: dict[tuple[tuple[str, ...], tuple[str, ...]], list[int]] = {}
    for idx, identity in identities.items():
        key = identity.exact_key
        if key is None:
            continue
        by_key.setdefault(key, []).append(idx)

    groups: list[list[int]] = []
    for indices in by_key.values():
        for component in _spatial_components(
            indices, positions_world, spatial_radius_m
        ):
            if len(component) < int(min_anchor_observations):
                continue
            keyframes = np.unique(
                object_keyframe_ids[np.asarray(component, dtype=np.int64)]
            )
            if keyframes.size < int(min_anchor_keyframes):
                continue
            groups.append(component)
    return groups


def _filter_supported_ocr_candidate_clusters(
    *,
    candidate_indices: np.ndarray,
    cluster_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
) -> np.ndarray:
    kept: list[int] = []
    for label in np.unique(cluster_ids[candidate_indices]).tolist():
        indices = candidate_indices[cluster_ids[candidate_indices] == int(label)]
        if indices.size < int(min_anchor_observations):
            continue
        keyframe_count = np.unique(object_keyframe_ids[indices]).size
        if keyframe_count < int(min_anchor_keyframes):
            continue
        kept.extend(indices.tolist())
    return np.asarray(kept, dtype=np.int64)


def _limit_ocr_candidates_per_cluster(
    *,
    candidate_indices: np.ndarray,
    cluster_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    textness: np.ndarray,
    max_candidates_per_cluster: int,
) -> np.ndarray:
    kept: list[int] = []
    limit = int(max_candidates_per_cluster)
    for label in np.unique(cluster_ids[candidate_indices]).tolist():
        indices = candidate_indices[cluster_ids[candidate_indices] == int(label)]
        ordered = indices[np.argsort(-textness[indices])]
        selected: list[int] = []
        seen_keyframes: set[int] = set()
        for idx in ordered.tolist():
            keyframe_id = int(object_keyframe_ids[idx])
            if keyframe_id in seen_keyframes:
                continue
            selected.append(idx)
            seen_keyframes.add(keyframe_id)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            selected_set = set(selected)
            for idx in ordered.tolist():
                if idx in selected_set:
                    continue
                selected.append(idx)
                if len(selected) >= limit:
                    break
        kept.extend(selected)
    return np.asarray(kept, dtype=np.int64)


def _cluster_ocr_consensus(
    *,
    candidate_indices: np.ndarray,
    cluster_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    ocr_keys: np.ndarray,
    min_anchor_observations: int,
    min_anchor_keyframes: int,
    min_support_fraction: float,
    min_margin: int,
) -> dict[int, OcrConsensus]:
    keys = np.asarray(ocr_keys).astype(str).ravel()
    result: dict[int, OcrConsensus] = {}
    for label in np.unique(cluster_ids[candidate_indices]).tolist():
        cluster_candidate_indices = candidate_indices[
            cluster_ids[candidate_indices] == int(label)
        ]
        non_empty = [
            int(idx)
            for idx in cluster_candidate_indices.tolist()
            if idx < keys.shape[0] and keys[idx]
        ]
        if not non_empty:
            result[int(label)] = OcrConsensus(
                cluster_id=int(label),
                key="",
                support_count=0,
                total_count=0,
                keyframe_count=0,
                margin=0,
                support_fraction=0.0,
                accepted=False,
            )
            continue

        counts = Counter(keys[non_empty].tolist())
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        best_key, best_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        supporting_indices = np.asarray(
            [idx for idx in non_empty if keys[idx] == best_key],
            dtype=np.int64,
        )
        keyframe_count = int(np.unique(object_keyframe_ids[supporting_indices]).size)
        total_count = len(non_empty)
        support_fraction = float(best_count / total_count) if total_count else 0.0
        margin = int(best_count - second_count)
        accepted = (
            int(best_count) >= int(min_anchor_observations)
            and keyframe_count >= int(min_anchor_keyframes)
            and support_fraction >= float(min_support_fraction)
            and margin >= int(min_margin)
        )
        result[int(label)] = OcrConsensus(
            cluster_id=int(label),
            key=str(best_key),
            support_count=int(best_count),
            total_count=int(total_count),
            keyframe_count=keyframe_count,
            margin=margin,
            support_fraction=support_fraction,
            accepted=accepted,
        )
    return result


def _fallback_candidates_without_consensus(
    *,
    candidate_indices: np.ndarray,
    cluster_ids: np.ndarray,
    consensus_clusters: set[int],
) -> np.ndarray:
    if not consensus_clusters:
        return np.asarray(candidate_indices, dtype=np.int64)
    kept = [
        int(idx)
        for idx in candidate_indices.tolist()
        if int(cluster_ids[int(idx)]) not in consensus_clusters
    ]
    return np.asarray(kept, dtype=np.int64)


def _spatial_components(
    indices: Iterable[int],
    positions_world: np.ndarray,
    spatial_radius_m: float,
) -> list[list[int]]:
    ordered = list(indices)
    if len(ordered) <= 1:
        return [ordered] if ordered else []

    from sklearn.neighbors import NearestNeighbors

    positions = np.asarray(
        positions_world[np.asarray(ordered, dtype=np.int64)], dtype=np.float32
    )
    neighbor_search = NearestNeighbors(radius=float(spatial_radius_m))
    neighbor_search.fit(positions)

    parent = np.arange(len(ordered), dtype=np.int32)
    rank = np.zeros(len(ordered), dtype=np.int8)

    def find(idx: int) -> int:
        root = int(idx)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[idx]) != root:
            next_idx = int(parent[idx])
            parent[idx] = root
            idx = next_idx
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    radius = float(spatial_radius_m)
    chunk_size = 4096
    for start in range(0, len(ordered), chunk_size):
        end = min(start + chunk_size, len(ordered))
        neighbor_indices = neighbor_search.radius_neighbors(
            positions[start:end],
            radius=radius,
            return_distance=False,
        )
        for local_left, neighbors in enumerate(neighbor_indices):
            left = start + local_left
            for right in neighbors.tolist():
                if int(right) > left:
                    union(left, int(right))

    by_root: dict[int, list[int]] = {}
    for local_idx, original_idx in enumerate(ordered):
        by_root.setdefault(find(local_idx), []).append(original_idx)
    return [sorted(component) for component in by_root.values()]


def _compatible_anchor_edges(
    groups: list[list[int]],
    representatives: list[OcrIdentity],
    centroids: list[np.ndarray],
    spatial_radius_m: float,
) -> list[tuple[int, int]]:
    """Edges between spatially-close, OCR-compatible, unambiguous group pairs.

    A group is dropped if it is OCR-compatible with more than one distinct exact
    key (ambiguous), since merging it could fuse genuinely different signs.
    """
    possible_edges: list[tuple[int, int]] = []
    compatible_keys_by_group: list[set[tuple[tuple[str, ...], tuple[str, ...]]]] = [
        set() for _ in groups
    ]
    for left_idx in range(len(groups)):
        for right_idx in range(left_idx + 1, len(groups)):
            distance = float(np.linalg.norm(centroids[left_idx] - centroids[right_idx]))
            if distance > float(spatial_radius_m):
                continue
            left_identity = representatives[left_idx]
            right_identity = representatives[right_idx]
            if not ocr_compatible(left_identity, right_identity):
                continue
            possible_edges.append((left_idx, right_idx))
            if right_identity.exact_key is not None:
                compatible_keys_by_group[left_idx].add(right_identity.exact_key)
            if left_identity.exact_key is not None:
                compatible_keys_by_group[right_idx].add(left_identity.exact_key)

    ambiguous_groups = {
        group_idx
        for group_idx, compatible_keys in enumerate(compatible_keys_by_group)
        if len(compatible_keys) > 1
    }
    return [
        (left_idx, right_idx)
        for left_idx, right_idx in possible_edges
        if left_idx not in ambiguous_groups and right_idx not in ambiguous_groups
    ]


def _merge_groups_by_edges(
    groups: list[list[int]], group_edges: list[tuple[int, int]]
) -> list[list[int]]:
    """Union groups connected by edges into merged, sorted index lists."""
    adjacency: list[list[int]] = [[] for _ in groups]
    for left_idx, right_idx in group_edges:
        adjacency[left_idx].append(right_idx)
        adjacency[right_idx].append(left_idx)

    merged: list[list[int]] = []
    seen: set[int] = set()
    for group_idx in range(len(groups)):
        if group_idx in seen:
            continue
        stack = [group_idx]
        seen.add(group_idx)
        merged_group: list[int] = []
        while stack:
            current = stack.pop()
            merged_group.extend(groups[current])
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        merged.append(sorted(merged_group))
    return merged


def _merge_compatible_anchor_groups(
    groups: list[list[int]],
    *,
    identities: dict[int, OcrIdentity],
    positions_world: np.ndarray,
    spatial_radius_m: float,
) -> list[list[int]]:
    if len(groups) <= 1:
        return groups

    representatives = [identities[group[0]] for group in groups]
    centroids = [
        positions_world[np.asarray(group, dtype=np.int64)].mean(axis=0)
        for group in groups
    ]
    group_edges = _compatible_anchor_edges(
        groups, representatives, centroids, spatial_radius_m
    )
    if not group_edges:
        return groups
    return _merge_groups_by_edges(groups, group_edges)


def _choose_anchor_label(
    *,
    idx: int,
    identity: OcrIdentity | None,
    position: np.ndarray,
    anchor_labels: list[int],
    anchor_centroids: list[np.ndarray],
    anchor_identities: list[list[OcrIdentity]],
    spatial_radius_m: float,
    assignment_margin_m: float,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for label, centroid, identities in zip(
        anchor_labels, anchor_centroids, anchor_identities
    ):
        distance = float(np.linalg.norm(position - centroid))
        if distance > float(spatial_radius_m):
            continue
        if identity is not None and not any(
            ocr_compatible(identity, anchor_identity) for anchor_identity in identities
        ):
            continue
        candidates.append((distance, label))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) == 1:
        return candidates[0][1]
    nearest_distance, nearest_label = candidates[0]
    second_distance = candidates[1][0]
    if second_distance - nearest_distance >= float(assignment_margin_m):
        return nearest_label
    return None


def _relabel_compact(cluster_ids: np.ndarray) -> np.ndarray:
    relabeled = np.full(cluster_ids.shape, -1, dtype=np.int32)
    labels = np.unique(cluster_ids)
    labels = labels[labels >= 0]
    for new_label, old_label in enumerate(labels.tolist()):
        relabeled[cluster_ids == old_label] = new_label
    return relabeled
