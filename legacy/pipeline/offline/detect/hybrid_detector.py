"""YOLO-World and GroundingDINO detector classes for the hybrid offline pipeline.

Both classes run on the *full equirectangular image* (ERP-direct) and return
:class:`~common.HybridDetection` objects in ERP pixel coordinates.  The
calling code in ``build_index.py`` is responsible for post-processing
(NMS, projection, capping) using helpers from :mod:`common`.

Imports
-------
Geometry helpers, filters, and the HybridDetection record live in
:mod:`common`.  All vocabulary constants and GDINO prompts live in
:mod:`prompts`.  This file contains only model-loading logic and the two
inference classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image

from pipeline.core.detectors import Detection, postprocess_detections
from pipeline.core.logging import logger
from pipeline.core.models.base_model import load_model
from pipeline.core.models.detection.grounding_dino import GroundingDINOModel
from pipeline.offline.detect.common import (  # noqa: F401  (re-exported for callers)
    FACE_NAME_TO_LOCAL_INDEX,
    BBox,
    HybridDetection,
    _area_filter,
    _aspect_ratio_filter,
    _clip_bboxes_to_image,
    _normalize_label,
    erp_pixel_to_pano_ray,
    pano_ray_to_face_pixel,
    project_erp_bbox_to_face,
)
from pipeline.offline.detect.prompts import (  # noqa: F401  (re-exported for callers)
    AIRPORT_OPERATIONS_VOCAB,
    AIRPORT_SPECIFIC_VOCAB,
    BROAD_VOCAB,
    DEFAULT_GDINO_LABEL,
    DEFAULT_VENUE,
    DROP_LABELS_LOWER,
    GENERIC_OBJECTS_VOCAB,
    HOTEL_FURNITURE_VOCAB,
    HOTEL_SPECIFIC_VOCAB,
    PUBLIC_INFRA_VOCAB,
    SAFETY_SECURITY_VOCAB,
    TECHNICAL_MEP_VOCAB,
    VENUE_PROMPTS,
    VENUE_YOLO_VOCABS,
    _build_yolo_vocabulary_passes,
    _dedupe_normalized,
)

# ---------------------------------------------------------------------------
# YOLO-World detector
# ---------------------------------------------------------------------------


class YoloWorldDetector:
    """Multi-vocabulary YOLO-World detector running on the full ERP.

    Two super-vocab passes (BROAD + venue-specific) replace the original
    five fine-grained passes to halve ``set_classes()`` text-encoder re-runs.
    ``agnostic_nms=False`` (class-aware NMS) preserves recall when many
    classes share a single forward pass.

    Venue selection
    ~~~~~~~~~~~~~~~
    Pass ``venue="hotel"`` (or any key in :data:`VENUE_YOLO_VOCABS`) to
    automatically select the matching specific vocab.  Alternatively, supply
    an explicit ``specific_vocab`` tuple.  The legacy ``enable_airport_specific``
    keyword still works for backward compatibility.
    """

    def __init__(
        self,
        *,
        weights: str,
        device: str,
        imgsz: int,
        yolo_iou: float,
        max_det: int,
        enable_broad: bool = True,
        venue: Optional[str] = None,
        specific_vocab: Optional[Tuple[str, ...]] = None,
        conf_broad: float = 0.05,
        conf_specific: float = 0.03,
        # Backward-compat aliases (airport-era API)
        enable_airport_specific: Optional[bool] = None,
        conf_airport_specific: Optional[float] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "YOLO-World requires ultralytics. Install with: pip install ultralytics"
            ) from exc

        # Resolve backward-compat airport kwargs
        if (
            enable_airport_specific is not None
            and venue is None
            and specific_vocab is None
        ):
            venue = "airport" if enable_airport_specific else None
        if conf_airport_specific is not None:
            conf_specific = conf_airport_specific

        # Resolve the venue-specific vocab
        resolved_vocab: Optional[Tuple[str, ...]] = specific_vocab
        if resolved_vocab is None and venue is not None:
            resolved_vocab = VENUE_YOLO_VOCABS.get(venue)
            if resolved_vocab is None:
                logger.warning(
                    "Unknown venue %r — no venue-specific YOLO vocab; using BROAD only",
                    venue,
                )

        logger.info(
            "Loading YOLO-World weights: %s  venue=%s  specific_vocab=%s cats",
            weights,
            venue,
            len(resolved_vocab) if resolved_vocab else 0,
        )
        self.model = YOLO(weights)
        self.model.to(device)
        self._base_kw = {
            "verbose": False,
            "imgsz": int(imgsz),
            "iou": float(yolo_iou),
            "max_det": int(max_det),
            "device": device,
            "agnostic_nms": False,
        }
        self.passes = _build_yolo_vocabulary_passes(
            enable_broad=enable_broad,
            resolved_vocab=resolved_vocab,
            conf_broad=conf_broad,
            conf_specific=conf_specific,
        )

    def detect_batch(
        self,
        images: List[np.ndarray],
    ) -> List[List[HybridDetection]]:
        """Run every vocab pass over the full image batch in one call each.

        Returns one detection list per input image (same order as input).
        """
        if not images:
            return []
        per_image: List[List[HybridDetection]] = [[] for _ in images]
        for vocab, conf in self.passes:
            self.model.set_classes(list(vocab))
            kw = dict(self._base_kw)
            kw["conf"] = conf
            results = self.model.predict(images, **kw)
            names = self.model.names
            for i, r in enumerate(results):
                per_image[i].extend(self._result_to_hybrid(r, names=names))
        return per_image

    def _result_to_hybrid(self, r: Any, *, names: Any) -> List[HybridDetection]:
        """Convert one Ultralytics result object to normalised HybridDetections."""
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        out: List[HybridDetection] = []
        for j in range(len(xyxy)):
            raw_label = str(names[int(cls_ids[j])])
            norm_label = _normalize_label(raw_label)
            if norm_label in DROP_LABELS_LOWER:
                continue
            x0, y0, x1, y1 = (float(v) for v in xyxy[j])
            out.append(
                HybridDetection(
                    bbox=(x0, y0, x1, y1),
                    score=float(confs[j]),
                    label=norm_label,
                    raw_label=raw_label,
                    source="yolo",
                )
            )
        return out


# ---------------------------------------------------------------------------
# GroundingDINO venue-specific detector
# ---------------------------------------------------------------------------


class GdinoVenueDetector:
    """Single-prompt GDINO detector running on the full ERP.

    Optimised direct-forward path: prompt is pre-tokenised once at init, and
    ``self.model.model(...)`` is called directly to avoid per-call tokenisation
    and logger spam from the wrapper's ``detect()`` method.

    All detections receive ``synthetic_label`` (default ``"gdino_venue"``)
    because HF GDINO label extraction is unreliable on long prompts; the bbox
    is the proposal, not the label.

    Dummy padding for partial final batches reuses ``pils[0]`` (the first real
    image) instead of a small white square so the HF processor's output tensor
    shape stays invariant across batches, preventing torch.compile
    recompilation on every partial batch.
    """

    def __init__(
        self,
        *,
        prompt: str,
        device: str,
        score: float = 0.06,
        min_confidence: float = 0.10,
        iou_threshold: float = 0.5,
        min_area_ratio: float = 0.0,
        max_area_ratio: float = 0.95,
        synthetic_label: str = DEFAULT_GDINO_LABEL,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        model_conf: Dict[str, object] = {
            "name": "grounding_dino",
            "device": device,
            "batch_size": self.batch_size,
            "prompt": prompt,
            "threshold": float(score),
        }
        logger.info(
            "Loading GroundingDINO (batch_size=%d, prompt: %s...)",
            self.batch_size,
            prompt[:120],
        )
        self.model = cast(GroundingDINOModel, load_model(model_conf, device=device))
        self.min_confidence = float(min_confidence)
        self.iou_threshold = float(iou_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.synthetic_label = _normalize_label(synthetic_label)
        self._text_inputs = self._tokenize_prompt(prompt)

    def _tokenize_prompt(self, prompt: str) -> dict:
        """Tokenise once; reuse for every image batch."""
        return dict(
            self.model.processor.tokenizer(
                [prompt] * self.batch_size,
                padding="max_length",
                max_length=self.model.conf.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
        )

    def _prepare_pils(
        self,
        images: List[np.ndarray],
    ) -> Tuple[List[Image.Image], int]:
        """Convert numpy ERPs to PIL and pad the last partial batch."""
        pils: List[Image.Image] = [Image.fromarray(img) for img in images]
        n_real = len(pils)
        if n_real > self.batch_size:
            raise ValueError(
                f"detect_batch got {n_real} images but batch_size={self.batch_size}"
            )
        if n_real < self.batch_size:
            # Pad with the first real image so the HF processor always
            # produces the same pixel_values tensor shape, keeping
            # torch.compile's compiled graph stable across batches.
            pils.extend(pils[0] for _ in range(self.batch_size - n_real))
        return pils, n_real

    def _prepare_image_inputs(self, pils: List[Image.Image]) -> dict:
        """Run GDINO's image processor and cast pixel values to fp16."""
        image_inputs = self.model.processor.image_processor(
            images=pils,
            return_tensors="pt",
        )
        image_inputs = image_inputs.to(self.model.device)
        image_inputs["pixel_values"] = image_inputs["pixel_values"].half()
        return dict(image_inputs)

    @torch.inference_mode()
    def detect_batch(
        self,
        images: List[np.ndarray],
    ) -> List[List[HybridDetection]]:
        """Run one batched forward+decode for up to ``self.batch_size`` ERPs."""
        if not images:
            return []
        pils, n_real = self._prepare_pils(images)
        image_inputs = self._prepare_image_inputs(pils)
        inputs = {**image_inputs, **self._text_inputs}
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model.model(**inputs)
        results = self.model.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=self._text_inputs["input_ids"],
            threshold=self.model.conf.threshold,
            target_sizes=[pil.size[::-1] for pil in pils],
        )
        if n_real < self.batch_size:
            results = results[:n_real]
        if not results:
            return [[] for _ in range(n_real)]
        per_image: List[List[HybridDetection]] = []
        for i in range(n_real):
            pil = pils[i]
            res = results[i]
            raw_detections: List[Detection] = [
                Detection(bbox=tuple(box.tolist()), confidence=sc.item(), label=lbl)
                for box, sc, lbl in zip(res["boxes"], res["scores"], res["text_labels"])
            ]
            kept = postprocess_detections(
                raw_detections,
                pil.width,
                pil.height,
                min_confidence=self.min_confidence,
                min_area_ratio=self.min_area_ratio,
                max_area_ratio=self.max_area_ratio,
                iou_threshold=self.iou_threshold,
            )
            per_image.append(
                [
                    HybridDetection(
                        bbox=cast(
                            Tuple[float, float, float, float],
                            tuple(float(v) for v in det.bbox),
                        ),
                        score=float(det.confidence),
                        label=self.synthetic_label,
                        raw_label=str(det.label),
                        source="gdino",
                    )
                    for det in kept
                ]
            )
        return per_image
