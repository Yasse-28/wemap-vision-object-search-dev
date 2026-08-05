"""Lightweight PP-OCR recognizer (optional dependency)."""

from __future__ import annotations

import tempfile

from PIL import Image

from pipeline.core.logging import logger
from pipeline.core.models.base_model import OcrSingletonModel

from .identity import LightweightOcrRead, ocr_identity
from .paddle_utils import (
    crop_expanded_bbox,
    is_lightweight_ocr_accepted,
    merge_lightweight_predictions,
    release_ocr_memory,
)


class LightweightPaddleOcrRecognizer(OcrSingletonModel):
    def __init__(
        self,
        *,
        lang: str = "fr",
        device: str = "gpu:0",
        min_score: float = 0.7,
        single_letter_min_score: float = 0.85,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "The `paddleocr` package is required for lightweight PP-OCR refinement."
            ) from exc

        kwargs = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": device,
        }
        try:
            self.ocr = PaddleOCR(**kwargs)
        except TypeError as exc:
            logger.warning(
                "PaddleOCR constructor rejected one or more v3 options: %s", exc
            )
            kwargs.pop("device", None)
            self.ocr = PaddleOCR(**kwargs)
        self.min_score = float(min_score)
        self.single_letter_min_score = float(single_letter_min_score)

    def read(self, image: Image.Image) -> LightweightOcrRead:
        image = image.convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            predictions = self.ocr.predict(tmp.name)
        text, score, bbox = merge_lightweight_predictions(predictions)
        identity = ocr_identity(text)
        text_region = crop_expanded_bbox(image, bbox) if bbox is not None else None
        return LightweightOcrRead(
            text=text,
            score=score,
            identity=identity,
            accepted=is_lightweight_ocr_accepted(
                identity,
                score=score,
                min_score=self.min_score,
                single_letter_min_score=self.single_letter_min_score,
            ),
            text_region=text_region,
        )

    def close(self) -> None:
        ocr = getattr(self, "ocr", None)
        if ocr is not None:
            close = getattr(ocr, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.warning("Lightweight PP-OCR close failed: %s", exc)
        self.ocr = None
        release_ocr_memory("lightweight PP-OCR")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
