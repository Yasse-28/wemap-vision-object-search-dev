"""PaddleOCR-VL recognizer (optional heavy dependency)."""

from __future__ import annotations

import os
import tempfile

from PIL import Image

from pipeline.core.logging import logger
from pipeline.core.models.base_model import OcrSingletonModel

from .paddle_utils import merge_paddleocr_vl_texts, release_ocr_memory


class PaddleOcrVlRecognizer(OcrSingletonModel):
    def __init__(
        self,
        *,
        device: str,
        max_new_tokens: int = 32,
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "The official `paddleocr` package with PaddleOCRVL is required"
                " for OCR-VL fallback. Install `paddleocr[doc-parser]` as"
                " described in the PaddleOCR-VL documentation."
            ) from exc

        paddle_device = "gpu:0" if device.startswith("cuda") else device
        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            device=paddle_device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
            format_block_content=False,
            use_queues=False,
        )
        self.max_new_tokens = int(max_new_tokens)

    def read_text(self, image: Image.Image) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("RGB").save(tmp.name)
            outputs = self.pipeline.predict(
                tmp.name,
                use_layout_detection=False,
                prompt_label="ocr",
                max_new_tokens=self.max_new_tokens,
            )
            return merge_paddleocr_vl_texts(outputs)

    def read_texts(self, images: list[Image.Image]) -> list[str]:
        return [self.read_text(image) for image in images]

    def close(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.warning("PaddleOCR-VL close failed: %s", exc)
        self.pipeline = None
        release_ocr_memory("PaddleOCR-VL")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
