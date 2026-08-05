from pipeline.core.models.base_model import OcrSingletonModel
from pipeline.core.models.ocr.identity import (
    LightweightOcrRead,
    OcrIdentity,
    normalize_ocr_text,
    ocr_compatible,
    ocr_identity,
    ocr_key_string,
    ocr_token_string,
)
from pipeline.core.models.ocr.paddle_lightweight import LightweightPaddleOcrRecognizer
from pipeline.core.models.ocr.paddle_utils import (
    default_lightweight_paddle_device,
    release_ocr_memory,
)
from pipeline.core.models.ocr.paddle_vl import PaddleOcrVlRecognizer

__all__ = [
    "LightweightOcrRead",
    "LightweightPaddleOcrRecognizer",
    "OcrIdentity",
    "OcrSingletonModel",
    "PaddleOcrVlRecognizer",
    "default_lightweight_paddle_device",
    "normalize_ocr_text",
    "ocr_compatible",
    "ocr_identity",
    "ocr_key_string",
    "ocr_token_string",
    "release_ocr_memory",
]
