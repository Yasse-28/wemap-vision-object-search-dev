"""MetaCLIP model and shared runtime (device resolution, OOM fallback,
singleton load)."""

from __future__ import annotations

import gc
import time
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.metaclip_2.modeling_metaclip_2 import MetaClip2Model

from pipeline.core.logging import logger
from pipeline.core.models.base_model import SingletonModel, load_model

# --- Model (module-level `torch` is used by tests that patch
# resolve_metaclip_device paths)

_METACLIP_DEVICE_OVERRIDE: str | None = None


class MetaCLIP(SingletonModel):
    default_conf = {
        "name": "metaclip2",
        "pretrained_model_name_or_path": "facebook/metaclip-2-worldwide-huge-quickgelu",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, **conf: Any):
        super().__init__(**conf)
        logger.info("MetaCLIP2 loading model")
        start_time = time.time()
        self.device = self.conf.device
        self.model = (
            MetaClip2Model.from_pretrained(
                self.conf.pretrained_model_name_or_path,
                dtype=torch.float16,
            )
            .eval()
            .to(self.device)
        )
        self.projection_dim = self.model.config.projection_dim
        self.processor = AutoProcessor.from_pretrained(
            self.conf.pretrained_model_name_or_path,
            use_fast=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.pretrained_model_name_or_path,
            use_fast=True,
        )
        logger.info("MetaCLIP2 model loaded in %.2fs", time.time() - start_time)

    def eval(self) -> "MetaCLIP":
        self.model.eval()
        return self

    def to(self, device: Any) -> "MetaCLIP":
        self.model.to(device)
        return self

    def _feature_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(f"Unsupported MetaCLIP feature output type: {type(output)!r}")

    # Cap on images per forward pass. Offline builds can hand thousands of
    # crops in a single call; encoding them all at once allocates ViT-H
    # activations for the whole set and can OOM a 24 GB GPU. Sub-batching keeps
    # peak VRAM bounded and independent of the caller's batch size.
    IMAGE_FEATURE_BATCH_SIZE = 256

    @torch.inference_mode()
    def get_image_features(
        self, images: Any, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        images = list(images)
        if not images:
            return torch.empty((0, self.projection_dim), dtype=torch.float32)

        bs = int(batch_size or self.IMAGE_FEATURE_BATCH_SIZE)
        if bs <= 0 or len(images) <= bs:
            return self._encode_image_batch(images)

        chunks = [
            self._encode_image_batch(images[i : i + bs])
            for i in range(0, len(images), bs)
        ]
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def _encode_image_batch(self, images: Any) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        image_features = self._feature_tensor(self.model.get_image_features(**inputs))
        return F.normalize(image_features, p=2, dim=-1).cpu()

    @torch.inference_mode()
    def get_text_features(self, text: Any) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        text_features = self._feature_tensor(self.model.get_text_features(**inputs))
        return F.normalize(text_features, p=2, dim=-1).cpu()

    def warmup(self, iterations: int = 5) -> None:
        logger.info("MetaCLIP2 warming up with %d iterations", iterations)
        start_time = time.time()
        self.warming_up = True
        try:
            for _ in range(iterations):
                self.get_text_features("A picture of a window")
        finally:
            self.warming_up = False
            self.warmed_up = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("MetaCLIP2 warmed up in %.2fs", time.time() - start_time)


# --- Runtime (shared instance, device, OOM recovery) --------------------------


def resolve_metaclip_device(preferred_device: Optional[str] = None) -> str:
    if _METACLIP_DEVICE_OVERRIDE is not None:
        return _METACLIP_DEVICE_OVERRIDE

    if preferred_device:
        return preferred_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_metaclip_device_override(device: str | None) -> None:
    global _METACLIP_DEVICE_OVERRIDE
    _METACLIP_DEVICE_OVERRIDE = device


def get_metaclip_device_override() -> str | None:
    return _METACLIP_DEVICE_OVERRIDE


def get_shared_metaclip(preferred_device: Optional[str] = None) -> MetaCLIP:
    device = resolve_metaclip_device(preferred_device)
    try:
        return cast(
            MetaCLIP,
            load_model({"name": "metaclip2"}, device=device),
        )
    except Exception as exc:
        if not _is_cuda_oom(exc) or str(device).startswith("cpu"):
            raise
        _cleanup_after_failed_cuda_load()
        set_metaclip_device_override("cpu")
        return cast(
            MetaCLIP,
            load_model({"name": "metaclip2"}, device="cpu"),
        )


def get_loaded_metaclip_device(preferred_device: Optional[str] = None) -> str:
    model = get_shared_metaclip(preferred_device)
    try:
        return str(model.model.device)
    except Exception:
        return resolve_metaclip_device(preferred_device)


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _cleanup_after_failed_cuda_load() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
