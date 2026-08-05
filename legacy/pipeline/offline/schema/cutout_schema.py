"""Cutout records and id encoding (safe for lightweight tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from pipeline.core.types import DEFAULT_ID_STRIDE


def encode_cutout_id(
    keyframe_id: int, local_index: int, id_stride: int = DEFAULT_ID_STRIDE
) -> int:
    return int(keyframe_id) * int(id_stride) + int(local_index)


def decode_keyframe_id(cutout_id: int, id_stride: int = DEFAULT_ID_STRIDE) -> int:
    return int(cutout_id) // int(id_stride)


def decode_local_index(cutout_id: int, id_stride: int = DEFAULT_ID_STRIDE) -> int:
    return int(cutout_id) % int(id_stride)


@dataclass(frozen=True)
class CutoutRecord:
    cutout_id: int
    keyframe_id: int
    face_label: str
    image: Image.Image
    center_xy: tuple[float, float] | None = None
    rotation_cutout_to_equirect: np.ndarray | None = None
