"""Depth-map I/O and per-pixel decoding.

Vendored from `backend/depth/service/decode.py` — see ../PROVENANCE.md.

PORT NOTE: production reads the TIFF off a Django `VideoKeyframe.depth_map`
storage field. Here depth maps are plain files inside a map directory, so the
`VideoKeyframe` / `ClientError` paths (`load_depth_map`, `load_depth_maps`) are
dropped and only the path-based reader survives. The decode math is untouched.

The on-disk format is the frozen `sqrt + tiff_zstd L9 + 10 bits` produced by
`third_party/depth/io.py::save_depth` in the backend. Those constants are frozen
by spec; if they ever change, this copy must change in lock-step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


class DepthMapNotFound(Exception):
    """No depth TIFF on disk for this keyframe."""


# Frozen depth-quantization constants — mirror third_party/depth/io.py.
_DMIN_M = 0.1
_DMAX_M = 200.0
_EFFECTIVE_BITS = 10
_LEVELS = (1 << _EFFECTIVE_BITS) - 1
_SHIFT = 16 - _EFFECTIVE_BITS
_SQRT_LO = float(np.sqrt(_DMIN_M))
_SQRT_HI = float(np.sqrt(_DMAX_M))


def load_depth_map_from_path(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Read one depth TIFF; return ``(uint16 [H, W], (H, W))``.

    Raises ``DepthMapNotFound`` when the file is absent, so callers can skip a
    keyframe the same way production skips one with no ``depth_map`` set.
    """
    path = Path(path)
    if not path.is_file():
        raise DepthMapNotFound(f"No depth map at '{path}'.")
    arr = tifffile.imread(str(path))
    if arr.dtype != np.uint16:
        raise ValueError(f"Expected uint16 TIFF, got dtype {arr.dtype}")
    return arr, arr.shape


def decode_uint16_meters(code: np.ndarray | int) -> np.ndarray:
    """Decode uint16 depth code(s) to float32 meters. 0 stays 0 (invalid sentinel)."""
    code = np.asarray(code)
    valid = code != 0
    raw = code.astype(np.int32) >> _SHIFT
    normed = (raw - 1).astype(np.float32) / max(_LEVELS - 1, 1)
    depth = (_SQRT_LO + normed * (_SQRT_HI - _SQRT_LO)) ** 2
    return np.where(valid, depth.astype(np.float32), np.float32(0.0))


def sample_depth_meters(depth_uint16: np.ndarray, x: float, y: float) -> float | None:
    """Sample one pixel of a uint16 depth map; return meters, or ``None`` if invalid."""
    h, w = depth_uint16.shape
    xi = int(round(float(x))) % w
    yi = max(0, min(int(round(float(y))), h - 1))
    raw = int(depth_uint16[yi, xi])
    if raw == 0:
        return None
    return float(decode_uint16_meters(np.array([raw], dtype=np.uint16))[0])
