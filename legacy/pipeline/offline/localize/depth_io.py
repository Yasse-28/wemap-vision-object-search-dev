from __future__ import annotations

import time

import numpy as np
import tifffile
import zarr

# Frozen sqrt-quantization constants — mirror
# wemap-vision-backend/depth/service/decode.py.
# The on-disk format is fixed by spec (sqrt + tiff_zstd L9 + 10 bits); update both
# copies in lock-step if the spec ever changes.
_DMIN_M = 0.1
_DMAX_M = 200.0
_EFFECTIVE_BITS = 10
_LEVELS = (1 << _EFFECTIVE_BITS) - 1  # 1023
_SHIFT = 16 - _EFFECTIVE_BITS  # 6
_SQRT_LO = float(np.sqrt(_DMIN_M))
_SQRT_HI = float(np.sqrt(_DMAX_M))


def load_depth_from_tif(tif_path: str) -> tuple[np.ndarray, float]:
    """Load a frozen sqrt-quantized uint16 TIFF depth map and decode to float32 meters.

    Pixel value 0 is the invalid sentinel and is decoded to NaN.
    Returns (depth_meters [H, W] float32, elapsed_seconds).
    """
    t_start = time.perf_counter()
    arr = tifffile.imread(tif_path)
    if arr.dtype != np.uint16:
        raise ValueError(f"Expected uint16 TIFF, got dtype {arr.dtype}")
    valid = arr != 0
    raw = arr.astype(np.int32) >> _SHIFT
    normed = (raw - 1).astype(np.float32) / (_LEVELS - 1)
    decoded = (_SQRT_LO + normed * (_SQRT_HI - _SQRT_LO)) ** 2
    depth_meters = np.where(valid, decoded, np.float32(np.nan))
    return depth_meters, time.perf_counter() - t_start


def load_depth_from_zarr(
    zarr_path: str, decode_to_meters: bool = True
) -> tuple[np.ndarray, float]:
    t_start = time.perf_counter()
    root = zarr.open_group(zarr_path, mode="r")
    depth_ds = root["depth"]
    enc = depth_ds[:]
    valid_mask = root["valid_mask"][:] if "valid_mask" in root else None

    if decode_to_meters and "min_m" in depth_ds.attrs and "max_m" in depth_ds.attrs:
        dmin = float(depth_ds.attrs["min_m"])
        dmax = float(depth_ds.attrs["max_m"])
        if (
            depth_ds.attrs.get("encoding", "") == "quantized_uint16"
            and "step_m" in depth_ds.attrs
        ):
            depth = (enc.astype(np.float32) * float(depth_ds.attrs["step_m"])) + dmin
        else:
            scale = 65535.0 / (dmax - dmin) if dmax > dmin else 1.0
            depth = (enc.astype(np.float32) / scale) + dmin
        if valid_mask is not None:
            depth = np.where(valid_mask, depth, np.nan)
    else:
        depth = enc
    return depth, time.perf_counter() - t_start
