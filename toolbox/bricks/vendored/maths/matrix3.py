"""3x3 matrix primitives (numpy float64), used mainly for rotation matrices."""

from __future__ import annotations

from typing import NewType

import numpy as np

Matrix3 = NewType("Matrix3", np.ndarray)  # (3, 3)
Matrix3Batch = NewType("Matrix3Batch", np.ndarray)  # (N, 3, 3)


def cast(arr) -> Matrix3:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError(f"Matrix3 must have shape (3, 3), got {a.shape}")
    return Matrix3(np.ascontiguousarray(a))


def cast_batch(arr) -> Matrix3Batch:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError(f"Matrix3Batch must have shape (N, 3, 3), got {a.shape}")
    return Matrix3Batch(np.ascontiguousarray(a))


def identity() -> Matrix3:
    return Matrix3(np.eye(3, dtype=np.float64))
