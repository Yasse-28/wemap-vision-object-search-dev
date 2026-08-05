"""3D vector primitives (numpy float64)."""

from __future__ import annotations

from typing import NewType

import numpy as np

Vector3 = NewType("Vector3", np.ndarray)  # (3,)
Vector3Batch = NewType("Vector3Batch", np.ndarray)  # (N, 3)


def cast(arr) -> Vector3:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"Vector3 must have shape (3,), got {a.shape}")
    return Vector3(np.ascontiguousarray(a))


def cast_batch(arr) -> Vector3Batch:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Vector3Batch must have shape (N, 3), got {a.shape}")
    return Vector3Batch(np.ascontiguousarray(a))


def zero() -> Vector3:
    return Vector3(np.zeros(3, dtype=np.float64))


def from_xyz(x: float, y: float, z: float) -> Vector3:
    return Vector3(np.array([x, y, z], dtype=np.float64))
