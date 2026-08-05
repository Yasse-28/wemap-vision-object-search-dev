"""Unit quaternion primitives in [w, x, y, z] order.

Matches the storage format of ``QuaternionField`` and the SFM ``output.json``
produced by ``third_party/sfm/scripts/export.py``.
"""

from __future__ import annotations

from typing import NewType

import numpy as np

from .matrix3 import Matrix3
from .vector3 import Vector3, Vector3Batch

Quaternion = NewType("Quaternion", np.ndarray)  # (4,) — [w, x, y, z]
QuaternionBatch = NewType("QuaternionBatch", np.ndarray)  # (N, 4)


def cast(arr) -> Quaternion:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {a.shape}")
    return Quaternion(np.ascontiguousarray(a))


def cast_batch(arr) -> QuaternionBatch:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"QuaternionBatch must have shape (N, 4), got {a.shape}")
    return QuaternionBatch(np.ascontiguousarray(a))


def identity() -> Quaternion:
    return Quaternion(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))


def from_axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
    n = float(np.linalg.norm(axis))
    if n == 0.0:
        return identity()
    a = axis / n
    half = float(angle_rad) / 2.0
    s = float(np.sin(half))
    return Quaternion(
        np.array([np.cos(half), a[0] * s, a[1] * s, a[2] * s], dtype=np.float64)
    )


def normalize(q: Quaternion) -> Quaternion:
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return identity()
    return Quaternion(q / n)


def multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    w2, x2, y2, z2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return Quaternion(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def to_matrix3(q: Quaternion) -> Matrix3:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return Matrix3(
        np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    )


def from_matrix3(m: Matrix3) -> Quaternion:
    # Shepperd's method — numerically stable across all branches.
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * float(np.sqrt(trace + 1.0))
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return Quaternion(np.array([w, x, y, z], dtype=np.float64))


def rotate(q: Quaternion, v: Vector3) -> Vector3:
    return Vector3(to_matrix3(q) @ v)


def rotate_vector_batch(q_batch: QuaternionBatch, v: Vector3) -> Vector3Batch:
    """Apply each quaternion in the batch to the same 3-vector. Returns (N, 3)."""
    w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ox = (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - w * z) * vy
        + 2 * (x * z + w * y) * vz
    )
    oy = (
        2 * (x * y + w * z) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - w * x) * vz
    )
    oz = (
        2 * (x * z - w * y) * vx
        + 2 * (y * z + w * x) * vy
        + (1 - 2 * (x * x + y * y)) * vz
    )
    return Vector3Batch(np.column_stack([ox, oy, oz]))


def rotate_vectors_batch(
    q_batch: QuaternionBatch, v_batch: Vector3Batch
) -> Vector3Batch:
    """Apply each quaternion q_batch[i] to its corresponding vector v_batch[i]. Returns (N, 3)."""
    w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]
    vx, vy, vz = v_batch[:, 0], v_batch[:, 1], v_batch[:, 2]
    ox = (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - w * z) * vy
        + 2 * (x * z + w * y) * vz
    )
    oy = (
        2 * (x * y + w * z) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - w * x) * vz
    )
    oz = (
        2 * (x * z - w * y) * vx
        + 2 * (y * z + w * x) * vy
        + (1 - 2 * (x * x + y * y)) * vz
    )
    return Vector3Batch(np.column_stack([ox, oy, oz]))
