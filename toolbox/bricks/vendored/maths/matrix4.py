"""4x4 homogeneous matrix primitives (numpy float64).

Used to represent rigid-body poses (rotation + translation). ``invert`` assumes
a rigid transform: it uses ``R^T`` for the rotation part rather than the general
matrix inverse. For non-rigid 4x4 matrices, build your own inverse.
"""

from __future__ import annotations

from typing import NewType

import numpy as np

from . import vector3
from .matrix3 import Matrix3, Matrix3Batch
from .vector3 import Vector3, Vector3Batch

Matrix4 = NewType("Matrix4", np.ndarray)  # (4, 4)
Matrix4Batch = NewType("Matrix4Batch", np.ndarray)  # (N, 4, 4)


def cast(arr) -> Matrix4:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != (4, 4):
        raise ValueError(f"Matrix4 must have shape (4, 4), got {a.shape}")
    return Matrix4(np.ascontiguousarray(a))


def cast_batch(arr) -> Matrix4Batch:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4):
        raise ValueError(f"Matrix4Batch must have shape (N, 4, 4), got {a.shape}")
    return Matrix4Batch(np.ascontiguousarray(a))


def identity() -> Matrix4:
    return Matrix4(np.eye(4, dtype=np.float64))


def compose(translation: Vector3, rotation: Matrix3) -> Matrix4:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return Matrix4(m)


def from_matrix3(rotation: Matrix3) -> Matrix4:
    return compose(vector3.zero(), rotation)


def invert(m: Matrix4) -> Matrix4:
    r = m[:3, :3]
    t = m[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return Matrix4(out)


def multiply(a: Matrix4, b: Matrix4) -> Matrix4:
    return Matrix4(a @ b)


def multiply_vector3(m: Matrix4, v: Vector3) -> Vector3:
    return Vector3(m[:3, :3] @ v + m[:3, 3])


def extract_translation(m: Matrix4) -> Vector3:
    return Vector3(np.ascontiguousarray(m[:3, 3]))


def extract_rotation(m: Matrix4) -> Matrix3:
    return Matrix3(np.ascontiguousarray(m[:3, :3]))


# ---- Batch ------------------------------------------------------------------
# All batch helpers assume the leading axis is the batch dim. Shapes documented
# per function. Vectorised via numpy broadcasting / einsum — no Python loop.


def multiply_batch(a: Matrix4Batch, b: Matrix4Batch) -> Matrix4Batch:
    return Matrix4Batch(a @ b)


def multiply_vector3_batch(m: Matrix4Batch, v: Vector3Batch) -> Vector3Batch:
    r = m[:, :3, :3]  # (N, 3, 3)
    t = m[:, :3, 3]  # (N, 3)
    return Vector3Batch(np.einsum("nij,nj->ni", r, v) + t)


def invert_batch(m: Matrix4Batch) -> Matrix4Batch:
    r = m[:, :3, :3]  # (N, 3, 3)
    t = m[:, :3, 3]  # (N, 3)
    r_t = np.transpose(r, (0, 2, 1))
    out = np.zeros_like(m)
    out[:, :3, :3] = r_t
    out[:, :3, 3] = -np.einsum("nij,nj->ni", r_t, t)
    out[:, 3, 3] = 1.0
    return Matrix4Batch(out)


def extract_translation_batch(m: Matrix4Batch) -> Vector3Batch:
    return Vector3Batch(np.ascontiguousarray(m[:, :3, 3]))


def extract_rotation_batch(m: Matrix4Batch) -> Matrix3Batch:
    return Matrix3Batch(np.ascontiguousarray(m[:, :3, :3]))
