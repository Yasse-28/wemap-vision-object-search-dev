"""Equirectangular (ERP) pixel ↔ camera ray helpers.

Convention shared with ``third_party/object_search/prepare/convention.py``:

- ``theta = atan2(x, z)`` — 0 at the ERP horizontal centre.
- ``phi = asin(y)`` — 0 at the horizon, +π/2 at the top of the image.

That ERP frame is *left-handed* relative to the OpenGL camera frame stored in
``GeoKeyframe.orientation`` (OpenGL ``-Z`` is forward; the ERP centre pixel is
the camera's forward direction). The helpers here absorb the sign flip and
return rays already in the OpenGL camera frame, so callers can pass the result
straight to ``Pose.rotate`` / ``Pose.apply``.
"""

from __future__ import annotations

import numpy as np

from toolbox.bricks.vendored.maths import Vector3, Vector3Batch, vector3


def pixel_to_theta_phi(
    x: float, y: float, width: int, height: int
) -> tuple[float, float]:
    """ERP pixel → ``(theta, phi)`` in the depth / object_search convention."""
    u = float(x) / (width - 1)
    v = float(y) / (height - 1)
    theta = u * 2.0 * np.pi - np.pi
    phi = np.pi / 2.0 - v * np.pi
    return float(theta), float(phi)


def theta_phi_to_opengl_ray_batch(theta: np.ndarray, phi: np.ndarray) -> Vector3Batch:
    """Vectorized ``theta_phi_to_opengl_ray``. Returns (N, 3) in OpenGL camera frame."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    cos_phi = np.cos(phi)
    return vector3.cast_batch(
        np.column_stack(
            [cos_phi * np.sin(theta), np.sin(phi), -cos_phi * np.cos(theta)]
        )
    )


def theta_phi_to_opengl_ray(theta: float, phi: float) -> Vector3:
    """``(theta, phi)`` → unit ray in OpenGL camera frame (-Z = forward)."""
    return Vector3(theta_phi_to_opengl_ray_batch(np.array([theta]), np.array([phi]))[0])


def pixel_to_opengl_ray(x: float, y: float, width: int, height: int) -> Vector3:
    """ERP pixel → unit ray in OpenGL camera frame."""
    theta, phi = pixel_to_theta_phi(x, y, width, height)
    return theta_phi_to_opengl_ray(theta, phi)
