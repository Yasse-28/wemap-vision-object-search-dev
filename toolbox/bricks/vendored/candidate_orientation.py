"""Cutout direction (theta, phi) → quaternion.

Vendored from `backend/object_search/v1_legacy.py` — see ../PROVENANCE.md.

Only `candidate_orientation` is taken. `v1_attitude_quaternion` and
`KIOSK_HEADING_FLIP_QUATERNION` are deliberately left behind: they bake in the v1
kiosk handler's hard-coded "+180°", and the v1.5 wire format this repo serves does
not (livemap applies the flip client-side). See `localize.v1_5_observation_quaternion`.
"""

from __future__ import annotations

import numpy as np

from toolbox.bricks.vendored.erp import theta_phi_to_opengl_ray
from toolbox.bricks.vendored.geo_transform import OPENGL_FORWARD_VECTOR
from toolbox.bricks.vendored.maths import Quaternion, quaternion, vector3


def candidate_orientation(theta: float, phi: float) -> Quaternion:
    """Quaternion that rotates camera forward (-Z, body frame) to the direction
    ``(theta, phi)`` in the ERP / object_search convention."""
    target = theta_phi_to_opengl_ray(theta, phi)
    forward = OPENGL_FORWARD_VECTOR
    dot = float(np.dot(forward, target))
    if dot >= 1.0 - 1e-12:
        return quaternion.identity()
    if dot <= -1.0 + 1e-12:
        # 180° rotation around any axis perpendicular to forward; use +Y.
        return quaternion.from_axis_angle(vector3.from_xyz(0.0, 1.0, 0.0), float(np.pi))
    axis = np.cross(forward, target)
    angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    return quaternion.from_axis_angle(vector3.cast(axis), angle)
