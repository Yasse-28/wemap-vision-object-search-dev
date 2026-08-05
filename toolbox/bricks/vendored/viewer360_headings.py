"""Keyframe orientation → compass heading.

Vendored from `backend/api/viewer360/v1_legacy.py` — see ../PROVENANCE.md.
Only `headings_from_orientations` is needed; the rest of that module converts v2
types to the v1 360-viewer graph JSON, which this repo does not serve.
"""

from __future__ import annotations

import numpy as np

from toolbox.bricks.vendored.geo_transform import (
    OPENGL_FORWARD_VECTOR,
    eus_compass_heading_radians_batch,
)
from toolbox.bricks.vendored.maths import quaternion


def headings_from_orientations(orientations_wxyz: np.ndarray) -> np.ndarray:
    """Compass heading (degrees in [0, 360)) for each keyframe orientation.

    ``orientations_wxyz`` is ``(N, 4)``, quaternions in ``[w, x, y, z]`` order
    that rotate from CameraFrame (OpenGL, -Z = forward) to LocalFrame (EUS).
    Output is ``(N,)``.
    """
    forward_eus = quaternion.rotate_vector_batch(
        quaternion.cast_batch(orientations_wxyz), OPENGL_FORWARD_VECTOR
    )
    headings = np.degrees(eus_compass_heading_radians_batch(forward_eus))
    return np.where(headings < 0.0, headings + 360.0, headings)
