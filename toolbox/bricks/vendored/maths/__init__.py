"""Linear algebra primitives for the v2 backend.

Each subpackage is a namespace for a primitive type:

- ``maths.vector3``  — 3-vectors, plus batched (N, 3).
- ``maths.matrix3``  — 3x3 rotations, plus batched (N, 3, 3).
- ``maths.matrix4``  — 4x4 homogeneous poses, plus batched (N, 4, 4).
- ``maths.quaternion`` — unit quaternions in ``[w, x, y, z]`` order, plus batched (N, 4).

Numpy-native (``float64`` everywhere), batch-first via broadcasting. The
quaternion ordering matches ``QuaternionField`` and the SFM ``output.json``.

Types are exported here for short imports::

    from toolbox.bricks.vendored.maths import Matrix4, Vector3, Quaternion
    from toolbox.bricks.vendored.maths import matrix4, vector3, quaternion
    m = matrix4.compose(vector3.zero(), matrix3.identity())
"""

from . import matrix3, matrix4, quaternion, vector3
from .matrix3 import Matrix3, Matrix3Batch
from .matrix4 import Matrix4, Matrix4Batch
from .quaternion import Quaternion, QuaternionBatch
from .vector3 import Vector3, Vector3Batch

__all__ = [
    "Matrix3",
    "Matrix3Batch",
    "Matrix4",
    "Matrix4Batch",
    "Quaternion",
    "QuaternionBatch",
    "Vector3",
    "Vector3Batch",
    "matrix3",
    "matrix4",
    "quaternion",
    "vector3",
]
