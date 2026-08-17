"""Two evidences for "is this one object or two", scored rather than thresholded.

Both replace a distance threshold — the parameter that today decides the number of
objects, and that has a different optimum per venue (1.25-1.5 m in a hotel, ~3 m in an
airport).

`score_1v2` asks whether *merging improves the fit of the whole cluster*, not whether
two members happen to be close. A model-selection score, so the arbitration is
explicit: fidelity against the number of objects, with an MDL penalty per object.

`covisibility_conflict` mines the information no cue in this pipeline uses: a keyframe
that sees object A and does *not* see B where B is claimed to be. It is deliberately a
score and not a hard constraint — without an occlusion test a wall between the camera
and B produces a false conflict, which a filter would turn into a lost merge.
"""

from __future__ import annotations

import numpy as np

#: Metres of positional slack that grow with range: pose error plus depth error.
BASE_SIGMA_M = 0.5
SIGMA_PER_METRE = 0.05
#: Object extent, held **fixed** during merges on purpose. Letting it float would let
#: two neighbouring objects grow into one plausible big object.
DEFAULT_OBJECT_EXTENT_M = 1.0
#: MDL cost of claiming one more object, in the same units as the residual sum.
DEFAULT_OBJECT_PENALTY = 6.0
#: Guard rail on an *observed* extent, not a tuning knob: an aberrant depth turns a
#: 30 deg box into a 20 m object, and one such detection would justify any merge.
EXTENT_BOUNDS_M = (0.2, 5.0)
#: Huber knee on the Mahalanobis residual: beyond it an aberrant depth stops deciding.
HUBER_KNEE = 2.0


def _sigmas(points: np.ndarray, origins: np.ndarray | None) -> np.ndarray:
    """Per-point positional sigma, larger for detections seen from far away."""
    if origins is None:
        return np.full(points.shape[0], BASE_SIGMA_M)
    ranges = np.linalg.norm(points - origins, axis=1)
    return BASE_SIGMA_M + SIGMA_PER_METRE * ranges


def _huber(squared: np.ndarray, knee: float = HUBER_KNEE) -> np.ndarray:
    """Quadratic below the knee, linear above, in squared-Mahalanobis units."""
    return np.where(
        squared <= knee**2,
        squared,
        knee * (2.0 * np.sqrt(np.maximum(squared, 0)) - knee),
    )


def latent_cost(
    points: np.ndarray,
    origins: np.ndarray | None = None,
    *,
    object_extent_m: float | np.ndarray = DEFAULT_OBJECT_EXTENT_M,
    iterations: int = 12,
) -> float:
    """Robust cost of explaining a set of 3D points by **one** latent centre.

    The scale is `sigma_i^2 + extent_i^2`: a detection is allowed to sit anywhere on
    an object of that size without being charged for it, which is what stops a large
    object from looking like two.

    `object_extent_m` accepts an array with one value per point, for callers that
    observe the size instead of assuming it. A scalar keeps the previous behaviour
    exactly.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return 0.0
    extents = np.asarray(object_extent_m, dtype=np.float64)
    if extents.ndim not in (0, 1) or (
        extents.ndim == 1 and extents.shape[0] != points.shape[0]
    ):
        raise ValueError("object_extent_m must be a scalar or one value per point")
    scales = _sigmas(points, origins) ** 2 + extents**2
    centre = points.mean(axis=0)
    for _ in range(iterations):
        squared = np.einsum("ni,ni->n", points - centre, points - centre) / scales
        # IRLS weights of the Huber loss, so one aberrant depth cannot pull the centre.
        weights = np.where(
            squared <= HUBER_KNEE**2,
            1.0,
            HUBER_KNEE / np.sqrt(np.maximum(squared, 1e-12)),
        )
        weights = weights / scales
        updated = (points * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(updated - centre) < 1e-4:
            centre = updated
            break
        centre = updated
    squared = np.einsum("ni,ni->n", points - centre, points - centre) / scales
    return float(_huber(squared).sum())


def observed_extent_m(
    ranges_m: np.ndarray,
    angular_width: np.ndarray,
    angular_height: np.ndarray,
) -> np.ndarray:
    """Half the metric size a detection's box subtends at its own range.

    The size the constant `DEFAULT_OBJECT_EXTENT_M` assumes is in fact observable:
    a box of angular diagonal `a` seen at range `r` covers about `r * a` metres.
    Clipped to `EXTENT_BOUNDS_M`.
    """
    diagonal = np.hypot(
        np.asarray(angular_width, dtype=np.float64),
        np.asarray(angular_height, dtype=np.float64),
    )
    extents = 0.5 * np.asarray(ranges_m, dtype=np.float64) * diagonal
    clipped: np.ndarray = np.clip(extents, *EXTENT_BOUNDS_M)
    return clipped


def cluster_extent_m(member_extents_m: np.ndarray) -> float:
    """The **median** member extent, which is not the mean and not the maximum.

    A detection whose object is cut by the edge of the panorama under-estimates the
    size, and a mean lets those pull the whole cluster down; the median does not.
    """
    values = np.asarray(member_extents_m, dtype=np.float64)
    if values.size == 0:
        return DEFAULT_OBJECT_EXTENT_M
    return float(np.median(values))


def score_1v2(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_origins: np.ndarray | None = None,
    right_origins: np.ndarray | None = None,
    *,
    object_extent_m: float = DEFAULT_OBJECT_EXTENT_M,
    left_extent_m: float | None = None,
    right_extent_m: float | None = None,
    object_penalty: float = DEFAULT_OBJECT_PENALTY,
) -> float:
    """`[L(A) + L(B) + pen(2)] - [L(A∪B) + pen(1)]`; positive favours merging.

    Note what this is *not*: a distance. Two clusters 3 m apart merge when a single
    centre still explains both within their own spread, and stay apart when it does
    not — the decision scales with the objects rather than with the venue.

    `left_extent_m`/`right_extent_m` override the shared extent per side, for callers
    that measure each cluster's size. The union is scored with each side keeping its
    own value: recomputing one extent on the union is exactly the drift the fixed
    constant exists to prevent — two neighbours would grow into one plausible big
    object and every merge would look justified.
    """
    left = np.asarray(left_points, dtype=np.float64)
    right = np.asarray(right_points, dtype=np.float64)
    union = np.vstack([left, right])
    union_origins = (
        None
        if left_origins is None or right_origins is None
        else np.vstack([np.asarray(left_origins), np.asarray(right_origins)])
    )
    left_extent = object_extent_m if left_extent_m is None else left_extent_m
    right_extent = object_extent_m if right_extent_m is None else right_extent_m
    union_extent: float | np.ndarray = (
        object_extent_m
        if left_extent_m is None and right_extent_m is None
        else np.concatenate(
            (np.full(left.shape[0], left_extent), np.full(right.shape[0], right_extent))
        )
    )
    separate = (
        latent_cost(left, left_origins, object_extent_m=left_extent)
        + latent_cost(right, right_origins, object_extent_m=right_extent)
        + 2.0 * object_penalty
    )
    together = (
        latent_cost(union, union_origins, object_extent_m=union_extent) + object_penalty
    )
    return float(separate - together)


def covisibility_conflict(
    viewer_origins: np.ndarray,
    viewer_directions: np.ndarray,
    viewer_half_extents: np.ndarray,
    other_point: np.ndarray,
    *,
    margin: float = 1.5,
) -> float:
    """Share of A's viewpoints from which B lies clearly outside A's own box.

    For every keyframe that detected A, B's point is turned into a direction and
    compared with the direction A was detected in. Landing far outside A's angular
    extent means that viewpoint saw two different places — evidence against merging.

    Returns a value in `[0, 1]`; 0 when no viewpoint disagrees. **No occlusion test**:
    a wall between the camera and B reads as a conflict here, which is why the caller
    must use this as a cost and never as a veto.
    """
    origins = np.asarray(viewer_origins, dtype=np.float64)
    directions = np.asarray(viewer_directions, dtype=np.float64)
    half = np.asarray(viewer_half_extents, dtype=np.float64)
    if origins.shape[0] == 0:
        return 0.0
    offsets = np.asarray(other_point, dtype=np.float64) - origins
    norms = np.linalg.norm(offsets, axis=1)
    usable = norms > 1e-6
    if not usable.any():
        return 0.0
    cosines = np.einsum(
        "ni,ni->n", offsets[usable] / norms[usable, None], directions[usable]
    )
    separation = np.arccos(np.clip(cosines, -1.0, 1.0))
    conflicts = separation > margin * np.maximum(half[usable], 1e-6)
    return float(conflicts.mean())
