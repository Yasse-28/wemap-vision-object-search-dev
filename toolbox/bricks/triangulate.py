"""Robust N-view triangulation of an object from its detection rays.

The cost is **angular**, not metric: the observation noise is the wobble of a bbox
centre (median 5.3 deg, p90 12.8 deg on this data), so a residual in metres would
under-penalise a far, badly aimed ray and over-penalise a near, good one. It also
makes the inlier threshold a constant rather than a per-venue one — the metric
equivalent scales with how far things typically are, which is why `clustering_eps_m`
had to be retuned between a hotel (1.25-1.5 m) and an airport (~3 m).

Solved by IRLS on the closed-form "closest point to a set of lines": weighting each
ray by `1 / range^2` turns that metric solve into a first-order approximation of the
angular one, and a Huber weight on the angular residual makes it robust. Three
unknowns, so this is ~25 lines of normal equations rather than a scipy dependency.

The depth-map point of a detection can seed the solve and, optionally, hold it: at
near-zero parallax the ray system is singular and the depth prior is the only thing
that keeps the answer at a plausible range.

Why hand-rolled at all: `poselib` (2.0.5) exposes pose estimators only and no
multi-view point triangulation; `pycolmap` would bring an SfM pipeline for one solve.

Frame: EUS, the same one `localize` clusters in. Distances are metres, angles degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Calibrated on the measured aiming error rather than picked round: half of p90.
DEFAULT_INLIER_THRESHOLD_DEG = 8.0
#: Huber knee, in degrees — beyond it a ray keeps pulling, but linearly.
DEFAULT_HUBER_DEG = 4.0


@dataclass(frozen=True)
class TriangulationResult:
    """`point_eus` is None when no consensus reached the minimum ray count."""

    point_eus: tuple[float, float, float] | None
    inlier_indices: tuple[int, ...]
    """Angle between each ray and the solution — the quantity actually minimised."""
    residuals_deg: tuple[float, ...]
    """Perpendicular distance to each ray. Reported, not optimised: metres are what a
    human reads off a map, degrees are what the estimator believes."""
    residuals_m: tuple[float, ...]
    mean_inlier_residual_deg: float | None
    mean_inlier_residual_m: float | None
    """Largest angle between two inlier rays — the parallax the answer rests on."""
    max_parallax_deg: float | None


def _closest_point(
    origins: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray | None = None,
    depth_points: np.ndarray | None = None,
    depth_weight: float = 0.0,
) -> tuple[float, float, float] | None:
    """Weighted least-squares point closest to a set of rays.

    Sum of `(I - dd^T)` projectors, solved once. Singular when the rays are parallel —
    two views of the same direction pin nothing down — unless a depth prior is given,
    which adds `lambda * I` to the normal equations and makes them solvable again.
    """
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    if weights is not None:
        projectors = projectors * weights[:, None, None]
    lhs = projectors.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projectors, origins)
    if depth_points is not None and depth_weight > 0.0:
        usable = ~np.isnan(depth_points).any(axis=1)
        if usable.any():
            lhs = lhs + depth_weight * usable.sum() * np.eye(3)
            rhs = rhs + depth_weight * depth_points[usable].sum(axis=0)
    try:
        point = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return (float(point[0]), float(point[1]), float(point[2]))


def _angular_residuals(
    point: tuple[float, float, float], origins: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Angle between each ray and the direction from its origin to `point`, degrees.

    A solution behind the camera needs no special case here: it lands past 90 deg on
    its own, which no sane threshold accepts. That is the whole reason the metric
    version needed an explicit `inf`.
    """
    offsets = np.asarray(point) - origins
    ranges = np.linalg.norm(offsets, axis=1)
    safe = np.where(ranges > 0, ranges, 1.0)
    cosines = np.einsum("ni,ni->n", offsets / safe[:, None], directions)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def _metric_residuals(
    point: tuple[float, float, float], origins: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Perpendicular distance to each ray, kept for reporting only."""
    offsets = np.asarray(point) - origins
    along = np.einsum("ni,ni->n", offsets, directions)
    perpendicular = offsets - along[:, None] * directions
    return np.linalg.norm(perpendicular, axis=1)


def _refine_point(
    origins: np.ndarray,
    directions: np.ndarray,
    initial: tuple[float, float, float],
    *,
    huber_deg: float = DEFAULT_HUBER_DEG,
    depth_points: np.ndarray | None = None,
    depth_weight: float = 0.0,
    iterations: int = 20,
) -> tuple[float, float, float] | None:
    """IRLS towards the angular optimum, starting from `initial`.

    `1 / range^2` converts the metric solve into the angular one to first order; the
    Huber factor stops one badly aimed ray from dragging the point. Converged in
    practice within a handful of passes, so the iteration cap is a guard, not a knob.
    """
    point = initial
    for _ in range(iterations):
        residuals = _angular_residuals(point, origins, directions)
        ranges = np.linalg.norm(np.asarray(point) - origins, axis=1)
        ranges = np.where(ranges > 1e-6, ranges, 1e-6)
        huber = np.where(
            residuals <= huber_deg, 1.0, huber_deg / np.maximum(residuals, 1e-9)
        )
        weights = huber / (ranges**2)
        updated = _closest_point(
            origins, directions, weights, depth_points, depth_weight
        )
        if updated is None:
            return point
        moved = float(np.linalg.norm(np.asarray(updated) - np.asarray(point)))
        point = updated
        if moved < 1e-4:
            break
    return point


def _max_parallax_deg(directions: np.ndarray) -> float:
    if directions.shape[0] < 2:
        return 0.0
    cosines = np.clip(directions @ directions.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines.min())))


EMPTY_RESULT = TriangulationResult(None, (), (), (), None, None, None)


def triangulate_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
    min_rays: int = 2,
    iterations: int = 200,
    depth_points: np.ndarray | None = None,
    depth_weight: float = 0.0,
    seed: int = 0,
) -> TriangulationResult:
    """RANSAC over seeds, then IRLS refit on the consensus set.

    Seeds are ray pairs *and*, when available, each detection's own depth point: a
    pair with almost no baseline proposes a meaningless point, whereas the depth point
    is always a plausible one. That is what keeps low-parallax sets from returning
    nothing at all.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    count = origins.shape[0]
    if count < max(2, min_rays):
        return EMPTY_RESULT

    rng = np.random.default_rng(seed)
    seeds: list[tuple[float, float, float]] = []
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if len(pairs) > iterations:
        chosen = rng.choice(len(pairs), size=iterations, replace=False)
        pairs = [pairs[index] for index in chosen]
    for i, j in pairs:
        candidate = _closest_point(origins[[i, j]], directions[[i, j]])
        if candidate is not None:
            seeds.append(candidate)
    if depth_points is not None:
        for row in np.asarray(depth_points, dtype=np.float64):
            if not np.isnan(row).any():
                seeds.append((float(row[0]), float(row[1]), float(row[2])))
    if not seeds:
        return EMPTY_RESULT

    best_inliers: np.ndarray = np.array([], dtype=int)
    for candidate in seeds:
        inliers = np.flatnonzero(
            _angular_residuals(candidate, origins, directions) <= inlier_threshold_deg
        )
        if inliers.size > best_inliers.size:
            best_inliers = inliers
    if best_inliers.size < max(2, min_rays):
        return EMPTY_RESULT

    initial = _closest_point(
        origins[best_inliers],
        directions[best_inliers],
        None,
        None if depth_points is None else np.asarray(depth_points)[best_inliers],
        depth_weight,
    )
    if initial is None:
        return EMPTY_RESULT
    point = _refine_point(
        origins[best_inliers],
        directions[best_inliers],
        initial,
        depth_points=(
            None if depth_points is None else np.asarray(depth_points)[best_inliers]
        ),
        depth_weight=depth_weight,
    )
    if point is None:
        return EMPTY_RESULT

    residuals_deg = _angular_residuals(point, origins, directions)
    # Refitting moves the point, so the consensus set is recomputed rather than
    # carried over — otherwise a ray can be reported as an inlier of a point it is
    # no longer close to.
    inliers = np.flatnonzero(residuals_deg <= inlier_threshold_deg)
    if inliers.size < max(2, min_rays):
        return EMPTY_RESULT
    residuals_m = _metric_residuals(point, origins, directions)
    return TriangulationResult(
        point_eus=point,
        inlier_indices=tuple(int(index) for index in inliers),
        residuals_deg=tuple(float(value) for value in residuals_deg),
        residuals_m=tuple(float(value) for value in residuals_m),
        mean_inlier_residual_deg=float(residuals_deg[inliers].mean()),
        mean_inlier_residual_m=float(residuals_m[inliers].mean()),
        max_parallax_deg=_max_parallax_deg(directions[inliers]),
    )


@dataclass(frozen=True)
class Hypothesis:
    """One object found in a set of rays: its point and the rays that voted for it."""

    point_eus: tuple[float, float, float]
    """Positions in the *input* ray list, not in the consensus set."""
    member_indices: tuple[int, ...]
    mean_residual_deg: float | None
    mean_residual_m: float | None
    max_parallax_deg: float | None


def triangulate_multi(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
    min_rays: int = 2,
    max_models: int = 8,
    depth_points: np.ndarray | None = None,
    depth_weight: float = 0.0,
    seed: int = 0,
) -> tuple[list[Hypothesis], list[int]]:
    """Sequential RANSAC: fit, remove the consensus, refit on what is left.

    Single-model RANSAC cannot answer "how many objects are in this set" — it fits one
    point and calls everything else an outlier, which is why two real objects in one
    basket come back as a consensus plus rejects rather than as two answers. Fitting
    the remainder repeatedly is the smallest change that turns the same estimator into
    a partition.

    Known weakness, and the reason `max_models` exists: the order matters. The first
    hypothesis is free to take rays that a later one would have claimed, so a returned
    partition is *a* reading of the set, not the provably best one. J-linkage-style
    joint fitting is the upgrade if that shows up in practice.

    Returns the hypotheses, largest consensus first, and the indices no model claimed.
    """
    count = np.asarray(origins).shape[0]
    remaining = list(range(count))
    hypotheses: list[Hypothesis] = []

    while len(remaining) >= max(2, min_rays) and len(hypotheses) < max_models:
        result = triangulate_rays(
            np.asarray(origins)[remaining],
            np.asarray(directions)[remaining],
            inlier_threshold_deg=inlier_threshold_deg,
            min_rays=min_rays,
            depth_points=(
                None if depth_points is None else np.asarray(depth_points)[remaining]
            ),
            depth_weight=depth_weight,
            seed=seed + len(hypotheses),
        )
        if result.point_eus is None:
            break
        members = [remaining[local] for local in result.inlier_indices]
        if len(members) < max(2, min_rays):
            break
        hypotheses.append(
            Hypothesis(
                point_eus=result.point_eus,
                member_indices=tuple(members),
                mean_residual_deg=result.mean_inlier_residual_deg,
                mean_residual_m=result.mean_inlier_residual_m,
                max_parallax_deg=result.max_parallax_deg,
            )
        )
        claimed = set(members)
        remaining = [index for index in remaining if index not in claimed]

    return hypotheses, remaining


# ── J-linkage / T-linkage ────────────────────────────────────────────────────
#
# Sequential RANSAC decides one object at a time, so the first model can steal rays a
# later one would have claimed. Linkage methods avoid the ordering by describing each
# ray by *which hypotheses it prefers*, then clustering those descriptions: rays of one
# object prefer the same hypotheses, whatever order they were generated in.
#
# J-linkage uses binary preference sets and Jaccard distance; T-linkage replaces both
# with continuous preferences and Tanimoto distance, which stops the answer depending
# on rays sitting a hair either side of the threshold.

MethodName = str


def _sampled_hypothesis_points(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Points from minimal sets (ray pairs), the hypothesis pool both methods score."""
    count = origins.shape[0]
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if not pairs:
        return np.empty((0, 3))
    rng = np.random.default_rng(seed)
    if len(pairs) > sample_count:
        chosen = rng.choice(len(pairs), size=sample_count, replace=False)
        pairs = [pairs[index] for index in chosen]
    points = []
    for i, j in pairs:
        point = _closest_point(origins[[i, j]], directions[[i, j]])
        if point is not None:
            points.append(point)
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3))


def _preference_matrix(
    points: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    threshold: float,
    soft: bool,
) -> np.ndarray:
    """`(rays, hypotheses)` preferences: binary for J-linkage, decaying for T."""
    residuals = np.stack(
        [_angular_residuals(tuple(point), origins, directions) for point in points],
        axis=1,
    )
    if not soft:
        return (residuals <= threshold).astype(np.float64)
    # T-linkage's own definition: exponential decay, cut at 5 tau so a far hypothesis
    # contributes exactly zero rather than a vanishing tail that still links clusters.
    tau = threshold / 5.0 if threshold > 0 else 1.0
    preferences = np.exp(-residuals / tau)
    preferences[residuals > 5 * tau] = 0.0
    return preferences


def _pairwise_distance(left: np.ndarray, right: np.ndarray, *, soft: bool) -> float:
    """Jaccard (binary) or Tanimoto (continuous) distance between two rows."""
    if soft:
        dot = float(np.dot(left, right))
        denominator = float(np.dot(left, left) + np.dot(right, right) - dot)
        return 1.0 if denominator <= 0 else 1.0 - dot / denominator
    intersection = float(np.sum(np.minimum(left, right)))
    union = float(np.sum(np.maximum(left, right)))
    return 1.0 if union <= 0 else 1.0 - intersection / union


def _linkage_clusters(preferences: np.ndarray, *, soft: bool) -> list[list[int]]:
    """Agglomerate while two clusters still share a preference; merge by intersection.

    The merged cluster's preference is the element-wise minimum — the models *both*
    parts agree on. That is what stops the chaining the plain graph approaches suffer
    from: a cluster's description can only shrink, so it cannot keep absorbing.
    """
    clusters = [[index] for index in range(preferences.shape[0])]
    profiles = [preferences[index].copy() for index in range(preferences.shape[0])]

    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _pairwise_distance(profiles[i], profiles[j], soft=soft)
                if best is None or distance < best[0]:
                    best = (distance, i, j)
        if best is None or best[0] >= 1.0:
            break
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        profiles[i] = np.minimum(profiles[i], profiles[j])
        del clusters[j]
        del profiles[j]
    return clusters


def triangulate_linkage(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
    min_rays: int = 2,
    method: MethodName = "jlinkage",
    sample_count: int = 400,
    depth_points: np.ndarray | None = None,
    depth_weight: float = 0.0,
    seed: int = 0,
) -> tuple[list[Hypothesis], list[int]]:
    """J-linkage (`"jlinkage"`) or T-linkage (`"tlinkage"`) partition of the rays.

    Unlike sequential RANSAC this is order-free: clusters emerge from agreement between
    preference sets, so no model gets first pick. Returns hypotheses sorted by size and
    the rays whose cluster was too small to be an object.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if origins.shape[0] < max(2, min_rays):
        return [], list(range(origins.shape[0]))

    points = _sampled_hypothesis_points(
        origins, directions, sample_count=sample_count, seed=seed
    )
    # The depth points join the hypothesis pool for the same reason they seed RANSAC:
    # a low-baseline pair proposes nothing usable, a depth point always does.
    if depth_points is not None:
        usable = np.asarray(depth_points, dtype=np.float64)
        usable = usable[~np.isnan(usable).any(axis=1)]
        if usable.size:
            points = np.vstack([points, usable]) if points.size else usable
    if not points.size:
        return [], list(range(origins.shape[0]))

    soft = method == "tlinkage"
    preferences = _preference_matrix(
        points, origins, directions, threshold=inlier_threshold_deg, soft=soft
    )
    clusters = _linkage_clusters(preferences, soft=soft)

    hypotheses: list[Hypothesis] = []
    unassigned: list[int] = []
    for members in clusters:
        if len(members) < max(2, min_rays):
            unassigned.extend(members)
            continue
        member_depths = (
            None if depth_points is None else np.asarray(depth_points)[members]
        )
        initial = _closest_point(
            origins[members], directions[members], None, member_depths, depth_weight
        )
        point = (
            None
            if initial is None
            else _refine_point(
                origins[members],
                directions[members],
                initial,
                depth_points=member_depths,
                depth_weight=depth_weight,
            )
        )
        if point is None:
            unassigned.extend(members)
            continue
        residuals = _angular_residuals(point, origins[members], directions[members])
        metric = _metric_residuals(point, origins[members], directions[members])
        hypotheses.append(
            Hypothesis(
                point_eus=point,
                member_indices=tuple(sorted(int(index) for index in members)),
                mean_residual_deg=float(residuals.mean()) if residuals.size else None,
                mean_residual_m=float(metric.mean()) if metric.size else None,
                max_parallax_deg=_max_parallax_deg(directions[members]),
            )
        )
    hypotheses.sort(key=lambda item: len(item.member_indices), reverse=True)
    return hypotheses, sorted(unassigned)
