from __future__ import annotations

from collections import Counter
from typing import cast

import numpy as np

from pipeline.core.types import UNRESOLVED_LEVEL_SENTINEL


def build_cluster_cutout_membership(
    *,
    object_cluster_ids: np.ndarray,
    object_cutout_ids: np.ndarray,
    object_keyframe_ids: np.ndarray,
    detection_levels: np.ndarray | None,
) -> dict[str, np.ndarray]:
    cluster_ids = np.asarray(object_cluster_ids, dtype=np.int32).ravel()
    cutout_ids = np.asarray(object_cutout_ids, dtype=np.int64).ravel()
    keyframe_ids = np.asarray(object_keyframe_ids, dtype=np.int64).ravel()
    levels = (
        np.asarray(detection_levels, dtype=np.int32).ravel()
        if detection_levels is not None
        else np.full(cluster_ids.shape, UNRESOLVED_LEVEL_SENTINEL, dtype=np.int32)
    )

    if not (
        cluster_ids.shape[0]
        == cutout_ids.shape[0]
        == keyframe_ids.shape[0]
        == levels.shape[0]
    ):
        raise ValueError(
            "Cluster cutout membership inputs must have aligned row counts"
        )

    grouped: dict[tuple[int, int], dict[str, list[int] | int]] = {}
    for cluster_id, cutout_id, keyframe_id, level in zip(
        cluster_ids.tolist(),
        cutout_ids.tolist(),
        keyframe_ids.tolist(),
        levels.tolist(),
    ):
        if cluster_id < 0:
            continue
        key = (int(cluster_id), int(cutout_id))
        bucket = grouped.setdefault(
            key,
            {
                "keyframe_ids": [],
                "levels": [],
                "count": 0,
            },
        )
        cast(list[int], bucket["keyframe_ids"]).append(int(keyframe_id))
        if int(level) != UNRESOLVED_LEVEL_SENTINEL:
            cast(list[int], bucket["levels"]).append(int(level))
        bucket["count"] = cast(int, bucket["count"]) + 1

    rows: list[tuple[int, int, int, int, int]] = []
    for (cluster_id, cutout_id), bucket in sorted(grouped.items()):
        unique_keyframes = sorted(set(cast(list[int], bucket["keyframe_ids"])))
        if len(unique_keyframes) != 1:
            raise ValueError(
                f"Corrupted cluster-cutout membership: cluster_id={cluster_id} "
                f"cutout_id={cutout_id} spans multiple keyframes {unique_keyframes}"
            )
        bucket_levels = cast(list[int], bucket["levels"])
        if not bucket_levels:
            raise ValueError(
                f"Cluster-cutout membership missing resolved level: "
                f"cluster_id={cluster_id} cutout_id={cutout_id}"
            )
        level_counts = Counter(bucket_levels)
        resolved_level = sorted(
            level_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        rows.append(
            (
                int(cluster_id),
                int(cutout_id),
                int(unique_keyframes[0]),
                int(resolved_level),
                cast(int, bucket["count"]),
            )
        )

    if not rows:
        return {
            "cluster_cutout_cluster_ids": np.zeros(0, dtype=np.int32),
            "cluster_cutout_ids": np.zeros(0, dtype=np.int64),
            "cluster_cutout_keyframe_ids": np.zeros(0, dtype=np.int64),
            "cluster_cutout_levels": np.zeros(0, dtype=np.int32),
            "cluster_cutout_observation_counts": np.zeros(0, dtype=np.int32),
        }

    return {
        "cluster_cutout_cluster_ids": np.asarray(
            [row[0] for row in rows], dtype=np.int32
        ),
        "cluster_cutout_ids": np.asarray([row[1] for row in rows], dtype=np.int64),
        "cluster_cutout_keyframe_ids": np.asarray(
            [row[2] for row in rows], dtype=np.int64
        ),
        "cluster_cutout_levels": np.asarray([row[3] for row in rows], dtype=np.int32),
        "cluster_cutout_observation_counts": np.asarray(
            [row[4] for row in rows], dtype=np.int32
        ),
    }
