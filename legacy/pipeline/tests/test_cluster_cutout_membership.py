from __future__ import annotations

import numpy as np
import pytest

from pipeline.core.types import UNRESOLVED_LEVEL_SENTINEL
from pipeline.offline.localize.cluster_cutout_membership import (
    build_cluster_cutout_membership,
)


def test_cluster_cutout_membership_deduplicates_and_keeps_level() -> None:
    membership = build_cluster_cutout_membership(
        object_cluster_ids=np.array([0, 0, 0, 1, -1], dtype=np.int32),
        object_cutout_ids=np.array([1001, 1001, 1002, 2001, 9999], dtype=np.int64),
        object_keyframe_ids=np.array([10, 10, 10, 20, 99], dtype=np.int64),
        detection_levels=np.array([3, 3, 3, 5, 7], dtype=np.int32),
    )

    assert membership["cluster_cutout_cluster_ids"].tolist() == [0, 0, 1]
    assert membership["cluster_cutout_ids"].tolist() == [1001, 1002, 2001]
    assert membership["cluster_cutout_keyframe_ids"].tolist() == [10, 10, 20]
    assert membership["cluster_cutout_levels"].tolist() == [3, 3, 5]
    assert membership["cluster_cutout_observation_counts"].tolist() == [2, 1, 1]


def test_cluster_cutout_membership_uses_most_frequent_level() -> None:
    membership = build_cluster_cutout_membership(
        object_cluster_ids=np.array([0, 0, 0], dtype=np.int32),
        object_cutout_ids=np.array([1001, 1001, 1001], dtype=np.int64),
        object_keyframe_ids=np.array([10, 10, 10], dtype=np.int64),
        detection_levels=np.array([2, 2, 4], dtype=np.int32),
    )

    assert membership["cluster_cutout_levels"].tolist() == [2]
    assert membership["cluster_cutout_observation_counts"].tolist() == [3]


def test_cluster_cutout_membership_rejects_mixed_keyframes_per_cutout() -> None:
    with pytest.raises(ValueError, match="spans multiple keyframes"):
        build_cluster_cutout_membership(
            object_cluster_ids=np.array([0, 0], dtype=np.int32),
            object_cutout_ids=np.array([1001, 1001], dtype=np.int64),
            object_keyframe_ids=np.array([10, 11], dtype=np.int64),
            detection_levels=np.array([2, 2], dtype=np.int32),
        )


def test_cluster_cutout_membership_rejects_missing_level() -> None:
    with pytest.raises(ValueError, match="missing resolved level"):
        build_cluster_cutout_membership(
            object_cluster_ids=np.array([0], dtype=np.int32),
            object_cutout_ids=np.array([1001], dtype=np.int64),
            object_keyframe_ids=np.array([10], dtype=np.int64),
            detection_levels=np.array([UNRESOLVED_LEVEL_SENTINEL], dtype=np.int32),
        )
