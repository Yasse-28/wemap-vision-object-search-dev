from __future__ import annotations

import numpy as np

from pipeline.offline.refine.visual_refine import refine_clusters_with_visual_features


def test_visual_refinement_splits_supported_components_and_drops_singleton_outlier():
    cluster_ids = np.asarray([0, 0, 0, 0, 0], dtype=np.int32)
    valid_mask = np.ones(5, dtype=bool)
    visual_valid_mask = np.ones(5, dtype=bool)
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.98, 0.02],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = refine_clusters_with_visual_features(
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        visual_embeddings=embeddings,
        visual_embedding_valid_mask=visual_valid_mask,
        similarity_threshold=0.9,
        min_cluster_observations=3,
        min_component_observations=2,
    )

    assert result.cluster_ids[0] == result.cluster_ids[1]
    assert result.cluster_ids[2] == result.cluster_ids[3]
    assert result.cluster_ids[0] != result.cluster_ids[2]
    assert result.cluster_ids[4] == -1


def test_visual_refinement_preserves_small_clusters():
    result = refine_clusters_with_visual_features(
        cluster_ids=np.asarray([0, 0], dtype=np.int32),
        valid_mask=np.ones(2, dtype=bool),
        visual_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        visual_embedding_valid_mask=np.ones(2, dtype=bool),
        similarity_threshold=0.9,
        min_cluster_observations=3,
        min_component_observations=2,
    )

    assert result.cluster_ids.tolist() == [0, 0]
