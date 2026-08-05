import numpy as np

from pipeline.core.search_numpy import (
    compute_cosine_similarities,
    top_k_cutout,
    top_k_object_by_cutout,
)


def test_cosine_and_topk_cutout():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    s = compute_cosine_similarities(q, emb)
    assert s[0] > 0.99 and s[1] < 0.01 and s[2] < -0.99
    ids = np.array([10, 20, 30], dtype=np.int64)
    top = top_k_cutout(ids, s, 2)
    assert [t[0] for t in top] == ["10", "20"]


def test_object_aggregate_by_cutout():
    cutout_ids = np.array([1, 1, 2, 3], dtype=np.int64)
    scores = np.array([0.5, 0.9, 0.3, 0.4], dtype=np.float32)
    top = top_k_object_by_cutout(cutout_ids, scores, k=2)
    # cutout 1 -> max 0.9, cutout 2 -> 0.3, cutout 3 -> 0.4 → top2 ids 1 and 3
    assert [t[0] for t in top] == ["1", "3"]
