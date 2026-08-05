"""Pure NumPy cosine retrieval (no torch)."""

from __future__ import annotations

from typing import List

import numpy as np

from pipeline.core.types import ResultRow


def compute_cosine_similarities(
    query_features: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query_features, dtype=np.float32).ravel()
    E = (
        embeddings
        if embeddings.dtype == np.float32
        else np.asarray(embeddings, dtype=np.float32)
    )
    if E.size == 0:
        return np.zeros(E.shape[0], dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0:
        return np.zeros(E.shape[0], dtype=np.float32)

    q_unit = (q / q_norm).astype(np.float32)
    dots = E @ q_unit  # single BLAS gemv, no copy

    # Fast path: if embeddings are pre-normalised (all row norms ≈ 1), skip division.
    # Check a small sample to avoid computing all norms on every call.
    sample = E[: min(64, E.shape[0])]
    sample_norms = np.linalg.norm(sample, axis=1)
    if np.allclose(sample_norms, 1.0, atol=1e-3):
        return dots

    emb_norms = np.linalg.norm(E, axis=1)
    mask = emb_norms > 0
    dots[mask] /= emb_norms[mask]
    dots[~mask] = 0.0
    return dots


def top_k_cutout(
    cutout_ids: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> List[ResultRow]:
    if cutout_ids.size == 0 or k <= 0:
        return []
    k = min(k, cutout_ids.size)
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(str(int(cutout_ids[i])), float(scores[i])) for i in top_idx]


def top_k_object_by_cutout(
    cutout_ids: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> List[ResultRow]:
    """One score per cutout_id = max similarity over object rows (matches
    legacy sqlite behavior)."""
    if cutout_ids.size == 0 or k <= 0:
        return []
    best: dict[int, float] = {}
    for cid, sc in zip(cutout_ids.tolist(), scores.tolist()):
        c = int(cid)
        if c not in best or sc > best[c]:
            best[c] = float(sc)
    if not best:
        return []
    ids_arr = np.array(list(best.keys()), dtype=np.int64)
    sc_arr = np.array(list(best.values()), dtype=np.float32)
    return top_k_cutout(ids_arr, sc_arr, k)
