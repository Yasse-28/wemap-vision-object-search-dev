"""DINO-guided visual refinement for localized object clusters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.core.logging import logger


@dataclass
class VisualRefinementResult:
    cluster_ids: np.ndarray
    visual_similarity_scores: np.ndarray
    visual_candidate_mask: np.ndarray
    visual_assigned_mask: np.ndarray


class DinoImageEmbedder:
    def __init__(
        self,
        *,
        repo: str = "facebookresearch/dinov2",
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        image_size: int = 518,
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from torchvision import transforms

        self.torch = torch
        self.functional = functional
        self.device = (
            "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.model = torch.hub.load(repo, model_name).to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (int(image_size), int(image_size)),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        tensors = [self.transform(image.convert("RGB")) for image in images]
        batch = self.torch.stack(tensors, dim=0).to(self.device)
        with self.torch.inference_mode():
            features = self.model(batch)
            features = self.functional.normalize(features, dim=-1)
        return np.asarray(features.detach().float().cpu().numpy().astype(np.float32))


def compute_dino_embeddings(
    *,
    object_crops: Sequence[Image.Image | None],
    device: str,
    repo: str = "facebookresearch/dinov2",
    model_name: str = "dinov2_vitb14",
    image_size: int = 518,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    started_at = time.perf_counter()
    candidate_indices_getter = getattr(object_crops, "candidate_indices", None)
    if callable(candidate_indices_getter):
        valid_indices = [int(idx) for idx in candidate_indices_getter()]
    else:
        valid_indices = [
            idx for idx, crop in enumerate(object_crops) if crop is not None
        ]
    valid_mask = np.zeros(len(object_crops), dtype=bool)
    if not valid_indices:
        logger.info("DINO visual embeddings skipped: no object crops")
        return np.zeros((len(object_crops), 0), dtype=np.float32), valid_mask

    batch_size = max(1, int(batch_size))
    embedder = DinoImageEmbedder(
        repo=repo,
        model_name=model_name,
        device=device,
        image_size=image_size,
    )
    embeddings_by_index: dict[int, np.ndarray] = {}
    for offset in tqdm(
        range(0, len(valid_indices), int(batch_size)),
        desc="DINO visual embeddings",
        unit="batch",
    ):
        batch_indices = valid_indices[offset : offset + int(batch_size)]
        batch_items = [
            (idx, crop)
            for idx in batch_indices
            if (crop := object_crops[idx]) is not None
        ]
        if not batch_items:
            continue
        batch_images = [crop for _idx, crop in batch_items]
        batch_embeddings = embedder.encode(batch_images)
        for (idx, _crop), embedding in zip(batch_items, batch_embeddings):
            embeddings_by_index[idx] = embedding
            valid_mask[idx] = True

    if not embeddings_by_index:
        del embedder
        logger.info("DINO visual embeddings skipped: no readable object crops")
        return np.zeros((len(object_crops), 0), dtype=np.float32), valid_mask

    dim = len(next(iter(embeddings_by_index.values())))
    embeddings = np.zeros((len(object_crops), dim), dtype=np.float32)
    for idx, embedding in embeddings_by_index.items():
        embeddings[idx] = embedding

    del embedder
    logger.info(
        "DINO visual embeddings computed for %d/%d crops in %.2fs",
        int(valid_mask.sum()),
        len(object_crops),
        time.perf_counter() - started_at,
    )
    return embeddings, valid_mask


def _chunked_similarity_edges(
    embeddings: np.ndarray,
    threshold: float,
    chunk_size: int = 2000,
) -> list[tuple[int, int]]:
    """
    Compute similarity edges in chunks to avoid OOM for large clusters.
    Returns list of (i, j) pairs where similarity >= threshold.
    """
    n = embeddings.shape[0]
    edges: list[tuple[int, int]] = []
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk_i = embeddings[i_start:i_end]
        for j_start in range(i_start, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            chunk_j = embeddings[j_start:j_end]
            sim_block = chunk_i @ chunk_j.T
            rows, cols = np.where(sim_block >= threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                gi = i_start + r
                gj = j_start + c
                if gi < gj:
                    edges.append((gi, gj))
    return edges


def _similarity_components_from_edges(
    n_items: int,
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """Build connected components from edge list."""
    adjacency: list[set[int]] = [set() for _ in range(n_items)]
    for left_idx, right_idx in edges:
        adjacency[left_idx].add(right_idx)
        adjacency[right_idx].add(left_idx)

    components: list[list[int]] = []
    seen: set[int] = set()
    for idx in range(n_items):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _compute_medoid_scores_chunked(
    embeddings: np.ndarray,
    component: list[int],
    chunk_size: int = 2000,
) -> tuple[int, np.ndarray]:
    """Compute medoid and similarity scores for a component using chunked processing."""
    comp_indices = np.asarray(component, dtype=np.int64)
    comp_emb = embeddings[comp_indices]
    n = len(component)

    if n <= chunk_size:
        sim = comp_emb @ comp_emb.T
        medoid_local = int(sim.mean(axis=1).argmax())
        scores = sim[:, medoid_local].astype(np.float32)
        full_scores = np.zeros(embeddings.shape[0], dtype=np.float32)
        full_scores[comp_indices] = scores
        return comp_indices[medoid_local], full_scores[comp_indices]

    row_sums = np.zeros(n, dtype=np.float32)
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk_i = comp_emb[i_start:i_end]
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            chunk_j = comp_emb[j_start:j_end]
            sim_block = chunk_i @ chunk_j.T
            row_sums[i_start:i_end] += sim_block.sum(axis=1)

    medoid_local = int(row_sums.argmax())
    medoid_emb = comp_emb[medoid_local : medoid_local + 1]
    scores = (comp_emb @ medoid_emb.T).ravel().astype(np.float32)
    return comp_indices[medoid_local], scores


@dataclass
class _VisualRefineCounters:
    """Mutable tallies accumulated across per-cluster visual refinement."""

    next_label: int
    changed_clusters: int = 0
    noise_count: int = 0
    split_count: int = 0
    large_cluster_count: int = 0


def _refine_one_cluster(
    *,
    label: int,
    refined: np.ndarray,
    visual_scores: np.ndarray,
    assigned_mask: np.ndarray,
    candidate_mask: np.ndarray,
    visual_embeddings: np.ndarray,
    counters: _VisualRefineCounters,
    similarity_threshold: float,
    min_cluster_observations: int,
    min_component_observations: int,
    large_cluster_threshold: int,
    chunk_size: int,
) -> None:
    """Split/clean one cluster by DINO visual similarity.

    Mutates refined / visual_scores / assigned_mask in place and updates
    counters. Connected components below min_component_observations are dropped
    to noise; multiple supported components split the cluster into new labels.
    """
    indices = np.where(candidate_mask & (refined == int(label)))[0]
    if indices.size < int(min_cluster_observations):
        assigned_mask[indices] = True
        visual_scores[indices] = _cluster_medoid_scores(visual_embeddings[indices])
        return

    embeddings = np.asarray(visual_embeddings[indices], dtype=np.float32)

    use_chunked = indices.size > large_cluster_threshold
    if use_chunked:
        counters.large_cluster_count += 1
        edges = _chunked_similarity_edges(
            embeddings, float(similarity_threshold), chunk_size
        )
        components = _similarity_components_from_edges(indices.size, edges)
    else:
        sim = embeddings @ embeddings.T
        components = _similarity_components(sim, float(similarity_threshold))

    if len(components) == 1:
        if use_chunked:
            _, comp_scores = _compute_medoid_scores_chunked(
                embeddings, components[0], chunk_size
            )
            visual_scores[indices] = comp_scores
        else:
            medoid_local = _component_medoid(sim, components[0])
            visual_scores[indices] = sim[:, medoid_local].astype(np.float32)
        assigned_mask[indices] = True
        return

    supported = [
        component
        for component in components
        if len(component) >= int(min_component_observations)
    ]
    if len(supported) <= 1:
        if supported:
            keep_local = supported[0]
            keep_indices = indices[np.asarray(keep_local, dtype=np.int64)]
            drop_local = sorted(set(range(indices.size)) - set(keep_local))
            drop_indices = indices[np.asarray(drop_local, dtype=np.int64)]
            refined[drop_indices] = -1
            assigned_mask[keep_indices] = True
            counters.noise_count += int(drop_indices.size)
            counters.changed_clusters += 1
            if use_chunked:
                _, comp_scores = _compute_medoid_scores_chunked(
                    embeddings, keep_local, chunk_size
                )
                visual_scores[indices[np.asarray(keep_local, dtype=np.int64)]] = (
                    comp_scores
                )
            else:
                medoid_local = _component_medoid(sim, keep_local)
                visual_scores[indices] = sim[:, medoid_local].astype(np.float32)
        else:
            assigned_mask[indices] = True
            visual_scores[indices] = _cluster_medoid_scores(
                visual_embeddings[indices]
            )
        return

    counters.changed_clusters += 1
    counters.split_count += len(supported) - 1
    refined[indices] = -1
    for component_idx, component in enumerate(supported):
        component_indices = indices[np.asarray(component, dtype=np.int64)]
        target_label = int(label) if component_idx == 0 else counters.next_label
        if component_idx > 0:
            counters.next_label += 1
        refined[component_indices] = target_label
        assigned_mask[component_indices] = True
        if use_chunked:
            _, comp_scores = _compute_medoid_scores_chunked(
                embeddings, component, chunk_size
            )
            visual_scores[component_indices] = comp_scores
        else:
            medoid_local = _component_medoid(sim, component)
            visual_scores[component_indices] = sim[
                np.asarray(component, dtype=np.int64), medoid_local
            ].astype(np.float32)

    supported_locals = {local for component in supported for local in component}
    unsupported = sorted(set(range(indices.size)) - supported_locals)
    if unsupported:
        unsupported_indices = indices[np.asarray(unsupported, dtype=np.int64)]
        refined[unsupported_indices] = -1
        counters.noise_count += int(unsupported_indices.size)
        if use_chunked:
            unsup_emb = embeddings[np.asarray(unsupported, dtype=np.int64)]
            nearest_scores = np.zeros(len(unsupported), dtype=np.float32)
            for i, ue in enumerate(unsup_emb):
                max_sim = float((embeddings @ ue).max())
                nearest_scores[i] = max_sim
            visual_scores[unsupported_indices] = nearest_scores
        else:
            nearest_scores = sim[np.asarray(unsupported, dtype=np.int64)].max(axis=1)
            visual_scores[unsupported_indices] = nearest_scores.astype(np.float32)


def refine_clusters_with_visual_features(
    *,
    cluster_ids: np.ndarray,
    valid_mask: np.ndarray,
    visual_embeddings: np.ndarray,
    visual_embedding_valid_mask: np.ndarray,
    similarity_threshold: float = 0.35,
    min_cluster_observations: int = 3,
    min_component_observations: int = 2,
    large_cluster_threshold: int = 5000,
    chunk_size: int = 2000,
) -> VisualRefinementResult:
    refined = np.asarray(cluster_ids, dtype=np.int32).copy()
    n_objects = int(refined.shape[0])
    visual_scores = np.full(n_objects, np.nan, dtype=np.float32)
    candidate_mask = (
        valid_mask
        & (cluster_ids >= 0)
        & np.asarray(visual_embedding_valid_mask, dtype=bool)
    )
    assigned_mask = np.zeros(n_objects, dtype=bool)

    labels = np.unique(refined[candidate_mask])
    labels = labels[labels >= 0]
    counters = _VisualRefineCounters(
        next_label=int(refined[refined >= 0].max(initial=-1)) + 1
    )

    for label in tqdm(labels.tolist(), desc="DINO visual refinement", unit="cluster"):
        _refine_one_cluster(
            label=int(label),
            refined=refined,
            visual_scores=visual_scores,
            assigned_mask=assigned_mask,
            candidate_mask=candidate_mask,
            visual_embeddings=visual_embeddings,
            counters=counters,
            similarity_threshold=similarity_threshold,
            min_cluster_observations=min_cluster_observations,
            min_component_observations=min_component_observations,
            large_cluster_threshold=large_cluster_threshold,
            chunk_size=chunk_size,
        )

    refined = _relabel_compact(refined)
    logger.info(
        "DINO visual refinement applied: candidate_clusters=%d"
        " changed_clusters=%d splits=%d noise=%d clusters_before=%d"
        " clusters_after=%d large_clusters_chunked=%d",
        int(labels.size),
        counters.changed_clusters,
        counters.split_count,
        counters.noise_count,
        int(np.unique(cluster_ids[cluster_ids >= 0]).size),
        int(np.unique(refined[refined >= 0]).size),
        counters.large_cluster_count,
    )
    return VisualRefinementResult(
        cluster_ids=refined,
        visual_similarity_scores=visual_scores,
        visual_candidate_mask=candidate_mask,
        visual_assigned_mask=assigned_mask,
    )


def _similarity_components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    n_items = int(similarity.shape[0])
    adjacency: list[set[int]] = [set() for _ in range(n_items)]
    for left_idx in range(n_items):
        for right_idx in range(left_idx + 1, n_items):
            if float(similarity[left_idx, right_idx]) >= threshold:
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

    components: list[list[int]] = []
    seen: set[int] = set()
    for idx in range(n_items):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _component_medoid(similarity: np.ndarray, component: Iterable[int]) -> int:
    indices = np.asarray(list(component), dtype=np.int64)
    if indices.size == 1:
        return int(indices[0])
    component_sim = similarity[np.ix_(indices, indices)]
    return int(indices[int(component_sim.mean(axis=1).argmax())])


def _cluster_medoid_scores(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if embeddings.shape[0] == 1:
        return np.ones(1, dtype=np.float32)
    similarity = embeddings @ embeddings.T
    medoid = int(similarity.mean(axis=1).argmax())
    return np.asarray(similarity[:, medoid].astype(np.float32))


def _relabel_compact(cluster_ids: np.ndarray) -> np.ndarray:
    relabeled = np.full(cluster_ids.shape, -1, dtype=np.int32)
    labels = np.unique(cluster_ids)
    labels = labels[labels >= 0]
    for new_label, old_label in enumerate(labels.tolist()):
        relabeled[cluster_ids == old_label] = new_label
    return relabeled
