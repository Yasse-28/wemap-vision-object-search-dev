"""Agglomerative clustering of a signed graph, with average linkage (GASP).

The association already in production contracts a signed graph greedily
(`greedy_additive_edge_contraction`, i.e. GAEC), where the cost between two clusters
is the **sum** of the edges between them. That is what lets a single strong edge act
as a bridge: sum grows with the number of edges, so a large cluster attracts
everything near it and the partition chains — the failure mode that ended the keypoint
graph experiments of 2026-08-11 (a component of 486 detections out of 500).

Average linkage removes the incentive: the cost between two clusters is the *mean* of
the edges between them, so adding members does not by itself make a cluster more
attractive, and one bridge among fifty repulsive edges loses. Same graph, same costs,
one line of difference in the update rule — which is why this is worth trying before
anything heavier.

Cannot-link pairs are absolute and propagate through merges: if A cannot link with C,
then neither can any cluster containing A. That is the mechanism that makes repulsive
evidence survive transitivity, instead of being outvoted by a chain of attractions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Pair:
    total: float
    count: int

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def average_linkage_clusters(
    node_count: int,
    edges: list[tuple[int, int, float]],
    forbidden_pairs: set[tuple[int, int]] | None = None,
) -> list[int]:
    """Merge the most attractive cluster pair until no positive average edge remains.

    Args:
        node_count: Number of nodes; nodes with no edge come back as singletons.
        edges: `(i, j, cost)` with cost > 0 attractive and < 0 repulsive.
        forbidden_pairs: Node pairs that must never share a cluster.

    Returns:
        One compact label per node.
    """
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    pairs: dict[tuple[int, int], _Pair] = {}
    for i, j, cost in edges:
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        entry = pairs.get(key)
        if entry is None:
            pairs[key] = _Pair(float(cost), 1)
        else:
            entry.total += float(cost)
            entry.count += 1

    blocked: set[tuple[int, int]] = set()
    for i, j in forbidden_pairs or ():
        blocked.add((min(i, j), max(i, j)))

    while True:
        best_key: tuple[int, int] | None = None
        best_average = 0.0
        for key, entry in pairs.items():
            if key in blocked:
                continue
            average = entry.average
            if average > best_average:
                best_average, best_key = average, key
        if best_key is None:
            break

        left, right = best_key
        parent[right] = left
        # Re-key every edge touching the absorbed cluster onto the survivor, summing
        # totals and counts so the average stays the mean over *all* original edges
        # between the two sides — that is the whole point of average linkage.
        moved: dict[tuple[int, int], _Pair] = {}
        for key in list(pairs):
            if right not in key:
                continue
            entry = pairs.pop(key)
            other = key[0] if key[1] == right else key[1]
            if other == left:
                continue
            new_key = (min(left, other), max(left, other))
            existing = moved.get(new_key) or pairs.get(new_key)
            if existing is None:
                moved[new_key] = _Pair(entry.total, entry.count)
            else:
                existing.total += entry.total
                existing.count += entry.count
                moved[new_key] = existing
        pairs.pop(best_key, None)
        pairs.update(moved)

        # Cannot-link is inherited: a merged cluster carries every prohibition of its
        # parts, which is what stops a chain of attractions from routing around it.
        for key in list(blocked):
            if right not in key:
                continue
            other = key[0] if key[1] == right else key[1]
            blocked.discard(key)
            if other != left:
                blocked.add((min(left, other), max(left, other)))

    roots: dict[int, int] = {}
    labels: list[int] = []
    for node in range(node_count):
        root = find(node)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels
