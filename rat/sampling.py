"""Farthest Point Sampling for anchor selection."""

from __future__ import annotations

import numpy as np


def farthest_point_sampling(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """Select k anchor indices via Farthest Point Sampling.

    Uses cosine distance (1 - cosine_similarity) as the distance metric.

    Parameters
    ----------
    embeddings : (N, D) L2-normalized embeddings
    k : number of anchors to select
    seed : random seed for initial point selection

    Returns
    -------
    (k,) int array of selected indices
    """
    N = embeddings.shape[0]
    if k > N:
        raise ValueError(f"k={k} exceeds number of embeddings N={N}")

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(N))]

    # cosine similarity matrix: (N,) distances to nearest selected point
    # Initialize with distance to first selected point
    sim = embeddings @ embeddings[selected[0]]  # (N,)
    min_dist = 1.0 - sim  # cosine distance

    for _ in range(k - 1):
        # Pick the point farthest from all selected points
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        # Update distances
        sim = embeddings @ embeddings[idx]  # (N,)
        dist = 1.0 - sim
        min_dist = np.minimum(min_dist, dist)

    return np.array(selected, dtype=np.intp)
