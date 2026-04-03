"""Z-score normalization and adaptive decision logic."""

from __future__ import annotations

import numpy as np


def compute_sim_mean(anchor_embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity among anchors (diagonal excluded).

    Parameters
    ----------
    anchor_embeddings : (K, D) L2-normalized

    Returns
    -------
    float : mean off-diagonal cosine similarity
    """
    sim = anchor_embeddings @ anchor_embeddings.T  # (K, K)
    K = sim.shape[0]
    if K < 2:
        return 0.0
    mask = ~np.eye(K, dtype=bool)
    return float(sim[mask].mean())


def normalize_zscore(relative_repr: np.ndarray) -> np.ndarray:
    """Per-vector z-score normalization.

    Each vector is independently normalized to mean=0, std=1
    across its K dimensions. Stateless — no precomputed statistics needed.

    Parameters
    ----------
    relative_repr : (N, K) or (1, K)

    Returns
    -------
    Normalized array, same shape as input.
    """
    mean = relative_repr.mean(axis=1, keepdims=True)
    std = relative_repr.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (relative_repr - mean) / std


def recommend_zscore(
    sim_mean: float,
    threshold: float = 0.15,
    harmful_threshold: float = 0.65,
) -> str:
    """Recommend z-score normalization based on sim_mean.

    Parameters
    ----------
    sim_mean : mean pairwise cosine similarity of anchors
    threshold : kept for backward compatibility but no longer used
        in the decision. Previously, sim_mean below this was "not_needed".
    harmful_threshold : above this, z-score is harmful

    Returns
    -------
    "harmful" or "recommended"
    """
    if sim_mean >= harmful_threshold:
        return "harmful"
    return "recommended"
