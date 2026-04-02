import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def to_relative(
    embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    kernel: str = "cosine",
    **kernel_params,
) -> np.ndarray:
    """
    embeddingをアンカーとのカーネル類似度ベクトル（相対表現）に変換する。

    embeddings: (N, D_model) - L2正規化済み
    anchor_embeddings: (K, D_model) - L2正規化済み
    kernel: "cosine", "rbf", "poly"
    returns: (N, K)
    """
    if kernel == "cosine":
        return _kernel_cosine(embeddings, anchor_embeddings)
    elif kernel == "rbf":
        gamma = kernel_params.get("gamma", None)
        return _kernel_rbf(embeddings, anchor_embeddings, gamma)
    elif kernel == "poly":
        degree = kernel_params.get("degree", 2)
        coef0 = kernel_params.get("coef0", 1.0)
        return _kernel_poly(embeddings, anchor_embeddings, degree, coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _kernel_cosine(embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """コサイン類似度（正規化済みならdot product）。"""
    return embeddings @ anchors.T


def _kernel_rbf(
    embeddings: np.ndarray,
    anchors: np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """
    RBFカーネル: exp(-gamma * ||x - a||^2)
    gamma=None の場合、中央値ヒューリスティックで設定。
    """
    dists_sq = euclidean_distances(embeddings, anchors, squared=True)  # (N, K)

    if gamma is None:
        # 中央値ヒューリスティック: gamma = 1 / (2 * median(||x - a||^2))
        median_dist_sq = np.median(dists_sq)
        if median_dist_sq == 0:
            median_dist_sq = 1.0
        gamma = 1.0 / (2.0 * median_dist_sq)

    return np.exp(-gamma * dists_sq)


def _kernel_poly(
    embeddings: np.ndarray,
    anchors: np.ndarray,
    degree: int = 2,
    coef0: float = 1.0,
) -> np.ndarray:
    """多項式カーネル: (x・a + coef0)^degree"""
    dot = embeddings @ anchors.T
    return (dot + coef0) ** degree


def to_relative_subset(
    embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    num_anchors: int,
    kernel: str = "cosine",
    **kernel_params,
) -> np.ndarray:
    """アンカーの先頭num_anchors個だけを使って相対表現に変換する。"""
    return to_relative(
        embeddings, anchor_embeddings[:num_anchors], kernel=kernel, **kernel_params
    )
