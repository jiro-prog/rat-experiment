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


# ========================================
# 後処理: 類似度潰れ対策
# ========================================

def normalize_zscore(rel: np.ndarray) -> np.ndarray:
    """各クエリの相対表現ベクトルをz-score正規化する。"""
    mean = rel.mean(axis=1, keepdims=True)
    std = rel.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (rel - mean) / std


def normalize_rank(rel: np.ndarray) -> np.ndarray:
    """類似度の絶対値を順位に変換する。分布の形に依存しない。"""
    n, k = rel.shape
    ranks = np.zeros_like(rel)
    for i in range(n):
        order = np.argsort(-rel[i])  # 降順
        ranks[i, order] = np.arange(k, dtype=np.float64)
    # 0〜K-1を0〜1に正規化
    return 1.0 - ranks / (k - 1)


def normalize_topk_mask(rel: np.ndarray, top_k: int = 50) -> np.ndarray:
    """上位k個のアンカーのみ値を残し、それ以外を0にする。"""
    n, k_total = rel.shape
    masked = np.zeros_like(rel)
    for i in range(n):
        top_indices = np.argsort(-rel[i])[:top_k]
        masked[i, top_indices] = rel[i, top_indices]
    return masked


def normalize_softmax(rel: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """温度付きsoftmaxで差を増幅する。"""
    scaled = rel / temperature
    # 数値安定性のため各行の最大値を引く
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


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
