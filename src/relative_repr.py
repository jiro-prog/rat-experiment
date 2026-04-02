import numpy as np


def to_relative(embeddings: np.ndarray, anchor_embeddings: np.ndarray) -> np.ndarray:
    """
    embeddingをアンカーとのコサイン類似度ベクトル（相対表現）に変換する。

    embeddings: (N, D_model) - L2正規化済み
    anchor_embeddings: (K, D_model) - L2正規化済み
    returns: (N, K) - 各アンカーとのコサイン類似度
    """
    return embeddings @ anchor_embeddings.T


def to_relative_subset(
    embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    num_anchors: int,
) -> np.ndarray:
    """アンカーの先頭num_anchors個だけを使って相対表現に変換する。"""
    return to_relative(embeddings, anchor_embeddings[:num_anchors])
