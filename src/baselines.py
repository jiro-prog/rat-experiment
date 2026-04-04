"""
ベースライン手法: 直接アライメント（Procrustes / Ridge / Affine）

Experiment D1 用。アンカーペアの埋め込みから変換行列を学習し、
クエリ埋め込みを変換して検索精度を評価する。
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge


def fit_procrustes(
    anchor_A: np.ndarray,
    anchor_B: np.ndarray,
) -> np.ndarray:
    """
    Orthogonal Procrustes: W = argmin ||A @ W - B||^2  s.t. W^T W = I

    anchor_A: (K, D) — ソース空間のアンカー埋め込み
    anchor_B: (K, D) — ターゲット空間のアンカー埋め込み（同次元のみ）
    returns: W (D, D) 直交変換行列
    """
    W, _ = orthogonal_procrustes(anchor_A, anchor_B)
    return W


def transform_procrustes(
    query_emb: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """query_emb @ W でターゲット空間に変換。"""
    return query_emb @ W


def fit_ridge(
    anchor_A: np.ndarray,
    anchor_B: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Ridge Regression: W = (A^T A + αI)^{-1} A^T B

    次元が異なっていてもOK。
    anchor_A: (K, D_a), anchor_B: (K, D_b)
    returns: W (D_a, D_b)
    """
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(anchor_A, anchor_B)
    return model.coef_.T  # sklearn: (D_b, D_a) → transpose to (D_a, D_b)


def transform_ridge(
    query_emb: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """query_emb @ W でターゲット空間に変換。"""
    return query_emb @ W


def fit_affine(
    anchor_A: np.ndarray,
    anchor_B: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Affine: W + bias。Ridge + intercept。

    anchor_A: (K, D_a), anchor_B: (K, D_b)
    returns: (W (D_a, D_b), bias (D_b,))
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(anchor_A, anchor_B)
    W = model.coef_.T  # (D_a, D_b)
    bias = model.intercept_  # (D_b,)
    return W, bias


def transform_affine(
    query_emb: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """query_emb @ W + bias でターゲット空間に変換。"""
    return query_emb @ W + bias
