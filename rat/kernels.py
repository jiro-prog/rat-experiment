"""Kernel functions for computing relative representations."""

from __future__ import annotations

from functools import partial

import numpy as np


def poly_kernel(
    X: np.ndarray,
    A: np.ndarray,
    degree: int = 2,
    coef0: float = 1.0,
) -> np.ndarray:
    """Polynomial kernel: (X @ A.T + coef0) ** degree.

    Parameters
    ----------
    X : (N, D) input embeddings
    A : (K, D) anchor embeddings
    degree : polynomial degree
    coef0 : constant term

    Returns
    -------
    (N, K) kernel matrix
    """
    return (X @ A.T + coef0) ** degree


def cosine_kernel(
    X: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Cosine similarity kernel: X @ A.T (assumes L2-normalized inputs).

    Parameters
    ----------
    X : (N, D) input embeddings, L2-normalized
    A : (K, D) anchor embeddings, L2-normalized

    Returns
    -------
    (N, K) similarity matrix
    """
    return X @ A.T


def rbf_kernel(
    X: np.ndarray,
    A: np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """RBF (Gaussian) kernel: exp(-gamma * ||x - a||^2).

    Parameters
    ----------
    X : (N, D) input embeddings
    A : (K, D) anchor embeddings
    gamma : kernel width. None uses 1/D

    Returns
    -------
    (N, K) kernel matrix
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    # ||x - a||^2 = ||x||^2 + ||a||^2 - 2 x·a
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)  # (K, 1)
    dist_sq = X_sq + A_sq.T - 2.0 * (X @ A.T)    # (N, K)
    dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety
    return np.exp(-gamma * dist_sq)


def get_kernel(name: str, params: dict | None = None):
    """Factory: return a callable(X, A) -> np.ndarray for the named kernel.

    Parameters
    ----------
    name : "poly", "cosine", or "rbf"
    params : optional kwargs forwarded to the kernel function

    Returns
    -------
    callable(X, A) -> np.ndarray
    """
    params = params or {}
    kernels = {
        "poly": poly_kernel,
        "cosine": cosine_kernel,
        "rbf": rbf_kernel,
    }
    if name not in kernels:
        raise ValueError(f"Unknown kernel '{name}'. Choose from: {list(kernels.keys())}")
    return partial(kernels[name], **params)
