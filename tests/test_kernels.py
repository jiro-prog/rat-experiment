"""Tests for rat.kernels."""

import numpy as np
import pytest

from rat.kernels import cosine_kernel, get_kernel, poly_kernel, rbf_kernel


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def data(rng):
    X = rng.standard_normal((5, 4))
    A = rng.standard_normal((3, 4))
    return X, A


class TestPolyKernel:
    def test_shape(self, data):
        X, A = data
        result = poly_kernel(X, A)
        assert result.shape == (5, 3)

    def test_hand_calculation(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        # (X @ A.T + 1)^2
        # X @ A.T = [[1, 0], [0, 1]]
        # + 1 = [[2, 1], [1, 2]]
        # ^2 = [[4, 1], [1, 4]]
        result = poly_kernel(X, A, degree=2, coef0=1.0)
        expected = np.array([[4.0, 1.0], [1.0, 4.0]])
        np.testing.assert_allclose(result, expected)

    def test_custom_params(self, data):
        X, A = data
        result = poly_kernel(X, A, degree=3, coef0=0.5)
        expected = (X @ A.T + 0.5) ** 3
        np.testing.assert_allclose(result, expected)


class TestCosineKernel:
    def test_matches_matmul(self, data):
        X, A = data
        result = cosine_kernel(X, A)
        np.testing.assert_allclose(result, X @ A.T)

    def test_shape(self, data):
        X, A = data
        assert cosine_kernel(X, A).shape == (5, 3)


class TestRbfKernel:
    def test_identical_vectors(self):
        X = np.array([[1.0, 0.0, 0.0]])
        result = rbf_kernel(X, X)
        np.testing.assert_allclose(result, [[1.0]])

    def test_orthogonal_vectors(self):
        X = np.array([[1.0, 0.0]])
        A = np.array([[0.0, 1.0]])
        gamma = 1.0
        # ||x - a||^2 = 2.0
        expected = np.exp(-gamma * 2.0)
        result = rbf_kernel(X, A, gamma=gamma)
        np.testing.assert_allclose(result, [[expected]])

    def test_default_gamma(self, data):
        X, A = data
        result = rbf_kernel(X, A)
        expected = rbf_kernel(X, A, gamma=1.0 / X.shape[1])
        np.testing.assert_allclose(result, expected)

    def test_shape(self, data):
        X, A = data
        assert rbf_kernel(X, A).shape == (5, 3)


class TestGetKernel:
    def test_returns_callable(self):
        fn = get_kernel("poly")
        assert callable(fn)

    def test_poly_with_params(self, data):
        X, A = data
        fn = get_kernel("poly", {"degree": 3, "coef0": 0.5})
        result = fn(X, A)
        expected = poly_kernel(X, A, degree=3, coef0=0.5)
        np.testing.assert_allclose(result, expected)

    def test_cosine(self, data):
        X, A = data
        fn = get_kernel("cosine")
        np.testing.assert_allclose(fn(X, A), cosine_kernel(X, A))

    def test_rbf(self, data):
        X, A = data
        fn = get_kernel("rbf", {"gamma": 0.5})
        np.testing.assert_allclose(fn(X, A), rbf_kernel(X, A, gamma=0.5))

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            get_kernel("invalid")
