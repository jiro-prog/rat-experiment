"""Tests for rat.normalize."""

import numpy as np
import pytest

from rat.normalize import compute_sim_mean, normalize_zscore, recommend_zscore


class TestComputeSimMean:
    def test_orthogonal_vectors(self):
        emb = np.eye(4)
        result = compute_sim_mean(emb)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_identical_vectors(self):
        emb = np.ones((5, 3)) / np.sqrt(3)
        result = compute_sim_mean(emb)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_single_anchor(self):
        emb = np.array([[1.0, 0.0]])
        assert compute_sim_mean(emb) == 0.0


class TestNormalizeZscore:
    def test_basic(self):
        """Each row should have mean=0, std=1 after normalization."""
        rel = np.array([[1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0]])
        result = normalize_zscore(rel)
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(axis=1), 1.0, atol=1e-10)

    def test_single_vector(self):
        """Works with a single vector."""
        rel = np.array([[1.0, 3.0, 5.0, 7.0]])
        result = normalize_zscore(rel)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)

    def test_constant_vector(self):
        """Constant vector (std=0) doesn't cause division by zero."""
        rel = np.array([[5.0, 5.0, 5.0, 5.0]])
        result = normalize_zscore(rel)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_preserves_shape(self):
        rng = np.random.default_rng(0)
        rel = rng.standard_normal((50, 20))
        result = normalize_zscore(rel)
        assert result.shape == rel.shape

    def test_matches_src_implementation(self):
        """Exact match with src/relative_repr.py normalize_zscore."""
        rng = np.random.default_rng(42)
        rel = rng.standard_normal((30, 500)) * 3 + 10
        result = normalize_zscore(rel)
        # Reproduce src/ logic
        mean = rel.mean(axis=1, keepdims=True)
        std = rel.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        expected = (rel - mean) / std
        np.testing.assert_allclose(result, expected)


class TestRecommendZscore:
    def test_not_needed(self):
        assert recommend_zscore(0.14) == "not_needed"

    def test_recommended_at_threshold(self):
        assert recommend_zscore(0.15) == "recommended"

    def test_recommended_mid(self):
        assert recommend_zscore(0.40) == "recommended"

    def test_harmful_at_threshold(self):
        assert recommend_zscore(0.65) == "harmful"

    def test_harmful_above(self):
        assert recommend_zscore(0.90) == "harmful"

    def test_custom_harmful_threshold(self):
        assert recommend_zscore(0.50, harmful_threshold=0.50) == "harmful"
        assert recommend_zscore(0.49, harmful_threshold=0.50) == "recommended"

    def test_custom_threshold(self):
        assert recommend_zscore(0.09, threshold=0.10) == "not_needed"
        assert recommend_zscore(0.10, threshold=0.10) == "recommended"
