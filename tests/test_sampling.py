"""Tests for rat.sampling."""

import numpy as np
import pytest

from rat.sampling import farthest_point_sampling


@pytest.fixture
def normalized_embeddings():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


class TestFarthestPointSampling:
    def test_shape(self, normalized_embeddings):
        result = farthest_point_sampling(normalized_embeddings, k=10)
        assert result.shape == (10,)

    def test_no_duplicates(self, normalized_embeddings):
        result = farthest_point_sampling(normalized_embeddings, k=20)
        assert len(set(result.tolist())) == 20

    def test_reproducibility(self, normalized_embeddings):
        r1 = farthest_point_sampling(normalized_embeddings, k=10, seed=42)
        r2 = farthest_point_sampling(normalized_embeddings, k=10, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self, normalized_embeddings):
        r1 = farthest_point_sampling(normalized_embeddings, k=10, seed=0)
        r2 = farthest_point_sampling(normalized_embeddings, k=10, seed=1)
        # Different seeds should (almost certainly) give different results
        assert not np.array_equal(r1, r2)

    def test_k_equals_n(self, normalized_embeddings):
        N = normalized_embeddings.shape[0]
        result = farthest_point_sampling(normalized_embeddings, k=N)
        assert result.shape == (N,)
        assert len(set(result.tolist())) == N

    def test_k_exceeds_n_raises(self, normalized_embeddings):
        N = normalized_embeddings.shape[0]
        with pytest.raises(ValueError, match="exceeds"):
            farthest_point_sampling(normalized_embeddings, k=N + 1)

    def test_k_one(self, normalized_embeddings):
        result = farthest_point_sampling(normalized_embeddings, k=1)
        assert result.shape == (1,)
