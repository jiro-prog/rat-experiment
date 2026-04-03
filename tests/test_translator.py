"""Tests for rat.translator."""

import tempfile

import numpy as np
import pytest

from rat.translator import RATranslator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_normalized(rng, n, d):
    emb = rng.standard_normal((n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def translator(rng):
    """Fitted translator with K=20, D_a=16, D_b=32."""
    t = RATranslator(kernel="poly", normalize="auto")
    anchor_a = _make_normalized(rng, 20, 16)
    anchor_b = _make_normalized(rng, 20, 32)
    t.fit_embeddings(anchor_a, anchor_b)
    return t


class TestFitEmbeddings:
    def test_anchor_count_mismatch(self, rng):
        t = RATranslator()
        with pytest.raises(ValueError, match="Anchor count mismatch"):
            t.fit_embeddings(_make_normalized(rng, 20, 16), _make_normalized(rng, 15, 32))

    def test_l2_warning(self, rng):
        t = RATranslator()
        bad = rng.standard_normal((10, 16)) * 5
        good = _make_normalized(rng, 10, 32)
        with pytest.warns(UserWarning, match="L2-normalized"):
            t.fit_embeddings(bad, good)

    def test_method_chaining(self, rng):
        t = RATranslator()
        result = t.fit_embeddings(_make_normalized(rng, 10, 16), _make_normalized(rng, 10, 32))
        assert result is t


class TestTransform:
    def test_shape(self, translator, rng):
        emb = _make_normalized(rng, 50, 16)
        result = translator.transform(emb, "a")
        assert result.shape == (50, 20)

    def test_shape_b(self, translator, rng):
        emb = _make_normalized(rng, 50, 32)
        result = translator.transform(emb, "b")
        assert result.shape == (50, 20)

    def test_before_fit_raises(self, rng):
        t = RATranslator()
        with pytest.raises(RuntimeError, match="fit"):
            t.transform(_make_normalized(rng, 5, 16), "a")

    def test_invalid_source_raises(self, translator, rng):
        with pytest.raises(ValueError, match="source"):
            translator.transform(_make_normalized(rng, 5, 16), "c")

    def test_invalid_role_raises(self, translator, rng):
        with pytest.raises(ValueError, match="role"):
            translator.transform(_make_normalized(rng, 5, 16), "a", role="invalid")


class TestRoleOverride:
    def test_source_a_default_no_zscore(self, rng):
        """source='a' with role=None should behave like role='query' (no z-score)."""
        t = RATranslator(kernel="poly", normalize="always")
        anchor_a = _make_normalized(rng, 20, 16)
        anchor_b = _make_normalized(rng, 20, 32)
        t.fit_embeddings(anchor_a, anchor_b)

        emb = _make_normalized(rng, 10, 16)
        default_result = t.transform(emb, "a")
        query_result = t.transform(emb, "a", role="query")
        np.testing.assert_allclose(default_result, query_result)

    def test_source_b_default_zscore(self, rng):
        """source='b' with role=None should behave like role='db'."""
        t = RATranslator(kernel="poly", normalize="always")
        anchor_a = _make_normalized(rng, 20, 16)
        anchor_b = _make_normalized(rng, 20, 32)
        t.fit_embeddings(anchor_a, anchor_b)

        emb = _make_normalized(rng, 10, 32)
        default_result = t.transform(emb, "b")
        db_result = t.transform(emb, "b", role="db")
        np.testing.assert_allclose(default_result, db_result)

    def test_source_a_role_db_applies_zscore(self, rng):
        """source='a' with role='db' should apply z-score (override)."""
        t = RATranslator(kernel="poly", normalize="always")
        anchor_a = _make_normalized(rng, 20, 16)
        anchor_b = _make_normalized(rng, 20, 32)
        t.fit_embeddings(anchor_a, anchor_b)

        emb = _make_normalized(rng, 10, 16)
        query_result = t.transform(emb, "a", role="query")
        db_result = t.transform(emb, "a", role="db")
        # With normalize="always", db should differ from query due to z-score
        assert not np.allclose(query_result, db_result)

    def test_source_b_role_query_skips_zscore(self, rng):
        """source='b' with role='query' should skip z-score (override)."""
        t = RATranslator(kernel="poly", normalize="always")
        anchor_a = _make_normalized(rng, 20, 16)
        anchor_b = _make_normalized(rng, 20, 32)
        t.fit_embeddings(anchor_a, anchor_b)

        emb = _make_normalized(rng, 10, 32)
        query_result = t.transform(emb, "b", role="query")
        db_result = t.transform(emb, "b", role="db")
        # They should differ when normalize="always"
        assert not np.allclose(query_result, db_result)

    def test_db_role_produces_per_vector_zscore(self, rng):
        """role='db' should produce per-vector z-scored output (mean≈0 per row)."""
        t = RATranslator(kernel="poly", normalize="always")
        anchor_a = _make_normalized(rng, 20, 16)
        anchor_b = _make_normalized(rng, 20, 32)
        t.fit_embeddings(anchor_a, anchor_b)

        emb = _make_normalized(rng, 10, 32)
        result = t.transform(emb, "b", role="db")
        # Per-vector z-score: each row should have mean≈0, std≈1
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(axis=1), 1.0, atol=1e-10)


class TestRetrieve:
    def test_shape_and_descending(self, translator, rng):
        q = _make_normalized(rng, 3, 16)
        d = _make_normalized(rng, 100, 32)
        result = translator.retrieve(q, d, top_k=5)
        assert result["indices"].shape == (3, 5)
        assert result["scores"].shape == (3, 5)
        for i in range(3):
            assert np.all(result["scores"][i, :-1] >= result["scores"][i, 1:])

    def test_roundtrip_recall(self, rng):
        """Same embeddings as both A and B should give perfect recall."""
        D, K, N = 16, 20, 50
        t = RATranslator(kernel="poly", normalize="never")
        anchors = _make_normalized(rng, K, D)
        t.fit_embeddings(anchors, anchors)

        emb = _make_normalized(rng, N, D)
        result = t.retrieve(emb, emb, top_k=1)
        recall = np.mean(result["indices"][:, 0] == np.arange(N))
        assert recall >= 0.95, f"Recall@1={recall}, expected ≈ 1.0"


class TestEstimateCompatibility:
    def test_returns_fields(self, translator):
        result = translator.estimate_compatibility()
        assert "sim_mean_a" in result
        assert "sim_mean_b" in result
        assert isinstance(result["estimated_recall_at_1"], float)
        assert "compatibility" in result
        assert result["estimate_reliability"] == "low"

    def test_same_family_passthrough(self, translator):
        result = translator.estimate_compatibility(same_family=True)
        assert result["same_family_bonus"] == 10.0


class TestSaveLoad:
    def test_roundtrip(self, translator, rng):
        emb_a = _make_normalized(rng, 10, 16)
        emb_b = _make_normalized(rng, 10, 32)
        orig_a = translator.transform(emb_a, "a")
        orig_b = translator.transform(emb_b, "b")

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            translator.save(f.name)
            loaded = RATranslator.load(f.name)

        np.testing.assert_allclose(loaded.transform(emb_a, "a"), orig_a)
        np.testing.assert_allclose(loaded.transform(emb_b, "b"), orig_b)


class TestHubConsistency:
    def test_translator_matches_hub(self, rng):
        """RATranslator and RATHub should give identical results for same input."""
        from rat.hub import RATHub

        K, D_a, D_b = 20, 16, 32
        anchor_a = _make_normalized(rng, K, D_a)
        anchor_b = _make_normalized(rng, K, D_b)

        translator = RATranslator(kernel="poly", normalize="always")
        translator.fit_embeddings(anchor_a, anchor_b)

        hub = RATHub(kernel="poly", normalize="always")
        hub.set_anchors("__model_a__", anchor_a)
        hub.set_anchors("__model_b__", anchor_b)

        emb = _make_normalized(rng, 10, D_a)
        t_result = translator.transform(emb, "a")
        h_result = hub.transform("__model_a__", emb, role="query")
        np.testing.assert_allclose(t_result, h_result)
