"""Tests for rat.hub."""

import tempfile

import numpy as np
import pytest

from rat.hub import RATHub


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_normalized(rng, n, d):
    emb = rng.standard_normal((n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def hub_3models(rng):
    """Hub with 3 models registered, K=20 anchors each."""
    hub = RATHub(kernel="poly")
    hub.set_anchors("model_x", _make_normalized(rng, 20, 32))
    hub.set_anchors("model_y", _make_normalized(rng, 20, 64))
    hub.set_anchors("model_z", _make_normalized(rng, 20, 16))
    return hub


class TestSetAnchors:
    def test_anchor_count_mismatch(self, rng):
        hub = RATHub()
        hub.set_anchors("a", _make_normalized(rng, 20, 32))
        with pytest.raises(ValueError, match="Anchor count mismatch"):
            hub.set_anchors("b", _make_normalized(rng, 15, 64))

    def test_l2_warning(self, rng):
        hub = RATHub()
        bad_emb = rng.standard_normal((10, 16)) * 5  # not normalized
        with pytest.warns(UserWarning, match="L2-normalized"):
            hub.set_anchors("bad", bad_emb)


class TestTransform:
    def test_shape(self, hub_3models, rng):
        emb = _make_normalized(rng, 50, 32)
        result = hub_3models.transform("model_x", emb, role="query")
        assert result.shape == (50, 20)

    def test_any_pair(self, hub_3models, rng):
        # Can transform from any registered model
        for name, dim in [("model_x", 32), ("model_y", 64), ("model_z", 16)]:
            emb = _make_normalized(rng, 5, dim)
            result = hub_3models.transform(name, emb)
            assert result.shape == (5, 20)

    def test_unregistered_model_raises(self, hub_3models, rng):
        emb = _make_normalized(rng, 5, 32)
        with pytest.raises(RuntimeError, match="not registered"):
            hub_3models.transform("unknown", emb)

    def test_invalid_role_raises(self, hub_3models, rng):
        emb = _make_normalized(rng, 5, 32)
        with pytest.raises(ValueError, match="role"):
            hub_3models.transform("model_x", emb, role="invalid")


class TestRetrieve:
    def test_returns_correct_shape(self, hub_3models, rng):
        q = _make_normalized(rng, 3, 32)
        d = _make_normalized(rng, 100, 64)
        result = hub_3models.retrieve(q, d, "model_x", "model_y", top_k=5)
        assert result["indices"].shape == (3, 5)
        assert result["scores"].shape == (3, 5)

    def test_scores_descending(self, hub_3models, rng):
        q = _make_normalized(rng, 2, 32)
        d = _make_normalized(rng, 50, 64)
        result = hub_3models.retrieve(q, d, "model_x", "model_y", top_k=10)
        for i in range(2):
            scores = result["scores"][i]
            assert np.all(scores[:-1] >= scores[1:])


class TestEstimateCompatibility:
    def test_returns_fields(self, hub_3models):
        result = hub_3models.estimate_compatibility("model_x", "model_y")
        assert "sim_mean_a" in result
        assert "sim_mean_b" in result
        assert "max_sim_mean" in result
        assert "compatibility" in result
        assert "compatibility_description" in result
        assert isinstance(result["estimated_recall_at_1"], float)
        assert isinstance(result["confidence_band"], tuple)
        assert len(result["confidence_band"]) == 2
        assert result["estimate_reliability"] == "low"
        assert result["same_family_bonus"] is None
        assert result["z_score_recommendation"] in ("recommended", "harmful")
        assert isinstance(result["warnings"], list)

    def test_unregistered_raises(self, hub_3models):
        with pytest.raises(RuntimeError):
            hub_3models.estimate_compatibility("model_x", "nonexistent")

    def test_tier_high(self, rng):
        """max_sim_mean < 0.45 → high tier."""
        hub = RATHub(kernel="poly")
        # Use very low-dim embeddings that are orthogonal → low sim_mean
        anchors_a = _make_normalized(rng, 20, 128)
        anchors_b = _make_normalized(rng, 20, 128)
        hub.set_anchors("a", anchors_a)
        hub.set_anchors("b", anchors_b)
        result = hub.estimate_compatibility("a", "b")
        # Random 128-dim vectors have sim_mean close to 0 → should be "high"
        assert result["compatibility"] == "high"

    def test_tier_boundaries(self):
        """Test tier computation directly."""
        assert RATHub._compute_tier(0.44) == "high"
        assert RATHub._compute_tier(0.45) == "moderate"
        assert RATHub._compute_tier(0.71) == "moderate"
        assert RATHub._compute_tier(0.72) == "low"

    def test_same_family_upgrades_tier(self):
        assert RATHub._upgrade_tier("low") == "moderate"
        assert RATHub._upgrade_tier("moderate") == "high"
        assert RATHub._upgrade_tier("high") == "high"

    def test_same_family_bonus(self, hub_3models):
        result_no_sf = hub_3models.estimate_compatibility("model_x", "model_y")
        result_sf = hub_3models.estimate_compatibility("model_x", "model_y", same_family=True)
        assert result_no_sf["same_family_bonus"] is None
        assert result_sf["same_family_bonus"] == 10.0
        # same_family caps at 99.0, base caps at 100.0, so sf can be <= base
        assert result_sf["estimated_recall_at_1"] <= 99.0

    def test_same_family_clipping(self, rng):
        """same_family bonus should not exceed 99.0."""
        hub = RATHub(kernel="poly")
        # Low sim_mean → high base estimate → bonus could push past 99
        anchors = _make_normalized(rng, 20, 256)
        hub.set_anchors("a", anchors)
        hub.set_anchors("b", _make_normalized(rng, 20, 256))
        result = hub.estimate_compatibility("a", "b", same_family=True)
        assert result["estimated_recall_at_1"] <= 99.0

    def test_confidence_band_clipping(self, hub_3models):
        result = hub_3models.estimate_compatibility("model_x", "model_y")
        lo, hi = result["confidence_band"]
        assert lo >= 0.0
        assert hi <= 100.0
        assert lo <= result["estimated_recall_at_1"] <= hi


class TestRetrieveMulti:
    def test_returns_correct_shape(self, hub_3models, rng):
        q = _make_normalized(rng, 3, 32)
        db1 = _make_normalized(rng, 40, 64)
        db2 = _make_normalized(rng, 30, 16)
        result = hub_3models.retrieve_multi(
            q,
            databases=[(db1, "model_y"), (db2, "model_z")],
            query_model="model_x",
            top_k=5,
        )
        assert result["indices"].shape == (3, 5)
        assert result["scores"].shape == (3, 5)
        assert result["db_labels"].shape == (3, 5)

    def test_db_labels_valid(self, hub_3models, rng):
        q = _make_normalized(rng, 2, 32)
        db1 = _make_normalized(rng, 20, 64)
        db2 = _make_normalized(rng, 15, 16)
        result = hub_3models.retrieve_multi(
            q,
            databases=[(db1, "model_y"), (db2, "model_z")],
            query_model="model_x",
            top_k=10,
        )
        # db_labels should be 0 or 1
        assert np.all(result["db_labels"] >= 0)
        assert np.all(result["db_labels"] <= 1)

    def test_scores_descending(self, hub_3models, rng):
        q = _make_normalized(rng, 2, 32)
        db1 = _make_normalized(rng, 30, 64)
        db2 = _make_normalized(rng, 20, 16)
        result = hub_3models.retrieve_multi(
            q,
            databases=[(db1, "model_y"), (db2, "model_z")],
            query_model="model_x",
            top_k=10,
        )
        for i in range(2):
            scores = result["scores"][i]
            assert np.all(scores[:-1] >= scores[1:])

    def test_global_indices_valid(self, hub_3models, rng):
        q = _make_normalized(rng, 2, 32)
        db1 = _make_normalized(rng, 25, 64)
        db2 = _make_normalized(rng, 20, 16)
        result = hub_3models.retrieve_multi(
            q,
            databases=[(db1, "model_y"), (db2, "model_z")],
            query_model="model_x",
            top_k=10,
        )
        assert np.all(result["indices"] >= 0)
        assert np.all(result["indices"] < 45)  # 25 + 20


class TestSaveLoad:
    def test_roundtrip(self, hub_3models, rng):
        emb = _make_normalized(rng, 10, 32)
        original = hub_3models.transform("model_x", emb, role="db")

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            hub_3models.save(f.name)
            loaded = RATHub.load(f.name)

        restored = loaded.transform("model_x", emb, role="db")
        np.testing.assert_allclose(original, restored)
