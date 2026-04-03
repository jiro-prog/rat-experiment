"""Integration test: reproduce experiment results via rat/ library.

Loads cached embeddings from data/d2_matrix/, runs the same pipeline
as experiments/run_d2a_matrix.py, and verifies that rat/ produces
identical results to src/.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rat.kernels import poly_kernel
from rat.normalize import normalize_zscore, compute_sim_mean
from rat.translator import RATranslator
from rat.hub import RATHub

DATA_DIR = Path(__file__).parent.parent / "data" / "d2_matrix"
RESULTS_PATH = Path(__file__).parent.parent / "results" / "d2a_matrix.json"
K = 500


def _skip_if_no_data():
    if not DATA_DIR.exists():
        pytest.skip("d2_matrix cache not found")


def _load_emb(label: str):
    """Load query and candidate embeddings for a model label."""
    q = np.load(DATA_DIR / f"query_{label}.npy")
    c = np.load(DATA_DIR / f"cand_{label}.npy")
    return q, c


def _fps_src_compat(candidate_embeddings: np.ndarray, k: int, seed: int = 42):
    """FPS matching src/anchor_sampler.py (uses legacy RandomState)."""
    rng = np.random.RandomState(seed)
    n = len(candidate_embeddings)
    first = rng.randint(n)
    selected = [first]

    min_sim = candidate_embeddings @ candidate_embeddings[first]

    for _ in range(k - 1):
        min_sim_copy = min_sim.copy()
        for idx in selected:
            min_sim_copy[idx] = np.inf
        next_idx = int(np.argmin(min_sim_copy))
        selected.append(next_idx)
        new_sim = candidate_embeddings @ candidate_embeddings[next_idx]
        min_sim = np.maximum(min_sim, new_sim)

    return np.array(selected)


def _src_evaluate_retrieval(rel_A, rel_B):
    """Reproduce src/evaluator.py evaluate_retrieval logic."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(rel_A, rel_B)
    ranks = []
    for i in range(len(rel_A)):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


# ── Test: kernel + z-score exact match with src/ ──

class TestKernelMatchesSrc:
    """Verify rat/ kernels produce identical output to src/ to_relative."""

    def test_poly_kernel_matches_src(self):
        _skip_if_no_data()
        query_a, cand_a = _load_emb("A")
        fps_idx = _fps_src_compat(cand_a, K)
        anchors = cand_a[fps_idx]

        # src/ style
        dot = query_a @ anchors.T
        src_result = (dot + 1.0) ** 2

        # rat/ style
        rat_result = poly_kernel(query_a, anchors, degree=2, coef0=1.0)

        np.testing.assert_allclose(rat_result, src_result, rtol=1e-10)

    def test_zscore_matches_src(self):
        _skip_if_no_data()
        query_b, cand_b = _load_emb("B")
        query_a, cand_a = _load_emb("A")
        fps_idx = _fps_src_compat(cand_a, K)
        anchors_b = cand_b[fps_idx]

        rel = poly_kernel(query_b, anchors_b, degree=2, coef0=1.0)

        # src/ z-score
        mean = rel.mean(axis=1, keepdims=True)
        std = rel.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        src_z = (rel - mean) / std

        # rat/ z-score
        rat_z = normalize_zscore(rel)

        np.testing.assert_allclose(rat_z, src_z, rtol=1e-10)


# ── Test: RATranslator reproduces paper numbers ──

class TestReproducePaperResults:
    """Verify key paper results can be reproduced via RATranslator."""

    @pytest.fixture
    def reference_results(self):
        """Load d2a_matrix.json reference results."""
        import json
        if not RESULTS_PATH.exists():
            pytest.skip("d2a_matrix.json not found")
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        # Build lookup: (query_label, db_label) -> metrics
        lookup = {}
        for r in data["pair_results"]:
            lookup[(r["query"], r["db"])] = r
        return lookup

    def _run_pair(self, query_label, db_label):
        """Run RAT pipeline for a pair and return metrics."""
        _skip_if_no_data()
        query_x, cand_x = _load_emb(query_label)
        query_y, cand_y = _load_emb(db_label)

        # FPS on query side (matching src/ behavior)
        fps_idx = _fps_src_compat(cand_x, K)
        anchor_x = cand_x[fps_idx]
        anchor_y = cand_y[fps_idx]

        # --- RATranslator pathway ---
        t = RATranslator(kernel="poly", normalize="never")
        t.fit_embeddings(anchor_x, anchor_y)

        rel_x = t.transform(query_x, "a")  # query side, no z-score
        rel_y = t.transform(query_y, "b")  # db side, but normalize="never"

        baseline = _src_evaluate_retrieval(rel_x, rel_y)

        # z-score DB-side only
        rel_y_z = normalize_zscore(rel_y)
        zscore_db = _src_evaluate_retrieval(rel_x, rel_y_z)

        return baseline, zscore_db

    @pytest.mark.parametrize("ql,dl,expected_baseline,expected_zscore", [
        # Core paper pairs (from README / Phase 3)
        ("A", "B", 0.70, 0.70),   # A×B: ~77-80% baseline, ~76% zscore
        ("A", "C", 0.95, 0.95),   # A×C: ~98% both
        ("B", "C", 0.05, 0.50),   # B×C: ~14% baseline, ~64% zscore
    ])
    def test_paper_pairs_minimum(self, ql, dl, expected_baseline, expected_zscore):
        """Verify key pairs meet minimum thresholds (not exact, since FPS is stochastic)."""
        baseline, zscore = self._run_pair(ql, dl)
        assert baseline["recall_at_1"] >= expected_baseline, (
            f"{ql}→{dl} baseline R@1={baseline['recall_at_1']:.3f}, expected >= {expected_baseline}"
        )
        assert zscore["recall_at_1"] >= expected_zscore, (
            f"{ql}→{dl} zscore R@1={zscore['recall_at_1']:.3f}, expected >= {expected_zscore}"
        )

    def test_exact_match_ab(self, reference_results):
        """A→B: exact match with d2a_matrix.json reference."""
        ref = reference_results.get(("A", "B"))
        if ref is None:
            pytest.skip("A→B not in reference results")

        baseline, zscore = self._run_pair("A", "B")

        assert baseline["recall_at_1"] == pytest.approx(ref["baseline_r1"], abs=0.001), (
            f"A→B baseline: got {baseline['recall_at_1']:.4f}, "
            f"ref {ref['baseline_r1']:.4f}"
        )
        assert zscore["recall_at_1"] == pytest.approx(ref["zscore_db_r1"], abs=0.001), (
            f"A→B zscore: got {zscore['recall_at_1']:.4f}, "
            f"ref {ref['zscore_db_r1']:.4f}"
        )

    def test_exact_match_ac(self, reference_results):
        """A→C: exact match with d2a_matrix.json reference."""
        ref = reference_results.get(("A", "C"))
        if ref is None:
            pytest.skip("A→C not in reference results")

        baseline, zscore = self._run_pair("A", "C")

        assert baseline["recall_at_1"] == pytest.approx(ref["baseline_r1"], abs=0.001)
        assert zscore["recall_at_1"] == pytest.approx(ref["zscore_db_r1"], abs=0.001)

    def test_exact_match_bc(self, reference_results):
        """B→C: exact match (the hardest pair, z-score critical)."""
        ref = reference_results.get(("B", "C"))
        if ref is None:
            pytest.skip("B→C not in reference results")

        baseline, zscore = self._run_pair("B", "C")

        assert baseline["recall_at_1"] == pytest.approx(ref["baseline_r1"], abs=0.001)
        assert zscore["recall_at_1"] == pytest.approx(ref["zscore_db_r1"], abs=0.001)


# ── Test: RATHub gives same results as RATranslator ──

class TestHubMatchesTranslator:
    """RATHub and RATranslator should produce identical results."""

    def test_hub_vs_translator(self):
        _skip_if_no_data()
        query_a, cand_a = _load_emb("A")
        query_b, cand_b = _load_emb("B")
        fps_idx = _fps_src_compat(cand_a, K)

        anchor_a = cand_a[fps_idx]
        anchor_b = cand_b[fps_idx]

        # RATranslator
        t = RATranslator(kernel="poly", normalize="always")
        t.fit_embeddings(anchor_a, anchor_b)
        t_rel_a = t.transform(query_a, "a")
        t_rel_b = t.transform(query_b, "b")

        # RATHub
        hub = RATHub(kernel="poly", normalize="always")
        hub.set_anchors("model_a", anchor_a)
        hub.set_anchors("model_b", anchor_b)
        h_rel_a = hub.transform("model_a", query_a, role="query")
        h_rel_b = hub.transform("model_b", query_b, role="db")

        np.testing.assert_allclose(t_rel_a, h_rel_a)
        np.testing.assert_allclose(t_rel_b, h_rel_b)
