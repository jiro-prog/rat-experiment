"""RATHub: multi-model relative anchor translation."""

from __future__ import annotations

import json
import warnings

import numpy as np

from rat.kernels import get_kernel
from rat.normalize import compute_sim_mean, normalize_zscore, recommend_zscore


class RATHub:
    """Manage N embedding models and translate between any pair via RAT.

    Each model's anchor embeddings are registered once. After that,
    any model's embeddings can be transformed to the shared relative space.
    """

    def __init__(
        self,
        kernel: str = "poly",
        kernel_params: dict | None = None,
        normalize: str = "auto",
        normalize_threshold: float = 0.15,
        normalize_harmful_threshold: float = 0.65,
        verbose: bool = False,
    ):
        self._kernel_name = kernel
        self._kernel_params = kernel_params or {}
        self._kernel_fn = get_kernel(kernel, kernel_params)
        self._normalize = normalize
        self._normalize_threshold = normalize_threshold
        self._normalize_harmful_threshold = normalize_harmful_threshold
        self._verbose = verbose

        # Per-model state: {model_name: {...}}
        self._models: dict[str, dict] = {}
        self._anchor_k: int | None = None  # enforced anchor count

    # ── Registration ──

    def set_anchors(
        self,
        model_name: str,
        anchor_embeddings: np.ndarray,
    ) -> None:
        """Register anchor embeddings for a model.

        All models must use anchors from the **same texts** in the
        **same order**. For example, select K anchor indices via FPS
        on one model's candidates, then use those same indices to
        extract anchors from every model. Mismatched anchor texts
        will cause cross-model retrieval to fail silently.

        Parameters
        ----------
        model_name : identifier for this model
        anchor_embeddings : (K, D) L2-normalized
        """
        _check_l2_normalized(anchor_embeddings)

        K = anchor_embeddings.shape[0]
        if self._anchor_k is None:
            self._anchor_k = K
        elif K != self._anchor_k:
            raise ValueError(
                f"Anchor count mismatch: expected {self._anchor_k}, got {K} for '{model_name}'"
            )

        sim_mean = compute_sim_mean(anchor_embeddings)

        self._models[model_name] = {
            "anchors": anchor_embeddings,
            "sim_mean": sim_mean,
        }

        if self._verbose:
            rec = recommend_zscore(sim_mean, self._normalize_threshold, self._normalize_harmful_threshold)
            print(f"[RAT] {model_name} sim_mean={sim_mean:.3f} → z-score: {rec}")
            print(
                f"[RAT] Anchor shape: {model_name}="
                f"({anchor_embeddings.shape[0]}, {anchor_embeddings.shape[1]})"
            )

    # ── Transform ──

    def transform(
        self,
        model_name: str,
        embeddings: np.ndarray,
        role: str = "auto",
    ) -> np.ndarray:
        """Transform embeddings to relative representation.

        Parameters
        ----------
        model_name : which model produced these embeddings
        embeddings : (N, D) L2-normalized
        role : "query" (skip z-score), "db" (apply z-score per normalize setting),
               "auto" (treat as db)

        Returns
        -------
        (N, K) relative representation
        """
        if model_name not in self._models:
            raise RuntimeError(
                f"Model '{model_name}' not registered. Call set_anchors() first."
            )
        if role not in ("query", "db", "auto"):
            raise ValueError(f"role must be 'query', 'db', or 'auto', got '{role}'")

        _check_l2_normalized(embeddings)

        state = self._models[model_name]
        anchors = state["anchors"]
        rel = self._kernel_fn(embeddings, anchors)  # (N, K)

        if self._should_apply_zscore(model_name, role):
            rel = normalize_zscore(rel)

        return rel

    def _should_apply_zscore(self, model_name: str, role: str) -> bool:
        """Decide whether to apply z-score for this model/role combination."""
        if role == "query":
            return False

        # role is "db" or "auto" → check normalize setting
        state = self._models[model_name]
        sim_mean = state["sim_mean"]

        if self._normalize == "never":
            return False
        if self._normalize == "always":
            return True

        # "auto": apply z-score unless harmful
        rec = recommend_zscore(sim_mean, self._normalize_threshold, self._normalize_harmful_threshold)
        should = rec != "harmful"

        if self._verbose:
            if rec == "harmful":
                print(
                    f"[RAT] {model_name} sim_mean={sim_mean:.3f} >= "
                    f"harmful_threshold {self._normalize_harmful_threshold} "
                    f"→ z-score: harmful (skipping)"
                )
            else:
                print(
                    f"[RAT] {model_name} sim_mean={sim_mean:.3f} "
                    f"→ applying z-score"
                )

        return should

    # ── Retrieval ──

    def retrieve(
        self,
        query_emb: np.ndarray,
        db_emb: np.ndarray,
        from_model: str,
        to_model: str,
        top_k: int = 10,
    ) -> dict:
        """Transform + cosine similarity retrieval.

        Parameters
        ----------
        query_emb : (N, D_from) query embeddings from from_model
        db_emb : (M, D_to) database embeddings from to_model
        from_model : model name for queries
        to_model : model name for database
        top_k : number of results per query

        Returns
        -------
        {"indices": (N, top_k), "scores": (N, top_k)}
        """
        q_rel = self.transform(from_model, query_emb, role="query")
        d_rel = self.transform(to_model, db_emb, role="db")

        # L2-normalize relative representations for cosine similarity
        q_norm = q_rel / (np.linalg.norm(q_rel, axis=1, keepdims=True) + 1e-10)
        d_norm = d_rel / (np.linalg.norm(d_rel, axis=1, keepdims=True) + 1e-10)

        scores = q_norm @ d_norm.T  # (N, M)
        top_k = min(top_k, scores.shape[1])
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)

        return {"indices": indices, "scores": sorted_scores}

    def retrieve_multi(
        self,
        query_emb: np.ndarray,
        databases: list[tuple[np.ndarray, str]],
        query_model: str,
        top_k: int = 10,
    ) -> dict:
        """Search across multiple databases built with different models.

        Uses per-database score normalization to make scores comparable
        across databases. Do NOT vstack relative representations from
        different models — their score scales differ.

        Parameters
        ----------
        query_emb : (N, D) query embeddings from query_model
        databases : list of (db_embeddings, db_model_name) tuples.
            Each db_embeddings is (M_i, D_i) from the corresponding model.
            Each database should have at least ~50 documents for stable
            score normalization.
        query_model : model name for queries
        top_k : number of results per query

        Returns
        -------
        {"indices": (N, top_k), "scores": (N, top_k), "db_labels": (N, top_k)}
            indices are global (offset by cumulative DB sizes).
            db_labels[i][j] is the database index (0, 1, ...) for each result.
        """
        q_rel = self.transform(query_model, query_emb, role="query")
        q_norm = q_rel / (np.linalg.norm(q_rel, axis=1, keepdims=True) + 1e-10)

        n_queries = len(query_emb)
        total_docs = sum(db_emb.shape[0] for db_emb, _ in databases)
        merged_scores = np.empty((n_queries, total_docs), dtype=np.float32)

        offset = 0
        for db_emb, db_model in databases:
            if db_emb.shape[0] < 50:
                warnings.warn(
                    f"Database '{db_model}' has only {db_emb.shape[0]} documents. "
                    "Per-database score normalization may be unstable with fewer than "
                    "~50 documents.",
                    stacklevel=2,
                )
            d_rel = self.transform(db_model, db_emb, role="db")
            d_norm = d_rel / (np.linalg.norm(d_rel, axis=1, keepdims=True) + 1e-10)
            sim = q_norm @ d_norm.T  # (N, M_i)

            # Per-query z-score normalization for cross-DB comparability
            mu = sim.mean(axis=1, keepdims=True)
            sigma = sim.std(axis=1, keepdims=True)
            sigma[sigma == 0] = 1.0
            merged_scores[:, offset:offset + db_emb.shape[0]] = (sim - mu) / sigma

            offset += db_emb.shape[0]

        top_k = min(top_k, total_docs)
        indices = np.argsort(-merged_scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(merged_scores, indices, axis=1)

        # Map global indices to DB labels
        db_labels = np.empty_like(indices)
        offset = 0
        for db_idx, (db_emb, _) in enumerate(databases):
            m = db_emb.shape[0]
            mask = (indices >= offset) & (indices < offset + m)
            db_labels[mask] = db_idx
            offset += m

        return {"indices": indices, "scores": sorted_scores, "db_labels": db_labels}

    # ── Compatibility ──

    # Regression coefficients (linear, N=90, CLIP excluded, R²=0.17)
    _COMPAT_SLOPE = -45.37
    _COMPAT_INTERCEPT = 109.87
    _COMPAT_RESID_P16 = -11.6  # 16th percentile of residuals
    _COMPAT_RESID_P84 = 13.9   # 84th percentile of residuals
    _COMPAT_SAME_FAMILY_BONUS = 10.0

    # Tier boundaries (max_sim_mean thresholds)
    _TIER_HIGH_UPPER = 0.45
    _TIER_LOW_LOWER = 0.72

    _TIER_DESCRIPTIONS: dict[tuple[str, bool], str] = {
        ("high", False): "Pairs in this compression range typically achieve Recall@1 91-98%.",
        ("high", True): "Same-family pairs in this range typically achieve >95%.",
        ("moderate", False): (
            "Pairs in this range typically achieve Recall@1 73-91%, with wide variation."
        ),
        ("moderate", True): (
            "Same-family pairs in this range are expected to perform at the high end."
        ),
        ("low", False): (
            "High-compression pairs typically achieve Recall@1 61-82%. "
            "Consider z-score normalization."
        ),
        ("low", True): (
            "Same-family pairs partially offset compression effects, "
            "typically achieving 80-95%."
        ),
    }

    def estimate_compatibility(
        self,
        model_a: str,
        model_b: str,
        same_family: bool | None = None,
    ) -> dict:
        """Rough compatibility estimate based on anchor similarity compression.

        Based on linear regression over N=90 text-model pairs (CLIP excluded),
        R²=0.17. The compatibility tier is more reliable than the point
        estimate. Cross-modal and CLIP-text models may deviate significantly.

        Parameters
        ----------
        model_a, model_b : registered model names
        same_family : whether the two models belong to the same family
            (e.g. both BGE, both E5). If True, adds +10pt bonus to the
            point estimate and raises the compatibility tier by one level.
            If None (default), no adjustment is applied.

        Returns
        -------
        dict with compatibility tier, point estimate, confidence band, and
        z-score recommendation.
        """
        if model_a not in self._models:
            raise RuntimeError(f"Model '{model_a}' not registered.")
        if model_b not in self._models:
            raise RuntimeError(f"Model '{model_b}' not registered.")

        sm_a = self._models[model_a]["sim_mean"]
        sm_b = self._models[model_b]["sim_mean"]
        max_sm = max(sm_a, sm_b)
        rec = recommend_zscore(max_sm, self._normalize_threshold, self._normalize_harmful_threshold)

        # Point estimate (clipped to [0, 100])
        est = self._COMPAT_SLOPE * max_sm + self._COMPAT_INTERCEPT
        est = max(0.0, min(est, 100.0))
        bonus: float | None = None
        if same_family is True:
            bonus = self._COMPAT_SAME_FAMILY_BONUS
            est = min(est + bonus, 99.0)

        # Confidence band (16th-84th percentile, clipped to [0, 100])
        band_lo = max(est + self._COMPAT_RESID_P16, 0.0)
        band_hi = min(est + self._COMPAT_RESID_P84, 100.0)

        # Tier
        tier = self._compute_tier(max_sm)
        if same_family is True:
            tier = self._upgrade_tier(tier)

        sf_key = same_family is True
        description = self._TIER_DESCRIPTIONS.get((tier, sf_key), "")

        warn_list: list[str] = []
        if max_sm > self._normalize_harmful_threshold:
            warn_list.append(
                f"High sim_mean ({max_sm:.3f}) detected. "
                "Z-score normalization may be harmful for this pair."
            )

        return {
            "sim_mean_a": sm_a,
            "sim_mean_b": sm_b,
            "max_sim_mean": max_sm,
            "compatibility": tier,
            "compatibility_description": description,
            "estimated_recall_at_1": round(est, 1),
            "confidence_band": (round(band_lo, 1), round(band_hi, 1)),
            "estimate_reliability": "low",
            "same_family_bonus": bonus,
            "z_score_recommendation": rec,
            "warnings": warn_list,
        }

    @staticmethod
    def _compute_tier(max_sim_mean: float) -> str:
        if max_sim_mean < RATHub._TIER_HIGH_UPPER:
            return "high"
        if max_sim_mean < RATHub._TIER_LOW_LOWER:
            return "moderate"
        return "low"

    @staticmethod
    def _upgrade_tier(tier: str) -> str:
        if tier == "low":
            return "moderate"
        if tier == "moderate":
            return "high"
        return "high"  # high stays high

    # ── Persistence ──

    def save(self, path: str) -> None:
        """Save hub state to .npz file."""
        save_dict: dict[str, object] = {}

        model_names = list(self._models.keys())
        for name in model_names:
            state = self._models[name]
            save_dict[f"anchor_{name}"] = state["anchors"]

        config = {
            "kernel": self._kernel_name,
            "kernel_params": self._kernel_params,
            "normalize": self._normalize,
            "normalize_threshold": self._normalize_threshold,
            "normalize_harmful_threshold": self._normalize_harmful_threshold,
            "model_names": model_names,
            "sim_means": {n: self._models[n]["sim_mean"] for n in model_names},
            "version": "0.1.0",
        }
        save_dict["config"] = np.array(json.dumps(config))

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "RATHub":
        """Load hub state from .npz file."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)
        config = json.loads(str(data["config"]))

        hub = cls(
            kernel=config["kernel"],
            kernel_params=config.get("kernel_params"),
            normalize=config["normalize"],
            normalize_threshold=config["normalize_threshold"],
            normalize_harmful_threshold=config.get("normalize_harmful_threshold", 0.65),
        )

        for name in config["model_names"]:
            anchors = data[f"anchor_{name}"]
            hub.set_anchors(name, anchors)
            # Override sim_mean from saved config (avoids recomputation drift)
            hub._models[name]["sim_mean"] = config["sim_means"][name]

        return hub


def _check_l2_normalized(embeddings: np.ndarray) -> None:
    """Warn if embeddings don't appear L2-normalized."""
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        warnings.warn(
            "Input embeddings may not be L2-normalized. RAT expects normalized embeddings.",
            stacklevel=3,
        )
