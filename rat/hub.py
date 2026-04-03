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

    # ── Compatibility ──

    def estimate_compatibility(
        self,
        model_a: str,
        model_b: str,
    ) -> dict:
        """Estimate RAT compatibility between two registered models.

        Returns
        -------
        dict with sim_mean_a, sim_mean_b, max_sim_mean, estimated_recall_at_1,
        z_score_recommendation, warnings
        """
        if model_a not in self._models:
            raise RuntimeError(f"Model '{model_a}' not registered.")
        if model_b not in self._models:
            raise RuntimeError(f"Model '{model_b}' not registered.")

        sm_a = self._models[model_a]["sim_mean"]
        sm_b = self._models[model_b]["sim_mean"]
        max_sm = max(sm_a, sm_b)
        rec = recommend_zscore(max_sm, self._normalize_threshold, self._normalize_harmful_threshold)

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
            "estimated_recall_at_1": None,  # Phase 2
            "z_score_recommendation": rec,
            "warnings": warn_list,
        }

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
