"""RATranslator: simple 2-model API wrapping RATHub."""

from __future__ import annotations

import json

import numpy as np

from rat.hub import RATHub


_MODEL_A = "__model_a__"
_MODEL_B = "__model_b__"


class RATranslator:
    """Zero-shot embedding space translation between two models.

    RAT uses relative distances to shared anchors to make embeddings
    from different models comparable, without any additional training.

    Typical usage: model_a = query model, model_b = database model.
    This convention determines default z-score behavior, but can be
    overridden via the ``role`` parameter in :meth:`transform`.
    """

    def __init__(
        self,
        kernel: str = "poly",
        kernel_params: dict | None = None,
        normalize: str = "auto",
        normalize_harmful_threshold: float = 0.65,
        verbose: bool = False,
    ):
        self._hub = RATHub(
            kernel=kernel,
            kernel_params=kernel_params,
            normalize=normalize,
            normalize_harmful_threshold=normalize_harmful_threshold,
            verbose=verbose,
        )
        self._fitted = False

    # ── Setup ──

    def fit(
        self,
        anchor_texts: list[str],
        model_a: str,
        model_b: str,
        *,
        model_a_prefix: str = "",
        model_b_prefix: str = "",
    ) -> "RATranslator":
        """Encode anchors with both models and set up the translator.

        Requires sentence-transformers. For pre-computed embeddings,
        use :meth:`fit_embeddings` instead.

        Note: anchor embeddings encoded here use the given prefixes.
        Embeddings passed to :meth:`transform` must use the same prefix
        setting to ensure consistency.

        Returns self for method chaining.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for fit(). "
                "Install with: pip install rat-embed[models]\n"
                "Or use fit_embeddings() with pre-computed embeddings."
            )

        st_a = SentenceTransformer(model_a)
        st_b = SentenceTransformer(model_b)

        texts_a = [model_a_prefix + t for t in anchor_texts] if model_a_prefix else anchor_texts
        texts_b = [model_b_prefix + t for t in anchor_texts] if model_b_prefix else anchor_texts

        emb_a = st_a.encode(texts_a, normalize_embeddings=True)
        emb_b = st_b.encode(texts_b, normalize_embeddings=True)

        return self.fit_embeddings(np.asarray(emb_a), np.asarray(emb_b))

    def fit_embeddings(
        self,
        anchor_emb_a: np.ndarray,
        anchor_emb_b: np.ndarray,
    ) -> "RATranslator":
        """Set up from pre-computed anchor embeddings (numpy only).

        Parameters
        ----------
        anchor_emb_a : (K, D_a) L2-normalized anchors from model A
        anchor_emb_b : (K, D_b) L2-normalized anchors from model B

        Returns self for method chaining.
        """
        if anchor_emb_a.shape[0] != anchor_emb_b.shape[0]:
            raise ValueError(
                f"Anchor count mismatch: a={anchor_emb_a.shape[0]}, b={anchor_emb_b.shape[0]}"
            )
        self._hub.set_anchors(_MODEL_A, anchor_emb_a)
        self._hub.set_anchors(_MODEL_B, anchor_emb_b)
        self._fitted = True
        return self

    # ── Transform ──

    def transform(
        self,
        embeddings: np.ndarray,
        source: str,
        *,
        role: str | None = None,
    ) -> np.ndarray:
        """Transform embeddings to relative representation.

        Parameters
        ----------
        embeddings : (N, D) L2-normalized
        source : "a" or "b" — which model's anchors to use
        role : z-score control.
            None (default): source="a" → query (no z-score),
                            source="b" → db (z-score per normalize setting)
            "query": skip z-score
            "db": apply z-score per normalize setting

            ``source`` and ``role`` are independent concerns:
            source selects anchors, role controls normalization.

        Returns
        -------
        (N, K) relative representation
        """
        self._check_fitted()

        if source not in ("a", "b"):
            raise ValueError(f"source must be 'a' or 'b', got '{source}'")
        if role is not None and role not in ("query", "db"):
            raise ValueError(f"role must be 'query', 'db', or None, got '{role}'")

        model_name = _MODEL_A if source == "a" else _MODEL_B

        # Resolve role
        if role is None:
            hub_role = "query" if source == "a" else "db"
        else:
            hub_role = role

        if self._hub._verbose and role is not None:
            default_role = "query" if source == "a" else "db"
            if role != default_role:
                print(f'[RAT] transform source="{source}" role={role} → override')

        return self._hub.transform(model_name, embeddings, role=hub_role)

    # ── Retrieval ──

    def retrieve(
        self,
        query_emb: np.ndarray,
        db_emb: np.ndarray,
        top_k: int = 10,
    ) -> dict:
        """Transform + cosine similarity retrieval.

        Parameters
        ----------
        query_emb : (N, D_a) from model A (query side)
        db_emb : (M, D_b) from model B (database side)
        top_k : results per query

        Returns
        -------
        {"indices": (N, top_k), "scores": (N, top_k)}
        """
        self._check_fitted()
        return self._hub.retrieve(query_emb, db_emb, _MODEL_A, _MODEL_B, top_k)

    # ── Compatibility ──

    def estimate_compatibility(self, same_family: bool | None = None) -> dict:
        """Estimate RAT compatibility between the two fitted models.

        Parameters
        ----------
        same_family : whether the two models belong to the same family.
            See :meth:`RATHub.estimate_compatibility` for details.
        """
        self._check_fitted()
        return self._hub.estimate_compatibility(_MODEL_A, _MODEL_B, same_family=same_family)

    # ── Persistence ──

    def save(self, path: str) -> None:
        """Save translator state to .npz file."""
        self._check_fitted()
        self._hub.save(path)

    @classmethod
    def load(cls, path: str) -> "RATranslator":
        """Load translator from .npz file."""
        hub = RATHub.load(path)
        translator = cls.__new__(cls)
        translator._hub = hub
        translator._fitted = True
        return translator

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_embeddings() first.")
