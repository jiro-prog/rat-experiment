import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

import config


def _get_prefix(model_name: str) -> str:
    """モデルに対応するプレフィクスを取得する。"""
    if model_name in config.MODEL_CONFIGS:
        return config.MODEL_CONFIGS[model_name]["prefix"]
    return ""


def _prepare_texts(model_name: str, texts: list[str]) -> list[str]:
    """モデルに応じてプレフィクスを付与する。"""
    prefix = _get_prefix(model_name)
    if prefix:
        return [prefix + t for t in texts]
    return texts


def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    """テキストをembeddingに変換する。L2正規化済み。"""
    model = SentenceTransformer(model_name)
    prepared = _prepare_texts(model_name, texts)
    embeddings = model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return embeddings


def embed_and_save(
    model_name: str,
    anchors: list[str],
    queries: list[str],
    model_label: str,
    data_dir: Path = config.DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """アンカーとクエリをembedしてnpyで保存する。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Model: {model_name} ({model_label})")
    print(f"{'='*50}")

    print(f"アンカー {len(anchors)}件をembed中...")
    anchor_emb = embed_texts(model_name, anchors)
    np.save(data_dir / f"anchor_emb_{model_label}.npy", anchor_emb)

    print(f"クエリ {len(queries)}件をembed中...")
    query_emb = embed_texts(model_name, queries)
    np.save(data_dir / f"query_emb_{model_label}.npy", query_emb)

    print(f"Embedding shape: anchor={anchor_emb.shape}, query={query_emb.shape}")
    return anchor_emb, query_emb
