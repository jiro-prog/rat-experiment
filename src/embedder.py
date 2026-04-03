import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from PIL import Image

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
    kwargs = {}
    if model_name in config.MODEL_CONFIGS:
        if config.MODEL_CONFIGS[model_name].get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_name, **kwargs)
    prepared = _prepare_texts(model_name, texts)
    embeddings = model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return embeddings


def embed_images_clip(
    images: list[Image.Image],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
) -> np.ndarray:
    """CLIPの画像エンコーダで画像をembeddingに変換する。L2正規化済み。"""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            # visual_projection で CLIP の共有空間に射影
            pooled = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled)
        # L2正規化
        embs = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
        print(f"  画像embed: {min(i + batch_size, len(images))}/{len(images)}", end="\r")

    print()
    return np.vstack(all_embs)


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
