"""
実験 C-3: 非対称z-score

Phase 4で発見した「z-scoreはクエリ側のentropyが低いときに効く」を検証。

4パターン:
  none:     z-score なし（ベースライン）
  both:     クエリ・DB両方にz-score
  query:    クエリ側のみz-score
  db:       DB側のみz-score

テキスト3ペア（A×B, A×C, B×C）+ クロスモーダル2方向（Text→Image, Image→Text）

加えて、エントロピー閾値ルールの検証:
  「entropy < 2.0 のモデル側にのみz-scoreを適用」
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from datasets import load_dataset

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K = 500
CANDIDATE_POOL = 2000
ENTROPY_THRESHOLD = 2.0


def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def load_coco_pairs(num_pairs: int, offset: int = 0, seed: int = 42) -> list[dict]:
    """COCO Karpathy split から(画像, キャプション)ペアをサンプル。"""
    ds = load_dataset("yerevann/coco-karpathy", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))
    candidate_indices = indices[offset : offset + num_pairs + 200]

    candidates = []
    for idx in candidate_indices:
        item = ds[int(idx)]
        sentences = item["sentences"]
        caption = sentences[rng.randint(len(sentences))]
        if isinstance(caption, dict):
            caption = caption.get("raw", str(caption))
        candidates.append({"url": item["url"], "caption": caption.strip()})

    def _download(i):
        return i, download_image(candidates[i]["url"])

    pairs = []
    fail_count = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_download, i): i for i in range(len(candidates))}
        for future in as_completed(futures):
            i, img = future.result()
            if img is None:
                fail_count += 1
            else:
                candidates[i]["image"] = img

    for c in candidates:
        if "image" in c and len(pairs) < num_pairs:
            pairs.append({"image": c["image"], "caption": c["caption"]})

    print(f"  COCO: {len(pairs)}組取得 (失敗: {fail_count})")
    return pairs


def compute_entropy(embeddings: np.ndarray) -> float:
    """アンカー間類似度のエントロピーを計算。"""
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)
    triu_idx = np.triu_indices_from(sim, k=1)
    sims = sim[triu_idx]
    hist, _ = np.histogram(sims, bins=50, range=(-1, 1), density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(entropy(hist))


def eval_asymmetric(rel_query_raw, rel_db_raw, zscore_query, zscore_db):
    """非対称z-score適用後に検索評価。"""
    rq = normalize_zscore(rel_query_raw) if zscore_query else rel_query_raw
    rd = normalize_zscore(rel_db_raw) if zscore_db else rel_db_raw
    return evaluate_retrieval(rq, rd)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 C-3: 非対称z-score")
    print("=" * 60)

    models = {
        "A": config.MODEL_A,
        "B": config.MODEL_B,
        "C": config.MODEL_C,
    }

    # ========================================
    # Part 1: テキスト3ペア
    # ========================================
    print("\n--- Part 1: テキスト3ペア ---")
    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )

    cand_embs = {}
    query_embs = {}
    for label, model_name in models.items():
        short = model_name.split("/")[-1]
        print(f"  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)

    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in models}

    # エントロピー計算
    entropies = {}
    for label in models:
        entropies[label] = compute_entropy(anchor_embs[label])
        print(f"  Model {label} entropy: {entropies[label]:.4f}")

    # テキストペア評価
    text_pairs = [("A×B", "A", "B"), ("A×C", "A", "C"), ("B×C", "B", "C")]
    patterns = [
        ("none", False, False),
        ("both", True, True),
        ("query_only", True, False),
        ("db_only", False, True),
    ]

    text_results = []
    print(f"\n{'ペア':<8} {'pattern':<14} {'R@1':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 55)

    for pair_label, x, y in text_pairs:
        rel_x_raw = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y_raw = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)

        for pat_name, zq, zd in patterns:
            m = eval_asymmetric(rel_x_raw, rel_y_raw, zq, zd)
            text_results.append({
                "pair": pair_label,
                "pattern": pat_name,
                "query_model": x,
                "db_model": y,
                **m,
            })
            print(
                f"{pair_label:<8} {pat_name:<14} "
                f"{m['recall_at_1']*100:>7.1f}% {m['recall_at_10']*100:>7.1f}% "
                f"{m['mrr']:>8.3f}"
            )

    # ========================================
    # Part 2: クロスモーダル
    # ========================================
    print("\n--- Part 2: クロスモーダル (Text ↔ Image) ---")

    anchor_pairs = load_coco_pairs(500, offset=0, seed=config.RANDOM_SEED)
    query_pairs = load_coco_pairs(500, offset=500, seed=config.RANDOM_SEED)

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    print("  テキスト embedding...")
    anchor_emb_text = embed_texts(config.MODEL_A, anchor_captions)
    query_emb_text = embed_texts(config.MODEL_A, query_captions)

    print("  画像 embedding...")
    anchor_emb_img = embed_images_clip(anchor_images)
    query_emb_img = embed_images_clip(query_images)

    fps_cm_indices, _ = select_anchors_fps(anchor_emb_text, anchor_captions, K)
    anchor_text_fps = anchor_emb_text[fps_cm_indices]
    anchor_img_fps = anchor_emb_img[fps_cm_indices]

    ent_text = compute_entropy(anchor_text_fps)
    ent_img = compute_entropy(anchor_img_fps)
    print(f"  Text entropy: {ent_text:.4f}")
    print(f"  Image entropy: {ent_img:.4f}")

    # 相対表現（生）
    rel_text_q_raw = to_relative(query_emb_text, anchor_text_fps, kernel="poly", degree=2, coef0=1.0)
    rel_img_q_raw = to_relative(query_emb_img, anchor_img_fps, kernel="poly", degree=2, coef0=1.0)

    cross_results = []
    print(f"\n{'方向':<20} {'pattern':<14} {'R@1':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 60)

    for dir_label, rel_q_raw, rel_d_raw, q_model, d_model in [
        ("Text→Image", rel_text_q_raw, rel_img_q_raw, "text", "image"),
        ("Image→Text", rel_img_q_raw, rel_text_q_raw, "image", "text"),
    ]:
        for pat_name, zq, zd in patterns:
            m = eval_asymmetric(rel_q_raw, rel_d_raw, zq, zd)
            cross_results.append({
                "direction": dir_label,
                "pattern": pat_name,
                "query_model": q_model,
                "db_model": d_model,
                **m,
            })
            print(
                f"{dir_label:<20} {pat_name:<14} "
                f"{m['recall_at_1']*100:>7.1f}% {m['recall_at_10']*100:>7.1f}% "
                f"{m['mrr']:>8.3f}"
            )

    # ========================================
    # エントロピー閾値ルール検証
    # ========================================
    print("\n" + "=" * 60)
    print(f"エントロピー閾値ルール (threshold={ENTROPY_THRESHOLD})")
    print("=" * 60)
    print(f"  ルール: entropy < {ENTROPY_THRESHOLD} のモデル側にのみz-scoreを適用")

    # 全モデルのエントロピー
    all_entropies = {**entropies, "text_cm": ent_text, "image_cm": ent_img}
    print(f"\n  エントロピー一覧:")
    for label, ent in sorted(all_entropies.items()):
        marker = "← z-score対象" if ent < ENTROPY_THRESHOLD else ""
        print(f"    {label}: {ent:.4f} {marker}")

    # 閾値ルール適用
    print(f"\n{'ペア/方向':<20} {'adaptive R@1':>12} {'both R@1':>12} {'差分':>8}")
    print("-" * 60)

    adaptive_results = []

    # テキストペア
    for pair_label, x, y in text_pairs:
        rel_x_raw = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y_raw = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)

        zq = entropies[x] < ENTROPY_THRESHOLD
        zd = entropies[y] < ENTROPY_THRESHOLD
        m_adaptive = eval_asymmetric(rel_x_raw, rel_y_raw, zq, zd)

        # both for comparison
        m_both = eval_asymmetric(rel_x_raw, rel_y_raw, True, True)

        diff = (m_adaptive["recall_at_1"] - m_both["recall_at_1"]) * 100
        adaptive_results.append({
            "pair": pair_label,
            "query_zscore": zq,
            "db_zscore": zd,
            **m_adaptive,
        })
        print(
            f"{pair_label:<20} {m_adaptive['recall_at_1']*100:>11.1f}% "
            f"{m_both['recall_at_1']*100:>11.1f}% {diff:>+7.1f}%"
        )

    # クロスモーダル
    for dir_label, rel_q_raw, rel_d_raw, q_ent, d_ent in [
        ("Text→Image", rel_text_q_raw, rel_img_q_raw, ent_text, ent_img),
        ("Image→Text", rel_img_q_raw, rel_text_q_raw, ent_img, ent_text),
    ]:
        zq = q_ent < ENTROPY_THRESHOLD
        zd = d_ent < ENTROPY_THRESHOLD
        m_adaptive = eval_asymmetric(rel_q_raw, rel_d_raw, zq, zd)
        m_both = eval_asymmetric(rel_q_raw, rel_d_raw, True, True)

        diff = (m_adaptive["recall_at_1"] - m_both["recall_at_1"]) * 100
        adaptive_results.append({
            "pair": dir_label,
            "query_zscore": zq,
            "db_zscore": zd,
            **m_adaptive,
        })
        print(
            f"{dir_label:<20} {m_adaptive['recall_at_1']*100:>11.1f}% "
            f"{m_both['recall_at_1']*100:>11.1f}% {diff:>+7.1f}%"
        )

    # 保存
    elapsed = time.time() - start_time
    output = {
        "entropies": all_entropies,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "text_results": text_results,
        "cross_modal_results": cross_results,
        "adaptive_results": adaptive_results,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "c3_asymmetric_zscore.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
