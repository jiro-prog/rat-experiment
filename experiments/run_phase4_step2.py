"""
RAT Phase 4 Step 2: クロスモーダル検索（テキスト × 画像）

ロゼッタストーン方式:
  同じ概念の(画像, テキスト)ペアをアンカーとし、
  各モデルは自分のモダリティでアンカーをencodeする。

テキスト側: MiniLM (384d) — キャプションをencode
画像側: CLIP画像エンコーダ (512d) — 画像をencode

FPS + poly + z-score, K=500
成功基準: Recall@1 > 10%
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
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K = 500
NUM_ANCHORS = 500
NUM_QUERIES = 500
CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"


def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    """URLから画像をダウンロードしてPIL Imageで返す。"""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def load_coco_pairs(num_pairs: int, offset: int = 0, seed: int = 42) -> list[dict]:
    """
    COCO Captions (Karpathy split) から(画像, キャプション)ペアをサンプルする。
    各画像に対しキャプション1つをランダム選択。画像はURLからダウンロード。
    """
    print("  COCO Karpathy split をロード中...")
    ds = load_dataset("yerevann/coco-karpathy", split="test")

    rng = np.random.RandomState(seed)

    # 全件のインデックスをシャッフル
    indices = rng.permutation(len(ds))
    print(f"  データセット: {len(ds)}件")

    # offset から十分な候補を取得（ダウンロード失敗に備えてバッファ）
    candidate_indices = indices[offset : offset + num_pairs + 200]

    # キャプション選択を先に決定
    candidates = []
    for idx in candidate_indices:
        item = ds[int(idx)]
        sentences = item["sentences"]
        caption = sentences[rng.randint(len(sentences))]
        if isinstance(caption, dict):
            caption = caption.get("raw", str(caption))
        candidates.append({
            "url": item["url"],
            "caption": caption.strip(),
        })

    # 並列画像ダウンロード
    print(f"  画像を並列ダウンロード中 ({len(candidates)}件)...")

    def _download(i):
        img = download_image(candidates[i]["url"])
        return i, img

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

    # 元の順序を維持してペアを構築
    for c in candidates:
        if "image" in c and len(pairs) < num_pairs:
            pairs.append({"image": c["image"], "caption": c["caption"]})

    print(f"  取得: {len(pairs)}組 (失敗: {fail_count})")
    return pairs


def compute_anchor_sim_stats(embeddings: np.ndarray, label: str) -> dict:
    """アンカー間類似度行列の統計量を計算する。"""
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)
    triu_idx = np.triu_indices_from(sim, k=1)
    sims = sim[triu_idx]

    hist, _ = np.histogram(sims, bins=50, range=(-1, 1), density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    ent = float(entropy(hist))

    return {
        "label": label,
        "mean_sim": float(np.mean(sims)),
        "std_sim": float(np.std(sims)),
        "min_sim": float(np.min(sims)),
        "max_sim": float(np.max(sims)),
        "entropy": ent,
    }


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4 Step 2: クロスモーダル検索 (Text × Image)")
    print("  ロゼッタストーン方式")
    print("=" * 60)

    # ========================================
    # データ準備: COCO Captions
    # ========================================
    print("\n--- データ準備 ---")

    # アンカー用: 最初の500組
    print(f"\nアンカー用 {NUM_ANCHORS}組:")
    anchor_pairs = load_coco_pairs(NUM_ANCHORS, offset=0, seed=config.RANDOM_SEED)
    print(f"  取得: {len(anchor_pairs)}組")

    # クエリ用: 次の500組（重複なし）
    print(f"\nクエリ用 {NUM_QUERIES}組:")
    query_pairs = load_coco_pairs(NUM_QUERIES, offset=NUM_ANCHORS, seed=config.RANDOM_SEED)
    print(f"  取得: {len(query_pairs)}組")

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    # 画像がPIL Imageであることを確認
    print(f"\n  アンカーキャプション例: {anchor_captions[0][:80]}...")
    print(f"  クエリキャプション例: {query_captions[0][:80]}...")

    # ========================================
    # Embedding
    # ========================================
    print("\n--- Embedding ---")

    # MiniLM: テキストキャプションをencode
    print("\n  MiniLM (テキスト):")
    print("    アンカーキャプション...")
    anchor_emb_text = embed_texts(config.MODEL_A, anchor_captions)
    print(f"    shape: {anchor_emb_text.shape}")
    print("    クエリキャプション...")
    query_emb_text = embed_texts(config.MODEL_A, query_captions)
    print(f"    shape: {query_emb_text.shape}")

    # CLIP画像エンコーダ: 画像をencode
    print("\n  CLIP画像エンコーダ:")
    print("    アンカー画像...")
    anchor_emb_img = embed_images_clip(anchor_images, CLIP_VISION_MODEL)
    print(f"    shape: {anchor_emb_img.shape}")
    print("    クエリ画像...")
    query_emb_img = embed_images_clip(query_images, CLIP_VISION_MODEL)
    print(f"    shape: {query_emb_img.shape}")

    # ========================================
    # FPSアンカー選定（MiniLMテキスト空間で）
    # ========================================
    print("\n--- FPSアンカー選定 (MiniLMテキスト空間, K=500) ---")

    # アンカー候補 = アンカーの全500組（候補プール=アンカー数の場合、全数選択）
    # ここではFPSで順序を決めて、500個すべて使う
    fps_indices, _ = select_anchors_fps(anchor_emb_text, anchor_captions, K)

    # FPSの順序でアンカーを並び替え
    anchor_emb_text_fps = anchor_emb_text[fps_indices]
    anchor_emb_img_fps = anchor_emb_img[fps_indices]

    # ========================================
    # アンカー間類似度分析
    # ========================================
    print("\n--- アンカー間類似度分析 ---")

    stats_text = compute_anchor_sim_stats(anchor_emb_text_fps, "MiniLM (text anchors)")
    stats_img = compute_anchor_sim_stats(anchor_emb_img_fps, "CLIP-image (image anchors)")

    for stats, label in [(stats_text, "MiniLM テキスト"), (stats_img, "CLIP 画像")]:
        print(f"\n  {label}:")
        print(f"    Mean sim:  {stats['mean_sim']:.4f}")
        print(f"    Std sim:   {stats['std_sim']:.4f}")
        print(f"    Range:     [{stats['min_sim']:.4f}, {stats['max_sim']:.4f}]")
        print(f"    Entropy:   {stats['entropy']:.4f}")

    # ========================================
    # クロスモーダル検索
    # ========================================
    print("\n" + "=" * 60)
    print("クロスモーダル検索")
    print("=" * 60)

    all_results = []

    # --- ベースライン (z-scoreなし) ---
    print("\n--- ベースライン (FPS + poly, z-scoreなし) ---")

    # Text→Image: MiniLMクエリ → CLIP画像DB
    rel_text_q = to_relative(query_emb_text, anchor_emb_text_fps, kernel="poly", degree=2, coef0=1.0)
    rel_img_db = to_relative(query_emb_img, anchor_emb_img_fps, kernel="poly", degree=2, coef0=1.0)
    metrics_t2i = evaluate_retrieval(rel_text_q, rel_img_db)
    print(f"  Text→Image: Recall@1={metrics_t2i['recall_at_1']*100:.1f}%, R@10={metrics_t2i['recall_at_10']*100:.1f}%, MRR={metrics_t2i['mrr']:.3f}")
    all_results.append({"method": "baseline", "direction": "Text→Image", **metrics_t2i})

    # Image→Text: CLIP画像クエリ → MiniLM テキストDB
    metrics_i2t = evaluate_retrieval(rel_img_db, rel_text_q)
    print(f"  Image→Text: Recall@1={metrics_i2t['recall_at_1']*100:.1f}%, R@10={metrics_i2t['recall_at_10']*100:.1f}%, MRR={metrics_i2t['mrr']:.3f}")
    all_results.append({"method": "baseline", "direction": "Image→Text", **metrics_i2t})

    # --- z-score正規化あり ---
    print("\n--- FPS + poly + z-score ---")

    rel_text_q_z = normalize_zscore(rel_text_q)
    rel_img_db_z = normalize_zscore(rel_img_db)

    metrics_t2i_z = evaluate_retrieval(rel_text_q_z, rel_img_db_z)
    print(f"  Text→Image: Recall@1={metrics_t2i_z['recall_at_1']*100:.1f}%, R@10={metrics_t2i_z['recall_at_10']*100:.1f}%, MRR={metrics_t2i_z['mrr']:.3f}")
    all_results.append({"method": "zscore", "direction": "Text→Image", **metrics_t2i_z})

    metrics_i2t_z = evaluate_retrieval(rel_img_db_z, rel_text_q_z)
    print(f"  Image→Text: Recall@1={metrics_i2t_z['recall_at_1']*100:.1f}%, R@10={metrics_i2t_z['recall_at_10']*100:.1f}%, MRR={metrics_i2t_z['mrr']:.3f}")
    all_results.append({"method": "zscore", "direction": "Image→Text", **metrics_i2t_z})

    # --- 参考: CLIP内部のText→Image（上限推定） ---
    print("\n--- 参考: CLIP内部 Text→Image（直接コサイン、RATなし） ---")
    # CLIPテキストエンコーダでキャプションをencode
    print("  CLIPテキストエンコーダでキャプションをencode...")
    query_emb_clip_text = embed_texts(config.MODEL_D, query_captions)
    sim_clip_direct = cosine_similarity(query_emb_clip_text, query_emb_img)
    ranks = []
    for i in range(len(query_emb_clip_text)):
        sorted_idx = np.argsort(-sim_clip_direct[i])
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    clip_direct = {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    print(f"  CLIP直接: Recall@1={clip_direct['recall_at_1']*100:.1f}%, R@10={clip_direct['recall_at_10']*100:.1f}%, MRR={clip_direct['mrr']:.3f}")

    # ========================================
    # 結果テーブル
    # ========================================
    print("\n" + "=" * 100)
    print("  Phase 4 Step 2 結果: クロスモーダル検索 (MiniLM Text ↔ CLIP Image)")
    print("=" * 100)

    print(f"\n{'方向':<25} {'Method':<12} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['direction']:<25} {r['method']:<12} "
            f"{r['recall_at_1']*100:>9.1f}% {r['recall_at_5']*100:>9.1f}% "
            f"{r['recall_at_10']*100:>9.1f}% {r['mrr']:>8.3f}"
        )

    print("-" * 80)
    print(
        f"{'CLIP直接 (上限参考)':<25} {'direct':<12} "
        f"{clip_direct['recall_at_1']*100:>9.1f}% {clip_direct['recall_at_5']*100:>9.1f}% "
        f"{clip_direct['recall_at_10']*100:>9.1f}% {clip_direct['mrr']:>8.3f}"
    )
    print("=" * 100)

    # ========================================
    # 判定
    # ========================================
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    zscore_results = [r for r in all_results if r["method"] == "zscore"]
    best = max(zscore_results, key=lambda r: r["recall_at_1"])
    best_r1 = best["recall_at_1"]

    print(f"\n  最良: {best['direction']} (Recall@1={best_r1*100:.1f}%)")

    for r in zscore_results:
        status = "✓ PASS" if r["recall_at_1"] > 0.1 else "✗ FAIL"
        print(f"  {r['direction']}: Recall@1={r['recall_at_1']*100:.1f}% → {status} (基準: >10%)")

    print(f"\n  CLIP直接検索（上限参考）: Recall@1={clip_direct['recall_at_1']*100:.1f}%")
    if best_r1 > 0:
        ratio = best_r1 / max(clip_direct["recall_at_1"], 1e-9)
        print(f"  RAT達成率（対CLIP直接）: {ratio*100:.1f}%")

    if best_r1 > 0.1:
        print(f"\n  → 成功! RATがモダリティの壁を越えてテキスト-画像検索を実現")
        print(f"  → MiniLMのテキスト入力だけでCLIP画像空間を検索できる")
    elif best_r1 > 0.02:
        print(f"\n  → ランダム(0.2%)の10倍以上。シグナルはあるが実用レベルではない")
    else:
        print(f"\n  → クロスモーダルRATは現構成では機能しない")

    # 保存
    elapsed = time.time() - start_time

    output = {
        "anchor_sim_stats": {
            "text": stats_text,
            "image": stats_img,
        },
        "results": all_results,
        "clip_direct_baseline": clip_direct,
        "best_direction": best["direction"],
        "best_recall_at_1": best_r1,
        "pass_10pct": best_r1 > 0.1,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "phase4_step2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
