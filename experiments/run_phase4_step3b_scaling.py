"""
RAT Phase 4 Step 3b: アンカースケーリング + 片側z-score診断

2つの実験を同時に実行:
  1. A×E (MiniLM × CLIP-image) K=500,1000,2000,3000 スケーリング
     - 片側z-score (query側のみ / DB側のみ / 両側) を網羅
  2. D×E (CLIP-text × CLIP-image) 片側z-score再計測
     - Step 3で両側z-scoreのみだったのを修正

アンカー: 最大3000, クエリ: 500 (重複なし)
FPS + poly(degree=2, coef0=1.0)
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

from run_phase4_step2 import load_coco_pairs, compute_anchor_sim_stats

CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"
ANCHOR_COUNTS = [500, 1000, 2000, 3000]
NUM_QUERIES = 500
MAX_ANCHORS = 3000


def run_retrieval_full(rel_q, rel_db, label):
    """baseline + 片側z-score + 両側z-score、双方向で検索。"""
    results = []

    rel_q_z = normalize_zscore(rel_q)
    rel_db_z = normalize_zscore(rel_db)

    configs = [
        ("baseline",    rel_q,   rel_db),
        ("zscore_q",    rel_q_z, rel_db),     # query側のみ
        ("zscore_db",   rel_q,   rel_db_z),   # DB側のみ
        ("zscore_both", rel_q_z, rel_db_z),   # 両側
    ]

    for method, q, db in configs:
        m = evaluate_retrieval(q, db)
        results.append({"method": method, "direction": label, **m})

        m_rev = evaluate_retrieval(db, q)
        results.append({"method": method, "direction": f"{label}_rev", **m_rev})

    return results


def print_results_table(results, title):
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(f"  {'Direction':<28} {'Method':<14} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-' * 82}")
    for r in results:
        print(
            f"  {r['direction']:<28} {r['method']:<14} "
            f"{r['recall_at_1']*100:>7.1f}% {r['recall_at_5']*100:>7.1f}% "
            f"{r['recall_at_10']*100:>7.1f}% {r['mrr']:>8.3f}"
        )
    print(f"{'=' * 95}")


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_needed = MAX_ANCHORS + NUM_QUERIES

    print("=" * 60)
    print("Phase 4 Step 3b: スケーリング + 片側z-score診断")
    print(f"  アンカー: 最大{MAX_ANCHORS}, クエリ: {NUM_QUERIES}")
    print("=" * 60)

    # ========================================
    # データ準備
    # ========================================
    print(f"\n--- データ準備: COCO {total_needed}組 ---")
    all_pairs = load_coco_pairs(total_needed, offset=0, seed=config.RANDOM_SEED)
    print(f"  取得: {len(all_pairs)}組")

    available = len(all_pairs) - NUM_QUERIES
    adjusted_counts = [k for k in ANCHOR_COUNTS if k <= available]
    if len(adjusted_counts) < len(ANCHOR_COUNTS):
        print(f"  アンカー数を調整: {adjusted_counts} (利用可能: {available})")

    anchor_pairs = all_pairs[:MAX_ANCHORS]
    query_pairs = all_pairs[MAX_ANCHORS:MAX_ANCHORS + NUM_QUERIES]

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    # ========================================
    # Embedding（全モデル一括）
    # ========================================
    print("\n--- Embedding ---")

    print("  CLIP-text アンカー...")
    anchor_emb_clip_text = embed_texts(config.MODEL_D, anchor_captions)
    print(f"    shape: {anchor_emb_clip_text.shape}")

    print("  CLIP-text クエリ...")
    query_emb_clip_text = embed_texts(config.MODEL_D, query_captions)
    print(f"    shape: {query_emb_clip_text.shape}")

    print("  CLIP-image アンカー...")
    anchor_emb_clip_img = embed_images_clip(anchor_images, CLIP_VISION_MODEL)
    print(f"    shape: {anchor_emb_clip_img.shape}")

    print("  CLIP-image クエリ...")
    query_emb_clip_img = embed_images_clip(query_images, CLIP_VISION_MODEL)
    print(f"    shape: {query_emb_clip_img.shape}")

    print("  MiniLM アンカー...")
    anchor_emb_minilm = embed_texts(config.MODEL_A, anchor_captions)
    print(f"    shape: {anchor_emb_minilm.shape}")

    print("  MiniLM クエリ...")
    query_emb_minilm = embed_texts(config.MODEL_A, query_captions)
    print(f"    shape: {query_emb_minilm.shape}")

    # ========================================
    # メイン実験
    # ========================================
    all_experiment_results = {}

    for K in adjusted_counts:
        print(f"\n{'=' * 60}")
        print(f"  K = {K}")
        print(f"{'=' * 60}")

        # FPS（CLIP-text空間で選定、MiniLMにも同じインデックスを適用）
        print(f"\n  FPSアンカー選定 (CLIP-text空間, K={K})...")
        fps_indices, _ = select_anchors_fps(anchor_emb_clip_text[:K], anchor_captions[:K], K)

        a_clip_text = anchor_emb_clip_text[:K][fps_indices]
        a_clip_img = anchor_emb_clip_img[:K][fps_indices]
        a_minilm = anchor_emb_minilm[:K][fps_indices]

        # 統計
        stats = {
            "clip_text": compute_anchor_sim_stats(a_clip_text, f"CLIP-text (K={K})"),
            "clip_img": compute_anchor_sim_stats(a_clip_img, f"CLIP-image (K={K})"),
            "minilm": compute_anchor_sim_stats(a_minilm, f"MiniLM (K={K})"),
        }
        for s in stats.values():
            print(f"    {s['label']}: mean={s['mean_sim']:.4f}, entropy={s['entropy']:.4f}")

        # --- D×E: CLIP-text × CLIP-image（片側z-score） ---
        print(f"\n  --- D×E: CLIP-text × CLIP-image (K={K}) ---")
        rel_ct = to_relative(query_emb_clip_text, a_clip_text, kernel="poly", degree=2, coef0=1.0)
        rel_ci = to_relative(query_emb_clip_img, a_clip_img, kernel="poly", degree=2, coef0=1.0)
        dxe_results = run_retrieval_full(rel_ct, rel_ci, "CLIPtext→CLIPimg")
        print_results_table(dxe_results, f"D×E (K={K})")

        # --- A×E: MiniLM × CLIP-image（片側z-score） ---
        print(f"\n  --- A×E: MiniLM × CLIP-image (K={K}) ---")
        rel_ml = to_relative(query_emb_minilm, a_minilm, kernel="poly", degree=2, coef0=1.0)
        rel_ci2 = to_relative(query_emb_clip_img, a_clip_img, kernel="poly", degree=2, coef0=1.0)
        axe_results = run_retrieval_full(rel_ml, rel_ci2, "MiniLM→CLIPimg")
        print_results_table(axe_results, f"A×E (K={K})")

        all_experiment_results[f"K={K}"] = {
            "anchor_stats": stats,
            "DxE": dxe_results,
            "AxE": axe_results,
        }

    # ========================================
    # 参考: CLIP直接
    # ========================================
    print("\n--- 参考: CLIP直接 Text→Image ---")
    sim_direct = cosine_similarity(query_emb_clip_text, query_emb_clip_img)
    ranks = []
    for i in range(len(query_emb_clip_text)):
        rank = int(np.where(np.argsort(-sim_direct[i]) == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    clip_direct = {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    print(f"  CLIP直接: R@1={clip_direct['recall_at_1']*100:.1f}%, R@10={clip_direct['recall_at_10']*100:.1f}%")

    # ========================================
    # サマリー
    # ========================================
    print("\n" + "=" * 95)
    print("  サマリー")
    print("=" * 95)
    print(f"\n  CLIP直接 (上限): R@1={clip_direct['recall_at_1']*100:.1f}%\n")

    for k_label, data in all_experiment_results.items():
        print(f"  {k_label}:")
        for pair_name, pair_key in [("D×E", "DxE"), ("A×E", "AxE")]:
            results = data[pair_key]
            best = max(results, key=lambda r: r["recall_at_1"])
            print(f"    {pair_name} best: R@1={best['recall_at_1']*100:.1f}% ({best['direction']}, {best['method']})")

            # z-score方式別ベスト
            for method in ["baseline", "zscore_q", "zscore_db", "zscore_both"]:
                method_results = [r for r in results if r["method"] == method]
                if method_results:
                    best_m = max(method_results, key=lambda r: r["recall_at_1"])
                    print(f"      {method:<14} R@1={best_m['recall_at_1']*100:.1f}% ({best_m['direction']})")
        print()

    # ========================================
    # 保存
    # ========================================
    elapsed = time.time() - start_time

    output = {
        "experiment_results": all_experiment_results,
        "clip_direct_baseline": clip_direct,
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "phase4_step3b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
