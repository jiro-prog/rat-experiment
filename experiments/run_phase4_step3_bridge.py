"""
RAT Phase 4 Step 3: CLIP内クロスモーダルRAT (CLIP-text × CLIP-image)

目的:
  ブリッジ戦略の前提検証として、CLIP-text-RATとCLIP-image-RATの
  直接一致度を計測する。

  同じ概念の(画像, テキスト)ペアをアンカーとし:
    テキスト側: CLIP-text (512d) — キャプションをencode
    画像側: CLIP-image (512d) — 画像をencode

  CLIPは同じ訓練でtext-imageが整列されているため、
  RATアンカーとの距離パターンも一致しやすい期待がある。

  同時に A×E (MiniLM × CLIP-image) をK=1000,2000でスケーリング。

FPS + poly + z-score
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

import config
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

# Step 2と同じCOCOデータローダを流用
from run_phase4_step2 import load_coco_pairs, compute_anchor_sim_stats

CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"
ANCHOR_COUNTS = [500, 1000, 2000]
NUM_QUERIES = 500


def run_retrieval(rel_q, rel_db, label):
    """baseline + z-score で検索し結果を返す。"""
    results = []

    # baseline
    m = evaluate_retrieval(rel_q, rel_db)
    results.append({"method": "baseline", "direction": f"{label}", **m})

    m_rev = evaluate_retrieval(rel_db, rel_q)
    results.append({"method": "baseline", "direction": f"{label}_rev", **m_rev})

    # z-score
    rel_q_z = normalize_zscore(rel_q)
    rel_db_z = normalize_zscore(rel_db)

    m_z = evaluate_retrieval(rel_q_z, rel_db_z)
    results.append({"method": "zscore", "direction": f"{label}", **m_z})

    m_z_rev = evaluate_retrieval(rel_db_z, rel_q_z)
    results.append({"method": "zscore", "direction": f"{label}_rev", **m_z_rev})

    return results


def print_results_table(results, title):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Direction':<30} {'Method':<10} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-' * 78}")
    for r in results:
        print(
            f"  {r['direction']:<30} {r['method']:<10} "
            f"{r['recall_at_1']*100:>7.1f}% {r['recall_at_5']*100:>7.1f}% "
            f"{r['recall_at_10']*100:>7.1f}% {r['mrr']:>8.3f}"
        )
    print(f"{'=' * 90}")


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    max_anchors = max(ANCHOR_COUNTS)
    total_needed = max_anchors + NUM_QUERIES

    print("=" * 60)
    print("Phase 4 Step 3: CLIP内クロスモーダルRAT + アンカースケーリング")
    print("=" * 60)

    # ========================================
    # データ準備
    # ========================================
    print(f"\n--- データ準備: COCO {total_needed}組 ---")
    all_pairs = load_coco_pairs(total_needed, offset=0, seed=config.RANDOM_SEED)
    print(f"  取得: {len(all_pairs)}組")

    if len(all_pairs) < total_needed:
        print(f"  警告: {total_needed}組必要だが{len(all_pairs)}組しか取得できず")
        # アンカー数を調整
        available_anchors = len(all_pairs) - NUM_QUERIES
        adjusted_counts = [k for k in ANCHOR_COUNTS if k <= available_anchors]
        if not adjusted_counts:
            print("  エラー: アンカーが足りません")
            return
        print(f"  アンカー数を調整: {adjusted_counts}")
    else:
        adjusted_counts = ANCHOR_COUNTS

    anchor_pairs = all_pairs[:max_anchors]
    query_pairs = all_pairs[max_anchors:max_anchors + NUM_QUERIES]

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    # ========================================
    # Embedding
    # ========================================
    print("\n--- Embedding ---")

    # CLIP-text: キャプション
    print("  CLIP-text アンカー...")
    anchor_emb_clip_text = embed_texts(config.MODEL_D, anchor_captions)
    print(f"    shape: {anchor_emb_clip_text.shape}")

    print("  CLIP-text クエリ...")
    query_emb_clip_text = embed_texts(config.MODEL_D, query_captions)
    print(f"    shape: {query_emb_clip_text.shape}")

    # CLIP-image: 画像
    print("  CLIP-image アンカー...")
    anchor_emb_clip_img = embed_images_clip(anchor_images, CLIP_VISION_MODEL)
    print(f"    shape: {anchor_emb_clip_img.shape}")

    print("  CLIP-image クエリ...")
    query_emb_clip_img = embed_images_clip(query_images, CLIP_VISION_MODEL)
    print(f"    shape: {query_emb_clip_img.shape}")

    # MiniLM: キャプション（スケーリング比較用）
    print("  MiniLM アンカー...")
    anchor_emb_minilm = embed_texts(config.MODEL_A, anchor_captions)
    print(f"    shape: {anchor_emb_minilm.shape}")

    print("  MiniLM クエリ...")
    query_emb_minilm = embed_texts(config.MODEL_A, query_captions)
    print(f"    shape: {query_emb_minilm.shape}")

    # ========================================
    # メイン実験: K別
    # ========================================
    all_experiment_results = {}

    for K in adjusted_counts:
        print(f"\n{'=' * 60}")
        print(f"  K = {K}")
        print(f"{'=' * 60}")

        # FPSアンカー選定（CLIP-text空間で）
        print(f"\n  FPSアンカー選定 (CLIP-text空間, K={K})...")
        fps_indices, _ = select_anchors_fps(anchor_emb_clip_text[:K], anchor_captions[:K], K)

        # アンカー取得
        a_clip_text = anchor_emb_clip_text[:K][fps_indices]
        a_clip_img = anchor_emb_clip_img[:K][fps_indices]
        a_minilm = anchor_emb_minilm[:K][fps_indices]

        # アンカー間統計
        stats_clip_text = compute_anchor_sim_stats(a_clip_text, f"CLIP-text (K={K})")
        stats_clip_img = compute_anchor_sim_stats(a_clip_img, f"CLIP-image (K={K})")
        stats_minilm = compute_anchor_sim_stats(a_minilm, f"MiniLM (K={K})")

        print(f"\n  アンカー間類似度:")
        for s in [stats_clip_text, stats_clip_img, stats_minilm]:
            print(f"    {s['label']}: mean={s['mean_sim']:.4f}, entropy={s['entropy']:.4f}")

        # --- D×E: CLIP-text × CLIP-image ---
        print(f"\n  --- D×E: CLIP-text × CLIP-image (K={K}) ---")
        rel_clip_text = to_relative(query_emb_clip_text, a_clip_text, kernel="poly", degree=2, coef0=1.0)
        rel_clip_img = to_relative(query_emb_clip_img, a_clip_img, kernel="poly", degree=2, coef0=1.0)

        dxe_results = run_retrieval(rel_clip_text, rel_clip_img, "CLIPtext→CLIPimg")
        print_results_table(dxe_results, f"D×E: CLIP-text × CLIP-image (K={K})")

        # --- A×E: MiniLM × CLIP-image ---
        print(f"\n  --- A×E: MiniLM × CLIP-image (K={K}) ---")
        rel_minilm = to_relative(query_emb_minilm, a_minilm, kernel="poly", degree=2, coef0=1.0)
        # CLIP-image側は同じアンカー画像に対するRAT
        rel_clip_img_for_minilm = to_relative(query_emb_clip_img, a_clip_img, kernel="poly", degree=2, coef0=1.0)

        axe_results = run_retrieval(rel_minilm, rel_clip_img_for_minilm, "MiniLM→CLIPimg")
        print_results_table(axe_results, f"A×E: MiniLM × CLIP-image (K={K})")

        all_experiment_results[f"K={K}"] = {
            "anchor_stats": {
                "clip_text": stats_clip_text,
                "clip_img": stats_clip_img,
                "minilm": stats_minilm,
            },
            "DxE_clip_text_x_clip_image": dxe_results,
            "AxE_minilm_x_clip_image": axe_results,
        }

    # ========================================
    # 参考: CLIP直接検索（RATなし）
    # ========================================
    print("\n--- 参考: CLIP直接 Text→Image ---")
    sim_clip_direct = cosine_similarity(query_emb_clip_text, query_emb_clip_img)
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
    print(f"  CLIP直接: R@1={clip_direct['recall_at_1']*100:.1f}%, R@10={clip_direct['recall_at_10']*100:.1f}%")

    # ========================================
    # サマリー
    # ========================================
    print("\n" + "=" * 90)
    print("  サマリー: ブリッジ前提検証")
    print("=" * 90)

    print(f"\n  CLIP直接 (上限): R@1={clip_direct['recall_at_1']*100:.1f}%")
    print()

    for k_label, data in all_experiment_results.items():
        dxe = data["DxE_clip_text_x_clip_image"]
        axe = data["AxE_minilm_x_clip_image"]

        best_dxe = max(dxe, key=lambda r: r["recall_at_1"])
        best_axe = max(axe, key=lambda r: r["recall_at_1"])

        print(f"  {k_label}:")
        print(f"    D×E best: R@1={best_dxe['recall_at_1']*100:.1f}% ({best_dxe['direction']}, {best_dxe['method']})")
        print(f"    A×E best: R@1={best_axe['recall_at_1']*100:.1f}% ({best_axe['direction']}, {best_axe['method']})")
        print()

    # 判定
    best_dxe_overall = max(
        [r for data in all_experiment_results.values() for r in data["DxE_clip_text_x_clip_image"]],
        key=lambda r: r["recall_at_1"],
    )
    dxe_r1 = best_dxe_overall["recall_at_1"]

    if dxe_r1 > 0.3:
        print("  → D×E > 30%: ブリッジ戦略は有望。MiniLM→CLIP-textの接続を検討する価値あり。")
    elif dxe_r1 > 0.18:
        print("  → D×E > 18% (A×Eの現状): ブリッジの天井はA×E直接より高い。")
        print("    ただし MiniLM→CLIP-text の変換損失を考えると実質的なゲインは限定的。")
    else:
        print("  → D×E ≤ 18%: ブリッジの天井がA×E直接と同等以下。ブリッジ戦略は不採用。")
        print("    アンカースケーリングとカーネルチューニングに集中すべき。")

    # ========================================
    # 保存
    # ========================================
    elapsed = time.time() - start_time

    output = {
        "experiment_results": {},
        "clip_direct_baseline": clip_direct,
        "elapsed_seconds": elapsed,
    }

    # numpy/dict を JSON serializable に
    for k_label, data in all_experiment_results.items():
        output["experiment_results"][k_label] = {
            "anchor_stats": data["anchor_stats"],
            "DxE": data["DxE_clip_text_x_clip_image"],
            "AxE": data["AxE_minilm_x_clip_image"],
        }

    out_path = config.RESULTS_DIR / "phase4_step3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
