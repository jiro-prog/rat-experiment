"""
RAT Phase 0: 相対アンカー変換の仮説検証実験

異なるembeddingモデル間で、共通アンカーとの相対距離表現を用いた
zero-shotクロスモデル検索が成立するかを検証する。
"""
import sys
import json
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from src.anchor_sampler import sample_anchors_and_queries, save_data
from src.embedder import embed_and_save
from src.relative_repr import to_relative, to_relative_subset
from src.evaluator import evaluate_retrieval, evaluate_neighbor_preservation, print_metrics
from src.visualizer import plot_similarity_heatmap, plot_tsne, plot_anchor_scaling


def print_summary_table(results: dict):
    """最終結果テーブルを出力する。"""
    print("\n" + "=" * 80)
    print("  最終結果サマリ")
    print("=" * 80)

    header = f"{'実験':<45} {'Recall@1':>9} {'Recall@10':>10} {'MRR':>7} {'備考'}"
    print(header)
    print("-" * 80)

    for row in results["rows"]:
        r1 = f"{row['recall_at_1']*100:.1f}%" if row["recall_at_1"] is not None else "-"
        r10 = f"{row['recall_at_10']*100:.1f}%" if row["recall_at_10"] is not None else "-"
        mrr = f"{row['mrr']:.3f}" if row["mrr"] is not None else "-"
        print(f"{row['label']:<45} {r1:>9} {r10:>10} {mrr:>7} {row.get('note', '')}")

    if "overlap_rows" in results:
        print()
        header2 = f"{'近傍構造保存率':<45} {'Overlap@10':>10} {'Std':>7} {'Median':>8}"
        print(header2)
        print("-" * 80)
        for row in results["overlap_rows"]:
            o10 = f"{row['overlap_at_10']*100:.1f}%"
            std = f"{row['std']*100:.1f}%"
            med = f"{row['median']*100:.1f}%"
            print(f"{row['label']:<45} {o10:>10} {std:>7} {med:>8}")

    print("=" * 80)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Step 1: データ準備
    # ========================================
    print("\n" + "=" * 60)
    print("Step 1: データ準備")
    print("=" * 60)

    anchors, queries = sample_anchors_and_queries()
    save_data(anchors, queries)

    # ========================================
    # Step 2: Embedding取得
    # ========================================
    print("\n" + "=" * 60)
    print("Step 2: Embedding取得")
    print("=" * 60)

    anchor_emb_A, query_emb_A = embed_and_save(
        config.MODEL_A, anchors, queries, model_label="A"
    )
    anchor_emb_B, query_emb_B = embed_and_save(
        config.MODEL_B, anchors, queries, model_label="B"
    )

    # ========================================
    # Step 3: 相対表現への変換
    # ========================================
    print("\n" + "=" * 60)
    print("Step 3: 相対表現への変換")
    print("=" * 60)

    query_rel_A = to_relative(query_emb_A, anchor_emb_A)
    query_rel_B = to_relative(query_emb_B, anchor_emb_B)

    print(f"相対表現 shape: A={query_rel_A.shape}, B={query_rel_B.shape}")
    np.save(config.DATA_DIR / "query_rel_A.npy", query_rel_A)
    np.save(config.DATA_DIR / "query_rel_B.npy", query_rel_B)

    # ========================================
    # Step 4: 評価
    # ========================================
    print("\n" + "=" * 60)
    print("Step 4: 評価")
    print("=" * 60)

    # メイン: Cross-Model Retrieval (A → B) with full anchors
    cross_metrics = evaluate_retrieval(query_rel_A, query_rel_B)
    print_metrics(cross_metrics, f"Cross-Model A→B (K={config.NUM_ANCHORS})")

    # ランダムベースライン
    random_recall1 = 1.0 / len(queries)
    print(f"\nRandom Baseline Recall@1: {random_recall1:.4f} ({random_recall1*100:.2f}%)")

    # ========================================
    # Step 4b: 近傍構造保存率 (Overlap@10)
    # ========================================
    print("\n" + "=" * 60)
    print("Step 4b: 近傍構造保存率 (Same-Model Baseline)")
    print("=" * 60)

    # Model A: 元空間 vs 相対表現の近傍一致率（全アンカー）
    overlap_A_full = evaluate_neighbor_preservation(query_emb_A, query_rel_A, k=10)
    print(f"\nModel A 近傍構造保存率 (K={config.NUM_ANCHORS}):")
    print(f"  Overlap@10:  {overlap_A_full['overlap_at_10']:.4f} ({overlap_A_full['overlap_at_10']*100:.1f}%)")
    print(f"  Std:         {overlap_A_full['overlap_at_10_std']:.4f}")
    print(f"  Median:      {overlap_A_full['overlap_at_10_median']:.4f}")

    # Model B: 元空間 vs 相対表現の近傍一致率（全アンカー）
    overlap_B_full = evaluate_neighbor_preservation(query_emb_B, query_rel_B, k=10)
    print(f"\nModel B 近傍構造保存率 (K={config.NUM_ANCHORS}):")
    print(f"  Overlap@10:  {overlap_B_full['overlap_at_10']:.4f} ({overlap_B_full['overlap_at_10']*100:.1f}%)")
    print(f"  Std:         {overlap_B_full['overlap_at_10_std']:.4f}")
    print(f"  Median:      {overlap_B_full['overlap_at_10_median']:.4f}")

    # K=500でのOverlap@10も計測（前回結果との比較用）
    overlap_A_500 = None
    overlap_B_500 = None
    if config.NUM_ANCHORS > 500:
        rel_A_500 = to_relative_subset(query_emb_A, anchor_emb_A, 500)
        rel_B_500 = to_relative_subset(query_emb_B, anchor_emb_B, 500)
        overlap_A_500 = evaluate_neighbor_preservation(query_emb_A, rel_A_500, k=10)
        overlap_B_500 = evaluate_neighbor_preservation(query_emb_B, rel_B_500, k=10)
        print(f"\nModel A 近傍構造保存率 (K=500):")
        print(f"  Overlap@10:  {overlap_A_500['overlap_at_10']:.4f} ({overlap_A_500['overlap_at_10']*100:.1f}%)")
        print(f"\nModel B 近傍構造保存率 (K=500):")
        print(f"  Overlap@10:  {overlap_B_500['overlap_at_10']:.4f} ({overlap_B_500['overlap_at_10']*100:.1f}%)")

    all_metrics = {
        "cross_model_A_to_B": cross_metrics,
        "neighbor_preservation_A": overlap_A_full,
        "neighbor_preservation_B": overlap_B_full,
        "random_baseline_recall_at_1": random_recall1,
    }
    if overlap_A_500:
        all_metrics["neighbor_preservation_A_K500"] = overlap_A_500
        all_metrics["neighbor_preservation_B_K500"] = overlap_B_500

    # ========================================
    # Step 5: 可視化 + アンカースケーリング
    # ========================================
    print("\n" + "=" * 60)
    print("Step 5: 可視化 + アンカースケーリング")
    print("=" * 60)

    plot_similarity_heatmap(
        query_rel_A, query_rel_B,
        config.RESULTS_DIR / "sim_matrix.png",
    )
    plot_tsne(
        query_rel_A, query_rel_B,
        config.RESULTS_DIR / "tsne_plot.png",
    )

    # アンカー数スケーリング実験（Cross-Model + Overlap@10）
    print("\nアンカー数スケーリング実験...")
    scaling_results = {}
    scaling_overlap_A = {}
    scaling_overlap_B = {}
    for n_anchors in config.ANCHOR_COUNTS:
        if n_anchors > len(anchors):
            print(f"  アンカー {n_anchors} > 利用可能数 {len(anchors)}, スキップ")
            continue
        rel_A_sub = to_relative_subset(query_emb_A, anchor_emb_A, n_anchors)
        rel_B_sub = to_relative_subset(query_emb_B, anchor_emb_B, n_anchors)

        metrics = evaluate_retrieval(rel_A_sub, rel_B_sub)
        scaling_results[n_anchors] = metrics

        ov_A = evaluate_neighbor_preservation(query_emb_A, rel_A_sub, k=10)
        ov_B = evaluate_neighbor_preservation(query_emb_B, rel_B_sub, k=10)
        scaling_overlap_A[n_anchors] = ov_A
        scaling_overlap_B[n_anchors] = ov_B

        print(
            f"  K={n_anchors:5d}: Recall@1={metrics['recall_at_1']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, "
            f"Overlap@10(A)={ov_A['overlap_at_10']:.4f}, "
            f"Overlap@10(B)={ov_B['overlap_at_10']:.4f}"
        )

    all_metrics["anchor_scaling"] = {str(k): v for k, v in scaling_results.items()}
    all_metrics["anchor_scaling_overlap_A"] = {str(k): v for k, v in scaling_overlap_A.items()}
    all_metrics["anchor_scaling_overlap_B"] = {str(k): v for k, v in scaling_overlap_B.items()}

    plot_anchor_scaling(
        scaling_results,
        config.RESULTS_DIR / "anchor_scaling.png",
    )

    # ========================================
    # 結果テーブル出力
    # ========================================
    summary_rows = []

    # K=500のCross-Model（スケーリング結果から取得）
    if 500 in scaling_results:
        m500 = scaling_results[500]
        summary_rows.append({
            "label": "Cross-Model A→B (K=500)",
            "recall_at_1": m500["recall_at_1"],
            "recall_at_10": m500["recall_at_10"],
            "mrr": m500["mrr"],
            "note": "",
        })

    # K=1000のCross-Model
    if config.NUM_ANCHORS >= 1000 and 1000 in scaling_results:
        m1000 = scaling_results[1000]
        summary_rows.append({
            "label": f"Cross-Model A→B (K={config.NUM_ANCHORS})",
            "recall_at_1": m1000["recall_at_1"],
            "recall_at_10": m1000["recall_at_10"],
            "mrr": m1000["mrr"],
            "note": "新規",
        })
    elif config.NUM_ANCHORS < 1000:
        summary_rows.append({
            "label": f"Cross-Model A→B (K={config.NUM_ANCHORS})",
            "recall_at_1": cross_metrics["recall_at_1"],
            "recall_at_10": cross_metrics["recall_at_10"],
            "mrr": cross_metrics["mrr"],
            "note": "",
        })

    # ランダムベースライン
    summary_rows.append({
        "label": "Random Baseline",
        "recall_at_1": random_recall1,
        "recall_at_10": None,
        "mrr": None,
        "note": f"1/{len(queries)}",
    })

    # Overlap行
    overlap_rows = []
    if overlap_A_500:
        overlap_rows.append({
            "label": "Model A 近傍構造保存率 (K=500)",
            "overlap_at_10": overlap_A_500["overlap_at_10"],
            "std": overlap_A_500["overlap_at_10_std"],
            "median": overlap_A_500["overlap_at_10_median"],
        })
    overlap_rows.append({
        "label": f"Model A 近傍構造保存率 (K={config.NUM_ANCHORS})",
        "overlap_at_10": overlap_A_full["overlap_at_10"],
        "std": overlap_A_full["overlap_at_10_std"],
        "median": overlap_A_full["overlap_at_10_median"],
    })
    if overlap_B_500:
        overlap_rows.append({
            "label": "Model B 近傍構造保存率 (K=500)",
            "overlap_at_10": overlap_B_500["overlap_at_10"],
            "std": overlap_B_500["overlap_at_10_std"],
            "median": overlap_B_500["overlap_at_10_median"],
        })
    overlap_rows.append({
        "label": f"Model B 近傍構造保存率 (K={config.NUM_ANCHORS})",
        "overlap_at_10": overlap_B_full["overlap_at_10"],
        "std": overlap_B_full["overlap_at_10_std"],
        "median": overlap_B_full["overlap_at_10_median"],
    })

    print_summary_table({"rows": summary_rows, "overlap_rows": overlap_rows})

    # ========================================
    # 結果の保存
    # ========================================
    elapsed = time.time() - start_time

    with open(config.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指標保存: {config.RESULTS_DIR / 'metrics.json'}")

    # 実行ログ
    log_lines = [
        f"実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model A: {config.MODEL_A}",
        f"Model B: {config.MODEL_B}",
        f"アンカー数: {config.NUM_ANCHORS}",
        f"クエリ数: {config.NUM_QUERIES}",
        f"シード: {config.RANDOM_SEED}",
        f"実行時間: {elapsed:.1f}秒",
        "",
        f"--- Cross-Model A→B (K={config.NUM_ANCHORS}) ---",
        *[f"  {k}: {v}" for k, v in cross_metrics.items()],
        "",
        f"--- 近傍構造保存率 Model A (K={config.NUM_ANCHORS}) ---",
        *[f"  {k}: {v}" for k, v in overlap_A_full.items()],
        "",
        f"--- 近傍構造保存率 Model B (K={config.NUM_ANCHORS}) ---",
        *[f"  {k}: {v}" for k, v in overlap_B_full.items()],
    ]
    with open(config.RESULTS_DIR / "experiment_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # 判定
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)
    r1 = cross_metrics["recall_at_1"] * 100
    if r1 > 60:
        print(f"  Recall@1 = {r1:.1f}% → 目標達成! Phase 1に進む価値あり")
    elif r1 > 30:
        print(f"  Recall@1 = {r1:.1f}% → 有望。仮説に根拠あり、Phase 1で改善を検討")
    elif r1 > 10:
        print(f"  Recall@1 = {r1:.1f}% → 可能性あり。アンカー選定やカーネル関数の変更を検討")
    else:
        print(f"  Recall@1 = {r1:.1f}% → コサイン類似度ベースでは不十分。線形変換の追加を検討")

    ov = overlap_A_full["overlap_at_10"] * 100
    if ov > 90:
        print(f"  Overlap@10(A) = {ov:.1f}% → 情報ロス小。精度低下はモデル間構造差に起因")
    elif ov > 70:
        print(f"  Overlap@10(A) = {ov:.1f}% → 情報ロス中程度。アンカー増加/最適化で改善余地あり")
    else:
        print(f"  Overlap@10(A) = {ov:.1f}% → 情報ロス大。アンカー選定の改善が先決")

    print(f"\n実行時間: {elapsed:.1f}秒")
    print("完了!")


if __name__ == "__main__":
    main()
