"""
RAT Phase 1: アンカー選定とカーネル関数の最適化

3軸を独立に検証：
  1. アンカー選定（ランダム / k-means / FPS / 両モデル合議）
  2. カーネル関数（cosine / RBF / poly）
  3. 最良アンカー × 最良カーネルの組み合わせ

全実験 K=500 で統一。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from src.anchor_sampler import (
    sample_anchors_and_queries,
    save_data,
    select_anchors_kmeans,
    select_anchors_fps,
    select_anchors_consensus,
)
from src.embedder import embed_texts
from src.relative_repr import to_relative
from src.evaluator import evaluate_retrieval, evaluate_neighbor_preservation

K = 500  # 全実験で統一


def run_single_experiment(
    query_emb_A: np.ndarray,
    query_emb_B: np.ndarray,
    anchor_emb_A: np.ndarray,
    anchor_emb_B: np.ndarray,
    kernel: str = "cosine",
    **kernel_params,
) -> dict:
    """1つの実験条件でCross-Model RetrievalとOverlap@10を計測する。"""
    rel_A = to_relative(query_emb_A, anchor_emb_A, kernel=kernel, **kernel_params)
    rel_B = to_relative(query_emb_B, anchor_emb_B, kernel=kernel, **kernel_params)

    cross = evaluate_retrieval(rel_A, rel_B)
    overlap_A = evaluate_neighbor_preservation(query_emb_A, rel_A, k=10)
    overlap_B = evaluate_neighbor_preservation(query_emb_B, rel_B, k=10)

    return {
        "recall_at_1": cross["recall_at_1"],
        "recall_at_5": cross["recall_at_5"],
        "recall_at_10": cross["recall_at_10"],
        "mrr": cross["mrr"],
        "overlap_at_10_A": overlap_A["overlap_at_10"],
        "overlap_at_10_B": overlap_B["overlap_at_10"],
    }


def print_results_table(rows: list[dict]):
    """結果テーブルを出力する。"""
    print("\n" + "=" * 100)
    print("  Phase 1 結果サマリ")
    print("=" * 100)

    header = (
        f"{'実験':<12} {'アンカー選定':<14} {'カーネル':<10} "
        f"{'Recall@1':>9} {'Recall@10':>10} {'MRR':>7} "
        f"{'Overlap@10(A)':>14} {'Overlap@10(B)':>14}"
    )
    print(header)
    print("-" * 100)

    for row in rows:
        r1 = f"{row['recall_at_1']*100:.1f}%"
        r10 = f"{row['recall_at_10']*100:.1f}%"
        mrr = f"{row['mrr']:.3f}"
        oa = f"{row['overlap_at_10_A']*100:.1f}%"
        ob = f"{row['overlap_at_10_B']*100:.1f}%"
        print(
            f"{row['label']:<12} {row['anchor']:<14} {row['kernel']:<10} "
            f"{r1:>9} {r10:>10} {mrr:>7} {oa:>14} {ob:>14}"
        )

    print("=" * 100)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================
    # データ準備: アンカー候補プールとクエリ
    # ========================================
    print("=" * 60)
    print("データ準備")
    print("=" * 60)

    # 大きめの候補プールを確保（選定用）
    # K=500をいろいろな方法で選ぶので、候補は多めに
    CANDIDATE_POOL = 2000
    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )

    print(f"\n候補プール: {len(candidates)}件, クエリ: {len(queries)}件")
    print("候補プールをembedding中...")

    cand_emb_A = embed_texts(config.MODEL_A, candidates)
    cand_emb_B = embed_texts(config.MODEL_B, candidates)
    query_emb_A = embed_texts(config.MODEL_A, queries)
    query_emb_B = embed_texts(config.MODEL_B, queries)

    print(f"候補embedding: A={cand_emb_A.shape}, B={cand_emb_B.shape}")
    print(f"クエリembedding: A={query_emb_A.shape}, B={query_emb_B.shape}")

    # ========================================
    # 軸1: アンカー選定の最適化（カーネル=cosine固定）
    # ========================================
    print("\n" + "=" * 60)
    print("軸1: アンカー選定の最適化 (K=500, cosineカーネル)")
    print("=" * 60)

    all_rows = []

    # a) ランダム
    print("\n--- ランダム ---")
    import random
    rng = random.Random(config.RANDOM_SEED)
    random_indices = rng.sample(range(CANDIDATE_POOL), K)
    anchor_emb_A_rand = cand_emb_A[random_indices]
    anchor_emb_B_rand = cand_emb_B[random_indices]

    res = run_single_experiment(query_emb_A, query_emb_B, anchor_emb_A_rand, anchor_emb_B_rand)
    res["label"] = "1a"
    res["anchor"] = "ランダム"
    res["kernel"] = "cosine"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # b) k-means（Model Aの空間）
    print("\n--- k-means ---")
    kmeans_indices, _ = select_anchors_kmeans(cand_emb_A, candidates, K)
    anchor_emb_A_km = cand_emb_A[kmeans_indices]
    anchor_emb_B_km = cand_emb_B[kmeans_indices]

    res = run_single_experiment(query_emb_A, query_emb_B, anchor_emb_A_km, anchor_emb_B_km)
    res["label"] = "1b"
    res["anchor"] = "k-means"
    res["kernel"] = "cosine"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # c) Farthest Point Sampling（Model Aの空間）
    print("\n--- FPS ---")
    fps_indices, _ = select_anchors_fps(cand_emb_A, candidates, K)
    anchor_emb_A_fps = cand_emb_A[fps_indices]
    anchor_emb_B_fps = cand_emb_B[fps_indices]

    res = run_single_experiment(query_emb_A, query_emb_B, anchor_emb_A_fps, anchor_emb_B_fps)
    res["label"] = "1c"
    res["anchor"] = "FPS"
    res["kernel"] = "cosine"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # d) 両モデル合議
    print("\n--- 両モデル合議 ---")
    cons_indices, _ = select_anchors_consensus(cand_emb_A, cand_emb_B, candidates, K)
    anchor_emb_A_cons = cand_emb_A[cons_indices]
    anchor_emb_B_cons = cand_emb_B[cons_indices]

    res = run_single_experiment(query_emb_A, query_emb_B, anchor_emb_A_cons, anchor_emb_B_cons)
    res["label"] = "1d"
    res["anchor"] = "両モデル合議"
    res["kernel"] = "cosine"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # 軸1の勝者を特定
    axis1_results = all_rows[:4]
    best_anchor_row = max(axis1_results, key=lambda r: r["recall_at_1"])
    best_anchor_name = best_anchor_row["anchor"]
    print(f"\n軸1 勝者: {best_anchor_name} (Recall@1={best_anchor_row['recall_at_1']*100:.1f}%)")

    # 勝者のアンカーembeddingを保持
    anchor_map = {
        "ランダム": (anchor_emb_A_rand, anchor_emb_B_rand),
        "k-means": (anchor_emb_A_km, anchor_emb_B_km),
        "FPS": (anchor_emb_A_fps, anchor_emb_B_fps),
        "両モデル合議": (anchor_emb_A_cons, anchor_emb_B_cons),
    }

    # ========================================
    # 軸2: カーネル関数の変更（アンカー=ランダム500固定）
    # ========================================
    print("\n" + "=" * 60)
    print("軸2: カーネル関数の変更 (ランダムアンカーK=500)")
    print("=" * 60)

    # a) cosine（既に計測済み、再利用）
    # all_rows[0] がランダム+cosine

    # b) RBF
    print("\n--- RBFカーネル ---")
    res = run_single_experiment(
        query_emb_A, query_emb_B, anchor_emb_A_rand, anchor_emb_B_rand,
        kernel="rbf",
    )
    res["label"] = "2b"
    res["anchor"] = "ランダム"
    res["kernel"] = "rbf"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # c) 多項式
    print("\n--- 多項式カーネル ---")
    res = run_single_experiment(
        query_emb_A, query_emb_B, anchor_emb_A_rand, anchor_emb_B_rand,
        kernel="poly", degree=2, coef0=1.0,
    )
    res["label"] = "2c"
    res["anchor"] = "ランダム"
    res["kernel"] = "poly"
    all_rows.append(res)
    print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Overlap@10(A)={res['overlap_at_10_A']*100:.1f}%")

    # 軸2の勝者を特定（ランダム+cosineも含めて比較）
    axis2_results = [all_rows[0], all_rows[4], all_rows[5]]  # cosine, rbf, poly
    best_kernel_row = max(axis2_results, key=lambda r: r["recall_at_1"])
    best_kernel_name = best_kernel_row["kernel"]
    print(f"\n軸2 勝者: {best_kernel_name} (Recall@1={best_kernel_row['recall_at_1']*100:.1f}%)")

    # ========================================
    # 軸3: 組み合わせ（最良アンカー × 最良カーネル）
    # ========================================
    print("\n" + "=" * 60)
    print(f"軸3: 組み合わせ ({best_anchor_name} × {best_kernel_name})")
    print("=" * 60)

    best_anchor_A, best_anchor_B = anchor_map[best_anchor_name]

    # 既に計測済みの組み合わせでなければ実行
    already_measured = (
        best_anchor_name == best_anchor_row["anchor"]
        and best_kernel_name == best_anchor_row["kernel"]
    )
    if already_measured:
        combo_res = best_anchor_row.copy()
        combo_res["label"] = "3"
        print(f"  (軸1の勝者と同一条件、結果を再利用)")
    else:
        combo_res = run_single_experiment(
            query_emb_A, query_emb_B, best_anchor_A, best_anchor_B,
            kernel=best_kernel_name,
        )
        combo_res["label"] = "3"
        combo_res["anchor"] = best_anchor_name
        combo_res["kernel"] = best_kernel_name

    all_rows.append(combo_res)
    print(f"  Recall@1={combo_res['recall_at_1']*100:.1f}%, Overlap@10(A)={combo_res['overlap_at_10_A']*100:.1f}%")

    # ========================================
    # 結果出力
    # ========================================
    print_results_table(all_rows)

    # 判定
    best_r1 = combo_res["recall_at_1"] * 100
    baseline_r1 = all_rows[0]["recall_at_1"] * 100
    improvement = best_r1 - baseline_r1

    print(f"\n最良組み合わせ: {combo_res.get('anchor', best_anchor_name)} × {combo_res.get('kernel', best_kernel_name)}")
    print(f"  Recall@1: {baseline_r1:.1f}% → {best_r1:.1f}% ({improvement:+.1f}pt)")

    if best_r1 > 60:
        print(f"  → 目標達成! Phase 2に進む価値あり")
    elif best_r1 > 50:
        print(f"  → 大幅改善。目標60%まであと{60-best_r1:.1f}pt")
    else:
        print(f"  → 改善はあるが目標60%には未到達。軽量線形変換の追加を検討")

    # 保存
    elapsed = time.time() - start_time

    results_data = {
        "experiments": all_rows,
        "best_anchor": best_anchor_name,
        "best_kernel": best_kernel_name,
        "elapsed_seconds": elapsed,
    }
    with open(config.RESULTS_DIR / "phase1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {config.RESULTS_DIR / 'phase1_metrics.json'}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
