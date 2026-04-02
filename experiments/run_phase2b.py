"""
RAT Phase 2b: FPS基準モデル依存性の診断とモデル非依存アンカー選定

Step 1: FPS基準をA/B/Cで変えてシーソー現象を確認
Step 2: TF-IDF FPSとブートストラップFPSで全ペアを検証
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
    select_anchors_fps,
    select_anchors_tfidf_fps,
    select_anchors_bootstrap_fps,
)
from src.embedder import embed_texts
from src.relative_repr import to_relative
from src.evaluator import evaluate_retrieval, evaluate_neighbor_preservation

K = 500
CANDIDATE_POOL = 2000

PAIRS = [
    ("A×B", "A", "B"),
    ("A×C", "A", "C"),
    ("B×C", "B", "C"),
]


def evaluate_pair(
    query_embs: dict, anchor_embs: dict, x: str, y: str,
) -> dict:
    """1ペアをFPS+polyで評価する。"""
    rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
    rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)
    cross = evaluate_retrieval(rel_x, rel_y)
    ov_x = evaluate_neighbor_preservation(query_embs[x], rel_x, k=10)
    ov_y = evaluate_neighbor_preservation(query_embs[y], rel_y, k=10)
    return {
        "recall_at_1": cross["recall_at_1"],
        "recall_at_5": cross["recall_at_5"],
        "recall_at_10": cross["recall_at_10"],
        "mrr": cross["mrr"],
        "overlap_at_10_x": ov_x["overlap_at_10"],
        "overlap_at_10_y": ov_y["overlap_at_10"],
    }


def evaluate_all_pairs(query_embs, anchor_embs):
    """全3ペアを評価してdict[pair_label, metrics]を返す。"""
    results = {}
    for pair_label, x, y in PAIRS:
        results[pair_label] = evaluate_pair(query_embs, anchor_embs, x, y)
    return results


def print_diagnosis_table(all_results: dict):
    """診断テーブルを出力する。"""
    print("\n" + "=" * 80)
    print("  FPS基準モデル別 Recall@1 一覧")
    print("=" * 80)
    header = f"{'FPS基準':<20} {'A×B':>10} {'A×C':>10} {'B×C':>10} {'平均':>10} {'最小':>10}"
    print(header)
    print("-" * 80)

    for method, pair_results in all_results.items():
        r_ab = pair_results["A×B"]["recall_at_1"] * 100
        r_ac = pair_results["A×C"]["recall_at_1"] * 100
        r_bc = pair_results["B×C"]["recall_at_1"] * 100
        avg = (r_ab + r_ac + r_bc) / 3
        mn = min(r_ab, r_ac, r_bc)
        print(f"{method:<20} {r_ab:>9.1f}% {r_ac:>9.1f}% {r_bc:>9.1f}% {avg:>9.1f}% {mn:>9.1f}%")

    print("=" * 80)


def print_full_table(all_results: dict):
    """全指標の詳細テーブル。"""
    print("\n" + "=" * 100)
    print("  全指標詳細")
    print("=" * 100)
    header = (
        f"{'方法':<20} {'ペア':<8} "
        f"{'Recall@1':>9} {'Recall@10':>10} {'MRR':>7} "
        f"{'Ov@10(X)':>10} {'Ov@10(Y)':>10}"
    )
    print(header)
    print("-" * 100)

    for method, pair_results in all_results.items():
        for pair_label, metrics in pair_results.items():
            r1 = f"{metrics['recall_at_1']*100:.1f}%"
            r10 = f"{metrics['recall_at_10']*100:.1f}%"
            mrr = f"{metrics['mrr']:.3f}"
            ox = f"{metrics['overlap_at_10_x']*100:.1f}%"
            oy = f"{metrics['overlap_at_10_y']*100:.1f}%"
            print(f"{method:<20} {pair_label:<8} {r1:>9} {r10:>10} {mrr:>7} {ox:>10} {oy:>10}")
        print("-" * 100)

    print("=" * 100)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "A": config.MODEL_A,
        "B": config.MODEL_B,
        "C": config.MODEL_C,
    }

    # ========================================
    # データ準備
    # ========================================
    print("=" * 60)
    print("データ準備")
    print("=" * 60)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )

    cand_embs = {}
    query_embs = {}
    for label, model_name in models.items():
        short = model_name.split("/")[-1]
        print(f"\n  Model {label} ({short}):")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)
        print(f"    cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    all_results = {}

    # ========================================
    # Step 1: FPS基準モデルの診断
    # ========================================
    print("\n" + "=" * 60)
    print("Step 1: FPS基準モデルの診断")
    print("=" * 60)

    for basis_label in ["A", "B", "C"]:
        basis_name = models[basis_label].split("/")[-1]
        print(f"\n--- FPS基準: Model {basis_label} ({basis_name}) ---")

        fps_indices, _ = select_anchors_fps(cand_embs[basis_label], candidates, K)

        anchor_embs = {}
        for label in models:
            anchor_embs[label] = cand_embs[label][fps_indices]

        pair_results = evaluate_all_pairs(query_embs, anchor_embs)
        method_name = f"FPS(Model {basis_label})"
        all_results[method_name] = pair_results

        for pl, m in pair_results.items():
            print(f"  {pl}: Recall@1={m['recall_at_1']*100:.1f}%")

    print_diagnosis_table(all_results)

    # ========================================
    # Step 2: モデル非依存アンカー選定
    # ========================================
    print("\n" + "=" * 60)
    print("Step 2: モデル非依存アンカー選定")
    print("=" * 60)

    # 2a) TF-IDF FPS
    print("\n--- TF-IDF FPS ---")
    tfidf_indices, _ = select_anchors_tfidf_fps(candidates, K)
    anchor_embs_tfidf = {}
    for label in models:
        anchor_embs_tfidf[label] = cand_embs[label][tfidf_indices]

    tfidf_results = evaluate_all_pairs(query_embs, anchor_embs_tfidf)
    all_results["TF-IDF FPS"] = tfidf_results
    for pl, m in tfidf_results.items():
        print(f"  {pl}: Recall@1={m['recall_at_1']*100:.1f}%")

    # 2b) ブートストラップFPS
    print("\n--- ブートストラップFPS ---")
    bootstrap_indices, _ = select_anchors_bootstrap_fps(
        cand_embs, candidates, K, bootstrap_k=200
    )
    anchor_embs_bootstrap = {}
    for label in models:
        anchor_embs_bootstrap[label] = cand_embs[label][bootstrap_indices]

    bootstrap_results = evaluate_all_pairs(query_embs, anchor_embs_bootstrap)
    all_results["Bootstrap FPS"] = bootstrap_results
    for pl, m in bootstrap_results.items():
        print(f"  {pl}: Recall@1={m['recall_at_1']*100:.1f}%")

    # ========================================
    # 結果出力
    # ========================================
    print_diagnosis_table(all_results)
    print_full_table(all_results)

    # 勝者判定（最小ペアのRecall@1が最も高い方法）
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    best_method = None
    best_min_r1 = -1
    for method, pair_results in all_results.items():
        r1s = [m["recall_at_1"] for m in pair_results.values()]
        min_r1 = min(r1s)
        avg_r1 = sum(r1s) / len(r1s)
        if min_r1 > best_min_r1:
            best_min_r1 = min_r1
            best_method = method
        print(f"  {method:<20}: min={min_r1*100:.1f}%, avg={avg_r1*100:.1f}%")

    print(f"\n  最良方法（最小ペア基準）: {best_method} (min Recall@1={best_min_r1*100:.1f}%)")

    all_pass = best_min_r1 > 0.6
    if all_pass:
        print(f"  → 全ペアRecall@1 > 60%: 汎用プロトコルとして成立")
    else:
        print(f"  → 全ペア60%未達。最弱ペアの改善が必要")

    # 保存
    elapsed = time.time() - start_time
    output = {
        "results": {k: v for k, v in all_results.items()},
        "best_method": best_method,
        "best_min_recall_at_1": best_min_r1,
        "all_pairs_pass_60": all_pass,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "phase2b_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
