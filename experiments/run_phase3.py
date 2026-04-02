"""
RAT Phase 3: E5-largeの類似度潰れ対策

相対表現の後処理でB×Cペアの性能を回復できるか検証する。
全手法 K=500, FPS+poly ベース。

手法:
  1. z-score正規化
  2. ランク変換
  3. Top-kマスク (k=50)
  4. 温度softmax (T=0.05, 0.1, 0.2)
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import (
    to_relative,
    normalize_zscore,
    normalize_rank,
    normalize_topk_mask,
    normalize_softmax,
)
from src.evaluator import evaluate_retrieval

K = 500
CANDIDATE_POOL = 2000

PAIRS = [
    ("A×B", "A", "B"),
    ("A×C", "A", "C"),
    ("B×C", "B", "C"),
]


def evaluate_with_normalizer(
    query_embs: dict,
    anchor_embs: dict,
    normalizer_fn,
    label: str,
) -> dict:
    """全ペアを正規化手法付きで評価する。"""
    results = {"label": label}
    for pair_label, x, y in PAIRS:
        # polyカーネルで相対表現を計算
        rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)

        # 後処理を適用
        if normalizer_fn is not None:
            rel_x = normalizer_fn(rel_x)
            rel_y = normalizer_fn(rel_y)

        cross = evaluate_retrieval(rel_x, rel_y)
        results[pair_label] = {
            "recall_at_1": cross["recall_at_1"],
            "recall_at_5": cross["recall_at_5"],
            "recall_at_10": cross["recall_at_10"],
            "mrr": cross["mrr"],
        }
    return results


def print_results_table(all_results: list[dict]):
    """結果テーブルを出力する。"""
    print("\n" + "=" * 100)
    print("  Phase 3 結果: 類似度潰れ対策")
    print("=" * 100)

    header = (
        f"{'手法':<25} "
        f"{'B×C R@1':>9} {'B×C R@10':>9} {'B×C MRR':>8} "
        f"{'A×B R@1':>9} {'A×C R@1':>9}"
    )
    print(header)
    print("-" * 100)

    for row in all_results:
        bc = row["B×C"]
        ab = row["A×B"]
        ac = row["A×C"]
        print(
            f"{row['label']:<25} "
            f"{bc['recall_at_1']*100:>8.1f}% {bc['recall_at_10']*100:>8.1f}% {bc['mrr']:>8.3f} "
            f"{ab['recall_at_1']*100:>8.1f}% {ac['recall_at_1']*100:>8.1f}%"
        )

    print("=" * 100)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "A": config.MODEL_A,
        "B": config.MODEL_B,
        "C": config.MODEL_C,
    }

    # データ準備
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
        print(f"  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)

    # FPSアンカー選定（Model A基準）
    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in models}

    # 実験実行
    print("\n" + "=" * 60)
    print("類似度潰れ対策の検証")
    print("=" * 60)

    all_results = []

    # ベースライン（正規化なし）
    print("\n--- ベースライン ---")
    res = evaluate_with_normalizer(query_embs, anchor_embs, None, "ベースライン (FPS+poly)")
    all_results.append(res)
    print(f"  B×C: Recall@1={res['B×C']['recall_at_1']*100:.1f}%")

    # 1. z-score正規化
    print("\n--- z-score正規化 ---")
    res = evaluate_with_normalizer(query_embs, anchor_embs, normalize_zscore, "z-score正規化")
    all_results.append(res)
    print(f"  B×C: Recall@1={res['B×C']['recall_at_1']*100:.1f}%")

    # 2. ランク変換
    print("\n--- ランク変換 ---")
    res = evaluate_with_normalizer(query_embs, anchor_embs, normalize_rank, "ランク変換")
    all_results.append(res)
    print(f"  B×C: Recall@1={res['B×C']['recall_at_1']*100:.1f}%")

    # 3. Top-kマスク
    print("\n--- Top-kマスク (k=50) ---")
    res = evaluate_with_normalizer(
        query_embs, anchor_embs,
        lambda r: normalize_topk_mask(r, top_k=50),
        "Top-kマスク (k=50)",
    )
    all_results.append(res)
    print(f"  B×C: Recall@1={res['B×C']['recall_at_1']*100:.1f}%")

    # 4. 温度softmax
    for temp in [0.05, 0.1, 0.2]:
        print(f"\n--- 温度softmax (T={temp}) ---")
        res = evaluate_with_normalizer(
            query_embs, anchor_embs,
            lambda r, t=temp: normalize_softmax(r, temperature=t),
            f"温度softmax (T={temp})",
        )
        all_results.append(res)
        print(f"  B×C: Recall@1={res['B×C']['recall_at_1']*100:.1f}%")

    # 結果出力
    print_results_table(all_results)

    # 勝者判定
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    baseline_bc = all_results[0]["B×C"]["recall_at_1"]
    best_row = max(all_results, key=lambda r: r["B×C"]["recall_at_1"])
    best_bc = best_row["B×C"]["recall_at_1"]
    improvement = (best_bc - baseline_bc) * 100

    print(f"\n  最良手法: {best_row['label']}")
    print(f"  B×C Recall@1: {baseline_bc*100:.1f}% → {best_bc*100:.1f}% ({improvement:+.1f}pt)")

    # 他のペアへの影響チェック
    baseline_ab = all_results[0]["A×B"]["recall_at_1"]
    baseline_ac = all_results[0]["A×C"]["recall_at_1"]
    best_ab = best_row["A×B"]["recall_at_1"]
    best_ac = best_row["A×C"]["recall_at_1"]

    print(f"  A×B: {baseline_ab*100:.1f}% → {best_ab*100:.1f}% ({(best_ab-baseline_ab)*100:+.1f}pt)")
    print(f"  A×C: {baseline_ac*100:.1f}% → {best_ac*100:.1f}% ({(best_ac-baseline_ac)*100:+.1f}pt)")

    if best_bc > 0.6:
        print(f"\n  → B×Cが60%突破! 潰れ対策で全ペア目標達成")
    elif best_bc > 0.3:
        print(f"\n  → B×C大幅改善。完全解決には至っていないが有効な対策")
    else:
        print(f"\n  → 改善限定的。後処理だけでは不十分、構造的アプローチが必要")

    # 保存
    elapsed = time.time() - start_time

    output = {
        "experiments": all_results,
        "best_method": best_row["label"],
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "phase3_normalization.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
