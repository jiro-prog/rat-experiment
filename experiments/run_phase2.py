"""
RAT Phase 2: 3モデルクロス検証

FPS+polyプロトコルが複数のモデルペアで汎用的に機能するかを検証する。

Models:
  A: sentence-transformers/all-MiniLM-L6-v2 (384d)
  B: intfloat/multilingual-e5-large (1024d)
  C: BAAI/bge-small-en-v1.5 (384d)

全ペア: A×B, A×C, B×C
条件: FPS (K=500) + poly kernel ((x·a+1)^2)
判定基準: 全ペアでRecall@1 > 60%
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
from src.relative_repr import to_relative
from src.evaluator import evaluate_retrieval, evaluate_neighbor_preservation

K = 500
CANDIDATE_POOL = 2000


def run_pair(
    label: str,
    model_x_name: str,
    model_y_name: str,
    query_emb_x: np.ndarray,
    query_emb_y: np.ndarray,
    anchor_emb_x: np.ndarray,
    anchor_emb_y: np.ndarray,
) -> dict:
    """1ペアの評価を実行する。"""
    rel_x = to_relative(query_emb_x, anchor_emb_x, kernel="poly", degree=2, coef0=1.0)
    rel_y = to_relative(query_emb_y, anchor_emb_y, kernel="poly", degree=2, coef0=1.0)

    cross = evaluate_retrieval(rel_x, rel_y)
    overlap_x = evaluate_neighbor_preservation(query_emb_x, rel_x, k=10)
    overlap_y = evaluate_neighbor_preservation(query_emb_y, rel_y, k=10)

    return {
        "pair": label,
        "model_x": model_x_name,
        "model_y": model_y_name,
        "recall_at_1": cross["recall_at_1"],
        "recall_at_5": cross["recall_at_5"],
        "recall_at_10": cross["recall_at_10"],
        "mrr": cross["mrr"],
        "median_rank": cross["median_rank"],
        "overlap_at_10_x": overlap_x["overlap_at_10"],
        "overlap_at_10_y": overlap_y["overlap_at_10"],
    }


def print_results_table(rows: list[dict]):
    """結果テーブルを出力する。"""
    print("\n" + "=" * 110)
    print("  Phase 2 結果: 3モデルクロス検証 (FPS K=500 + poly kernel)")
    print("=" * 110)

    header = (
        f"{'ペア':<8} {'Model X':<14} {'Model Y':<14} "
        f"{'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>7} "
        f"{'Overlap@10(X)':>14} {'Overlap@10(Y)':>14}"
    )
    print(header)
    print("-" * 110)

    for row in rows:
        # モデル名を短縮
        mx = row["model_x"].split("/")[-1]
        my = row["model_y"].split("/")[-1]
        r1 = f"{row['recall_at_1']*100:.1f}%"
        r5 = f"{row['recall_at_5']*100:.1f}%"
        r10 = f"{row['recall_at_10']*100:.1f}%"
        mrr = f"{row['mrr']:.3f}"
        ox = f"{row['overlap_at_10_x']*100:.1f}%"
        oy = f"{row['overlap_at_10_y']*100:.1f}%"
        print(
            f"{row['pair']:<8} {mx:<14} {my:<14} "
            f"{r1:>9} {r5:>9} {r10:>10} {mrr:>7} {ox:>14} {oy:>14}"
        )

    print("=" * 110)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "A": config.MODEL_A,  # MiniLM (384d)
        "B": config.MODEL_B,  # E5-large (1024d)
        "C": config.MODEL_C,  # BGE-small (384d)
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
    print(f"候補プール: {len(candidates)}件, クエリ: {len(queries)}件")

    # 全モデルでembedding
    print("\n全モデルでembedding取得中...")
    cand_embs = {}
    query_embs = {}
    for label, model_name in models.items():
        short = model_name.split("/")[-1]
        print(f"\n  Model {label} ({short}):")

        print(f"    候補 {len(candidates)}件...")
        cand_embs[label] = embed_texts(model_name, candidates)

        print(f"    クエリ {len(queries)}件...")
        query_embs[label] = embed_texts(model_name, queries)

        print(f"    → cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    # ========================================
    # FPSアンカー選定（Model Aの空間で実施）
    # ========================================
    print("\n" + "=" * 60)
    print("FPSアンカー選定 (Model Aの空間, K=500)")
    print("=" * 60)

    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)

    # 選定されたアンカーのembeddingを全モデルから取得
    anchor_embs = {}
    for label in models:
        anchor_embs[label] = cand_embs[label][fps_indices]
        print(f"  Model {label} アンカー: {anchor_embs[label].shape}")

    # ========================================
    # 全ペア評価
    # ========================================
    print("\n" + "=" * 60)
    print("全ペア評価 (FPS + poly kernel)")
    print("=" * 60)

    pairs = [
        ("A×B", "A", "B"),
        ("A×C", "A", "C"),
        ("B×C", "B", "C"),
    ]

    results = []
    for pair_label, x, y in pairs:
        print(f"\n--- {pair_label}: {models[x].split('/')[-1]} × {models[y].split('/')[-1]} ---")
        res = run_pair(
            pair_label,
            models[x], models[y],
            query_embs[x], query_embs[y],
            anchor_embs[x], anchor_embs[y],
        )
        results.append(res)
        print(f"  Recall@1={res['recall_at_1']*100:.1f}%, Recall@10={res['recall_at_10']*100:.1f}%, MRR={res['mrr']:.3f}")

    # ========================================
    # 結果出力
    # ========================================
    print_results_table(results)

    # 判定
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    all_pass = True
    for res in results:
        r1 = res["recall_at_1"] * 100
        status = "✓ PASS" if r1 > 60 else "✗ FAIL"
        if r1 <= 60:
            all_pass = False
        print(f"  {res['pair']}: Recall@1={r1:.1f}% {status}")

    print()
    if all_pass:
        print("  全ペアでRecall@1 > 60%: RATプロトコル(FPS+poly)の汎用性を確認")
    else:
        failed = [r["pair"] for r in results if r["recall_at_1"] <= 0.6]
        print(f"  一部ペアが未達: {', '.join(failed)}")
        print("  → モデルペア固有の問題か、プロトコルの限界かを要分析")

    # A×Bの再現性確認
    ab_r1 = results[0]["recall_at_1"] * 100
    print(f"\n  A×B再現性: Phase 1=77.2% → Phase 2={ab_r1:.1f}%", end="")
    if abs(ab_r1 - 77.2) < 1.0:
        print(" (再現OK)")
    else:
        print(f" (差異: {ab_r1 - 77.2:+.1f}pt — 候補プール/サンプリングの違いによる)")

    # 保存
    elapsed = time.time() - start_time

    output = {
        "protocol": "FPS (K=500) + poly kernel ((x·a+1)^2)",
        "models": {k: v for k, v in models.items()},
        "fps_anchor_space": "Model A",
        "results": results,
        "all_pairs_pass_60": all_pass,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "phase2_cross_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
