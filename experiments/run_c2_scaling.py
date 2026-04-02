"""
実験 C-2: 最終プロトコルのスケーリングカーブ

A×Bペアで K=[100, 200, 500, 1000] に対して：
  a) ランダム + cosine（Phase 0相当）
  b) FPS + poly + z-score（最終プロトコル）

2本のカーブを同一グラフにプロットして比較。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K_VALUES = [100, 200, 500, 1000]
CANDIDATE_POOL = 2000


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 C-2: スケーリングカーブ（ランダム+cosine vs FPS+poly+z-score）")
    print("=" * 60)

    # データ準備
    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )
    print(f"候補: {len(candidates)}文, クエリ: {len(queries)}文")

    # A×Bペアのembedding
    print("\nEmbedding...")
    cand_emb_A = embed_texts(config.MODEL_A, candidates)
    cand_emb_B = embed_texts(config.MODEL_B, candidates)
    query_emb_A = embed_texts(config.MODEL_A, queries)
    query_emb_B = embed_texts(config.MODEL_B, queries)
    print(f"  A: cand={cand_emb_A.shape}, query={query_emb_A.shape}")
    print(f"  B: cand={cand_emb_B.shape}, query={query_emb_B.shape}")

    # FPSアンカー（最大K=1000をまず選定、サブセットで使う）
    max_k = max(K_VALUES)
    print(f"\nFPSアンカー選定 (K={max_k})...")
    fps_indices, _ = select_anchors_fps(cand_emb_A, candidates, max_k)

    results_random = []
    results_fps = []

    rng = np.random.RandomState(config.RANDOM_SEED)

    for K in K_VALUES:
        print(f"\n--- K={K} ---")

        # (a) ランダム + cosine
        rand_indices = rng.choice(CANDIDATE_POOL, size=K, replace=False)
        anchor_A_rand = cand_emb_A[rand_indices]
        anchor_B_rand = cand_emb_B[rand_indices]

        rel_A_rand = to_relative(query_emb_A, anchor_A_rand, kernel="cosine")
        rel_B_rand = to_relative(query_emb_B, anchor_B_rand, kernel="cosine")
        m_rand = evaluate_retrieval(rel_A_rand, rel_B_rand)
        results_random.append({"K": K, **m_rand})
        print(
            f"  ランダム+cosine:       R@1={m_rand['recall_at_1']*100:.1f}%, "
            f"R@10={m_rand['recall_at_10']*100:.1f}%, MRR={m_rand['mrr']:.3f}"
        )

        # (b) FPS + poly + z-score
        fps_sub = fps_indices[:K]
        anchor_A_fps = cand_emb_A[fps_sub]
        anchor_B_fps = cand_emb_B[fps_sub]

        rel_A_fps = to_relative(
            query_emb_A, anchor_A_fps, kernel="poly", degree=2, coef0=1.0
        )
        rel_B_fps = to_relative(
            query_emb_B, anchor_B_fps, kernel="poly", degree=2, coef0=1.0
        )
        rel_A_fps = normalize_zscore(rel_A_fps)
        rel_B_fps = normalize_zscore(rel_B_fps)
        m_fps = evaluate_retrieval(rel_A_fps, rel_B_fps)
        results_fps.append({"K": K, **m_fps})
        print(
            f"  FPS+poly+z-score:      R@1={m_fps['recall_at_1']*100:.1f}%, "
            f"R@10={m_fps['recall_at_10']*100:.1f}%, MRR={m_fps['mrr']:.3f}"
        )

        gain = (m_fps["recall_at_1"] - m_rand["recall_at_1"]) * 100
        print(f"  FPSの貢献: +{gain:.1f}%")

    # 結果テーブル
    print("\n" + "=" * 80)
    print("  スケーリング比較 (A×B)")
    print("=" * 80)
    print(f"\n{'K':>6} {'Random R@1':>12} {'FPS+proto R@1':>14} {'差分':>8} {'FPS MRR':>10}")
    print("-" * 55)
    for r, f in zip(results_random, results_fps):
        diff = (f["recall_at_1"] - r["recall_at_1"]) * 100
        print(
            f"{r['K']:>6} {r['recall_at_1']*100:>11.1f}% "
            f"{f['recall_at_1']*100:>13.1f}% {diff:>+7.1f}% "
            f"{f['mrr']:>10.3f}"
        )

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ks = K_VALUES

    # Recall@1
    ax = axes[0]
    r1_rand = [r["recall_at_1"] * 100 for r in results_random]
    r1_fps = [r["recall_at_1"] * 100 for r in results_fps]
    ax.plot(ks, r1_rand, "o--", color="#888888", linewidth=2, markersize=8,
            label="Random + cosine (Phase 0)")
    ax.plot(ks, r1_fps, "s-", color="#2196F3", linewidth=2, markersize=8,
            label="FPS + poly + z-score (Final)")
    ax.set_xlabel("Number of Anchors (K)", fontsize=12)
    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("A×B: Recall@1 Scaling", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    # MRR
    ax = axes[1]
    mrr_rand = [r["mrr"] for r in results_random]
    mrr_fps = [r["mrr"] for r in results_fps]
    ax.plot(ks, mrr_rand, "o--", color="#888888", linewidth=2, markersize=8,
            label="Random + cosine (Phase 0)")
    ax.plot(ks, mrr_fps, "s-", color="#2196F3", linewidth=2, markersize=8,
            label="FPS + poly + z-score (Final)")
    ax.set_xlabel("Number of Anchors (K)", fontsize=12)
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title("A×B: MRR Scaling", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    plt.tight_layout()
    save_path = config.RESULTS_DIR / "scaling_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nグラフ保存: {save_path}")

    # 保存
    elapsed = time.time() - start_time
    output = {
        "pair": "A×B",
        "K_values": K_VALUES,
        "random_cosine": results_random,
        "fps_poly_zscore": results_fps,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "c2_scaling.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
