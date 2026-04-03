"""
実験 5c: Few-shot 線形補正

相対表現空間でridge回帰（正則化付き）を適用。
元embedding空間でのProcrustesはモデル次元が異なるペアでは適用不可のため除外。

論文 §6 Discussion 用: "N paired examples improve R@1 by +X%"
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
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

NUM_SHOTS = [0, 5, 10, 20, 50]
NUM_ANCHORS = 500
NUM_TRIALS = 5
CANDIDATE_POOL = 2000

MODEL_PAIRS = [
    ("A×B", config.MODEL_A, config.MODEL_B),
    ("A×C", config.MODEL_A, config.MODEL_C),
    ("B×C", config.MODEL_B, config.MODEL_C),
]


def diagonal_scaling(
    rel_src: np.ndarray,
    rel_tgt: np.ndarray,
    paired_indices: list[int],
) -> np.ndarray:
    """
    各アンカー次元を独立にスケーリング+バイアス補正。
    パラメータ数 = 2K（K=アンカー数）で、少数ペアでも安定。
    """
    src_train = rel_src[paired_indices]  # (n_pairs, K)
    tgt_train = rel_tgt[paired_indices]  # (n_pairs, K)

    # 各次元で独立に scale, bias を最小二乗で求める
    K = rel_src.shape[1]
    scale = np.ones(K)
    bias = np.zeros(K)

    for k in range(K):
        x = src_train[:, k]
        y = tgt_train[:, k]
        # y = scale * x + bias の最小二乗解
        A = np.stack([x, np.ones_like(x)], axis=1)  # (n_pairs, 2)
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        scale[k] = sol[0]
        bias[k] = sol[1]

    return rel_src * scale[None, :] + bias[None, :]


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 5c: Few-shot Ridge補正")
    print(f"  ショット数: {NUM_SHOTS}")
    print(f"  試行回数: {NUM_TRIALS}")
    print(f"  方式: 対角スケーリング (2Kパラメータ)")
    print("=" * 60)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES,
    )
    print(f"候補: {len(candidates)}文, クエリ: {len(queries)}文")

    all_results = []

    for pair_name, model_1, model_2 in MODEL_PAIRS:
        print(f"\n{'='*60}")
        print(f"ペア: {pair_name}")
        print(f"{'='*60}")

        # Embedding
        cand_emb_1 = embed_texts(model_1, candidates)
        cand_emb_2 = embed_texts(model_2, candidates)
        query_emb_1 = embed_texts(model_1, queries)
        query_emb_2 = embed_texts(model_2, queries)

        # FPSアンカー選定
        dummy_texts = [f"d{i}" for i in range(len(candidates))]
        fps_indices, _ = select_anchors_fps(cand_emb_1, dummy_texts, NUM_ANCHORS)
        anchor_1 = cand_emb_1[fps_indices]
        anchor_2 = cand_emb_2[fps_indices]

        # 相対表現（zero-shot用 & ridge用）
        rel_1 = to_relative(query_emb_1, anchor_1, kernel="poly", degree=2, coef0=1.0)
        rel_2 = to_relative(query_emb_2, anchor_2, kernel="poly", degree=2, coef0=1.0)
        rel_1_z = normalize_zscore(rel_1)
        rel_2_z = normalize_zscore(rel_2)

        for n_shots in NUM_SHOTS:
            if n_shots == 0:
                metrics = evaluate_retrieval(rel_1_z, rel_2_z)
                result = {
                    "pair": pair_name, "n_shots": 0, "method": "zero-shot",
                    "trial": "mean", **metrics,
                }
                all_results.append(result)
                print(
                    f"  {n_shots:>3} shots (zero-shot):  "
                    f"R@1={metrics['recall_at_1']*100:5.1f}%  "
                    f"R@10={metrics['recall_at_10']*100:5.1f}%  "
                    f"MRR={metrics['mrr']:.3f}"
                )
                continue

            # Ridge in relative space
            trial_metrics = []
            for trial in range(NUM_TRIALS):
                rng = np.random.RandomState(config.RANDOM_SEED + trial)
                paired = rng.choice(len(queries), size=n_shots, replace=False).tolist()

                rel_ridge = diagonal_scaling(rel_1_z, rel_2_z, paired)
                m = evaluate_retrieval(rel_ridge, rel_2_z)
                trial_metrics.append(m)

            mean_m = {}
            for key in trial_metrics[0]:
                vals = [m[key] for m in trial_metrics]
                mean_m[key] = float(np.mean(vals))
                mean_m[f"{key}_std"] = float(np.std(vals))

            all_results.append({
                "pair": pair_name, "n_shots": n_shots, "method": "diag_scale",
                "trial": "mean", **mean_m,
            })
            print(
                f"  {n_shots:>3} shots (diag):      "
                f"R@1={mean_m['recall_at_1']*100:5.1f}% (±{mean_m['recall_at_1_std']*100:.1f})  "
                f"R@10={mean_m['recall_at_10']*100:5.1f}%  "
                f"MRR={mean_m['mrr']:.3f}"
            )

    # --- サマリーテーブル ---
    print("\n" + "=" * 80)
    print("  Diagonal Scaling few-shot 結果サマリー (R@1%)")
    print("=" * 80)

    for pair_name, _, _ in MODEL_PAIRS:
        pair_results = [r for r in all_results if r["pair"] == pair_name]
        zero = next(r for r in pair_results if r["n_shots"] == 0)

        print(f"\n  {pair_name} (zero-shot: {zero['recall_at_1']*100:.1f}%)")
        print(f"  {'shots':>5}  {'Diag R@1':>16}")
        print(f"  {'-'*25}")
        for n in NUM_SHOTS:
            if n == 0:
                continue
            r = next(r for r in pair_results if r["n_shots"] == n)
            delta = (r["recall_at_1"] - zero["recall_at_1"]) * 100
            print(f"  {n:>5}  {r['recall_at_1']*100:5.1f}% (±{r['recall_at_1_std']*100:.1f})  Δ{delta:+.1f}%")

    # --- JSON保存 ---
    elapsed = time.time() - start_time
    output = {
        "protocol": "FPS+poly+z-score + diagonal_scaling",
        "num_anchors": NUM_ANCHORS,
        "num_shots": NUM_SHOTS,
        "num_trials": NUM_TRIALS,
        "methods": ["zero-shot", "diag_scale"],
        "model_pairs": [(n, m1, m2) for n, m1, m2 in MODEL_PAIRS],
        "results": all_results,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "procrustes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
