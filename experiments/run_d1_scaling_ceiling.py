"""
Experiment D1 Scaling Ceiling: K=1000, 2000 でのスケーリング天井確認

代表的な6有向ペア（3ペア×双方向）で K=[50,100,200,500,1000,2000] を実行。
K=2000 は候補プール全体（2000点）を使うため、FPSの理論上の最大値。

比較手法:
  - RAT (poly kernel + adaptive z-score)
  - Ridge (best λ from [1e-4, 1e-2, 1.0, 100.0])
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval
from src.baselines import fit_ridge, transform_ridge
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# 設定
# ========================================
ANCHOR_COUNTS = [50, 100, 200, 500, 1000, 2000]
RIDGE_ALPHAS = [1e-4, 1e-2, 1.0, 100.0]
SEED = 42
DATA_DIR = config.DATA_DIR / "d2_matrix"

# 代表的ペア（双方向）
PAIRS = [
    ("A", "C"),  # 同次元384, K=500でRAT 97.2%
    ("C", "A"),
    ("A", "B"),  # 異次元384→1024, K=500でRAT 79.0%
    ("B", "A"),
    ("E", "I"),  # 同次元1024, 同ファミリー高精度
    ("I", "E"),
]


def compute_sim_mean(anchor_emb: np.ndarray) -> float:
    sim = cosine_similarity(anchor_emb)
    n = len(sim)
    return float((sim.sum() - n) / (n * (n - 1)))


def main():
    start_time = time.time()

    print("=" * 70)
    print("D1 Scaling Ceiling: K=[50..2000] for 6 directed pairs")
    print("=" * 70)

    # メタデータ
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    # 必要なモデルだけロード
    needed_labels = sorted(set(l for pair in PAIRS for l in pair))
    print(f"必要モデル: {needed_labels}")

    cand_embs = {}
    query_embs = {}
    for label in needed_labels:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
        info = config.MATRIX_MODELS[label]
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape} "
              f"({info['family']}, dim={info['dim']})")

    n_cand = cand_embs[needed_labels[0]].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    # FPSインデックス事前計算（K=2000 = 全候補なのでK=2000まで）
    # K=2000の場合、FPSで全候補を選ぶ = 全候補使用
    print(f"\nFPSインデックス計算 (max K={max(ANCHOR_COUNTS)}, seed={SEED})...")
    fps_cache = {}
    for label in needed_labels:
        max_k = min(max(ANCHOR_COUNTS), n_cand)
        fps_idx, _ = select_anchors_fps(
            cand_embs[label], candidates_dummy, max_k, seed=SEED,
        )
        fps_cache[label] = np.array(fps_idx)
        print(f"  {label}: FPS {len(fps_cache[label])} indices computed")

    # 実験実行
    all_results = []

    for lx, ly in PAIRS:
        info_x = config.MATRIX_MODELS[lx]
        info_y = config.MATRIX_MODELS[ly]
        print(f"\n{'='*60}")
        print(f"{lx}→{ly}: {info_x['family']}(dim={info_x['dim']}) → "
              f"{info_y['family']}(dim={info_y['dim']})")
        print(f"{'='*60}")

        for K in ANCHOR_COUNTS:
            anchor_idx = fps_cache[lx][:K]
            anc_x = cand_embs[lx][anchor_idx]
            anc_y = cand_embs[ly][anchor_idx]

            sm = compute_sim_mean(anc_x)

            row_base = {
                "query_model": lx,
                "db_model": ly,
                "dim_x": int(info_x["dim"]),
                "dim_y": int(info_y["dim"]),
                "K": K,
                "seed": SEED,
                "sim_mean": round(sm, 4),
            }

            # --- RAT (poly + adaptive z-score) ---
            rel_x = to_relative(query_embs[lx], anc_x, kernel="poly", degree=2, coef0=1.0)
            rel_y = to_relative(query_embs[ly], anc_y, kernel="poly", degree=2, coef0=1.0)
            if sm < 0.65:
                rel_x_z = normalize_zscore(rel_x)
                rel_y_z = normalize_zscore(rel_y)
                rat_metrics = evaluate_retrieval(rel_x_z, rel_y_z)
                zscore_applied = True
            else:
                rat_metrics = evaluate_retrieval(rel_x, rel_y)
                zscore_applied = False

            rat_row = {
                **row_base,
                "method": "RAT",
                "zscore_applied": zscore_applied,
                "alpha": None,
                **rat_metrics,
            }
            all_results.append(rat_row)

            # --- Ridge (best λ) ---
            best_ridge = None
            best_ridge_r1 = -1.0
            for alpha in RIDGE_ALPHAS:
                W = fit_ridge(anc_x, anc_y, alpha=alpha)
                aligned = transform_ridge(query_embs[lx], W)
                ridge_metrics = evaluate_retrieval(aligned, query_embs[ly])
                if ridge_metrics["recall_at_1"] > best_ridge_r1:
                    best_ridge_r1 = ridge_metrics["recall_at_1"]
                    best_ridge = {
                        **row_base,
                        "method": "Ridge",
                        "zscore_applied": None,
                        "alpha": alpha,
                        **ridge_metrics,
                    }
            all_results.append(best_ridge)

            print(f"  K={K:>4}: RAT R@1={rat_row['recall_at_1']*100:5.1f}%  "
                  f"Ridge R@1={best_ridge['recall_at_1']*100:5.1f}% (α={best_ridge['alpha']})")

    elapsed = time.time() - start_time

    # テーブル表示
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE: R@1 (%) by pair × K")
    print(f"{'='*90}")

    header = f"{'Pair':>8} {'Method':>6}"
    for K in ANCHOR_COUNTS:
        header += f"  K={K:>4}"
    print(header)
    print("-" * 90)

    for lx, ly in PAIRS:
        pair_label = f"{lx}→{ly}"
        for method in ["RAT", "Ridge"]:
            line = f"{pair_label:>8} {method:>6}"
            for K in ANCHOR_COUNTS:
                r = [x for x in all_results
                     if x["query_model"] == lx and x["db_model"] == ly
                     and x["K"] == K and x["method"] == method]
                if r:
                    line += f"  {r[0]['recall_at_1']*100:5.1f}"
                else:
                    line += f"     --"
            print(line)
        pair_label = ""  # don't repeat pair label for Ridge

    # K=500→1000→2000 の伸び分析
    print(f"\n{'='*90}")
    print("DELTA ANALYSIS: R@1 gain from K=500")
    print(f"{'='*90}")
    print(f"{'Pair':>8} {'Method':>6}  K500→K1000  K500→K2000")
    print("-" * 50)
    for lx, ly in PAIRS:
        pair_label = f"{lx}→{ly}"
        for method in ["RAT", "Ridge"]:
            r500 = [x for x in all_results
                    if x["query_model"] == lx and x["db_model"] == ly
                    and x["K"] == 500 and x["method"] == method]
            r1000 = [x for x in all_results
                     if x["query_model"] == lx and x["db_model"] == ly
                     and x["K"] == 1000 and x["method"] == method]
            r2000 = [x for x in all_results
                     if x["query_model"] == lx and x["db_model"] == ly
                     and x["K"] == 2000 and x["method"] == method]
            if r500 and r1000 and r2000:
                d1 = (r1000[0]["recall_at_1"] - r500[0]["recall_at_1"]) * 100
                d2 = (r2000[0]["recall_at_1"] - r500[0]["recall_at_1"]) * 100
                print(f"{pair_label:>8} {method:>6}  {d1:+6.1f}%p     {d2:+6.1f}%p")
            pair_label = ""

    # 結果保存
    out_dir = config.RESULTS_DIR / "d1_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "anchor_counts": ANCHOR_COUNTS,
            "ridge_alphas": RIDGE_ALPHAS,
            "seed": SEED,
            "kernel": "poly(degree=2, coef0=1.0)",
            "candidate_pool": n_cand,
            "num_queries": int(query_embs[needed_labels[0]].shape[0]),
            "fps_space": "query_model",
            "zscore": "adaptive (threshold=0.65)",
            "pairs": [f"{lx}→{ly}" for lx, ly in PAIRS],
        },
        "models": {
            label: {
                "name": config.MATRIX_MODELS[label]["name"],
                "family": config.MATRIX_MODELS[label]["family"],
                "dim": config.MATRIX_MODELS[label]["dim"],
            }
            for label in needed_labels
        },
        "results": all_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = out_dir / "d1_scaling_ceiling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {json_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
