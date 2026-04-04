"""
M (GTE-Qwen2) + N (Arctic) 追加検証実験

2 new models × 12 existing models = 24 directed pairs (+ M↔N = 26 total)
Methods: RAT (poly + adaptive z-score) のみ（D1比較は既存132ペアで確定済み）
K = [50, 100, 200, 500]
Seeds = [42, 123, 7]
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
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# 設定
# ========================================
ANCHOR_COUNTS = [50, 100, 200, 500]
SEEDS = [42, 123, 7]
DATA_DIR = config.DATA_DIR / "d2_matrix"

NEW_LABELS = ["M", "N"]
ALL_LABELS = sorted(config.MATRIX_MODELS.keys())


def compute_sim_mean(anchor_emb: np.ndarray) -> float:
    sim = cosine_similarity(anchor_emb)
    n = len(sim)
    return float((sim.sum() - n) / (n * (n - 1)))


def main():
    start_time = time.time()

    print("=" * 70)
    print("M+N Validation: 2 new models × all existing models")
    print("=" * 70)

    # Load embeddings
    cand_embs = {}
    query_embs = {}
    for label in ALL_LABELS:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
        info = config.MATRIX_MODELS[label]
        print(f"  {label}: cand={cand_embs[label].shape} ({info['family']}, dim={info['dim']})")

    n_cand = cand_embs["A"].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    # Build pair list: all pairs involving M or N
    pairs = []
    for new in NEW_LABELS:
        for other in ALL_LABELS:
            if new == other:
                continue
            pairs.append((new, other))
            pairs.append((other, new))
    # Add M↔N (already covered above, deduplicate)
    pairs = list(dict.fromkeys(pairs))
    print(f"\nTotal directed pairs: {len(pairs)}")

    # FPS index cache
    print(f"\nFPS index computation (max K={max(ANCHOR_COUNTS)})...")
    fps_cache = {}  # (label, seed) -> indices
    for label in ALL_LABELS:
        for seed in SEEDS:
            max_k = min(max(ANCHOR_COUNTS), n_cand)
            fps_idx, _ = select_anchors_fps(
                cand_embs[label], candidates_dummy, max_k, seed=seed,
            )
            fps_cache[(label, seed)] = np.array(fps_idx)

    # Run experiments
    all_results = []
    total = len(pairs) * len(ANCHOR_COUNTS) * len(SEEDS)
    count = 0

    for lx, ly in pairs:
        info_x = config.MATRIX_MODELS[lx]
        info_y = config.MATRIX_MODELS[ly]

        for seed in SEEDS:
            for K in ANCHOR_COUNTS:
                count += 1
                anchor_idx = fps_cache[(lx, seed)][:K]
                anc_x = cand_embs[lx][anchor_idx]
                anc_y = cand_embs[ly][anchor_idx]

                sm = compute_sim_mean(anc_x)

                # RAT (poly + adaptive z-score)
                rel_x = to_relative(query_embs[lx], anc_x, kernel="poly", degree=2, coef0=1.0)
                rel_y = to_relative(query_embs[ly], anc_y, kernel="poly", degree=2, coef0=1.0)
                if sm < 0.65:
                    rel_x_z = normalize_zscore(rel_x)
                    rel_y_z = normalize_zscore(rel_y)
                    metrics = evaluate_retrieval(rel_x_z, rel_y_z)
                    zscore_applied = True
                else:
                    metrics = evaluate_retrieval(rel_x, rel_y)
                    zscore_applied = False

                row = {
                    "query_model": lx,
                    "db_model": ly,
                    "family_x": info_x["family"],
                    "family_y": info_y["family"],
                    "dim_x": int(info_x["dim"]),
                    "dim_y": int(info_y["dim"]),
                    "K": K,
                    "seed": seed,
                    "method": "RAT",
                    "sim_mean": round(sm, 4),
                    "zscore_applied": zscore_applied,
                    **metrics,
                }
                all_results.append(row)

                if seed == 42 and K == 500:
                    print(f"  [{count}/{total}] {lx}→{ly} K={K}: "
                          f"R@1={metrics['recall_at_1']*100:5.1f}% sm={sm:.3f}")

    elapsed = time.time() - start_time

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print("SUMMARY: M and N pairs, seed=42, K=500")
    print(f"{'='*70}")

    primary = [r for r in all_results if r["seed"] == 42 and r["K"] == 500]

    # M as query
    print("\n--- M (GTE-Qwen2, 1536d) as FPS space ---")
    m_as_query = sorted([r for r in primary if r["query_model"] == "M"],
                        key=lambda r: r["recall_at_1"], reverse=True)
    for r in m_as_query:
        print(f"  M→{r['db_model']}: R@1={r['recall_at_1']*100:5.1f}% "
              f"(dim {r['dim_x']}→{r['dim_y']}, sm={r['sim_mean']:.3f})")

    # M as target
    print("\n--- M (GTE-Qwen2, 1536d) as target ---")
    m_as_target = sorted([r for r in primary if r["db_model"] == "M"],
                         key=lambda r: r["recall_at_1"], reverse=True)
    for r in m_as_target:
        print(f"  {r['query_model']}→M: R@1={r['recall_at_1']*100:5.1f}% "
              f"(dim {r['dim_x']}→{r['dim_y']}, sm={r['sim_mean']:.3f})")

    # N as query
    print("\n--- N (Arctic, 768d) as FPS space ---")
    n_as_query = sorted([r for r in primary if r["query_model"] == "N"],
                        key=lambda r: r["recall_at_1"], reverse=True)
    for r in n_as_query:
        print(f"  N→{r['db_model']}: R@1={r['recall_at_1']*100:5.1f}% "
              f"(dim {r['dim_x']}→{r['dim_y']}, sm={r['sim_mean']:.3f})")

    # N as target
    print("\n--- N (Arctic, 768d) as target ---")
    n_as_target = sorted([r for r in primary if r["db_model"] == "N"],
                         key=lambda r: r["recall_at_1"], reverse=True)
    for r in n_as_target:
        print(f"  {r['query_model']}→N: R@1={r['recall_at_1']*100:5.1f}% "
              f"(dim {r['dim_x']}→{r['dim_y']}, sm={r['sim_mean']:.3f})")

    # Bidirectional best for undirected pairs involving M or N
    print(f"\n--- RAT_auto (bidirectional best) K=500, seed=42 ---")
    for new in NEW_LABELS:
        for other in ALL_LABELS:
            if new == other:
                continue
            r_fwd = [r for r in primary if r["query_model"] == new and r["db_model"] == other]
            r_rev = [r for r in primary if r["query_model"] == other and r["db_model"] == new]
            if r_fwd and r_rev:
                best = max(r_fwd[0]["recall_at_1"], r_rev[0]["recall_at_1"])
                fwd_v = r_fwd[0]["recall_at_1"]
                rev_v = r_rev[0]["recall_at_1"]
                arrow = "→" if fwd_v >= rev_v else "←"
                print(f"  {new}↔{other}: RAT_auto={best*100:5.1f}% "
                      f"({new}→{other}={fwd_v*100:.1f}%, {other}→{new}={rev_v*100:.1f}%, best={arrow})")

    # Overall stats
    all_r1 = [r["recall_at_1"] for r in primary]
    m_query_r1 = [r["recall_at_1"] for r in primary if r["query_model"] == "M"]
    n_query_r1 = [r["recall_at_1"] for r in primary if r["query_model"] == "N"]

    print(f"\n--- Overall K=500 stats ---")
    print(f"  All new pairs: mean R@1 = {np.mean(all_r1)*100:.1f}%")
    print(f"  M as FPS space: mean R@1 = {np.mean(m_query_r1)*100:.1f}%")
    print(f"  N as FPS space: mean R@1 = {np.mean(n_query_r1)*100:.1f}%")

    # Save results
    out_dir = config.RESULTS_DIR / "mn_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "anchor_counts": ANCHOR_COUNTS,
            "seeds": SEEDS,
            "kernel": "poly(degree=2, coef0=1.0)",
            "candidate_pool": n_cand,
            "num_queries": int(query_embs["A"].shape[0]),
            "zscore": "adaptive (threshold=0.65)",
            "new_models": {
                label: {
                    "name": config.MATRIX_MODELS[label]["name"],
                    "family": config.MATRIX_MODELS[label]["family"],
                    "dim": config.MATRIX_MODELS[label]["dim"],
                    "params": config.MATRIX_MODELS[label]["params"],
                }
                for label in NEW_LABELS
            },
        },
        "results": all_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = out_dir / "mn_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {json_path}")
    print(f"レコード数: {len(all_results)}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
