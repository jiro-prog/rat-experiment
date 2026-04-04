"""
Experiment D1: RAT vs 直接アライメント比較

目的: 「なぜ単純な線形写像じゃダメなのか？」に定量的に答える。
     RATが勝つ条件と線形手法が勝つ条件を、アンカー数・モデルペア属性で切り分ける。

比較手法:
  1. RAT (FPS + poly kernel + per-vector z-score)
  2. Orthogonal Procrustes (同次元ペアのみ)
  3. Ridge Regression (λグリッドサーチ、ペアごとbest)
  4. Affine (Ridge + bias、ペアごとbest)

実験グリッド:
  - 12×11 = 132有向ペア（全量）
  - K ∈ {10, 25, 50, 100, 200, 500} アンカー数
  - 3 seeds (FPS初期点) で分散確認
  - 評価: Recall@1, @5, @10 (N=500クエリ)

出力:
  - d1_results.json (全データ)
  - d1_results.csv (ペア×K×手法のフラットテーブル)

データ: data/d2_matrix/ のキャッシュ済み埋め込みを再利用。
"""
import csv
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval
from src.baselines import (
    fit_procrustes, transform_procrustes,
    fit_ridge, transform_ridge,
    fit_affine, transform_affine,
)

# ========================================
# 設定
# ========================================
ANCHOR_COUNTS = [10, 25, 50, 100, 200, 500]
RIDGE_ALPHAS = [1e-4, 1e-2, 1.0, 100.0]
SEEDS = [42, 123, 7]
DATA_DIR = config.DATA_DIR / "d2_matrix"
MAX_FPS_K = 500


def compute_sim_mean(anchor_emb: np.ndarray) -> float:
    """アンカー間の平均コサイン類似度。"""
    sim = cosine_similarity(anchor_emb)
    n = len(sim)
    return float((sim.sum() - n) / (n * (n - 1)))


def evaluate_pair_all_methods(
    label_x: str,
    label_y: str,
    cand_x: np.ndarray,
    cand_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    fps_indices: np.ndarray,
    seed: int,
) -> list[dict]:
    """1ペア×1シードの全手法×全アンカー数を評価する。"""
    dim_x = cand_x.shape[1]
    dim_y = cand_y.shape[1]
    same_dim = (dim_x == dim_y)
    results = []

    for K in ANCHOR_COUNTS:
        anchor_idx = fps_indices[:K]
        anc_x = cand_x[anchor_idx]
        anc_y = cand_y[anchor_idx]

        sm = compute_sim_mean(anc_x)

        row_base = {
            "query_model": label_x,
            "db_model": label_y,
            "dim_x": dim_x,
            "dim_y": dim_y,
            "same_dim": same_dim,
            "K": K,
            "seed": seed,
            "sim_mean": sm,
        }

        # --- RAT (poly + adaptive z-score) ---
        rel_x = to_relative(query_x, anc_x, kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_y, anc_y, kernel="poly", degree=2, coef0=1.0)
        if sm < 0.65:
            rel_x_z = normalize_zscore(rel_x)
            rel_y_z = normalize_zscore(rel_y)
            rat_metrics = evaluate_retrieval(rel_x_z, rel_y_z)
            zscore_applied = True
        else:
            rat_metrics = evaluate_retrieval(rel_x, rel_y)
            zscore_applied = False

        results.append({
            **row_base,
            "method": "RAT",
            "zscore_applied": zscore_applied,
            "alpha": None,
            **rat_metrics,
        })

        # --- Orthogonal Procrustes (同次元のみ) ---
        if same_dim:
            W = fit_procrustes(anc_x, anc_y)
            aligned = transform_procrustes(query_x, W)
            proc_metrics = evaluate_retrieval(aligned, query_y)
            results.append({
                **row_base,
                "method": "Procrustes",
                "zscore_applied": None,
                "alpha": None,
                **proc_metrics,
            })

        # --- Ridge (best λ) ---
        best_ridge = None
        best_ridge_r1 = -1.0
        for alpha in RIDGE_ALPHAS:
            W = fit_ridge(anc_x, anc_y, alpha=alpha)
            aligned = transform_ridge(query_x, W)
            ridge_metrics = evaluate_retrieval(aligned, query_y)
            if ridge_metrics["recall_at_1"] > best_ridge_r1:
                best_ridge_r1 = ridge_metrics["recall_at_1"]
                best_ridge = {
                    **row_base,
                    "method": "Ridge",
                    "zscore_applied": None,
                    "alpha": alpha,
                    **ridge_metrics,
                }
        results.append(best_ridge)

        # --- Affine (best λ) ---
        best_affine = None
        best_affine_r1 = -1.0
        for alpha in RIDGE_ALPHAS:
            W, bias = fit_affine(anc_x, anc_y, alpha=alpha)
            aligned = transform_affine(query_x, W, bias)
            affine_metrics = evaluate_retrieval(aligned, query_y)
            if affine_metrics["recall_at_1"] > best_affine_r1:
                best_affine_r1 = affine_metrics["recall_at_1"]
                best_affine = {
                    **row_base,
                    "method": "Affine",
                    "zscore_applied": None,
                    "alpha": alpha,
                    **affine_metrics,
                }
        results.append(best_affine)

    return results


def save_csv(all_results: list[dict], path: Path):
    """全結果をCSVに保存。"""
    fields = [
        "query_model", "db_model", "dim_x", "dim_y", "same_dim",
        "K", "seed", "sim_mean", "method", "zscore_applied", "alpha",
        "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)


def print_summary(all_results: list[dict], available: list[str]):
    """サマリー統計を出力。"""
    # seed=42の結果でサマリー（代表seed）
    primary = [r for r in all_results if r["seed"] == SEEDS[0]]

    print(f"\n{'='*70}")
    print(f"サマリー: K=500, seed={SEEDS[0]} でのR@1平均（手法別）")
    print(f"{'='*70}")
    k500 = [r for r in primary if r["K"] == 500]
    for method in ["RAT", "Procrustes", "Ridge", "Affine"]:
        mrs = [r["recall_at_1"] for r in k500 if r["method"] == method]
        if mrs:
            print(f"  {method:>10}: mean={np.mean(mrs)*100:.1f}%, "
                  f"median={np.median(mrs)*100:.1f}%, "
                  f"min={np.min(mrs)*100:.1f}%, max={np.max(mrs)*100:.1f}% "
                  f"(N={len(mrs)})")

    # seed間分散（K=500）
    print(f"\n{'='*70}")
    print("Seed間分散: K=500でのR@1 std（手法別、全ペア平均）")
    print(f"{'='*70}")
    k500_all = [r for r in all_results if r["K"] == 500]
    for method in ["RAT", "Procrustes", "Ridge", "Affine"]:
        stds = []
        pairs = set()
        for r in k500_all:
            if r["method"] == method:
                pairs.add((r["query_model"], r["db_model"]))
        for qm, dm in pairs:
            vals = [
                r["recall_at_1"] for r in k500_all
                if r["method"] == method and r["query_model"] == qm and r["db_model"] == dm
            ]
            if len(vals) == len(SEEDS):
                stds.append(np.std(vals))
        if stds:
            print(f"  {method:>10}: mean_std={np.mean(stds)*100:.2f}%p, "
                  f"max_std={np.max(stds)*100:.2f}%p")

    # RAT vs Best Linear（K別の勝率）
    print(f"\n{'='*70}")
    print("RAT勝率 vs Best Linear（seed=42）")
    print(f"{'='*70}")
    for K in ANCHOR_COUNTS:
        k_results = [r for r in primary if r["K"] == K]
        rat_wins = 0
        total = 0
        rat_advantages = []
        for r in k_results:
            if r["method"] != "RAT":
                continue
            qm, dm = r["query_model"], r["db_model"]
            linear = [
                lr for lr in k_results
                if lr["query_model"] == qm and lr["db_model"] == dm
                and lr["method"] in ("Procrustes", "Ridge", "Affine")
            ]
            if linear:
                best_lin = max(lr["recall_at_1"] for lr in linear)
                delta = r["recall_at_1"] - best_lin
                rat_advantages.append(delta)
                if delta > 0:
                    rat_wins += 1
                total += 1
        if total:
            print(f"  K={K:>3}: RAT wins {rat_wins}/{total} "
                  f"({rat_wins/total*100:.0f}%), "
                  f"mean Δ={np.mean(rat_advantages)*100:+.1f}%p")

    # 方向非対称性（Procrustes）
    print(f"\n{'='*70}")
    print("方向非対称性: |R@1(X→Y) - R@1(Y→X)| (K=500, seed=42, Procrustes vs RAT)")
    print(f"{'='*70}")
    for method in ["RAT", "Procrustes"]:
        asymmetries = []
        mrs = [r for r in primary if r["K"] == 500 and r["method"] == method]
        pairs_seen = set()
        for r in mrs:
            qm, dm = r["query_model"], r["db_model"]
            if (dm, qm) in pairs_seen:
                continue
            reverse = [
                rr for rr in mrs
                if rr["query_model"] == dm and rr["db_model"] == qm
            ]
            if reverse:
                asym = abs(r["recall_at_1"] - reverse[0]["recall_at_1"])
                asymmetries.append(asym)
                pairs_seen.add((qm, dm))
        if asymmetries:
            print(f"  {method:>10}: mean={np.mean(asymmetries)*100:.1f}%p, "
                  f"max={np.max(asymmetries)*100:.1f}%p, "
                  f"N={len(asymmetries)} undirected pairs")


def main():
    start_time = time.time()

    print("=" * 70)
    print("Experiment D1: RAT vs Direct Alignment Comparison")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    # メタデータ読み込み
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("ERROR: まず run_d2a_embed_all.py を実行してください")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    available = meta["completed"]
    print(f"利用可能モデル: {len(available)} — {', '.join(available)}")
    print(f"アンカー数: {ANCHOR_COUNTS}")
    print(f"Ridgeα: {RIDGE_ALPHAS}")

    # embedding読み込み
    cand_embs = {}
    query_embs = {}
    for label in available:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
        info = config.MATRIX_MODELS[label]
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape} "
              f"({info['family']}, dim={info['dim']})")

    n_cand = cand_embs[available[0]].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    all_results = []
    n_pairs = len(available) * (len(available) - 1)

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"Seed {seed_idx+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*70}")

        # FPSインデックス事前計算（seedごとに異なる）
        fps_cache = {}
        for label in available:
            fps_idx, _ = select_anchors_fps(
                cand_embs[label], candidates_dummy, MAX_FPS_K, seed=seed,
            )
            fps_cache[label] = np.array(fps_idx)

        done = 0
        for lx in available:
            for ly in available:
                if lx == ly:
                    continue
                done += 1
                info_x = config.MATRIX_MODELS[lx]
                info_y = config.MATRIX_MODELS[ly]

                if seed_idx == 0:  # 最初のseedだけ詳細表示
                    print(f"\n[{done}/{n_pairs}] {lx}→{ly}: "
                          f"{info_x['family']}(dim={info_x['dim']}) → "
                          f"{info_y['family']}(dim={info_y['dim']})")

                pair_results = evaluate_pair_all_methods(
                    lx, ly,
                    cand_embs[lx], cand_embs[ly],
                    query_embs[lx], query_embs[ly],
                    fps_cache[lx],
                    seed=seed,
                )
                all_results.extend(pair_results)

                if seed_idx == 0:
                    k500 = [r for r in pair_results if r["K"] == 500]
                    for r in k500:
                        alpha_str = f"  α={r['alpha']}" if r['alpha'] is not None else ""
                        print(f"  {r['method']:>10}: R@1={r['recall_at_1']*100:5.1f}%  "
                              f"R@5={r['recall_at_5']*100:5.1f}%  "
                              f"MRR={r['mrr']:.3f}{alpha_str}")

        elapsed_seed = time.time() - start_time
        print(f"\nSeed {seed} 完了 (累計 {elapsed_seed:.0f}s)")

    # 結果保存
    elapsed = time.time() - start_time
    out_dir = config.RESULTS_DIR / "d1_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    output = {
        "config": {
            "anchor_counts": ANCHOR_COUNTS,
            "ridge_alphas": RIDGE_ALPHAS,
            "seeds": SEEDS,
            "kernel": "poly(degree=2, coef0=1.0)",
            "candidate_pool": n_cand,
            "num_queries": query_embs[available[0]].shape[0],
            "fps_space": "query_model",
            "zscore": "adaptive (threshold=0.65)",
        },
        "models": {
            label: {
                "name": config.MATRIX_MODELS[label]["name"],
                "family": config.MATRIX_MODELS[label]["family"],
                "params": config.MATRIX_MODELS[label]["params"],
                "dim": config.MATRIX_MODELS[label]["dim"],
            }
            for label in available
        },
        "results": all_results,
        "elapsed_seconds": elapsed,
    }

    json_path = out_dir / "d1_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = out_dir / "d1_results.csv"
    save_csv(all_results, csv_path)

    print(f"\n{'='*70}")
    print(f"結果保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"総レコード数: {len(all_results)}")
    print(f"実行時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")

    # サマリー統計
    print_summary(all_results, available)


if __name__ == "__main__":
    main()
