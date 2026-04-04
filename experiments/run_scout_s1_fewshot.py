"""
Scout S1 Few-shot: Ridge回帰 vs Procrustes でクロスクラスター壁突破

核心の問い: 直交制約を外した線形写像（Ridge）なら、何ペアで壁を越えられるか？

設計根拠:
- S1でProcrustes（直交変換）は2000ペアでもクロスクラスター0%
  → 直交制約が問題。距離構造を保存する変換では、距離構造が無相関な空間を橋渡しできない
- Ridge（非制約線形写像）は空間を伸縮・剪断できる
  → 構造を保存するのではなく新たに学習する
- Procrustesは対照条件として残し、「制約を緩めた効果」を定量化する
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.evaluator import evaluate_retrieval

DATA_DIR = config.DATA_DIR / "d2_matrix"

# ============================================================
# 実験設定
# ============================================================

CROSS_CLUSTER_PAIRS = [
    ("A", "O", "MiniLM(384)→Arctic-xs(384)", "same"),
    ("C", "P", "BGE-s(384)→Arctic-s(384)", "same"),
    ("A", "N", "MiniLM(384)→Arctic-m(768)", "different"),
]

CONTROL_PAIRS = [
    ("A", "B", "MiniLM(384)→E5-large(1024)", "different"),
]

N_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]
NUM_TRIALS = 5
BASE_SEED = 42

# RidgeCV alphas (広い範囲で探索)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


# ============================================================
# データ読み込み
# ============================================================

def load_embeddings(labels: list[str]) -> tuple[dict, dict]:
    cand_embs, query_embs = {}, {}
    for label in labels:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
    return cand_embs, query_embs


# ============================================================
# Few-shot Ridge回帰
# ============================================================

def fewshot_ridge(
    query_src: np.ndarray, query_tgt: np.ndarray,
    train_indices: np.ndarray,
) -> dict:
    """
    Few-shot Ridge回帰: ラベルペアからW (D_src, D_tgt) を学習。
    次元不一致でもOK。RidgeCVで正則化係数を自動選定。
    """
    X_train = query_src[train_indices]
    Y_train = query_tgt[train_indices]

    # n_shots < n_features の場合、CVの分割数を調整
    n_cv = min(5, len(train_indices))
    if n_cv < 2:
        # 1-shotではCVできない → 最大の正則化を使用
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=10000.0, fit_intercept=False)
        model.fit(X_train, Y_train)
    else:
        model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False, cv=n_cv)
        model.fit(X_train, Y_train)

    # 全クエリを変換
    query_transformed = query_src @ model.coef_.T  # (N, D_tgt)

    # L2正規化（コサイン類似度検索のため）
    norms = np.linalg.norm(query_transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_transformed = query_transformed / norms

    metrics = evaluate_retrieval(query_transformed, query_tgt)

    alpha_used = model.alpha_ if hasattr(model, 'alpha_') else 10000.0
    metrics["alpha"] = float(alpha_used)
    return metrics


# ============================================================
# Few-shot Procrustes (対照条件)
# ============================================================

def fewshot_procrustes_same_dim(
    query_src: np.ndarray, query_tgt: np.ndarray,
    train_indices: np.ndarray,
) -> dict:
    """同次元ペア用Procrustes。"""
    X_train = query_src[train_indices]
    Y_train = query_tgt[train_indices]

    W, _ = orthogonal_procrustes(X_train, Y_train)
    query_transformed = query_src @ W

    return evaluate_retrieval(query_transformed, query_tgt)


def fewshot_procrustes_zeropad(
    query_src: np.ndarray, query_tgt: np.ndarray,
    train_indices: np.ndarray,
) -> dict:
    """異次元ペア用Procrustes（ゼロパディング）。"""
    dim_src = query_src.shape[1]
    dim_tgt = query_tgt.shape[1]
    dim_max = max(dim_src, dim_tgt)

    def pad(arr, target_dim):
        if arr.shape[1] == target_dim:
            return arr
        padded = np.zeros((arr.shape[0], target_dim))
        padded[:, :arr.shape[1]] = arr
        return padded

    X_pad = pad(query_src[train_indices], dim_max)
    Y_pad = pad(query_tgt[train_indices], dim_max)

    W, _ = orthogonal_procrustes(X_pad, Y_pad)
    query_transformed = pad(query_src, dim_max) @ W
    query_transformed = query_transformed[:, :dim_tgt]

    return evaluate_retrieval(query_transformed, query_tgt)


# ============================================================
# 集約ユーティリティ
# ============================================================

def aggregate_trials(trial_results: list[dict]) -> dict:
    keys = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    agg = {}
    for key in keys:
        vals = [r[key] for r in trial_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    agg["median_rank_mean"] = float(np.mean([r["median_rank"] for r in trial_results]))
    if "alpha" in trial_results[0]:
        agg["alpha_mean"] = float(np.mean([r["alpha"] for r in trial_results]))
    agg["n_trials"] = len(trial_results)
    return agg


# ============================================================
# メイン
# ============================================================

def run_experiment():
    print("=" * 60)
    print("Scout S1 Few-shot: Ridge vs Procrustes")
    print("=" * 60)
    t_start = time.time()

    all_labels = set()
    for src, tgt, _, _ in CROSS_CLUSTER_PAIRS + CONTROL_PAIRS:
        all_labels.update([src, tgt])

    print(f"対象モデル: {sorted(all_labels)}")
    cand_embs, query_embs = load_embeddings(sorted(all_labels))
    for label in sorted(all_labels):
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    n_queries = query_embs[list(query_embs.keys())[0]].shape[0]

    results = []

    all_pairs = [
        *[(s, t, d, dm, "cross_cluster") for s, t, d, dm in CROSS_CLUSTER_PAIRS],
        *[(s, t, d, dm, "control") for s, t, d, dm in CONTROL_PAIRS],
    ]

    for src, tgt, desc, dim_match, pair_type in all_pairs:
        print(f"\n{'='*60}")
        print(f"ペア: {src}→{tgt} ({desc}) [{pair_type}]")
        print(f"  dim: {query_embs[src].shape[1]} → {query_embs[tgt].shape[1]}")
        print(f"{'='*60}")

        qs = query_embs[src]
        qt = query_embs[tgt]

        for n_shots in N_SHOTS:
            if n_shots >= n_queries:
                continue

            ridge_trials = []
            proc_trials = []

            for trial in range(NUM_TRIALS):
                rng = np.random.RandomState(BASE_SEED + trial)
                train_idx = rng.choice(n_queries, size=n_shots, replace=False)

                # Ridge
                m_ridge = fewshot_ridge(qs, qt, train_idx)
                ridge_trials.append(m_ridge)

                # Procrustes
                if dim_match == "same":
                    m_proc = fewshot_procrustes_same_dim(qs, qt, train_idx)
                else:
                    m_proc = fewshot_procrustes_zeropad(qs, qt, train_idx)
                proc_trials.append(m_proc)

            # Ridge集約
            agg_ridge = aggregate_trials(ridge_trials)
            results.append({
                "pair": f"{src}→{tgt}", "pair_type": pair_type,
                "method": "ridge", "n_shots": n_shots, **agg_ridge,
            })

            # Procrustes集約
            agg_proc = aggregate_trials(proc_trials)
            proc_variant = "procrustes_same" if dim_match == "same" else "procrustes_zeropad"
            results.append({
                "pair": f"{src}→{tgt}", "pair_type": pair_type,
                "method": proc_variant, "n_shots": n_shots, **agg_proc,
            })

            alpha_str = f" α={agg_ridge.get('alpha_mean', 0):.0f}" if "alpha_mean" in agg_ridge else ""
            print(f"  n={n_shots:>3d}: Ridge R@1={agg_ridge['recall_at_1_mean']*100:5.1f}%"
                  f"±{agg_ridge['recall_at_1_std']*100:.1f}{alpha_str}"
                  f"  |  Proc R@1={agg_proc['recall_at_1_mean']*100:5.1f}%"
                  f"±{agg_proc['recall_at_1_std']*100:.1f}")

    elapsed = time.time() - t_start

    # 結果保存
    output = {
        "experiment": "scout_s1_fewshot",
        "description": "Few-shot Ridge vs Procrustes: クロスクラスター壁突破",
        "config": {
            "n_shots": N_SHOTS,
            "num_trials": NUM_TRIALS,
            "base_seed": BASE_SEED,
            "ridge_alphas": RIDGE_ALPHAS,
            "n_queries": n_queries,
            "cross_cluster_pairs": [f"{s}→{t}" for s, t, _, _ in CROSS_CLUSTER_PAIRS],
            "control_pairs": [f"{s}→{t}" for s, t, _, _ in CONTROL_PAIRS],
        },
        "results": results,
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "scout_s1_fewshot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ============================================
    # サマリーテーブル
    # ============================================
    print(f"\n{'='*70}")
    print("サマリー: R@1 (%) — Ridge vs Procrustes")
    print(f"{'='*70}")

    for src, tgt, desc, _, pair_type in all_pairs:
        pair_key = f"{src}→{tgt}"
        pair_results = [r for r in results if r["pair"] == pair_key]

        print(f"\n  {pair_key} ({desc}) [{pair_type}]")
        print(f"  {'n_shots':>7s}  {'Ridge R@1':>12s}  {'Proc R@1':>12s}  {'Delta':>8s}")
        print(f"  {'-'*45}")

        for n in N_SHOTS:
            ridge_r = next((r for r in pair_results if r["method"] == "ridge" and r["n_shots"] == n), None)
            proc_r = next((r for r in pair_results
                          if r["method"].startswith("procrustes") and r["n_shots"] == n), None)
            if ridge_r and proc_r:
                r_r1 = ridge_r["recall_at_1_mean"] * 100
                p_r1 = proc_r["recall_at_1_mean"] * 100
                delta = r_r1 - p_r1
                print(f"  {n:>7d}  {r_r1:>5.1f}%±{ridge_r['recall_at_1_std']*100:.1f}"
                      f"     {p_r1:>5.1f}%±{proc_r['recall_at_1_std']*100:.1f}"
                      f"     {delta:>+5.1f}%")

    print(f"\n完了! {elapsed:.1f}秒")
    print(f"結果: {out_path}")


if __name__ == "__main__":
    run_experiment()
