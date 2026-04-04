"""
S2: Few-shot クロスクラスター壁突破

目的: ゼロショットでは不可能なクロスクラスター翻訳が、最小限の教師信号で
可能になるかを検証。「構造保存」(Procrustes) vs「構造学習」(Ridge) を定量化。

設計根拠:
- S1: ゼロショットではProcrustes/RAT/RBF全滅（R@1≈0.4%, ランダム水準）
- S1診断: ローカル近傍もクラスタ構造も完全に無相関
- 仮説: 構造が存在しないなら「保存」ではなく「学習」が必要
  → Ridge（非制約線形写像）vs Procrustes（直交制約）の対比

重要な実装ポイント:
- shot_indicesは評価から除外（情報リーク防止）
- RidgeCVでα自動選定
- 変換後はL2正規化してコサイン類似度検索
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# sklearn の R^2 warning を抑制（n_shots=1でのCV時）
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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
NUM_TRIALS = 3
BASE_SEED = 42

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


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
# 評価（shot除外版）
# ============================================================

def evaluate_retrieval_exclude(
    query_transformed: np.ndarray,
    query_target: np.ndarray,
    eval_mask: np.ndarray,
) -> dict:
    """
    shot_indicesを除外して検索精度を評価。

    query_transformed[i] と query_target[i] が同じテキストに対応。
    eval_maskがTrueのインデックスのみ評価対象。

    検索空間: eval_mask内の query_target のみ（学習ペアは検索候補からも除外）
    """
    eval_indices = np.where(eval_mask)[0]
    n_eval = len(eval_indices)

    # 評価対象のみ抽出
    qt_eval = query_transformed[eval_indices]  # (n_eval, D)
    tt_eval = query_target[eval_indices]        # (n_eval, D)

    # コサイン類似度行列
    sim_matrix = cosine_similarity(qt_eval, tt_eval)  # (n_eval, n_eval)

    # 各クエリの正解ランクを計算
    # qt_eval[i] の正解は tt_eval[i]（対角要素）
    ranks = []
    for i in range(n_eval):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": int(np.median(ranks)),
        "n_eval": n_eval,
    }


# ============================================================
# Few-shot Ridge
# ============================================================

def fewshot_ridge(
    query_src: np.ndarray,
    query_tgt: np.ndarray,
    shot_indices: np.ndarray,
) -> tuple[dict, float]:
    """
    Few-shot Ridge回帰。
    Returns: (metrics, alpha_used)
    """
    X_train = query_src[shot_indices]
    Y_train = query_tgt[shot_indices]
    n_shots = len(shot_indices)

    # α選定
    if n_shots < 3:
        # CVできない → 最大αで強い正則化
        model = Ridge(alpha=1000.0, fit_intercept=False)
        model.fit(X_train, Y_train)
        alpha_used = 1000.0
    else:
        n_cv = min(5, n_shots)
        model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False, cv=n_cv)
        model.fit(X_train, Y_train)
        alpha_used = float(model.alpha_)

    # 全クエリを変換
    query_transformed = query_src @ model.coef_.T

    # L2正規化
    norms = np.linalg.norm(query_transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_transformed = query_transformed / norms

    # L2正規化（ターゲットも念のため）
    tgt_norms = np.linalg.norm(query_tgt, axis=1, keepdims=True)
    tgt_norms[tgt_norms == 0] = 1.0
    query_tgt_normed = query_tgt / tgt_norms

    # 評価（shot除外）
    eval_mask = np.ones(len(query_src), dtype=bool)
    eval_mask[shot_indices] = False
    metrics = evaluate_retrieval_exclude(query_transformed, query_tgt_normed, eval_mask)

    return metrics, alpha_used


# ============================================================
# Few-shot Procrustes
# ============================================================

def fewshot_procrustes(
    query_src: np.ndarray,
    query_tgt: np.ndarray,
    shot_indices: np.ndarray,
    dim_match: str,
) -> dict:
    """
    Few-shot Procrustes。同次元は直接、異次元はゼロパディング。
    """
    X_train = query_src[shot_indices]
    Y_train = query_tgt[shot_indices]

    dim_src = query_src.shape[1]
    dim_tgt = query_tgt.shape[1]

    if dim_match == "same":
        W, _ = orthogonal_procrustes(X_train, Y_train)
        query_transformed = query_src @ W
    else:
        dim_max = max(dim_src, dim_tgt)

        def pad(arr, target_dim):
            if arr.shape[1] == target_dim:
                return arr
            padded = np.zeros((arr.shape[0], target_dim))
            padded[:, :arr.shape[1]] = arr
            return padded

        X_pad = pad(X_train, dim_max)
        Y_pad = pad(Y_train, dim_max)
        W, _ = orthogonal_procrustes(X_pad, Y_pad)
        query_transformed = pad(query_src, dim_max) @ W
        query_transformed = query_transformed[:, :dim_tgt]

    # L2正規化
    norms = np.linalg.norm(query_transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_transformed = query_transformed / norms

    tgt_norms = np.linalg.norm(query_tgt, axis=1, keepdims=True)
    tgt_norms[tgt_norms == 0] = 1.0
    query_tgt_normed = query_tgt / tgt_norms

    # 評価（shot除外）
    eval_mask = np.ones(len(query_src), dtype=bool)
    eval_mask[shot_indices] = False
    return evaluate_retrieval_exclude(query_transformed, query_tgt_normed, eval_mask)


# ============================================================
# 集約
# ============================================================

def aggregate_trials(trial_results: list[dict]) -> dict:
    keys = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    agg = {}
    for key in keys:
        vals = [r[key] for r in trial_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    agg["median_rank_mean"] = float(np.mean([r["median_rank"] for r in trial_results]))
    agg["n_eval"] = trial_results[0]["n_eval"]
    agg["n_trials"] = len(trial_results)
    return agg


# ============================================================
# メイン
# ============================================================

def run_experiment():
    print("=" * 60)
    print("S2: Few-shot クロスクラスター壁突破")
    print("  Ridge (構造学習) vs Procrustes (構造保存)")
    print("=" * 60)
    t_start = time.time()

    all_labels = set()
    for src, tgt, _, _ in CROSS_CLUSTER_PAIRS + CONTROL_PAIRS:
        all_labels.update([src, tgt])

    print(f"\n対象モデル: {sorted(all_labels)}")
    cand_embs, query_embs = load_embeddings(sorted(all_labels))
    for label in sorted(all_labels):
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    n_queries = query_embs[list(query_embs.keys())[0]].shape[0]
    print(f"クエリ数: {n_queries}")

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
            alpha_list = []

            for trial in range(NUM_TRIALS):
                rng = np.random.RandomState(BASE_SEED + trial)
                shot_idx = rng.choice(n_queries, size=n_shots, replace=False)

                # Ridge
                m_ridge, alpha_used = fewshot_ridge(qs, qt, shot_idx)
                ridge_trials.append(m_ridge)
                alpha_list.append(alpha_used)

                # Procrustes
                m_proc = fewshot_procrustes(qs, qt, shot_idx, dim_match)
                proc_trials.append(m_proc)

            # Ridge集約
            agg_ridge = aggregate_trials(ridge_trials)
            agg_ridge["alpha_mean"] = float(np.mean(alpha_list))
            agg_ridge["alpha_values"] = alpha_list
            results.append({
                "pair": f"{src}→{tgt}", "pair_type": pair_type,
                "method": "ridge", "n_shots": n_shots, **agg_ridge,
            })

            # Procrustes集約
            agg_proc = aggregate_trials(proc_trials)
            proc_name = "procrustes" if dim_match == "same" else "procrustes_zeropad"
            results.append({
                "pair": f"{src}→{tgt}", "pair_type": pair_type,
                "method": proc_name, "n_shots": n_shots, **agg_proc,
            })

            r_r1 = agg_ridge["recall_at_1_mean"] * 100
            p_r1 = agg_proc["recall_at_1_mean"] * 100
            alpha_str = f"α={agg_ridge['alpha_mean']:.0f}"
            print(f"  n={n_shots:>3d} (eval={agg_ridge['n_eval']:>3d}): "
                  f"Ridge {r_r1:>5.1f}%±{agg_ridge['recall_at_1_std']*100:.1f} ({alpha_str})  "
                  f"Proc {p_r1:>5.1f}%±{agg_proc['recall_at_1_std']*100:.1f}  "
                  f"Δ={r_r1-p_r1:+.1f}%")

    elapsed = time.time() - t_start

    # ============================================
    # ランダムベースライン（各n_eval条件で）
    # ============================================
    random_baselines = {}
    for n_shots in N_SHOTS:
        n_eval = n_queries - n_shots
        random_baselines[n_shots] = {
            "recall_at_1": 1.0 / n_eval,
            "recall_at_5": 5.0 / n_eval,
            "recall_at_10": 10.0 / n_eval,
            "n_eval": n_eval,
        }

    # ============================================
    # 結果保存
    # ============================================
    output = {
        "experiment": "scout_s2_fewshot",
        "description": "Few-shot Ridge vs Procrustes: クロスクラスター壁突破",
        "design_rationale": {
            "s1_finding": "ゼロショットでは全手法がクロスクラスターで全滅 (R@1≈0.4%)",
            "s1_diag_finding": "ローカル近傍もクラスタ構造も完全に無相関",
            "hypothesis": "構造保存(Procrustes)ではなく構造学習(Ridge)が必要",
            "key_question": "壁を越えるのに最低何ペア必要か",
        },
        "config": {
            "n_shots": N_SHOTS,
            "num_trials": NUM_TRIALS,
            "base_seed": BASE_SEED,
            "ridge_alphas": RIDGE_ALPHAS,
            "n_queries": n_queries,
            "shot_excluded_from_eval": True,
            "cross_cluster_pairs": [f"{s}→{t}" for s, t, _, _ in CROSS_CLUSTER_PAIRS],
            "control_pairs": [f"{s}→{t}" for s, t, _, _ in CONTROL_PAIRS],
        },
        "random_baselines": random_baselines,
        "results": results,
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "scout_s2_fewshot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ============================================
    # サマリーテーブル
    # ============================================
    print(f"\n{'='*80}")
    print("S2 サマリー: R@1 (%) — Ridge vs Procrustes (shot除外評価)")
    print(f"{'='*80}")

    for src, tgt, desc, _, pair_type in all_pairs:
        pair_key = f"{src}→{tgt}"
        pair_results = [r for r in results if r["pair"] == pair_key]

        print(f"\n  {pair_key} ({desc}) [{pair_type}]")
        print(f"  {'n':>5s} {'eval':>5s}  {'Ridge R@1':>14s}  {'Proc R@1':>14s}  {'Δ':>7s}  {'random':>7s}")
        print(f"  {'-'*60}")

        for n in N_SHOTS:
            ridge_r = next((r for r in pair_results if r["method"] == "ridge" and r["n_shots"] == n), None)
            proc_r = next((r for r in pair_results if r["method"].startswith("procrustes") and r["n_shots"] == n), None)
            if ridge_r and proc_r:
                r_r1 = ridge_r["recall_at_1_mean"] * 100
                p_r1 = proc_r["recall_at_1_mean"] * 100
                delta = r_r1 - p_r1
                rand = random_baselines[n]["recall_at_1"] * 100
                n_eval = ridge_r["n_eval"]
                print(f"  {n:>5d} {n_eval:>5d}  "
                      f"{r_r1:>5.1f}%±{ridge_r['recall_at_1_std']*100:>4.1f}  "
                      f"{p_r1:>5.1f}%±{proc_r['recall_at_1_std']*100:>4.1f}  "
                      f"{delta:>+5.1f}%  "
                      f"{rand:>5.2f}%")

    # ============================================
    # R@5/R@10 詳細（クロスクラスターの最良ペア）
    # ============================================
    print(f"\n{'='*80}")
    print("詳細: A→O (代表クロスクラスター) 全指標")
    print(f"{'='*80}")
    ao_results = [r for r in results if r["pair"] == "A→O"]
    print(f"  {'n':>5s} {'method':>12s}  {'R@1':>8s}  {'R@5':>8s}  {'R@10':>8s}  {'MRR':>8s}  {'MedR':>5s}")
    print(f"  {'-'*65}")
    for n in N_SHOTS:
        for method_prefix in ["ridge", "procrustes"]:
            r = next((r for r in ao_results
                      if r["method"].startswith(method_prefix) and r["n_shots"] == n), None)
            if r:
                m_label = "Ridge" if method_prefix == "ridge" else "Proc"
                print(f"  {n:>5d} {m_label:>12s}  "
                      f"{r['recall_at_1_mean']*100:>6.1f}%  "
                      f"{r['recall_at_5_mean']*100:>6.1f}%  "
                      f"{r['recall_at_10_mean']*100:>6.1f}%  "
                      f"{r['mrr_mean']:>6.4f}  "
                      f"{r['median_rank_mean']:>5.0f}")

    print(f"\n完了! {len(results)}条件, {elapsed:.1f}秒")
    print(f"結果: {out_path}")


if __name__ == "__main__":
    run_experiment()
