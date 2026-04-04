"""
Scout Experiment S1: ゼロショット・クロスクラスター壁突破の探索

目的: クロスクラスター間（RDM ρ ≈ 0）で追加学習なしに検索精度を改善できるか検証
手法: Orthogonal Procrustes (1a/1b/1c), RBF kernel RAT, 高K RAT
判定: Go/No-go基準に基づく最小コスト実験
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

# ============================================================
# 実験設定
# ============================================================

DATA_DIR = config.DATA_DIR / "d2_matrix"

# クロスクラスターペア（Cluster1 → Cluster2）
CROSS_CLUSTER_PAIRS = [
    # (src_label, tgt_label, description, dim_match)
    ("A", "O", "MiniLM(384)→Arctic-xs(384)", "same"),     # 1a: 同次元
    ("C", "P", "BGE-s(384)→Arctic-s(384)", "same"),       # 1a: 同次元
    ("A", "N", "MiniLM(384)→Arctic-m(768)", "different"),  # 1b/1c: 異次元
]

# 同クラスター制御ペア
CONTROL_PAIRS = [
    ("A", "B", "MiniLM(384)→E5-large(1024)", "different"),
]

# アンカー数条件
ANCHOR_COUNTS = [50, 100, 200, 500, 1000, 2000]

# RBFカーネルgamma条件
RBF_GAMMAS = ["auto", 0.1, 0.01]

# 試行数
NUM_TRIALS = 3
BASE_SEED = 42


# ============================================================
# データ読み込み
# ============================================================

def load_embeddings(labels: list[str]) -> tuple[dict, dict]:
    """候補・クエリ埋め込みを読み込む。"""
    cand_embs, query_embs = {}, {}
    for label in labels:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape}")
    return cand_embs, query_embs


# ============================================================
# アンカー選定
# FPSはソース側モデルの候補埋め込みで実行。
# ターゲット側は同一テキストインデックスで対応点を取得。
# ソース空間の分布カバレッジを保証する設計。
# ============================================================

def select_anchors(cand_emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    """FPSでアンカーインデックスを選定（ソース側候補埋め込みで実行）。"""
    dummy_texts = [str(i) for i in range(len(cand_emb))]
    indices, _ = select_anchors_fps(cand_emb, dummy_texts, k, seed=seed)
    return np.array(indices)


# ============================================================
# 手法1: Orthogonal Procrustes
# ============================================================

def run_procrustes_same_dim(
    query_src: np.ndarray, query_tgt: np.ndarray,
    cand_src: np.ndarray, cand_tgt: np.ndarray,
    anchor_indices: np.ndarray,
) -> dict:
    """手法1a: 同次元Procrustes。ソース空間をターゲット空間に直交変換。"""
    anchor_src = cand_src[anchor_indices]
    anchor_tgt = cand_tgt[anchor_indices]

    W, _ = orthogonal_procrustes(anchor_src, anchor_tgt)
    query_transformed = query_src @ W

    # ターゲット空間で検索: 変換済みクエリ vs ターゲットクエリ
    return evaluate_retrieval(query_transformed, query_tgt)


def run_procrustes_zeropad(
    query_src: np.ndarray, query_tgt: np.ndarray,
    cand_src: np.ndarray, cand_tgt: np.ndarray,
    anchor_indices: np.ndarray,
) -> dict:
    """手法1b: ゼロパディングProcrustes。低次元側を高次元にパディング。"""
    dim_src = query_src.shape[1]
    dim_tgt = query_tgt.shape[1]
    dim_max = max(dim_src, dim_tgt)

    def pad(arr, target_dim):
        if arr.shape[1] == target_dim:
            return arr
        padded = np.zeros((arr.shape[0], target_dim))
        padded[:, :arr.shape[1]] = arr
        return padded

    anchor_src_pad = pad(cand_src[anchor_indices], dim_max)
    anchor_tgt_pad = pad(cand_tgt[anchor_indices], dim_max)

    W, _ = orthogonal_procrustes(anchor_src_pad, anchor_tgt_pad)
    query_src_pad = pad(query_src, dim_max)
    query_transformed = query_src_pad @ W

    # ターゲット次元にスライスして戻す（padded次元のまま検索しない）
    query_transformed = query_transformed[:, :dim_tgt]
    return evaluate_retrieval(query_transformed, query_tgt)


def run_procrustes_pca(
    query_src: np.ndarray, query_tgt: np.ndarray,
    cand_src: np.ndarray, cand_tgt: np.ndarray,
    anchor_indices: np.ndarray,
    pca_fit: str = "all",
) -> dict:
    """手法1c: PCA次元削減 + Procrustes。高次元側を低次元に落としてからProcrustes。"""
    dim_src = query_src.shape[1]
    dim_tgt = query_tgt.shape[1]
    target_dim = min(dim_src, dim_tgt)

    # どちらが高次元か判定
    if dim_src < dim_tgt:
        # ターゲット側をPCAで次元削減
        if pca_fit == "all":
            pca = PCA(n_components=target_dim)
            pca.fit(cand_tgt)
        else:  # anchor_only
            n_comp = min(target_dim, len(anchor_indices))
            pca = PCA(n_components=n_comp)
            pca.fit(cand_tgt[anchor_indices])
            target_dim = n_comp  # 次元をアンカー数で制限
        query_tgt_reduced = pca.transform(query_tgt)
        anchor_tgt_reduced = pca.transform(cand_tgt[anchor_indices])
        if pca_fit == "anchor_only" and target_dim < dim_src:
            # ソース側もPCAで同次元に合わせる
            pca_src = PCA(n_components=target_dim)
            pca_src.fit(cand_src)
            anchor_src_use = pca_src.transform(cand_src[anchor_indices])
            query_src_use = pca_src.transform(query_src)
        else:
            anchor_src_use = cand_src[anchor_indices]
            query_src_use = query_src
    else:
        # ソース側をPCAで次元削減
        if pca_fit == "all":
            pca = PCA(n_components=target_dim)
            pca.fit(cand_src)
        else:
            n_comp = min(target_dim, len(anchor_indices))
            pca = PCA(n_components=n_comp)
            pca.fit(cand_src[anchor_indices])
            target_dim = n_comp
        query_src_use = pca.transform(query_src)
        anchor_src_use = pca.transform(cand_src[anchor_indices])
        if pca_fit == "anchor_only" and target_dim < dim_tgt:
            pca_tgt = PCA(n_components=target_dim)
            pca_tgt.fit(cand_tgt)
            anchor_tgt_reduced = pca_tgt.transform(cand_tgt[anchor_indices])
            query_tgt_reduced = pca_tgt.transform(query_tgt)
        else:
            anchor_tgt_reduced = cand_tgt[anchor_indices]
            query_tgt_reduced = query_tgt

    W, _ = orthogonal_procrustes(anchor_src_use, anchor_tgt_reduced)
    query_transformed = query_src_use @ W

    return evaluate_retrieval(query_transformed, query_tgt_reduced)


# ============================================================
# 手法2: RBF kernel RAT
# ============================================================

def run_rat_rbf(
    query_src: np.ndarray, query_tgt: np.ndarray,
    cand_src: np.ndarray, cand_tgt: np.ndarray,
    anchor_indices: np.ndarray,
    gamma: float | str = "auto",
) -> dict:
    """RBFカーネルRAT。gamma="auto"は1/dimを使用。"""
    anchor_src = cand_src[anchor_indices]
    anchor_tgt = cand_tgt[anchor_indices]

    gamma_src = 1.0 / query_src.shape[1] if gamma == "auto" else gamma
    gamma_tgt = 1.0 / query_tgt.shape[1] if gamma == "auto" else gamma

    rel_src = to_relative(query_src, anchor_src, kernel="rbf", gamma=gamma_src)
    rel_tgt = to_relative(query_tgt, anchor_tgt, kernel="rbf", gamma=gamma_tgt)

    rel_src = normalize_zscore(rel_src)
    rel_tgt = normalize_zscore(rel_tgt)

    return evaluate_retrieval(rel_src, rel_tgt)


# ============================================================
# 手法3: 高K RAT (poly kernel)
# ============================================================

def run_rat_poly(
    query_src: np.ndarray, query_tgt: np.ndarray,
    cand_src: np.ndarray, cand_tgt: np.ndarray,
    anchor_indices: np.ndarray,
) -> dict:
    """標準poly kernel RAT（z-score正規化付き）。"""
    anchor_src = cand_src[anchor_indices]
    anchor_tgt = cand_tgt[anchor_indices]

    rel_src = to_relative(query_src, anchor_src, kernel="poly", degree=2, coef0=1.0)
    rel_tgt = to_relative(query_tgt, anchor_tgt, kernel="poly", degree=2, coef0=1.0)

    rel_src = normalize_zscore(rel_src)
    rel_tgt = normalize_zscore(rel_tgt)

    return evaluate_retrieval(rel_src, rel_tgt)


# ============================================================
# ランダムベースライン
# ============================================================

def compute_random_baseline(n_queries: int, n_permutations: int = 1000) -> dict:
    """ランダム検索のベースライン（理論値 + permutation test）。"""
    theoretical = {
        "recall_at_1": 1.0 / n_queries,
        "recall_at_5": 5.0 / n_queries,
        "recall_at_10": 10.0 / n_queries,
        "mrr": sum(1.0 / k for k in range(1, n_queries + 1)) / n_queries,
    }

    # Permutation test
    rng = np.random.RandomState(42)
    r1s, r5s, r10s, mrrs = [], [], [], []
    for _ in range(n_permutations):
        perm = rng.permutation(n_queries)
        ranks = np.array([np.where(perm == i)[0][0] + 1 for i in range(n_queries)])
        r1s.append(np.mean(ranks == 1))
        r5s.append(np.mean(ranks <= 5))
        r10s.append(np.mean(ranks <= 10))
        mrrs.append(np.mean(1.0 / ranks))

    return {
        "theoretical": theoretical,
        "permutation_mean": {
            "recall_at_1": float(np.mean(r1s)),
            "recall_at_5": float(np.mean(r5s)),
            "recall_at_10": float(np.mean(r10s)),
            "mrr": float(np.mean(mrrs)),
        },
        "permutation_std": {
            "recall_at_1": float(np.std(r1s)),
            "recall_at_5": float(np.std(r5s)),
            "recall_at_10": float(np.std(r10s)),
            "mrr": float(np.std(mrrs)),
        },
    }


# ============================================================
# 集約ユーティリティ
# ============================================================

def aggregate_trials(trial_results: list[dict]) -> dict:
    """複数トライアルの平均±stdを計算。"""
    keys = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    agg = {}
    for key in keys:
        vals = [r[key] for r in trial_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    agg["median_rank_mean"] = float(np.mean([r["median_rank"] for r in trial_results]))
    agg["n_trials"] = len(trial_results)
    agg["raw_trials"] = trial_results
    return agg


# ============================================================
# メイン実験ループ
# ============================================================

def run_experiment():
    print("=" * 60)
    print("Scout S1: ゼロショット・クロスクラスター壁突破")
    print("=" * 60)
    t_start = time.time()

    # 必要なモデルラベルを収集
    all_labels = set()
    for src, tgt, _, _ in CROSS_CLUSTER_PAIRS + CONTROL_PAIRS:
        all_labels.update([src, tgt])

    print(f"\n対象モデル: {sorted(all_labels)}")
    print("埋め込み読み込み中...")
    cand_embs, query_embs = load_embeddings(sorted(all_labels))

    n_queries = query_embs[list(query_embs.keys())[0]].shape[0]
    print(f"\nクエリ数: {n_queries}")

    results = []

    # --------------------------------------------------------
    # 全ペアに対して実験実行
    # --------------------------------------------------------
    all_pairs = [
        *[(src, tgt, desc, dm, "cross_cluster") for src, tgt, desc, dm in CROSS_CLUSTER_PAIRS],
        *[(src, tgt, desc, dm, "control") for src, tgt, desc, dm in CONTROL_PAIRS],
    ]

    for src, tgt, desc, dim_match, pair_type in all_pairs:
        print(f"\n{'='*60}")
        print(f"ペア: {src}→{tgt} ({desc}) [{pair_type}]")
        print(f"{'='*60}")

        cs, ct = cand_embs[src], cand_embs[tgt]
        qs, qt = query_embs[src], query_embs[tgt]

        # ============================================
        # 手法1: Procrustes variants
        # ============================================
        for K in ANCHOR_COUNTS:
            if K > cs.shape[0]:
                continue

            trial_results_1a = []
            trial_results_1b = []
            trial_results_1c_all = []
            trial_results_1c_anchor = []

            for trial in range(NUM_TRIALS):
                seed = BASE_SEED + trial
                anchor_idx = select_anchors(cs, K, seed)

                # 1a: 同次元Procrustes（同次元ペアのみ）
                if dim_match == "same":
                    m = run_procrustes_same_dim(qs, qt, cs, ct, anchor_idx)
                    trial_results_1a.append(m)

                # 1b: ゼロパディングProcrustes（異次元ペアのみ）
                if dim_match == "different":
                    m = run_procrustes_zeropad(qs, qt, cs, ct, anchor_idx)
                    trial_results_1b.append(m)

                    # 1c: PCA + Procrustes（異次元ペアのみ）
                    m_all = run_procrustes_pca(qs, qt, cs, ct, anchor_idx, pca_fit="all")
                    trial_results_1c_all.append(m_all)

                    m_anc = run_procrustes_pca(qs, qt, cs, ct, anchor_idx, pca_fit="anchor_only")
                    trial_results_1c_anchor.append(m_anc)

            if trial_results_1a:
                agg = aggregate_trials(trial_results_1a)
                results.append({
                    "pair": f"{src}→{tgt}", "pair_type": pair_type,
                    "method": "procrustes_same_dim", "variant": "1a",
                    "K": K, **agg,
                })
                print(f"  1a Procrustes K={K}: R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

            if trial_results_1b:
                agg = aggregate_trials(trial_results_1b)
                results.append({
                    "pair": f"{src}→{tgt}", "pair_type": pair_type,
                    "method": "procrustes_zeropad", "variant": "1b",
                    "K": K, **agg,
                })
                print(f"  1b ZeroPad   K={K}: R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

            if trial_results_1c_all:
                agg = aggregate_trials(trial_results_1c_all)
                results.append({
                    "pair": f"{src}→{tgt}", "pair_type": pair_type,
                    "method": "procrustes_pca", "variant": "1c",
                    "pca_fit": "all", "K": K, **agg,
                })
                print(f"  1c PCA(all)  K={K}: R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

            if trial_results_1c_anchor:
                agg = aggregate_trials(trial_results_1c_anchor)
                results.append({
                    "pair": f"{src}→{tgt}", "pair_type": pair_type,
                    "method": "procrustes_pca", "variant": "1c",
                    "pca_fit": "anchor_only", "K": K, **agg,
                })
                print(f"  1c PCA(anc)  K={K}: R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

        # ============================================
        # 手法2: RBF kernel RAT (K=500のみ)
        # ============================================
        K_rbf = 500
        for gamma in RBF_GAMMAS:
            trial_results_rbf = []
            for trial in range(NUM_TRIALS):
                seed = BASE_SEED + trial
                anchor_idx = select_anchors(cs, K_rbf, seed)
                m = run_rat_rbf(qs, qt, cs, ct, anchor_idx, gamma=gamma)
                trial_results_rbf.append(m)

            agg = aggregate_trials(trial_results_rbf)
            gamma_label = f"1/{qs.shape[1]}" if gamma == "auto" else str(gamma)
            results.append({
                "pair": f"{src}→{tgt}", "pair_type": pair_type,
                "method": "rat_rbf", "variant": "2",
                "K": K_rbf, "gamma": gamma_label, **agg,
            })
            print(f"  2  RBF γ={gamma_label:>8s} K={K_rbf}: R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

        # ============================================
        # 手法3: 高K poly RAT (K=2000)
        # ============================================
        K_high = 2000
        trial_results_highk = []
        for trial in range(NUM_TRIALS):
            seed = BASE_SEED + trial
            anchor_idx = select_anchors(cs, K_high, seed)
            m = run_rat_poly(qs, qt, cs, ct, anchor_idx)
            trial_results_highk.append(m)

        agg = aggregate_trials(trial_results_highk)
        results.append({
            "pair": f"{src}→{tgt}", "pair_type": pair_type,
            "method": "rat_poly_highK", "variant": "3",
            "K": K_high, **agg,
        })
        print(f"  3  Poly K={K_high}:     R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

        # ============================================
        # ベースライン: 標準RAT (poly K=500)
        # ============================================
        K_std = 500
        trial_results_std = []
        for trial in range(NUM_TRIALS):
            seed = BASE_SEED + trial
            anchor_idx = select_anchors(cs, K_std, seed)
            m = run_rat_poly(qs, qt, cs, ct, anchor_idx)
            trial_results_std.append(m)

        agg = aggregate_trials(trial_results_std)
        results.append({
            "pair": f"{src}→{tgt}", "pair_type": pair_type,
            "method": "rat_poly_baseline", "variant": "baseline",
            "K": K_std, **agg,
        })
        print(f"  BL Poly K={K_std}:      R@1={agg['recall_at_1_mean']:.4f}±{agg['recall_at_1_std']:.4f}")

    # --------------------------------------------------------
    # ランダムベースライン
    # --------------------------------------------------------
    print(f"\nランダムベースライン計算中...")
    random_bl = compute_random_baseline(n_queries)
    print(f"  理論値 R@1={random_bl['theoretical']['recall_at_1']:.4f}")

    # --------------------------------------------------------
    # 結果保存
    # --------------------------------------------------------
    elapsed = time.time() - t_start

    output = {
        "experiment": "scout_s1",
        "description": "ゼロショット・クロスクラスター壁突破の探索実験",
        "config": {
            "cross_cluster_pairs": [f"{s}→{t}" for s, t, _, _ in CROSS_CLUSTER_PAIRS],
            "control_pairs": [f"{s}→{t}" for s, t, _, _ in CONTROL_PAIRS],
            "anchor_counts": ANCHOR_COUNTS,
            "rbf_gammas": [str(g) for g in RBF_GAMMAS],
            "num_trials": NUM_TRIALS,
            "base_seed": BASE_SEED,
            "n_queries": n_queries,
            "n_candidates": cs.shape[0],
        },
        "random_baseline": random_bl,
        "results": results,
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "scout_s1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完了! {len(results)}条件, {elapsed:.1f}秒")
    print(f"結果: {out_path}")
    print(f"{'='*60}")

    # --------------------------------------------------------
    # サマリー表示
    # --------------------------------------------------------
    print("\n=== サマリー ===")
    print(f"{'ペア':>12s} {'手法':>22s} {'K':>5s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR':>8s}")
    print("-" * 80)
    for r in results:
        k_str = str(r.get("K", ""))
        gamma_str = f" γ={r['gamma']}" if "gamma" in r else ""
        pca_str = f" pca={r['pca_fit']}" if "pca_fit" in r else ""
        method_str = f"{r['method']}{gamma_str}{pca_str}"
        print(f"{r['pair']:>12s} {method_str:>22s} {k_str:>5s} "
              f"{r['recall_at_1_mean']:>7.4f} {r['recall_at_5_mean']:>7.4f} "
              f"{r['recall_at_10_mean']:>7.4f} {r['mrr_mean']:>7.4f}")


if __name__ == "__main__":
    run_experiment()
