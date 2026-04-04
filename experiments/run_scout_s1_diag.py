"""
Scout S1 診断: クロスクラスター間のローカル/クラスタ構造一致度

目的: RDM ρ≈0（グローバル距離構造が無相関）でも、
ローカル近傍構造やクラスタ構造が一致しているか検証する。
これがゼロなら、ゼロショットの限界は確定。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

DATA_DIR = config.DATA_DIR / "d2_matrix"

# 実験対象ペア
PAIRS = [
    # (src, tgt, description, type)
    ("A", "O", "MiniLM(384)→Arctic-xs(384)", "cross_cluster"),
    ("C", "P", "BGE-s(384)→Arctic-s(384)", "cross_cluster"),
    ("A", "N", "MiniLM(384)→Arctic-m(768)", "cross_cluster"),
    ("A", "B", "MiniLM(384)→E5-large(1024)", "control_same_cluster"),
]

# k-NN のk値
KNN_KS = [5, 10, 20, 50, 100]

# k-means のクラスタ数
KMEANS_KS = [10, 20, 50, 100]


def load_embeddings(labels):
    embs = {}
    for label in labels:
        cand = np.load(DATA_DIR / f"cand_{label}.npy")
        query = np.load(DATA_DIR / f"query_{label}.npy")
        # 候補+クエリを結合して全データで診断
        embs[label] = np.vstack([cand, query])
    return embs


def compute_knn_indices(emb: np.ndarray, k: int) -> np.ndarray:
    """各点のk近傍インデックスを返す（自身を除く）。"""
    sim = cosine_similarity(emb)
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def mutual_knn_overlap(knn_src: np.ndarray, knn_tgt: np.ndarray, k: int) -> dict:
    """同一テキストのk近傍がどれだけ一致しているか。"""
    n = len(knn_src)
    overlaps = np.zeros(n)
    for i in range(n):
        src_set = set(knn_src[i])
        tgt_set = set(knn_tgt[i])
        overlaps[i] = len(src_set & tgt_set) / k

    return {
        "mean": float(np.mean(overlaps)),
        "std": float(np.std(overlaps)),
        "median": float(np.median(overlaps)),
        "p95": float(np.percentile(overlaps, 95)),
        "p99": float(np.percentile(overlaps, 99)),
        "max": float(np.max(overlaps)),
        "zero_frac": float(np.mean(overlaps == 0)),
    }


def random_knn_overlap_baseline(n: int, k: int, n_permutations: int = 100) -> dict:
    """ランダムに近傍を選んだ場合の期待overlap。"""
    # 超幾何分布の期待値: k * (k / (n-1))
    expected = k * k / (n - 1)
    expected_frac = expected / k  # = k / (n-1)
    return {
        "expected_overlap_frac": float(expected_frac),
        "expected_overlap_count": float(expected),
    }


def cluster_agreement(emb_src: np.ndarray, emb_tgt: np.ndarray, n_clusters: int,
                      seed: int = 42) -> dict:
    """両空間でk-meansし、クラスタ割り当ての一致度を測定。"""
    km_src = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km_tgt = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

    labels_src = km_src.fit_predict(emb_src)
    labels_tgt = km_tgt.fit_predict(emb_tgt)

    ari = adjusted_rand_score(labels_src, labels_tgt)
    nmi = normalized_mutual_info_score(labels_src, labels_tgt)

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
    }


def run_diagnostics():
    print("=" * 60)
    print("Scout S1 診断: ローカル/クラスタ構造一致度")
    print("=" * 60)
    t_start = time.time()

    all_labels = set()
    for src, tgt, _, _ in PAIRS:
        all_labels.update([src, tgt])

    print(f"\n対象モデル: {sorted(all_labels)}")
    print("埋め込み読み込み中...")
    embs = load_embeddings(sorted(all_labels))
    for label, emb in embs.items():
        print(f"  {label}: {emb.shape}")

    n = embs[list(embs.keys())[0]].shape[0]
    print(f"データ点数: {n} (候補2000 + クエリ500)")

    results = []

    for src, tgt, desc, pair_type in PAIRS:
        print(f"\n{'='*60}")
        print(f"ペア: {src}→{tgt} ({desc}) [{pair_type}]")
        print(f"{'='*60}")

        emb_src = embs[src]
        emb_tgt = embs[tgt]

        pair_result = {
            "pair": f"{src}→{tgt}",
            "description": desc,
            "pair_type": pair_type,
            "knn_overlap": {},
            "cluster_agreement": {},
        }

        # ============================================
        # 1. Mutual k-NN overlap
        # ============================================
        print("\n--- Mutual k-NN Overlap ---")

        # k-NNインデックスを最大kで一度計算し、スライスで再利用
        max_k = max(KNN_KS)
        print(f"  k-NN計算中 (k_max={max_k})...")
        knn_src = compute_knn_indices(emb_src, max_k)
        knn_tgt = compute_knn_indices(emb_tgt, max_k)

        for k in KNN_KS:
            overlap = mutual_knn_overlap(knn_src[:, :k], knn_tgt[:, :k], k)
            baseline = random_knn_overlap_baseline(n, k)
            pair_result["knn_overlap"][k] = {
                **overlap,
                "random_expected": baseline["expected_overlap_frac"],
            }
            ratio = overlap["mean"] / baseline["expected_overlap_frac"] if baseline["expected_overlap_frac"] > 0 else 0
            print(f"  k={k:>3d}: overlap={overlap['mean']:.4f} (random={baseline['expected_overlap_frac']:.4f}, "
                  f"ratio={ratio:.2f}x, zero_frac={overlap['zero_frac']:.2f}, p95={overlap['p95']:.4f})")

        # ============================================
        # 2. Cluster Agreement
        # ============================================
        print("\n--- Cluster Agreement (k-means) ---")

        for n_clusters in KMEANS_KS:
            ca = cluster_agreement(emb_src, emb_tgt, n_clusters)
            pair_result["cluster_agreement"][n_clusters] = ca
            print(f"  k={n_clusters:>3d}: ARI={ca['adjusted_rand_index']:.4f}, "
                  f"NMI={ca['normalized_mutual_info']:.4f}")

        results.append(pair_result)

    elapsed = time.time() - t_start

    # 結果保存
    output = {
        "experiment": "scout_s1_diagnostics",
        "description": "クロスクラスター間のローカル/クラスタ構造一致度診断",
        "config": {
            "knn_ks": KNN_KS,
            "kmeans_ks": KMEANS_KS,
            "n_data_points": n,
        },
        "results": results,
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "scout_s1_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # サマリー
    print(f"\n{'='*60}")
    print("サマリー: k-NN overlap ratio (実測/ランダム期待値)")
    print(f"{'='*60}")
    print(f"{'ペア':>12s}", end="")
    for k in KNN_KS:
        print(f"  k={k:>3d}", end="")
    print()
    for r in results:
        print(f"{r['pair']:>12s}", end="")
        for k in KNN_KS:
            ov = r["knn_overlap"][k]
            ratio = ov["mean"] / ov["random_expected"] if ov["random_expected"] > 0 else 0
            print(f"  {ratio:>5.1f}x", end="")
        print(f"  [{r['pair_type']}]")

    print(f"\nサマリー: Cluster Agreement (ARI)")
    print(f"{'ペア':>12s}", end="")
    for nc in KMEANS_KS:
        print(f"  k={nc:>3d}", end="")
    print()
    for r in results:
        print(f"{r['pair']:>12s}", end="")
        for nc in KMEANS_KS:
            ari = r["cluster_agreement"][nc]["adjusted_rand_index"]
            print(f"  {ari:>5.3f}", end="")
        print(f"  [{r['pair_type']}]")

    print(f"\n完了! {elapsed:.1f}秒")
    print(f"結果: {out_path}")


if __name__ == "__main__":
    run_diagnostics()
