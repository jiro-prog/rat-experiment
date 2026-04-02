"""
RAT B×C 失敗の原因分析

1. アンカーembedding分布のt-SNE可視化（3モデル比較）
2. アンカー間類似度行列のエントロピー分析
3. モデルペア互換性スコアの定義と計測
4. 相対表現の相関構造の比較
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy, spearmanr

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative


K = 500
CANDIDATE_POOL = 2000


def compute_sim_matrix_stats(embeddings: np.ndarray, label: str) -> dict:
    """アンカー間類似度行列の統計量を計算する。"""
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)  # 自己類似度は除外

    # 上三角のみ取得
    triu_idx = np.triu_indices_from(sim, k=1)
    sims = sim[triu_idx]

    # 類似度のヒストグラムからエントロピーを計算
    hist, _ = np.histogram(sims, bins=50, range=(-1, 1), density=True)
    hist = hist / hist.sum()  # 確率分布に正規化
    hist = hist[hist > 0]  # 0を除外
    ent = float(entropy(hist))

    return {
        "label": label,
        "mean_sim": float(np.mean(sims)),
        "std_sim": float(np.std(sims)),
        "min_sim": float(np.min(sims)),
        "max_sim": float(np.max(sims)),
        "entropy": ent,
    }


def compute_relative_repr_correlation(
    rel_X: np.ndarray, rel_Y: np.ndarray,
) -> dict:
    """
    2つのモデルの相対表現間の相関を計測する。

    相対表現 rel_X[i] と rel_Y[i] は同じ文のもの。
    各文について、アンカーへの類似度プロファイルの順位相関を取る。
    """
    n = len(rel_X)
    spearman_corrs = []
    for i in range(n):
        corr, _ = spearmanr(rel_X[i], rel_Y[i])
        if not np.isnan(corr):
            spearman_corrs.append(corr)

    corrs = np.array(spearman_corrs)
    return {
        "mean_spearman": float(np.mean(corrs)),
        "std_spearman": float(np.std(corrs)),
        "median_spearman": float(np.median(corrs)),
    }


def compute_pairwise_sim_correlation(
    emb_X: np.ndarray, emb_Y: np.ndarray,
) -> float:
    """
    モデルXとYで同じ文ペア間の類似度順序がどの程度保存されるか。
    文ペア間コサイン類似度の順位相関（Spearman）。
    """
    sim_X = cosine_similarity(emb_X)
    sim_Y = cosine_similarity(emb_Y)

    triu = np.triu_indices_from(sim_X, k=1)
    corr, _ = spearmanr(sim_X[triu], sim_Y[triu])
    return float(corr)


def plot_anchor_tsne(
    anchor_embs: dict[str, np.ndarray],
    model_names: dict[str, str],
    save_path: Path,
):
    """各モデルのアンカーembeddingをt-SNEで並べて可視化する。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"A": "blue", "B": "red", "C": "green"}

    for idx, (label, emb) in enumerate(anchor_embs.items()):
        tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30)
        coords = tsne.fit_transform(emb)

        ax = axes[idx]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors[label], alpha=0.4, s=10)
        short = model_names[label].split("/")[-1]
        ax.set_title(f"Model {label}: {short}\n(dim={emb.shape[1]})")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Anchor Embedding Distribution (t-SNE per model)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"保存: {save_path}")


def plot_sim_distributions(
    anchor_embs: dict[str, np.ndarray],
    model_names: dict[str, str],
    save_path: Path,
):
    """アンカー間類似度の分布を重ねてプロットする。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "blue", "B": "red", "C": "green"}

    for label, emb in anchor_embs.items():
        sim = cosine_similarity(emb)
        triu = np.triu_indices_from(sim, k=1)
        sims = sim[triu]
        short = model_names[label].split("/")[-1]
        ax.hist(sims, bins=80, alpha=0.4, label=f"{label}: {short}", color=colors[label], density=True)

    ax.set_xlabel("Cosine Similarity between Anchors")
    ax.set_ylabel("Density")
    ax.set_title("Anchor Inter-Similarity Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"保存: {save_path}")


def plot_relative_repr_comparison(
    rels: dict[str, np.ndarray],
    model_names: dict[str, str],
    save_path: Path,
    sample_idx: int = 0,
):
    """同じ文の相対表現プロファイルを3モデル並べて可視化する。"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    colors = {"A": "blue", "B": "red", "C": "green"}

    for idx, (label, rel) in enumerate(rels.items()):
        ax = axes[idx]
        profile = rel[sample_idx]  # 1文のアンカーとの類似度プロファイル
        ax.bar(range(len(profile)), profile, color=colors[label], alpha=0.6, width=1.0)
        short = model_names[label].split("/")[-1]
        ax.set_ylabel(f"Model {label}\n({short})")
        ax.set_ylim(0, max(profile) * 1.3)

    axes[-1].set_xlabel("Anchor Index")
    axes[0].set_title(f"Relative Representation Profile (Query #{sample_idx})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"保存: {save_path}")


def plot_compatibility_matrix(
    compat_scores: dict[str, dict[str, float]],
    save_path: Path,
):
    """モデル間互換性スコアのヒートマップ。"""
    labels = sorted(compat_scores.keys())
    n = len(labels)
    matrix = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li == lj:
                matrix[i, j] = 1.0
            elif lj in compat_scores.get(li, {}):
                matrix[i, j] = compat_scores[li][lj]
            elif li in compat_scores.get(lj, {}):
                matrix[i, j] = compat_scores[lj][li]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=12)

    ax.set_title("Model Compatibility Score\n(Pairwise Similarity Order Correlation)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"保存: {save_path}")


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "A": config.MODEL_A,
        "B": config.MODEL_B,
        "C": config.MODEL_C,
    }

    # ========================================
    # データ準備（Phase 2と同じ条件）
    # ========================================
    print("=" * 60)
    print("データ準備")
    print("=" * 60)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )

    cand_embs = {}
    query_embs = {}
    for label, model_name in models.items():
        short = model_name.split("/")[-1]
        print(f"  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)

    # FPSアンカー選定（Model A基準）
    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in models}

    # ========================================
    # 分析1: アンカー間類似度分布
    # ========================================
    print("\n" + "=" * 60)
    print("分析1: アンカー間類似度分布")
    print("=" * 60)

    sim_stats = {}
    for label in models:
        stats = compute_sim_matrix_stats(anchor_embs[label], label)
        sim_stats[label] = stats
        short = models[label].split("/")[-1]
        print(f"\n  Model {label} ({short}):")
        print(f"    Mean sim:  {stats['mean_sim']:.4f}")
        print(f"    Std sim:   {stats['std_sim']:.4f}")
        print(f"    Range:     [{stats['min_sim']:.4f}, {stats['max_sim']:.4f}]")
        print(f"    Entropy:   {stats['entropy']:.4f}")

    plot_sim_distributions(anchor_embs, models, config.RESULTS_DIR / "analysis_sim_dist.png")

    # ========================================
    # 分析2: モデル間の空間構造互換性
    # ========================================
    print("\n" + "=" * 60)
    print("分析2: モデル間の空間構造互換性（ペアワイズ類似度順位相関）")
    print("=" * 60)

    # クエリ間の類似度順序がモデル間でどの程度保存されるか
    compat_scores = {}
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for x, y in pairs:
        corr = compute_pairwise_sim_correlation(query_embs[x], query_embs[y])
        if x not in compat_scores:
            compat_scores[x] = {}
        compat_scores[x][y] = corr
        short_x = models[x].split("/")[-1]
        short_y = models[y].split("/")[-1]
        print(f"  {x}×{y} ({short_x} × {short_y}): Spearman ρ = {corr:.4f}")

    plot_compatibility_matrix(compat_scores, config.RESULTS_DIR / "analysis_compat.png")

    # ========================================
    # 分析3: 相対表現の相関構造
    # ========================================
    print("\n" + "=" * 60)
    print("分析3: 相対表現プロファイルの相関（同一文のアンカー類似度パターン）")
    print("=" * 60)

    # 各モデルの相対表現を計算
    rels = {}
    for label in models:
        rels[label] = to_relative(query_embs[label], anchor_embs[label], kernel="poly", degree=2, coef0=1.0)

    rel_corrs = {}
    for x, y in pairs:
        corr_stats = compute_relative_repr_correlation(rels[x], rels[y])
        rel_corrs[f"{x}×{y}"] = corr_stats
        short_x = models[x].split("/")[-1]
        short_y = models[y].split("/")[-1]
        print(f"\n  {x}×{y} ({short_x} × {short_y}):")
        print(f"    Mean Spearman:   {corr_stats['mean_spearman']:.4f}")
        print(f"    Median Spearman: {corr_stats['median_spearman']:.4f}")
        print(f"    Std:             {corr_stats['std_spearman']:.4f}")

    # ========================================
    # 分析4: 可視化
    # ========================================
    print("\n" + "=" * 60)
    print("分析4: 可視化")
    print("=" * 60)

    plot_anchor_tsne(anchor_embs, models, config.RESULTS_DIR / "analysis_anchor_tsne.png")
    plot_relative_repr_comparison(rels, models, config.RESULTS_DIR / "analysis_rel_profile.png")

    # ========================================
    # 総合判定
    # ========================================
    print("\n" + "=" * 60)
    print("総合判定")
    print("=" * 60)

    print("\n  空間構造互換性 vs RAT性能:")
    print(f"  {'ペア':<8} {'構造互換性(ρ)':>14} {'相対表現相関':>14} {'Recall@1':>10}")
    print(f"  {'-'*50}")

    # Phase 2の結果を参照用にハードコード（同一シード・同一条件）
    phase2_r1 = {"A×B": 79.4, "A×C": 98.4, "B×C": 15.2}

    for x, y in pairs:
        pair = f"{x}×{y}"
        compat = compat_scores.get(x, {}).get(y, 0)
        rel_corr = rel_corrs[pair]["mean_spearman"]
        r1 = phase2_r1.get(pair, 0)
        print(f"  {pair:<8} {compat:>14.4f} {rel_corr:>14.4f} {r1:>9.1f}%")

    print(f"\n  分析:")
    ab_compat = compat_scores["A"]["B"]
    ac_compat = compat_scores["A"]["C"]
    bc_compat = compat_scores["B"]["C"]
    ab_rel = rel_corrs["A×B"]["mean_spearman"]
    bc_rel = rel_corrs["B×C"]["mean_spearman"]

    print(f"  - A×CとA×Bの構造互換性の差: {ac_compat - ab_compat:.4f}")
    print(f"  - A×BとB×Cの構造互換性の差: {ab_compat - bc_compat:.4f}")
    print(f"  - A×BとB×Cの相対表現相関の差: {ab_rel - bc_rel:.4f}")

    if bc_compat < ab_compat * 0.8:
        print(f"\n  → B×Cの空間構造互換性が低い。E5-largeの意味空間は")
        print(f"    384dモデルとは根本的に異なる順序構造を持っている。")
    if bc_rel < ab_rel * 0.5:
        print(f"  → B×Cの相対表現相関が極めて低い。同じアンカーに対する")
        print(f"    類似度パターンがモデル間で一致しないため、検索が機能しない。")

    # 保存
    elapsed = time.time() - start_time

    output = {
        "anchor_sim_stats": sim_stats,
        "compatibility_scores": compat_scores,
        "relative_repr_correlations": rel_corrs,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "analysis_bc_failure.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
