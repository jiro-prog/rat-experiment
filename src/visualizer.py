import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

import config


def plot_similarity_heatmap(
    rel_A: np.ndarray,
    rel_B: np.ndarray,
    save_path: Path,
    max_display: int = 100,
):
    """類似度行列のヒートマップ。対角線が明るければ成功。"""
    sim = cosine_similarity(rel_A[:max_display], rel_B[:max_display])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    ax.set_xlabel("Model B queries")
    ax.set_ylabel("Model A queries")
    ax.set_title("Cross-Model Similarity Matrix (Relative Repr.)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ヒートマップ保存: {save_path}")


def plot_tsne(
    rel_A: np.ndarray,
    rel_B: np.ndarray,
    save_path: Path,
    num_points: int = 200,
    seed: int = config.RANDOM_SEED,
):
    """Model AとBの相対表現をt-SNEで同一プロットに描画。"""
    n = min(num_points, len(rel_A))
    combined = np.vstack([rel_A[:n], rel_B[:n]])

    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, n - 1))
    coords = tsne.fit_transform(combined)

    coords_A = coords[:n]
    coords_B = coords[n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords_A[:, 0], coords_A[:, 1], c="blue", alpha=0.5, s=20, label="Model A")
    ax.scatter(coords_B[:, 0], coords_B[:, 1], c="red", alpha=0.5, s=20, label="Model B")

    # 同じ文のペアを線で結ぶ（最初の20本だけ）
    for i in range(min(20, n)):
        ax.plot(
            [coords_A[i, 0], coords_B[i, 0]],
            [coords_A[i, 1], coords_B[i, 1]],
            c="gray", alpha=0.3, linewidth=0.5,
        )

    ax.set_title("t-SNE of Relative Representations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNEプロット保存: {save_path}")


def plot_anchor_scaling(
    scaling_results: dict[int, dict],
    save_path: Path,
):
    """アンカー数 vs 各指標のグラフ。"""
    counts = sorted(scaling_results.keys())
    recall1 = [scaling_results[k]["recall_at_1"] for k in counts]
    recall10 = [scaling_results[k]["recall_at_10"] for k in counts]
    mrr = [scaling_results[k]["mrr"] for k in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, recall1, "o-", label="Recall@1")
    ax.plot(counts, recall10, "s-", label="Recall@10")
    ax.plot(counts, mrr, "^-", label="MRR")
    ax.set_xlabel("Number of Anchors")
    ax.set_ylabel("Score")
    ax.set_title("Anchor Count Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"スケーリンググラフ保存: {save_path}")
