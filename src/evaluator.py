import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _get_topk_indices(sim_matrix: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """各行のtop-kインデックスを返す。exclude_self=Trueなら対角要素を除外。"""
    if exclude_self:
        sim_matrix = sim_matrix.copy()
        np.fill_diagonal(sim_matrix, -np.inf)
    # (N, k)
    return np.argsort(-sim_matrix, axis=1)[:, :k]


def evaluate_neighbor_preservation(
    query_emb: np.ndarray,
    query_rel: np.ndarray,
    k: int = 10,
) -> dict:
    """
    元空間と相対表現空間の近傍構造保存率を測定する。

    query_emb: (N, D) 元のembedding（L2正規化済み）
    query_rel: (N, K) 相対表現
    k: top-kの近傍を比較

    自分自身は除外して計算する。
    """
    # 元空間でのtop-k近傍
    sim_orig = cosine_similarity(query_emb, query_emb)
    topk_orig = _get_topk_indices(sim_orig, k, exclude_self=True)

    # 相対表現空間でのtop-k近傍
    sim_rel = cosine_similarity(query_rel, query_rel)
    topk_rel = _get_topk_indices(sim_rel, k, exclude_self=True)

    # Overlap@k: 各クエリのtop-kの重なり率
    overlaps = []
    for i in range(len(query_emb)):
        orig_set = set(topk_orig[i])
        rel_set = set(topk_rel[i])
        overlaps.append(len(orig_set & rel_set) / k)

    overlaps = np.array(overlaps)
    return {
        f"overlap_at_{k}": float(np.mean(overlaps)),
        f"overlap_at_{k}_std": float(np.std(overlaps)),
        f"overlap_at_{k}_min": float(np.min(overlaps)),
        f"overlap_at_{k}_median": float(np.median(overlaps)),
    }


def evaluate_retrieval(rel_A: np.ndarray, rel_B: np.ndarray) -> dict:
    """
    相対表現間のクロスモデル検索精度を評価する。

    rel_A[i] と rel_B[i] は同じ文に対応。
    rel_A[i] に最も近い rel_B のインデックスが i であれば正解。
    """
    sim_matrix = cosine_similarity(rel_A, rel_B)  # (N, N)

    # 各クエリの正解ランクを計算
    ranks = []
    for i in range(len(rel_A)):
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
    }


def print_metrics(metrics: dict, label: str = ""):
    """評価結果を見やすく表示する。"""
    header = f"  {label}  " if label else ""
    print(f"\n{'='*50}")
    print(f"  {header}")
    print(f"{'='*50}")
    print(f"  Recall@1:    {metrics['recall_at_1']:.4f} ({metrics['recall_at_1']*100:.1f}%)")
    print(f"  Recall@5:    {metrics['recall_at_5']:.4f} ({metrics['recall_at_5']*100:.1f}%)")
    print(f"  Recall@10:   {metrics['recall_at_10']:.4f} ({metrics['recall_at_10']*100:.1f}%)")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print(f"  Median Rank: {metrics['median_rank']}")
    print(f"{'='*50}")
