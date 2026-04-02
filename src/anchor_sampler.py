import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets

import config


def load_sts_sentences() -> list[str]:
    """STSBenchmarkのtrain/dev/testから全文を取得し、重複を除去する。"""
    ds = load_dataset(config.DATASET_NAME)
    all_sentences = set()
    for split in ds:
        for row in ds[split]:
            all_sentences.add(row["sentence1"])
            all_sentences.add(row["sentence2"])
    return list(all_sentences)


def sample_anchors_and_queries(
    num_anchors: int = config.NUM_ANCHORS,
    num_queries: int = config.NUM_QUERIES,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """アンカーとクエリを重複なしでサンプリングする。"""
    sentences = load_sts_sentences()
    print(f"STSBenchmarkから {len(sentences)} 件のユニーク文を取得")

    total_needed = num_anchors + num_queries
    if len(sentences) < total_needed:
        raise ValueError(
            f"文が足りません: {len(sentences)} < {total_needed} (anchors={num_anchors} + queries={num_queries})"
        )

    rng = random.Random(seed)
    sampled = rng.sample(sentences, total_needed)
    anchors = sampled[:num_anchors]
    queries = sampled[num_anchors:]

    return anchors, queries


def select_anchors_kmeans(
    candidate_embeddings: np.ndarray,
    candidates: list[str],
    k: int,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[int], list[str]]:
    """
    k-meansクラスタリングでアンカーを選定する。
    各クラスタ中心に最も近い文をアンカーとして選ぶ。

    candidate_embeddings: (N, D) 候補文のembedding
    candidates: 候補文のリスト
    k: 選定するアンカー数
    returns: (選定されたインデックス, 選定された文)
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(candidate_embeddings)
    centers = kmeans.cluster_centers_  # (k, D)

    # 各クラスタ中心に最も近い候補を選ぶ
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(centers, candidate_embeddings)  # (k, N)
    selected_indices = np.argmax(sim, axis=1).tolist()

    # 重複除去（稀にあり得る）
    seen = set()
    unique_indices = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    selected_indices = unique_indices[:k]

    selected_texts = [candidates[i] for i in selected_indices]
    print(f"k-meansアンカー選定: {len(selected_texts)}件")
    return selected_indices, selected_texts


def select_anchors_fps(
    candidate_embeddings: np.ndarray,
    candidates: list[str],
    k: int,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[int], list[str]]:
    """
    Farthest Point Samplingでアンカーを選定する。
    意味空間で最大限分散したアンカーセットを構築。

    candidate_embeddings: (N, D) L2正規化済み
    """
    rng = np.random.RandomState(seed)
    n = len(candidate_embeddings)

    # 初期点をランダムに選択
    first = rng.randint(n)
    selected = [first]

    # コサイン距離 = 1 - コサイン類似度（正規化済みならdot product）
    # 各点から選択済み点群への最小距離を管理
    sim_to_selected = candidate_embeddings @ candidate_embeddings[first]  # (N,)
    min_sim = sim_to_selected.copy()  # 類似度が高い=距離が近い

    for _ in range(k - 1):
        # 最も類似度が低い（=最も遠い）点を選択
        # 選択済みの点は除外
        min_sim_copy = min_sim.copy()
        for idx in selected:
            min_sim_copy[idx] = np.inf
        next_idx = np.argmin(min_sim_copy)
        selected.append(next_idx)

        # 新しく選んだ点との類似度で min_sim を更新
        new_sim = candidate_embeddings @ candidate_embeddings[next_idx]
        min_sim = np.maximum(min_sim, new_sim)

    selected_texts = [candidates[i] for i in selected]
    print(f"FPSアンカー選定: {len(selected_texts)}件")
    return selected, selected_texts


def select_anchors_consensus(
    candidate_emb_A: np.ndarray,
    candidate_emb_B: np.ndarray,
    candidates: list[str],
    k: int,
) -> tuple[list[int], list[str]]:
    """
    両モデル合議でアンカーを選定する。
    Model AとModel Bの両空間で平均カバレッジが最大になるよう貪欲法で選定。

    カバレッジ: 各未選択点について、選択済みアンカーへの最大類似度の平均。
    これを最小化する（=最も遠い点をカバーする）アンカーを順次選ぶ。
    """
    n = len(candidates)
    selected = []

    # 各空間での「選択済みアンカーへの最大類似度」を管理
    max_sim_A = np.full(n, -np.inf)
    max_sim_B = np.full(n, -np.inf)

    for step in range(k):
        best_idx = -1
        best_score = np.inf

        # 候補を評価（全候補をスキャンする貪欲法）
        # スコア = 追加後の「全点の最近傍アンカーへの平均距離」の両モデル平均
        # 効率化: 各候補iを追加した場合のスコア改善量で判定
        for i in range(n):
            if i in set(selected):
                continue

            sim_A_i = candidate_emb_A @ candidate_emb_A[i]  # (N,)
            sim_B_i = candidate_emb_B @ candidate_emb_B[i]  # (N,)

            new_max_A = np.maximum(max_sim_A, sim_A_i)
            new_max_B = np.maximum(max_sim_B, sim_B_i)

            # 平均カバレッジ（類似度が高い=カバーされている）
            # 両モデルの平均最大類似度の平均を最大化したい
            score_A = np.mean(new_max_A)
            score_B = np.mean(new_max_B)
            # 両モデルで均等にカバーするために調和平均的にmin取る代わりに単純平均
            avg_coverage = (score_A + score_B) / 2

            # カバレッジを最大化 → 負にして最小化
            if -avg_coverage < best_score:
                best_score = -avg_coverage
                best_idx = i

        selected.append(best_idx)
        max_sim_A = np.maximum(max_sim_A, candidate_emb_A @ candidate_emb_A[best_idx])
        max_sim_B = np.maximum(max_sim_B, candidate_emb_B @ candidate_emb_B[best_idx])

        if (step + 1) % 100 == 0:
            print(f"  合議法: {step + 1}/{k} 選定済み")

    selected_texts = [candidates[i] for i in selected]
    print(f"両モデル合議アンカー選定: {len(selected_texts)}件")
    return selected, selected_texts


def save_data(anchors: list[str], queries: list[str], data_dir: Path = config.DATA_DIR):
    """アンカーとクエリをJSONで保存する。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "anchors.json", "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    with open(data_dir / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"保存完了: anchors={len(anchors)}件, queries={len(queries)}件 → {data_dir}")
