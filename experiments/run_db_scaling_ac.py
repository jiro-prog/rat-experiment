"""
実験 5a-2: DB規模スケーリング (A×C ペア)

A×Bと同一設計でA×C（MiniLM × BGE-small、同系統384次元ペア）のカーブを取得。
A×Bの55%@100kに対して、同系統ペアの優位性を示す。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative, normalize_zscore

DB_SIZES = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
NUM_QUERIES = 500
NUM_ANCHORS = 500
CACHE_DIR = config.DATA_DIR / "db_scaling_ac_cache"


def load_distractor_texts(n: int) -> list[str]:
    """AllNLI triplet からユニーク文を取得。"""
    from datasets import load_dataset

    print(f"AllNLI からユニーク文を収集中 (目標: {n}件)...")
    ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    seen = set()
    texts = []
    for row in ds:
        for key in ["anchor", "positive", "negative"]:
            t = row[key].strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
                if len(texts) >= n:
                    print(f"  取得完了: {len(texts)}件")
                    return texts
    if len(texts) < n:
        raise ValueError(f"文不足: {len(texts)} < {n}")
    return texts


def embed_with_cache(model_name: str, texts: list[str], label: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{label}.npy"
    if cache_path.exists():
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            print(f"  キャッシュ読込: {cache_path} ({emb.shape})")
            return emb
        print(f"  キャッシュサイズ不一致 ({emb.shape[0]} != {len(texts)}), 再計算")
    print(f"  {label}: {len(texts)}文をembed中...")
    emb = embed_texts(model_name, texts)
    np.save(cache_path, emb)
    print(f"  キャッシュ保存: {cache_path} ({emb.shape})")
    return emb


def evaluate_at_db_size(query_rel, db_rel, db_size, num_queries):
    db_subset = db_rel[:db_size]
    sim_matrix = cosine_similarity(query_rel, db_subset)
    ranks = []
    for i in range(num_queries):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "db_size": db_size,
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "recall_at_50": float(np.mean(ranks <= 50)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": int(np.median(ranks)),
    }


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 5a-2: DB規模スケーリング (A×C)")
    print(f"  A={config.MODEL_A}")
    print(f"  C={config.MODEL_C}")
    print(f"  クエリ: {NUM_QUERIES}, アンカー: {NUM_ANCHORS}")
    print(f"  DB規模: {DB_SIZES}")
    print("=" * 60)

    max_db = max(DB_SIZES)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=max(NUM_ANCHORS + NUM_QUERIES, 2000),
        num_queries=NUM_QUERIES,
    )

    num_distractors = max_db - NUM_QUERIES
    distractor_texts = load_distractor_texts(num_distractors)
    db_texts = queries + distractor_texts[:num_distractors]
    print(f"DB構成: クエリ一致={NUM_QUERIES}件 + distractor={num_distractors}件 = {len(db_texts)}件")

    print("\nEmbedding計算...")
    anchor_cand_emb_A = embed_with_cache(config.MODEL_A, candidates, "anchor_cand_A")
    anchor_cand_emb_C = embed_with_cache(config.MODEL_C, candidates, "anchor_cand_C")
    query_emb_A = embed_with_cache(config.MODEL_A, queries, "query_A")
    db_emb_C = embed_with_cache(config.MODEL_C, db_texts, "db_C")

    print(f"\nFPSアンカー選定 (K={NUM_ANCHORS})...")
    dummy_texts = [f"dummy_{i}" for i in range(len(candidates))]
    fps_indices, _ = select_anchors_fps(anchor_cand_emb_A, dummy_texts, NUM_ANCHORS)
    anchor_emb_A = anchor_cand_emb_A[fps_indices]
    anchor_emb_C = anchor_cand_emb_C[fps_indices]

    print("\n相対表現計算...")
    rel_query = to_relative(query_emb_A, anchor_emb_A, kernel="poly", degree=2, coef0=1.0)
    rel_query = normalize_zscore(rel_query)
    print(f"  クエリ相対表現: {rel_query.shape}")

    rel_db = to_relative(db_emb_C, anchor_emb_C, kernel="poly", degree=2, coef0=1.0)
    rel_db = normalize_zscore(rel_db)
    print(f"  DB相対表現: {rel_db.shape}")

    print("\n評価...")
    results = []
    for db_size in DB_SIZES:
        if db_size > len(db_texts):
            print(f"  DB={db_size}: スキップ (テキスト不足)")
            continue
        metrics = evaluate_at_db_size(rel_query, rel_db, db_size, NUM_QUERIES)
        results.append(metrics)
        print(
            f"  DB={db_size:>7,}: R@1={metrics['recall_at_1']*100:5.1f}%  "
            f"R@10={metrics['recall_at_10']*100:5.1f}%  "
            f"R@50={metrics['recall_at_50']*100:5.1f}%  "
            f"MRR={metrics['mrr']:.3f}  medRank={metrics['median_rank']}"
        )

    # A×Bの結果を読んで比較プロット
    ab_path = config.RESULTS_DIR / "db_scaling.json"
    ab_results = None
    if ab_path.exists():
        with open(ab_path) as f:
            ab_data = json.load(f)
        ab_results = ab_data["results"]

    # テーブル
    random_baselines = []
    print("\n" + "=" * 90)
    print("  DB規模スケーリング結果 (A×C, FPS+poly+z-score, K=500)")
    print("=" * 90)
    print(f"{'DB size':>10} {'A×C R@1':>9} {'A×B R@1':>9} {'A×C MRR':>9} {'medRank':>8} {'vs random':>10}")
    print("-" * 60)
    for r in results:
        expected_r1 = 1.0 / r["db_size"]
        random_baselines.append({"db_size": r["db_size"], "expected_recall_at_1": expected_r1})
        ratio = r["recall_at_1"] / expected_r1
        ab_r1 = ""
        if ab_results:
            ab_match = next((x for x in ab_results if x["db_size"] == r["db_size"]), None)
            if ab_match:
                ab_r1 = f"{ab_match['recall_at_1']*100:7.1f}%"
        print(
            f"{r['db_size']:>10,} "
            f"{r['recall_at_1']*100:>8.1f}% "
            f"{ab_r1:>9} "
            f"{r['mrr']:>9.3f} "
            f"{r['median_rank']:>8} "
            f"{ratio:>9.0f}x"
        )

    # プロット（A×BとA×Cの比較）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    db_sizes_ac = [r["db_size"] for r in results]
    r1_ac = [r["recall_at_1"] * 100 for r in results]
    mrr_ac = [r["mrr"] for r in results]

    ax = axes[0]
    ax.semilogx(db_sizes_ac, r1_ac, "s-", color="#4CAF50", linewidth=2, markersize=8, label="A×C (MiniLM×BGE-small)")
    if ab_results:
        db_sizes_ab = [r["db_size"] for r in ab_results]
        r1_ab = [r["recall_at_1"] * 100 for r in ab_results]
        ax.semilogx(db_sizes_ab, r1_ab, "o--", color="#2196F3", linewidth=2, markersize=8, label="A×B (MiniLM×E5-large)")
    ax.set_xlabel("Database Size", fontsize=12)
    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("Recall@1 vs Database Scale", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(db_sizes_ac, mrr_ac, "s-", color="#4CAF50", linewidth=2, markersize=8, label="A×C")
    if ab_results:
        mrr_ab = [r["mrr"] for r in ab_results]
        ax.semilogx(db_sizes_ab, mrr_ab, "o--", color="#2196F3", linewidth=2, markersize=8, label="A×B")
    ax.set_xlabel("Database Size", fontsize=12)
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title("MRR vs Database Scale", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.RESULTS_DIR / "db_scaling_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nグラフ保存: {save_path}")

    elapsed = time.time() - start_time
    output = {
        "pair": "A×C",
        "protocol": "FPS+poly+z-score",
        "num_queries": NUM_QUERIES,
        "num_anchors": NUM_ANCHORS,
        "db_sizes": DB_SIZES,
        "results": results,
        "random_baselines": random_baselines,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "db_scaling_ac.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
