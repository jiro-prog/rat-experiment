"""
実験 5a: DB規模スケーリング

500クエリに対して DB=500/1k/5k/10k/50k/100k でRecall@1の劣化カーブを測定。
distractorは BeIR/msmarco corpus から取得（フォールバック: AllNLI triplet）。
FPS+poly+z-score プロトコルで A×B ペア。

論文 §4 の Table + Figure 用。
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

# DB規模のバリエーション
DB_SIZES = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
NUM_QUERIES = 500
NUM_ANCHORS = 500  # C-2で飽和確認済み
CACHE_DIR = config.DATA_DIR / "db_scaling_cache"


def load_distractor_texts(n: int) -> list[str]:
    """BeIR/msmarco corpus から n 件のユニークパッセージを取得。失敗時は AllNLI フォールバック。"""
    try:
        return _load_from_beir_msmarco(n)
    except Exception as e:
        print(f"BeIR/msmarco 取得失敗: {e}")
        print("フォールバック: AllNLI triplet からユニーク文を収集")
        return _load_from_allnli(n)


def _load_from_beir_msmarco(n: int) -> list[str]:
    from datasets import load_dataset

    print(f"BeIR/msmarco corpus からパッセージ取得中 (目標: {n}件)...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
    texts = []
    seen = set()
    for row in ds:
        t = row["text"].strip()
        if t and t not in seen:
            seen.add(t)
            texts.append(t)
            if len(texts) >= n:
                break
    if len(texts) < n:
        raise ValueError(f"パッセージ不足: {len(texts)} < {n}")
    print(f"  取得完了: {len(texts)}件")
    return texts


def _load_from_allnli(n: int) -> list[str]:
    from datasets import load_dataset

    print(f"AllNLI からユニーク文を収集中 (目標: {n}件)...")
    ds = load_dataset(
        "sentence-transformers/all-nli", "triplet", split="train",
        trust_remote_code=True,
    )
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
    """embeddingをキャッシュ付きで計算。"""
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


def evaluate_at_db_size(
    query_rel: np.ndarray,
    db_rel: np.ndarray,
    db_size: int,
    num_queries: int,
) -> dict:
    """
    query_rel: (num_queries, K) クエリの相対表現
    db_rel: (N_total, K) DB全体の相対表現。先頭 num_queries 件が正解。
    db_size: 評価に使うDB件数（先頭 num_queries 件 + distractor）

    db_rel[:num_queries] がクエリと1:1対応する正解。
    db_rel[num_queries:db_size] がdistractor。
    """
    db_subset = db_rel[:db_size]  # (db_size, K)
    sim_matrix = cosine_similarity(query_rel, db_subset)  # (num_queries, db_size)

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
    print("実験 5a: DB規模スケーリング")
    print(f"  クエリ: {NUM_QUERIES}, アンカー: {NUM_ANCHORS}")
    print(f"  DB規模: {DB_SIZES}")
    print("=" * 60)

    max_db = max(DB_SIZES)

    # --- Step 1: クエリとアンカー候補の準備 ---
    candidates, queries = sample_anchors_and_queries(
        num_anchors=max(NUM_ANCHORS + NUM_QUERIES, 2000),
        num_queries=NUM_QUERIES,
    )
    anchor_candidates = candidates  # FPS選定用の候補プール

    # --- Step 2: distractor テキストの取得 ---
    # DB = queries(500) + distractors で最大 max_db 件
    num_distractors = max_db - NUM_QUERIES
    distractor_texts = load_distractor_texts(num_distractors)

    # DB全体: 先頭500件 = クエリと同一テキスト, 残り = distractor
    db_texts = queries + distractor_texts[:num_distractors]
    print(f"DB構成: クエリ一致={NUM_QUERIES}件 + distractor={len(distractor_texts[:num_distractors])}件 = {len(db_texts)}件")

    # --- Step 3: Embedding ---
    print("\nEmbedding計算...")
    # アンカー候補
    anchor_cand_emb_A = embed_with_cache(config.MODEL_A, anchor_candidates, "anchor_cand_A")
    anchor_cand_emb_B = embed_with_cache(config.MODEL_B, anchor_candidates, "anchor_cand_B")

    # クエリ
    query_emb_A = embed_with_cache(config.MODEL_A, queries, "query_A")

    # DB全体（Model Bで埋め込み — クエリはModel A, DBはModel B）
    db_emb_B = embed_with_cache(config.MODEL_B, db_texts, "db_B")

    # --- Step 4: FPSアンカー選定 ---
    print(f"\nFPSアンカー選定 (K={NUM_ANCHORS})...")
    dummy_texts = [f"dummy_{i}" for i in range(len(anchor_candidates))]
    fps_indices, _ = select_anchors_fps(anchor_cand_emb_A, dummy_texts, NUM_ANCHORS)
    anchor_emb_A = anchor_cand_emb_A[fps_indices]
    anchor_emb_B = anchor_cand_emb_B[fps_indices]

    # --- Step 5: 相対表現の計算 ---
    print("\n相対表現計算...")
    rel_query = to_relative(query_emb_A, anchor_emb_A, kernel="poly", degree=2, coef0=1.0)
    rel_query = normalize_zscore(rel_query)
    print(f"  クエリ相対表現: {rel_query.shape}")

    rel_db = to_relative(db_emb_B, anchor_emb_B, kernel="poly", degree=2, coef0=1.0)
    rel_db = normalize_zscore(rel_db)
    print(f"  DB相対表現: {rel_db.shape}")

    # --- Step 6: 各DB規模で評価 ---
    print("\n評価...")
    results = []
    for db_size in DB_SIZES:
        if db_size > len(db_texts):
            print(f"  DB={db_size}: スキップ (テキスト不足: {len(db_texts)}件)")
            continue
        metrics = evaluate_at_db_size(rel_query, rel_db, db_size, NUM_QUERIES)
        results.append(metrics)
        print(
            f"  DB={db_size:>7,}: R@1={metrics['recall_at_1']*100:5.1f}%  "
            f"R@10={metrics['recall_at_10']*100:5.1f}%  "
            f"R@50={metrics['recall_at_50']*100:5.1f}%  "
            f"MRR={metrics['mrr']:.3f}  medRank={metrics['median_rank']}"
        )

    # --- ランダムベースライン（参考値） ---
    print("\nランダムベースライン計算...")
    random_baselines = []
    for db_size in DB_SIZES:
        if db_size > len(db_texts):
            continue
        expected_r1 = 1.0 / db_size
        random_baselines.append({
            "db_size": db_size,
            "expected_recall_at_1": expected_r1,
        })
        print(f"  DB={db_size:>7,}: ランダム期待R@1={expected_r1*100:.4f}%  RAT比={results[len(random_baselines)-1]['recall_at_1']/expected_r1:.0f}倍")

    # --- 結果テーブル ---
    print("\n" + "=" * 90)
    print("  DB規模スケーリング結果 (A×B, FPS+poly+z-score, K=500)")
    print("=" * 90)
    print(f"{'DB size':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'MRR':>8} {'medRank':>8} {'vs random':>10}")
    print("-" * 75)
    for r, rb in zip(results, random_baselines):
        ratio = r["recall_at_1"] / rb["expected_recall_at_1"]
        print(
            f"{r['db_size']:>10,} "
            f"{r['recall_at_1']*100:>7.1f}% "
            f"{r['recall_at_5']*100:>7.1f}% "
            f"{r['recall_at_10']*100:>7.1f}% "
            f"{r['recall_at_50']*100:>7.1f}% "
            f"{r['mrr']:>8.3f} "
            f"{r['median_rank']:>8} "
            f"{ratio:>9.0f}x"
        )

    # --- プロット ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    db_sizes_actual = [r["db_size"] for r in results]
    r1_vals = [r["recall_at_1"] * 100 for r in results]
    mrr_vals = [r["mrr"] for r in results]

    # Recall@1 vs DB size (log scale)
    ax = axes[0]
    ax.semilogx(db_sizes_actual, r1_vals, "s-", color="#2196F3", linewidth=2, markersize=8, label="RAT (FPS+poly+z-score)")
    ax.set_xlabel("Database Size", fontsize=12)
    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("Recall@1 vs Database Scale", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # MRR vs DB size (log scale)
    ax = axes[1]
    ax.semilogx(db_sizes_actual, mrr_vals, "s-", color="#FF9800", linewidth=2, markersize=8, label="RAT (FPS+poly+z-score)")
    ax.set_xlabel("Database Size", fontsize=12)
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title("MRR vs Database Scale", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.RESULTS_DIR / "db_scaling.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nグラフ保存: {save_path}")

    # --- JSON保存 ---
    elapsed = time.time() - start_time
    output = {
        "pair": "A×B",
        "protocol": "FPS+poly+z-score",
        "num_queries": NUM_QUERIES,
        "num_anchors": NUM_ANCHORS,
        "db_sizes": DB_SIZES,
        "results": results,
        "random_baselines": random_baselines,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "db_scaling.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
