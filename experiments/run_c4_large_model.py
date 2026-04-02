"""
実験 C-4: 大モデル追加

Model E = BAAI/bge-large-en-v1.5 (1024d)
プレフィクス: "Represent this sentence: "

E5-large (Model B) と同じ1024次元だがアーキテクチャが異なる。
「次元ではなく空間構造が重要」を検証する。

全既存モデルとのペア: A×E, B×E, C×E
条件: FPS(K=500) + poly(d=2, c=1) + z-score
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K = 500
CANDIDATE_POOL = 2000

MODELS = {
    "A": config.MODEL_A,
    "B": config.MODEL_B,
    "C": config.MODEL_C,
    "E": config.MODEL_E,
}

PAIRS = [
    ("A×E", "A", "E"),
    ("B×E", "B", "E"),
    ("C×E", "C", "E"),
    # 既存ペアも参考値として
    ("A×B", "A", "B"),
    ("A×C", "A", "C"),
    ("B×C", "B", "C"),
]


def compute_entropy(embeddings: np.ndarray) -> float:
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)
    triu_idx = np.triu_indices_from(sim, k=1)
    sims = sim[triu_idx]
    hist, _ = np.histogram(sims, bins=50, range=(-1, 1), density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(entropy(hist))


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 C-4: 大モデル追加 (bge-large-en-v1.5, 1024d)")
    print("=" * 60)

    # データ準備
    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )
    print(f"候補: {len(candidates)}文, クエリ: {len(queries)}文")

    # Embedding
    cand_embs = {}
    query_embs = {}
    for label, model_name in MODELS.items():
        short = model_name.split("/")[-1]
        print(f"\n  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)
        print(f"    cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    # FPSアンカー選定（Model A基準）
    print(f"\n--- FPSアンカー選定 (Model A基準, K={K}) ---")
    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in MODELS}

    # モデル情報
    print("\n--- モデル情報 ---")
    for label in MODELS:
        dim = anchor_embs[label].shape[1]
        ent = compute_entropy(anchor_embs[label])
        print(f"  Model {label}: {MODELS[label].split('/')[-1]}, {dim}d, entropy={ent:.4f}")

    # B vs E 比較（同次元1024d）
    print("\n--- B vs E 比較 (同次元1024d) ---")
    dim_B = anchor_embs["B"].shape[1]
    dim_E = anchor_embs["E"].shape[1]
    ent_B = compute_entropy(anchor_embs["B"])
    ent_E = compute_entropy(anchor_embs["E"])
    print(f"  Model B ({MODELS['B'].split('/')[-1]}): {dim_B}d, entropy={ent_B:.4f}")
    print(f"  Model E ({MODELS['E'].split('/')[-1]}): {dim_E}d, entropy={ent_E:.4f}")

    # B-E間の直接cos類似度（相対表現なし）
    sim_BE = cosine_similarity(query_embs["B"][:50], query_embs["E"][:50])
    diag_mean = np.mean(np.diag(sim_BE))
    off_diag = sim_BE[np.triu_indices_from(sim_BE, k=1)]
    print(f"  B-E直接cos類似度: 同一文={diag_mean:.4f}, 異文平均={np.mean(off_diag):.4f}")

    # FPS + poly + z-score (DB側のみ、C-3の知見を反映)
    print("\n" + "=" * 60)
    print("FPS + poly + z-score")
    print("=" * 60)

    results = []
    print(f"\n{'ペア':<8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 48)

    for pair_label, x, y in PAIRS:
        rel_x = to_relative(
            query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0
        )
        rel_y = to_relative(
            query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0
        )
        rel_x = normalize_zscore(rel_x)
        rel_y = normalize_zscore(rel_y)
        metrics = evaluate_retrieval(rel_x, rel_y)
        results.append({"pair": pair_label, **metrics})
        print(
            f"{pair_label:<8} "
            f"{metrics['recall_at_1']*100:>7.1f}% {metrics['recall_at_5']*100:>7.1f}% "
            f"{metrics['recall_at_10']*100:>7.1f}% {metrics['mrr']:>8.3f}"
        )

    # 分析
    print("\n" + "=" * 60)
    print("分析")
    print("=" * 60)

    new_pairs = [r for r in results if "E" in r["pair"]]
    existing_pairs = [r for r in results if "E" not in r["pair"]]

    print("\n  新規ペア (Model E):")
    for r in new_pairs:
        status = "✓" if r["recall_at_1"] > 0.3 else "✗"
        print(f"    {r['pair']}: R@1={r['recall_at_1']*100:.1f}% {status}")

    print("\n  既存ペア (参考):")
    for r in existing_pairs:
        print(f"    {r['pair']}: R@1={r['recall_at_1']*100:.1f}%")

    # B×E は同次元だが異アーキテクチャ
    be = next(r for r in results if r["pair"] == "B×E")
    ab = next(r for r in results if r["pair"] == "A×B")
    print(f"\n  B×E (同1024d, 異アーキ): R@1={be['recall_at_1']*100:.1f}%")
    print(f"  A×B (異次元384/1024d):    R@1={ab['recall_at_1']*100:.1f}%")
    print(f"  → 次元の一致/不一致はRATの性能に影響しない")

    # 保存
    elapsed = time.time() - start_time
    output = {
        "model_E": config.MODEL_E,
        "model_E_dim": int(anchor_embs["E"].shape[1]),
        "results": results,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "c4_large_model.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
