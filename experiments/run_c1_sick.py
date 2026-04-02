"""
実験 C-1: 別データセット（AllNLI）でのRAT再現実験

STSBenchmark以外のデータセットで同じプロトコル（FPS+poly+z-score）が
機能することを確認する。

データ: sentence-transformers/all-nli (pair-score, test split)
条件: FPS(K=500) + poly(degree=2, coef0=1.0) + z-score
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset

import config
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K = 500
CANDIDATE_POOL = 2000
NUM_QUERIES = 500

MODELS = {
    "A": config.MODEL_A,
    "B": config.MODEL_B,
    "C": config.MODEL_C,
}

PAIRS = [
    ("A×B", "A", "B"),
    ("A×C", "A", "C"),
    ("B×C", "B", "C"),
]

# Phase 3のSTSBenchmark結果（比較用）
STS_RESULTS = {
    "A×B": 0.762,
    "A×C": 0.980,
    "B×C": 0.640,
}


def load_allnli_sentences(seed: int = 42) -> list[str]:
    """AllNLIからユニークな文を抽出する。"""
    ds = load_dataset("sentence-transformers/all-nli", "pair-score", split="test")
    sentences = set()
    for row in ds:
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])
    sentences = sorted(sentences)
    print(f"AllNLI test: {len(ds)}ペア → {len(sentences)}ユニーク文")
    return sentences


def sample_candidates_and_queries(
    sentences: list[str], num_candidates: int, num_queries: int, seed: int = 42
) -> tuple[list[str], list[str]]:
    """候補とクエリを重複なしでサンプリングする。"""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(sentences))
    total = num_candidates + num_queries
    assert total <= len(sentences), f"文が足りない: {len(sentences)} < {total}"
    selected = indices[:total]
    candidates = [sentences[i] for i in selected[:num_candidates]]
    queries = [sentences[i] for i in selected[num_candidates:total]]
    return candidates, queries


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 C-1: AllNLIでのRAT再現")
    print("=" * 60)

    # データ準備
    sentences = load_allnli_sentences()
    candidates, queries = sample_candidates_and_queries(
        sentences, CANDIDATE_POOL, NUM_QUERIES
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

    # FPS + poly + z-score で検索
    print("\n" + "=" * 60)
    print("FPS + poly + z-score")
    print("=" * 60)

    results = []
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
            f"  {pair_label}: Recall@1={metrics['recall_at_1']*100:.1f}%, "
            f"R@5={metrics['recall_at_5']*100:.1f}%, "
            f"R@10={metrics['recall_at_10']*100:.1f}%, "
            f"MRR={metrics['mrr']:.3f}"
        )

    # 比較テーブル
    print("\n" + "=" * 80)
    print("  STSBenchmark vs AllNLI 比較")
    print("=" * 80)
    print(f"\n{'ペア':<8} {'STS R@1':>10} {'AllNLI R@1':>12} {'差分':>8}")
    print("-" * 45)
    for r in results:
        pair = r["pair"]
        sts = STS_RESULTS[pair]
        nli = r["recall_at_1"]
        diff = nli - sts
        print(
            f"{pair:<8} {sts*100:>9.1f}% {nli*100:>11.1f}% {diff*100:>+7.1f}%"
        )

    # 判定
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)
    all_pass = all(r["recall_at_1"] > 0.3 for r in results)
    for r in results:
        status = "✓" if r["recall_at_1"] > 0.3 else "✗"
        print(f"  {r['pair']}: {r['recall_at_1']*100:.1f}% {status}")
    if all_pass:
        print("\n  → 全ペアで30%超。RATプロトコルはデータセット非依存で機能する")
    else:
        print("\n  → 一部ペアが基準未達。データ特性の影響を分析する必要あり")

    # 保存
    elapsed = time.time() - start_time
    output = {
        "dataset": "sentence-transformers/all-nli",
        "split": "test",
        "num_candidates": CANDIDATE_POOL,
        "num_queries": NUM_QUERIES,
        "K": K,
        "method": "FPS + poly(d=2,c=1) + z-score",
        "results": results,
        "sts_reference": STS_RESULTS,
        "all_pass_30pct": all_pass,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "c1_allnli.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
