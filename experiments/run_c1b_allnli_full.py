"""
実験 C-1b: AllNLIでの全ペア計測（Table 1の空欄埋め）

C-1ではA×B, A×C, B×Cのみ。Model Eを追加して
A×E, B×E, C×Eも計測する。
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
    "E": config.MODEL_E,
}

PAIRS = [
    ("A×B", "A", "B"),
    ("A×C", "A", "C"),
    ("A×E", "A", "E"),
    ("B×C", "B", "C"),
    ("B×E", "B", "E"),
    ("C×E", "C", "E"),
]


def load_allnli_sentences() -> list[str]:
    ds = load_dataset("sentence-transformers/all-nli", "pair-score", split="test")
    sentences = set()
    for row in ds:
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])
    sentences = sorted(sentences)
    print(f"AllNLI test: {len(sentences)}ユニーク文")
    return sentences


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("C-1b: AllNLI 全ペア計測")
    print("=" * 60)

    sentences = load_allnli_sentences()
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(sentences))
    total = CANDIDATE_POOL + NUM_QUERIES
    candidates = [sentences[i] for i in indices[:CANDIDATE_POOL]]
    queries = [sentences[i] for i in indices[CANDIDATE_POOL:total]]
    print(f"候補: {len(candidates)}文, クエリ: {len(queries)}文")

    cand_embs = {}
    query_embs = {}
    for label, model_name in MODELS.items():
        short = model_name.split("/")[-1]
        print(f"\n  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)
        print(f"    cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    print(f"\n--- FPSアンカー選定 (Model A基準, K={K}) ---")
    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in MODELS}

    print("\n" + "=" * 60)
    print("FPS + poly + z-score")
    print("=" * 60)

    results = []
    print(f"\n{'ペア':<8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 48)

    for pair_label, x, y in PAIRS:
        rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)
        rel_x = normalize_zscore(rel_x)
        rel_y = normalize_zscore(rel_y)
        metrics = evaluate_retrieval(rel_x, rel_y)
        results.append({"pair": pair_label, **metrics})
        print(
            f"{pair_label:<8} "
            f"{metrics['recall_at_1']*100:>7.1f}% {metrics['recall_at_5']*100:>7.1f}% "
            f"{metrics['recall_at_10']*100:>7.1f}% {metrics['mrr']:>8.3f}"
        )

    elapsed = time.time() - start_time
    output = {
        "dataset": "sentence-transformers/all-nli",
        "results": results,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "c1b_allnli_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
