"""
実験 5b: Moschella et al. との直接比較

先行研究のデフォルト設定（ランダムアンカー + cosine類似度）を再現し、
RATプロトコル（FPS + poly + z-score）の各コンポーネントの改善幅を定量化。

比較する手法:
  1. Moschella (random + cosine)           — 先行研究の再現
  2. +FPS (FPS + cosine)                    — アンカー選定の効果
  3. +poly (FPS + poly)                     — カーネルの効果
  4. +z-score (FPS + poly + z-score)        — 正規化の効果（=最終プロトコル）

2データセット (STS, AllNLI) × 3モデルペア × K=[100, 200, 500]

論文 §4 Table 1 拡張用。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.evaluator import evaluate_retrieval

# --- 手法の自前定義（Moschella再現のため、既存コードの正規化が混入しないようにする） ---

def relative_cosine(embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Moschella et al. のデフォルト: コサイン類似度ベースの相対表現。"""
    return embeddings @ anchors.T  # L2正規化済み前提


def relative_poly(embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """多項式カーネル: (x·a + 1)^2"""
    dot = embeddings @ anchors.T
    return (dot + 1.0) ** 2


def zscore_normalize(rel: np.ndarray) -> np.ndarray:
    """z-score正規化。"""
    mean = rel.mean(axis=1, keepdims=True)
    std = rel.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (rel - mean) / std


# --- 手法定義 ---

METHODS = {
    "moschella": {
        "label": "Moschella (random+cosine)",
        "anchor": "random",
        "kernel": "cosine",
        "normalize": False,
    },
    "+fps": {
        "label": "+FPS (FPS+cosine)",
        "anchor": "fps",
        "kernel": "cosine",
        "normalize": False,
    },
    "+poly": {
        "label": "+poly (FPS+poly)",
        "anchor": "fps",
        "kernel": "poly",
        "normalize": False,
    },
    "+zscore": {
        "label": "+z-score (FPS+poly+z-score)",
        "anchor": "fps",
        "kernel": "poly",
        "normalize": True,
    },
}

K_VALUES = [100, 200, 500]
CANDIDATE_POOL = 2000

MODEL_PAIRS = [
    ("A×B", config.MODEL_A, config.MODEL_B),
    ("A×C", config.MODEL_A, config.MODEL_C),
    ("B×C", config.MODEL_B, config.MODEL_C),
]


def load_allnli_sentences(n: int = 5000, seed: int = config.RANDOM_SEED) -> list[str]:
    """AllNLI triplet からユニーク文を取得。"""
    from datasets import load_dataset
    import random

    ds = load_dataset(
        "sentence-transformers/all-nli", "triplet", split="dev",
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
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n]


def run_method(
    method_cfg: dict,
    query_emb_1: np.ndarray,
    query_emb_2: np.ndarray,
    cand_emb_1: np.ndarray,
    cand_emb_2: np.ndarray,
    fps_indices: list[int],
    K: int,
    rng: np.random.RandomState,
) -> dict:
    """1手法 × 1ペア × 1K の評価を実行。"""
    # アンカー選定
    if method_cfg["anchor"] == "random":
        indices = rng.choice(len(cand_emb_1), size=K, replace=False)
    else:  # fps
        indices = fps_indices[:K]

    anchor_1 = cand_emb_1[indices]
    anchor_2 = cand_emb_2[indices]

    # カーネル
    if method_cfg["kernel"] == "cosine":
        rel_1 = relative_cosine(query_emb_1, anchor_1)
        rel_2 = relative_cosine(query_emb_2, anchor_2)
    else:  # poly
        rel_1 = relative_poly(query_emb_1, anchor_1)
        rel_2 = relative_poly(query_emb_2, anchor_2)

    # 正規化
    if method_cfg["normalize"]:
        rel_1 = zscore_normalize(rel_1)
        rel_2 = zscore_normalize(rel_2)

    return evaluate_retrieval(rel_1, rel_2)


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("実験 5b: Moschella et al. 比較 (ablation)")
    print("=" * 60)

    # --- データセット準備 ---
    datasets = {}

    # STS
    print("\n[STS] データ準備...")
    sts_candidates, sts_queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES,
    )
    datasets["STS"] = (sts_candidates, sts_queries)

    # AllNLI
    print("\n[AllNLI] データ準備...")
    allnli_all = load_allnli_sentences(CANDIDATE_POOL + config.NUM_QUERIES)
    allnli_candidates = allnli_all[:CANDIDATE_POOL]
    allnli_queries = allnli_all[CANDIDATE_POOL:CANDIDATE_POOL + config.NUM_QUERIES]
    datasets["AllNLI"] = (allnli_candidates, allnli_queries)

    all_results = []

    for ds_name, (candidates, queries) in datasets.items():
        print(f"\n{'='*60}")
        print(f"データセット: {ds_name} (候補={len(candidates)}, クエリ={len(queries)})")
        print(f"{'='*60}")

        for pair_name, model_1, model_2 in MODEL_PAIRS:
            print(f"\n--- {pair_name} ---")

            # Embedding
            cand_emb_1 = embed_texts(model_1, candidates)
            cand_emb_2 = embed_texts(model_2, candidates)
            query_emb_1 = embed_texts(model_1, queries)
            query_emb_2 = embed_texts(model_2, queries)

            # FPS（Model 1空間で選定）
            max_k = max(K_VALUES)
            dummy_texts = [f"d{i}" for i in range(len(candidates))]
            fps_indices, _ = select_anchors_fps(cand_emb_1, dummy_texts, max_k)

            rng = np.random.RandomState(config.RANDOM_SEED)

            for K in K_VALUES:
                row = {"dataset": ds_name, "pair": pair_name, "K": K}
                print(f"\n  K={K}:")

                for method_key, method_cfg in METHODS.items():
                    # ランダム手法は同じrngで再現性確保
                    method_rng = np.random.RandomState(config.RANDOM_SEED + K)
                    metrics = run_method(
                        method_cfg, query_emb_1, query_emb_2,
                        cand_emb_1, cand_emb_2, fps_indices, K, method_rng,
                    )
                    row[method_key] = metrics
                    print(
                        f"    {method_cfg['label']:>35}: "
                        f"R@1={metrics['recall_at_1']*100:5.1f}%  "
                        f"R@10={metrics['recall_at_10']*100:5.1f}%  "
                        f"MRR={metrics['mrr']:.3f}"
                    )

                all_results.append(row)

    # --- 改善幅サマリー ---
    print("\n" + "=" * 90)
    print("  コンポーネント別改善幅 (Recall@1, K=500)")
    print("=" * 90)
    print(f"{'Dataset':>8} {'Pair':>5} {'Moschella':>10} {'+FPS':>8} {'+poly':>8} {'+z-score':>10} {'Total Δ':>8}")
    print("-" * 60)
    for r in all_results:
        if r["K"] != 500:
            continue
        m = r["moschella"]["recall_at_1"] * 100
        f = r["+fps"]["recall_at_1"] * 100
        p = r["+poly"]["recall_at_1"] * 100
        z = r["+zscore"]["recall_at_1"] * 100
        print(
            f"{r['dataset']:>8} {r['pair']:>5} "
            f"{m:>9.1f}% "
            f"{f-m:>+7.1f}% "
            f"{p-f:>+7.1f}% "
            f"{z-p:>+9.1f}% "
            f"{z-m:>+7.1f}%"
        )

    # --- JSON保存 ---
    elapsed = time.time() - start_time
    output = {
        "methods": list(METHODS.keys()),
        "K_values": K_VALUES,
        "model_pairs": [(n, m1, m2) for n, m1, m2 in MODEL_PAIRS],
        "datasets": list(datasets.keys()),
        "results": all_results,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "baseline_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
