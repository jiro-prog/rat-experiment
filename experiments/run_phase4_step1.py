"""
RAT Phase 4 Step 1: CLIPテキストエンコーダ vs Sentence-BERT（テキスト同士）

CLIPのテキストエンコーダ（contrastive learning）がRAT相対表現で
Sentence-BERTモデルと互換性を持つか検証する。

Model D = sentence-transformers/clip-ViT-B-32 (512d)
条件: FPS + poly + z-score, K=500
成功基準: いずれかのペアでRecall@1 > 30%
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from transformers import CLIPTokenizer

import config
from src.anchor_sampler import sample_anchors_and_queries, select_anchors_fps
from src.embedder import embed_texts
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

K = 500
CANDIDATE_POOL = 2000

PAIRS = [
    ("A×D", "A", "D"),
    ("B×D", "B", "D"),
    ("C×D", "C", "D"),
]


def check_clip_token_lengths(texts: list[str]) -> dict:
    """CLIPトークナイザで入力テキストのトークン長を確認する。"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    lengths = [len(tokenizer.encode(t)) for t in texts]
    lengths = np.array(lengths)
    over_77 = int(np.sum(lengths > 77))
    return {
        "max_tokens": int(np.max(lengths)),
        "mean_tokens": float(np.mean(lengths)),
        "median_tokens": int(np.median(lengths)),
        "over_77": over_77,
        "over_77_pct": float(over_77 / len(texts) * 100),
        "total": len(texts),
    }


def compute_anchor_sim_stats(embeddings: np.ndarray, label: str) -> dict:
    """アンカー間類似度行列の統計量を計算する。"""
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)
    triu_idx = np.triu_indices_from(sim, k=1)
    sims = sim[triu_idx]

    hist, _ = np.histogram(sims, bins=50, range=(-1, 1), density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    ent = float(entropy(hist))

    return {
        "label": label,
        "mean_sim": float(np.mean(sims)),
        "std_sim": float(np.std(sims)),
        "min_sim": float(np.min(sims)),
        "max_sim": float(np.max(sims)),
        "entropy": ent,
    }


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "A": config.MODEL_A,
        "B": config.MODEL_B,
        "C": config.MODEL_C,
        "D": config.MODEL_D,
    }

    # ========================================
    # データ準備
    # ========================================
    print("=" * 60)
    print("Phase 4 Step 1: CLIP Text Encoder vs Sentence-BERT")
    print("=" * 60)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=config.NUM_QUERIES
    )

    # CLIPトークン長チェック
    print("\n--- CLIPトークン長チェック ---")
    all_texts = list(set(candidates + queries))
    token_stats = check_clip_token_lengths(all_texts)
    print(f"  テキスト数: {token_stats['total']}")
    print(f"  最大トークン長: {token_stats['max_tokens']}")
    print(f"  平均トークン長: {token_stats['mean_tokens']:.1f}")
    print(f"  中央値: {token_stats['median_tokens']}")
    print(f"  77トークン超過: {token_stats['over_77']}件 ({token_stats['over_77_pct']:.1f}%)")
    if token_stats["over_77"] > 0:
        print(f"  ⚠ {token_stats['over_77']}件が77トークンを超過 → CLIPで切り詰めが発生")

    # Embedding
    cand_embs = {}
    query_embs = {}
    for label, model_name in models.items():
        short = model_name.split("/")[-1]
        print(f"\n  Model {label} ({short})...")
        cand_embs[label] = embed_texts(model_name, candidates)
        query_embs[label] = embed_texts(model_name, queries)
        print(f"    cand={cand_embs[label].shape}, query={query_embs[label].shape}")

    # FPSアンカー選定（Model A基準）
    print("\n--- FPSアンカー選定 (Model A基準, K=500) ---")
    fps_indices, _ = select_anchors_fps(cand_embs["A"], candidates, K)
    anchor_embs = {label: cand_embs[label][fps_indices] for label in models}

    # ========================================
    # Model D アンカー間類似度分析
    # ========================================
    print("\n" + "=" * 60)
    print("Model D アンカー間類似度分析")
    print("=" * 60)

    # 全モデルの統計を出す（比較用）
    for label in models:
        stats = compute_anchor_sim_stats(anchor_embs[label], label)
        short = models[label].split("/")[-1]
        dim = anchor_embs[label].shape[1]
        print(f"\n  Model {label} ({short}, {dim}d):")
        print(f"    Mean sim:  {stats['mean_sim']:.4f}")
        print(f"    Std sim:   {stats['std_sim']:.4f}")
        print(f"    Range:     [{stats['min_sim']:.4f}, {stats['max_sim']:.4f}]")
        print(f"    Entropy:   {stats['entropy']:.4f}")

    # ========================================
    # クロスモデル検索 (FPS + poly + z-score)
    # ========================================
    print("\n" + "=" * 60)
    print("クロスモデル検索: FPS + poly + z-score")
    print("=" * 60)

    all_results = []

    # ベースライン（z-scoreなし）
    print("\n--- ベースライン (FPS + poly, z-scoreなし) ---")
    for pair_label, x, y in PAIRS:
        rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)
        metrics = evaluate_retrieval(rel_x, rel_y)
        print(f"  {pair_label}: Recall@1={metrics['recall_at_1']*100:.1f}%, R@10={metrics['recall_at_10']*100:.1f}%, MRR={metrics['mrr']:.3f}")
        all_results.append({
            "method": "baseline",
            "pair": pair_label,
            **metrics,
        })

    # z-score正規化あり
    print("\n--- FPS + poly + z-score ---")
    for pair_label, x, y in PAIRS:
        rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)
        rel_x = normalize_zscore(rel_x)
        rel_y = normalize_zscore(rel_y)
        metrics = evaluate_retrieval(rel_x, rel_y)
        print(f"  {pair_label}: Recall@1={metrics['recall_at_1']*100:.1f}%, R@10={metrics['recall_at_10']*100:.1f}%, MRR={metrics['mrr']:.3f}")
        all_results.append({
            "method": "zscore",
            "pair": pair_label,
            **metrics,
        })

    # 既存ペアも参考値として計測
    print("\n--- 参考: 既存ペア (FPS + poly + z-score) ---")
    existing_pairs = [("A×B", "A", "B"), ("A×C", "A", "C"), ("B×C", "B", "C")]
    for pair_label, x, y in existing_pairs:
        rel_x = to_relative(query_embs[x], anchor_embs[x], kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[y], anchor_embs[y], kernel="poly", degree=2, coef0=1.0)
        rel_x = normalize_zscore(rel_x)
        rel_y = normalize_zscore(rel_y)
        metrics = evaluate_retrieval(rel_x, rel_y)
        print(f"  {pair_label}: Recall@1={metrics['recall_at_1']*100:.1f}%, R@10={metrics['recall_at_10']*100:.1f}%, MRR={metrics['mrr']:.3f}")
        all_results.append({
            "method": "zscore_reference",
            "pair": pair_label,
            **metrics,
        })

    # ========================================
    # 結果テーブル
    # ========================================
    print("\n" + "=" * 100)
    print("  Phase 4 Step 1 結果: CLIP Text Encoder × Sentence-BERT")
    print("=" * 100)

    print(f"\n{'ペア':<10} {'Method':<12} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 70)

    for r in all_results:
        print(
            f"{r['pair']:<10} {r['method']:<12} "
            f"{r['recall_at_1']*100:>9.1f}% {r['recall_at_5']*100:>9.1f}% "
            f"{r['recall_at_10']*100:>9.1f}% {r['mrr']:>8.3f}"
        )

    # ========================================
    # 判定
    # ========================================
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    zscore_results = [r for r in all_results if r["method"] == "zscore"]
    best = max(zscore_results, key=lambda r: r["recall_at_1"])
    best_r1 = best["recall_at_1"]

    print(f"\n  最良ペア: {best['pair']} (Recall@1={best_r1*100:.1f}%)")

    for r in zscore_results:
        status = "✓ PASS" if r["recall_at_1"] > 0.3 else "✗ FAIL"
        print(f"  {r['pair']}: Recall@1={r['recall_at_1']*100:.1f}% → {status} (基準: >30%)")

    if best_r1 > 0.3:
        print(f"\n  → 成功基準達成! CLIPテキストエンコーダとSentence-BERT間でRATが機能する")
        print(f"  → Step 2（テキスト×画像）に進む根拠あり")
    elif best_r1 > 0.1:
        print(f"\n  → 部分的に機能。CLIPテキスト空間の特殊性を追加分析する必要あり")
    else:
        print(f"\n  → CLIPテキスト空間はSentence-BERTと互換性なし。Step 2は困難")

    # 保存
    elapsed = time.time() - start_time

    # Model D のアンカー間統計も保存
    d_stats = compute_anchor_sim_stats(anchor_embs["D"], "D")

    output = {
        "token_stats": token_stats,
        "anchor_sim_stats_D": d_stats,
        "results": all_results,
        "best_pair": best["pair"],
        "best_recall_at_1": best_r1,
        "pass_30pct": best_r1 > 0.3,
        "elapsed_seconds": elapsed,
    }
    out_path = config.RESULTS_DIR / "phase4_step1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
