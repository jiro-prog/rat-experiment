"""
RAT Phase 4 Step 3c: K=4000スケーリング（テーブル補完用）

COCO Karpathy test=5000件 → アンカー4000 + クエリ500 + バッファ500
A×E (MiniLM × CLIP-image) baseline のみ。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative
from src.evaluator import evaluate_retrieval
from run_phase4_step2 import load_coco_pairs, compute_anchor_sim_stats

CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"
NUM_QUERIES = 500
MAX_ANCHORS = 4000
ANCHOR_COUNTS = [4000]


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_needed = MAX_ANCHORS + NUM_QUERIES

    print("=" * 60)
    print(f"Phase 4 Step 3c: K={MAX_ANCHORS} スケーリング")
    print("=" * 60)

    print(f"\n--- データ準備: COCO {total_needed}組 ---")
    all_pairs = load_coco_pairs(total_needed, offset=0, seed=config.RANDOM_SEED)
    print(f"  取得: {len(all_pairs)}組")

    anchor_pairs = all_pairs[:MAX_ANCHORS]
    query_pairs = all_pairs[MAX_ANCHORS:MAX_ANCHORS + NUM_QUERIES]

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    print("\n--- Embedding ---")

    print("  MiniLM...")
    anchor_emb_ml = embed_texts(config.MODEL_A, anchor_captions)
    query_emb_ml = embed_texts(config.MODEL_A, query_captions)

    print("  CLIP-image...")
    anchor_emb_ci = embed_images_clip(anchor_images, CLIP_VISION_MODEL)
    query_emb_ci = embed_images_clip(query_images, CLIP_VISION_MODEL)

    print("  CLIP-text...")
    anchor_emb_ct = embed_texts(config.MODEL_D, anchor_captions)

    results = {}

    for K in ANCHOR_COUNTS:
        print(f"\n--- K={K} ---")
        fps_idx, _ = select_anchors_fps(anchor_emb_ct[:K], anchor_captions[:K], K)

        a_ml = anchor_emb_ml[:K][fps_idx]
        a_ci = anchor_emb_ci[:K][fps_idx]

        rel_ml = to_relative(query_emb_ml, a_ml, kernel="poly", degree=2, coef0=1.0)
        rel_ci = to_relative(query_emb_ci, a_ci, kernel="poly", degree=2, coef0=1.0)

        # A×E baseline (MiniLM→CLIPimg)
        m = evaluate_retrieval(rel_ml, rel_ci)
        print(f"  A×E baseline: R@1={m['recall_at_1']*100:.1f}%, R@10={m['recall_at_10']*100:.1f}%, MRR={m['mrr']:.3f}")

        # 逆方向
        m_rev = evaluate_retrieval(rel_ci, rel_ml)
        print(f"  A×E reverse:  R@1={m_rev['recall_at_1']*100:.1f}%, R@10={m_rev['recall_at_10']*100:.1f}%, MRR={m_rev['mrr']:.3f}")

        stats_ml = compute_anchor_sim_stats(a_ml, f"MiniLM (K={K})")
        stats_ci = compute_anchor_sim_stats(a_ci, f"CLIP-image (K={K})")

        results[f"K={K}"] = {
            "AxE_baseline": m,
            "AxE_reverse": m_rev,
            "anchor_stats": {"minilm": stats_ml, "clip_img": stats_ci},
        }

    elapsed = time.time() - start_time

    output = {"results": results, "elapsed_seconds": elapsed}
    out_path = config.RESULTS_DIR / "phase4_step3c.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
