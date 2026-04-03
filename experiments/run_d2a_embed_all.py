"""
Direction 2 実験群A Step 1: 全12モデルのembedding一括取得

STSBenchmarkのテキスト（アンカー候補2000件 + クエリ500件）を
全12モデルでembedし、.npyで保存する。

これが全ペアマトリクス計算の前提となるボトルネックステップ。
モデルダウンロード + GPU computeで時間がかかるため、先に回しておく。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from src.anchor_sampler import sample_anchors_and_queries
from src.embedder import embed_texts

# 設定
CANDIDATE_POOL = 2000
NUM_QUERIES = 500
OUT_DIR = config.DATA_DIR / "d2_matrix"


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================
    # データサンプリング（全モデル共通）
    # ========================================
    print("=" * 60)
    print("Direction 2A: 全モデルembedding一括取得")
    print("=" * 60)

    candidates, queries = sample_anchors_and_queries(
        num_anchors=CANDIDATE_POOL, num_queries=NUM_QUERIES
    )
    print(f"候補プール: {len(candidates)}件, クエリ: {len(queries)}件")

    # テキストを保存（再現性のため）
    texts_path = OUT_DIR / "texts.json"
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump({"candidates": candidates, "queries": queries}, f, ensure_ascii=False)
    print(f"テキスト保存: {texts_path}")

    # ========================================
    # 全モデルでembedding
    # ========================================
    all_texts = candidates + queries  # 一括embed、後でスライス
    n_cand = len(candidates)

    completed = []
    failed = []

    for label in sorted(config.MATRIX_MODELS.keys()):
        info = config.MATRIX_MODELS[label]
        model_name = info["name"]
        short = model_name.split("/")[-1]

        # 既にembedが存在する場合はスキップ
        cand_path = OUT_DIR / f"cand_{label}.npy"
        query_path = OUT_DIR / f"query_{label}.npy"
        if cand_path.exists() and query_path.exists():
            print(f"\n[SKIP] Model {label} ({short}): 既存embedを使用")
            completed.append(label)
            continue

        print(f"\n{'='*60}")
        print(f"Model {label}: {short}")
        print(f"  Family={info['family']}, Params={info['params']}, "
              f"Dim={info['dim']}, Training={info['training']}, Lang={info['lang']}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            embs = embed_texts(model_name, all_texts)
            cand_embs = embs[:n_cand]
            query_embs = embs[n_cand:]

            np.save(cand_path, cand_embs)
            np.save(query_path, query_embs)

            elapsed_model = time.time() - t0
            print(f"  → cand={cand_embs.shape}, query={query_embs.shape} ({elapsed_model:.1f}秒)")
            completed.append(label)

        except Exception as e:
            elapsed_model = time.time() - t0
            print(f"  [ERROR] {e} ({elapsed_model:.1f}秒)")
            failed.append({"label": label, "model": model_name, "error": str(e)})

    # ========================================
    # サマリー
    # ========================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完了: {len(completed)}/{len(config.MATRIX_MODELS)}モデル ({elapsed:.1f}秒)")
    if failed:
        print(f"失敗: {len(failed)}モデル")
        for f_info in failed:
            print(f"  {f_info['label']}: {f_info['model']} — {f_info['error']}")
    print(f"{'='*60}")

    # メタデータ保存
    meta = {
        "candidate_pool": CANDIDATE_POOL,
        "num_queries": NUM_QUERIES,
        "models": {
            label: {
                "name": info["name"],
                "family": info["family"],
                "params": info["params"],
                "dim": info["dim"],
                "training": info["training"],
                "lang": info["lang"],
            }
            for label, info in config.MATRIX_MODELS.items()
        },
        "completed": completed,
        "failed": [f["label"] for f in failed],
        "elapsed_seconds": elapsed,
    }
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"メタデータ保存: {meta_path}")


if __name__ == "__main__":
    main()
