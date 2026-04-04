"""
D2 Phase 1: 新規モデル埋め込み生成 + 全新規ペア基礎評価

Step 1: O(xs), P(s), Q(l) の埋め込み生成 (2000 cand + 500 query)
Step 2: 新規ペアのRAT R@1/R@5/MRR評価 (K=500)
  - O,P,Q × 既存14モデル + O×P, O×Q, P×Q 双方向
  - RATauto (bidirectional best) も計算

データは data/d2_matrix/ に保存（既存フォーマットと統一）。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.embedder import embed_texts
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

# 設定
DATA_DIR = config.DATA_DIR / "d2_matrix"
OUT_DIR = config.RESULTS_DIR / "d2_scale"
K = 500
SEED = 42  # Phase 1はseed=42のみ（Phase 2で複数seed）

# D2新規モデル
D2_NEW_MODELS = ["O", "P", "Q"]
# 全モデル（A-Q）
ALL_LABELS = sorted(config.MATRIX_MODELS.keys())


def compute_sim_mean(anchor_emb: np.ndarray) -> float:
    """アンカー間の平均コサイン類似度。"""
    sim = cosine_similarity(anchor_emb)
    n = len(sim)
    return float((sim.sum() - n) / (n * (n - 1)))


def generate_embeddings():
    """新規モデルの埋め込みを生成する。"""
    # テキスト読み込み（既存と同一）
    texts_path = DATA_DIR / "texts.json"
    with open(texts_path, encoding="utf-8") as f:
        texts = json.load(f)
    candidates = texts["candidates"]
    queries = texts["queries"]
    all_texts = candidates + queries
    n_cand = len(candidates)

    print(f"テキスト: {n_cand} candidates + {len(queries)} queries")

    generated = []
    for label in D2_NEW_MODELS:
        cand_path = DATA_DIR / f"cand_{label}.npy"
        query_path = DATA_DIR / f"query_{label}.npy"

        if cand_path.exists() and query_path.exists():
            print(f"\n[SKIP] Model {label}: 既存embedを使用")
            generated.append(label)
            continue

        info = config.MATRIX_MODELS[label]
        model_name = info["name"]
        short = model_name.split("/")[-1]

        print(f"\n{'='*60}")
        print(f"Model {label}: {short} ({info['params']}, {info['dim']}d)")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            embs = embed_texts(model_name, all_texts)
            cand_embs = embs[:n_cand]
            query_embs = embs[n_cand:]

            np.save(cand_path, cand_embs)
            np.save(query_path, query_embs)

            elapsed = time.time() - t0
            print(f"  → cand={cand_embs.shape}, query={query_embs.shape} ({elapsed:.1f}s)")
            generated.append(label)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] {e} ({elapsed:.1f}s)")
            raise

    return generated


def evaluate_all_new_pairs():
    """新規ペアのRAT基礎評価。"""
    # 全モデルの埋め込みを読み込み
    available = [l for l in ALL_LABELS
                 if (DATA_DIR / f"cand_{l}.npy").exists()]
    print(f"\n利用可能モデル: {len(available)} — {', '.join(available)}")

    cand_embs = {}
    query_embs = {}
    for label in available:
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")

    n_cand = cand_embs[available[0]].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    # FPSインデックス事前計算
    print("\nFPSインデックス計算中...")
    fps_cache = {}
    for label in available:
        fps_idx, _ = select_anchors_fps(
            cand_embs[label], candidates_dummy, K, seed=SEED)
        fps_cache[label] = np.array(fps_idx)

    # 新規ペアの列挙: 少なくとも1つがD2_NEW_MODELSに含まれるペア
    new_pairs = []
    for lx in available:
        for ly in available:
            if lx == ly:
                continue
            if lx in D2_NEW_MODELS or ly in D2_NEW_MODELS:
                new_pairs.append((lx, ly))

    print(f"新規ペア数: {len(new_pairs)}")

    results = []
    for i, (lx, ly) in enumerate(new_pairs):
        info_x = config.MATRIX_MODELS[lx]
        info_y = config.MATRIX_MODELS[ly]

        anchor_idx = fps_cache[lx][:K]
        anc_x = cand_embs[lx][anchor_idx]
        anc_y = cand_embs[ly][anchor_idx]

        sm = compute_sim_mean(anc_x)

        # RAT (poly + adaptive z-score)
        rel_x = to_relative(query_embs[lx], anc_x, kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_embs[ly], anc_y, kernel="poly", degree=2, coef0=1.0)

        if sm < 0.65:
            rel_x_z = normalize_zscore(rel_x)
            rel_y_z = normalize_zscore(rel_y)
            metrics = evaluate_retrieval(rel_x_z, rel_y_z)
            zscore = True
        else:
            metrics = evaluate_retrieval(rel_x, rel_y)
            zscore = False

        row = {
            "query_model": lx,
            "db_model": ly,
            "family_x": info_x["family"],
            "family_y": info_y["family"],
            "dim_x": info_x["dim"],
            "dim_y": info_y["dim"],
            "params_x": info_x["params"],
            "params_y": info_y["params"],
            "K": K,
            "seed": SEED,
            "sim_mean": round(sm, 4),
            "zscore_applied": zscore,
            **metrics,
        }
        results.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(new_pairs)}] {lx}→{ly}: "
                  f"R@1={metrics['recall_at_1']*100:.1f}% sim_mean={sm:.3f}")

    return results


def compute_rat_auto(results: list[dict]) -> list[dict]:
    """双方向のbest（RATauto）を計算。"""
    auto_results = []
    pairs_seen = set()

    for r in results:
        lx, ly = r["query_model"], r["db_model"]
        if (lx, ly) in pairs_seen or (ly, lx) in pairs_seen:
            continue

        # 逆方向を探す
        reverse = [rr for rr in results
                   if rr["query_model"] == ly and rr["db_model"] == lx]
        if not reverse:
            continue

        rr = reverse[0]
        if r["recall_at_1"] >= rr["recall_at_1"]:
            best = r
            best_dir = f"{lx}→{ly}"
        else:
            best = rr
            best_dir = f"{ly}→{lx}"

        auto_results.append({
            "model_a": lx,
            "model_b": ly,
            "best_direction": best_dir,
            "recall_at_1": best["recall_at_1"],
            "recall_at_5": best["recall_at_5"],
            "mrr": best["mrr"],
            "sim_mean": best["sim_mean"],
            "r1_forward": r["recall_at_1"],
            "r1_reverse": rr["recall_at_1"],
            "asymmetry": abs(r["recall_at_1"] - rr["recall_at_1"]),
        })
        pairs_seen.add((lx, ly))

    return auto_results


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("D2 Phase 1: Embedding Generation + Basic Evaluation")
    print("=" * 60)

    # Step 1: 埋め込み生成
    print("\n" + "=" * 60)
    print("Step 1: 新規モデル埋め込み生成")
    print("=" * 60)
    generated = generate_embeddings()

    # Step 2: 基礎評価
    print("\n" + "=" * 60)
    print("Step 2: 新規ペアRAT基礎評価 (K=500)")
    print("=" * 60)
    results = evaluate_all_new_pairs()

    # RATauto
    auto_results = compute_rat_auto(results)

    # Arctic系列サマリー
    print("\n" + "=" * 60)
    print("Arctic系列サマリー (K=500)")
    print("=" * 60)
    arctic = ["O", "P", "N", "Q"]
    arctic_names = {
        "O": "xs(22M,384d)", "P": "s(33M,384d)",
        "N": "m(109M,768d)", "Q": "l(335M,1024d)",
    }
    for lx in arctic:
        for ly in arctic:
            if lx == ly:
                continue
            r = [rr for rr in results
                 if rr["query_model"] == lx and rr["db_model"] == ly]
            if r:
                r = r[0]
                print(f"  {arctic_names[lx]:>16} → {arctic_names[ly]:<16}: "
                      f"R@1={r['recall_at_1']*100:5.1f}%  MRR={r['mrr']:.3f}")

    # 結果保存
    elapsed = time.time() - start_time
    output = {
        "config": {
            "K": K,
            "seed": SEED,
            "kernel": "poly(degree=2, coef0=1.0)",
            "zscore": "adaptive (threshold=0.65)",
            "new_models": D2_NEW_MODELS,
        },
        "n_pairs": len(results),
        "results": results,
        "rat_auto": auto_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = OUT_DIR / "d2_phase1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"総ペア数: {len(results)}, RATautoペア: {len(auto_results)}")
    print(f"実行時間: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
