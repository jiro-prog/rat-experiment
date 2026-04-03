"""
Direction 2 実験群A Step 2: 全ペアRAT精度マトリクス

run_d2a_embed_all.py で保存した .npy を読み込み、
12モデル × 12モデルの全132有向ペアについてRAT検索精度を評価する。

各ペア (X→Y):
  - クエリ側 = Model X, DB側 = Model Y
  - FPSアンカー選定はModel Xの空間で実施（クエリ側基準）
  - poly kernel (u·a+1)², K=500
  - baseline + z-score DB-side の2条件

出力:
  - 12×12 精度マトリクス (Recall@1, MRR)
  - ペア別詳細メトリクス
  - 分析用メタデータ（similarity collapse指標、アンカーentropy等）
"""
import sys
import json
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import entropy as sp_entropy

import config
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval

# 設定
K = 500  # アンカー数
DATA_DIR = config.DATA_DIR / "d2_matrix"


def compute_sim_stats(embs: np.ndarray) -> dict:
    """embedding空間の類似度分布統計（similarity collapse検出用）。"""
    # ランダムに1000ペアサンプル
    n = len(embs)
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, n, size=min(1000, n * (n - 1) // 2))
    idx_b = rng.randint(0, n, size=min(1000, n * (n - 1) // 2))
    # 同一インデックスを避ける
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    sims = np.sum(embs[idx_a] * embs[idx_b], axis=1)
    return {
        "sim_mean": float(sims.mean()),
        "sim_std": float(sims.std()),
        "sim_min": float(sims.min()),
        "sim_max": float(sims.max()),
        "sim_range": float(sims.max() - sims.min()),
    }


def compute_anchor_entropy(rel: np.ndarray) -> float:
    """相対表現のアンカー利用分布のentropy。
    各クエリについてsoftmax(rel)のentropyを計算し平均。
    高いentropy → アンカーが均等に使われている。
    低いentropy → 少数のアンカーに集中（潰れの兆候）。
    """
    # 数値安定のためmax引いてからsoftmax
    shifted = rel - rel.max(axis=1, keepdims=True)
    exp_rel = np.exp(shifted)
    probs = exp_rel / exp_rel.sum(axis=1, keepdims=True)
    ents = sp_entropy(probs, axis=1)  # 各クエリのentropy
    return float(np.mean(ents))


def evaluate_pair(
    label_x: str,
    label_y: str,
    cand_embs: dict[str, np.ndarray],
    query_embs: dict[str, np.ndarray],
    fps_cache: dict[str, np.ndarray],
    candidates_dummy: list[str],
) -> dict:
    """1有向ペア (X→Y) を評価する。"""
    # FPSはクエリ側(X)空間で実施（キャッシュ）
    if label_x not in fps_cache:
        fps_indices, _ = select_anchors_fps(
            cand_embs[label_x], candidates_dummy, K
        )
        fps_cache[label_x] = np.array(fps_indices)

    fps_idx = fps_cache[label_x]
    anchor_x = cand_embs[label_x][fps_idx]
    anchor_y = cand_embs[label_y][fps_idx]

    # 相対表現
    rel_x = to_relative(query_embs[label_x], anchor_x, kernel="poly", degree=2, coef0=1.0)
    rel_y = to_relative(query_embs[label_y], anchor_y, kernel="poly", degree=2, coef0=1.0)

    # baseline
    baseline = evaluate_retrieval(rel_x, rel_y)

    # z-score DB-side only
    rel_y_z = normalize_zscore(rel_y)
    zscore_db = evaluate_retrieval(rel_x, rel_y_z)

    # アンカーentropy
    ent_x = compute_anchor_entropy(rel_x)
    ent_y = compute_anchor_entropy(rel_y)

    return {
        "query_model": label_x,
        "db_model": label_y,
        "baseline": baseline,
        "zscore_db": zscore_db,
        "anchor_entropy_query": ent_x,
        "anchor_entropy_db": ent_y,
    }


def print_matrix(labels: list[str], matrix: np.ndarray, title: str):
    """12×12マトリクスをテーブル表示。"""
    print(f"\n{title}")
    print("Query↓  DB→  ", end="")
    for l in labels:
        print(f"  {l:>5}", end="")
    print()
    print("-" * (14 + 7 * len(labels)))
    for i, lx in enumerate(labels):
        print(f"  {lx:<10}  ", end="")
        for j, ly in enumerate(labels):
            if i == j:
                print(f"    - ", end="")
            else:
                print(f" {matrix[i, j]:5.1f}", end="")
        print()


def main():
    start_time = time.time()

    # ========================================
    # embedding読み込み
    # ========================================
    print("=" * 70)
    print("Direction 2A: 全ペアRAT精度マトリクス")
    print("=" * 70)

    # メタデータ読み込み
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("ERROR: まず run_d2a_embed_all.py を実行してください")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    available = meta["completed"]
    print(f"利用可能モデル: {len(available)} — {', '.join(available)}")

    # embedding読み込み
    cand_embs = {}
    query_embs = {}
    for label in available:
        cand_path = DATA_DIR / f"cand_{label}.npy"
        query_path = DATA_DIR / f"query_{label}.npy"
        cand_embs[label] = np.load(cand_path)
        query_embs[label] = np.load(query_path)
        info = config.MATRIX_MODELS[label]
        print(f"  {label}: cand={cand_embs[label].shape}, query={query_embs[label].shape} "
              f"({info['family']}, {info['params']})")

    # FPSのためにダミーのテキストリスト（インデックスだけ使うので中身は不要）
    n_cand = cand_embs[available[0]].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    # ========================================
    # 各モデルのsimilarity collapse統計
    # ========================================
    print(f"\n{'='*70}")
    print("類似度分布統計（similarity collapse検出）")
    print(f"{'='*70}")

    sim_stats = {}
    for label in available:
        stats = compute_sim_stats(query_embs[label])
        sim_stats[label] = stats
        info = config.MATRIX_MODELS[label]
        print(f"  {label} ({info['family']:<6} {info['params']:>5}): "
              f"mean={stats['sim_mean']:.4f}, std={stats['sim_std']:.4f}, "
              f"range={stats['sim_range']:.4f}")

    # ========================================
    # 全ペア評価
    # ========================================
    print(f"\n{'='*70}")
    print(f"全ペア評価 (K={K}, poly kernel, {len(available)}×{len(available)-1} = "
          f"{len(available)*(len(available)-1)} 有向ペア)")
    print(f"{'='*70}")

    fps_cache = {}
    pair_results = []
    n_pairs = len(available) * (len(available) - 1)
    done = 0

    for lx in available:
        for ly in available:
            if lx == ly:
                continue
            done += 1
            info_x = config.MATRIX_MODELS[lx]
            info_y = config.MATRIX_MODELS[ly]
            print(f"\n[{done}/{n_pairs}] {lx}→{ly}: "
                  f"{info_x['family']}({info_x['params']}) → "
                  f"{info_y['family']}({info_y['params']})")

            res = evaluate_pair(lx, ly, cand_embs, query_embs, fps_cache, candidates_dummy)
            pair_results.append(res)

            r1_base = res["baseline"]["recall_at_1"] * 100
            r1_z = res["zscore_db"]["recall_at_1"] * 100
            print(f"  baseline R@1={r1_base:.1f}%, zscore_db R@1={r1_z:.1f}%, "
                  f"ent_q={res['anchor_entropy_query']:.2f}, ent_db={res['anchor_entropy_db']:.2f}")

    # ========================================
    # マトリクス構築
    # ========================================
    n = len(available)
    r1_baseline = np.zeros((n, n))
    r1_zscore = np.zeros((n, n))
    mrr_baseline = np.zeros((n, n))

    label_to_idx = {l: i for i, l in enumerate(available)}
    for res in pair_results:
        i = label_to_idx[res["query_model"]]
        j = label_to_idx[res["db_model"]]
        r1_baseline[i, j] = res["baseline"]["recall_at_1"] * 100
        r1_zscore[i, j] = res["zscore_db"]["recall_at_1"] * 100
        mrr_baseline[i, j] = res["baseline"]["mrr"] * 100

    print_matrix(available, r1_baseline, "Recall@1 (%) — Baseline")
    print_matrix(available, r1_zscore, "Recall@1 (%) — Z-score DB")
    print_matrix(available, mrr_baseline, "MRR (×100) — Baseline")

    # ========================================
    # 分析: 事前定義した比較軸
    # ========================================
    print(f"\n{'='*70}")
    print("分析サマリー")
    print(f"{'='*70}")

    # best of baseline vs zscore per pair
    best_r1 = np.maximum(r1_baseline, r1_zscore)
    print_matrix(available, best_r1, "Recall@1 (%) — Best of Baseline/Zscore")

    # 1. 同一ファミリー内サイズ効果
    print(f"\n--- 同一ファミリー内サイズ効果 ---")
    families = {}
    for label in available:
        fam = config.MATRIX_MODELS[label]["family"]
        families.setdefault(fam, []).append(label)
    for fam, members in sorted(families.items()):
        if len(members) < 2:
            continue
        print(f"\n  {fam}: {', '.join(members)}")
        # ファミリー内ペアの精度
        for lx in members:
            for ly in members:
                if lx == ly:
                    continue
                i, j = label_to_idx[lx], label_to_idx[ly]
                print(f"    {lx}→{ly}: R@1={best_r1[i,j]:.1f}%")
        # ファミリー内 vs ファミリー外の平均精度
        in_family = []
        out_family = []
        for lx in members:
            for ly in available:
                if lx == ly:
                    continue
                i, j = label_to_idx[lx], label_to_idx[ly]
                if ly in members:
                    in_family.append(best_r1[i, j])
                else:
                    out_family.append(best_r1[i, j])
        if in_family:
            print(f"    ファミリー内平均: {np.mean(in_family):.1f}%")
        print(f"    ファミリー外平均: {np.mean(out_family):.1f}%")

    # 2. 同サイズ異ファミリー
    print(f"\n--- 同サイズ異ファミリー比較 ---")
    size_groups = {}
    for label in available:
        params = config.MATRIX_MODELS[label]["params"]
        size_groups.setdefault(params, []).append(label)
    for size, members in sorted(size_groups.items()):
        if len(members) < 2:
            continue
        print(f"\n  {size}: {', '.join(members)}")
        for lx in members:
            for ly in members:
                if lx == ly:
                    continue
                i, j = label_to_idx[lx], label_to_idx[ly]
                print(f"    {lx}→{ly}: R@1={best_r1[i,j]:.1f}%")

    # 3. 多言語 vs 英語
    print(f"\n--- 多言語 vs 英語 ---")
    multi_labels = [l for l in available if config.MATRIX_MODELS[l]["lang"] == "multi"]
    en_labels = [l for l in available if config.MATRIX_MODELS[l]["lang"] == "en"]
    if multi_labels:
        # 多言語モデルがクエリ側の場合の平均精度
        multi_as_query = []
        en_as_query = []
        for lx in available:
            for ly in available:
                if lx == ly:
                    continue
                i, j = label_to_idx[lx], label_to_idx[ly]
                if lx in multi_labels:
                    multi_as_query.append(best_r1[i, j])
                else:
                    en_as_query.append(best_r1[i, j])
        print(f"  多言語モデルがクエリ: mean R@1={np.mean(multi_as_query):.1f}% ({multi_labels})")
        print(f"  英語モデルがクエリ:   mean R@1={np.mean(en_as_query):.1f}%")

        # E5内 EN vs MULTI比較
        e5_members = families.get("E5", [])
        if len(e5_members) >= 2:
            print(f"\n  E5内比較:")
            for lx in e5_members:
                for ly in e5_members:
                    if lx == ly:
                        continue
                    i, j = label_to_idx[lx], label_to_idx[ly]
                    info_x = config.MATRIX_MODELS[lx]
                    info_y = config.MATRIX_MODELS[ly]
                    print(f"    {lx}({info_x['lang']},{info_x['params']})→"
                          f"{ly}({info_y['lang']},{info_y['params']}): R@1={best_r1[i,j]:.1f}%")

    # 4. similarity collapse相関
    print(f"\n--- Similarity Collapse vs RAT精度 ---")
    print(f"  モデル: sim_std → 平均R@1(as query), 平均R@1(as db)")
    for label in available:
        std = sim_stats[label]["sim_std"]
        idx = label_to_idx[label]
        # as query: row average (excluding diagonal)
        as_query = [best_r1[idx, j] for j in range(n) if j != idx]
        # as db: column average
        as_db = [best_r1[i, idx] for i in range(n) if i != idx]
        print(f"  {label} ({config.MATRIX_MODELS[label]['family']:>6}): "
              f"sim_std={std:.4f} → query_avg={np.mean(as_query):.1f}%, "
              f"db_avg={np.mean(as_db):.1f}%")

    # 5. アンカーentropy vs 精度
    print(f"\n--- アンカーEntropy vs RAT精度 ---")
    for res in pair_results:
        lx, ly = res["query_model"], res["db_model"]
        r1 = res["baseline"]["recall_at_1"] * 100
        ent_q = res["anchor_entropy_query"]
        ent_db = res["anchor_entropy_db"]
        # 出力は多すぎるので上位・下位5ペアのみ
    # entropy-精度相関
    ent_qs = [r["anchor_entropy_query"] for r in pair_results]
    ent_dbs = [r["anchor_entropy_db"] for r in pair_results]
    r1s = [r["baseline"]["recall_at_1"] * 100 for r in pair_results]
    from scipy.stats import spearmanr
    rho_q, p_q = spearmanr(ent_qs, r1s)
    rho_db, p_db = spearmanr(ent_dbs, r1s)
    rho_sum, p_sum = spearmanr(np.array(ent_qs) + np.array(ent_dbs), r1s)
    print(f"  Spearman ρ(entropy_query, R@1) = {rho_q:.3f} (p={p_q:.2e})")
    print(f"  Spearman ρ(entropy_db, R@1)    = {rho_db:.3f} (p={p_db:.2e})")
    print(f"  Spearman ρ(entropy_sum, R@1)   = {rho_sum:.3f} (p={p_sum:.2e})")

    # ========================================
    # 結果保存
    # ========================================
    elapsed = time.time() - start_time

    output = {
        "config": {
            "K": K,
            "kernel": "poly(degree=2, coef0=1.0)",
            "candidate_pool": n_cand,
            "num_queries": query_embs[available[0]].shape[0],
            "fps_space": "query_model",
        },
        "models": {
            label: {
                "name": config.MATRIX_MODELS[label]["name"],
                "family": config.MATRIX_MODELS[label]["family"],
                "params": config.MATRIX_MODELS[label]["params"],
                "dim": config.MATRIX_MODELS[label]["dim"],
                "training": config.MATRIX_MODELS[label]["training"],
                "lang": config.MATRIX_MODELS[label]["lang"],
            }
            for label in available
        },
        "labels": available,
        "matrix_r1_baseline": r1_baseline.tolist(),
        "matrix_r1_zscore": r1_zscore.tolist(),
        "matrix_mrr_baseline": mrr_baseline.tolist(),
        "sim_stats": sim_stats,
        "pair_results": [
            {
                "query": r["query_model"],
                "db": r["db_model"],
                "baseline_r1": r["baseline"]["recall_at_1"],
                "baseline_r5": r["baseline"]["recall_at_5"],
                "baseline_r10": r["baseline"]["recall_at_10"],
                "baseline_mrr": r["baseline"]["mrr"],
                "zscore_db_r1": r["zscore_db"]["recall_at_1"],
                "zscore_db_r5": r["zscore_db"]["recall_at_5"],
                "zscore_db_r10": r["zscore_db"]["recall_at_10"],
                "zscore_db_mrr": r["zscore_db"]["mrr"],
                "anchor_entropy_query": r["anchor_entropy_query"],
                "anchor_entropy_db": r["anchor_entropy_db"],
            }
            for r in pair_results
        ],
        "elapsed_seconds": elapsed,
    }

    out_path = config.RESULTS_DIR / "d2a_matrix.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
