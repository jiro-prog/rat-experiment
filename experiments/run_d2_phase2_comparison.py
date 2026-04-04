"""
D2 Phase 2: RAT vs Ridge vs Procrustes 比較（D1拡張）

D1と同一プロトコルで50方向ペア × 6K × 3seeds = 900レコード。
D1とは異なるseedセットを使用して再現性を検証。

ペア選定:
  - Arctic内スケール: O↔P, O↔N, O↔Q, P↔N, P↔Q, N↔Q (12方向)
  - Arctic↔BERT低圧縮: O/P/N/Q × A,J (16方向)
  - Arctic↔BERT高圧縮: O/P/N/Q × B,G (16方向)
  - 対照（再現性）: A↔C, A↔B, B↔C 双方向 (6方向)
  合計: 50方向
"""
import csv
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.anchor_sampler import select_anchors_fps
from src.relative_repr import to_relative, normalize_zscore
from src.evaluator import evaluate_retrieval
from src.baselines import (
    fit_procrustes, transform_procrustes,
    fit_ridge, transform_ridge,
    fit_affine, transform_affine,
)

# ========================================
# 設定
# ========================================
ANCHOR_COUNTS = [10, 25, 50, 100, 200, 500]
RIDGE_ALPHAS = [1e-4, 1e-2, 1.0, 100.0]
SEEDS = [314, 999, 2025]  # D1 [42, 123, 7] とは異なるセット
DATA_DIR = config.DATA_DIR / "d2_matrix"
OUT_DIR = config.RESULTS_DIR / "d2_scale"
MAX_FPS_K = 500

# ペア選定
def build_pair_list() -> list[tuple[str, str]]:
    """D2 Phase 2の50方向ペアを構築する。"""
    pairs = set()

    # Arctic内スケール (12方向)
    arctic = ["O", "P", "N", "Q"]
    for i, lx in enumerate(arctic):
        for ly in arctic[i+1:]:
            pairs.add((lx, ly))
            pairs.add((ly, lx))

    # Arctic↔BERT低圧縮: A(MiniLM), J(MPNet) — sim_mean低い側
    for a in arctic:
        for b in ["A", "J"]:
            pairs.add((a, b))
            pairs.add((b, a))

    # Arctic↔BERT高圧縮: B(E5-large), G(E5-small-multi) — sim_mean高い側
    for a in arctic:
        for b in ["B", "G"]:
            pairs.add((a, b))
            pairs.add((b, a))

    # 対照: D1再現性チェック（異なるseed）
    for lx, ly in [("A", "C"), ("C", "A"), ("A", "B"), ("B", "A"),
                    ("B", "C"), ("C", "B")]:
        pairs.add((lx, ly))

    return sorted(pairs)


def compute_sim_mean(anchor_emb: np.ndarray) -> float:
    sim = cosine_similarity(anchor_emb)
    n = len(sim)
    return float((sim.sum() - n) / (n * (n - 1)))


def evaluate_pair_all_methods(
    label_x: str, label_y: str,
    cand_x: np.ndarray, cand_y: np.ndarray,
    query_x: np.ndarray, query_y: np.ndarray,
    fps_indices: np.ndarray, seed: int,
) -> list[dict]:
    """1ペア×1シードの全手法×全K評価（D1と同一ロジック）。"""
    dim_x = cand_x.shape[1]
    dim_y = cand_y.shape[1]
    same_dim = (dim_x == dim_y)
    results = []

    for K_val in ANCHOR_COUNTS:
        anchor_idx = fps_indices[:K_val]
        anc_x = cand_x[anchor_idx]
        anc_y = cand_y[anchor_idx]
        sm = compute_sim_mean(anc_x)

        row_base = {
            "query_model": label_x,
            "db_model": label_y,
            "dim_x": dim_x, "dim_y": dim_y,
            "same_dim": same_dim,
            "K": K_val, "seed": seed,
            "sim_mean": sm,
        }

        # --- RAT ---
        rel_x = to_relative(query_x, anc_x, kernel="poly", degree=2, coef0=1.0)
        rel_y = to_relative(query_y, anc_y, kernel="poly", degree=2, coef0=1.0)
        if sm < 0.65:
            rat_metrics = evaluate_retrieval(normalize_zscore(rel_x),
                                             normalize_zscore(rel_y))
            zscore = True
        else:
            rat_metrics = evaluate_retrieval(rel_x, rel_y)
            zscore = False

        results.append({
            **row_base, "method": "RAT",
            "zscore_applied": zscore, "alpha": None,
            **rat_metrics,
        })

        # --- Procrustes (同次元のみ) ---
        if same_dim:
            W = fit_procrustes(anc_x, anc_y)
            aligned = transform_procrustes(query_x, W)
            proc_metrics = evaluate_retrieval(aligned, query_y)
            results.append({
                **row_base, "method": "Procrustes",
                "zscore_applied": None, "alpha": None,
                **proc_metrics,
            })

        # --- Ridge (best λ) ---
        best_ridge = None
        best_r1 = -1.0
        for alpha in RIDGE_ALPHAS:
            W = fit_ridge(anc_x, anc_y, alpha=alpha)
            aligned = transform_ridge(query_x, W)
            m = evaluate_retrieval(aligned, query_y)
            if m["recall_at_1"] > best_r1:
                best_r1 = m["recall_at_1"]
                best_ridge = {
                    **row_base, "method": "Ridge",
                    "zscore_applied": None, "alpha": alpha, **m,
                }
        results.append(best_ridge)

        # --- Affine (best λ) ---
        best_affine = None
        best_r1 = -1.0
        for alpha in RIDGE_ALPHAS:
            W, bias = fit_affine(anc_x, anc_y, alpha=alpha)
            aligned = transform_affine(query_x, W, bias)
            m = evaluate_retrieval(aligned, query_y)
            if m["recall_at_1"] > best_r1:
                best_r1 = m["recall_at_1"]
                best_affine = {
                    **row_base, "method": "Affine",
                    "zscore_applied": None, "alpha": alpha, **m,
                }
        results.append(best_affine)

    return results


def save_csv(all_results: list[dict], path: Path):
    fields = [
        "query_model", "db_model", "dim_x", "dim_y", "same_dim",
        "K", "seed", "sim_mean", "method", "zscore_applied", "alpha",
        "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = build_pair_list()
    # 利用可能なモデルでフィルタ
    available = [l for l in sorted(config.MATRIX_MODELS.keys())
                 if (DATA_DIR / f"cand_{l}.npy").exists()]
    pairs = [(lx, ly) for lx, ly in pairs
             if lx in available and ly in available]

    print("=" * 70)
    print("D2 Phase 2: RAT vs Ridge vs Procrustes Comparison")
    print(f"Pairs: {len(pairs)}, K: {ANCHOR_COUNTS}, Seeds: {SEEDS}")
    print(f"Expected records: {len(pairs) * len(ANCHOR_COUNTS) * len(SEEDS) * 3}+")
    print("=" * 70)

    # 全ペアで使うモデルを特定
    models_needed = set()
    for lx, ly in pairs:
        models_needed.add(lx)
        models_needed.add(ly)

    # embedding読み込み
    cand_embs = {}
    query_embs = {}
    for label in sorted(models_needed):
        cand_embs[label] = np.load(DATA_DIR / f"cand_{label}.npy")
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")
        info = config.MATRIX_MODELS[label]
        print(f"  {label}: cand={cand_embs[label].shape} ({info['family']})")

    n_cand = cand_embs[list(models_needed)[0]].shape[0]
    candidates_dummy = [str(i) for i in range(n_cand)]

    all_results = []
    n_pairs = len(pairs)

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"Seed {seed_idx+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*70}")

        # FPSインデックス事前計算
        fps_cache = {}
        for label in sorted(models_needed):
            fps_idx, _ = select_anchors_fps(
                cand_embs[label], candidates_dummy, MAX_FPS_K, seed=seed)
            fps_cache[label] = np.array(fps_idx)

        for i, (lx, ly) in enumerate(pairs):
            pair_results = evaluate_pair_all_methods(
                lx, ly,
                cand_embs[lx], cand_embs[ly],
                query_embs[lx], query_embs[ly],
                fps_cache[lx], seed=seed,
            )
            all_results.extend(pair_results)

            if seed_idx == 0 and (i + 1) % 10 == 0:
                k500 = [r for r in pair_results if r["K"] == 500]
                rat = [r for r in k500 if r["method"] == "RAT"]
                ridge = [r for r in k500 if r["method"] == "Ridge"]
                if rat and ridge:
                    print(f"  [{i+1}/{n_pairs}] {lx}→{ly}: "
                          f"RAT={rat[0]['recall_at_1']*100:.1f}% "
                          f"Ridge={ridge[0]['recall_at_1']*100:.1f}%")

        elapsed_seed = time.time() - start_time
        print(f"Seed {seed} 完了 (累計 {elapsed_seed:.0f}s)")

    # 結果保存
    elapsed = time.time() - start_time

    # JSON
    output = {
        "config": {
            "anchor_counts": ANCHOR_COUNTS,
            "ridge_alphas": RIDGE_ALPHAS,
            "seeds": SEEDS,
            "kernel": "poly(degree=2, coef0=1.0)",
            "zscore": "adaptive (threshold=0.65)",
            "pair_selection": {
                "arctic_internal": 12,
                "arctic_bert_low_compression": 16,
                "arctic_bert_high_compression": 16,
                "control_reproducibility": 6,
                "total": len(pairs),
            },
        },
        "n_records": len(all_results),
        "results": all_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = OUT_DIR / "d2_phase2_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = OUT_DIR / "d2_phase2_results.csv"
    save_csv(all_results, csv_path)

    print(f"\n{'='*70}")
    print(f"結果保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"総レコード数: {len(all_results)}")
    print(f"実行時間: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # サマリー: RAT勝率 vs Ridge (K別)
    print(f"\n{'='*70}")
    print("RAT勝率 vs Best Linear（seed=314）")
    print(f"{'='*70}")
    primary = [r for r in all_results if r["seed"] == SEEDS[0]]
    for K_val in ANCHOR_COUNTS:
        k_results = [r for r in primary if r["K"] == K_val]
        rat_wins = 0
        total = 0
        for r in k_results:
            if r["method"] != "RAT":
                continue
            qm, dm = r["query_model"], r["db_model"]
            linear = [lr for lr in k_results
                      if lr["query_model"] == qm and lr["db_model"] == dm
                      and lr["method"] in ("Procrustes", "Ridge", "Affine")]
            if linear:
                best_lin = max(lr["recall_at_1"] for lr in linear)
                if r["recall_at_1"] > best_lin:
                    rat_wins += 1
                total += 1
        if total:
            print(f"  K={K_val:>3}: RAT wins {rat_wins}/{total} "
                  f"({rat_wins/total*100:.0f}%)")


if __name__ == "__main__":
    main()
