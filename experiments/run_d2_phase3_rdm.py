"""
D2 Phase 3: RDM相関分析 + 階層的回帰 + スケール軸可視化

分析1: 17×17 RDM相関行列（Mantel test, Spearman）
分析2: RDM相関 vs RAT R@1 / Ridge R@1 散布図
分析3: 階層的回帰
  - Model 1: R@1 ~ sim_mean
  - Model 2: R@1 ~ sim_mean + RDM
  - Model 3: R@1 ~ sim_mean + RDM + same_family
分析4: スケール軸プロット（Arctic params vs RDM vs R@1）

入力: Phase 1/2の結果 + 既存D1結果 + 埋め込み
出力: figures + JSON + analysis tables
"""
import sys
import json
import time
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

DATA_DIR = config.DATA_DIR / "d2_matrix"
OUT_DIR = config.RESULTS_DIR / "d2_scale"
PHASE1_PATH = OUT_DIR / "d2_phase1_results.json"
PHASE2_PATH = OUT_DIR / "d2_phase2_results.json"
D1_CSV_PATH = config.RESULTS_DIR / "d1_alignment" / "d1_results.csv"


def compute_rdm(query_embs: np.ndarray) -> np.ndarray:
    """500クエリ間の距離行列（1 - cosine similarity）。"""
    sim = cosine_similarity(query_embs)
    return 1.0 - sim


def compute_rdm_correlation_matrix(labels: list[str],
                                    query_embs: dict) -> dict:
    """全ペアのRDM Spearman相関を計算する。"""
    rdms = {}
    for label in labels:
        rdm = compute_rdm(query_embs[label])
        # 上三角のみ取得（対角=0を除く）
        rdms[label] = squareform(rdm, checks=False)

    n = len(labels)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            elif j > i:
                rho, p = spearmanr(rdms[labels[i]], rdms[labels[j]])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
                p_matrix[i, j] = p
                p_matrix[j, i] = p

    return {
        "labels": labels,
        "matrix": corr_matrix,
        "p_values": p_matrix,
        "rdm_vectors": rdms,
    }


def load_all_results():
    """Phase 1, Phase 2, D1の結果を統合する。"""
    import csv

    all_pairs = {}  # (lx, ly) -> {K -> {method -> metrics}}

    # D1結果読み込み
    if D1_CSV_PATH.exists():
        with open(D1_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["query_model"], row["db_model"])
                K = int(row["K"])
                method = row["method"]
                seed = int(row["seed"]) if row["seed"] else 42

                if key not in all_pairs:
                    all_pairs[key] = {}
                if K not in all_pairs[key]:
                    all_pairs[key][K] = {}
                if method not in all_pairs[key][K]:
                    all_pairs[key][K][method] = []

                all_pairs[key][K][method].append({
                    "recall_at_1": float(row["recall_at_1"]),
                    "recall_at_5": float(row["recall_at_5"]),
                    "mrr": float(row["mrr"]),
                    "sim_mean": float(row["sim_mean"]),
                    "seed": seed,
                    "source": "D1",
                })

    # Phase 2結果読み込み
    if PHASE2_PATH.exists():
        with open(PHASE2_PATH, encoding="utf-8") as f:
            p2 = json.load(f)
        for row in p2["results"]:
            key = (row["query_model"], row["db_model"])
            K = row["K"]
            method = row["method"]

            if key not in all_pairs:
                all_pairs[key] = {}
            if K not in all_pairs[key]:
                all_pairs[key][K] = {}
            if method not in all_pairs[key][K]:
                all_pairs[key][K][method] = []

            all_pairs[key][K][method].append({
                "recall_at_1": row["recall_at_1"],
                "recall_at_5": row["recall_at_5"],
                "mrr": row["mrr"],
                "sim_mean": row["sim_mean"],
                "seed": row["seed"],
                "source": "D2",
            })

    return all_pairs


def hierarchical_regression(pair_features: list[dict]):
    """階層的回帰: R@1 ~ sim_mean (+RDM +same_family)。"""
    from sklearn.linear_model import LinearRegression

    # 特徴量抽出
    y = np.array([p["r1"] for p in pair_features])
    X_sim = np.array([p["sim_mean"] for p in pair_features]).reshape(-1, 1)
    X_rdm = np.array([p["rdm_corr"] for p in pair_features]).reshape(-1, 1)
    X_fam = np.array([1.0 if p["same_family"] else 0.0
                       for p in pair_features]).reshape(-1, 1)

    results = {}

    # Model 1: R@1 ~ sim_mean
    m1 = LinearRegression().fit(X_sim, y)
    r2_1 = m1.score(X_sim, y)
    results["model1"] = {
        "formula": "R@1 ~ sim_mean",
        "R2": round(r2_1, 4),
        "coef_sim_mean": round(float(m1.coef_[0]), 4),
        "intercept": round(float(m1.intercept_), 4),
    }

    # Model 2: R@1 ~ sim_mean + RDM
    X2 = np.hstack([X_sim, X_rdm])
    m2 = LinearRegression().fit(X2, y)
    r2_2 = m2.score(X2, y)
    delta_r2 = r2_2 - r2_1
    results["model2"] = {
        "formula": "R@1 ~ sim_mean + RDM",
        "R2": round(r2_2, 4),
        "delta_R2": round(delta_r2, 4),
        "coef_sim_mean": round(float(m2.coef_[0]), 4),
        "coef_rdm": round(float(m2.coef_[1]), 4),
        "intercept": round(float(m2.intercept_), 4),
    }

    # Model 3: R@1 ~ sim_mean + RDM + same_family
    X3 = np.hstack([X_sim, X_rdm, X_fam])
    m3 = LinearRegression().fit(X3, y)
    r2_3 = m3.score(X3, y)
    results["model3"] = {
        "formula": "R@1 ~ sim_mean + RDM + same_family",
        "R2": round(r2_3, 4),
        "delta_R2_from_m2": round(r2_3 - r2_2, 4),
        "coef_sim_mean": round(float(m3.coef_[0]), 4),
        "coef_rdm": round(float(m3.coef_[1]), 4),
        "coef_same_family": round(float(m3.coef_[2]), 4),
        "intercept": round(float(m3.intercept_), 4),
    }

    # 偏相関: RDM vs R@1 controlling for sim_mean
    resid_y = y - LinearRegression().fit(X_sim, y).predict(X_sim)
    resid_rdm = X_rdm.ravel() - LinearRegression().fit(X_sim, X_rdm.ravel()).predict(X_sim)
    partial_rho, partial_p = spearmanr(resid_rdm, resid_y)
    results["partial_correlation"] = {
        "rdm_vs_r1_controlling_sim_mean": {
            "spearman_rho": round(float(partial_rho), 4),
            "p_value": float(partial_p),
        },
    }

    return results


def plot_rdm_heatmap(corr_result: dict, out_path: Path):
    """17×17 RDM相関ヒートマップ。"""
    labels = corr_result["labels"]
    matrix = corr_result["matrix"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 値をセル内に表示
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if matrix[i,j] > 0.6 else "black")

    # Arctic系列をハイライト
    arctic_idx = [labels.index(l) for l in ["O", "P", "N", "Q"] if l in labels]
    for idx in arctic_idx:
        ax.get_xticklabels()[idx].set_color("red")
        ax.get_yticklabels()[idx].set_color("red")

    plt.colorbar(im, label="RDM Spearman rho")
    ax.set_title("RDM Correlation Matrix (17 models)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rdm_vs_r1(pair_features: list[dict], out_path: Path):
    """RDM相関 vs RAT R@1 散布図。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RAT
    ax = axes[0]
    rdm = [p["rdm_corr"] for p in pair_features]
    r1_rat = [p["r1"] for p in pair_features]
    sm = [p["sim_mean"] for p in pair_features]

    sc = ax.scatter(rdm, r1_rat, c=sm, cmap="viridis", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="sim_mean")
    ax.set_xlabel("RDM Spearman rho")
    ax.set_ylabel("RAT R@1")
    ax.set_title("RDM vs RAT R@1 (K=500)")

    # Arctic系列をハイライト
    arctic = {"O", "P", "N", "Q"}
    for p in pair_features:
        if p["lx"] in arctic or p["ly"] in arctic:
            ax.annotate(f"{p['lx']}{p['ly']}", (p["rdm_corr"], p["r1"]),
                       fontsize=5, alpha=0.7)

    # Ridge
    ax = axes[1]
    r1_ridge = [p.get("r1_ridge", 0) for p in pair_features]
    sc = ax.scatter(rdm, r1_ridge, c=sm, cmap="viridis", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="sim_mean")
    ax.set_xlabel("RDM Spearman rho")
    ax.set_ylabel("Ridge R@1")
    ax.set_title("RDM vs Ridge R@1 (K=500)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scale_axis(rdm_result: dict, all_pairs: dict, out_path: Path):
    """Arctic スケール軸プロット。"""
    arctic_models = [
        ("O", "xs\n22M\n384d"),
        ("P", "s\n33M\n384d"),
        ("N", "m\n109M\n768d"),
        ("Q", "l\n335M\n1024d"),
    ]
    arctic_labels = [a[0] for a in arctic_models]
    params = [22, 33, 109, 335]  # millions

    labels = rdm_result["labels"]
    matrix = rdm_result["matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: RDM相関 vs scale (対Model A)
    ax = axes[0]
    target_models = [("A", "MiniLM", "tab:blue"),
                     ("C", "BGE-s", "tab:orange"),
                     ("J", "MPNet", "tab:green")]

    for target, tname, color in target_models:
        if target not in labels:
            continue
        t_idx = labels.index(target)
        rhos = []
        for a_label in arctic_labels:
            if a_label in labels:
                a_idx = labels.index(a_label)
                rhos.append(matrix[a_idx, t_idx])
            else:
                rhos.append(np.nan)
        ax.plot(params, rhos, "o-", label=f"vs {target}({tname})", color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Arctic Parameters (M)")
    ax.set_ylabel("RDM Spearman rho")
    ax.set_title("Scale vs RDM Correlation")
    ax.set_xticks(params)
    ax.set_xticklabels([a[1] for a in arctic_models], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: R@1 vs scale
    ax = axes[1]
    for K_val, ls in [(100, "--"), (500, "-")]:
        for target, tname, color in target_models:
            r1s = []
            for a_label in arctic_labels:
                key = (a_label, target)
                if key in all_pairs and K_val in all_pairs[key]:
                    rat_data = all_pairs[key][K_val].get("RAT", [])
                    if rat_data:
                        r1s.append(np.mean([d["recall_at_1"] for d in rat_data]))
                    else:
                        r1s.append(np.nan)
                else:
                    r1s.append(np.nan)
            ax.plot(params, r1s, f"o{ls}", label=f"vs {target} K={K_val}",
                    color=color, alpha=0.8 if ls == "-" else 0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Arctic Parameters (M)")
    ax.set_ylabel("RAT R@1")
    ax.set_title("Scale vs RAT R@1")
    ax.set_xticks(params)
    ax.set_xticklabels([a[1] for a in arctic_models], fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("D2 Phase 3: RDM Correlation Analysis")
    print("=" * 60)

    # 全モデルのクエリ埋め込み読み込み
    available = [l for l in sorted(config.MATRIX_MODELS.keys())
                 if (DATA_DIR / f"query_{l}.npy").exists()]
    print(f"利用可能モデル: {len(available)} — {', '.join(available)}")

    query_embs = {}
    for label in available:
        query_embs[label] = np.load(DATA_DIR / f"query_{label}.npy")

    # ========================================
    # 分析1: RDM相関行列
    # ========================================
    print("\n--- 分析1: RDM相関行列 ---")
    rdm_result = compute_rdm_correlation_matrix(available, query_embs)

    # JSON保存（行列はリスト化）
    rdm_json = {
        "labels": rdm_result["labels"],
        "rdm_spearman": {
            li: {lj: round(float(rdm_result["matrix"][i, j]), 4)
                 for j, lj in enumerate(available)}
            for i, li in enumerate(available)
        },
        "n_samples": query_embs[available[0]].shape[0],
    }
    rdm_path = OUT_DIR / "d2_rdm_correlation.json"
    with open(rdm_path, "w", encoding="utf-8") as f:
        json.dump(rdm_json, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {rdm_path}")

    # ヒートマップ
    plot_rdm_heatmap(rdm_result, OUT_DIR / "fig_rdm_heatmap.png")

    # Arctic系列のRDMサマリー
    print("\n  Arctic系列 RDM相関:")
    arctic = ["O", "P", "N", "Q"]
    arctic_avail = [l for l in arctic if l in available]
    for i, li in enumerate(arctic_avail):
        for lj in arctic_avail[i+1:]:
            idx_i = available.index(li)
            idx_j = available.index(lj)
            rho = rdm_result["matrix"][idx_i, idx_j]
            print(f"    {li}↔{lj}: ρ={rho:.4f}")

    # Q1判定: Arctic-l vs Arctic-xs のRDM変化
    if "O" in available and "Q" in available and "A" in available:
        idx_o = available.index("O")
        idx_q = available.index("Q")
        idx_a = available.index("A")
        delta_rho = rdm_result["matrix"][idx_q, idx_a] - rdm_result["matrix"][idx_o, idx_a]
        print(f"\n  Q1判定: Δρ(Arctic-l vs Arctic-xs, 対A) = {delta_rho:.4f}")
        if abs(delta_rho) > 0.1:
            print(f"    → 明確に変化 (|Δρ| > 0.1)")
        elif abs(delta_rho) > 0.05:
            print(f"    → 微弱な変化 (0.05 < |Δρ| ≤ 0.1)")
        else:
            print(f"    → 変化なし (|Δρ| ≤ 0.05)")

    # xs→s (同次元) の純粋パラメータ効果
    if "O" in available and "P" in available and "A" in available:
        idx_o = available.index("O")
        idx_p = available.index("P")
        idx_a = available.index("A")
        delta_xs_s = rdm_result["matrix"][idx_p, idx_a] - rdm_result["matrix"][idx_o, idx_a]
        print(f"  副問: Δρ(s vs xs, 対A, 同次元384d) = {delta_xs_s:.4f}")

    # ========================================
    # 分析2-3: RDM vs R@1 + 階層的回帰
    # ========================================
    print("\n--- 分析2-3: RDM vs R@1 + 階層的回帰 ---")

    all_pairs = load_all_results()

    # ペア特徴量の構築
    pair_features = []
    for (lx, ly), k_data in all_pairs.items():
        if lx not in available or ly not in available:
            continue
        if 500 not in k_data:
            continue
        rat_data = k_data[500].get("RAT", [])
        ridge_data = k_data[500].get("Ridge", [])
        if not rat_data:
            continue

        idx_x = available.index(lx)
        idx_y = available.index(ly)
        rdm_corr = rdm_result["matrix"][idx_x, idx_y]

        info_x = config.MATRIX_MODELS.get(lx, {})
        info_y = config.MATRIX_MODELS.get(ly, {})
        same_family = info_x.get("family") == info_y.get("family")

        pair_features.append({
            "lx": lx, "ly": ly,
            "rdm_corr": rdm_corr,
            "sim_mean": np.mean([d["sim_mean"] for d in rat_data]),
            "r1": np.mean([d["recall_at_1"] for d in rat_data]),
            "r1_ridge": np.mean([d["recall_at_1"] for d in ridge_data]) if ridge_data else 0,
            "same_family": same_family,
        })

    print(f"  ペア特徴量: {len(pair_features)} pairs")

    if len(pair_features) >= 10:
        # 散布図
        plot_rdm_vs_r1(pair_features, OUT_DIR / "fig_rdm_vs_r1.png")

        # 階層的回帰
        reg_results = hierarchical_regression(pair_features)
        print("\n  階層的回帰結果:")
        for name, res in reg_results.items():
            if name == "partial_correlation":
                pc = res["rdm_vs_r1_controlling_sim_mean"]
                print(f"    偏相関 (RDM|sim_mean): ρ={pc['spearman_rho']:.4f}, p={pc['p_value']:.4e}")
            else:
                print(f"    {res['formula']}: R²={res['R2']:.4f}"
                      + (f", ΔR²={res.get('delta_R2', 0):.4f}" if "delta_R2" in res else ""))

        # Q2判定
        delta_r2 = reg_results["model2"]["delta_R2"]
        print(f"\n  Q2判定: ΔR² = {delta_r2:.4f}")
        if delta_r2 >= 0.05:
            print(f"    → RDMは独立予測因子 (ΔR² ≥ 0.05)")
        else:
            print(f"    → RDMは冗長 (ΔR² < 0.05)")
    else:
        reg_results = {"error": "insufficient pairs for regression"}
        print("  [SKIP] ペア数不足（回帰には10+ペアが必要）")

    # ========================================
    # 分析4: スケール軸プロット
    # ========================================
    print("\n--- 分析4: スケール軸プロット ---")
    plot_scale_axis(rdm_result, all_pairs, OUT_DIR / "fig_scale_axis.png")

    # ========================================
    # 結果保存
    # ========================================
    elapsed = time.time() - start_time
    output = {
        "rdm_correlation_path": str(rdm_path),
        "pair_features": pair_features,
        "hierarchical_regression": reg_results,
        "q1_delta_rho": {
            "arctic_l_vs_xs_to_A": round(delta_rho, 4) if "O" in available and "Q" in available else None,
            "arctic_s_vs_xs_to_A_same_dim": round(delta_xs_s, 4) if "O" in available and "P" in available else None,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path = OUT_DIR / "d2_phase3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
