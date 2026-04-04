"""
D2 Q3 + 分析D + 分析E: K交差点分析 + 修正版集計 + 方向非対称性

Q3: K交差点はRDM相関の関数か？
分析D: K=500 RAT vs best-linear 勝率（クラスター別）
分析E: 方向非対称性の定量化
"""
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT_DIR = config.RESULTS_DIR / "d2_scale"
PHASE2_PATH = OUT_DIR / "d2_phase2_results.json"
D1_CSV_PATH = config.RESULTS_DIR / "d1_alignment" / "d1_results.csv"
RDM_PATH = OUT_DIR / "d2_rdm_correlation.json"

ANCHOR_COUNTS = [10, 25, 50, 100, 200, 500]
BERT_MODELS = set("ABCDEFGHIJK")
ARCTIC_MODELS = set("OPNQ")


def load_data():
    with open(PHASE2_PATH) as f:
        p2 = json.load(f)
    with open(RDM_PATH) as f:
        rdm = json.load(f)

    # D1もロード
    d1_results = []
    if D1_CSV_PATH.exists():
        with open(D1_CSV_PATH) as f:
            for row in csv.DictReader(f):
                d1_results.append(row)

    return p2, rdm, d1_results


def get_pair_results(results, lx, ly):
    """ペアの全K×seed結果をstructured dictで返す。"""
    out = {}  # K -> seed -> method -> metrics
    for r in results:
        if r["query_model"] == lx and r["db_model"] == ly:
            K = r["K"] if isinstance(r["K"], int) else int(r["K"])
            seed = r["seed"] if isinstance(r["seed"], int) else int(r["seed"])
            method = r["method"]
            if K not in out:
                out[K] = {}
            if seed not in out[K]:
                out[K][seed] = {}
            out[K][seed][method] = {
                "recall_at_1": float(r["recall_at_1"]),
                "sim_mean": float(r["sim_mean"]),
            }
    return out


def find_crossover(pair_data, seeds):
    """K交差点を見つける。3seedの中央値で判定。"""
    crossover_results = []
    for K in ANCHOR_COUNTS:
        if K not in pair_data:
            continue
        rat_vals = []
        best_lin_vals = []
        for seed in seeds:
            if seed not in pair_data[K]:
                continue
            methods = pair_data[K][seed]
            rat_r1 = methods.get("RAT", {}).get("recall_at_1", 0)
            ridge_r1 = methods.get("Ridge", {}).get("recall_at_1", 0)
            proc_r1 = methods.get("Procrustes", {}).get("recall_at_1", 0)
            affine_r1 = methods.get("Affine", {}).get("recall_at_1", 0)
            best_lin = max(ridge_r1, proc_r1, affine_r1)
            rat_vals.append(rat_r1)
            best_lin_vals.append(best_lin)

        if rat_vals:
            crossover_results.append({
                "K": K,
                "rat_median": np.median(rat_vals),
                "lin_median": np.median(best_lin_vals),
                "rat_wins": np.median(rat_vals) > np.median(best_lin_vals),
            })

    # 交差点を見つける
    if not crossover_results:
        return None, crossover_results

    last_rat_win_K = None
    first_lin_win_K = None
    for cr in crossover_results:
        if cr["rat_wins"]:
            last_rat_win_K = cr["K"]
        elif first_lin_win_K is None:
            first_lin_win_K = cr["K"]

    if last_rat_win_K is None:
        crossover_K = 5  # <10 — linear always wins
    elif first_lin_win_K is None:
        crossover_K = 750  # >500 — RAT always wins
    else:
        crossover_K = (last_rat_win_K + first_lin_win_K) / 2

    return crossover_K, crossover_results


def main():
    p2, rdm_data, d1_results = load_data()
    rdm_matrix = rdm_data["rdm_spearman"]

    p2_seeds = [314, 999, 2025]
    d1_seeds = [42, 123, 7]

    # ========================================
    # Q3: K交差点 vs RDM相関
    # ========================================
    print("=" * 70)
    print("Q3: K交差点 vs RDM相関")
    print("=" * 70)

    # Phase 2ペア
    p2_pairs = set()
    for r in p2["results"]:
        p2_pairs.add((r["query_model"], r["db_model"]))

    crossover_data = []

    for lx, ly in sorted(p2_pairs):
        pair_data = get_pair_results(p2["results"], lx, ly)
        crossover_K, details = find_crossover(pair_data, p2_seeds)
        if crossover_K is None:
            continue

        rdm_rho = float(rdm_matrix.get(lx, {}).get(ly, 0))

        # sim_mean (K=500, median over seeds)
        sm_vals = []
        for seed in p2_seeds:
            if 500 in pair_data and seed in pair_data[500]:
                rat = pair_data[500][seed].get("RAT", {})
                if "sim_mean" in rat:
                    sm_vals.append(rat["sim_mean"])
        sim_mean = np.median(sm_vals) if sm_vals else 0

        # クラスター分類
        if lx in BERT_MODELS and ly in BERT_MODELS:
            cluster = "BERT-BERT"
        elif lx in ARCTIC_MODELS and ly in ARCTIC_MODELS:
            cluster = "Arctic-Arctic"
        else:
            cluster = "Cross"

        # K=500のR@1 (median)
        rat_k500 = []
        lin_k500 = []
        if 500 in pair_data:
            for seed in p2_seeds:
                if seed in pair_data[500]:
                    m = pair_data[500][seed]
                    rat_k500.append(m.get("RAT", {}).get("recall_at_1", 0))
                    lin_k500.append(max(
                        m.get("Ridge", {}).get("recall_at_1", 0),
                        m.get("Procrustes", {}).get("recall_at_1", 0),
                        m.get("Affine", {}).get("recall_at_1", 0),
                    ))

        crossover_data.append({
            "lx": lx, "ly": ly,
            "crossover_K": crossover_K,
            "rdm_rho": rdm_rho,
            "sim_mean": sim_mean,
            "cluster": cluster,
            "rat_k500": np.median(rat_k500) if rat_k500 else 0,
            "lin_k500": np.median(lin_k500) if lin_k500 else 0,
        })

    # テーブル出力
    print(f"\n{'Pair':>6} {'Cluster':>14} {'RDM':>6} {'sim':>6} "
          f"{'CrossK':>7} {'RAT@500':>8} {'Lin@500':>8}")
    print("-" * 65)

    # クロスクラスター除外でソート
    intra = [d for d in crossover_data if d["cluster"] != "Cross"]
    cross = [d for d in crossover_data if d["cluster"] == "Cross"]

    for d in sorted(intra, key=lambda x: -x["rdm_rho"]):
        print(f"{d['lx']}→{d['ly']:>2} {d['cluster']:>14} {d['rdm_rho']:>6.3f} "
              f"{d['sim_mean']:>6.3f} {d['crossover_K']:>7.0f} "
              f"{d['rat_k500']*100:>7.1f}% {d['lin_k500']*100:>7.1f}%")

    print(f"\n  Cross-cluster pairs ({len(cross)}): 全て RAT≈0%, Linear≈0%, 交差点不定")

    # Spearman相関（クラスター内のみ）
    rdm_vals = [d["rdm_rho"] for d in intra]
    cross_vals = [d["crossover_K"] for d in intra]
    sim_vals = [d["sim_mean"] for d in intra]

    rho_rdm, p_rdm = spearmanr(rdm_vals, cross_vals)
    rho_sim, p_sim = spearmanr(sim_vals, cross_vals)

    print(f"\n  クラスター内ペア (N={len(intra)}):")
    print(f"    交差点K vs RDM ρ:     Spearman ρ = {rho_rdm:.4f} (p={p_rdm:.4e})")
    print(f"    交差点K vs sim_mean:  Spearman ρ = {rho_sim:.4f} (p={p_sim:.4e})")

    # ========================================
    # Q3 散布図
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"BERT-BERT": "tab:blue", "Arctic-Arctic": "tab:red"}
    markers = {"BERT-BERT": "o", "Arctic-Arctic": "s"}

    # Plot 1: 交差点K vs RDM
    ax = axes[0]
    for d in intra:
        ax.scatter(d["rdm_rho"], d["crossover_K"],
                   c=colors[d["cluster"]], marker=markers[d["cluster"]],
                   s=40, alpha=0.7)
        if d["cluster"] == "Arctic-Arctic":
            ax.annotate(f"{d['lx']}{d['ly']}", (d["rdm_rho"], d["crossover_K"]),
                       fontsize=7, alpha=0.8, xytext=(3, 3),
                       textcoords="offset points")

    ax.set_xlabel("RDM Spearman ρ")
    ax.set_ylabel("Crossover K (RAT→Linear)")
    ax.set_title(f"RDM vs Crossover K\nSpearman ρ={rho_rdm:.3f} (p={p_rdm:.3e})")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="tab:blue", linestyle="", label="BERT-BERT"),
        plt.Line2D([0], [0], marker="s", color="tab:red", linestyle="", label="Arctic-Arctic"),
    ], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 800)

    # Plot 2: 交差点K vs sim_mean
    ax = axes[1]
    for d in intra:
        ax.scatter(d["sim_mean"], d["crossover_K"],
                   c=colors[d["cluster"]], marker=markers[d["cluster"]],
                   s=40, alpha=0.7)

    ax.set_xlabel("sim_mean")
    ax.set_ylabel("Crossover K (RAT→Linear)")
    ax.set_title(f"sim_mean vs Crossover K\nSpearman ρ={rho_sim:.3f} (p={p_sim:.3e})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 800)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_q3_crossover.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {fig_path}")

    # ========================================
    # 分析D: K=500 RAT vs best-linear 勝率（修正版）
    # ========================================
    print("\n" + "=" * 70)
    print("分析D: K=500 RAT vs Best Linear 勝率（修正版）")
    print("=" * 70)

    for cluster_name, cluster_filter in [
        ("BERT内 (A-K)", lambda lx, ly: lx in BERT_MODELS and ly in BERT_MODELS),
        ("Arctic内 (O,P,N,Q)", lambda lx, ly: lx in ARCTIC_MODELS and ly in ARCTIC_MODELS),
        ("全クラスター内", lambda lx, ly: (lx in BERT_MODELS and ly in BERT_MODELS) or
                                          (lx in ARCTIC_MODELS and ly in ARCTIC_MODELS)),
    ]:
        wins = 0
        total = 0
        deltas = []
        for d in crossover_data:
            if not cluster_filter(d["lx"], d["ly"]):
                continue
            total += 1
            delta = d["rat_k500"] - d["lin_k500"]
            deltas.append(delta)
            if delta > 0:
                wins += 1

        if total:
            print(f"\n  {cluster_name}: RAT wins {wins}/{total} ({wins/total*100:.0f}%)")
            print(f"    mean Δ = {np.mean(deltas)*100:+.1f}%p, "
                  f"median Δ = {np.median(deltas)*100:+.1f}%p")

    # D1との比較
    print("\n  --- D1結果との比較 (BERT内, seeds 42,123,7) ---")
    d1_k500_rat_wins = 0
    d1_k500_total = 0
    d1_k500_deltas = []
    d1_pair_data = {}

    for row in d1_results:
        if int(row["K"]) != 500:
            continue
        key = (row["query_model"], row["db_model"], int(row["seed"]))
        method = row["method"]
        r1 = float(row["recall_at_1"])
        if key not in d1_pair_data:
            d1_pair_data[key] = {}
        d1_pair_data[key][method] = r1

    for key, methods in d1_pair_data.items():
        lx, ly, seed = key
        if lx not in BERT_MODELS or ly not in BERT_MODELS:
            continue
        if "RAT" not in methods:
            continue
        rat = methods["RAT"]
        best_lin = max(methods.get("Ridge", 0), methods.get("Procrustes", 0),
                       methods.get("Affine", 0))
        delta = rat - best_lin
        d1_k500_deltas.append(delta)
        d1_k500_total += 1
        if delta > 0:
            d1_k500_rat_wins += 1

    if d1_k500_total:
        print(f"  D1 BERT内: RAT wins {d1_k500_rat_wins}/{d1_k500_total} "
              f"({d1_k500_rat_wins/d1_k500_total*100:.0f}%)")
        print(f"    mean Δ = {np.mean(d1_k500_deltas)*100:+.1f}%p")

    # ========================================
    # 分析E: 方向非対称性
    # ========================================
    print("\n" + "=" * 70)
    print("分析E: 方向非対称性（K=500, median over seeds）")
    print("=" * 70)

    def get_asymmetry(data_list, lx, ly):
        fwd = [d for d in data_list if d["lx"] == lx and d["ly"] == ly]
        rev = [d for d in data_list if d["lx"] == ly and d["ly"] == lx]
        if fwd and rev:
            return abs(fwd[0]["rat_k500"] - rev[0]["rat_k500"])
        return None

    # Arctic内
    print("\n  --- Arctic内 ---")
    arctic_list = ["O", "P", "N", "Q"]
    arctic_names = {"O": "xs(22M,384d)", "P": "s(33M,384d)",
                    "N": "m(109M,768d)", "Q": "l(335M,1024d)"}
    arctic_asymmetries = []
    for i, lx in enumerate(arctic_list):
        for ly in arctic_list[i+1:]:
            fwd = [d for d in crossover_data if d["lx"] == lx and d["ly"] == ly]
            rev = [d for d in crossover_data if d["lx"] == ly and d["ly"] == lx]
            if fwd and rev:
                asym = abs(fwd[0]["rat_k500"] - rev[0]["rat_k500"])
                arctic_asymmetries.append(asym)
                same_dim = config.MATRIX_MODELS[lx]["dim"] == config.MATRIX_MODELS[ly]["dim"]
                print(f"    {arctic_names[lx]:>16} ↔ {arctic_names[ly]:<16}: "
                      f"fwd={fwd[0]['rat_k500']*100:5.1f}% rev={rev[0]['rat_k500']*100:5.1f}% "
                      f"|Δ|={asym*100:5.1f}%p {'[同次元]' if same_dim else ''}")

    # BERT内（Phase 2の対照ペア）
    print("\n  --- BERT対照ペア ---")
    bert_asymmetries = []
    for lx, ly in [("A", "C"), ("A", "B"), ("B", "C")]:
        fwd = [d for d in crossover_data if d["lx"] == lx and d["ly"] == ly]
        rev = [d for d in crossover_data if d["lx"] == ly and d["ly"] == lx]
        if fwd and rev:
            asym = abs(fwd[0]["rat_k500"] - rev[0]["rat_k500"])
            bert_asymmetries.append(asym)
            print(f"    {lx:>2} ↔ {ly:<2}: "
                  f"fwd={fwd[0]['rat_k500']*100:5.1f}% rev={rev[0]['rat_k500']*100:5.1f}% "
                  f"|Δ|={asym*100:5.1f}%p")

    print(f"\n  Arctic内 非対称性: mean={np.mean(arctic_asymmetries)*100:.1f}%p, "
          f"max={np.max(arctic_asymmetries)*100:.1f}%p (N={len(arctic_asymmetries)})")
    if bert_asymmetries:
        print(f"  BERT対照 非対称性: mean={np.mean(bert_asymmetries)*100:.1f}%p, "
              f"max={np.max(bert_asymmetries)*100:.1f}%p (N={len(bert_asymmetries)})")

    # 非対称性 vs 次元差
    print("\n  --- 非対称性 vs 次元不一致 ---")
    same_dim_asym = [a for a, d in zip(arctic_asymmetries,
        [(lx, ly) for i, lx in enumerate(arctic_list) for ly in arctic_list[i+1:]])
        if config.MATRIX_MODELS[d[0]]["dim"] == config.MATRIX_MODELS[d[1]]["dim"]]
    diff_dim_asym = [a for a, d in zip(arctic_asymmetries,
        [(lx, ly) for i, lx in enumerate(arctic_list) for ly in arctic_list[i+1:]])
        if config.MATRIX_MODELS[d[0]]["dim"] != config.MATRIX_MODELS[d[1]]["dim"]]
    if same_dim_asym:
        print(f"    同次元ペア: mean |Δ|={np.mean(same_dim_asym)*100:.1f}%p (N={len(same_dim_asym)})")
    if diff_dim_asym:
        print(f"    異次元ペア: mean |Δ|={np.mean(diff_dim_asym)*100:.1f}%p (N={len(diff_dim_asym)})")

    # ========================================
    # 結果保存
    # ========================================
    output = {
        "q3_crossover": {
            "intra_cluster_pairs": [{k: v for k, v in d.items()} for d in intra],
            "cross_cluster_count": len(cross),
            "spearman_rdm_vs_crossK": {"rho": round(float(rho_rdm), 4), "p": float(p_rdm)},
            "spearman_sim_vs_crossK": {"rho": round(float(rho_sim), 4), "p": float(p_sim)},
        },
        "analysis_d_k500_winrate": {
            "note": "See printed output for details",
        },
        "analysis_e_asymmetry": {
            "arctic_mean": round(float(np.mean(arctic_asymmetries)) * 100, 1),
            "arctic_max": round(float(np.max(arctic_asymmetries)) * 100, 1),
            "bert_control_mean": round(float(np.mean(bert_asymmetries)) * 100, 1) if bert_asymmetries else None,
        },
    }
    out_path = OUT_DIR / "d2_q3_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()
