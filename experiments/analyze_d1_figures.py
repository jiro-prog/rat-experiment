"""
D1実験: RAT_auto計算 + メインサマリー + Figure 1 & 2 生成

Step 1: RAT_auto = max(A→B, B→A) for each undirected pair × K
Step 2: メインサマリーテーブル（5手法 × 6K）
Step 3: Figure 1 (R@1 vs K) + Figure 2 (sim_mean vs R@1 scatter)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import config

# ========================================
# データ読み込み
# ========================================
RESULTS_PATH = config.RESULTS_DIR / "d1_alignment" / "d1_results.json"
OUT_DIR = config.RESULTS_DIR / "d1_alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

results = data["results"]
models = data["models"]
ANCHOR_COUNTS = data["config"]["anchor_counts"]
SEEDS = data["config"]["seeds"]

labels = list(models.keys())
undirected_pairs = []
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i < j:
            undirected_pairs.append((a, b))


def get_results(method, qm, dm, K, seed=42):
    return [
        r for r in results
        if r["method"] == method and r["query_model"] == qm
        and r["db_model"] == dm and r["K"] == K and r["seed"] == seed
    ]


# ========================================
# Step 1: RAT_auto (bidirectional best)
# ========================================
print("=" * 70)
print("Step 1: RAT_auto計算")
print("=" * 70)

# 3 seeds分のRAT_autoを計算
rat_auto_records = []  # 無向ペアベースの集計用
for seed in SEEDS:
    for K in ANCHOR_COUNTS:
        for a, b in undirected_pairs:
            r_ab = [r for r in results if r["method"] == "RAT"
                    and r["query_model"] == a and r["db_model"] == b
                    and r["K"] == K and r["seed"] == seed]
            r_ba = [r for r in results if r["method"] == "RAT"
                    and r["query_model"] == b and r["db_model"] == a
                    and r["K"] == K and r["seed"] == seed]
            if not r_ab or not r_ba:
                continue
            r_ab, r_ba = r_ab[0], r_ba[0]

            # bidirectional best
            if r_ab["recall_at_1"] >= r_ba["recall_at_1"]:
                best = r_ab
            else:
                best = r_ba

            rat_auto_records.append({
                "pair": f"{a}-{b}",
                "K": K,
                "seed": seed,
                "recall_at_1": best["recall_at_1"],
                "recall_at_5": best["recall_at_5"],
                "recall_at_10": best["recall_at_10"],
                "mrr": best["mrr"],
                "chosen_dir": f"{best['query_model']}→{best['db_model']}",
                "sim_mean": best["sim_mean"],
            })

# ========================================
# Step 2: メインサマリーテーブル
# ========================================
print(f"\n{'='*70}")
print("Step 2: メインサマリーテーブル (seed=42)")
print(f"{'='*70}")

primary = [r for r in results if r["seed"] == 42]
primary_auto = [r for r in rat_auto_records if r["seed"] == 42]

# 手法別×K別の平均R@1集計
method_stats = {}  # method -> {K -> [r1_values]}

# RAT_orig: 132有向ペアの平均
for K in ANCHOR_COUNTS:
    for method in ["RAT", "Procrustes", "Ridge", "Affine"]:
        key = method
        if key not in method_stats:
            method_stats[key] = {}
        vals = [r["recall_at_1"] for r in primary
                if r["method"] == method and r["K"] == K]
        method_stats[key][K] = vals

# RAT_auto: 66無向ペアのbidirectional best
method_stats["RAT_auto"] = {}
for K in ANCHOR_COUNTS:
    vals = [r["recall_at_1"] for r in primary_auto if r["K"] == K]
    method_stats["RAT_auto"][K] = vals

# テーブル出力
print(f"\n{'Method':<14} {'N':>4}", end="")
for K in ANCHOR_COUNTS:
    print(f"  K={K:>3}", end="")
print()
print("-" * (18 + 8 * len(ANCHOR_COUNTS)))

for method in ["RAT", "RAT_auto", "Procrustes", "Ridge", "Affine"]:
    n_sample = len(method_stats[method][ANCHOR_COUNTS[0]])
    print(f"{method:<14} {n_sample:>4}", end="")
    for K in ANCHOR_COUNTS:
        vals = method_stats[method][K]
        print(f"  {np.mean(vals)*100:5.1f}%", end="")
    print()

# RAT_auto vs BestLinear の勝率テーブル
print(f"\n--- RAT_auto勝率 vs Best Linear ---")
for K in ANCHOR_COUNTS:
    auto_k = [r for r in primary_auto if r["K"] == K]
    wins = 0
    total = 0
    deltas = []
    for ar in auto_k:
        a, b = ar["pair"].split("-")
        # best linear for both directions
        lin_candidates = []
        for method in ["Procrustes", "Ridge", "Affine"]:
            for qm, dm in [(a, b), (b, a)]:
                matches = [r for r in primary if r["method"] == method
                           and r["query_model"] == qm and r["db_model"] == dm
                           and r["K"] == K]
                if matches:
                    lin_candidates.append(matches[0]["recall_at_1"])

        if lin_candidates:
            best_lin = max(lin_candidates)
            delta = ar["recall_at_1"] - best_lin
            deltas.append(delta)
            if delta > 0:
                wins += 1
            total += 1

    if total:
        print(f"  K={K:>3}: RAT_auto wins {wins}/{total} ({wins/total*100:.0f}%), "
              f"mean Δ={np.mean(deltas)*100:+.1f}%p")

# Seed間分散（RAT_auto）
print(f"\n--- RAT_auto Seed間分散 ---")
for K in ANCHOR_COUNTS:
    stds = []
    for a, b in undirected_pairs:
        vals = [r["recall_at_1"] for r in rat_auto_records
                if r["pair"] == f"{a}-{b}" and r["K"] == K]
        if len(vals) == len(SEEDS):
            stds.append(np.std(vals))
    if stds:
        print(f"  K={K:>3}: mean_std={np.mean(stds)*100:.2f}%p, "
              f"max_std={np.max(stds)*100:.2f}%p")

# ========================================
# Step 3: Figure 1 — R@1 vs K
# ========================================
print(f"\n{'='*70}")
print("Step 3: Figure生成")
print(f"{'='*70}")

fig, ax = plt.subplots(figsize=(8, 5))

# 色とスタイル定義
plot_config = {
    "RAT": {"color": "#2196F3", "marker": "o", "ls": "--", "lw": 1.5, "label": "RAT (single direction)"},
    "RAT_auto": {"color": "#1565C0", "marker": "s", "ls": "-", "lw": 2.5, "label": "RAT (bidirectional best)"},
    "Procrustes": {"color": "#FF9800", "marker": "^", "ls": "-.", "lw": 1.5, "label": "Procrustes (same-dim only)"},
    "Ridge": {"color": "#4CAF50", "marker": "D", "ls": "-", "lw": 1.5, "label": "Ridge (best λ)"},
    "Affine": {"color": "#9C27B0", "marker": "v", "ls": "-", "lw": 1.5, "label": "Affine (best λ)"},
}

for method, cfg in plot_config.items():
    means = []
    stds_vals = []
    for K in ANCHOR_COUNTS:
        # 3 seeds平均
        if method == "RAT_auto":
            all_seed_means = []
            for seed in SEEDS:
                vals = [r["recall_at_1"] for r in rat_auto_records
                        if r["K"] == K and r["seed"] == seed]
                all_seed_means.append(np.mean(vals))
            means.append(np.mean(all_seed_means) * 100)
            stds_vals.append(np.std(all_seed_means) * 100)
        else:
            all_seed_means = []
            for seed in SEEDS:
                vals = [r["recall_at_1"] for r in results
                        if r["method"] == method and r["K"] == K and r["seed"] == seed]
                if vals:
                    all_seed_means.append(np.mean(vals))
            if all_seed_means:
                means.append(np.mean(all_seed_means) * 100)
                stds_vals.append(np.std(all_seed_means) * 100)
            else:
                means.append(None)
                stds_vals.append(None)

    # Noneを除外してプロット
    valid_k = [K for K, m in zip(ANCHOR_COUNTS, means) if m is not None]
    valid_means = [m for m in means if m is not None]
    valid_stds = [s for s in stds_vals if s is not None]

    ax.errorbar(valid_k, valid_means, yerr=valid_stds,
                color=cfg["color"], marker=cfg["marker"],
                ls=cfg["ls"], lw=cfg["lw"],
                label=cfg["label"], markersize=6, capsize=3)

ax.set_xscale("log")
ax.set_xticks(ANCHOR_COUNTS)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xlabel("Number of Anchors (K)", fontsize=12)
ax.set_ylabel("Recall@1 (%)", fontsize=12)
ax.set_title("Cross-Model Retrieval: RAT vs Direct Alignment Methods", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# 交差点の領域をハイライト
ax.axvspan(50, 100, alpha=0.08, color="gray")
ax.text(70, 5, "crossover\nzone", ha="center", va="bottom",
        fontsize=9, color="gray", style="italic")

plt.tight_layout()
fig1_path = OUT_DIR / "fig1_r1_vs_k.png"
fig.savefig(fig1_path, dpi=150)
print(f"  Figure 1 saved: {fig1_path}")
plt.close()


# ========================================
# Figure 2: sim_mean vs R@1 (better/worse scatter)
# ========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, K in enumerate([50, 100, 500]):
    ax = axes[idx]
    better_sm, better_r1 = [], []
    worse_sm, worse_r1 = [], []

    for a, b in undirected_pairs:
        r_ab = [r for r in primary if r["method"] == "RAT"
                and r["query_model"] == a and r["db_model"] == b and r["K"] == K]
        r_ba = [r for r in primary if r["method"] == "RAT"
                and r["query_model"] == b and r["db_model"] == a and r["K"] == K]
        if not r_ab or not r_ba:
            continue
        r_ab, r_ba = r_ab[0], r_ba[0]

        if r_ab["recall_at_1"] >= r_ba["recall_at_1"]:
            better_sm.append(r_ab["sim_mean"])
            better_r1.append(r_ab["recall_at_1"] * 100)
            worse_sm.append(r_ba["sim_mean"])
            worse_r1.append(r_ba["recall_at_1"] * 100)
        else:
            better_sm.append(r_ba["sim_mean"])
            better_r1.append(r_ba["recall_at_1"] * 100)
            worse_sm.append(r_ab["sim_mean"])
            worse_r1.append(r_ab["recall_at_1"] * 100)

    ax.scatter(worse_sm, worse_r1, c="#E53935", alpha=0.6, s=30,
               label="Worse direction", edgecolors="white", linewidth=0.5)
    ax.scatter(better_sm, better_r1, c="#1E88E5", alpha=0.6, s=30,
               label="Better direction", edgecolors="white", linewidth=0.5)

    # sim_mean=0.65 境界線
    ax.axvline(x=0.65, color="gray", ls="--", alpha=0.5, lw=1)
    ax.text(0.66, 95, "harmful\nzone", fontsize=8, color="gray", va="top")

    ax.set_xlabel("sim_mean of FPS space", fontsize=11)
    ax.set_title(f"K = {K}", fontsize=12)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2)

    if idx == 0:
        ax.set_ylabel("Recall@1 (%)", fontsize=11)
        ax.legend(fontsize=9, loc="center right")

plt.suptitle("Direction Asymmetry: FPS Space Quality Determines RAT Performance",
             fontsize=13, y=1.02)
plt.tight_layout()
fig2_path = OUT_DIR / "fig2_simmean_scatter.png"
fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"  Figure 2 saved: {fig2_path}")
plt.close()


# ========================================
# Figure 3 (bonus): 非対称性の分布ヒストグラム
# ========================================
fig, ax = plt.subplots(figsize=(7, 4))

for method, color, label in [
    ("RAT", "#2196F3", "RAT"),
    ("Ridge", "#4CAF50", "Ridge"),
    ("Affine", "#9C27B0", "Affine"),
]:
    asyms = []
    for a, b in undirected_pairs:
        r_ab = [r for r in primary if r["method"] == method
                and r["query_model"] == a and r["db_model"] == b and r["K"] == 500]
        r_ba = [r for r in primary if r["method"] == method
                and r["query_model"] == b and r["db_model"] == a and r["K"] == 500]
        if r_ab and r_ba:
            asyms.append(abs(r_ab[0]["recall_at_1"] - r_ba[0]["recall_at_1"]) * 100)
    ax.hist(asyms, bins=20, alpha=0.5, color=color, label=f"{label} (mean={np.mean(asyms):.1f}%p)",
            range=(0, 100))

ax.set_xlabel("|R@1(X→Y) - R@1(Y→X)| (%p)", fontsize=11)
ax.set_ylabel("Number of pairs", fontsize=11)
ax.set_title("Direction Asymmetry Distribution (K=500)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig3_path = OUT_DIR / "fig3_asymmetry_hist.png"
fig.savefig(fig3_path, dpi=150)
print(f"  Figure 3 saved: {fig3_path}")
plt.close()

# ========================================
# RAT_auto結果をCSVに保存
# ========================================
import csv
csv_path = OUT_DIR / "d1_rat_auto.csv"
fields = ["pair", "K", "seed", "recall_at_1", "recall_at_5", "recall_at_10",
          "mrr", "chosen_dir", "sim_mean"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rat_auto_records)
print(f"\n  RAT_auto CSV saved: {csv_path}")
print(f"  Records: {len(rat_auto_records)}")

print(f"\n{'='*70}")
print("完了")
print(f"{'='*70}")
