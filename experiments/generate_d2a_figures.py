"""
Direction 2A: フィギュア生成（修正版）

Figure 1: max(sim_mean) vs Recall@1 — CLIP除外でもρ=-0.62で成立
Figure 2: z-score効果 vs DB側sim_mean
Figure 3: 11×11 ヒートマップ
Figure 4: ファミリー効果の分離（サイズ梯子）
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config

# データ読み込み
with open(config.RESULTS_DIR / "d2a_matrix.json") as f:
    data = json.load(f)

sim_stats = data["sim_stats"]
models = data["models"]
pair_results = data["pair_results"]

EXCLUDE = {"L"}
OUT_DIR = Path(__file__).parent.parent / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ペアデータ構築
points = []
for p in pair_results:
    q, d = p["query"], p["db"]
    if q in EXCLUDE or d in EXCLUDE:
        continue
    best_r1 = max(p["baseline_r1"], p["zscore_db_r1"])
    same_family = models[q]["family"] == models[d]["family"]
    has_clip = q == "D" or d == "D"

    points.append({
        "q": q, "d": d,
        "best_r1": best_r1,
        "same_family": same_family,
        "has_clip": has_clip,
        "max_sim_mean": max(sim_stats[q]["sim_mean"], sim_stats[d]["sim_mean"]),
        "min_ent": min(p["anchor_entropy_query"], p["anchor_entropy_db"]),
        "db_sim_mean": sim_stats[d]["sim_mean"],
        "delta_zscore": (p["zscore_db_r1"] - p["baseline_r1"]) * 100,
    })

# ========================================
# Figure 1: max(sim_mean) vs R@1
# ========================================
fig, ax = plt.subplots(figsize=(7, 5))

cat_same = [p for p in points if p["same_family"] and not p["has_clip"]]
cat_clip = [p for p in points if p["has_clip"]]
cat_diff = [p for p in points if not p["same_family"] and not p["has_clip"]]

ax.scatter(
    [p["max_sim_mean"] for p in cat_diff],
    [p["best_r1"] * 100 for p in cat_diff],
    c="#7B8FA1", marker="o", s=40, alpha=0.55, label="Cross-family", zorder=2,
)
ax.scatter(
    [p["max_sim_mean"] for p in cat_same],
    [p["best_r1"] * 100 for p in cat_same],
    c="#2196F3", marker="s", s=55, alpha=0.8, label="Same family", zorder=3,
    edgecolors="white", linewidths=0.5,
)
ax.scatter(
    [p["max_sim_mean"] for p in cat_clip],
    [p["best_r1"] * 100 for p in cat_clip],
    c="#FF5722", marker="^", s=55, alpha=0.8, label="CLIP-involved", zorder=4,
    edgecolors="white", linewidths=0.5,
)

# Spearman全体 + CLIP除外
all_x = np.array([p["max_sim_mean"] for p in points])
all_y = np.array([p["best_r1"] for p in points])
rho_all, _ = spearmanr(all_x, all_y)

nc = [p for p in points if not p["has_clip"]]
nc_x = np.array([p["max_sim_mean"] for p in nc])
nc_y = np.array([p["best_r1"] for p in nc])
rho_nc, _ = spearmanr(nc_x, nc_y)

ax.text(
    0.97, 0.97,
    f"All pairs: ρ = {rho_all:.2f} (N={len(points)})\n"
    f"w/o CLIP: ρ = {rho_nc:.2f} (N={len(nc)})",
    transform=ax.transAxes, fontsize=10,
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
)

ax.set_xlabel("max(sim_mean$_Q$, sim_mean$_{DB}$)", fontsize=12)
ax.set_ylabel("Best Recall@1 (%)", fontsize=12)
ax.legend(fontsize=10, loc="lower left")
ax.grid(alpha=0.2)
ax.set_ylim(-3, 103)

plt.tight_layout()
plt.savefig(OUT_DIR / "d2a_sim_mean_vs_r1.png", dpi=300, bbox_inches="tight")
print(f"保存: d2a_sim_mean_vs_r1.png")
plt.close()

# ========================================
# Figure 2: z-score効果 vs DB側sim_mean
# ========================================
fig2, ax2 = plt.subplots(figsize=(7, 5))

clip_pts = [p for p in points if p["has_clip"]]
other_pts = [p for p in points if not p["has_clip"]]

ax2.scatter(
    [p["db_sim_mean"] for p in other_pts],
    [p["delta_zscore"] for p in other_pts],
    c="#5C6BC0", marker="o", s=40, alpha=0.55, label="Text models", zorder=2,
)
ax2.scatter(
    [p["db_sim_mean"] for p in clip_pts],
    [p["delta_zscore"] for p in clip_pts],
    c="#FF5722", marker="^", s=55, alpha=0.8, label="CLIP-involved", zorder=3,
    edgecolors="white", linewidths=0.5,
)

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

# 注釈: 効くゾーンと壊れるゾーン
ax2.axvspan(-0.05, 0.15, alpha=0.06, color="green")
ax2.axvspan(0.65, 0.80, alpha=0.06, color="red")
ax2.text(0.07, -72, "z-score\nhelpful", fontsize=9, color="#2E7D32", ha="center", style="italic")
ax2.text(0.72, 65, "z-score\nharmful", fontsize=9, color="#C62828", ha="center", style="italic")

all_dbsm = np.array([p["db_sim_mean"] for p in points])
all_dz = np.array([p["delta_zscore"] for p in points])
rho_z, _ = spearmanr(all_dbsm, all_dz)
ax2.text(
    0.97, 0.03,
    f"ρ = {rho_z:.2f}",
    transform=ax2.transAxes, fontsize=11,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
)

ax2.set_xlabel("DB-side Cosine Similarity Mean", fontsize=12)
ax2.set_ylabel("Δ Recall@1 (z-score − baseline) [pp]", fontsize=12)
ax2.legend(fontsize=10, loc="upper left")
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / "d2a_zscore_effect.png", dpi=300, bbox_inches="tight")
print(f"保存: d2a_zscore_effect.png")
plt.close()

# ========================================
# Figure 3: 11×11 ヒートマップ
# ========================================
fig3, ax3 = plt.subplots(figsize=(8, 7))

labels_clean = [l for l in data["labels"] if l not in EXCLUDE]
n = len(labels_clean)
label_idx = {l: i for i, l in enumerate(labels_clean)}

matrix = np.full((n, n), np.nan)
for p in pair_results:
    q, d = p["query"], p["db"]
    if q in EXCLUDE or d in EXCLUDE:
        continue
    best = max(p["baseline_r1"], p["zscore_db_r1"]) * 100
    matrix[label_idx[q], label_idx[d]] = best

im = ax3.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="equal")

tick_labels = [f"{l} ({models[l]['family']})" for l in labels_clean]
ax3.set_xticks(range(n))
ax3.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax3.set_yticks(range(n))
ax3.set_yticklabels(tick_labels, fontsize=9)
ax3.set_xlabel("DB model", fontsize=11)
ax3.set_ylabel("Query model", fontsize=11)

for i in range(n):
    for j in range(n):
        if i == j:
            ax3.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
        else:
            val = matrix[i, j]
            color = "white" if val < 40 else "black"
            ax3.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color=color)

plt.colorbar(im, ax=ax3, shrink=0.8, label="Best Recall@1 (%)")
plt.tight_layout()
plt.savefig(OUT_DIR / "d2a_heatmap.png", dpi=300, bbox_inches="tight")
print(f"保存: d2a_heatmap.png")
plt.close()

# ========================================
# Figure 4: ファミリー内サイズ梯子
# ========================================
fig4, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

families = {
    "BGE": {"models": [("C", "33M"), ("K", "109M"), ("E", "335M")], "color": "#4CAF50"},
    "E5":  {"models": [("F", "33M"), ("G", "118M"), ("B", "560M")], "color": "#2196F3"},
    "GTE": {"models": [("H", "33M"), ("I", "335M")], "color": "#FF9800"},
}

pair_dict = {}
for p in pair_results:
    q, d = p["query"], p["db"]
    if q in EXCLUDE or d in EXCLUDE:
        continue
    pair_dict[(q, d)] = max(p["baseline_r1"], p["zscore_db_r1"]) * 100

for idx, (fam, info) in enumerate(families.items()):
    ax = axes[idx]
    members = info["models"]
    color = info["color"]

    # ファミリー内ペア
    in_labels = []
    in_vals = []
    for i, (lx, sx) in enumerate(members):
        for ly, sy in members:
            if lx == ly:
                continue
            key = (lx, ly)
            if key in pair_dict:
                in_labels.append(f"{lx}({sx})→{ly}({sy})")
                in_vals.append(pair_dict[key])

    # ファミリー外平均（各メンバーについて）
    all_labels_clean_set = set(labels_clean) - {m[0] for m in members}
    out_vals = []
    for lx, _ in members:
        for ly in all_labels_clean_set:
            if (lx, ly) in pair_dict:
                out_vals.append(pair_dict[(lx, ly)])

    x = np.arange(len(in_labels))
    bars = ax.bar(x, in_vals, color=color, edgecolor="white", width=0.7, alpha=0.85)
    ax.axhline(np.mean(out_vals), color="gray", linestyle="--", linewidth=1.2,
               label=f"Cross-family avg ({np.mean(out_vals):.0f}%)")

    for bar, val in zip(bars, in_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(in_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"{fam} Family", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.2)

axes[0].set_ylabel("Recall@1 (%)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "d2a_family_size.png", dpi=300, bbox_inches="tight")
print(f"保存: d2a_family_size.png")
plt.close()

print("\n全フィギュア生成完了")
