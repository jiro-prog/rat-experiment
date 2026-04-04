"""
M+N検証: 論文用Figure生成

Figure A: RDM correlation heatmap (14×14) with cluster boundaries
Figure B: RDM correlation vs RAT R@1 scatter (all 182 directed pairs)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import config

OUT_DIR = config.RESULTS_DIR / "mn_validation"

# ========================================
# Load data
# ========================================
with open(OUT_DIR / "rdm_correlation.json") as f:
    rdm_data = json.load(f)

labels = rdm_data["labels"]
rdm_mat = np.array([[rdm_data["rdm_spearman"][l1][l2] for l2 in labels] for l1 in labels])

# Load RAT results: existing D1 + new MN
with open(config.RESULTS_DIR / "d1_alignment" / "d1_results.json") as f:
    d1 = json.load(f)
d1_results = [r for r in d1["results"] if r["method"] == "RAT" and r["seed"] == 42 and r["K"] == 500]

with open(OUT_DIR / "mn_results.json") as f:
    mn = json.load(f)
mn_results = [r for r in mn["results"] if r["seed"] == 42 and r["K"] == 500]

all_rat = d1_results + mn_results

# ========================================
# Figure A: RDM correlation heatmap
# ========================================
fig, ax = plt.subplots(figsize=(8, 7))

# Reorder: BERT cluster first, then new cluster
bert_order = list("ABCDEFGHIJK")
new_order = list("LMN")
order = bert_order + new_order
idx_order = [labels.index(l) for l in order]
rdm_ordered = rdm_mat[np.ix_(idx_order, idx_order)]

# Model info for labels
model_info = config.MATRIX_MODELS
display_labels = []
for l in order:
    info = model_info[l]
    display_labels.append(f"{l} ({info['family']}, {info['dim']}d)")

norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=1.0)
im = ax.imshow(rdm_ordered, cmap="RdBu_r", norm=norm, aspect="equal")

# Cluster boundary
boundary = len(bert_order) - 0.5
ax.axhline(y=boundary, color="black", linewidth=2)
ax.axvline(x=boundary, color="black", linewidth=2)

# Cluster labels
ax.text(len(bert_order)/2, -1.2, "BERT-family cluster",
        ha="center", fontsize=10, fontweight="bold")
ax.text(len(bert_order) + len(new_order)/2, -1.2, "New-gen\ncluster",
        ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(order)))
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(display_labels, fontsize=7)

# Add correlation values
for i in range(len(order)):
    for j in range(len(order)):
        val = rdm_ordered[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        if i == j:
            continue
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=5.5, color=color)

cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("RDM Spearman ρ", fontsize=10)

ax.set_title("Representational Dissimilarity Matrix (RDM) Correlation\nBetween Embedding Spaces", fontsize=12)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_rdm_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'fig_rdm_heatmap.png'}")
plt.close()


# ========================================
# Figure B: RDM correlation vs RAT R@1
# ========================================
fig, ax = plt.subplots(figsize=(8, 5))

# Build (rdm_rho, r1) pairs for all directed pairs
points_bert = []  # intra-BERT
points_new = []   # intra-new
points_cross = [] # cross-cluster

bert_set = set("ABCDEFGHIJK")
new_set = set("LMN")

for r in all_rat:
    qm, dm = r["query_model"], r["db_model"]
    rho = rdm_data["rdm_spearman"][qm][dm]
    r1 = r["recall_at_1"] * 100

    if qm in bert_set and dm in bert_set:
        points_bert.append((rho, r1))
    elif qm in new_set and dm in new_set:
        points_new.append((rho, r1))
    else:
        points_cross.append((rho, r1))

for pts, color, label, marker, size, alpha in [
    (points_cross, "#BDBDBD", f"Cross-cluster (n={len(points_cross)})", "x", 40, 0.7),
    (points_bert, "#1E88E5", f"Intra-BERT (n={len(points_bert)})", "o", 30, 0.5),
    (points_new, "#E53935", f"Intra-NewGen (n={len(points_new)})", "s", 50, 0.8),
]:
    if pts:
        xs, ys = zip(*pts)
        kwargs = {"c": color, "alpha": alpha, "s": size, "label": label, "marker": marker}
        if marker != "x":
            kwargs["edgecolors"] = "white"
            kwargs["linewidth"] = 0.5
        ax.scatter(xs, ys, **kwargs)

# Threshold annotation
ax.axvline(x=0.1, color="gray", ls="--", alpha=0.5, lw=1)
ax.text(0.12, 95, "ρ ≈ 0.1\nthreshold", fontsize=8, color="gray", va="top")

ax.set_xlabel("RDM Spearman ρ (representational similarity)", fontsize=11)
ax.set_ylabel("RAT Recall@1 (%) at K=500", fontsize=11)
ax.set_title("Representational Compatibility Determines RAT Performance", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.15, 1.0)
ax.set_ylim(-5, 105)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_rdm_vs_r1.png", dpi=150)
print(f"Saved: {OUT_DIR / 'fig_rdm_vs_r1.png'}")
plt.close()

print("\nDone.")
