"""バケット分析の棒グラフを生成する（Figure 6用）。"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import config

# データ読み込み
data_path = config.RESULTS_DIR / "phase4_analysis_correlation.json"
with open(data_path) as f:
    data = json.load(f)

buckets = data["bucket_analysis"]

# 棒グラフ
labels = []
dxe_vals = []
axe_vals = []
for b in buckets:
    lo, hi = b["range"]
    labels.append(f"[{lo:.2f}, {hi:.2f})")
    dxe_vals.append(b["dxe_recall_at_1"] * 100)
    axe_vals.append(b["axe_recall_at_1"] * 100)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar(x - width/2, axe_vals, width, label="A×E (MiniLM × CLIP-img)", color="#2196F3", edgecolor="white")
bars2 = ax.bar(x + width/2, dxe_vals, width, label="D×E (CLIP-text × CLIP-img)", color="#FF9800", edgecolor="white")

ax.set_xlabel("CLIP Native Text-Image Similarity (Quartile)", fontsize=12)
ax.set_ylabel("RAT Recall@1 (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"], fontsize=11)
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, 35)
ax.grid(axis="y", alpha=0.3)

# 値ラベル
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
out_path = Path(__file__).parent.parent / "paper" / "figures" / "bucket_analysis.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"保存: {out_path}")
