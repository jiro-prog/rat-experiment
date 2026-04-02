"""
Figure 1: RAT概念図

上段: テキスト×テキスト（同一アンカーテキスト）
下段: クロスモーダル（ロゼッタストーンアンカー）

パイプライン:
  Input → Encoder → Absolute Space → Anchor Similarity → Relative Space → Cosine Search
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(14, 8.5))

# Colors
C_MODEL_A = "#2196F3"  # blue
C_MODEL_B = "#FF9800"  # orange
C_MODEL_IMG = "#4CAF50"  # green
C_ANCHOR = "#9C27B0"   # purple
C_RELATIVE = "#E91E63"  # pink
C_BG = "#FAFAFA"
C_ARROW = "#555555"
C_MATCH = "#4CAF50"

def draw_box(ax, xy, w, h, text, color, fontsize=9, alpha=0.15, textcolor="black", bold=False):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight=weight)

def draw_arrow(ax, start, end, color=C_ARROW):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                               color=color, lw=1.5))

# ============================================
# Top panel: Text x Text
# ============================================
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Text × Text: Same anchors, different encoders", fontsize=12, fontweight="bold", pad=10)

# --- Model A path (top) ---
y_a = 3.0
draw_box(ax, (0.2, y_a), 1.8, 0.9, 'Query text\n"A dog runs"', "#EEEEEE", fontsize=8)
draw_arrow(ax, (2.0, y_a+0.45), (2.6, y_a+0.45), C_MODEL_A)
draw_box(ax, (2.6, y_a), 2.0, 0.9, "Model A\nMiniLM\n384d", C_MODEL_A, fontsize=8, bold=True)
draw_arrow(ax, (4.6, y_a+0.45), (5.2, y_a+0.45), C_MODEL_A)
draw_box(ax, (5.2, y_a+0.05), 1.4, 0.8, "Absolute\nspace", C_MODEL_A, fontsize=8, alpha=0.08)

# --- Model B path (bottom) ---
y_b = 1.0
draw_box(ax, (0.2, y_b), 1.8, 0.9, 'Query text\n"A dog runs"', "#EEEEEE", fontsize=8)
draw_arrow(ax, (2.0, y_b+0.45), (2.6, y_b+0.45), C_MODEL_B)
draw_box(ax, (2.6, y_b), 2.0, 0.9, "Model B\nE5-large\n1024d", C_MODEL_B, fontsize=8, bold=True)
draw_arrow(ax, (4.6, y_b+0.45), (5.2, y_b+0.45), C_MODEL_B)
draw_box(ax, (5.2, y_b+0.05), 1.4, 0.8, "Absolute\nspace", C_MODEL_B, fontsize=8, alpha=0.08)

# --- Shared anchors ---
y_mid = 2.0
draw_box(ax, (7.0, y_mid-0.15), 1.6, 0.9, "K=500\nShared\nAnchors", C_ANCHOR, fontsize=8, bold=True, alpha=0.2)

# Arrows from absolute to anchors
draw_arrow(ax, (6.6, y_a+0.35), (7.0, y_mid+0.55), C_ANCHOR)
draw_arrow(ax, (6.6, y_b+0.55), (7.0, y_mid+0.15), C_ANCHOR)

# --- Kernel similarity ---
draw_arrow(ax, (8.6, y_mid+0.3), (9.2, y_mid+0.3), C_ANCHOR)
ax.text(8.9, y_mid+0.75, r"$(x \cdot a + 1)^2$", ha="center", fontsize=8, color=C_ANCHOR)

# --- Relative space ---
draw_box(ax, (9.2, y_a-0.1), 1.8, 0.9, "Relative repr.\n500d", C_RELATIVE, fontsize=8, alpha=0.12)
draw_box(ax, (9.2, y_b+0.1), 1.8, 0.9, "Relative repr.\n500d + z-score", C_RELATIVE, fontsize=8, alpha=0.12)

# z-score label
ax.text(10.1, y_b-0.15, "DB-side z-score", ha="center", fontsize=7, color=C_RELATIVE, style="italic")

# --- Cosine match ---
draw_arrow(ax, (11.0, y_a+0.3), (11.8, y_mid+0.5), C_MATCH)
draw_arrow(ax, (11.0, y_b+0.6), (11.8, y_mid+0.2), C_MATCH)
draw_box(ax, (11.8, y_mid-0.1), 1.8, 0.8, "Cosine\nSearch", C_MATCH, fontsize=9, bold=True, alpha=0.15)

# "Same space!" annotation
ax.annotate("Same 500d space!", xy=(10.1, y_mid+0.35), fontsize=8,
            color=C_RELATIVE, ha="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_RELATIVE, alpha=0.8))

# ============================================
# Bottom panel: Cross-modal (Rosetta Stone)
# ============================================
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(b) Cross-Modal: Rosetta Stone anchors (paired image-caption)", fontsize=12, fontweight="bold", pad=10)

# --- Text path (top) ---
y_t = 3.0
draw_box(ax, (0.2, y_t), 1.8, 0.9, 'Text query\n"A dog runs"', "#EEEEEE", fontsize=8)
draw_arrow(ax, (2.0, y_t+0.45), (2.6, y_t+0.45), C_MODEL_A)
draw_box(ax, (2.6, y_t), 2.0, 0.9, "MiniLM\n(text encoder)\n384d", C_MODEL_A, fontsize=8, bold=True)
draw_arrow(ax, (4.6, y_t+0.45), (5.2, y_t+0.45), C_MODEL_A)
draw_box(ax, (5.2, y_t+0.05), 1.4, 0.8, "Text\nspace", C_MODEL_A, fontsize=8, alpha=0.08)

# --- Image path (bottom) ---
y_i = 1.0
draw_box(ax, (0.2, y_i), 1.8, 0.9, 'Image query\n[dog photo]', "#EEEEEE", fontsize=8)
draw_arrow(ax, (2.0, y_i+0.45), (2.6, y_i+0.45), C_MODEL_IMG)
draw_box(ax, (2.6, y_i), 2.0, 0.9, "CLIP ViT\n(image encoder)\n512d", C_MODEL_IMG, fontsize=8, bold=True)
draw_arrow(ax, (4.6, y_i+0.45), (5.2, y_i+0.45), C_MODEL_IMG)
draw_box(ax, (5.2, y_i+0.05), 1.4, 0.8, "Image\nspace", C_MODEL_IMG, fontsize=8, alpha=0.08)

# --- Rosetta Stone anchors ---
y_mid = 2.0

# Caption anchors (linked to text path)
draw_box(ax, (6.8, y_mid+0.55), 1.0, 0.65, "500\ncaptions", C_MODEL_A, fontsize=7, alpha=0.15)
# Image anchors (linked to image path)
draw_box(ax, (6.8, y_mid-0.6), 1.0, 0.65, "500\nimages", C_MODEL_IMG, fontsize=7, alpha=0.15)

# Connecting bracket for "same concept"
ax.plot([8.0, 8.3, 8.3, 8.0], [y_mid+0.85, y_mid+0.85, y_mid-0.25, y_mid-0.25],
        color=C_ANCHOR, linewidth=1.5)
ax.text(8.5, y_mid+0.3, "Same\nconcept\npairs", ha="left", va="center",
        fontsize=7, color=C_ANCHOR, weight="bold")

# Arrows
draw_arrow(ax, (6.6, y_t+0.35), (6.8, y_mid+0.85), C_ANCHOR)
draw_arrow(ax, (6.6, y_i+0.55), (6.8, y_mid-0.15), C_ANCHOR)

# --- Kernel + relative space ---
draw_arrow(ax, (9.4, y_mid+0.3), (9.8, y_mid+0.3), C_ANCHOR)

draw_box(ax, (9.8, y_t-0.1), 1.8, 0.9, "Relative repr.\n500d", C_RELATIVE, fontsize=8, alpha=0.12)
draw_box(ax, (9.8, y_i+0.1), 1.8, 0.9, "Relative repr.\n500d + z-score", C_RELATIVE, fontsize=8, alpha=0.12)

ax.annotate("Same 500d space!", xy=(10.7, y_mid+0.35), fontsize=8,
            color=C_RELATIVE, ha="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_RELATIVE, alpha=0.8))

# --- Cosine match ---
draw_arrow(ax, (11.6, y_t+0.3), (12.2, y_mid+0.5), C_MATCH)
draw_arrow(ax, (11.6, y_i+0.6), (12.2, y_mid+0.2), C_MATCH)
draw_box(ax, (12.2, y_mid-0.1), 1.5, 0.8, "Cosine\nSearch", C_MATCH, fontsize=9, bold=True, alpha=0.15)

# Key insight annotation
ax.text(7.0, 0.15, "Text encoder never sees images. Image encoder never sees text.\n"
        "500 paired anchors bridge the modality gap via shared similarity profiles.",
        fontsize=8, color="#555555", style="italic", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FFC107", alpha=0.7))

plt.tight_layout(h_pad=1.5)
plt.savefig("results/figure1_rat_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Figure 1 saved: results/figure1_rat_pipeline.png")
