"""LaTeX表の生成: 11×11マトリクス + 回帰サマリー"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config

with open(config.RESULTS_DIR / "d2a_matrix.json") as f:
    data = json.load(f)

models = data["models"]
pair_results = data["pair_results"]
sim_stats = data["sim_stats"]
EXCLUDE = {"L"}
labels = [l for l in data["labels"] if l not in EXCLUDE]
n = len(labels)
li = {l: i for i, l in enumerate(labels)}

matrix = np.full((n, n), np.nan)
for p in pair_results:
    q, d = p["query"], p["db"]
    if q in EXCLUDE or d in EXCLUDE:
        continue
    matrix[li[q], li[d]] = max(p["baseline_r1"], p["zscore_db_r1"]) * 100

out = []

# ========================================
# Table: 11×11 matrix
# ========================================
out.append("% 11×11 RAT Recall@1 matrix (best of baseline/z-score)")
out.append(r"\begin{table*}[t]")
out.append(r"\centering")
out.append(r"\caption{RAT Recall@1 (\%) across 11 embedding models (110 directed pairs).")
out.append(r"Each cell: best of baseline and z-score DB-side normalization.")
out.append(r"Rows = query model, columns = DB model. \textbf{Bold} = same family.}")
out.append(r"\label{tab:matrix}")
out.append(r"\small")

cols = "l" + "r" * n
out.append(r"\begin{tabular}{" + cols + "}")
out.append(r"\toprule")

# Header row
cells = [r"Query $\downarrow$ DB $\rightarrow$"]
for l in labels:
    cells.append(r"\textbf{" + l + "}")
out.append(" & ".join(cells) + r" \\")

# Family sub-header
cells = [""]
for l in labels:
    cells.append(r"{\scriptsize " + models[l]["family"] + "}")
out.append(" & ".join(cells) + r" \\")
out.append(r"\midrule")

# Data rows
for i, lx in enumerate(labels):
    fam_x = models[lx]["family"]
    cells = [r"\textbf{" + lx + r"} {\scriptsize " + fam_x + "}"]
    for j, ly in enumerate(labels):
        if i == j:
            cells.append("---")
        else:
            val = matrix[i, j]
            same = models[lx]["family"] == models[ly]["family"]
            if same:
                cells.append(r"\textbf{" + f"{val:.0f}" + "}")
            elif val < 30:
                cells.append(r"{\color{red}" + f"{val:.0f}" + "}")
            else:
                cells.append(f"{val:.0f}")
    out.append(" & ".join(cells) + r" \\")

out.append(r"\bottomrule")
out.append(r"\end{tabular}")
out.append(r"\end{table*}")

# ========================================
# Table: Predictor summary
# ========================================
out.append("")
out.append("% Predictor summary table")
out.append(r"\begin{table}[t]")
out.append(r"\centering")
out.append(r"\caption{Predictors of RAT Recall@1 across model pairs (OLS regression).")
out.append(r"CLIP-excluded column verifies robustness beyond the CLIP--text model divide.}")
out.append(r"\label{tab:predictors}")
out.append(r"\small")
out.append(r"\begin{tabular}{lcc}")
out.append(r"\toprule")
out.append(r"Predictor & R$^2$ (all, $N$=110) & R$^2$ (w/o CLIP, $N$=90) \\")
out.append(r"\midrule")
out.append(r"max(sim\_mean) & 0.050 & \textbf{0.167} \\")
out.append(r"same\_family & 0.095 & 0.093 \\")
out.append(r"min(anchor entropy) & 0.586$^*$ & 0.033 \\")
out.append(r"max(sim\_mean) + same\_family & --- & 0.242 \\")
out.append(r"All features (7 vars) & 0.811 & --- \\")
out.append(r"\midrule")
out.append(r"\multicolumn{3}{l}{\textit{Spearman rank correlation:}} \\")
out.append(r"max(sim\_mean) vs R@1 & $\rho$=$-$0.34 & $\rho$=$-$\textbf{0.62} \\")
out.append(r"\bottomrule")
out.append(r"\multicolumn{3}{l}{\scriptsize $^*$Driven almost entirely by CLIP vs.\ text-model separation.} \\")
out.append(r"\end{tabular}")
out.append(r"\end{table}")

# ========================================
# Table: Model overview
# ========================================
out.append("")
out.append("% Model overview table")
out.append(r"\begin{table}[t]")
out.append(r"\centering")
out.append(r"\caption{Embedding models used in the 12-model diversity experiment. sim\_mean: mean pairwise cosine similarity among 500 query embeddings.}")
out.append(r"\label{tab:models}")
out.append(r"\small")
out.append(r"\begin{tabular}{clcccc}")
out.append(r"\toprule")
out.append(r"ID & Family & Params & Dim & Training & sim\_mean \\")
out.append(r"\midrule")

for l in labels:
    m = models[l]
    s = sim_stats[l]
    training_short = {
        "contrastive_distill": "Contr.+Distill",
        "weak_sup_distill": "WeakSup+Distill",
        "contrastive_retromae": "Contr.+RetroMAE",
        "clip_contrastive": "CLIP Contr.",
        "multistage_contrastive": "Multi-stage",
        "mse_distill": "MSE Distill",
    }.get(m["training"], m["training"])
    out.append(f"{l} & {m['family']} & {m['params']} & {m['dim']} & {training_short} & {s['sim_mean']:.3f}" + r" \\")

out.append(r"\bottomrule")
out.append(r"\end{tabular}")
out.append(r"\end{table}")

# 出力
text = "\n".join(out)
out_path = config.RESULTS_DIR / "d2a_latex_tables.tex"
with open(out_path, "w") as f:
    f.write(text)
print(f"保存: {out_path}")
print()
print(text)
