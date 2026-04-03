"""Phase 1: Regression analysis for estimate_compatibility().

Fits linear and degree-2 polynomial on max(sim_mean) vs best_r1.
Reports R², adjusted R², AIC, residual diagnostics, same_family effects.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

# ── Load data ──
df = pd.read_csv("results/d2a_pair_features.csv")
print(f"Total pairs: {len(df)}")

# Identify CLIP pairs (model D)
clip_mask = (df["query"] == "D") | (df["db"] == "D")
print(f"CLIP pairs: {clip_mask.sum()}")
print(f"Non-CLIP pairs: {(~clip_mask).sum()}")

df["max_sim_mean"] = np.maximum(df["query_sim_mean"], df["db_sim_mean"])

# ── Fit on 90 pairs (CLIP excluded) ──
df90 = df[~clip_mask].copy()
X = df90["max_sim_mean"].values
y = df90["best_r1"].values * 100  # convert to %

print(f"\n{'='*60}")
print(f"Fitting on {len(df90)} pairs (CLIP excluded)")
print(f"max_sim_mean range: [{X.min():.4f}, {X.max():.4f}]")
print(f"best_r1 range: [{y.min():.1f}%, {y.max():.1f}%]")

# ── Linear fit ──
slope, intercept, r_lin, p_lin, se_lin = stats.linregress(X, y)
y_pred_lin = slope * X + intercept
resid_lin = y - y_pred_lin
r2_lin = r_lin**2
n = len(X)
adj_r2_lin = 1 - (1 - r2_lin) * (n - 1) / (n - 2)
sse_lin = np.sum(resid_lin**2)
aic_lin = n * np.log(sse_lin / n) + 2 * 2  # 2 params

print(f"\n--- Linear: y = {slope:.2f}*x + {intercept:.2f} ---")
print(f"R² = {r2_lin:.4f}, adj R² = {adj_r2_lin:.4f}")
print(f"AIC = {aic_lin:.2f}")
print(f"Residual std = {resid_lin.std():.2f}%")

# ── Degree 2 polynomial fit ──
coeffs2 = np.polyfit(X, y, 2)
y_pred_poly2 = np.polyval(coeffs2, X)
resid_poly2 = y - y_pred_poly2
ss_res2 = np.sum(resid_poly2**2)
ss_tot = np.sum((y - y.mean())**2)
r2_poly2 = 1 - ss_res2 / ss_tot
adj_r2_poly2 = 1 - (1 - r2_poly2) * (n - 1) / (n - 3)
aic_poly2 = n * np.log(ss_res2 / n) + 2 * 3  # 3 params

print(f"\n--- Degree 2: y = {coeffs2[0]:.2f}*x² + {coeffs2[1]:.2f}*x + {coeffs2[2]:.2f} ---")
print(f"R² = {r2_poly2:.4f}, adj R² = {adj_r2_poly2:.4f}")
print(f"AIC = {aic_poly2:.2f}")
print(f"Residual std = {resid_poly2.std():.2f}%")

# ── Model comparison ──
print(f"\n{'='*60}")
print("Model Comparison:")
print(f"  {'':20s} {'Linear':>10s} {'Poly-2':>10s}")
print(f"  {'R²':20s} {r2_lin:10.4f} {r2_poly2:10.4f}")
print(f"  {'Adj R²':20s} {adj_r2_lin:10.4f} {adj_r2_poly2:10.4f}")
print(f"  {'AIC':20s} {aic_lin:10.2f} {aic_poly2:10.2f}")
print(f"  {'Residual std':20s} {resid_lin.std():10.2f} {resid_poly2.std():10.2f}")

delta_aic = aic_lin - aic_poly2
print(f"\n  ΔAIC (linear - poly2) = {delta_aic:.2f}")
if abs(delta_aic) < 2:
    print("  → Models are essentially equivalent. Prefer linear (simpler).")
elif delta_aic > 0:
    print("  → Poly-2 is better (ΔAIC > 2).")
else:
    print("  → Linear is better (ΔAIC < -2).")

# ── Residual normality (Shapiro-Wilk) ──
# Use whichever model we'd recommend
for label, resid in [("Linear", resid_lin), ("Poly-2", resid_poly2)]:
    sw_stat, sw_p = stats.shapiro(resid)
    print(f"\n  Shapiro-Wilk ({label}): W={sw_stat:.4f}, p={sw_p:.4f}")
    if sw_p < 0.05:
        print(f"    → Non-normal residuals (p < 0.05). Use percentile-based CI.")
    else:
        print(f"    → Normal residuals (p ≥ 0.05). ±1σ CI is valid.")

# ── Residual pattern by max_sim_mean bins ──
print(f"\n{'='*60}")
print("Residual pattern by max_sim_mean region (Linear model):")
bins = [(0.0, 0.2, "low"), (0.2, 0.5, "mid"), (0.5, 0.8, "high")]
for lo, hi, label in bins:
    mask = (X >= lo) & (X < hi)
    if mask.sum() == 0:
        continue
    r = resid_lin[mask]
    print(f"  {label:5s} [{lo:.1f}, {hi:.1f}): n={mask.sum():3d}, "
          f"mean_resid={r.mean():+.2f}, std_resid={r.std():.2f}, "
          f"min={r.min():.2f}, max={r.max():.2f}")

# ── same_family analysis ──
print(f"\n{'='*60}")
print("same_family effect (on 90 pairs, CLIP excluded):")
sf_mask = df90["same_family"].values.astype(bool)
print(f"  Same family: n={sf_mask.sum()}, mean R@1={y[sf_mask].mean():.1f}%, "
      f"mean residual (linear)={resid_lin[sf_mask].mean():+.2f}")
print(f"  Diff family: n=(~sf_mask).sum()={( ~sf_mask).sum()}, mean R@1={y[~sf_mask].mean():.1f}%, "
      f"mean residual (linear)={resid_lin[~sf_mask].mean():+.2f}")

# Check if same_family bonus is constant or varies with max_sim_mean
print("\n  same_family residual by compression region:")
for lo, hi, label in bins:
    mask_region = (X >= lo) & (X < hi)
    for sf_val, sf_label in [(True, "same"), (False, "diff")]:
        mask = mask_region & (sf_mask == sf_val)
        if mask.sum() == 0:
            continue
        r = resid_lin[mask]
        print(f"    {label:5s} {sf_label:4s}: n={mask.sum():3d}, mean_resid={r.mean():+.2f}")

# ── CLIP pairs analysis ──
print(f"\n{'='*60}")
print("CLIP pairs (excluded from fit):")
df_clip = df[clip_mask].copy()
X_clip = df_clip["max_sim_mean"].values
y_clip = df_clip["best_r1"].values * 100
y_clip_pred_lin = slope * X_clip + intercept
resid_clip = y_clip - y_clip_pred_lin
print(f"  n={len(df_clip)}")
print(f"  Actual R@1: mean={y_clip.mean():.1f}%, range=[{y_clip.min():.1f}, {y_clip.max():.1f}]")
print(f"  Predicted (linear): mean={y_clip_pred_lin.mean():.1f}%")
print(f"  Residual: mean={resid_clip.mean():+.2f}, std={resid_clip.std():.2f}")
print(f"  → CLIP pairs systematically {'under' if resid_clip.mean() < 0 else 'over'}-predicted "
      f"by {abs(resid_clip.mean()):.1f}%")

# ── Summary for decision ──
print(f"\n{'='*60}")
print("SUMMARY FOR DECISION:")
print(f"  1. Model choice: {'Linear' if delta_aic <= 2 else 'Poly-2'} recommended")
print(f"  2. Coefficients (linear): slope={slope:.4f}, intercept={intercept:.4f}")
print(f"  3. Coefficients (poly-2): {coeffs2}")
print(f"  4. Residual std (linear): {resid_lin.std():.2f}%")
print(f"  5. same_family mean residual: {resid_lin[sf_mask].mean():+.2f}%")

# ── Save results ──
results = {
    "n_pairs": int(n),
    "linear": {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2_lin),
        "adj_r2": float(adj_r2_lin),
        "aic": float(aic_lin),
        "residual_std": float(resid_lin.std()),
    },
    "poly2": {
        "coeffs": [float(c) for c in coeffs2],
        "r2": float(r2_poly2),
        "adj_r2": float(adj_r2_poly2),
        "aic": float(aic_poly2),
        "residual_std": float(resid_poly2.std()),
    },
    "delta_aic": float(delta_aic),
    "shapiro_wilk_linear": {
        "W": float(stats.shapiro(resid_lin)[0]),
        "p": float(stats.shapiro(resid_lin)[1]),
    },
    "same_family_mean_residual": float(resid_lin[sf_mask].mean()),
    "diff_family_mean_residual": float(resid_lin[~sf_mask].mean()),
    "clip_mean_residual": float(resid_clip.mean()),
}
with open("results/phase1_regression.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/phase1_regression.json")
