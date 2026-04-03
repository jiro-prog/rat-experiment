"""Phase 1 supplemental: residual percentiles + tier boundary analysis."""

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/d2a_pair_features.csv")
clip_mask = (df["query"] == "D") | (df["db"] == "D")
df90 = df[~clip_mask].copy()
df90["max_sim_mean"] = np.maximum(df90["query_sim_mean"], df90["db_sim_mean"])

X = df90["max_sim_mean"].values
y = df90["best_r1"].values * 100

# Linear fit
slope, intercept, *_ = stats.linregress(X, y)
resid = y - (slope * X + intercept)

# ── Residual percentiles ──
print("=" * 60)
print("Residual percentiles (linear, 90 pairs):")
for p in [5, 10, 16, 25, 50, 75, 84, 90, 95]:
    print(f"  {p:3d}th: {np.percentile(resid, p):+.2f}%")

print(f"\n  16th-84th band width: {np.percentile(resid, 84) - np.percentile(resid, 16):.2f}%")

# ── R@1 distribution by max_sim_mean bins ──
print(f"\n{'=' * 60}")
print("R@1 distribution by max_sim_mean bins (all 90 pairs):")

bins = [
    (0.00, 0.30), (0.30, 0.45), (0.45, 0.55),
    (0.55, 0.65), (0.65, 0.72), (0.72, 0.80),
]
sf = df90["same_family"].values.astype(bool)

print(f"\n  {'Bin':>12s}  {'n':>3s}  {'n_sf':>4s}  {'med':>5s}  {'p25':>5s}  {'p75':>5s}  {'min':>5s}  {'max':>5s}")
for lo, hi in bins:
    mask = (X >= lo) & (X < hi)
    if mask.sum() == 0:
        print(f"  [{lo:.2f},{hi:.2f})  {0:3d}  ---")
        continue
    vals = y[mask]
    n_sf = sf[mask].sum()
    print(f"  [{lo:.2f},{hi:.2f})  {mask.sum():3d}  {n_sf:4d}  "
          f"{np.median(vals):5.1f}  {np.percentile(vals, 25):5.1f}  "
          f"{np.percentile(vals, 75):5.1f}  {vals.min():5.1f}  {vals.max():5.1f}")

# ── Same family vs diff family by bin ──
print(f"\n{'=' * 60}")
print("R@1 by bin, split by same_family:")
print(f"  {'Bin':>12s}  {'group':>5s}  {'n':>3s}  {'med':>5s}  {'p25':>5s}  {'p75':>5s}  {'min':>5s}  {'max':>5s}")
for lo, hi in bins:
    mask = (X >= lo) & (X < hi)
    if mask.sum() == 0:
        continue
    for label, sf_val in [("same", True), ("diff", False)]:
        m = mask & (sf == sf_val)
        if m.sum() == 0:
            continue
        vals = y[m]
        print(f"  [{lo:.2f},{hi:.2f})  {label:>5s}  {m.sum():3d}  "
              f"{np.median(vals):5.1f}  {np.percentile(vals, 25):5.1f}  "
              f"{np.percentile(vals, 75):5.1f}  {vals.min():5.1f}  {vals.max():5.1f}")

# ── Tier boundary candidates ──
print(f"\n{'=' * 60}")
print("Tier boundary candidates:")
print()

candidates = [
    # (name, high_upper, moderate_upper)
    ("A: conservative", 0.40, 0.65),
    ("B: balanced",     0.45, 0.72),
    ("C: aggressive",   0.50, 0.72),
]

for name, hi_bound, mod_bound in candidates:
    print(f"  --- {name}: high < {hi_bound}, moderate < {mod_bound}, low >= {mod_bound} ---")
    tiers = {"high": (0, hi_bound), "moderate": (hi_bound, mod_bound), "low": (mod_bound, 1.0)}
    for tier, (lo, hi) in tiers.items():
        mask = (X >= lo) & (X < hi)
        if mask.sum() == 0:
            print(f"    {tier:10s}: n=0")
            continue
        vals = y[mask]
        n_sf = sf[mask].sum()
        print(f"    {tier:10s}: n={mask.sum():3d} (sf={n_sf:2d}), "
              f"median={np.median(vals):5.1f}, IQR=[{np.percentile(vals, 25):5.1f}, {np.percentile(vals, 75):5.1f}], "
              f"range=[{vals.min():5.1f}, {vals.max():5.1f}]")

    # Check tier separation quality
    for tier, (lo, hi) in tiers.items():
        mask = (X >= lo) & (X < hi)
        if mask.sum() == 0:
            continue
    print()
