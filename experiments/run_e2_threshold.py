#!/usr/bin/env python3
"""E2: sim_mean threshold sensitivity analysis.

Compares DB-side vs query-side sim_mean as z-score threshold criterion.

Input:  results/d2a_matrix.json (132 pairs, K=500, baseline & zscore R@1)
Output: results/e2_threshold_sensitivity.csv
"""

import json
import csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
D2A_PATH = RESULTS_DIR / "d2a_matrix.json"

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def load_data():
    with open(D2A_PATH) as f:
        d2a = json.load(f)

    sim_stats = d2a["sim_stats"]  # model -> {sim_mean, sim_std, ...}
    pair_results = d2a["pair_results"]  # list of {query, db, baseline_r1, zscore_db_r1, ...}

    # Build per-pair records with both sim_mean sides
    pairs = []
    for pr in pair_results:
        q, d = pr["query"], pr["db"]
        pairs.append({
            "query": q,
            "db": d,
            "baseline_r1": pr["baseline_r1"],
            "zscore_r1": pr["zscore_db_r1"],
            "query_sim_mean": sim_stats[q]["sim_mean"],
            "db_sim_mean": sim_stats[d]["sim_mean"],
        })
    return pairs


def adaptive_r1(pair, t_harmful, side):
    """Select baseline or z-score R@1 based on threshold.

    side: 'db_sim_mean' or 'query_sim_mean'
    Logic: sim_mean >= t_harmful → baseline (harmful zone, skip z-score)
           else → z-score
    Note: no lower threshold — d1_alignment implementation had none,
          and z-score gives the largest gains at very low sim_mean.
    """
    sm = pair[side]
    if sm >= t_harmful:
        return pair["baseline_r1"]
    else:
        return pair["zscore_r1"]


def run_e2():
    print("Loading d2a_matrix.json...")
    pairs = load_data()
    print(f"  Pairs: {len(pairs)}")

    # Show sim_mean distribution
    q_sms = sorted(set(p["query_sim_mean"] for p in pairs))
    d_sms = sorted(set(p["db_sim_mean"] for p in pairs))
    print(f"  Query sim_mean range: {min(q_sms):.3f} - {max(q_sms):.3f}")
    print(f"  DB sim_mean range:    {min(d_sms):.3f} - {max(d_sms):.3f}")

    # Compute per-pair delta
    deltas = [p["zscore_r1"] - p["baseline_r1"] for p in pairs]
    print(f"\n  Δ(zscore - baseline): mean={mean(deltas):.3f}, "
          f"min={min(deltas):.3f}, max={max(deltas):.3f}")
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    print(f"  z-score helps: {n_positive}, hurts: {n_negative}, neutral: {n_zero}")

    rows = []

    # Always baseline
    bl_r1s = [p["baseline_r1"] for p in pairs]
    bl_mean = mean(bl_r1s) * 100
    rows.append({
        "strategy": "always_baseline",
        "side": "-",
        "harmful_threshold": "-",
        "mean_r1": f"{bl_mean:.2f}",
        "n_zscore": 0,
        "n_baseline": len(pairs),
    })
    print(f"\n  Always baseline:  {bl_mean:.2f}%")

    # Always z-score
    zs_r1s = [p["zscore_r1"] for p in pairs]
    zs_mean = mean(zs_r1s) * 100
    rows.append({
        "strategy": "always_zscore",
        "side": "-",
        "harmful_threshold": "-",
        "mean_r1": f"{zs_mean:.2f}",
        "n_zscore": len(pairs),
        "n_baseline": 0,
    })
    print(f"  Always z-score:   {zs_mean:.2f}%")

    # Adaptive thresholds — both sides
    print(f"\n  {'side':<15} {'threshold':>9} {'mean_r1':>8} {'n_zs':>5} {'n_bl':>5}")
    print("  " + "-" * 48)

    for side in ["db_sim_mean", "query_sim_mean"]:
        for t in THRESHOLDS:
            r1s = [adaptive_r1(p, t, side) for p in pairs]
            m = mean(r1s) * 100
            n_zs = sum(
                1 for p in pairs
                if p[side] < t
            )
            n_bl = len(pairs) - n_zs
            rows.append({
                "strategy": "adaptive",
                "side": side,
                "harmful_threshold": f"{t:.2f}",
                "mean_r1": f"{m:.2f}",
                "n_zscore": n_zs,
                "n_baseline": n_bl,
            })
            print(f"  {side:<15} {t:>9.2f} {m:>8.2f}% {n_zs:>5} {n_bl:>5}")

    # Oracle per-pair
    oracle_r1s = [max(p["baseline_r1"], p["zscore_r1"]) for p in pairs]
    oracle_mean = mean(oracle_r1s) * 100
    rows.append({
        "strategy": "oracle_per_pair",
        "side": "-",
        "harmful_threshold": "-",
        "mean_r1": f"{oracle_mean:.2f}",
        "n_zscore": sum(1 for p in pairs if p["zscore_r1"] > p["baseline_r1"]),
        "n_baseline": sum(1 for p in pairs if p["baseline_r1"] >= p["zscore_r1"]),
    })
    print(f"\n  Oracle per-pair:  {oracle_mean:.2f}%")

    # Find best threshold per side
    print("\n  === Best thresholds ===")
    for side in ["db_sim_mean", "query_sim_mean"]:
        best_t = None
        best_r1 = -1
        for t in THRESHOLDS:
            r1s = [adaptive_r1(p, t, side) for p in pairs]
            m = mean(r1s) * 100
            if m > best_r1:
                best_r1 = m
                best_t = t
        print(f"  {side}: best threshold = {best_t:.2f} (mean R@1 = {best_r1:.2f}%)")

        # Sensitivity: R@1 at best ± 0.05
        for delta in [-0.05, 0, 0.05]:
            t_check = best_t + delta
            if t_check in THRESHOLDS:
                r1s = [adaptive_r1(p, t_check, side) for p in pairs]
                m = mean(r1s) * 100
                print(f"    t={t_check:.2f}: {m:.2f}%  (Δ from best: {m - best_r1:+.2f}pp)")

    # Write CSV
    out = RESULTS_DIR / "e2_threshold_sensitivity.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "side", "harmful_threshold", "mean_r1",
                         "n_zscore", "n_baseline"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    run_e2()
