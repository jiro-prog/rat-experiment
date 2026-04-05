#!/usr/bin/env python3
"""E1: RATauto non-oracle — direction selection heuristics for Table 7.

Input:  results/d1_alignment_v2/d1_results.json (272 directed pairs × 6K × 3 seeds)
Output: results/e1_ratauto_nonoracle.csv   — Table 7 追加行
        results/e1_direction_accuracy.csv   — heuristic vs oracle 一致率
"""

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
D1_PATH = RESULTS_DIR / "d1_alignment_v2" / "d1_results.json"

K_VALUES = [10, 25, 50, 100, 200, 500]
MODELS_11 = set("ABCDEFGHIJKL")  # 12-model subset (A-L)


def load_rat_records():
    with open(D1_PATH) as f:
        data = json.load(f)
    return [r for r in data["results"] if r["method"] == "RAT"]


def build_lookup(records):
    """Build lookup: (query, db, K, seed) -> record."""
    lut = {}
    for r in records:
        key = (r["query_model"], r["db_model"], r["K"], r["seed"])
        lut[key] = r
    return lut


def build_sim_mean_lookup(records):
    """Build lookup: (model, K, seed) -> sim_mean (query-side FPS space)."""
    lut = {}
    for r in records:
        key = (r["query_model"], r["K"], r["seed"])
        if key not in lut:
            lut[key] = r["sim_mean"]
    return lut


def get_undirected_pairs(models):
    """Generate undirected pairs from model set."""
    models = sorted(models)
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs.append((a, b))
    return pairs


def run_e1():
    print("Loading d1_results.json...")
    records = load_rat_records()
    print(f"  RAT records: {len(records)}")

    lut = build_lookup(records)
    sim_lut = build_sim_mean_lookup(records)

    # Get all models and seeds
    all_models = sorted(set(r["query_model"] for r in records))
    seeds = sorted(set(r["seed"] for r in records))
    print(f"  Models: {all_models}")
    print(f"  Seeds: {seeds}")

    # Check for score heuristic data (mean similarity score of retrieval results)
    has_score = "mean_score" in records[0] or "retrieval_mean_score" in records[0]
    score_key = None
    if has_score:
        score_key = "mean_score" if "mean_score" in records[0] else "retrieval_mean_score"
        print(f"  Score heuristic data available: {score_key}")
    else:
        print("  Score heuristic data NOT available — skipping score heuristic")

    # Two scopes
    scopes = {
        "11_subset": get_undirected_pairs(MODELS_11),
        "all_17": get_undirected_pairs(all_models),
    }

    table_rows = []
    direction_rows = []

    for scope_name, pairs in scopes.items():
        print(f"\n=== {scope_name}: {len(pairs)} undirected pairs ===")

        # Per K, per seed: compute heuristic results
        for K in K_VALUES:
            oracle_by_seed = []
            sim_mean_by_seed = []
            score_by_seed = []
            agree_sim_by_seed = []
            agree_score_by_seed = []

            for seed in seeds:
                oracle_vals = []
                sim_vals = []
                score_vals_list = []
                agree_sim = 0
                agree_score = 0

                for a, b in pairs:
                    rec_ab = lut.get((a, b, K, seed))
                    rec_ba = lut.get((b, a, K, seed))
                    if rec_ab is None or rec_ba is None:
                        continue

                    r1_ab = rec_ab["recall_at_1"]
                    r1_ba = rec_ba["recall_at_1"]

                    # Oracle: pick better direction
                    oracle_r1 = max(r1_ab, r1_ba)
                    oracle_dir = "ab" if r1_ab >= r1_ba else "ba"
                    oracle_vals.append(oracle_r1)

                    # sim_mean heuristic: FPS in lower sim_mean space
                    # sim_mean is query-side. Lower query sim_mean → pick that direction.
                    sm_a = sim_lut.get((a, K, seed))
                    sm_b = sim_lut.get((b, K, seed))
                    if sm_a is not None and sm_b is not None:
                        if sm_a < sm_b:
                            sim_dir = "ab"  # A as query (FPS in A's space)
                        elif sm_b < sm_a:
                            sim_dir = "ba"  # B as query
                        else:
                            sim_dir = oracle_dir  # tiebreak
                        sim_r1 = r1_ab if sim_dir == "ab" else r1_ba
                        sim_vals.append(sim_r1)
                        if sim_dir == oracle_dir:
                            agree_sim += 1

                    # Score heuristic (if available)
                    if score_key:
                        sc_ab = rec_ab.get(score_key)
                        sc_ba = rec_ba.get(score_key)
                        if sc_ab is not None and sc_ba is not None:
                            score_dir = "ab" if sc_ab >= sc_ba else "ba"
                            score_r1 = r1_ab if score_dir == "ab" else r1_ba
                            score_vals_list.append(score_r1)
                            if score_dir == oracle_dir:
                                agree_score += 1

                n_pairs = len(oracle_vals)
                oracle_by_seed.append(mean(oracle_vals) * 100 if oracle_vals else 0)
                sim_mean_by_seed.append(mean(sim_vals) * 100 if sim_vals else 0)
                if score_vals_list:
                    score_by_seed.append(mean(score_vals_list) * 100)
                agree_sim_by_seed.append(agree_sim / n_pairs * 100 if n_pairs else 0)
                if score_key:
                    agree_score_by_seed.append(
                        agree_score / n_pairs * 100 if n_pairs else 0
                    )

            # Aggregate across seeds
            def fmt(vals):
                if not vals:
                    return ""
                m = mean(vals)
                s = stdev(vals) if len(vals) > 1 else 0
                return f"{m:.1f}±{s:.1f}"

            # Store for CSV
            if K == K_VALUES[0]:
                # Initialize row dicts
                oracle_row = {
                    "scope": scope_name,
                    "method": "RATauto_oracle",
                    "N": len(pairs),
                }
                sim_row = {
                    "scope": scope_name,
                    "method": "RATauto_sim_mean",
                    "N": len(pairs),
                }
                score_row = {
                    "scope": scope_name,
                    "method": "RATauto_score",
                    "N": len(pairs),
                }
                dir_sim_row = {"scope": scope_name, "heuristic": "sim_mean"}
                dir_score_row = {"scope": scope_name, "heuristic": "score"}

            oracle_row[f"K{K}"] = fmt(oracle_by_seed)
            sim_row[f"K{K}"] = fmt(sim_mean_by_seed)
            dir_sim_row[f"K{K}"] = fmt(agree_sim_by_seed)

            if score_by_seed:
                score_row[f"K{K}"] = fmt(score_by_seed)
                dir_score_row[f"K{K}"] = fmt(agree_score_by_seed)

            # Print progress
            print(
                f"  K={K:>3}: oracle={fmt(oracle_by_seed):>10}  "
                f"sim_mean={fmt(sim_mean_by_seed):>10}  "
                f"agree={fmt(agree_sim_by_seed):>10}"
            )

        table_rows.append(oracle_row)
        table_rows.append(sim_row)
        if score_key:
            table_rows.append(score_row)
        direction_rows.append(dir_sim_row)
        if score_key:
            direction_rows.append(dir_score_row)

    # Write CSVs
    k_cols = [f"K{k}" for k in K_VALUES]

    out1 = RESULTS_DIR / "e1_ratauto_nonoracle.csv"
    with open(out1, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "method", "N"] + k_cols)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)
    print(f"\nWrote {out1}")

    out2 = RESULTS_DIR / "e1_direction_accuracy.csv"
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "heuristic"] + k_cols)
        writer.writeheader()
        for row in direction_rows:
            writer.writerow(row)
    print(f"Wrote {out2}")

    # Sanity check: 11-subset oracle K=500 should ≈ 61.8%
    for row in table_rows:
        if row["scope"] == "11_subset" and row["method"] == "RATauto_oracle":
            print(f"\nSanity check: 11-subset RATauto oracle K=500 = {row['K500']}")
            print("  Expected: ~61.8% (Table 7)")


if __name__ == "__main__":
    run_e1()
