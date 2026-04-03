"""C2b: Multi-model RAG — cross-DB unified search via RATHub.

Tests whether RATHub can unify search across 3 DBs built with different
embedding models, and whether scores are comparable across DBs.

Design: split the 500 query texts into 3 "DBs", each encoded by a
different model. Query all 500 texts with BGE-large. GT is text identity
(query[i] should match the DB entry for the same text).

DB1: texts[0:167]   — BGE-large (E)  [same model as query]
DB2: texts[167:334]  — MiniLM (A)    [cross-family]
DB3: texts[334:500]  — GTE-small (H) [cross-family]

Query model: BGE-large (E), all 500 texts

Key finding: naive vstack fails due to score scale mismatch between models.
Per-query score normalization (z-score on similarity scores per DB) resolves this.
"""

import json
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rat import RATHub
from rat.sampling import farthest_point_sampling

DATA_DIR = "data/d2_matrix"
RESULTS_DIR = "results"
K = 500
RRF_K = 60
TOP_K_PER_DB = 50

DB_SPLITS = [(0, 167), (167, 334), (334, 500)]
DB_MODELS = ["E", "A", "H"]
DB_NAMES = ["DB1(BGE-large)", "DB2(MiniLM)", "DB3(GTE-small)"]
QUERY_MODEL = "E"


def compute_r_at_k(ranked, gt_indices, k):
    return sum(1 for i, gt in enumerate(gt_indices) if gt in ranked[i, :k])


def compute_mrr(ranked, gt_indices):
    mrr = 0
    for i, gt in enumerate(gt_indices):
        pos = np.where(ranked[i] == gt)[0]
        if len(pos) > 0:
            mrr += 1.0 / (pos[0] + 1)
    return mrr / len(gt_indices)


def main():
    start_time = time.time()
    print("=" * 70)
    print("C2b: Multi-Model RAG — Cross-DB Unified Search")
    print("=" * 70)

    with open(f"{DATA_DIR}/metadata.json") as f:
        meta = json.load(f)

    # ── Setup: shared anchors (same text indices for all models) ──
    cand_E = np.load(f"{DATA_DIR}/cand_{QUERY_MODEL}.npy")
    anchor_idx = farthest_point_sampling(cand_E, K)

    hub = RATHub(kernel="poly")
    for model in set(DB_MODELS + [QUERY_MODEL]):
        cand = np.load(f"{DATA_DIR}/cand_{model}.npy")
        hub.set_anchors(model, cand[anchor_idx])

    query_emb = np.load(f"{DATA_DIR}/query_{QUERY_MODEL}.npy")
    rel_query = hub.transform(QUERY_MODEL, query_emb, role="query")

    n_queries = len(query_emb)
    gt_all = np.arange(n_queries)  # identity GT

    print(f"\nQuery model: {meta['models'][QUERY_MODEL]['name']} ({query_emb.shape})")
    for (start, end), model, name in zip(DB_SPLITS, DB_MODELS, DB_NAMES):
        print(f"  {name}: texts[{start}:{end}] ({end-start} docs), "
              f"model={meta['models'][model]['name']}")

    # ── Test A: Per-DB Pairwise Retrieval ──
    print("\n=== Test A: Per-DB Pairwise (isolated) ===")
    test_a = []
    per_db_sims = {}

    for (start, end), model, name in zip(DB_SPLITS, DB_MODELS, DB_NAMES):
        emb = np.load(f"{DATA_DIR}/query_{model}.npy")[start:end]
        rel_db = hub.transform(model, emb, role="db")

        # Full query set vs this DB
        sim_full = cosine_similarity(rel_query, rel_db)  # (500, n_db)
        per_db_sims[name] = sim_full

        # Per-DB R@1: only queries whose GT is in this DB
        n = end - start
        sim_sub = sim_full[start:end]  # (n, n)
        top1 = np.argmax(sim_sub, axis=1)
        gt_local = np.arange(n)
        r1 = (top1 == gt_local).mean() * 100

        ranked_sub = np.argsort(-sim_sub, axis=1)
        r5 = sum(1 for i in range(n) if i in ranked_sub[i, :5]) / n * 100
        r10 = sum(1 for i in range(n) if i in ranked_sub[i, :10]) / n * 100
        mrr = compute_mrr(ranked_sub, gt_local) * 100

        result = {"db": name, "model": model, "n": n,
                  "r1": round(r1, 1), "r5": round(r5, 1),
                  "r10": round(r10, 1), "mrr": round(mrr, 1)}
        test_a.append(result)
        print(f"  {name}: R@1={r1:.1f}% R@5={r5:.1f}% R@10={r10:.1f}% MRR={mrr:.1f}%")

    # ── Test B1: Naive vstack ──
    print("\n=== Test B1: Naive vstack ===")
    rel_dbs_raw = []
    for (start, end), model, name in zip(DB_SPLITS, DB_MODELS, DB_NAMES):
        emb = np.load(f"{DATA_DIR}/query_{model}.npy")[start:end]
        rel = hub.transform(model, emb, role="db")
        norms = np.linalg.norm(rel, axis=1)
        print(f"  {name}: norm mean={norms.mean():.1f} std={norms.std():.3f}")
        rel_dbs_raw.append(rel)

    rel_all_raw = np.vstack(rel_dbs_raw)
    scores_vstack = cosine_similarity(rel_query, rel_all_raw)
    ranked_vs = np.argsort(-scores_vstack, axis=1)

    r1_vs = compute_r_at_k(ranked_vs, gt_all, 1) / n_queries * 100
    r5_vs = compute_r_at_k(ranked_vs, gt_all, 5) / n_queries * 100
    r10_vs = compute_r_at_k(ranked_vs, gt_all, 10) / n_queries * 100
    mrr_vs = compute_mrr(ranked_vs, gt_all) * 100
    print(f"  R@1={r1_vs:.1f}% R@5={r5_vs:.1f}% R@10={r10_vs:.1f}% MRR={mrr_vs:.1f}%")
    print("  → Score scale mismatch causes cross-DB comparison failure")

    # Score distributions
    vstack_score_stats = []
    for (start, end), name in zip(DB_SPLITS, DB_NAMES):
        top = np.max(scores_vstack[:, start:end], axis=1)
        vstack_score_stats.append({
            "db": name,
            "mean": round(float(top.mean()), 4),
            "std": round(float(top.std()), 4),
        })
        print(f"    {name} top score: mean={top.mean():.4f} std={top.std():.4f}")

    # ── Test B2: Score Normalization ──
    print("\n=== Test B2: Per-DB Score Normalization ===")
    all_scores_normed = np.zeros((n_queries, n_queries))

    for (start, end), name in zip(DB_SPLITS, DB_NAMES):
        sim = per_db_sims[name]  # (500, n_db)
        # Per-query z-score
        mu = sim.mean(axis=1, keepdims=True)
        sigma = sim.std(axis=1, keepdims=True)
        sigma[sigma == 0] = 1.0
        all_scores_normed[:, start:end] = (sim - mu) / sigma

    ranked_sn = np.argsort(-all_scores_normed, axis=1)
    r1_sn = compute_r_at_k(ranked_sn, gt_all, 1) / n_queries * 100
    r5_sn = compute_r_at_k(ranked_sn, gt_all, 5) / n_queries * 100
    r10_sn = compute_r_at_k(ranked_sn, gt_all, 10) / n_queries * 100
    mrr_sn = compute_mrr(ranked_sn, gt_all) * 100
    print(f"  R@1={r1_sn:.1f}% R@5={r5_sn:.1f}% R@10={r10_sn:.1f}% MRR={mrr_sn:.1f}%")

    # Per-DB breakdown
    sn_per_db = []
    for (start, end), name in zip(DB_SPLITS, DB_NAMES):
        n = end - start
        hits = sum(1 for i in range(start, end) if ranked_sn[i, 0] == i)
        result = {"db": name, "r1": round(hits / n * 100, 1)}
        sn_per_db.append(result)
        print(f"    {name}: R@1={result['r1']:.1f}%")

    # ── Test B3: Reciprocal Rank Fusion ──
    print("\n=== Test B3: Reciprocal Rank Fusion (RRF) ===")
    rrf_scores = np.zeros((n_queries, n_queries))

    for (start, end), name in zip(DB_SPLITS, DB_NAMES):
        sim = per_db_sims[name]
        ranked_db = np.argsort(-sim, axis=1)
        for q in range(n_queries):
            for rank_pos in range(min(TOP_K_PER_DB, end - start)):
                local_idx = ranked_db[q, rank_pos]
                global_idx = start + local_idx
                rrf_scores[q, global_idx] += 1.0 / (RRF_K + rank_pos + 1)

    ranked_rrf = np.argsort(-rrf_scores, axis=1)
    r1_rrf = compute_r_at_k(ranked_rrf, gt_all, 1) / n_queries * 100
    r5_rrf = compute_r_at_k(ranked_rrf, gt_all, 5) / n_queries * 100
    r10_rrf = compute_r_at_k(ranked_rrf, gt_all, 10) / n_queries * 100
    mrr_rrf = compute_mrr(ranked_rrf, gt_all) * 100
    print(f"  R@1={r1_rrf:.1f}% R@5={r5_rrf:.1f}% R@10={r10_rrf:.1f}% MRR={mrr_rrf:.1f}%")

    rrf_per_db = []
    for (start, end), name in zip(DB_SPLITS, DB_NAMES):
        n = end - start
        hits = sum(1 for i in range(start, end) if ranked_rrf[i, 0] == i)
        result = {"db": name, "r1": round(hits / n * 100, 1)}
        rrf_per_db.append(result)
        print(f"    {name}: R@1={result['r1']:.1f}%")

    # ── Test C: Baseline ──
    print("\n=== Test C: Baseline (re-index all with BGE-large) ===")
    scores_bl = cosine_similarity(query_emb, query_emb)
    ranked_bl = np.argsort(-scores_bl, axis=1)
    r1_bl = compute_r_at_k(ranked_bl, gt_all, 1) / n_queries * 100
    r5_bl = compute_r_at_k(ranked_bl, gt_all, 5) / n_queries * 100
    r10_bl = compute_r_at_k(ranked_bl, gt_all, 10) / n_queries * 100
    mrr_bl = compute_mrr(ranked_bl, gt_all) * 100
    print(f"  R@1={r1_bl:.1f}% R@5={r5_bl:.1f}% R@10={r10_bl:.1f}% MRR={mrr_bl:.1f}%")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'Method':<25s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'Ret%':>6s}")
    print("-" * 60)
    print(f"{'Baseline (re-index)':<25s} {r1_bl:>5.1f}% {r5_bl:>5.1f}% {r10_bl:>5.1f}% {mrr_bl:>5.1f}%   {'—':>4s}")
    print(f"{'Score normalization':<25s} {r1_sn:>5.1f}% {r5_sn:>5.1f}% {r10_sn:>5.1f}% {mrr_sn:>5.1f}% {r1_sn/r1_bl*100:>5.1f}%")
    print(f"{'RRF':<25s} {r1_rrf:>5.1f}% {r5_rrf:>5.1f}% {r10_rrf:>5.1f}% {mrr_rrf:>5.1f}% {r1_rrf/r1_bl*100:>5.1f}%")
    print(f"{'Naive vstack':<25s} {r1_vs:>5.1f}% {r5_vs:>5.1f}% {r10_vs:>5.1f}% {mrr_vs:>5.1f}% {r1_vs/r1_bl*100:>5.1f}%")

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Save ──
    output = {
        "experiment": "C2b: Multi-Model RAG",
        "date": "2026-04-04",
        "config": {
            "K": K, "kernel": "poly(degree=2, coef0=1.0)",
            "query_model": QUERY_MODEL,
            "db_models": DB_MODELS,
            "db_splits": [list(s) for s in DB_SPLITS],
            "n_queries": n_queries,
            "anchor_selection": "shared (FPS on query model candidates)",
            "rrf_k": RRF_K, "top_k_per_db": TOP_K_PER_DB,
            "zscore": "auto (v0.1.1)",
            "gt": "identity (query[i] matches db[i], same text)",
        },
        "test_a_per_db_pairwise": test_a,
        "test_b1_vstack": {
            "r1": round(r1_vs, 1), "r5": round(r5_vs, 1),
            "r10": round(r10_vs, 1), "mrr": round(mrr_vs, 1),
            "score_stats": vstack_score_stats,
            "finding": "Score scale mismatch between models causes complete failure",
        },
        "test_b2_score_normalization": {
            "r1": round(r1_sn, 1), "r5": round(r5_sn, 1),
            "r10": round(r10_sn, 1), "mrr": round(mrr_sn, 1),
            "per_db": sn_per_db,
            "retention": round(r1_sn / r1_bl * 100, 1),
        },
        "test_b3_rrf": {
            "r1": round(r1_rrf, 1), "r5": round(r5_rrf, 1),
            "r10": round(r10_rrf, 1), "mrr": round(mrr_rrf, 1),
            "per_db": rrf_per_db,
            "retention": round(r1_rrf / r1_bl * 100, 1),
        },
        "test_c_baseline": {
            "r1": round(r1_bl, 1), "r5": round(r5_bl, 1),
            "r10": round(r10_bl, 1), "mrr": round(mrr_bl, 1),
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = f"{RESULTS_DIR}/c2b_multimodel_rag.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
