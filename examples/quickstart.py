"""RAT Quick Start: cross-model embedding search.

Demonstrates:
  1. Two-model translation (RATranslator)
  2. Compatibility check
  3. Multi-model hub (RATHub)

Uses cached embeddings from the paper experiments.
To run: python examples/quickstart.py (from the repo root)

Requires: pip install rat-embed
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rat import RATranslator, RATHub
from rat.sampling import farthest_point_sampling

data_dir = Path(__file__).parent.parent / "data" / "d2_matrix"

# ─────────────────────────────────────────────
# Part 1: Two-model translation (RATranslator)
# ─────────────────────────────────────────────

# Model A: all-MiniLM-L6-v2 (384d, query side)
# Model C: bge-small-en-v1.5 (384d, database side)
cand_a = np.load(data_dir / "cand_A.npy")   # (2000, 384) candidate pool
cand_c = np.load(data_dir / "cand_C.npy")   # (2000, 384) same texts, different model
query_a = np.load(data_dir / "query_A.npy")  # (500, 384)
query_c = np.load(data_dir / "query_C.npy")  # (500, 384)

# Select anchors via Farthest Point Sampling
anchor_idx = farthest_point_sampling(cand_a, k=500, seed=42)
anchor_a = cand_a[anchor_idx]
anchor_c = cand_c[anchor_idx]

# Fit and retrieve
translator = RATranslator(kernel="poly").fit_embeddings(anchor_a, anchor_c)
results = translator.retrieve(query_a, query_c, top_k=10)

# Evaluate: query_a[i] and query_c[i] are the same text → correct match is i
correct_at_1 = (results["indices"][:, 0] == np.arange(len(query_a))).mean()
correct_at_10 = np.any(
    results["indices"] == np.arange(len(query_a))[:, None], axis=1
).mean()

print("=" * 50)
print("Part 1: MiniLM queries → BGE-small database")
print(f"  Recall@1:  {correct_at_1:.1%}")
print(f"  Recall@10: {correct_at_10:.1%}")

# ─────────────────────────────────────────────
# Part 2: Compatibility check
# ─────────────────────────────────────────────

compat = translator.estimate_compatibility()
print(f"\nPart 2: Compatibility check")
print(f"  Tier: {compat['compatibility']}")
print(f"  Estimated R@1: {compat['estimated_recall_at_1']}% "
      f"(band: {compat['confidence_band'][0]}-{compat['confidence_band'][1]}%)")
print(f"  {compat['compatibility_description']}")

# ─────────────────────────────────────────────
# Part 3: Multi-model hub (RATHub)
# ─────────────────────────────────────────────

# Add a third model: E5-large (1024d)
cand_b = np.load(data_dir / "cand_B.npy")   # (2000, 1024)
query_b = np.load(data_dir / "query_B.npy")  # (500, 1024)
anchor_b = cand_b[anchor_idx]  # same anchor indices for all models

hub = RATHub(kernel="poly")
hub.set_anchors("minilm", anchor_a)
hub.set_anchors("bge-small", anchor_c)
hub.set_anchors("e5-large", anchor_b)

# Cross-model retrieval: MiniLM → E5-large
r = hub.retrieve(query_a, query_b, "minilm", "e5-large", top_k=10)
r1 = (r["indices"][:, 0] == np.arange(len(query_a))).mean()
print(f"\nPart 3: RATHub multi-model")
print(f"  MiniLM → E5-large R@1: {r1:.1%}")

# Multi-DB search: query with MiniLM, search across BGE and E5 databases
multi = hub.retrieve_multi(
    query_a,
    databases=[
        (query_c, "bge-small"),    # 500 docs from BGE-small
        (query_b, "e5-large"),     # 500 docs from E5-large
    ],
    query_model="minilm",
    top_k=10,
)
# In this setup, the correct document for query[i] exists in both DBs
# (at index i in DB0 and index i in DB1), so any hit counts
hit_at_1 = 0
for i in range(len(query_a)):
    idx = multi["indices"][i, 0]
    db = multi["db_labels"][i, 0]
    local_idx = idx - (500 * db)  # convert global → local
    if local_idx == i:
        hit_at_1 += 1
print(f"  Multi-DB search R@1: {hit_at_1 / len(query_a):.1%}")
print(f"  DB distribution in top-10: "
      f"BGE={np.sum(multi['db_labels'] == 0)}, "
      f"E5={np.sum(multi['db_labels'] == 1)}")
print("=" * 50)
