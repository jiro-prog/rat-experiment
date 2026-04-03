"""RAT Quick Start: cross-model embedding search in ~10 lines.

This example uses cached embeddings from the paper experiments.
To run: python examples/quickstart.py (from the repo root)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rat import RATranslator
from rat.sampling import farthest_point_sampling

# ── 1. Load embeddings ──
# Model A: all-MiniLM-L6-v2 (384d, query side)
# Model C: bge-small-en-v1.5 (384d, database side)
data_dir = Path(__file__).parent.parent / "data" / "d2_matrix"
cand_a = np.load(data_dir / "cand_A.npy")   # (2000, 384) candidate pool
cand_c = np.load(data_dir / "cand_C.npy")   # (2000, 384) same texts, different model
query_a = np.load(data_dir / "query_A.npy") # (500, 384) query embeddings
query_c = np.load(data_dir / "query_C.npy") # (500, 384) ground truth (same texts)

# ── 2. Select anchors via FPS ──
anchor_idx = farthest_point_sampling(cand_a, k=500, seed=42)
anchor_a = cand_a[anchor_idx]
anchor_c = cand_c[anchor_idx]

# ── 3. Fit translator ──
translator = RATranslator(kernel="poly").fit_embeddings(anchor_a, anchor_c)

# ── 4. Retrieve ──
results = translator.retrieve(query_a, query_c, top_k=10)

# ── 5. Evaluate ──
# query_a[i] and query_c[i] are the same text → correct match is index i
correct = results["indices"][:, 0] == np.arange(len(query_a))
recall_at_1 = correct.mean()
recall_at_10 = np.any(results["indices"] == np.arange(len(query_a))[:, None], axis=1).mean()

print(f"Cross-model search: MiniLM queries → BGE-small database")
print(f"  Recall@1:  {recall_at_1:.1%}")
print(f"  Recall@10: {recall_at_10:.1%}")
print(f"  (500 queries, 500 candidates, 500 anchors)")

# ── 6. Compatibility check ──
compat = translator.estimate_compatibility()
print(f"\nCompatibility estimate:")
print(f"  sim_mean_a (MiniLM):    {compat['sim_mean_a']:.3f}")
print(f"  sim_mean_b (BGE-small): {compat['sim_mean_b']:.3f}")
print(f"  z-score recommendation: {compat['z_score_recommendation']}")
