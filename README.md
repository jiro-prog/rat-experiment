# RAT — Relative Anchor Translation

[![PyPI](https://img.shields.io/pypi/v/rat-embed)](https://pypi.org/project/rat-embed/)
[![Python](https://img.shields.io/pypi/pyversions/rat-embed)](https://pypi.org/project/rat-embed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19401277.svg)](https://zenodo.org/records/19401277)

Zero-shot embedding space translation using relative distances to shared anchors. No additional training required.

## Install

```bash
pip install rat-embed            # core (numpy only)
pip install "rat-embed[models]"  # + sentence-transformers for fit()
pip install "rat-embed[dev]"     # + pytest, ruff
```

## Quick Start

```python
import numpy as np
from rat import RATranslator

# 1. Prepare anchor embeddings from both models (K anchors, L2-normalized)
anchor_a = ...  # (K, D_a) from model A
anchor_b = ...  # (K, D_b) from model B

# 2. Fit the translator
translator = RATranslator(kernel="poly").fit_embeddings(anchor_a, anchor_b)

# 3. Transform & retrieve
query_emb = ...  # (N, D_a) from model A
db_emb = ...     # (M, D_b) from model B
results = translator.retrieve(query_emb, db_emb, top_k=10)
# results["indices"]  → (N, 10) nearest neighbor indices
# results["scores"]   → (N, 10) cosine similarity scores
```

Or transform individually for more control:

```python
q_rel = translator.transform(query_emb, "a")              # query side (no z-score)
d_rel = translator.transform(db_emb, "b")                 # db side (z-score applied)
d_rel = translator.transform(db_emb, "b", role="query")   # override: skip z-score
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example.

## Compatibility Check

Before running a full pipeline, check expected accuracy:

```python
compat = translator.estimate_compatibility(same_family=True)
print(compat["compatibility"])           # "high" / "moderate" / "low"
print(compat["estimated_recall_at_1"])   # point estimate (%)
print(compat["confidence_band"])         # (lower, upper) at 68% coverage
```

The compatibility tier is based on similarity compression (`max_sim_mean`) and is more
reliable than the point estimate (R²=0.17). Cross-modal and CLIP-text models may
deviate significantly from the estimate.

| Tier | max_sim_mean | Typical R@1 |
|------|-------------|-------------|
| high | < 0.45 | 91-98% |
| moderate | 0.45-0.72 | 73-91% |
| low | >= 0.72 | 61-82% |

## Advanced: RATHub (multi-model)

```python
from rat import RATHub

hub = RATHub(kernel="poly")
hub.set_anchors("minilm", anchor_minilm)      # (K, 384)
hub.set_anchors("e5", anchor_e5)              # (K, 1024)
hub.set_anchors("bge", anchor_bge)            # (K, 384)

# Transform from any model
q = hub.transform("minilm", query_emb, role="query")
d = hub.transform("e5", db_emb, role="db")

# Or use retrieve directly
results = hub.retrieve(query_emb, db_emb, "minilm", "e5", top_k=10)
```

> **Important:** All models must use anchors from the **same texts** in the
> **same order**. Select anchor indices once (e.g., via FPS on one model),
> then use those indices for every model.

### Multi-DB search

Search across multiple databases built with different models:

```python
results = hub.retrieve_multi(
    query_emb,
    databases=[
        (db1_emb, "bge"),       # BGE database
        (db2_emb, "minilm"),    # MiniLM database
        (db3_emb, "e5"),        # E5 database
    ],
    query_model="bge",
    top_k=10,
)
# results["indices"]   → (N, 10) global indices
# results["scores"]    → (N, 10) normalized scores
# results["db_labels"] → (N, 10) which DB each result came from (0, 1, 2)
```

`retrieve_multi` uses per-database score normalization internally to make
scores comparable across databases. Each database should have at least ~50
documents for stable normalization. Do not vstack relative representations
from different models — their score scales differ.

## Paper

See the [Zenodo record](https://zenodo.org/records/19401277) for the full experiment report.

Experiment reproduction code is in `experiments/` (unchanged from the original research).

## License

MIT
