# RAT — Relative Anchor Translation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19401277.svg)](https://zenodo.org/records/19401277)

Zero-shot embedding space translation using relative distances to shared anchors. No additional training required.

## Install

```bash
pip install -e .            # core (numpy only)
pip install -e ".[models]"  # + sentence-transformers for fit()
pip install -e ".[dev]"     # + pytest, ruff
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

## Paper

See the [Zenodo record](https://zenodo.org/records/19401277) for the full experiment report.

Experiment reproduction code is in `experiments/` (unchanged from the original research).

## License

MIT
