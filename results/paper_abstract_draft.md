# Abstract — Draft

Different embedding models map inputs to incompatible vector spaces, forcing costly re-embedding when models are switched or combined. We present **Relative Anchor Translation (RAT)**, a zero-shot protocol that translates between embedding spaces by comparing similarity profiles to shared anchor points, requiring no training data or learned parameters. RAT combines Farthest Point Sampling for anchor selection, polynomial kernel similarity, and DB-side z-score normalization into a three-step pipeline that runs on CPU in seconds.

We evaluate RAT across five text embedding models (384d–1024d, spanning BERT, XLM-R, and CLIP architectures) on two datasets (STSBenchmark, AllNLI). All model pairs achieve Recall@1 between 55% and 98% on STSBenchmark, compared to a 0.2% random baseline. RAT with 100 anchors (55%) surpasses random anchor selection with 1,000 (48%), demonstrating 10× efficiency from proper anchor placement.

Analysis reveals three findings: (1) **Similarity collapse** — a failure mode where compressed similarity distributions flatten relative representations — is diagnosable via entropy and fully resolved by z-score normalization (+65 points). (2) z-score normalization is effective **only on the database side**; query-side normalization is consistently harmful. (3) Two models with identical dimensions (1024d) but zero direct cosine correlation achieve 55% retrieval via RAT, demonstrating shared relative structure invisible to absolute comparison.

We extend RAT to cross-modal retrieval using paired image-caption anchors: a text-only encoder (MiniLM) searches a vision encoder's space (CLIP ViT-B/32) at Recall@1 = 18.2% — 91× the random baseline — with zero visual training. Code and experiments are publicly available.

---

## タイトル候補

**Primary:** Zero-Shot Embedding Space Translation via Relative Anchor Similarity Profiles

**Alternatives:**
- RAT: Relative Anchor Translation for Zero-Shot Embedding Interoperability
- Bridging Embedding Spaces Without Training: Relative Anchor Translation
