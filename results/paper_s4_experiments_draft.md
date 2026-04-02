# §4 Experiments — Draft

## 4.1 Experimental Setup

**Models.** We evaluate five text embedding models spanning different architectures, training objectives, and dimensionalities (Table 1): sentence-transformers/all-MiniLM-L6-v2 (384d, English, distilled BERT; Model A), intfloat/multilingual-e5-large (1024d, multilingual, XLM-R; Model B), BAAI/bge-small-en-v1.5 (384d, English, BERT; Model C), sentence-transformers/clip-ViT-B-32 (512d, CLIP text encoder; Model D), and BAAI/bge-large-en-v1.5 (1024d, English, BERT; Model E). For cross-modal experiments, we additionally use the CLIP ViT-B/32 image encoder (512d). Models B and E share the same output dimensionality (1024d) but differ in architecture (XLM-RoBERTa vs BERT) and training data, enabling controlled analysis of dimension vs. structure effects.

**Datasets.** We use three datasets:
- **STSBenchmark** (Cer et al., 2017): 15,487 unique English sentences from semantic textual similarity annotations. We sample 2,000 candidate anchors and 500 non-overlapping queries (seed=42).
- **AllNLI** (Bowman et al., 2015; Williams et al., 2018): 26,115 unique sentences from the combined SNLI and MultiNLI test sets. Same sampling protocol. This dataset provides greater semantic diversity than STSBenchmark, with sentence pairs spanning entailment, contradiction, and neutral relations.
- **COCO Captions** (Lin et al., 2014; Karpathy & Fei-Fei, 2015): 5,000 image-caption pairs from the Karpathy test split. We sample 500 pairs as Rosetta Stone anchors and 500 as queries.

**Protocol.** Unless otherwise noted: FPS anchor selection (K=500, based on Model A), polynomial kernel κ(u,v) = (u⊤v + 1)², z-score normalization (DB-side only), cosine similarity retrieval. Random baseline for 500 queries: Recall@1 = 0.2%.

**Evaluation.** Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR). For each of 500 queries, we compute the rank of the correct match among all 500 database items.

## 4.2 Text-to-Text Retrieval

**Table 1: Cross-model retrieval results (Recall@1 %) for all model pairs.**

| Pair | Dimensions | STSBenchmark | AllNLI |
|------|-----------|-------------|--------|
| A×C (MiniLM × BGE-small) | 384 × 384 | 98.0 | 99.2 |
| C×E (BGE-small × BGE-large) | 384 × 1024 | 77.0 | 98.0 |
| A×E (MiniLM × BGE-large) | 384 × 1024 | 80.2 | 91.8 |
| A×B (MiniLM × E5-large) | 384 × 1024 | 76.2 | 71.8 |
| B×E (E5-large × BGE-large) | 1024 × 1024 | 55.2 | 76.4 |
| B×C (E5-large × BGE-small) | 1024 × 384 | 64.0 | 84.2 |

All pairs exceed Recall@1 > 55% on STSBenchmark and > 71% on AllNLI, compared to a random baseline of 0.2%.

**Dataset effect.** Pairs involving Model B (E5-large) show marked improvement on AllNLI: B×E +21.2 points, B×C +20.2 points, A×E +11.6 points. We attribute this to AllNLI's greater semantic diversity, which provides more discriminative anchor response patterns even in B's compressed similarity space (§5.1). The one exception is A×B, which decreases slightly (76.2% → 71.8%); this 4.4-point difference is within the 95% confidence interval for 500 binary trials (±4.4% at p=0.05) and we do not consider it significant.

**Ablation.** Each component of the RAT protocol contributes independently (measured on A×B, STSBenchmark):

| Component | Recall@1 | Δ |
|-----------|----------|---|
| Random anchors + cosine kernel | 43.2% | baseline |
| + FPS anchor selection | 66.4% | +23.2 |
| + Polynomial kernel | 77.2% | +10.8 |
| + z-score (DB-side) | 76.2% | −1.0 |
| **Full protocol on B×C** | **64.0%** | **(vs 14.4% without z-score: +49.6)** |

z-score has negligible effect on well-spread pairs (A×B: −1.0 point) but is transformative for collapsed pairs (B×C: +49.6 points). This asymmetric impact is analyzed in §5.1.

## 4.3 Anchor Efficiency: Scaling Curves

Figure 2 compares the naïve baseline (random anchors + cosine kernel) against the full RAT protocol across K ∈ {100, 200, 500, 1000} on the A×B pair (STSBenchmark).

| K | Random + cosine | RAT protocol | Δ |
|---|----------------|-------------|---|
| 100 | 26.0% | **55.0%** | +29.0 |
| 200 | 33.4% | **69.8%** | +36.4 |
| 500 | 45.2% | **79.6%** | +34.4 |
| 1000 | 48.2% | **79.6%** | +31.4 |

Two findings emerge. First, **RAT at K=100 (55.0%) surpasses random at K=1000 (48.2%)** — proper anchor selection and kernel choice achieve higher accuracy with 10× fewer anchors. Second, RAT saturates at K=500 (79.6% = 79.6% at K=1000), identifying K=500 as the cost-optimal operating point. The random baseline continues to improve slowly with K but never reaches RAT's K=200 performance even at K=1000.

## 4.4 Cross-Modal Retrieval

Using 500 Rosetta Stone anchor pairs from COCO Captions, we evaluate retrieval between MiniLM text representations and CLIP ViT-B/32 image representations.

**Table 2: Cross-modal retrieval results.**

| Direction | Method | Recall@1 | Recall@10 | MRR |
|-----------|--------|----------|-----------|-----|
| Text → Image | baseline (no z-score) | **16.4%** | 62.6% | 0.305 |
| Text → Image | z-score (both) | 15.6% | 60.4% | 0.297 |
| Image → Text | baseline (no z-score) | 3.8% | 23.8% | 0.110 |
| Image → Text | z-score (both) | **18.2%** | 64.6% | 0.325 |
| *CLIP direct (upper bound)* | *cosine in shared space* | *62.0%* | *95.0%* | *0.734* |

The text encoder (MiniLM) has never been exposed to visual data; the image encoder (CLIP ViT) has never processed text. Yet the 500 Rosetta Stone anchors enable Recall@1 = 18.2% — 91× the random baseline — with zero training.

The CLIP direct baseline (62.0%) uses CLIP's own text encoder paired with its image encoder in their jointly trained space, representing the ceiling achievable by a system trained on 400M image-text pairs. RAT recovers approximately 29% of this performance (18.2/62.0) using only 500 anchor pairs.

The z-score asymmetry (Text→Image: slight decrease, Image→Text: dramatic increase) is analyzed in §5.2.
