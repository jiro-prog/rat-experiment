# §5 Analysis — Draft

This section examines three phenomena that emerge from the experiments: why certain model pairs fail, why z-score normalization exhibits directional asymmetry, and what RAT reveals about the structure shared across independently trained embedding spaces.

## 5.1 Similarity Collapse

Without z-score normalization, the B×C pair (E5-large × BGE-small) achieves only Recall@1 = 8.4% — far below the other pairs (A×B: 80.6%, A×C: 98.0%). We trace this failure to the distribution of anchor inter-similarities in Model B's space.

**Table 3: Anchor inter-similarity statistics and the effect of normalization methods on B×C.**

| Model | Mean sim | Std | Range | Entropy |
|-------|----------|-----|-------|---------|
| A (MiniLM, 384d) | 0.018 | 0.12 | [−0.26, 0.30] | 2.04 |
| B (E5-large, 1024d) | **0.721** | 0.07 | [0.56, 0.89] | **1.40** |
| C (BGE-small, 384d) | 0.402 | 0.12 | [0.15, 0.75] | 1.89 |
| E (BGE-large, 1024d) | 0.453 | 0.11 | [0.12, 0.78] | 1.93 |

| Normalization | B×C R@1 | A×B R@1 | A×C R@1 |
|--------------|---------|---------|---------|
| None (baseline) | 8.4% | 80.6% | 98.0% |
| **z-score (DB-side)** | **73.4%** | **81.0%** | **98.2%** |
| Rank transform | 52.0%* | 67.2%* | 97.6%* |
| Top-k mask (k=50) | 50.2%* | 54.0%* | 83.0%* |
| Softmax (T=0.1) | 50.2%* | 34.6%* | 44.8%* |

*Rank, top-k, and softmax results from a separate run with both-side application (Appendix B); exact values differ from the z-score rows due to run-to-run variation (±3–5 points) but the relative ordering is stable across all runs.

In Model B's space, all anchor pairs have cosine similarity > 0.56, with a mean of 0.72 and an effective range of only 0.33 (Figure 3). This compression — a consequence of mapping diverse multilingual content into a shared space — renders relative representation profiles nearly flat: all anchors appear approximately equidistant from any query, destroying discriminability.

We term this phenomenon **similarity collapse**: the failure of relative representations when the anchor similarity distribution is compressed to a narrow range. The critical diagnostic is anchor inter-similarity entropy (Shannon entropy of the binned similarity histogram). Model B's entropy of 1.40 is far below Model A's 2.04, indicating a concentrated, low-information distribution.

**Why z-score is the right fix.** z-score normalization rescales each relative representation vector to zero mean and unit variance. When the original distribution is compressed (B), this stretching dramatically increases discriminability (+65 points on B×C). When it is already well-spread (A), the effect is negligible (+0.4 points on A×B). This "stretch if compressed, preserve if spread" property makes z-score uniquely safe across all pairs.

The alternative normalizations fail because they are **information-destructive**: rank transform discards magnitude differences; top-k masking zeros out most entries; softmax with low temperature over-sharpens, amplifying noise. z-score is the only method that preserves the relative ordering and magnitude structure while correcting the scale.

**Dataset interaction.** On AllNLI, B×C improves to 84.2% (+20.2 vs STSBenchmark). AllNLI's greater semantic diversity provides more varied anchor response patterns, partially alleviating collapse even before normalization. This suggests that similarity collapse severity depends on both the model's space geometry and the data's semantic coverage.

## 5.2 Asymmetric z-score: DB-side Sufficiency

The z-score results in §4 apply normalization to both query and database sides. We now decompose this into four configurations: no normalization, query-only, DB-only, and both.

**Table 4: Asymmetric z-score analysis (Recall@1 %).**

*Text pairs (STSBenchmark):*

| Pair | None | Query only | DB only | Both |
|------|------|-----------|---------|------|
| A×B | 80.6 | 72.0 | **81.0** | 81.0 |
| A×C | 98.0 | 90.2 | **98.2** | 98.2 |
| B×C | 8.4 | 52.8 | **73.4** | 73.4 |

*Cross-modal:*

| Direction | None | Query only | DB only | Both |
|-----------|------|-----------|---------|------|
| Text → Image | **16.4** | 8.2 | 15.6 | 15.6 |
| Image → Text | 3.8 | 10.0 | **18.2** | 18.2 |

A striking pattern emerges: **DB-only z-score exactly matches both-side z-score across all pairs and directions.** Adding query-side normalization provides zero additional benefit. Conversely, query-only z-score consistently underperforms, degrading well-spread models by 8–10 points (A×B: −8.6, A×C: −7.8).

**Explanation.** z-score normalization estimates distributional statistics (mean and variance) from a vector's K entries. On the DB side, the normalized representations are computed once offline and the statistics are stable — each of the $n$ database vectors is independently normalized using its own K-dimensional profile. The normalization ensures all database items are on a comparable scale, regardless of the underlying model's similarity compression.

On the query side, a single vector's K entries yield one mean and one variance estimate. When the query model's space is already well-spread (Model A, entropy=2.04), this normalization distorts the relative magnitudes that carry discriminative signal. The harm is smaller when the query model is compressed (Image→Text: image entropy=2.18 vs text entropy=2.42), but even then, DB-side normalization alone captures the full benefit.

**Practical implication.** The optimal RAT protocol applies z-score normalization only to the database side, simplifying the procedure: no per-query normalization is needed, and no direction-dependent rules are required.

## 5.3 Dimensions Don't Matter: Shared Relative Structure

Models B (E5-large) and E (BGE-large) share the same output dimensionality (1024d) but differ in architecture (XLM-RoBERTa vs BERT), training data (multilingual vs English), and training objective. Their direct cosine similarity on identical texts averages −0.02 — **indistinguishable from random projections**. The absolute coordinate systems are entirely uncorrelated.

Yet RAT achieves Recall@1 = 55.2% on B×E. Meanwhile, A×B (384d × 1024d, different dimensions) achieves 76.2%.

| Pair | Dimensions | Direct cos sim | RAT R@1 |
|------|-----------|---------------|---------|
| B×E | 1024 × 1024 | −0.02 | 55.2% |
| A×B | 384 × 1024 | — | 76.2% |
| A×E | 384 × 1024 | — | 80.2% |

This demonstrates two points:

1. **Dimensional agreement is irrelevant to RAT performance.** Same-dimension B×E (55.2%) underperforms cross-dimension A×B (76.2%) by 21 points. What matters is the structural compatibility of the underlying semantic spaces — how similarly the models organize concepts relative to each other.

2. **Relative structure persists even when absolute coordinates are uncorrelated.** B and E encode identical texts into vectors with zero correlation, yet their *patterns of similarity to common anchors* share enough structure for 55% retrieval accuracy. This is not a trivial observation: it means that the two models, trained independently on different data with different architectures, have converged on partially overlapping semantic geometries that are only visible through relative comparison.

This finding resonates with the Platonic Representation Hypothesis (Huh et al., 2024), which posits that models trained on sufficient data converge toward a shared statistical model of reality. RAT provides a concrete operationalization of this hypothesis: the Recall@1 between two models' relative representations is a quantitative, task-grounded measure of their structural alignment — one that is computable without any model-internal access, requiring only the ability to encode shared anchors. B×E's 55% — high enough to be useful, low enough to indicate meaningful differences — suggests that convergence is real but partial, and varies with training conditions. The spectrum of RAT scores across model pairs (55%–98%) offers a finer-grained picture than the binary "converged or not" framing of the original hypothesis.
