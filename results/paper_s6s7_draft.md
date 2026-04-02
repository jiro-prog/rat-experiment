# §6 Limitations — Draft

Several limitations qualify the scope of our findings.

**Evaluation scale.** All experiments use 500 queries drawn from English-language datasets (STSBenchmark, AllNLI). While we demonstrate consistency across two datasets with different linguistic properties, evaluation on larger benchmarks (e.g., MTEB; Muennighoff et al., 2023) and multilingual/cross-lingual settings remains necessary to establish generality. The 500-query evaluation provides statistical power for detecting differences above ~4 percentage points (95% CI); finer-grained comparisons would require larger query sets.

**Cross-modal performance.** The Recall@1 of 18.2% for cross-modal retrieval, while 91× the random baseline, remains far below supervised systems. This result is best interpreted as a proof of concept demonstrating that modality-bridging via relative representations is possible in principle, rather than a practical retrieval system. The gap to CLIP's 62.0% (trained on 400M pairs) indicates substantial room for improvement — potentially through larger anchor sets, multi-scale kernels, or anchor pair quality optimization.

**Anchor selection dependency.** FPS is performed in a single model's space (Model A), and the selected anchors are shared across all models. While this simplifies the protocol, anchor quality is biased toward Model A's geometry. Experiments with alternative FPS bases (Appendix B) show modest variation (±2–3 points), largely absorbed by z-score normalization. For cross-modal settings, FPS in the text space may not optimally cover the image space; joint or alternating selection strategies are unexplored.

**Computational cost.** RAT's online retrieval cost is $O(nK)$ per query for computing cosine similarities in the $K$-dimensional relative space — identical to brute-force search in a $K{=}500$ dimensional space. For corpora of millions of items, this is tractable with standard approximate nearest-neighbor (ANN) indices (e.g., FAISS; Johnson et al., 2019) applied to the 500d relative representations. The offline cost of encoding $n$ database items against $K$ anchors is $O(nK \cdot d)$ dot products, comparable to a single pass of embedding computation. The dominant cost in practice is the initial embedding of anchors by each model, which is a one-time operation.

**Theoretical grounding.** We provide empirical evidence that relative similarity profiles are shared across models, but no formal proof of why this occurs. Our findings are consistent with the Platonic Representation Hypothesis (Huh et al., 2024), and RAT's cross-model Recall@1 can be interpreted as an operational measure of structural convergence. However, a formal characterization of which model properties guarantee high relative-representation alignment remains open.

**Model updates.** When an embedding model is updated (e.g., a new version release), anchors must be re-encoded under the new model. The anchor texts themselves can be reused, but the relative representations of existing database items become stale and require recomputation — the same cost as the original indexing.

---

# §7 Conclusion — Draft

We have presented Relative Anchor Translation (RAT), a zero-shot protocol for translating between embedding spaces without any training. Three simple, composable steps — Farthest Point Sampling for anchor selection, polynomial kernel similarity, and DB-side z-score normalization — enable cross-model retrieval across five embedding models with Recall@1 ranging from 55% to 98%, compared to a 0.2% random baseline.

Our experiments yield three findings that extend beyond the RAT protocol itself:

1. **Similarity collapse** is a previously uncharacterized failure mode of relative representations, caused by compressed anchor similarity distributions in certain model spaces (notably multilingual models). z-score normalization resolves this completely (+65 points on the most affected pair) while leaving well-spread models unharmed — a design principle applicable to any method operating on inter-point similarity distributions.

2. **z-score normalization is a DB-side operation.** Across all model pairs and both cross-modal directions, normalizing only the database side exactly matches both-side normalization, while query-side normalization is consistently harmful. This eliminates direction-dependent heuristics from the protocol.

3. **Relative structure persists when absolute coordinates are uncorrelated.** Two models sharing the same dimensionality (1024d) but with zero direct cosine correlation achieve 55% cross-model retrieval via RAT. This demonstrates that independently trained models develop partially overlapping semantic geometries visible only through relative comparison — providing a concrete, operational measure of the structural convergence posited by the Platonic Representation Hypothesis.

Finally, Rosetta Stone anchors extend RAT across modalities: a text-only encoder retrieves from a vision-only encoder's space at Recall@1 = 18.2% with zero visual training, using only 500 paired anchors. This proof of concept suggests that the anchor-based relative framework can bridge not just model differences, but fundamental representational modalities.

RAT's practical appeal lies in its simplicity: adding a new model to an existing system requires only encoding the shared anchors — no paired data, no gradient updates, no GPU. We release all code and experimental scripts to support reproducibility and further investigation.
