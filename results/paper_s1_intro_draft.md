# §1 Introduction — Draft

The proliferation of embedding models presents a growing interoperability problem. Organizations accumulate vector databases indexed by one model, only to find that newer or specialized models produce incompatible representations. Switching models requires re-embedding entire corpora — a cost that scales linearly with database size and becomes prohibitive for large-scale systems. Multi-model deployments, where different components use different encoders, face the same barrier: vectors from one model cannot be directly compared with vectors from another.

Existing approaches to embedding space alignment rely on learning a transformation between spaces. Procrustes alignment (Conneau et al., 2018) fits an orthogonal mapping from parallel data; knowledge distillation (Hinton et al., 2015) trains a student to mimic a teacher's outputs. These methods require paired training data, model-specific optimization, and separate transforms for each model pair — costs that grow quadratically with the number of models in a system.

We propose **Relative Anchor Translation (RAT)**, a zero-shot protocol that translates between arbitrary embedding spaces without any training. The core idea is simple: instead of comparing vectors directly, we compare their **similarity profiles** — vectors of similarities to a shared set of anchor points. When two models encode the same text, their patterns of similarity to common reference concepts tend to align, even though their absolute coordinates do not. This transforms the problem from aligning high-dimensional spaces of different dimensions to comparing response patterns in a shared, low-dimensional anchor space.

RAT builds on the relative representation framework of Moschella et al. (2023), who showed that cosine similarities to random anchors can enable zero-shot model stitching. We extend this framework in three ways that collectively raise performance from proof-of-concept to practical levels:

- **Farthest Point Sampling** for anchor selection eliminates the density bias of random anchors, achieving with 100 anchors what random selection requires 1,000 to match (§4.3).
- **Polynomial kernel** similarity $\kappa(u,v) = (u^\top v + 1)^2$ amplifies discriminability at the top of the ranked list, adding +11 points over cosine similarity.
- **DB-side z-score normalization** resolves a previously uncharacterized failure mode — **similarity collapse** — where compressed similarity distributions in certain model spaces (e.g., multilingual encoders) flatten relative representations and destroy discriminability. This single correction recovers +50 points on the affected pair (§5.1).

We evaluate RAT on five text embedding models spanning different architectures (BERT, XLM-R, CLIP), dimensionalities (384d to 1024d), and training objectives (English, multilingual, contrastive). All model pairs achieve Recall@1 > 55% on STSBenchmark and > 71% on AllNLI, against a random baseline of 0.2%. We further extend RAT across modalities using **Rosetta Stone anchors** — paired (image, caption) concepts that allow a text-only encoder to search a vision encoder's space at Recall@1 = 18.2%, with zero visual training.

Three analytical findings emerge from the experiments:

1. **Similarity collapse** in multilingual model spaces is diagnosable via anchor inter-similarity entropy and systematically fixable with z-score normalization — a finding applicable beyond RAT to any method relying on inter-point similarities (§5.1).

2. **z-score is a DB-side operation**: normalizing only the database representations exactly matches both-side normalization across all pairs and modalities, while query-side normalization is consistently harmful. This simplifies the protocol to a single, direction-independent rule (§5.2).

3. **Relative structure is shared even when absolute coordinates are uncorrelated**: two models with identical output dimensions (1024d) but zero direct cosine correlation achieve 55% cross-model retrieval, demonstrating that independently trained encoders develop overlapping semantic geometries visible only through relative comparison. RAT's cross-model Recall@1 provides a concrete operationalization of the structural convergence posited by the Platonic Representation Hypothesis (Huh et al., 2024) (§5.3).
