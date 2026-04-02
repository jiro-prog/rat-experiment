# §2 Method — Draft

## 2.1 Relative Anchor Representation

Let $f_X: \mathcal{T} \to \mathbb{R}^{d_X}$ and $f_Y: \mathcal{T} \to \mathbb{R}^{d_Y}$ be two embedding models mapping inputs to L2-normalized vectors, where $d_X \neq d_Y$ in general. Given a shared set of $K$ anchor texts $\mathcal{A} = \{a_1, \dots, a_K\}$, we define the **relative anchor representation** of an input $x$ under model $X$ as:

$$r_X(x) = \left[ \kappa(f_X(x), f_X(a_1)), \dots, \kappa(f_X(x), f_X(a_K)) \right] \in \mathbb{R}^K$$

where $\kappa$ is a kernel function measuring similarity between embeddings. This transforms vectors from model-specific $d_X$-dimensional spaces into a shared $K$-dimensional space, where $K$ is independent of the original dimensions.

**Kernel choice.** We use the polynomial kernel $\kappa(u, v) = (u^\top v + 1)^2$, which outperforms cosine similarity ($\kappa = u^\top v$) by +15 percentage points on Recall@1 (Table A1 in Appendix). The quadratic nonlinearity amplifies differences among the highest-similarity anchors, improving discrimination at the top of the ranked list where retrieval precision matters most.

## 2.2 Anchor Selection via Farthest Point Sampling

Random anchor selection suffers from **density bias**: anchors cluster in high-density regions of the embedding space, leaving sparse regions unrepresented. We apply Farthest Point Sampling (FPS) to maximize spatial coverage:

1. Select $a_1$ uniformly at random from a candidate pool $\mathcal{C}$ ($|\mathcal{C}| = 2000$)
2. For $i = 2, \dots, K$: select $a_i = \arg\min_{c \in \mathcal{C} \setminus \{a_1, \dots, a_{i-1}\}} \max_{j < i} \, f_X(c)^\top f_X(a_j)$

FPS is executed in a single model's space (Model A); the selected indices are shared across all models. This greedy $O(KN)$ procedure runs in seconds on CPU.

**Efficiency gain.** FPS with $K{=}100$ anchors achieves Recall@1 = 55.0%, exceeding random selection with $K{=}1000$ (48.2%). The full protocol (FPS + polynomial kernel + z-score) at $K{=}100$ is competitive with random anchors at 10$\times$ the budget (§4.3, Figure 2).

## 2.3 DB-side z-score Normalization

When anchor inter-similarity is compressed into a narrow range (e.g., multilingual E5-large: mean pairwise similarity = 0.72, effective range = 0.33), relative representations become **flat** — all entries are similar, destroying discriminability. We term this failure mode **similarity collapse**.

We address this with z-score normalization applied **exclusively to the database (DB) side**:

$$\hat{r}_Y(y_j) = \frac{r_Y(y_j) - \mu_j}{\sigma_j}, \quad \mu_j = \frac{1}{K}\sum_k r_Y(y_j)_k, \quad \sigma_j = \text{std}(r_Y(y_j))$$

The query representation $r_X(x)$ is left unnormalized.

**Why DB-side only?** Exhaustive evaluation across all model pairs and both cross-modal directions reveals that `db_only` z-score matches `both` z-score exactly, while `query_only` z-score consistently degrades performance (−8 to −9 points on well-spread models; §5.2, Table 4). Intuitively, z-score normalizes the "catalog" for uniform browsability, while the query retains its original distribution for accurate matching.

## Algorithm 1: RAT Protocol

**Input:**
- Query $x$ embedded by model $X$: $f_X(x) \in \mathbb{R}^{d_X}$
- Database $\{y_1, \dots, y_n\}$ embedded by model $Y$: $f_Y(y_j) \in \mathbb{R}^{d_Y}$
- Candidate anchor texts $\mathcal{C} = \{c_1, \dots, c_M\}$, $M \geq K$

**Offline (one-time per model pair):**
1. Encode candidates: $E_X = f_X(\mathcal{C})$, $E_Y = f_Y(\mathcal{C})$
2. Select $K$ anchors via FPS on $E_X$: $\mathcal{A} = \text{FPS}(E_X, K)$
3. Store anchor embeddings: $A_X = f_X(\mathcal{A})$, $A_Y = f_Y(\mathcal{A})$
4. Compute DB relative representations: $R_Y = [(y_j \cdot a + 1)^2]_{j,a} \in \mathbb{R}^{n \times K}$
5. Apply z-score to $R_Y$ (row-wise)

**Online (per query):**
1. Compute query relative representation: $r_X(x) = [(x \cdot a + 1)^2]_a \in \mathbb{R}^K$
2. Retrieve: $\hat{j} = \arg\max_j \cos(r_X(x), \hat{R}_Y[j])$

**Complexity:** Offline: $O(KN)$ for FPS + $O(nK)$ for DB encoding. Online: $O(nK)$ per query — same as standard nearest-neighbor search in $K$ dimensions.

---

## §3 Cross-Modal Extension: Rosetta Stone Anchors

The text-only formulation requires anchors $\mathcal{A}$ that both models can encode. For cross-modal retrieval between a text encoder $f_T$ and an image encoder $f_I$, we introduce **Rosetta Stone anchors**: a set of $K$ concept-aligned pairs $\{(t_k, i_k)\}_{k=1}^K$ where $t_k$ is a text caption and $i_k$ is the corresponding image.

Each model encodes its own modality:

$$r_T(x) = \left[\kappa(f_T(x), f_T(t_1)), \dots, \kappa(f_T(x), f_T(t_K))\right]$$
$$r_I(v) = \left[\kappa(f_I(v), f_I(i_1)), \dots, \kappa(f_I(v), f_I(i_K))\right]$$

The key insight is that both representations encode **the same concept's response pattern**, just measured through different modalities. If concept $k$ is "a dog playing in water," the text encoder measures how text-similar the query is to that concept, while the image encoder measures how visually similar the query is. These response patterns align because the underlying semantic structure is shared (Huh et al., 2024).

We use image-caption pairs from the COCO Karpathy test split (Lin et al., 2014), with FPS applied in the text encoder's space.

**Result:** A text-only model (MiniLM, 384d) achieves Recall@1 = 18.2% when searching a CLIP image space (512d), with zero training on visual data. The text encoder has never seen an image; the image encoder has never seen text. The 500 Rosetta Stone anchors are the only bridge.

---

## 記法まとめ

| 記号 | 意味 |
|------|------|
| $f_X, f_Y$ | Embedding models (L2-normalized output) |
| $d_X, d_Y$ | Original embedding dimensions |
| $K$ | Number of anchors (default: 500) |
| $\mathcal{A}$ | Anchor set |
| $\kappa(u,v)$ | Kernel function: $(u^\top v + 1)^2$ |
| $r_X(x)$ | Relative representation of $x$ under model $X$ |
| $R_Y$ | DB-side relative representation matrix ($n \times K$) |
| $\hat{R}_Y$ | z-score normalized $R_Y$ |
| $(t_k, i_k)$ | Rosetta Stone anchor pair (caption, image) |
