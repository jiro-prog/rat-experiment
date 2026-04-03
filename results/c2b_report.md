# C2b: Multi-Model RAG — Cross-DB Unified Search

> **異なるモデルで構築された3つのDBを、RATHub で横断検索。per-DB スコア正規化で R@1=68.8%、R@5=95.6%。naive vstack は完全に失敗する。**

## 1. Overview

組織内に異なる embedding モデルで構築された複数のベクトル DB がある。
1つのクエリモデルで全 DB を横断検索したい。

RATHub は各 DB のアンカーだけ共通モデルで encode すれば、異なるモデルの
相対表現を同じ K 次元空間に写像する。しかし **スコアの cross-DB comparability**
は自明ではない。

本実験では、500 文を 3 つの DB に分割し、各 DB を異なるモデルで encode。
BGE-large のクエリで横断検索し、統合手法（vstack / score normalization / RRF）を比較。

---

## 2. Experimental Setup

| | DB1 | DB2 | DB3 |
|---|---|---|---|
| Texts | [0:167] | [167:334] | [334:500] |
| Model | BGE-large (E) | MiniLM (A) | GTE-small (H) |
| Dim | 1024 | 384 | 384 |
| Relation to query | Same model | Cross-family | Cross-family |

- **Query model**: BGE-large (E), all 500 texts
- **Anchors**: K=500, FPS on BGE-large candidates, **shared indices** across all models
- **Ground truth**: text identity (query[i] matches DB entry for same text)

---

## 3. Results

### Test A: Per-DB Pairwise (isolated search)

各 DB を独立に検索した場合の精度。RAT の基本性能。

| DB | Model | R@1 | R@5 | MRR |
|----|-------|-----|-----|-----|
| DB1 | BGE-large (same model) | **100.0%** | 100.0% | 100.0% |
| DB2 | MiniLM (cross-family) | **88.0%** | 97.6% | 92.7% |
| DB3 | GTE-small (cross-family) | **92.8%** | 100.0% | 96.0% |

RAT のペアワイズ検索は正常に動作する。

### Test B: Unified Search（横断検索）

3 つの DB の結果を統合して 500 件プールから検索。

| Method | R@1 | R@5 | R@10 | MRR | Retention |
|--------|-----|-----|------|-----|-----------|
| Baseline (re-index) | 100% | 100% | 100% | 100% | — |
| **Score normalization** | **68.8%** | **95.6%** | **99.2%** | 79.5% | 68.8% |
| RRF (k=60) | 33.4% | 96.8% | 98.4% | 60.2% | 33.4% |
| Naive vstack | 30.8% | 33.2% | 33.2% | 32.2% | 30.8% |

### Score normalization の per-DB 内訳

| DB | R@1 (isolated) | R@1 (unified) | Degradation |
|----|-----------------|---------------|-------------|
| DB1 (BGE-large) | 100.0% | 71.3% | -28.7pt |
| DB2 (MiniLM) | 88.0% | 85.0% | -3.0pt |
| DB3 (GTE-small) | 92.8% | 50.0% | -42.8pt |

DB3(GTE-small) の劣化が大きい。GTE の similarity compression が
score normalization 後も影響している可能性。

---

## 4. Why Naive vstack Fails

相対表現のノルムがモデル間で桁違いに異なる:

| DB | Model | Norm (mean) |
|----|-------|-------------|
| DB1 | BGE-large | 22.4 |
| DB2 | MiniLM | 22.4 |
| DB3 | GTE-small | **64.8** |

GTE-small は sim_mean=0.68（harmful 判定）のため z-score をスキップ。
生のカーネル値の大きさがそのまま残り、cosine similarity で
DB3 の文書が全クエリで top-1 を独占する（score=0.998）。

**RAT 自体は壊れていない。** ペアワイズ検索は各 DB で正常に動く。
問題は「異なるモデルの相対表現のスコアスケールが揃わない」こと。

---

## 5. Score Normalization vs RRF

### Score normalization（推奨）
各 DB に対する similarity score を、クエリごとに z-score 正規化してからマージ。
DB 間のスコアスケールを揃える。

```python
for each DB:
    sim = cosine_similarity(query_rel, db_rel)
    sim_normed = (sim - sim.mean(axis=1)) / sim.std(axis=1)
merged_scores = concat(sim_normed_per_db)
```

R@1=68.8% で RRF（33.4%）を大幅に上回る。
R@5=95.6% — top-5 にはほぼ正解が含まれる。

### RRF
各 DB の検索結果のランクのみを使ってマージ。
スコアの問題を回避するが、**DB 間の確信度の違いを無視**するため
R@1 が低い（DB1 の確信度 100% と DB2 の 88% が等価に扱われる）。
ただし R@5=96.8% は高く、reranking と組み合わせれば実用可能。

---

## 6. Design Implications for RATHub

### 現状の制約
`hub.transform()` の出力を vstack して統合検索すると失敗する。
これはドキュメントで明示すべき制約。

### 推奨アーキテクチャ
```
Query → hub.transform(query_model, query, role="query")
         ↓
    ┌────┼────┐
    DB1  DB2  DB3  ← 各 DB に独立に検索
    ↓    ↓    ↓
  scores scores scores ← per-DB score normalization
    └────┼────┘
         ↓
    merged & sorted ← 統合ランキング
```

### ライブラリへのフィードバック
1. `hub.retrieve_multi()` API の追加を検討 — 内部で per-DB score normalization
2. vstack しないことを README / docstring で明記
3. `estimate_compatibility` に score scale の警告を追加

---

## 7. Key Findings

1. **ペアワイズ RAT は正常動作** — per-DB R@1 は 88-100%
2. **naive vstack は完全に失敗** — スコアスケールの不一致が原因
3. **per-DB score normalization で R@1=68.8%, R@5=95.6%** — 実用可能
4. **RRF は R@1 が低い（33.4%）** が R@5 は 96.8% — reranking 前提なら使える
5. **GTE(harmful) が特にスケール問題を起こす** — z-score スキップ時のノルム差が原因

---

## Appendix: Confidence Intervals

DB サイズ 167 件のため、per-DB R@1 の標準誤差は:

| DB | R@1 | SE (binomial) | 95% CI |
|----|-----|---------------|--------|
| DB1 | 71.3% | 3.5% | [64.4%, 78.1%] |
| DB2 | 85.0% | 2.8% | [79.6%, 90.4%] |
| DB3 | 50.0% | 3.9% | [42.4%, 57.6%] |

SE = sqrt(p(1-p)/n), n=167
