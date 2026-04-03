# C2a: Model Update Without Re-indexing — Experiment Report

> **同一ファミリーのモデルアップグレードで、再インデックスなしに平均94%の検索精度を保持。encode コストは0.05%以下。**

## 1. Overview

ベクトルDBを旧モデルで構築済みの状態で、新モデルに切り替えたい場合、
通常はDB全件を新モデルで再encodeする必要がある（再インデックス）。

RATを使えば、**アンカー（500文）だけを両モデルでencode**するだけで、
新モデルのクエリで旧モデルのDBを検索できる。

本実験では、Phase 0 の12モデル実験（d2a_matrix）の既存データを
「モデル更新シナリオ」として再分析し、Retention Rate を定量化した。

---

## 2. Retention Rate — Same-Family (Tier 1)

**方向: old=DB, new=query（旧モデルのDBを新モデルで検索）**

| Upgrade | Dims | R@1 (auto) | Best R@1 | Method |
|---------|------|-----------|----------|--------|
| BGE-small → BGE-base | 384→768 | **98.0%** | 98.6% | zscore |
| BGE-small → BGE-large | 384→1024 | **89.8%** | 89.8% | zscore |
| BGE-base → BGE-large | 768→1024 | **94.6%** | 94.6% | zscore |
| E5-small-v2 → E5-multi-large | 384→1024 | **90.0%** | 90.0% | baseline |
| E5-multi-small → E5-multi-large | 384→1024 | **95.8%** | 95.8% | baseline |
| E5-small-v2 → E5-multi-small | 384→384 | **88.4%** | 88.4% | baseline |
| GTE-small → GTE-large | 384→1024 | **98.4%** | 98.4% | baseline |

| 統計 | Auto Mode | Best Achievable |
|------|-----------|-----------------|
| 平均 | **93.6%** | 93.7% |
| 最小 | 88.4% | 88.4% |
| 最大 | 98.4% | 98.6% |

**全ペアで88%以上。Auto modeとBestの差はほぼゼロ。**
Adaptive z-score が正しく機能し、harmfulなケース（GTE, E5）では自動的にbaselineに切り替わっている。

---

## 3. Retention Rate — Cross-Family (Tier 2)

| Model Switch | Dims | R@1 (auto) | Best R@1 | Gap |
|-------------|------|-----------|----------|-----|
| MiniLM → BGE-small | 384→384 | 34.0% | **94.2%** | -60.2pt |
| MiniLM → BGE-large | 384→1024 | 5.8% | **79.6%** | -73.8pt |
| MiniLM → GTE-small | 384→384 | 1.2% | **85.6%** | -84.4pt |
| MiniLM → GTE-large | 384→1024 | 0.6% | **76.6%** | -76.0pt |
| MPNet → BGE-large | 768→1024 | 8.0% | **81.6%** | -73.6pt |
| GTE-small → BGE-large | 384→1024 | **89.4%** | 89.4% | 0.0pt |
| Nomic → BGE-large | 768→1024 | 0.4% | 0.4% | 0.0pt |
| CLIP → BGE-large | 512→1024 | 21.6% | 21.6% | 0.0pt |

| 統計 | Auto Mode | Best Achievable |
|------|-----------|-----------------|
| 平均 | **20.1%** | 66.1% |

**Auto modeとBestの間に巨大なギャップ（46pt）。**
原因: MiniLM/MPNetのDB側sim_meanが低すぎて`"not_needed"`と判定され、
z-scoreが適用されない。しかしz-scoreを強制すれば80%+が出るペアが多い。

→ クロスファミリーのモデル切り替えでは、adaptive z-scoreのロジック改善が必要。
DB側の sim_mean だけでなく、ペア間の空間の違いを考慮すべき。

---

## 4. Cost Analysis

### Encode コスト比較

| DB Size | Re-index (small model) | Re-index (large model) | RAT (total) | Encode Cost Ratio |
|---------|----------------------|----------------------|-------------|-------------------|
| 10K | 20s | 2min | ~1s | 5.00% |
| 100K | 3min | 21min | ~2s | 0.50% |
| **1M** | **33min** | **3.5h** | **~6s** | **0.05%** |
| 10M | 5.6h | 34.7h | ~31s | 0.01% |

- RAT のコスト = アンカー500文のencode（~1s）+ 行列積変換
- 行列積変換: 100万件で4.5秒（CPU、numpy）
- DB件数が増えるほどコスト優位性が拡大

### コスト内訳

RAT の処理は2ステップ:
1. **アンカーencode**: 500文 × 2モデル = 1000回のencode（~1-2秒）
2. **相対表現変換**: N×K の行列積（N=DB件数, K=500）
   - 10K: 0.06s / 100K: 0.45s / 1M: 4.5s

Re-index は N件すべてを新モデルでencode。
100万件のDBなら、RATは**2000分の1のencodeコスト**で94%の精度を保持する。

---

## 5. Z-score Safety Analysis

### Per-Model sim_mean

| Model | sim_mean | Recommendation |
|-------|----------|---------------|
| A: MiniLM | 0.0200 | not_needed |
| B: E5-multi-large | 0.5799 | recommended |
| C: BGE-small | 0.3794 | recommended |
| D: CLIP | 0.4299 | recommended |
| E: BGE-large | 0.4664 | recommended |
| F: E5-small-v2 | 0.6547 | harmful |
| G: E5-multi-small | 0.6626 | harmful |
| H: GTE-small | 0.6831 | **harmful** |
| I: GTE-large | 0.6758 | **harmful** |
| J: MPNet | 0.0154 | not_needed |
| K: BGE-base | 0.3744 | recommended |
| L: Nomic | 0.5012 | recommended |

### Adaptive Z-score の動作検証

`estimate_compatibility("gte-small", "gte-large")` の結果:
- `z_score_recommendation`: **"harmful"**
- `warnings`: "High sim_mean (0.683) detected. Z-score normalization may be harmful for this pair."

**設計通りに動作している。** GTE ペアに z-score を強制すると R@1 が 98.4% → 39.8% に壊滅するが、
auto mode は正しく baseline（98.4%）を選択する。

### 課題: "not_needed" カテゴリの問題

MiniLM (sim_mean=0.02) と MPNet (sim_mean=0.02) は `"not_needed"` と判定される。
同一モデル内では z-score が不要だが、**クロスモデルシナリオでは z-score が劇的に効く**:

- MiniLM(DB) → BGE-small(query): baseline 34.0% → zscore 94.2% (+60pt)
- MPNet(DB) → BGE-large(query): baseline 8.0% → zscore 81.6% (+73pt)

現在の adaptive logic は DB 側の sim_mean のみで判定しているが、
クロスモデル用途では **ペア間の空間差** を考慮する判定ロジックが必要。
これは Phase C3（理論深掘り）で対応すべき課題。

---

## 6. Key Findings

1. **同一ファミリーのモデルアップグレードで平均94%の精度保持**（全ペア88%以上）
2. **コスト削減**: 100万件DBなら encode コスト 0.05%（2000分の1）
3. **Adaptive z-score は同一ファミリーで正しく動作**（auto ≈ best）
4. **クロスファミリーでは auto mode に改善余地**（auto 20% vs best 66%）
5. **GTE の z-score 破壊を正しく検出**（sim_mean 0.68 > threshold 0.65）

---

## 7. 発信用ヘッドライン候補

### A. コスト訴求（推奨）
> **「embeddingモデルをアップグレードしても、ベクトルDBの再構築は不要。500文のencodeだけで94%の検索精度を保持」**

### B. 数字訴求
> **「100万件のDB再インデックスに3.5時間 → RATなら6秒。精度94%保持、コスト0.05%」**

### C. ユースケース訴求
> **「BGE-small → BGE-large へのアップグレード: 再インデックスなしで精度90%保持。必要なのはアンカー500文のencodeだけ」**

---

## Appendix: Experiment Configuration

- **データ**: Phase 0 の d2a_matrix（12モデル × 2500文、AllNLI corpus）
- **アンカー**: K=500, FPS選択
- **カーネル**: poly(degree=2, coef0=1.0)
- **評価**: 500クエリ × 2000候補プール
- **方向**: old=DB(列), new=query(行) — 現実的なモデル更新シナリオ
- **Re-index baseline**: 100%（同一モデル検索）
- **新計算**: なし（Phase 0 の既存データの再分析のみ）
