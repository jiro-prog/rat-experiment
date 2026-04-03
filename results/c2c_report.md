# C2c: Lightweight Gateway — Cheap Query Model, Expensive DB

> **安い小モデル（MiniLM, 22M params）でクエリを処理し、高精度な大モデル（BGE-large, 335M params）のDBを検索。クエリ encode が6倍速く、精度82%保持。**

## 1. Overview

高精度な embedding モデルは encode コストが高い。
ベクトルDB は大モデルで構築して検索精度を最大化したいが、
クエリ処理のレイテンシやコストがボトルネックになる。

RAT を使えば、**クエリ側だけ安い小モデル**で処理し、
大モデルで構築済みの DB をそのまま検索できる。
DB の再構築は不要。アンカー500文の encode だけで接続できる。

本レポートは Phase 0 の d2a_matrix データを「軽量ゲートウェイ」として再分析したもの。
新実験は行っていない。

---

## 2. Gateway Performance

**方向: query=小モデル（安い・速い）, DB=大モデル（高精度）**

### Tier 1: Small → Large（主要シナリオ）

| Gateway | Family | Q dim | DB dim | R@1 | MRR | Query Speedup |
|---------|--------|-------|--------|-----|-----|---------------|
| GTE-small → GTE-large | same | 384 | 1024 | **98.4%** | 99.1% | 6.2x |
| MiniLM → BGE-large | cross | 384 | 1024 | **82.4%** | 86.7% | 6.2x |
| MiniLM → GTE-large | cross | 384 | 1024 | **87.6%** | 92.4% | 6.2x |
| MiniLM → E5-large | cross | 384 | 1024 | **79.8%** | 86.3% | 6.2x |
| BGE-small → BGE-large | same | 384 | 1024 | **80.8%** | 75.2% | 6.2x |
| E5-small → E5-large | same | 384 | 1024 | **79.2%** | 83.8% | 6.2x |

### Tier 2: Same-dim / Base → Large

| Gateway | Family | Q dim | DB dim | R@1 | MRR | Query Speedup |
|---------|--------|-------|--------|-----|-----|---------------|
| MiniLM → BGE-small | cross | 384 | 384 | **97.2%** | 98.1% | 1.0x |
| MiniLM → GTE-small | cross | 384 | 384 | **93.2%** | 95.9% | 1.0x |
| MPNet → BGE-large | cross | 768 | 1024 | **86.6%** | 89.2% | 2.5x |
| BGE-base → BGE-large | same | 768 | 1024 | **89.4%** | 86.8% | 2.5x |

---

## 3. Speedup Analysis

### Query Encode Speed

| Model | Params | Dim | CPU (docs/sec) | GPU (docs/sec) |
|-------|--------|-----|---------------|----------------|
| MiniLM | 22M | 384 | ~500 | ~3,000 |
| BGE-small | 33M | 384 | ~500 | ~3,000 |
| MPNet | 109M | 768 | ~200 | ~1,500 |
| BGE-large | 335M | 1024 | ~80 | ~800 |

### Speedup の意味

Gateway を使わない場合、クエリも DB と同じ大モデルで encode する必要がある。

| 構成 | Query encode | 1000クエリ処理 |
|------|-------------|---------------|
| BGE-large (直接) | 80 docs/sec | 12.5秒 |
| MiniLM → BGE-large (Gateway) | 500 docs/sec | **2.0秒** |
| 差分 | 6.2x 高速 | **-10.5秒** |

バッチ処理では差が小さいが、**リアルタイム検索**では 1クエリあたりの
レイテンシが encode 時間に直結するため、6.2x の差が体感に出る。

---

## 4. C2a との関係

C2a（モデル更新）と C2c（軽量ゲートウェイ）は同じ RAT メカニズムの異なる応用:

| | C2a: モデル更新 | C2c: 軽量ゲートウェイ |
|---|---|---|
| 動機 | 再インデックスコスト削減 | クエリレイテンシ削減 |
| 方向 | old=DB, new=query | cheap=query, expensive=DB |
| 主要指標 | Retention Rate | R@1 + Speedup |
| 同一ファミリー最良 | 98.6%（BGE-small→base） | 98.4%（GTE-small→large） |
| クロスファミリー最良 | 94.2%（MiniLM DB→BGE-small query） | 97.2%（MiniLM query→BGE-small DB） |

**データソースは同じ**（d2a_matrix）だが、行列の参照方向が異なる。
C2a は `r1[new, old]`、C2c は `r1[cheap, expensive]`。

---

## 5. 実用上の考慮

### Gateway が有効なケース

- **リアルタイム検索**: 1クエリあたりのレイテンシが重要。384d モデルなら CPU でも 2ms/query
- **高トラフィック API**: クエリ encode がボトルネック。小モデルでスループットを稼ぐ
- **エッジデバイス**: クライアント側で小モデル encode → サーバーの大モデル DB を検索

### Gateway が不要なケース

- **バッチ検索**: GPU で大モデルを使えばスループットは十分
- **精度最優先**: 82% では不十分で、100% が必要なケース
- **同一モデル使用可**: クエリと DB で同じモデルを使えるなら Gateway は不要

### 精度と速度のトレードオフ

| 構成 | R@1 | Query速度 | 推奨用途 |
|------|-----|-----------|----------|
| BGE-large 直接 | 100% | 1x (遅い) | 精度最優先 |
| BGE-base → BGE-large | 89.4% | 2.5x | バランス型 |
| MiniLM → BGE-large | 82.4% | 6.2x | レイテンシ優先 |
| GTE-small → GTE-large | 98.4% | 6.2x | **同一ファミリー最適** |

**同一ファミリーの small → large が最もコスパが良い。**
GTE-small → GTE-large は 98.4% の精度を維持しながら 6.2x の速度改善。

---

## 6. 発信用ヘッドライン候補

### A. レイテンシ訴求
> **「クエリ encode を6倍高速化。22Mパラメータの MiniLM で、335M の BGE-large DB を検索。精度82%保持」**

### B. 同一ファミリー訴求（推奨）
> **「GTE-small → GTE-large: クエリ速度6倍、精度98%。軽量モデルで高精度DBを検索する Gateway パターン」**

### C. コスト訴求
> **「GPU不要のクエリ処理。CPU上の小モデル（2ms/query）で、大モデルのベクトルDBを直接検索」**

---

## Appendix: Data Source

- **元データ**: Phase 0 d2a_matrix（12モデル × 2500文、AllNLI corpus）
- **アンカー**: K=500, FPS選択, poly kernel
- **z-score**: v0.1.1 auto mode（harmful 以外は適用）
- **新計算**: なし（d2a_matrix の再分析のみ）
- **Speedup 推定**: sentence-transformers 典型値（CPU, batch_size=64）
