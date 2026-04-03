# RAT: Embedding Space Translation via Relative Anchor Similarity Profiles

**Format**: Short paper (6 pages + references + appendix), arXiv preprint
**Target venues**: EMNLP Findings, ACL SRW, or standalone arXiv

---

## Abstract (15 lines)

- 問題: 異なるembeddingモデルの出力は直接比較できない。学習ベースのアラインメントは対訳データと訓練コストが必要
- 提案: RAT (Relative Anchor Translation) — 共通アンカーとの類似度プロファイルに変換するだけで、追加学習なしにzero-shotで空間変換
- プロトコル: FPS(K=500) + poly kernel + adaptive z-score(DB側)
- 結果サマリ:
  - 12モデル×110ペア検証: 同一ファミリーR@1中央値=95%、異ファミリー中央値=80%
  - 同系統ペア(A×C): R@1=98.4%（500件DB）、100kDBでも90.6%を維持
  - Moschella et al.比で最大+71.6pt改善（B×C: 5.0%→76.6%）
  - similarity compression分析: sim_mean vs R@1相関ρ=-0.62
  - 応用: モデル更新94%保持(コスト0.05%)、6x高速ゲートウェイ、マルチDB統合検索
- 意義: Platonic Representation Hypothesisの実験的証拠。モデル交換のコストを劇的に下げる

---

## §1 Introduction (1 page)

### 問題設定 (0.4p)
- Embedding modelは増殖し続けている。新モデルが出るたびに全データの再embeddingが必要
- モデル間の空間は次元もスケールも異なり、直接比較不能
- 既存解: Procrustes alignment、distillation → いずれも対訳データ + 訓練が必要

### 提案の核心 (0.3p)
- 共通のK個のアンカーポイントとの類似度プロファイルが、モデルを跨いで共有構造を持つ
- この「相対表現」への変換だけで、追加学習ゼロのzero-shot空間変換が可能
- Moschella et al. (2023) の Relative Representations を拡張し、(1) FPSアンカー選定、(2) 多項式カーネル、(3) adaptive z-score正規化の3要素で実用レベルに引き上げた

### Contributions (0.3p)
1. **RAT Protocol**: FPS + poly + adaptive z-score の3ステップで、12モデル110ペア検証。同一ファミリーR@1中央値95%
2. **Similarity Compression の発見と解決**: 高sim_meanモデルの相対表現弁別力低下を同定し、adaptive z-scoreで自動対応（sim_mean vs R@1: ρ=-0.62）
3. **応用シナリオの実証**: モデル更新(94%保持)、軽量ゲートウェイ(6x高速)、マルチDB統合検索(R@5=96%)
4. **オープンソース**: pip-installable ライブラリ `rat-embed` として公開

---

## §2 Related Work (0.5 page)

### Relative Representations (Moschella et al., 2023)
- RATの理論的基盤。cosineカーネル + ランダムアンカーでzero-shot stitching を提案
- RATの拡張: (1) FPSで10倍効率化、(2) polyカーネルで+12〜15pt、(3) adaptive z-scoreでCollapse解決
- **本論文の直接比較**: Table 1でMoschella手法の再現実装との定量比較を実施

### Platonic Representation Hypothesis (Huh et al., 2024)
- 「十分に大きいモデルは同じ表現に収束する」という仮説
- RATの結果は整合: 絶対座標が無相関でも相対構造が共有される
- 12モデル実験: 構造的互換性の度合いはペアごとに異なり、sim_meanで予測可能

### Model Stitching / Alignment
- Procrustes, CCA, Distillation — いずれも対訳データ + 訓練が必要
- RATはzero-shot・CPU・秒単位 — 根本的に異なるアプローチ

---

## §3 Method (1 page)

### 3.1 Relative Anchor Representation (0.3p)
- 定義: r(x) = [k(x, a_1), k(x, a_2), ..., k(x, a_K)]
- 任意のembedding空間の点をK次元ベクトルに変換
- 異なるモデルの出力が同一のK次元空間に落ちる
- カーネル選択: 多項式 k(x,a) = (x·a + 1)² — cosineより+12〜15pt

### 3.2 Farthest Point Sampling (0.3p)
- ランダムアンカーは密集バイアスを持つ
- FPS: 既選択アンカーから最も遠い点を貪欲に追加。K=100で、ランダムK=1000を超える
- **条件**: アンカー空間のエントロピーが十分高いモデルでのみ有効

### 3.3 Adaptive Z-score Normalization (0.4p)
- 類似度圧縮（Similarity Compression）への対策
- DB側の相対表現を行ごとに平均0、分散1に正規化
- **Adaptive判定**: アンカー間の平均コサイン類似度 sim_mean で自動判定
  - sim_mean < 0.65: z-score適用（推奨）
  - sim_mean ≥ 0.65: z-scoreスキップ（harmful — 弁別力をさらに破壊）
- DB側のみで十分、クエリ側は不要（A×Bで-9pt）
- 閾値0.65の根拠: GTE(sim_mean=0.68)でz-score適用時R@1が98.4%→39.8%に壊滅

### Algorithm 1: RAT Protocol
```
Input:  query embedding x in R^d1 (Model X)
        database embeddings Y in R^{n*d2} (Model Y)
        shared anchor texts T = {t_1, ..., t_K}

1. Encode anchors: A_X = ModelX(T), A_Y = ModelY(T)
2. FPS: Select K anchors from candidate pool using A_X
3. Relative repr: r_X(x) = [(x*a+1)^2 for a in A_X]
                  R_Y(Y) = [(y*a+1)^2 for a in A_Y]
4. Adaptive z-score: if sim_mean(A_Y) < 0.65:
                       R_Y <- (R_Y - mu_row) / sigma_row  (DB-side only)
5. Retrieve: argmax_j cos(r_X(x), R_Y[j])
```

---

## §4 Experiments (1.5 pages)

### 4.1 Setup (0.3p)
- **Models**: 12 text encoders spanning 5 families:
  - BGE: small(384d), base(768d), large(1024d)
  - E5: small-v2(384d), multilingual-small(384d), multilingual-large(1024d)
  - GTE: small(384d), large(1024d)
  - MiniLM-L6(384d), MPNet-base(768d), Nomic-v1.5(768d), CLIP-ViT-B/32(512d)
- **Data**: AllNLI (2500 texts), COCO Karpathy (5K image-caption pairs)
- **Protocol**: FPS(K=500) + poly(d=2, c=1) + adaptive z-score + cosine kNN
- **Metric**: Recall@1, MRR (500 queries × 2000 candidate pool)

### 4.2 Text × Text Results (0.6p)

**前半: 代表ペアでの主要傾向**

Table 1: 代表ペアの R@1（Moschella比較含む）

| Pair | Family | Dims | Moschella | RAT (ours) | Δ |
|------|--------|------|-----------|------------|---|
| BGE-small → BGE-base | same | 384→768 | — | **98.6%** | — |
| GTE-small → GTE-large | same | 384→1024 | — | **98.4%** | — |
| MiniLM → BGE-small | cross | 384→384 | 68.4% | **97.0%** | +28.6 |
| MiniLM → E5-large | cross | 384→1024 | 42.8% | **82.8%** | +40.0 |
| E5-large → BGE-small | cross | 1024→384 | 5.0% | **76.6%** | +71.6 |

- 同一ファミリー: 98%+（ほぼ劣化なし）
- 異ファミリー: z-score による Similarity Compression 解決が鍵（+71.6pt）

**後半: 110ペアの統計サマリー**

| Category | N pairs | R@1 median | R@1 min | R@1 max |
|----------|---------|-----------|---------|---------|
| Same family | 14 | 95.0% | 79.2% | 98.6% |
| Cross family | 96 | 80.4% | 0.2% | 97.0% |
| All | 110 | 82.8% | 0.2% | 98.6% |

- Figure 2: **sim_mean vs R@1 散布図**（ρ=-0.62）
  - sim_mean < 0.5 のペアは高精度、sim_mean > 0.65 は低精度ゾーン
  - adaptive z-score の閾値を視覚的に説明

**全110ペアの完全テーブルは Appendix A**

### 4.3 Database Scaling (0.3p)

Table 2: DB規模 vs R@1

| DB size | A×C R@1 | A×B R@1 | vs random |
|--------:|--------:|--------:|----------:|
|     500 |   98.4% |   81.2% |      492× |
|  10,000 |   94.4% |   68.6% |    9,440× |
| 100,000 | **90.6%** | **55.0%** | **90,600×** |

- A×C: 100kで90.6% — 実用水準
- 劣化は対数線形（壊滅的崩壊ではない）

### 4.4 Cross-Modal Results (0.3p)

Table 3: MiniLM × CLIP-image (A×E)

| K | R@1 | CLIP直接 |
|---|-----|---------|
| 500 | 16.8% | 59-62% |
| 3000 | **21.2%** | |

- MiniLMは画像の存在を知らないモデル。にもかかわらず21.2%で画像検索
- K=3000が最適、K=4000で次元の呪いにより低下

---

## §5 Analysis (1 page)

### 5.1 Similarity Compression (0.4p)
- 高sim_meanモデル（E5-large=0.58, GTE=0.68）は相対表現プロファイルがフラット化
- Figure 3: sim_mean vs R@1 散布図（§4.2のFig 2と同一、ここで詳細議論）
  - ρ=-0.62: sim_meanが高いほどR@1が低い
  - 閾値0.65以上で z-score が harmful に転じる理由: 既に圧縮された分布をさらに引き伸ばすと、ノイズが増幅される
- adaptive z-score の設計検証:
  - GTE pair: z-score適用で98.4%→39.8%（正しくスキップ）
  - MiniLM pair: z-score適用で34%→94%（正しく適用）

### 5.2 FPS: When It Helps and When It Doesn't (0.2p)
- 低sim_meanモデルで+11〜28pt、高sim_meanでは逆効果の場合あり
- z-scoreとの併用で全ペア安全

### 5.3 Scaling Behavior (0.2p)
- A×CとA×Bの劣化率: 500→100kでA×Cは-7.8pt、A×Bは-26.2pt
- DB規模が大きくなるほど、モデル間の構造的互換性の差が増幅される

### 5.4 Cross-Modal Analysis (0.2p)
- CLIP-text × CLIP-image (D×E) < MiniLM × CLIP-image (A×E)
- アンカー距離パターンρ: A×E=0.33 vs D×E=0.18
- CLIPのcontrastive lossは局所的距離順序を保存しない

---

## §6 Applications (0.5 page)

C2a-c の結果を3つの実用シナリオとして提示。新計算なし（既存データの再分析）。

### 6.1 Model Update Without Re-indexing (0.2p)
- ベクトルDBを旧モデルで構築済み → 新モデルに切り替え
- 通常: 全N件を再encode。RAT: アンカーK=500件のみ再encode
- **結果**: 同一ファミリー Retention Rate 平均94%（7ペア、全て88%以上）
- **コスト**: 100万件DB → re-index 3.5時間、RAT 6秒（encode比 0.05%）

Table 4: Retention Rate（同一ファミリー）

| Upgrade | R@1 | Retention |
|---------|-----|-----------|
| BGE-small → BGE-base | 98.0% | 98.0% |
| BGE-small → BGE-large | 89.8% | 89.8% |
| GTE-small → GTE-large | 98.4% | 98.4% |
| Mean (7 pairs) | — | **93.6%** |

### 6.2 Lightweight Query Gateway (0.15p)
- 安い小モデルでクエリ → 高精度大モデルのDBを検索
- GTE-small → GTE-large: **R@1=98.4%, 6.2x高速**
- MiniLM → BGE-large: R@1=82.4%, 6.2x高速（クロスファミリー）

### 6.3 Multi-Database Unified Search (0.15p)
- 異なるモデルで構築された複数DBの横断検索
- per-DB score normalization で R@1=68.8%, **R@5=95.6%**
- naive vstack は失敗（スコアスケール不一致）→ per-DB正規化が必須

---

## §7 Conclusion (0.3 page)

- RATは追加学習なしで異なるembedding空間を接続する軽量プロトコル
- 12モデル110ペア: 同一ファミリー中央値95%、異ファミリー中央値80%
- Similarity Compressionの発見とadaptive z-scoreによる自動対応
- 3つの応用シナリオ: モデル更新(94%保持)、ゲートウェイ(6x高速)、マルチDB(R@5=96%)
- `rat-embed` としてオープンソース公開（pip install rat-embed）
- 制限: 英語中心、テキスト中心、クロスモーダルは概念実証段階
- 今後: 大規模ベンチマーク（MTEB）、多言語、adaptive正規化の理論的基礎づけ

---

## Appendix

### A. Full 12-Model Matrix (110 pairs)
- R@1, MRR の完全マトリクス（baseline + z-score）
- 全モデルのsim_mean一覧

### B. Kernel Comparison
- cosine / RBF / poly(d=2) / poly(d=3) の全組み合わせ結果

### C. Anchor Selection Methods
- Random / k-means / FPS / Consensus の比較

### D. Multi-DB Score Normalization Comparison
- 3方式の比較テーブル:
  | Method | R@1 | R@5 | R@10 |
  |--------|-----|-----|------|
  | Per-DB score normalization | **68.8%** | **95.6%** | 99.2% |
  | Reciprocal Rank Fusion | 33.4% | 96.8% | 98.4% |
  | Naive vstack | 30.8% | 33.2% | 33.2% |
- z-score選択の根拠

### E. Cross-Modal Details
- アンカー距離パターンρ分析、バケット分析

### F. Reproducibility
- ハードウェア（CPU実行、WSL2）、ソフトウェアバージョン、乱数シード（seed=42）
- 全実験スクリプトのGitHubリンク
- `pip install rat-embed` でライブラリ再現

---

## Figures/Tables 配置計画

| ID | 内容 | 配置 | サイズ |
|----|------|------|--------|
| Fig 1 | ロゼッタストーン概念図（テキスト×テキスト + クロスモーダル） | §3 | 0.3p |
| Fig 2 | **sim_mean vs R@1 散布図** (ρ=-0.62, 110ペア) | §4.2/§5.1 | 0.3p |
| Fig 3 | DB scaling: A×C vs A×B (log-scale) | §4.3 | 0.2p |
| Table 1 | 代表ペア R@1 + Moschella比較 | §4.2 | 0.3p |
| Table 2 | DB scaling結果 (3規模×2ペア) | §4.3 | 0.15p |
| Table 3 | クロスモーダル結果 | §4.4 | 0.15p |
| Table 4 | Retention Rate（同一ファミリー7ペア） | §6.1 | 0.2p |

合計: Fig 0.8p + Table 0.8p = 1.6p → 本文 4.4p → 計6.0p

---

## ページ配分

| セクション | ページ | 備考 |
|-----------|--------|------|
| Abstract | 0.3 | |
| §1 Introduction | 0.8 | 問題→提案→貢献 |
| §2 Related Work | 0.5 | 3点 |
| §3 Method | 1.0 | Algorithm 1含む、adaptive z-score |
| §4 Experiments | 1.5 | Table 1,2,3 + Fig 2,3 |
| §5 Analysis | 1.0 | similarity compression + FPS + scaling |
| §6 Applications | 0.5 | Table 4 + C2c/C2b要約 |
| §7 Conclusion | 0.3 | |
| **合計** | **5.9** | References別。Appendix A-F |

---

## 想定される査読コメントと対策

| 想定コメント | 対策 |
|-------------|------|
| 「実験規模が小さい」 | **12モデル110ペア + 100k DBスケーリング**で対応。ショートペーパーのスコープとして明示 |
| 「Moschella et al.との差分が不明確」 | Table 1で再現実装との直接比較。4コンポーネントのablation |
| 「100kで55%は低い」 | A×Cの90.6%を先に出す。55%はランダム比55,000倍で文脈化 |
| 「adaptive z-scoreの閾値は恣意的」 | sim_mean vs R@1散布図で0.65が自然な境界であることを視覚的に示す |
| 「応用シナリオが机上の空論」 | 全て既存データの再分析。Retention Rateとコスト比で定量化 |
| 「マルチDB検索のR@1=68.8%は低い」 | R@5=95.6%を強調。rerankerとの組み合わせで実用的。Appendix Dで3方式比較 |
| 「クロスモーダル21%は低すぎる」 | zero-shot・学習なし。CLIP直接62%との差はクロスモーダルRATの限界として正直に議論 |
| 「Platonic仮説との関連は主張しすぎ」 | 「整合する」に留め、「証明した」とは書かない |
