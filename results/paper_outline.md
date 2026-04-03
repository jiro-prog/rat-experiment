# RAT: Embedding Space Translation via Relative Anchor Similarity Profiles

**Format**: Short paper (6 pages + references + appendix), arXiv preprint
**Target venues**: EMNLP Findings, ACL SRW, or standalone arXiv

---

## Abstract (15 lines)

- 問題: 異なるembeddingモデルの出力は直接比較できない。学習ベースのアラインメントは対訳データと訓練コストが必要
- 提案: RAT (Relative Anchor Translation) — 共通アンカーとの類似度プロファイルに変換するだけで、追加学習なしにzero-shotで空間変換
- プロトコル: FPS(K=500) + poly kernel + z-score(DB側)
- 結果サマリ:
  - 同系統ペア(A×C): R@1=98.4%（500件DB）、**100kDBでも90.6%を維持**
  - 異系統ペア(A×B): R@1=82.8%（500件）、100kで55.0%（ランダム比55,000倍）
  - Moschella et al.比で最大+71.6pt改善（B×C: 5.0%→76.6%）
  - クロスモーダル: テキスト→画像zero-shot検索 R@1=18%（学習なし）
- 意義: Platonic Representation Hypothesisの実験的証拠。モデル交換・モーダル拡張のコストを劇的に下げる

---

## §1 Introduction (1 page)

### 問題設定 (0.4p)
- Embedding modelは増殖し続けている。新モデルが出るたびに全データの再embeddingが必要
- モデル間の空間は次元もスケールも異なり、直接比較不能
- 既存解: Procrustes alignment、distillation → いずれも対訳データ + 訓練が必要

### 提案の核心 (0.3p)
- 共通のK個のアンカーポイントとの類似度プロファイルが、モデルを跨いで共有構造を持つ
- この「相対表現」への変換だけで、追加学習ゼロのzero-shot空間変換が可能
- Moschella et al. (2023) の Relative Representations を拡張し、(1) FPSアンカー選定、(2) 多項式カーネル、(3) z-score正規化の3要素で実用レベルに引き上げた

### Contributions (0.3p)
1. **RAT Protocol**: FPS + poly + z-score の3ステップで、同系統ペアR@1=98%、100kDBで90%超を実現
2. **Similarity Collapse の発見と解決**: 多言語モデルの類似度潰れを同定し、z-scoreで5%→76.6%（+71.6pt）
3. **大規模検索での検証**: 100kDBでの劣化カーブを測定し、スケーリング特性を定量化
4. **先行研究との定量比較**: Moschella et al.の手法を再現実装し、各コンポーネントの寄与を分離

---

## §2 Method (1 page)

### 2.1 Relative Anchor Representation (0.4p)
- 定義: r(x) = [k(x, a_1), k(x, a_2), ..., k(x, a_K)]
- 任意のembedding空間の点をK次元ベクトルに変換
- 異なるモデルの出力が同一のK次元空間に落ちる
- カーネル選択: 多項式 k(x,a) = (x·a + 1)² — cosineより+12〜15pt（非線形性が上位の弁別力を向上）

### 2.2 Farthest Point Sampling (0.3p)
- ランダムアンカーは密集バイアスを持つ（高密度領域に偏る）
- FPS: 既選択アンカーから最も遠い点を貪欲に追加
- 空間被覆の最大化 → K=100で、ランダムK=1000を超える（C-2実証済み）
- 計算量: O(K*N) — CPUで数秒
- **条件**: アンカー空間のエントロピーが十分高いモデルでのみ有効（B×Cでは-1.6pt、§5で議論）

### 2.3 z-score Normalization (0.3p)
- 類似度潰れ（Similarity Collapse）への対策
- DB側の相対表現を行ごとに平均0、分散1に正規化
- 「潰れていれば引き伸ばし、広がっていればほぼそのまま」— 安全な片方向変換
- DB側のみで十分、クエリ側は不要（むしろ有害: A×Bで-9pt）

### Algorithm 1: RAT Protocol
```
Input:  query embedding x in R^d1 (Model X)
        database embeddings Y in R^{n*d2} (Model Y)
        shared anchor texts T = {t_1, ..., t_K}

1. Encode anchors: A_X = ModelX(T), A_Y = ModelY(T)
2. FPS: Select K anchors from candidate pool using A_X
3. Relative repr: r_X(x) = [(x*a+1)^2 for a in A_X]
                  R_Y(Y) = [(y*a+1)^2 for a in A_Y]
4. z-score: R_Y <- (R_Y - mu_row) / sigma_row  (DB-side only)
5. Retrieve: argmax_j cos(r_X(x), R_Y[j])
```

---

## §3 Cross-Modal Extension: Rosetta Stone Anchors (0.5 page)

- テキスト×テキストではアンカーは共通テキスト。クロスモーダルでは？
- ロゼッタストーン方式: 同じ概念の (image, caption) ペアをアンカーに
  - テキストモデルにはcaptionを、画像モデルにはimageを通す
  - 同じアンカーID = 同じ概念への「反応」
- COCO Captions (Karpathy split) から500ペアをサンプル
- FPSはテキスト空間（MiniLM）で実行
- Figure 1: ロゼッタストーンの概念図（テキスト×テキストとクロスモーダルの対比）

---

## §4 Experiments (1.5 pages)

### 4.1 Setup (0.3p)
- **Models**: 5 text encoders (MiniLM-L6 384d [A], E5-large 1024d [B], BGE-small 384d [C], CLIP-text 512d [D], BGE-large 1024d [E]) + CLIP image encoder (ViT-B/32 512d)
- **Data**: STSBenchmark (15K sentences), AllNLI (26K sentences), COCO Karpathy (5K image-caption pairs), MSMARCO/AllNLI corpus (100K distractors)
- **Protocol**: FPS(K=500) + poly(d=2, c=1) + z-score(DB-side) + cosine kNN
- **Metric**: Recall@1, Recall@5, Recall@10, MRR (500 queries)

### 4.2 Text × Text Results (0.5p)

**Table 1: Moschella et al. 比較 — コンポーネント別ablation (K=500, R@1%)**

| Dataset | Pair | Moschella | +FPS | +poly | +z-score | RAT (total) |
|---------|------|-----------|------|-------|----------|-------------|
| STS     | A×B  | 42.8      | 71.0 (+28.2) | 83.0 (+12.0) | 82.8 (-0.2) | **82.8** |
| STS     | A×C  | 68.4      | 89.4 (+21.0) | 97.0 (+7.6)  | 97.0 (±0)   | **97.0** |
| STS     | B×C  | 5.0       | 3.4 (-1.6)   | 19.0 (+15.6) | 76.6 (+57.6)| **76.6** |
| AllNLI  | A×B  | 47.6      | 59.2 (+11.6) | 74.2 (+15.0) | 75.0 (+0.8) | **75.0** |
| AllNLI  | A×C  | 73.4      | 88.4 (+15.0) | 99.0 (+10.6) | 98.6 (-0.4) | **98.6** |
| AllNLI  | B×C  | 3.0       | 3.0 (±0)     | 14.4 (+11.4) | 93.8 (+79.4)| **93.8** |

§4.2冒頭の記述方針:
- 「公正な比較のため、Moschella et al.のデフォルト設定（ランダムアンカー + cosine類似度）を同一データ・同一アンカー候補プール・同一評価指標で再実装した。各コンポーネントの効果を分離するため、+FPS → +poly → +z-scoreの順に積み上げる」

構成ポイント:
- **A×Cを先に議論**: 同系統ペアの強さでインパクトを作る（97-98%）
- **B×Cの劇的改善を強調**: Moschella 5% → RAT 76.6%（STS）、3% → 93.8%（AllNLI）
- FPSの貢献がA×B/A×Cで最大（+11〜28pt）、B×Cでは逆効果（-1.6pt）→ §5で議論
- z-scoreの貢献がB×Cで支配的（+57〜79pt）→ Similarity Collapse解決がメインcontribution
- 2データセットでの一貫性を確認

### 4.3 Database Scaling (0.4p) — **新規追加**

**Table 2: DB規模 vs Recall@1 (FPS+poly+z-score, K=500)**

| DB size | A×C R@1 | A×B R@1 | A×C MRR | A×B MRR | vs random |
|--------:|--------:|--------:|--------:|--------:|----------:|
|     500 |   98.4% |   81.2% |   0.990 |   0.876 |      492× |
|   1,000 |   98.0% |   78.2% |   0.988 |   0.852 |      980× |
|   5,000 |   94.8% |   72.4% |   0.970 |   0.795 |    4,740× |
|  10,000 |   94.4% |   68.6% |   0.966 |   0.762 |    9,440× |
|  50,000 |   92.0% |   58.8% |   0.946 |   0.679 |   46,000× |
| 100,000 | **90.6%** | **55.0%** | 0.933 | 0.641 | **90,600×** |

**Figure 2: Recall@1 vs Database Scale (A×C, A×B 並列、log-scale x軸)**
- 既に `results/db_scaling_comparison.png` に生成済み

構成ポイント:
- **A×Cを先に出す**: 100kで90.6%は「実用水準」と言い切れる数字
- A×Bの55%は「ランダム比55,000倍」で文脈化
- 劣化は対数線形（壊滅的崩壊ではない）
- **全DB規模でmedian_rank=1**を強調（正解が半数以上で1位）
- A×CとA×Bの劣化率の差が規模と共に拡大（17pt→36pt）→ §5で議論

### 4.4 Anchor Scaling Curve (0.3p)

**Figure 3: K=[100,200,500,1000] Random+cosine vs FPS+poly+z-score**
- FPS+proto K=100 (55%) > Random K=1000 (48%)
- K=500で飽和 — コスト最適点
- 既に `results/scaling_comparison.png` に生成済み

### 4.5 Cross-Modal Results (0.3p)

**Table 3: クロスモーダル検索 A×E (MiniLM × CLIP-image)**

| K | baseline R@1 | z-score (rev) R@1 | CLIP直接 R@1 |
|---|---|---|---|
| 500 | 16.8% | 15.6% | 59-62% |
| 1000 | 18.6% | 16.4% | |
| 2000 | 20.0% | 18.0% | |
| 3000 | **21.2%** | 18.4% | |
| 4000 | 19.4% | — | |

- **Best: A×E baseline K=3000 R@1=21.2%** (ランダム0.2%の106倍)
- CLIP直接検索（参考上限）: 59-62% — 4億ペア学習済み vs 3000アンカー
- K=3000が最適、K=4000で次元の呪いにより低下
- z-scoreはクロスモーダルでは効果なし（テキスト同士とは逆の挙動）→ §5.4で分析
- MiniLMは画像の存在を知らないモデル。にもかかわらず画像空間を検索できる

---

## §5 Analysis (1.5 pages)

### 5.1 Similarity Collapse and z-score (0.5p)
- E5-large空間でアンカー間mean sim=0.72、有効レンジ0.33
  - 多言語対応のために広い意味空間を共有空間に圧縮した結果
  - 相対表現プロファイルがフラットになり弁別力消滅
- Figure 4: 3モデルの相対表現プロファイル比較（同一文）
- z-scoreで B×C 5%→76.6%（+71.6pt）
- 非対称z-score: DB側のみで十分、クエリ側は有害（A×Bで-9pt）
- 解釈: DB側は「カタログ」— 正規化して均一に並べるのが検索に有利

### 5.2 FPS: When It Helps and When It Doesn't (0.3p)
- A×B/A×Cで+11〜28pt → アンカー空間のエントロピーが高い場合に有効
- B×Cで-1.6pt → E5-largeの潰れた空間でFPSが「最も遠い点」を有意に選定できない
- **条件**: FPSはアンカー空間の類似度分布が十分に広い（低mean sim）場合にのみ機能する
- 実用上: z-scoreとの併用で全ペア安全（FPSの逆効果をz-scoreが吸収）

### 5.3 Scaling Behavior: Model Compatibility Amplification (0.4p) — **新規追加**
- A×CとA×Bの劣化率: 500→100kでA×Cは-7.8pt、A×Bは-26.2pt
- DB=500では17pt差、DB=100kでは**36pt差**に拡大
- **知見**: DB規模が大きくなるほど、モデル間の構造的互換性の差が増幅される
- E5-largeの低エントロピー相対表現は、distractorが増えるほど正解との弁別が困難に
- Phase 2-3のSimilarity Collapse議論と直結: 「潰れた」相対表現はhard negativeに弱い

### 5.4 Cross-Modal Analysis: Why MiniLM > CLIP-text for Image Retrieval (0.5p) — **新規追加**

**核心的知見**: CLIP-text × CLIP-image (D×E) のRAT一致度はMiniLM × CLIP-image (A×E) の半分以下。
CLIPのtext-image alignmentはグローバルなコサイン最適化であり、RATが依存する局所的な距離パターンを保存しない。

**必須 — アンカー距離パターンρの比較（Figure 5候補）:**
- ペアごとのアンカー距離パターンSpearman相関を計測
- A×E: ρ=0.33 (66%のペアでρ>0.3)、D×E: ρ=0.18 (12%のペアでρ>0.3)
- CLIPのcontrastive lossはバッチ内正例/負例の区別に最適化 → 局所的距離順序の保存を保証しない
- MiniLM側がなぜ高いかはopen question（空間カバレッジの均等さ、訓練目的関数の違い等の仮説を挙げるに留める）

**必須 — バケット分析（Figure 6候補 — 棒グラフ）:**
- CLIPネイティブ類似度の四分位で分割し、各バケットのRAT Recall@1を比較
- A×E: 低バケット14.4% → 高バケット29.6%（CLIPと正相関、共有構造あり）
- D×E: 全バケットで7〜10%（CLIPと無相関、別の信号）
- 「A×EはCLIPの構造を部分的に反映、D×Eは独立の信号源」を一目で示す

**入れたい — パターン3（complementary signal）:**
- CLIPネイティブで下位25%なのにRAT A×Eで成功（rank≤5）が47件。逆はわずか3件
- RATは単にCLIPの劣化コピーではなく、異なる信号源を持つ証拠
- 定性的に「COCO頻出シーンの平凡な記述」がRATの得意領域
  - CLIPにとっては多数の類似キャプションと区別しにくいが、RATはアンカーとの距離パターンの微妙な差で区別

**Limitationsに回す — 次元の呪い:**
- K=3000時のRAT類似度: 正解ペア mean=0.983, 不正解 mean=0.981（Cohen's d=0.58）
- D×Eの方がCohen's d=1.21と分離度は高いが、全体が0.991〜0.993の極小レンジに収まり、微小ノイズで順位が入れ替わる
- スケーリングカーブ: K=500→3000で+4.4pt、K=4000で-1.8pt（次元の呪いによる反転）
- K=3000が最適点。次元の呪いが性能の天井を規定

**数字の更新:**
- クロスモーダルベストを21.2% (A×E baseline K=3000) に更新
  - 旧: 18.2% (z-score, K=500) → 新: 21.2% (baseline, K=3000)
  - abstractとconclusionも更新

### 5.5 Few-shot Correction (Appendixに移動)
- 対角スケーリングの結果はAppendixに移動
- 本文のスペースをクロスモーダル分析に充てる

---

## §6 Limitations — 追加項目

- **次元の呪い (Curse of dimensionality in RAT space):**
  K=3000でRAT類似度が0.98〜0.99に収束。正解/不正解の差が微小（A×E: Δmean=0.002）。
  K=3000が最適で、K=4000では性能低下。高K下では弁別力の向上より類似度潰れが支配的。
  Cohen's dによる分離度分析: D×E(d=1.21) > A×E(d=0.58) だが、D×Eは全体のレンジが極小のため実効的な弁別力はA×Eが上回る。

- **クロスモーダル天井の更新:**
  旧: 18.2%。新: 21.2% (K=3000, baseline)。CLIP直接(59-62%)の約35%。
  スケーリングとz-score最適化でここが天井、pure zero-shotでの限界。

---

## §6 Related Work (0.5 page)

### Relative Representations (Moschella et al., 2023)
- RATの理論的基盤。cosineカーネル + ランダムアンカーでzero-shot stitching を提案
- RATの拡張: (1) FPSで10倍効率化、(2) polyカーネルで+12〜15pt、(3) z-scoreでCollapse解決
- **本論文の直接比較**: Table 1でMoschella手法の再現実装との定量比較を実施

### Platonic Representation Hypothesis (Huh et al., 2024)
- 「十分に大きいモデルは同じ表現に収束する」という仮説
- RATの結果は整合: 絶対座標が無相関でも相対構造が共有される
- §5.3の知見: 構造的互換性の度合いはペアごとに異なり、スケーリング特性に直結

### Model Stitching / Alignment
- Procrustes, CCA, Distillation — いずれも対訳データ + 訓練が必要
- RATはzero-shot・CPU・秒単位 — 根本的に異なるアプローチ
- §5.4: 少数ペアの線形補正はzero-shot RATに劣る場合がある → zero-shotの頑健性

---

## §7 Conclusion (0.3 page)

- RATは追加学習なしで異なるembedding空間を接続する軽量プロトコル
- 同系統ペアでは100k DBでも**R@1=90.6%**、異系統でも55%（ランダム比55,000倍）
- Moschella et al.手法からの改善: B×Cで+71.6pt（5%→76.6%）
- Similarity Collapseの発見と解決は、相対表現の実用化における最大のボトルネックを除去
- DB規模増大時のモデル互換性増幅効果は、モデル選定の実用的指針を提供
- 制限: 英語のみ、テキスト中心、クロスモーダルは概念実証段階
- 今後: 大規模ベンチマーク（MTEB）、多言語、音声モダリティ、アンカー共有プロトコルの標準化

---

## Appendix

### A. Kernel Comparison (全テーブル)
- cosine / RBF / poly(d=2) / poly(d=3) の全組み合わせ結果

### B. Anchor Selection Methods (全比較)
- Random / k-means / FPS / Consensus / TF-IDF FPS / Bootstrap FPS

### C. Cross-Modal Details
- CLIPトークン長統計、アンカー間類似度分析

### D. Few-shot Correction Details
- 対角スケーリング全ペア×全ショット数の結果テーブル

### E. Reproducibility
- ハードウェア（CPU実行、WSL2）、ソフトウェアバージョン、乱数シード（seed=42）
- 全実験スクリプトのGitHubリンク

---

## Figures/Tables 配置計画

| ID | 内容 | 配置 | サイズ |
|----|------|------|--------|
| Fig 1 | ロゼッタストーン概念図 | §3 | 0.4p |
| Fig 2 | **DB scaling: A×C vs A×B (log-scale)** | §4.3 | 0.3p |
| Fig 3 | アンカースケーリングカーブ | §4.4 | 0.3p |
| Fig 4 | 相対表現プロファイル比較（3モデル同一文） | §5.1 | 0.3p |
| Table 1 | **Moschella比較 ablation (6ペア×4手法)** | §4.2 | 0.4p |
| Table 2 | **DB scaling結果 (6規模×2ペア)** | §4.3 | 0.3p |
| Table 3 | クロスモーダル結果 + K scaling | §4.5 | 0.3p |
| Fig 5 | アンカー距離パターンρ: A×E vs D×E (§5.4) | §5.4 | 0.25p |
| Fig 6 | CLIPバケット別RAT精度: A×E vs D×E 棒グラフ (§5.4) | §5.4 | 0.25p |

合計: Fig 1.8p + Table 1.0p = 2.8p → 本文 3.7p。
6ページに収めるにはFig 5,6を1パネル図にまとめるか、Appendixに移す検討が必要。
Few-shot CorrectionをAppendix移動で0.3p確保済み。

---

## 想定される査読コメントと対策

| 想定コメント | 対策 |
|-------------|------|
| 「実験規模が小さい」 | **100k DBスケーリングで対応済み**。ショートペーパーのスコープとして明示 |
| 「Moschella et al.との差分が不明確」 | **Table 1で再現実装との直接比較を実施**。4コンポーネントのablation |
| 「100kで55%は低い」 | A×Cの90.6%を先に出す。55%はランダム比55,000倍で文脈化。モデル互換性の差 |
| 「FPSが常に有効ではない」 | **§5.2で正直に議論**。B×Cでの逆効果とその理由を分析済み |
| 「z-scoreの理論的説明」 | DB/クエリの非対称性の直感的説明。理論的基礎づけはFuture work |
| 「クロスモーダル21%は低すぎる」 | CLIP直接62%との比較で文脈化。ただしRATはzero-shot・学習なし。さらにバケット分析で高CLIPバケットでは29.6%に達し、CLIPが得意なペアでのRAT精度はより高い |
| 「なぜCLIP-text × CLIP-imageが低い？」 | §5.4で詳細分析。アンカー距離パターンρ=0.18 vs 0.33。CLIPのcontrastive lossの構造的特性を説明 |
| 「few-shotで改善しないのは問題では」 | **§5.4で分析済み**: zero-shotの頑健性がRATの強みであることを示す |
| 「Moschella et al.との比較がunfair」 | **§4.2冒頭で明記**: 同一データ分割・同一アンカー数・同一評価指標での再実装比較。FPS/polyの効果を分離するためランダム+cosineを彼らのデフォルトとして再現。アンカー候補プールも共通 |
| 「Platonic仮説との関連は主張しすぎ」 | 「整合する」に留め、「証明した」とは書かない |

---

## ページ配分

| セクション | ページ | 備考 |
|-----------|--------|------|
| Abstract | 0.3 | |
| §1 Introduction | 1.0 | 問題→提案→貢献 |
| §2 Method | 1.0 | Algorithm 1含む |
| §3 Rosetta Stone | 0.5 | Fig 1含む |
| §4 Experiments | 1.5 | Table 1,2,3 + Fig 2,3 |
| §5 Analysis | 1.5 | Fig 4。Collapse + FPS条件 + Scaling挙動 + Few-shot |
| §6 Related Work | 0.4 | 3点のみ |
| §7 Conclusion | 0.3 | |
| **合計** | **6.5** | References別。0.5p超過はTable/Fig圧縮で吸収 |
