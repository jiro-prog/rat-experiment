# RAT: Embedding Space Translation via Relative Anchor Similarity Profiles

**Format**: Short paper (6 pages + references + appendix), arXiv preprint
**Target venues**: EMNLP Findings, ACL SRW, or standalone arXiv

---

## Abstract (15 lines)

- 問題: 異なるembeddingモデルの出力は直接比較できない。学習ベースのアラインメントは対訳データと訓練コストが必要
- 提案: RAT (Relative Anchor Translation) — 共通アンカーとの類似度プロファイルに変換するだけで、追加学習なしにzero-shotで空間変換
- プロトコル: FPS(K=500) + poly kernel + z-score(DB側)
- 結果サマリ:
  - 5モデル全ペアでRecall@1 30%超（最大98%）、ランダムベースライン0.2%
  - FPS+proto K=100 > Random K=1000（10倍効率）
  - テキスト専用モデルからCLIP画像空間へのzero-shotクロスモーダル検索（R@1=18%、学習なし）
  - 直接cos類似度≈0の空間ペアでもRAT 55%（相対構造の共有を実証）
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
1. **RAT Protocol v0.1**: FPS + poly + z-score(DB側) の3ステップで、5モデル全ペアにzero-shot対応
2. **Similarity Collapse の発見と解決**: 多言語モデルの類似度潰れを同定し、z-scoreで+50pt改善
3. **クロスモーダル拡張**: ロゼッタストーンアンカーによる学習なしテキスト×画像検索
4. **空間構造の共有**: 直接cos≈0でも相対プロファイルが一致する実験的証拠

---

## §2 Method (1 page)

### 2.1 Relative Anchor Representation (0.4p)
- 定義: r(x) = [k(x, a_1), k(x, a_2), ..., k(x, a_K)]
- 任意のembedding空間の点をK次元ベクトルに変換
- 異なるモデルの出力が同一のK次元空間に落ちる
- カーネル選択: 多項式 k(x,a) = (x·a + 1)² — cosineより+15pt（非線形性が上位の弁別力を向上）

### 2.2 Farthest Point Sampling (0.3p)
- ランダムアンカーは密集バイアスを持つ（高密度領域に偏る）
- FPS: 既選択アンカーから最も遠い点を貪欲に追加
- 空間被覆の最大化 → アンカー100個で、ランダム1000個を超える（C-2）
- 計算量: O(K*N) — CPUで数秒

### 2.3 z-score Normalization (0.3p)
- 類似度潰れ（Similarity Collapse）への対策
- DB側の相対表現を行ごとに平均0、分散1に正規化
- 「潰れていれば引き伸ばし、広がっていればほぼそのまま」— 安全な片方向変換
- C-3の知見: DB側のみで十分、クエリ側は不要（むしろ有害）

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
- **Models**: 5 text encoders (MiniLM 384d, E5-large 1024d, BGE-small 384d, CLIP-text 512d, BGE-large 1024d) + CLIP image encoder (ViT-B/32 512d)
- **Data**: STSBenchmark (15K sentences), AllNLI (26K sentences), COCO Karpathy (5K image-caption pairs)
- **Protocol**: FPS(K=500) + poly(d=2, c=1) + z-score(DB-side) + cosine kNN
- **Metric**: Recall@1, Recall@10, MRR (500 queries, random baseline = 0.2%)

### 4.2 Text x Text Results (0.4p)
- Table 1: 5モデル全ペア結果（STS + AllNLI）
  - 全ペアR@1 > 30%、最大98%（A×C）
  - AllNLIで再現確認、B×Cは+20%改善（データ多様性の恩恵）
- Ablation: 各コンポーネントの寄与
  - Random->FPS: +23pt, cosine->poly: +15pt, none->z-score: +50pt (B×C)

### 4.3 Scaling Curve (0.3p)
- Figure 2: K=[100,200,500,1000]でのRandom+cosine vs FPS+proto
- **FPS+proto K=100 (55%) > Random K=1000 (48%)**
- K=500で飽和 — コスト最適点

### 4.4 Cross-Modal Results (0.5p)
- Table 2: Text->Image / Image->Text の全パターン
  - Best: Image->Text R@1=18.2% (z-score), Text->Image R@1=16.4% (baseline)
  - CLIP直接検索（参考上限）: 62% — 4億ペア学習済み vs 500アンカー
- MiniLMは画像の存在を知らないモデル。にもかかわらず画像空間を検索できる

---

## §5 Analysis (1.5 pages)

§4と同じ分量を割く。「なぜ動くか、どこで壊れるか、どう直すか」がRATの価値の半分。

### 5.1 Similarity Collapse (0.5p)
- E5-large空間でアンカー間mean sim=0.72、有効レンジ0.33
  - 多言語対応のために広い意味空間を共有空間に圧縮した結果
  - 相対表現プロファイルがフラットになり弁別力消滅
- Figure 3: 3モデルの相対表現プロファイル比較（同一文）
- z-score正規化で解決: 14%->64%（+50pt）
- Table 3: 4つの正規化手法比較 — z-scoreだけが全ペアに安全
  - ランク変換: 52%（類似度の絶対差を捨てる）
  - Top-k: 50%（潰れていないモデルの情報を破壊）
  - softmax: 50%（同上）

### 5.2 Asymmetric z-score: DB-side Only (0.5p)
- Table 4: none / query_only / db_only / both の4パターン × 全ペア + クロスモーダル
- **全ペアで db_only = both**（クエリ側は不要）
- クエリ側z-scoreは有害: A×Bで-9pt、A×Cで-8pt
- 解釈: DB側は「カタログ」— 正規化して均一に並べるのが検索に有利。クエリは「問い合わせ」— 元の分布のまま投げるほうが正確
- エントロピー閾値ルールは方向依存を扱えず不適切 → 「常にDB側のみ」が最も堅牢

### 5.3 Dimensions Don't Matter (0.5p)
- B×E: 同じ1024次元、直接cos≈0（完全に無相関な空間）→ RAT 55%
- A×B: 異次元（384 vs 1024）→ RAT 76%
- **同次元55% < 異次元76%**: 次元の一致/不一致はRAT性能に無関係
- RATが依存するのは「アンカーとの相対距離パターンの構造的類似性」
- Platonic Representation Hypothesis (Huh et al., 2024) の実験的裏付け: 異なるモデルが同じタスクを解くと、表面的に無相関でも深層の構造が収束する

---

## §6 Related Work (0.5 page)

0.5ページに絞る。3点のみ。

### Relative Representations (Moschella et al., 2023)
- RATの理論的基盤。cosineカーネル + ランダムアンカーでzero-shot stitching を提案
- RATの拡張: (1) FPSで10倍効率化、(2) polyカーネルで+15pt、(3) z-scoreで潰れ解決、(4) クロスモーダル拡張
- Moschella et al. では触れられていない failure mode（Similarity Collapse）を発見・解決

### Platonic Representation Hypothesis (Huh et al., 2024)
- 「十分に大きいモデルは同じ表現に収束する」という仮説
- RATの結果はこの仮説と整合: 絶対座標が無相関でも相対構造が共有される
- ただしRATは「収束の度合い」のモデルペアごとの差も定量化している（B×C崩壊はその反例）

### Model Stitching / Alignment
- Procrustes, CCA, Distillation — いずれも対訳データと訓練が必要
- RATはzero-shot・CPU・秒単位 — 根本的に異なるアプローチ
- トレードオフ: RATは高精度な1:1マッピングより「検索可能にする」ことに特化

---

## §7 Conclusion (0.3 page)

- RATは追加学習なしで異なるembedding空間を接続する軽量プロトコル
- 3つのシンプルなステップ（FPS + poly + z-score）で、5モデル間のzero-shot検索を実現
- Similarity Collapseの発見と解決、非対称z-scoreの知見はRATを超えて汎用的
- ロゼッタストーンアンカーによるクロスモーダル拡張はproof of conceptだが、モダリティの壁が原理的に越えられることを示した
- 制限: 英語のみ、小規模データ、画像側FPS未最適化
- 今後: 大規模ベンチマーク（MTEB）、多言語、音声モダリティ、アンカー共有プロトコルの標準化

---

## Appendix

本文には結論だけ書いて、詳細はここに逃がす。6ページに収めるための必須判断。

### A. Kernel Comparison (全テーブル)
- cosine / RBF / poly(d=2) / poly(d=3) の全組み合わせ結果
- RBFのgammaチューニング結果

### B. Anchor Selection Methods (全比較)
- Random / k-means / FPS / Consensus / TF-IDF FPS / Bootstrap FPS
- FPS基準モデルの変更実験（A基準 vs B基準 vs C基準）

### C. Phase 4 Step 1: CLIP Text Encoder Details
- CLIPトークン長統計
- 全4モデルのアンカー間類似度分析
- CLIPテキストエンコーダのentropy=2.60が高いのにRecallが低い分析

### D. Reproducibility
- ハードウェア（CPU実行、WSL2）
- ソフトウェアバージョン（requirements.txt）
- 乱数シード（seed=42全実験統一）
- 全実験スクリプトのGitHubリンク

---

## Figures/Tables 配置計画

| ID | 内容 | 配置 | サイズ |
|----|------|------|--------|
| Fig 1 | ロゼッタストーン概念図（テキスト×テキスト vs クロスモーダル） | §3 | 0.4p |
| Fig 2 | スケーリングカーブ（Random vs FPS+proto） | §4.3 | 0.3p |
| Fig 3 | 相対表現プロファイル比較（3モデル同一文） | §5.1 | 0.3p |
| Table 1 | テキスト×テキスト全ペア結果（STS + AllNLI） | §4.2 | 0.3p |
| Table 2 | クロスモーダル結果 | §4.4 | 0.2p |
| Table 3 | 正規化手法比較（z-score / rank / topk / softmax） | §5.1 | 0.2p |
| Table 4 | 非対称z-score 4パターン × 全ペア | §5.2 | 0.3p |

合計: Fig 1.0p + Table 1.3p = 2.3p → 本文 3.7p で6ページに収まる

---

## 想定される査読コメントと対策

| 想定コメント | 対策 |
|-------------|------|
| 「実験規模が小さい（5モデル、500クエリ）」 | ショートペーパーのスコープとして明示。Future workでMTEB規模の検証を約束 |
| 「Moschella et al.との差分が不明確」 | §6で4点の具体的拡張を明示。Similarity Collapse発見はNovelty |
| 「FPSの理論的保証は？」 | 実験的に十分。理論的にはepsilon-net argumentが使えるが、ショートペーパーでは実験で示す |
| 「z-scoreがなぜDB側だけで効くのか理論的説明を」 | §5.2で「カタログ vs 問い合わせ」の直感的説明 + 情報理論的な議論はFuture work |
| 「クロスモーダル18%は低すぎる」 | CLIP直接62%（4億ペア学習）との比較で文脈化。500アンカーのzero-shotとしては非自明 |
| 「Platonic仮説との関連は主張しすぎ」 | 「整合する」「実験的裏付け」に留め、「証明した」とは書かない |

---

## ページ配分

| セクション | ページ | 備考 |
|-----------|--------|------|
| Abstract | 0.3 | |
| §1 Introduction | 1.0 | 問題→提案→貢献 |
| §2 Method | 1.0 | Algorithm 1含む |
| §3 Rosetta Stone | 0.5 | Fig 1含む |
| §4 Experiments | 1.5 | Table 1,2 + Fig 2 |
| §5 Analysis | 1.5 | Table 3,4 + Fig 3。§4と同分量 |
| §6 Related Work | 0.5 | 3点のみ |
| §7 Conclusion | 0.3 | |
| **合計** | **6.6** | References別。0.6p超過はTable圧縮で吸収 |
