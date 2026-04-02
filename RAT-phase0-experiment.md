# Relative Anchor Translation (RAT) - Phase 0 実験手順

## 概要

異なるembeddingモデル間で、共通アンカーポイントとの相対距離だけを使い、追加学習なしにzero-shotで空間変換ができるかを検証する。

## 検証する仮説

「Model Aで埋め込んだテキストを、アンカーとの相対距離表現に変換すれば、Model Bの相対距離空間で最近傍検索して正しい対応文を特定できる」

## 環境

- Python 3.10+
- GPU不要（CPU実行可能なスケール）
- 主要ライブラリ: `sentence-transformers`, `numpy`, `scikit-learn`

## ディレクトリ構成

```
rat-experiment/
├── README.md
├── requirements.txt
├── config.py          # 実験パラメータ一元管理
├── data/
│   └── (実行時に自動生成)
├── src/
│   ├── anchor_sampler.py    # Step 1: アンカー生成
│   ├── embedder.py          # Step 2: embedding取得
│   ├── relative_repr.py     # Step 3: 相対表現変換
│   ├── evaluator.py         # Step 4: 検索精度評価
│   └── visualizer.py        # Step 5: 可視化
├── experiments/
│   └── run_phase0.py        # メイン実行スクリプト
└── results/
    └── (実行時に自動生成)
```

## Step 0: セットアップ

```
pip install sentence-transformers numpy scikit-learn matplotlib datasets
```

## Step 1: データ準備

### アンカーセット（ロゼッタストーン）

- ソース: HuggingFace `datasets` から `wikipedia` の英語記事、または `mteb/stsbenchmark-sts` 等の既存データセット
- 件数: 1000文（まず500で試して十分なら維持、不足なら増やす）
- サンプリング方針: 多様なトピックをカバーするようランダムサンプル
- `data/anchors.json` に保存

### 評価用クエリセット

- アンカーとは**別の**文を500件サンプル
- これが「翻訳対象」になる
- `data/queries.json` に保存

**重要**: アンカーとクエリは重複させないこと。

## Step 2: Embedding取得

### 使用モデル

| モデル | 次元数 | 用途 |
|--------|--------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Model A（軽量英語特化） |
| `intfloat/multilingual-e5-large` | 1024 | Model B（多言語大規模） |

### 処理

1. アンカー1000文を両モデルでembed → `anchor_emb_A`, `anchor_emb_B`
2. クエリ500文を両モデルでembed → `query_emb_A`, `query_emb_B`
3. 全embeddingを `.npy` で保存

```python
# embedder.py の骨格
from sentence_transformers import SentenceTransformer
import numpy as np

def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings
```

**注意**: `multilingual-e5-large` はプレフィクス `"query: "` または `"passage: "` が必要。アンカー・クエリ両方に `"passage: "` を付与して統一する。

## Step 3: 相対表現への変換

各embeddingをアンカーセットとのコサイン類似度ベクトルに変換する。

```python
# relative_repr.py の骨格
import numpy as np

def to_relative(embeddings: np.ndarray, anchor_embeddings: np.ndarray) -> np.ndarray:
    """
    embeddings: (N, D_model) - 元のembedding
    anchor_embeddings: (K, D_model) - アンカーのembedding
    returns: (N, K) - 各アンカーとのコサイン類似度ベクトル

    前提: embeddings, anchor_embeddings ともにL2正規化済み
    """
    # 正規化済みならdot productがコサイン類似度
    return embeddings @ anchor_embeddings.T
```

変換後のデータ:
- `query_rel_A`: (500, 1000) - Model Aのクエリを相対表現化
- `query_rel_B`: (500, 1000) - Model Bのクエリを相対表現化

**ポイント**: 変換後は両方とも同じ1000次元空間にいる。元の384次元 vs 1024次元の差は消えている。

## Step 4: 評価

### メイン評価: Cross-Model Retrieval

`query_rel_A[i]` に対して `query_rel_B` から最近傍を検索し、正解（同じ文のindex `i`）が返るかを測定。

```python
# evaluator.py の骨格
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_retrieval(rel_A: np.ndarray, rel_B: np.ndarray) -> dict:
    """
    rel_A: (N, K) Model Aの相対表現
    rel_B: (N, K) Model Bの相対表現
    rel_A[i] と rel_B[i] は同じ文に対応
    """
    sim_matrix = cosine_similarity(rel_A, rel_B)  # (N, N)
    ranks = []
    for i in range(len(rel_A)):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": int(np.median(ranks)),
    }
```

### ベースライン比較

以下も同時に計測して比較すること：

1. **Same-Model Baseline**: Model A同士で同じ評価 → 相対表現変換自体の情報ロスを測る
2. **Random Baseline**: ランダムなベクトルでの期待Recall@1 = 1/500 = 0.2%
3. **直接cosine（参考）**: 次元が違うので直接比較はできないが、もし次元が同じモデルペアで試す場合は元空間での直接検索も比較

### 成功基準

| 指標 | 最低ライン（有望） | 目標 |
|------|---------------------|------|
| Recall@1 | > 30% | > 60% |
| Recall@10 | > 60% | > 85% |
| MRR | > 0.4 | > 0.7 |

## Step 5: 可視化

1. **相対表現のt-SNE/UMAP**: Model AとModel Bの相対表現を同一プロットに描画。同じ文のペアが近くにあるか確認
2. **類似度行列のヒートマップ**: `sim_matrix` の対角線が明るければ成功
3. **アンカー数の影響**: アンカー数を [50, 100, 200, 500, 1000] で変えてRecall@1をプロット → 最適アンカー数の手がかり

可視化は `results/` に画像で保存。

## Step 6: 追加実験（Phase 0の結果が良好な場合）

### 6a. アンカー数スケーリング
アンカー数を [50, 100, 200, 500, 1000, 2000] で変化させ、精度の飽和点を探す。

### 6b. 3モデル目の追加
- `BAAI/bge-small-en-v1.5` (384次元) を追加し、3モデル間でのクロス検索を検証
- 同じアンカープロファイルで3モデル全ペアが動くか

### 6c. 日本語テスト
- アンカー・クエリを日本語に変更
- `intfloat/multilingual-e5-large` は日本語対応済み
- `all-MiniLM-L6-v2` は英語特化なので、代わりに `pkshatech/GLuCoSE-base-ja` 等を使用

## 実行方法

```bash
cd rat-experiment
python experiments/run_phase0.py
```

`run_phase0.py` は Step 1〜5 を順番に実行し、結果をコンソールとファイルに出力する。

## 出力物

- `results/metrics.json` - 全評価指標
- `results/sim_matrix.png` - 類似度行列ヒートマップ
- `results/tsne_plot.png` - t-SNE可視化
- `results/anchor_scaling.png` - アンカー数 vs 精度のグラフ
- `results/experiment_log.txt` - 実行ログ（モデル名、アンカー数、実行時間等）

## 判断基準

- **Recall@1 > 30%**: 仮説に根拠あり → Phase 1（アンカー最適化）に進む
- **Recall@1 10-30%**: 可能性はあるが改善が必要 → アンカー選定やカーネル関数の変更を検討
- **Recall@1 < 10%**: コサイン類似度ベースの相対表現だけでは不十分 → 軽量な線形変換の追加を検討（それでもダメなら仮説の修正が必要）
