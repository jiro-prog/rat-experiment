# Relative Anchor Translation (RAT)

異なるembeddingモデル間で、共通アンカーポイントとの相対距離だけを使い、追加学習なしにzero-shotで空間変換ができるかを検証する実験。

## 仮説

Model Aで埋め込んだテキストを、アンカーとのコサイン類似度ベクトル（相対表現）に変換すれば、Model Bの相対表現空間で最近傍検索して正しい対応文を特定できる。

## 仕組み

```
テキスト → Model A (384d) → アンカーとのcos sim → 相対表現 (K次元)
テキスト → Model B (1024d) → アンカーとのcos sim → 相対表現 (K次元)
                                                     ↑ 同じ空間
```

元の次元数が異なっていても、共通のK個のアンカーとの類似度プロファイルに変換することで、同一空間での比較が可能になる。

## Phase 0 結果

| 実験 | Recall@1 | Recall@10 | MRR |
|------|----------|-----------|-----|
| Cross-Model A→B (K=500) | 42.2% | 81.6% | 0.558 |
| Cross-Model A→B (K=1000) | 43.4% | 81.2% | 0.565 |
| Random Baseline | 0.2% | - | - |

| 近傍構造保存率 | Overlap@10 |
|---------------|------------|
| Model A (K=500) | 61.1% |
| Model A (K=1000) | 62.7% |
| Model B (K=500) | 50.5% |
| Model B (K=1000) | 50.8% |

**判定: 仮説は成立。** ランダムアンカー500個でRecall@1=42.2%（ランダムベースライン0.2%の200倍以上）。スケーリングカーブはK=500付近で飽和傾向を示しており、アンカー数増加よりもアンカー選定の最適化が次のレバーとなる。

### 類似度行列

![Similarity Matrix](results/sim_matrix.png)

対角線上の明るいバンドが、Cross-Modelでの正解ペアの類似度が高いことを示している。

### アンカー数スケーリング

![Anchor Scaling](results/anchor_scaling.png)

K=50→500で急速に改善するが、K=500→1000で飽和が始まっている。

### t-SNE可視化

![t-SNE](results/tsne_plot.png)

Model A（青）とModel B（赤）の相対表現。同じ文のペアを灰色の線で接続。

## 使用モデル

| モデル | 次元数 | 役割 |
|--------|--------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Model A（軽量英語特化） |
| `intfloat/multilingual-e5-large` | 1024 | Model B（多言語大規模） |

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_phase0.py
```

結果は `results/` に出力される。データは `data/` に自動生成される（gitでは追跡しない）。

## ディレクトリ構成

```
rat-experiment/
├── config.py                 # 実験パラメータ一元管理
├── src/
│   ├── anchor_sampler.py     # STSBenchmarkからアンカー・クエリをサンプリング
│   ├── embedder.py           # 2モデルでembedding取得
│   ├── relative_repr.py      # 相対表現への変換
│   ├── evaluator.py          # Recall@K, MRR, Overlap@10評価
│   └── visualizer.py         # ヒートマップ, t-SNE, スケーリングカーブ
├── experiments/
│   └── run_phase0.py         # メイン実行スクリプト
├── data/                     # 実行時に自動生成（git非追跡）
└── results/                  # 評価結果・可視化
```

## License

MIT
