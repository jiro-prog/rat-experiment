# D2: Scale Series & Cross-Cluster RAT Analysis

**実施日**: 2026-04-04
**実施者**: sojir
**ステータス**: 完了。論文v4へのストック（A方針: 追加実験を積んでから統合）

## 実験の問い

「モデルのスケールアップはRDM相関を高め、クラスター間RATを改善するか？」

- **Q1**: Arctic系列(22M→335M)でBERTクラスターとのRDM相関は変化するか？
- **Q2**: RDM相関はRAT精度の独立した予測因子か？
- **Q3**: K交差点（RAT→線形手法の優位が逆転するK値）はRDM相関の関数か？

## モデルセット（17モデル）

| ID | モデル | Params | Dim | sim_mean | Cluster | 備考 |
|----|--------|--------|-----|----------|---------|------|
| A | all-MiniLM-L6-v2 | 22M | 384 | 0.020 | BERT | |
| B | multilingual-e5-large | 560M | 1024 | 0.705 | BERT | |
| C | bge-small-en-v1.5 | 33M | 384 | 0.379 | BERT | |
| D | clip-ViT-B-32 | 63M | 512 | 0.426 | BERT | |
| E | bge-large-en-v1.5 | 335M | 1024 | — | BERT | |
| F | e5-small-v2 | 33M | 384 | — | BERT | |
| G | multilingual-e5-small | 118M | 384 | 0.722 | BERT | |
| H | gte-small | 33M | 384 | — | BERT | |
| I | gte-large | 335M | 1024 | — | BERT | |
| J | all-mpnet-base-v2 | 109M | 768 | 0.016 | BERT | |
| K | bge-base-en-v1.5 | 109M | 768 | — | BERT | |
| L | nomic-embed-text-v1.5 | 137M | 768 | — | 新世代 | |
| M | gte-Qwen2-1.5B-instruct | 1.5B | 1536 | 0.166 | 新世代 | decoder系 |
| **N** | **snowflake-arctic-embed-m** | **109M** | **768** | **0.623** | **Arctic** | 既存 |
| **O** | **snowflake-arctic-embed-xs** | **22M** | **384** | **0.648** | **Arctic** | D2新規 |
| **P** | **snowflake-arctic-embed-s** | **33M** | **384** | **0.615** | **Arctic** | D2新規 |
| **Q** | **snowflake-arctic-embed-l** | **335M** | **1024** | **0.610** | **Arctic** | D2新規 |

Arctic系列: 全サイズCLS pooling + L2正規化で一貫（Phase 0で検証済み）。

**注意**: sim_mean値はseed=314のFPSアンカー(K=500)間で計算。v3論文のsim_mean（seed=42ベース）とは微妙に異なる（例: A=0.020 vs v3の0.028）。E,F,H,I,Kは本表では省略、Phase 1結果JSONに全値あり。D(CLIP)のクラスター帰属はv3では独立クラスターとして議論されており、ここでの「BERT」は便宜的分類。v4統合時に再検討のこと。
STS sanity check: 全モデルPASS (ρ ≥ 0.65)。xs=0.778, s=0.794, m=0.749, l=0.742。

## 結果

### Q1: スケールとRDM相関 → 変化なし

Arctic全サイズとBERT代表モデルのRDM相関:

| BERT \ Arctic | O (xs, 22M) | P (s, 33M) | N (m, 109M) | Q (l, 335M) | range |
|---------------|-------------|------------|-------------|-------------|-------|
| A (MiniLM) | 0.006 | 0.001 | -0.007 | -0.006 | 0.013 |
| C (BGE-s) | -0.006 | -0.017 | -0.008 | -0.012 | 0.011 |
| J (MPNet) | 0.018 | 0.007 | 0.009 | 0.010 | 0.011 |
| B (E5-large) | 0.005 | 0.003 | 0.002 | -0.012 | 0.017 |
| E (BGE-large) | -0.002 | 0.000 | -0.022 | -0.029 | 0.030 |

- **Δρ(Arctic-l vs Arctic-xs, 対A) = -0.012** （閾値0.05以下: 変化なし）
- **Δρ(s vs xs, 同次元384d, 対A) = -0.005** （パラメータ1.5倍でもゼロ）
- 全BERT×Arctic組み合わせで ρ ≈ 0 (±0.03)

**結論**: 訓練レシピが埋め込み空間のトポロジーを決定し、パラメータ数15倍(22M→335M)でも越えられない。PRHの「スケールが収束を駆動する」予測に対する部分的反証（ただし335Mまでの範囲に限定）。

Arctic系列内のRDM相関:

| ペア | RDM ρ | RAT R@1 (K=500) |
|------|-------|-----------------|
| N↔Q (m↔l) | 0.872 | 82%/61% |
| O↔P (xs↔s, 同次元384d) | 0.777 | 91%/94% |
| O↔N (xs↔m) | 0.690 | 67%/54% |
| O↔Q (xs↔l) | 0.634 | 51%/38% |
| P↔N (s↔m) | 0.565 | 65%/41% |
| P↔Q (s↔l) | 0.500 | 54%/38% |

### Q2: 二段階診断モデル → RDMは独立予測因子（ただしレバレッジ効果あり）

階層的回帰（R@1 ~ 予測因子、K=500):

| サブセット | N | R²(sim_mean) | R²(RDM) | R²(両方) | ΔR²(RDM after sim) |
|-----------|---|-------------|---------|---------|-------------------|
| **全ペア** | 176 | 0.093 | 0.568 | 0.692 | **0.599** |
| **BERT内** | 110 | **0.253** | **0.256** | **0.571** | 0.318 |
| **Arctic内** | 12 | 0.105 | 0.455 | 0.504 | 0.399 |
| クロスクラスター | 32 | 0.006 | 0.030 | 0.031 | 0.025 |

偏相関（RDM vs R@1 | sim_mean）: ρ = 0.802, p < 10⁻⁴⁰ （全ペア）

**重要な解釈**: 全ペアでΔR²=0.599は巨大だが、これはクロスクラスターペア(R@1≈0%, RDM≈0)のレバレッジ効果。**BERT内ではsim_meanとRDMがほぼ同等の説明力**(各R²≈0.25)。正しい解釈は二段階モデル:

```
Layer 1: RDM相関（ゲートキーパー）
  ├── ρ ≈ 0 → 翻訳不可能（R@1 ≈ 0%、手法に関わらず）
  └── ρ > 0  → 翻訳可能 → Layer 2 へ

Layer 2: sim_mean（精度レンジの決定）+ RDM（相補的）
  └── クラスター内ペアの精度を予測（合わせてR²=0.57）

Layer 3: 構造的要因（残差の説明）
  ├── same_family → +9pp
  ├── 次元一致 → 方向非対称性を抑制
  └── RDM → クラスター内での微調整
```

**結論**: RDMはゲートキーパーとして翻訳可否を決定し、クラスター内ではsim_meanと相補的に精度を予測する。「sim_meanが冗長でRDMが全て」は誤り。

### Q3: K交差点 → RDMに非依存

| 指標 | Spearman ρ | p値 |
|------|-----------|-----|
| 交差点K vs RDM | 0.136 | 0.590 |
| 交差点K vs sim_mean | -0.458 | 0.056 |

代表的なK交差点（RAT→Best Linearの逆転K、クラスター内ペアのみ）:

| ペア | RDM ρ | sim_mean | 交差K |
|------|-------|----------|-------|
| O↔P (同次元Arctic) | 0.777 | 0.63-0.65 | ~350 |
| N↔Q | 0.872 | 0.61-0.62 | 75-150 |
| A↔C | 0.552 | 0.02-0.38 | ~150 |
| A→B | 0.446 | 0.02 | ~350 |
| B→A | 0.446 | 0.71 | <10 |

**結論**: K交差点はsim_meanに弱く依存し、RDMには依存しない。RATの有効K範囲はsim_mean（空間の圧縮度）で制御され、RDMは「翻訳可能かどうか」を決めるが「RAT vs 線形手法の優劣」は決めない。

### 副次的知見

- **同次元・異クラスター**: C(BGE,384d) ↔ O(Arctic-xs,384d) = R@1 0.4%。同次元でも異クラスターなら翻訳不可能
- **方向非対称性の次元効果**: 同次元|Δ|=2.6pp vs 異次元|Δ|=18.0pp。Arctic系列のsim_mean均一性(0.610-0.648)によりsim_mean効果を除去した条件で分離
- **K=500 RAT勝率**: クラスター内0/18。BERT 0/6, Arctic 0/12。D1結論（K≥200で線形手法優位）の完全再現
- **Arctic sim_mean均一性**: 全サイズが0.610-0.648の狭い範囲。スケール系列実験でsim_meanが交絡因子にならない条件を自然に提供
- **D1再現性**: A↔C K=500 RAT: D2 seeds(314,999,2025)で95.2-96.8%。D1 seed=42での値と±1%の安定性

## バグ・修正履歴

### 分析Cバグ（2026-04-04 発見・修正）

**症状**: Arctic内ペアのRidge R@1が0.6-1.8%と報告され、「Arctic空間は非線形性が強い」と誤認。

**原因**: Q3分析スクリプト内で、Phase 1結果（RAT onlyの68ペア、K=500固定）とPhase 2結果（RAT+Ridge+Procrustes、44ペア×6K×3seeds）のJSONを統合して分析した際、Phase 1にはRidgeレコードが存在しないペアに対してデフォルト値（0）が使用された。Phase 2のJSONデータ自体は正しかった（P→O Ridge=88.6%）。

**検証方法**: パイプライン外の簡易テスト（sklearn.linear_model.Ridgeで直接計算）により、P→O Ridge α=0.01 = 88.6%を確認。Phase 2のprint出力（Q→P Ridge=88.0%）とも整合。

**修正**: Q3分析スクリプト(`run_d2_q3_analysis.py`)でPhase 2のJSONのみを使用するよう修正。Phase 1の結果はRAT基礎評価としてのみ参照。

**教訓**: 異なるPhaseの結果JSONを統合する際は、メソッド列の有無を明示的にチェックすべき。Check 4（5分の簡易テスト）で即座にバグを検出できた — 「理論的に不自然な結果は検証する」プロセスの重要性。

## ファイル対応表

```
experiments/
  run_d2_phase0_sanity.py     → results/d2_scale/d2_phase0_results.json
                                 (pooling検証 + STS sanity check)

  run_d2_phase1_basic.py      → results/d2_scale/d2_phase1_results.json
                                 data/d2_matrix/cand_{O,P,Q}.npy
                                 data/d2_matrix/query_{O,P,Q}.npy
                                 (埋め込み生成 + 90ペアRAT基礎評価)

  run_d2_phase2_comparison.py → results/d2_scale/d2_phase2_results.json
                                 results/d2_scale/d2_phase2_results.csv
                                 (50ペア × 6K × 3seeds、RAT/Ridge/Procrustes/Affine)

  run_d2_phase3_rdm.py        → results/d2_scale/d2_rdm_correlation.json
                                 results/d2_scale/d2_phase3_results.json
                                 results/d2_scale/fig_rdm_heatmap.png
                                 results/d2_scale/fig_rdm_vs_r1.png
                                 results/d2_scale/fig_scale_axis.png
                                 (17×17 RDM行列 + 階層的回帰 + 可視化)

  run_d2_q3_analysis.py       → results/d2_scale/d2_q3_analysis.json
                                 results/d2_scale/fig_q3_crossover.png
                                 (K交差点分析 + 分析D/E)

config.py                      → MODEL O,P,Q の追加（既存A-Nに影響なし）
```

## 論文への統合方針

**方針A（採用）**: v3は凍結。D2はストックし、追加実験を積んでからv4として統合。

統合時の構成案:
- §4.6 → "Cluster Boundaries and Scale Invariance" （Q1）
  - §4.6.1: クラスター発見（既存）
  - §4.6.2: スケール不変性（D2新規）
- §5.3 → "Two-Layer Diagnostic Model" （Q2 + Q3）
  - RDMゲートキーパー + sim_mean精度予測
  - K交差点の否定的結果
- §5.4 → "Direction Asymmetry and Dimensionality" （分析E）

v4のタイミング目安: D2に加えてもう1つの実験的貢献（カーネル改良/MTEB規模/別ファミリー系列）が積めたとき。

note記事4本目はv4とは独立にカジュアル発信として出してよい。

## 再現手順

```bash
cd ~/projects/rat-experiment
source ~/rat-venv/bin/activate

# Phase 0: Pooling検証 + STS sanity check (約1分)
python experiments/run_d2_phase0_sanity.py

# Phase 1: 埋め込み生成 + 基礎評価 (約30秒、GPUあり)
# ※ cand_{O,P,Q}.npy が既に存在すればスキップされる
python experiments/run_d2_phase1_basic.py

# Phase 2: RAT vs Ridge vs Procrustes (約13分)
# ※ seeds = [314, 999, 2025]（D1の[42,123,7]とは異なる）
python experiments/run_d2_phase2_comparison.py

# Phase 3: RDM分析 + 階層的回帰 (約5秒)
python experiments/run_d2_phase3_rdm.py

# Q3 + 分析D/E (約1秒)
python experiments/run_d2_q3_analysis.py
```

注意: Phase 2のstderr に Ridge の `LinAlgWarning` が大量に出るが正常動作。K>dimのケースで条件数が悪化するため。
