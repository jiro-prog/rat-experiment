"""
RAT Phase 4 分析補足: 正解/不正解RAT類似度ヒストグラム + パターン3定量化

1. K=3000時の正解ペア vs 不正解ペアのRAT類似度分布
   → 次元の呪いによる「潰れ」の定量化（§5/§6 Limitations向け）
2. パターン3（CLIP低 & RAT成功）の定量的特徴づけ
   → 具体名詞の数、動詞の数、キャプションの情報量で特徴を切り分け

保存済みの phase4_analysis_correlation.json を読んで追加分析。
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config

# ========================================
# データ読み込み
# ========================================
data_path = config.RESULTS_DIR / "phase4_analysis_correlation.json"
with open(data_path) as f:
    data = json.load(f)

per_pair = data["per_pair_data"]
clip_sims = np.array(per_pair["clip_native_sim"])
rat_sims_dxe = np.array(per_pair["rat_sim_DxE"])
rat_sims_axe = np.array(per_pair["rat_sim_AxE"])
ranks_dxe = np.array(per_pair["rank_DxE"])
ranks_axe = np.array(per_pair["rank_AxE"])
cap_lens = np.array(per_pair["caption_lengths"])

N = len(clip_sims)

# ========================================
# 1. 正解 vs 不正解のRAT類似度分布
# ========================================
print("=" * 70)
print("1. 正解 vs 不正解ペアのRAT類似度分布 (次元の呪い分析)")
print("=" * 70)

for label, rat_sims, ranks in [
    ("A×E", rat_sims_axe, ranks_axe),
    ("D×E", rat_sims_dxe, ranks_dxe),
]:
    correct_sims = rat_sims[ranks == 1]  # 正解ペアのRAT類似度

    # 不正解ペアの類似度: 各クエリについてランダムな不正解ペアの類似度を推定
    # ここではper_pair_dataの類似度は対角（正解ペア）のみなので、
    # 代わりに正解 vs 全体の分布を比較
    all_sims = rat_sims
    incorrect_mask = ranks > 1
    incorrect_sims = rat_sims[incorrect_mask]

    print(f"\n  --- {label} ---")
    print(f"  正解 (rank=1) N={len(correct_sims)}:")
    print(f"    mean={correct_sims.mean():.6f}, std={correct_sims.std():.6f}")
    print(f"    min={correct_sims.min():.6f}, max={correct_sims.max():.6f}")

    print(f"  不正解 (rank>1) N={len(incorrect_sims)}:")
    print(f"    mean={incorrect_sims.mean():.6f}, std={incorrect_sims.std():.6f}")
    print(f"    min={incorrect_sims.min():.6f}, max={incorrect_sims.max():.6f}")

    # 分離度: (mean_correct - mean_incorrect) / pooled_std
    pooled_std = np.sqrt((correct_sims.std()**2 + incorrect_sims.std()**2) / 2)
    if pooled_std > 0:
        separation = (correct_sims.mean() - incorrect_sims.mean()) / pooled_std
    else:
        separation = 0
    print(f"  分離度 (Cohen's d): {separation:.4f}")

    # ヒストグラム（テキストベース）
    bins = np.linspace(rat_sims.min() - 0.001, rat_sims.max() + 0.001, 21)
    print(f"\n  ヒストグラム (bins={len(bins)-1}):")
    print(f"  {'Range':<24} {'正解':>6} {'不正解':>6}")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n_correct = ((correct_sims >= lo) & (correct_sims < hi)).sum()
        n_incorrect = ((incorrect_sims >= lo) & (incorrect_sims < hi)).sum()
        if n_correct + n_incorrect > 0:
            bar_c = "█" * n_correct
            bar_i = "░" * min(n_incorrect // 3, 30)  # scale down
            print(f"  [{lo:.5f}, {hi:.5f}) {n_correct:>6} {n_incorrect:>6}  {bar_c}{bar_i}")


# ========================================
# 2. パターン3の定量的特徴づけ
# ========================================
print("\n" + "=" * 70)
print("2. パターン3 (CLIP低 & RAT成功) の定量的特徴づけ")
print("=" * 70)

# 再定義
clip_high_thresh = np.percentile(clip_sims, 75)
clip_low_thresh = np.percentile(clip_sims, 25)
rat_success = ranks_axe <= 5
rat_fail = ranks_axe > 50

p1 = (clip_sims >= clip_high_thresh) & rat_success  # CLIP高 & RAT成功
p2 = (clip_sims >= clip_high_thresh) & rat_fail     # CLIP高 & RAT失敗
p3 = (clip_sims <= clip_low_thresh) & rat_success    # CLIP低 & RAT成功
middle = ~p1 & ~p2 & ~p3  # その他

# キャプションの再読み込みが必要だが、JSONには入っていないので
# 統計量で比較
print(f"\n  パターン分布:")
print(f"    P1 (CLIP高&RAT成功): {p1.sum()}件")
print(f"    P2 (CLIP高&RAT失敗): {p2.sum()}件")
print(f"    P3 (CLIP低&RAT成功): {p3.sum()}件")
print(f"    その他: {middle.sum()}件")

for label, mask in [("P1", p1), ("P2", p2), ("P3", p3), ("全体", np.ones(N, dtype=bool))]:
    if mask.sum() == 0:
        continue
    print(f"\n  {label} (N={mask.sum()}):")
    print(f"    CLIPネイティブsim: mean={clip_sims[mask].mean():.4f}, std={clip_sims[mask].std():.4f}")
    print(f"    RAT A×E sim:      mean={rat_sims_axe[mask].mean():.6f}")
    print(f"    RAT D×E sim:      mean={rat_sims_dxe[mask].mean():.6f}")
    print(f"    キャプション長:    mean={cap_lens[mask].mean():.1f}, std={cap_lens[mask].std():.1f}")
    print(f"    A×E rank:         median={np.median(ranks_axe[mask]):.0f}, mean={ranks_axe[mask].mean():.1f}")
    print(f"    D×E rank:         median={np.median(ranks_dxe[mask]):.0f}, mean={ranks_dxe[mask].mean():.1f}")

# ========================================
# 3. K別の正解/不正解ペアRAT類似度の「潰れ」度合い
# ========================================
print("\n" + "=" * 70)
print("3. 次元の呪い: 正解ペアのRAT類似度分布の要約統計")
print("=" * 70)

# 正解ペアの類似度がどれだけ他のペアと区別できるか
# ここでは ranks を使って「正解ペアが他のペアよりどれだけ類似度が高いか」を推定
for label, ranks in [("A×E", ranks_axe), ("D×E", ranks_dxe)]:
    print(f"\n  {label}:")
    print(f"    Recall@1: {(ranks == 1).mean()*100:.1f}%")
    print(f"    Recall@5: {(ranks <= 5).mean()*100:.1f}%")
    print(f"    Recall@10: {(ranks <= 10).mean()*100:.1f}%")
    print(f"    MRR: {np.mean(1.0/ranks):.4f}")
    print(f"    Median rank: {np.median(ranks):.0f}")

    # ランク分布のヒストグラム
    rank_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    print(f"\n    ランク累積分布:")
    for rb in rank_bins:
        pct = (ranks <= rb).mean() * 100
        bar = "█" * int(pct / 2)
        print(f"      rank ≤ {rb:>3}: {pct:>5.1f}% {bar}")


# ========================================
# 4. CLIPネイティブ vs RAT: ランク順序の一致度
# ========================================
print("\n" + "=" * 70)
print("4. CLIPネイティブ順位 vs RAT順位の一致度")
print("=" * 70)

# CLIPネイティブの「正解ペアが検索空間でどのランクか」は
# phase4_analysis_correlation.pyでは計算していない
# 代わりにペアレベル類似度の順位相関で代替

from scipy.stats import spearmanr

# ペアごとの「CLIPネイティブ類似度の順位」と「RAT順位」の相関
clip_ranks = len(clip_sims) + 1 - np.argsort(np.argsort(clip_sims))  # 高い方がrank 1

rho_clip_axe, p_clip_axe = spearmanr(clip_ranks, ranks_axe)
rho_clip_dxe, p_clip_dxe = spearmanr(clip_ranks, ranks_dxe)

print(f"\n  CLIPネイティブ順位 vs A×E RAT順位: Spearman ρ={rho_clip_axe:.4f} (p={p_clip_axe:.2e})")
print(f"  CLIPネイティブ順位 vs D×E RAT順位: Spearman ρ={rho_clip_dxe:.4f} (p={p_clip_dxe:.2e})")
print(f"\n  解釈: ρが正 → CLIPで検索しやすいペアはRATでも検索しやすい（共有構造あり）")
print(f"         ρが0に近い → 独立の信号源")
