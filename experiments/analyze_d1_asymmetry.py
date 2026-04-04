"""
D1結果の方向非対称性分析

分析1: 方向非対称性の原因（FPS空間のsim_meanとの相関）
分析2: オラクル方向選択の上界（max(A→B, B→A)）
分析3: 自動選択指標の探索（sim_meanベースの方向判定）
分析4: RAT大幅劣位ケースの個別分析
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr, pearsonr

import config

# 結果読み込み
RESULTS_PATH = config.RESULTS_DIR / "d1_alignment" / "d1_results.json"
with open(RESULTS_PATH) as f:
    data = json.load(f)

results = data["results"]
models = data["models"]

# seed=42 の結果に絞る（代表seed）
primary = [r for r in results if r["seed"] == 42]

# ========================================
# ヘルパー
# ========================================
def get_result(method, qm, dm, K, dataset=None):
    """指定条件のレコードを取得。"""
    ds = dataset or primary
    matches = [
        r for r in ds if r["method"] == method
        and r["query_model"] == qm and r["db_model"] == dm and r["K"] == K
    ]
    return matches[0] if matches else None


def get_best_linear(qm, dm, K, dataset=None):
    """指定ペア・KのBest Linear結果を返す。"""
    ds = dataset or primary
    linear = [
        r for r in ds
        if r["query_model"] == qm and r["db_model"] == dm and r["K"] == K
        and r["method"] in ("Procrustes", "Ridge", "Affine")
    ]
    if not linear:
        return None
    return max(linear, key=lambda x: x["recall_at_1"])


# 全無向ペアを収集
labels = list(models.keys())
undirected_pairs = []
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i < j:
            undirected_pairs.append((a, b))


# ========================================
# 分析1: 方向非対称性の原因
# ========================================
print("=" * 70)
print("分析1: 方向非対称性の原因")
print("=" * 70)

for K in [50, 100, 500]:
    print(f"\n--- K={K} ---")
    asymmetries = []
    for a, b in undirected_pairs:
        r_ab = get_result("RAT", a, b, K)
        r_ba = get_result("RAT", b, a, K)
        if r_ab is None or r_ba is None:
            continue

        r1_ab = r_ab["recall_at_1"]
        r1_ba = r_ba["recall_at_1"]
        asym = abs(r1_ab - r1_ba)

        # better/worse方向
        if r1_ab >= r1_ba:
            better_dir, worse_dir = f"{a}→{b}", f"{b}→{a}"
            better_sm, worse_sm = r_ab["sim_mean"], r_ba["sim_mean"]
            better_r1, worse_r1 = r1_ab, r1_ba
        else:
            better_dir, worse_dir = f"{b}→{a}", f"{a}→{b}"
            better_sm, worse_sm = r_ba["sim_mean"], r_ab["sim_mean"]
            better_r1, worse_r1 = r1_ba, r1_ab

        asymmetries.append({
            "pair": f"{a}-{b}",
            "better_dir": better_dir,
            "worse_dir": worse_dir,
            "better_r1": better_r1,
            "worse_r1": worse_r1,
            "asymmetry": asym,
            "better_sim_mean": better_sm,
            "worse_sim_mean": worse_sm,
            "dim_a": models[a]["dim"],
            "dim_b": models[b]["dim"],
            "family_a": models[a]["family"],
            "family_b": models[b]["family"],
        })

    # 非対称性の分布
    asyms = [a["asymmetry"] for a in asymmetries]
    print(f"  N={len(asymmetries)} undirected pairs")
    print(f"  asymmetry: mean={np.mean(asyms)*100:.1f}%p, "
          f"median={np.median(asyms)*100:.1f}%p, "
          f"max={np.max(asyms)*100:.1f}%p")

    # sim_meanとの相関
    better_sms = [a["better_sim_mean"] for a in asymmetries]
    worse_sms = [a["worse_sim_mean"] for a in asymmetries]
    better_r1s = [a["better_r1"] for a in asymmetries]
    worse_r1s = [a["worse_r1"] for a in asymmetries]

    # worse側のsim_meanが高い（=アンカー密集）→ R@1が低い？
    rho_worse_sm, p_worse = spearmanr(worse_sms, worse_r1s)
    rho_better_sm, p_better = spearmanr(better_sms, better_r1s)
    print(f"  Spearman(worse_sim_mean, worse_R@1)   = {rho_worse_sm:.3f} (p={p_worse:.2e})")
    print(f"  Spearman(better_sim_mean, better_R@1) = {rho_better_sm:.3f} (p={p_better:.2e})")

    # sim_mean差と非対称性の相関
    sm_diffs = [abs(a["better_sim_mean"] - a["worse_sim_mean"]) for a in asymmetries]
    rho_diff, p_diff = spearmanr(sm_diffs, asyms)
    print(f"  Spearman(|Δsim_mean|, asymmetry)      = {rho_diff:.3f} (p={p_diff:.2e})")

    # sim_mean ≥ 0.65 の空間がFPS側のとき vs そうでないとき
    threshold = 0.65
    worse_harmful = [a for a in asymmetries if a["worse_sim_mean"] >= threshold]
    worse_healthy = [a for a in asymmetries if a["worse_sim_mean"] < threshold]
    if worse_harmful:
        print(f"  worse側 sim_mean≥{threshold}: {len(worse_harmful)} pairs, "
              f"mean worse_R@1={np.mean([a['worse_r1'] for a in worse_harmful])*100:.1f}%")
    if worse_healthy:
        print(f"  worse側 sim_mean<{threshold}: {len(worse_healthy)} pairs, "
              f"mean worse_R@1={np.mean([a['worse_r1'] for a in worse_healthy])*100:.1f}%")

    # Top-10 最大非対称ペア
    if K == 500:
        print(f"\n  Top-10 非対称ペア (K={K}):")
        sorted_asym = sorted(asymmetries, key=lambda x: -x["asymmetry"])
        for i, a in enumerate(sorted_asym[:10]):
            print(f"    {i+1}. {a['better_dir']} R@1={a['better_r1']*100:.1f}% vs "
                  f"{a['worse_dir']} R@1={a['worse_r1']*100:.1f}%  "
                  f"(Δ={a['asymmetry']*100:.1f}%p, "
                  f"sm_better={a['better_sim_mean']:.3f}, sm_worse={a['worse_sim_mean']:.3f})")


# ========================================
# 分析2: オラクル方向選択の上界
# ========================================
print(f"\n{'='*70}")
print("分析2: オラクル方向選択の上界")
print(f"{'='*70}")

for K in [10, 25, 50, 100, 200, 500]:
    rat_original = []  # 全132有向ペア
    rat_oracle = []    # 66無向ペア × oracle
    best_linear_original = []
    best_linear_oracle = []

    for a, b in undirected_pairs:
        r_ab = get_result("RAT", a, b, K)
        r_ba = get_result("RAT", b, a, K)
        if r_ab is None or r_ba is None:
            continue

        # RAT original (両方向を含む)
        rat_original.append(r_ab["recall_at_1"])
        rat_original.append(r_ba["recall_at_1"])

        # RAT oracle (良い方を選択)
        rat_oracle.append(max(r_ab["recall_at_1"], r_ba["recall_at_1"]))

        # Best linear original (両方向)
        bl_ab = get_best_linear(a, b, K)
        bl_ba = get_best_linear(b, a, K)
        if bl_ab:
            best_linear_original.append(bl_ab["recall_at_1"])
        if bl_ba:
            best_linear_original.append(bl_ba["recall_at_1"])
        # Best linear oracle
        bls = []
        if bl_ab:
            bls.append(bl_ab["recall_at_1"])
        if bl_ba:
            bls.append(bl_ba["recall_at_1"])
        if bls:
            best_linear_oracle.append(max(bls))

    # 「両方向やってbetter側を選ぶ」RAT
    # 公平比較: best_linear_originalは132方向の平均、rat_oracleは66ペアのbetter方向
    print(f"  K={K:>3}: RAT_orig={np.mean(rat_original)*100:.1f}%  "
          f"RAT_oracle={np.mean(rat_oracle)*100:.1f}%  "
          f"BestLinear={np.mean(best_linear_original)*100:.1f}%  "
          f"BestLinear_oracle={np.mean(best_linear_oracle)*100:.1f}%  "
          f"Δ(oracle-linear)={((np.mean(rat_oracle) - np.mean(best_linear_original))*100):+.1f}%p")


# ========================================
# 分析3: 自動選択指標の探索
# ========================================
print(f"\n{'='*70}")
print("分析3: sim_meanベースの方向自動選択")
print(f"{'='*70}")

for K in [50, 100, 500]:
    # 「sim_meanが低い方の空間でFPSを走らせる」ルール
    correct = 0
    total = 0
    rat_auto = []

    for a, b in undirected_pairs:
        r_ab = get_result("RAT", a, b, K)
        r_ba = get_result("RAT", b, a, K)
        if r_ab is None or r_ba is None:
            continue

        total += 1
        sm_ab = r_ab["sim_mean"]  # FPS on A space
        sm_ba = r_ba["sim_mean"]  # FPS on B space

        # sim_mean低い方を選択
        if sm_ab <= sm_ba:
            chosen_r1 = r_ab["recall_at_1"]
            actual_better = r_ab["recall_at_1"] >= r_ba["recall_at_1"]
        else:
            chosen_r1 = r_ba["recall_at_1"]
            actual_better = r_ba["recall_at_1"] >= r_ab["recall_at_1"]

        rat_auto.append(chosen_r1)
        if actual_better:
            correct += 1

    accuracy = correct / total if total else 0
    print(f"  K={K:>3}: 「sim_mean低い方でFPS」ルール: "
          f"正解率={accuracy*100:.1f}% ({correct}/{total}), "
          f"mean R@1={np.mean(rat_auto)*100:.1f}%")

    # 「両方向やってsim_meanとR@1の相対比較で選ぶ」— 実質オラクルに近い
    # より実用的: 「両方向のFPSを走らせて、anchor entropy（or sim_mean）が低い方を選ぶ」

# 追加指標: anchor entropy相関
print(f"\n--- 追加: FPS側空間のsim_std、dim、family効果 ---")
K = 500
all_rat_k500 = [r for r in primary if r["method"] == "RAT" and r["K"] == K]

# sim_mean vs R@1
sms = [r["sim_mean"] for r in all_rat_k500]
r1s = [r["recall_at_1"] for r in all_rat_k500]
rho, p = spearmanr(sms, r1s)
print(f"  Spearman(sim_mean, R@1) K=500: ρ={rho:.3f} (p={p:.2e})")

# dim差 vs R@1
dim_diffs = [abs(r["dim_x"] - r["dim_y"]) for r in all_rat_k500]
rho_dim, p_dim = spearmanr(dim_diffs, r1s)
print(f"  Spearman(|Δdim|, R@1) K=500:   ρ={rho_dim:.3f} (p={p_dim:.2e})")

# 同次元 vs 異次元
same_dim = [r["recall_at_1"] for r in all_rat_k500 if r["same_dim"]]
diff_dim = [r["recall_at_1"] for r in all_rat_k500 if not r["same_dim"]]
print(f"  同次元ペア:  mean R@1={np.mean(same_dim)*100:.1f}% (N={len(same_dim)})")
print(f"  異次元ペア:  mean R@1={np.mean(diff_dim)*100:.1f}% (N={len(diff_dim)})")

# 同ファミリー vs 異ファミリー
same_fam = [r for r in all_rat_k500
            if models[r["query_model"]]["family"] == models[r["db_model"]]["family"]]
diff_fam = [r for r in all_rat_k500
            if models[r["query_model"]]["family"] != models[r["db_model"]]["family"]]
if same_fam:
    print(f"  同ファミリー: mean R@1={np.mean([r['recall_at_1'] for r in same_fam])*100:.1f}% (N={len(same_fam)})")
if diff_fam:
    print(f"  異ファミリー: mean R@1={np.mean([r['recall_at_1'] for r in diff_fam])*100:.1f}% (N={len(diff_fam)})")


# ========================================
# 分析4: RAT大幅劣位ケースの個別分析
# ========================================
print(f"\n{'='*70}")
print("分析4: RAT大幅劣位ケース (K=500, ΔRAT-Linear < -40%p)")
print(f"{'='*70}")

for r in all_rat_k500:
    qm, dm = r["query_model"], r["db_model"]
    bl = get_best_linear(qm, dm, 500)
    if bl is None:
        continue
    delta = (r["recall_at_1"] - bl["recall_at_1"]) * 100
    if delta < -40:
        # 逆方向も確認
        r_rev = get_result("RAT", dm, qm, 500)
        bl_rev = get_best_linear(dm, qm, 500)
        rev_r1 = r_rev["recall_at_1"] * 100 if r_rev else None
        rev_bl = bl_rev["recall_at_1"] * 100 if bl_rev else None

        print(f"  {qm}→{dm}: RAT={r['recall_at_1']*100:.1f}% vs "
              f"{bl['method']}={bl['recall_at_1']*100:.1f}% (Δ={delta:.1f}%p) "
              f"sim_mean={r['sim_mean']:.3f}, same_dim={r['same_dim']}")
        if rev_r1 is not None:
            print(f"    逆方向 {dm}→{qm}: RAT={rev_r1:.1f}%"
                  + (f", BestLinear={rev_bl:.1f}%" if rev_bl else ""))


# ========================================
# 分析5: K別 × sim_mean帯 の RAT vs Linear
# ========================================
print(f"\n{'='*70}")
print("分析5: sim_mean帯別の RAT vs Best Linear (seed=42)")
print(f"{'='*70}")

sm_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 1.0)]
for K in [25, 50, 100, 500]:
    print(f"\n  K={K}:")
    for lo, hi in sm_bins:
        rat_r1s = []
        lin_r1s = []
        rat_wins = 0
        total = 0
        for r in primary:
            if r["method"] != "RAT" or r["K"] != K:
                continue
            if not (lo <= r["sim_mean"] < hi):
                continue
            bl = get_best_linear(r["query_model"], r["db_model"], K)
            if bl is None:
                continue
            rat_r1s.append(r["recall_at_1"])
            lin_r1s.append(bl["recall_at_1"])
            if r["recall_at_1"] > bl["recall_at_1"]:
                rat_wins += 1
            total += 1
        if total:
            print(f"    sim_mean [{lo:.2f}, {hi:.2f}): N={total:>3}, "
                  f"RAT={np.mean(rat_r1s)*100:.1f}%, "
                  f"Linear={np.mean(lin_r1s)*100:.1f}%, "
                  f"RAT_wins={rat_wins}/{total} ({rat_wins/total*100:.0f}%)")


# ========================================
# 分析6: 線形手法の方向非対称性との比較
# ========================================
print(f"\n{'='*70}")
print("分析6: 手法別の方向非対称性 (K=500)")
print(f"{'='*70}")

for method in ["RAT", "Procrustes", "Ridge", "Affine"]:
    asyms = []
    for a, b in undirected_pairs:
        r_ab = get_result(method, a, b, 500)
        r_ba = get_result(method, b, a, 500)
        if r_ab and r_ba:
            asyms.append(abs(r_ab["recall_at_1"] - r_ba["recall_at_1"]))
    if asyms:
        print(f"  {method:>10}: mean={np.mean(asyms)*100:.1f}%p, "
              f"median={np.median(asyms)*100:.1f}%p, "
              f"max={np.max(asyms)*100:.1f}%p  (N={len(asyms)})")
