"""
Direction 2A 分析: RAT精度の予測変数特定

1. 全ペアの特徴量CSV出力
2. OLS回帰: R@1 = f(query_sim_mean, db_entropy, ...) のR²比較
3. 方向非対称性の系統的分析
4. ファミリー効果の分離（目的関数 / 言語設計 / サイズ）
"""
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr, pearsonr

import config

# ========================================
# データ読み込み
# ========================================
results_path = config.RESULTS_DIR / "d2a_matrix.json"
with open(results_path) as f:
    data = json.load(f)

labels = data["labels"]
sim_stats = data["sim_stats"]
pair_results = data["pair_results"]
models = data["models"]

# Nomic (L) を除外
EXCLUDE = {"L"}
labels_clean = [l for l in labels if l not in EXCLUDE]
pair_clean = [p for p in pair_results if p["query"] not in EXCLUDE and p["db"] not in EXCLUDE]

print(f"分析対象: {len(labels_clean)}モデル, {len(pair_clean)}ペア (Nomic除外)")

# ========================================
# 1. 全ペア特徴量テーブル
# ========================================
print("\n" + "=" * 70)
print("1. 全ペア特徴量テーブル")
print("=" * 70)

rows = []
for p in pair_clean:
    q, d = p["query"], p["db"]
    q_info = models[q]
    d_info = models[d]
    same_family = 1 if q_info["family"] == d_info["family"] else 0
    same_training = 1 if q_info["training"] == d_info["training"] else 0
    same_lang = 1 if q_info["lang"] == d_info["lang"] else 0

    best_r1 = max(p["baseline_r1"], p["zscore_db_r1"])

    rows.append({
        "query": q,
        "db": d,
        "query_family": q_info["family"],
        "db_family": d_info["family"],
        "query_params": q_info["params"],
        "db_params": d_info["params"],
        "query_training": q_info["training"],
        "db_training": d_info["training"],
        "query_lang": q_info["lang"],
        "db_lang": d_info["lang"],
        "query_sim_mean": sim_stats[q]["sim_mean"],
        "query_sim_std": sim_stats[q]["sim_std"],
        "db_sim_mean": sim_stats[d]["sim_mean"],
        "db_sim_std": sim_stats[d]["sim_std"],
        "anchor_ent_q": p["anchor_entropy_query"],
        "anchor_ent_db": p["anchor_entropy_db"],
        "same_family": same_family,
        "same_training": same_training,
        "same_lang": same_lang,
        "baseline_r1": p["baseline_r1"],
        "zscore_db_r1": p["zscore_db_r1"],
        "best_r1": best_r1,
    })

# CSV保存
csv_path = config.RESULTS_DIR / "d2a_pair_features.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV保存: {csv_path} ({len(rows)}行)")

# ========================================
# 2. OLS回帰: R@1 予測
# ========================================
print("\n" + "=" * 70)
print("2. OLS回帰: best_R@1 の予測変数分析")
print("=" * 70)

y = np.array([r["best_r1"] for r in rows])

# 特徴量候補
features = {
    "query_sim_mean": np.array([r["query_sim_mean"] for r in rows]),
    "db_sim_mean": np.array([r["db_sim_mean"] for r in rows]),
    "query_sim_std": np.array([r["query_sim_std"] for r in rows]),
    "db_sim_std": np.array([r["db_sim_std"] for r in rows]),
    "anchor_ent_q": np.array([r["anchor_ent_q"] for r in rows]),
    "anchor_ent_db": np.array([r["anchor_ent_db"] for r in rows]),
    "same_family": np.array([r["same_family"] for r in rows], dtype=float),
    "same_training": np.array([r["same_training"] for r in rows], dtype=float),
}


def ols_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS回帰のR²と係数を返す。"""
    # intercept追加
    ones = np.ones((X.shape[0], 1))
    X_full = np.hstack([ones, X]) if X.ndim == 2 else np.column_stack([ones, X])
    # 正規方程式
    try:
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, np.array([])
    y_hat = X_full @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2, beta


# 単変量回帰
print("\n  --- 単変量回帰 ---")
print(f"  {'変数':<20} {'R²':>8} {'Pearson r':>10} {'Spearman ρ':>10} {'方向':>6}")
single_results = []
for name, x in features.items():
    r2, beta = ols_r2(x.reshape(-1, 1), y)
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    direction = "+" if beta[1] > 0 else "-"
    print(f"  {name:<20} {r2:>8.3f} {pr:>10.3f} {sr:>10.3f} {direction:>6}")
    single_results.append((name, r2))

# 多変量回帰
print("\n  --- 多変量回帰 ---")
combos = [
    ("query_sim_mean + db_sim_mean", ["query_sim_mean", "db_sim_mean"]),
    ("query_sim_mean + same_family", ["query_sim_mean", "same_family"]),
    ("query_sim_mean + db_sim_mean + same_family", ["query_sim_mean", "db_sim_mean", "same_family"]),
    ("query_sim_std + db_sim_std", ["query_sim_std", "db_sim_std"]),
    ("query_sim_std + db_sim_std + same_family", ["query_sim_std", "db_sim_std", "same_family"]),
    ("anchor_ent_q + anchor_ent_db", ["anchor_ent_q", "anchor_ent_db"]),
    ("query_sim_mean + anchor_ent_db", ["query_sim_mean", "anchor_ent_db"]),
    ("query_sim_mean + anchor_ent_db + same_family", ["query_sim_mean", "anchor_ent_db", "same_family"]),
    ("ALL (mean, std, ent, family)", ["query_sim_mean", "db_sim_mean", "query_sim_std", "db_sim_std",
                                       "anchor_ent_q", "anchor_ent_db", "same_family"]),
    ("ALL + same_training", ["query_sim_mean", "db_sim_mean", "query_sim_std", "db_sim_std",
                              "anchor_ent_q", "anchor_ent_db", "same_family", "same_training"]),
]

print(f"  {'モデル':<50} {'R²':>8}")
for label, feat_names in combos:
    X = np.column_stack([features[f] for f in feat_names])
    r2, beta = ols_r2(X, y)
    print(f"  {label:<50} {r2:>8.3f}")
    if "ALL" in label:
        print(f"    係数: intercept={beta[0]:.3f}", end="")
        for fn, b in zip(feat_names, beta[1:]):
            print(f", {fn}={b:.3f}", end="")
        print()

# ========================================
# 3. 方向非対称性の系統的分析
# ========================================
print("\n" + "=" * 70)
print("3. 方向非対称性分析")
print("=" * 70)

# ペアごとに (X→Y, Y→X) を対にする
pair_dict = {}
for r in rows:
    pair_dict[(r["query"], r["db"])] = r

asymmetries = []
for i, lx in enumerate(labels_clean):
    for j, ly in enumerate(labels_clean):
        if i >= j:
            continue
        xy = pair_dict.get((lx, ly))
        yx = pair_dict.get((ly, lx))
        if xy and yx:
            delta_r1 = xy["best_r1"] - yx["best_r1"]
            delta_sim_mean = sim_stats[lx]["sim_mean"] - sim_stats[ly]["sim_mean"]
            delta_sim_std = sim_stats[lx]["sim_std"] - sim_stats[ly]["sim_std"]
            asymmetries.append({
                "pair": f"{lx}↔{ly}",
                "r1_xy": xy["best_r1"],
                "r1_yx": yx["best_r1"],
                "abs_delta_r1": abs(delta_r1),
                "delta_r1": delta_r1,  # positive = X→Y higher
                "delta_sim_mean": delta_sim_mean,
                "abs_delta_sim_mean": abs(delta_sim_mean),
                "delta_sim_std": delta_sim_std,
            })

# |ΔR@1| vs |Δsim_mean| の相関
abs_dr1 = np.array([a["abs_delta_r1"] for a in asymmetries])
abs_dsm = np.array([a["abs_delta_sim_mean"] for a in asymmetries])
delta_r1 = np.array([a["delta_r1"] for a in asymmetries])
delta_sm = np.array([a["delta_sim_mean"] for a in asymmetries])

rho_abs, p_abs = spearmanr(abs_dsm, abs_dr1)
rho_signed, p_signed = spearmanr(delta_sm, delta_r1)

print(f"\n  |ΔR@1| vs |Δsim_mean|: Spearman ρ = {rho_abs:.3f} (p={p_abs:.2e})")
print(f"   ΔR@1  vs  Δsim_mean:  Spearman ρ = {rho_signed:.3f} (p={p_signed:.2e})")
print(f"  解釈: 負のρ → sim_meanが高い側がクエリの時に精度が低い")

# 非対称性が大きいペア top10
print(f"\n  非対称性 top10:")
print(f"  {'ペア':<10} {'X→Y':>7} {'Y→X':>7} {'|ΔR@1|':>8} {'Δsim_mean':>10}")
for a in sorted(asymmetries, key=lambda x: -x["abs_delta_r1"])[:10]:
    print(f"  {a['pair']:<10} {a['r1_xy']*100:>6.1f}% {a['r1_yx']*100:>6.1f}% "
          f"{a['abs_delta_r1']*100:>7.1f}% {a['delta_sim_mean']:>10.4f}")

# 対称性が高いペア top10
print(f"\n  対称性 top10 (|ΔR@1| 最小):")
for a in sorted(asymmetries, key=lambda x: x["abs_delta_r1"])[:10]:
    print(f"  {a['pair']:<10} {a['r1_xy']*100:>6.1f}% {a['r1_yx']*100:>6.1f}% "
          f"{a['abs_delta_r1']*100:>7.1f}% {a['delta_sim_mean']:>10.4f}")

# ========================================
# 4. ファミリー効果の分離
# ========================================
print("\n" + "=" * 70)
print("4. ファミリー効果の分離")
print("=" * 70)

# 4a. 目的関数の効果: 同サイズ・同BERTベース・異なる学習
print("\n  --- 4a. 目的関数の効果 (同サイズ33M, BERTベース) ---")
small_models = ["A", "C", "F", "H"]  # 22M-33M class
print(f"  {'ペア':<10} {'Family':<15} {'Training':<35} {'R@1(→)':>8} {'R@1(←)':>8}")
for i, lx in enumerate(small_models):
    for ly in small_models[i+1:]:
        xy = pair_dict.get((lx, ly))
        yx = pair_dict.get((ly, lx))
        if xy and yx:
            fx = models[lx]["family"]
            fy = models[ly]["family"]
            tx = models[lx]["training"]
            ty = models[ly]["training"]
            print(f"  {lx}↔{ly:<7} {fx}↔{fy:<12} {tx}↔{ty:<32} "
                  f"{xy['best_r1']*100:>7.1f}% {yx['best_r1']*100:>7.1f}%")

# 4b. 多言語化の効果: E5内 EN vs MULTI
print("\n  --- 4b. 多言語化の効果 (E5ファミリー内) ---")
e5_models = ["F", "G", "B"]  # EN-small, MULTI-small, MULTI-large
print(f"  {'ペア':<10} {'Lang':<15} {'Params':<15} {'R@1(→)':>8} {'R@1(←)':>8}")
for i, lx in enumerate(e5_models):
    for ly in e5_models[i+1:]:
        xy = pair_dict.get((lx, ly))
        yx = pair_dict.get((ly, lx))
        if xy and yx:
            lnx = models[lx]["lang"]
            lny = models[ly]["lang"]
            px = models[lx]["params"]
            py = models[ly]["params"]
            print(f"  {lx}↔{ly:<7} {lnx}↔{lny:<12} {px}↔{py:<12} "
                  f"{xy['best_r1']*100:>7.1f}% {yx['best_r1']*100:>7.1f}%")

# 4c. サイズの効果: 同一ファミリー内
print("\n  --- 4c. サイズの効果 (同一ファミリー内) ---")
family_groups = {
    "BGE": [("C", "33M"), ("K", "109M"), ("E", "335M")],
    "E5":  [("F", "33M"), ("G", "118M"), ("B", "560M")],
    "GTE": [("H", "33M"), ("I", "335M")],
}
for fam, members in family_groups.items():
    print(f"\n  {fam}:")
    for i, (lx, sx) in enumerate(members):
        for ly, sy in members[i+1:]:
            xy = pair_dict.get((lx, ly))
            yx = pair_dict.get((ly, lx))
            if xy and yx:
                print(f"    {lx}({sx})→{ly}({sy}): {xy['best_r1']*100:.1f}%  "
                      f"{ly}({sy})→{lx}({sx}): {yx['best_r1']*100:.1f}%")

# 4d. ファミリー外: 同サイズ異ファミリーのクロス精度
print("\n  --- 4d. 同サイズ異ファミリーのクロス精度 ---")
size_groups = {
    "~33M": ["A", "C", "F", "H"],
    "~109M": ["J", "K"],
    "~335M": ["E", "I"],
}
for size, members in size_groups.items():
    if len(members) < 2:
        continue
    # ファミリー内 vs ファミリー外
    in_fam_r1s = []
    out_fam_r1s = []
    for lx in members:
        for ly in members:
            if lx == ly:
                continue
            r1 = pair_dict[(lx, ly)]["best_r1"]
            if models[lx]["family"] == models[ly]["family"]:
                in_fam_r1s.append(r1)
            else:
                out_fam_r1s.append(r1)

    print(f"\n  {size} ({', '.join(members)}):")
    if in_fam_r1s:
        print(f"    同ファミリー平均:  {np.mean(in_fam_r1s)*100:.1f}%")
    print(f"    異ファミリー平均:  {np.mean(out_fam_r1s)*100:.1f}%")
    # 個別ペア
    for lx in members:
        for ly in members:
            if lx >= ly:
                continue
            xy = pair_dict[(lx, ly)]
            yx = pair_dict[(ly, lx)]
            same = "★" if models[lx]["family"] == models[ly]["family"] else " "
            print(f"    {same} {lx}({models[lx]['family']})↔{ly}({models[ly]['family']}): "
                  f"{xy['best_r1']*100:.1f}% / {yx['best_r1']*100:.1f}%")

# ========================================
# 5. z-scoreの効果分析: いつ効くか
# ========================================
print("\n" + "=" * 70)
print("5. z-score DB-sideの効果分析")
print("=" * 70)

zscore_effects = []
for r in rows:
    delta = r["zscore_db_r1"] - r["baseline_r1"]
    zscore_effects.append({
        "query": r["query"],
        "db": r["db"],
        "baseline": r["baseline_r1"],
        "zscore": r["zscore_db_r1"],
        "delta": delta,
        "query_sim_mean": r["query_sim_mean"],
        "db_sim_mean": r["db_sim_mean"],
    })

# z-score効果 vs query_sim_mean
q_means = np.array([z["query_sim_mean"] for z in zscore_effects])
deltas = np.array([z["delta"] for z in zscore_effects])
rho_z, p_z = spearmanr(q_means, deltas)
print(f"\n  z-score効果(ΔR@1) vs query_sim_mean: Spearman ρ = {rho_z:.3f} (p={p_z:.2e})")

# z-score効果 vs db_sim_mean
d_means = np.array([z["db_sim_mean"] for z in zscore_effects])
rho_zd, p_zd = spearmanr(d_means, deltas)
print(f"  z-score効果(ΔR@1) vs db_sim_mean:    Spearman ρ = {rho_zd:.3f} (p={p_zd:.2e})")

# query_sim_meanの閾値で分割
thresh = 0.4
high_q = [z for z in zscore_effects if z["query_sim_mean"] > thresh]
low_q = [z for z in zscore_effects if z["query_sim_mean"] <= thresh]
print(f"\n  query_sim_mean > {thresh} ({len(high_q)}ペア):")
print(f"    平均Δ(zscore-baseline): {np.mean([z['delta'] for z in high_q])*100:+.1f}pt")
print(f"    baseline平均: {np.mean([z['baseline'] for z in high_q])*100:.1f}%")
print(f"    zscore平均:   {np.mean([z['zscore'] for z in high_q])*100:.1f}%")
print(f"  query_sim_mean ≤ {thresh} ({len(low_q)}ペア):")
print(f"    平均Δ(zscore-baseline): {np.mean([z['delta'] for z in low_q])*100:+.1f}pt")
print(f"    baseline平均: {np.mean([z['baseline'] for z in low_q])*100:.1f}%")
print(f"    zscore平均:   {np.mean([z['zscore'] for z in low_q])*100:.1f}%")

# z-scoreが大幅に効くペア
print(f"\n  z-score効果 top10 (改善):")
for z in sorted(zscore_effects, key=lambda x: -x["delta"])[:10]:
    print(f"    {z['query']}→{z['db']}: {z['baseline']*100:.1f}%→{z['zscore']*100:.1f}% "
          f"(Δ={z['delta']*100:+.1f}pt, q_mean={z['query_sim_mean']:.3f})")

print(f"\n  z-score効果 bottom10 (悪化):")
for z in sorted(zscore_effects, key=lambda x: x["delta"])[:10]:
    print(f"    {z['query']}→{z['db']}: {z['baseline']*100:.1f}%→{z['zscore']*100:.1f}% "
          f"(Δ={z['delta']*100:+.1f}pt, q_mean={z['query_sim_mean']:.3f})")

print("\n完了")
