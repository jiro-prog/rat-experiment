"""
RAT Phase 4 分析: CLIPネイティブ vs RAT のペアレベル相関

目的:
  CLIP直接コサイン類似度とRAT類似度のペアレベル相関を分析し、
  RATが拾えている構造と拾えていない構造を分離する。

分析内容:
  1. ペアごとのCLIPネイティブ類似度 vs RAT類似度の散布図データ
  2. RATが成功するペア / 失敗するペアの特徴分析
  3. D×E vs A×E の差がどこから来るかの構造的比較
  4. アンカーとの距離パターンの相関分析

論文§5.4向け
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

import config
from src.anchor_sampler import select_anchors_fps
from src.embedder import embed_texts, embed_images_clip
from src.relative_repr import to_relative
from run_phase4_step2 import load_coco_pairs, compute_anchor_sim_stats

CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"
K = 3000
NUM_QUERIES = 500


def pairwise_rat_similarity(rel_A, rel_B):
    """各ペア(i,i)のRAT表現間コサイン類似度を返す。"""
    # 行ごとにdot product（L2正規化してからdot）
    norms_A = np.linalg.norm(rel_A, axis=1, keepdims=True)
    norms_B = np.linalg.norm(rel_B, axis=1, keepdims=True)
    norms_A[norms_A == 0] = 1.0
    norms_B[norms_B == 0] = 1.0
    rel_A_n = rel_A / norms_A
    rel_B_n = rel_B / norms_B
    return np.sum(rel_A_n * rel_B_n, axis=1)  # (N,)


def retrieval_rank(rel_q, rel_db):
    """各クエリiの正解ランクを返す。"""
    sim = cosine_similarity(rel_q, rel_db)
    ranks = []
    for i in range(len(rel_q)):
        rank = int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1
        ranks.append(rank)
    return np.array(ranks)


def analyze_success_failure(ranks, clip_native_sims, captions, label, top_n=20):
    """成功/失敗ペアの特徴を分析する。"""
    print(f"\n  --- {label}: 成功 vs 失敗 分析 ---")

    success_mask = ranks == 1
    fail_mask = ranks > 10

    print(f"  成功 (rank=1): {success_mask.sum()}件")
    print(f"  失敗 (rank>10): {fail_mask.sum()}件")

    if success_mask.sum() > 0:
        print(f"    成功ペアのCLIPネイティブ類似度: mean={clip_native_sims[success_mask].mean():.4f}, "
              f"std={clip_native_sims[success_mask].std():.4f}")
    if fail_mask.sum() > 0:
        print(f"    失敗ペアのCLIPネイティブ類似度: mean={clip_native_sims[fail_mask].mean():.4f}, "
              f"std={clip_native_sims[fail_mask].std():.4f}")

    # キャプション長の分析
    cap_lens = np.array([len(c.split()) for c in captions])
    if success_mask.sum() > 0:
        print(f"    成功ペアのキャプション長: mean={cap_lens[success_mask].mean():.1f} words")
    if fail_mask.sum() > 0:
        print(f"    失敗ペアのキャプション長: mean={cap_lens[fail_mask].mean():.1f} words")

    # 最悪の失敗例（CLIPネイティブ類似度が高いのにRATで失敗）
    if fail_mask.sum() > 0:
        fail_indices = np.where(fail_mask)[0]
        fail_clip_sims = clip_native_sims[fail_indices]
        worst = fail_indices[np.argsort(-fail_clip_sims)[:top_n]]
        print(f"\n    CLIPで高類似度なのにRAT失敗 (top {min(top_n, len(worst))}):")
        for idx in worst[:5]:
            print(f"      rank={ranks[idx]:>4}, CLIP_sim={clip_native_sims[idx]:.4f}: {captions[idx][:80]}")

    return {
        "n_success": int(success_mask.sum()),
        "n_fail": int(fail_mask.sum()),
        "success_clip_sim_mean": float(clip_native_sims[success_mask].mean()) if success_mask.sum() > 0 else None,
        "fail_clip_sim_mean": float(clip_native_sims[fail_mask].mean()) if fail_mask.sum() > 0 else None,
        "success_cap_len_mean": float(cap_lens[success_mask].mean()) if success_mask.sum() > 0 else None,
        "fail_cap_len_mean": float(cap_lens[fail_mask].mean()) if fail_mask.sum() > 0 else None,
    }


def analyze_anchor_pattern_correlation(rel_A_text, rel_B_img, rel_D_text, rel_E_img, label_pairs):
    """
    異なるペア間でアンカー距離パターンの相関を比較する。

    RATの仮説: 同じアンカーとの距離パターンが似ていれば検索が成功する。
    ペアごとにアンカー距離パターンのSpearman相関を計算し、
    A×E vs D×E でパターン保存の程度を比較する。
    """
    print(f"\n  --- アンカー距離パターン相関 ---")

    for rel_text, rel_img, label in label_pairs:
        # 各ペアのアンカー距離パターン相関
        correlations = []
        for i in range(len(rel_text)):
            rho, _ = spearmanr(rel_text[i], rel_img[i])
            if not np.isnan(rho):
                correlations.append(rho)

        correlations = np.array(correlations)
        print(f"\n  {label}:")
        print(f"    ペアごとSpearman ρ: mean={correlations.mean():.4f}, "
              f"median={np.median(correlations):.4f}, std={correlations.std():.4f}")
        print(f"    ρ > 0.5 の割合: {(correlations > 0.5).mean()*100:.1f}%")
        print(f"    ρ > 0.3 の割合: {(correlations > 0.3).mean()*100:.1f}%")
        print(f"    ρ < 0 の割合: {(correlations < 0).mean()*100:.1f}%")

    return correlations


def main():
    start_time = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4 分析: CLIPネイティブ vs RAT 相関分析")
    print(f"  K={K}, クエリ={NUM_QUERIES}")
    print("=" * 60)

    # ========================================
    # データ準備
    # ========================================
    total_needed = K + NUM_QUERIES
    print(f"\n--- データ準備: COCO {total_needed}組 ---")
    all_pairs = load_coco_pairs(total_needed, offset=0, seed=config.RANDOM_SEED)

    anchor_pairs = all_pairs[:K]
    query_pairs = all_pairs[K:K + NUM_QUERIES]

    anchor_captions = [p["caption"] for p in anchor_pairs]
    anchor_images = [p["image"] for p in anchor_pairs]
    query_captions = [p["caption"] for p in query_pairs]
    query_images = [p["image"] for p in query_pairs]

    # ========================================
    # Embedding
    # ========================================
    print("\n--- Embedding ---")

    print("  CLIP-text...")
    anchor_emb_ct = embed_texts(config.MODEL_D, anchor_captions)
    query_emb_ct = embed_texts(config.MODEL_D, query_captions)

    print("  CLIP-image...")
    anchor_emb_ci = embed_images_clip(anchor_images, CLIP_VISION_MODEL)
    query_emb_ci = embed_images_clip(query_images, CLIP_VISION_MODEL)

    print("  MiniLM...")
    anchor_emb_ml = embed_texts(config.MODEL_A, anchor_captions)
    query_emb_ml = embed_texts(config.MODEL_A, query_captions)

    # FPS
    print(f"\n  FPS (K={K})...")
    fps_idx, _ = select_anchors_fps(anchor_emb_ct, anchor_captions, K)
    a_ct = anchor_emb_ct[fps_idx]
    a_ci = anchor_emb_ci[fps_idx]
    a_ml = anchor_emb_ml[fps_idx]

    # ========================================
    # RAT変換
    # ========================================
    print("\n--- RAT変換 (poly, degree=2) ---")

    rel_ct = to_relative(query_emb_ct, a_ct, kernel="poly", degree=2, coef0=1.0)
    rel_ci = to_relative(query_emb_ci, a_ci, kernel="poly", degree=2, coef0=1.0)
    rel_ml = to_relative(query_emb_ml, a_ml, kernel="poly", degree=2, coef0=1.0)

    # ========================================
    # 1. CLIPネイティブ類似度
    # ========================================
    print("\n" + "=" * 60)
    print("1. CLIPネイティブ vs RAT ペアレベル相関")
    print("=" * 60)

    # CLIPネイティブ: text-image ペアごとのコサイン類似度
    clip_native_sims = np.sum(query_emb_ct * query_emb_ci, axis=1)  # (N,) 正規化済みなのでdot=cosine

    print(f"\n  CLIPネイティブ類似度分布:")
    print(f"    mean={clip_native_sims.mean():.4f}, std={clip_native_sims.std():.4f}")
    print(f"    min={clip_native_sims.min():.4f}, max={clip_native_sims.max():.4f}")

    # RAT類似度: D×E, A×E
    rat_sims_dxe = pairwise_rat_similarity(rel_ct, rel_ci)
    rat_sims_axe = pairwise_rat_similarity(rel_ml, rel_ci)

    print(f"\n  RAT D×E ペア類似度:")
    print(f"    mean={rat_sims_dxe.mean():.4f}, std={rat_sims_dxe.std():.4f}")

    print(f"\n  RAT A×E ペア類似度:")
    print(f"    mean={rat_sims_axe.mean():.4f}, std={rat_sims_axe.std():.4f}")

    # 相関
    rho_dxe, p_dxe = spearmanr(clip_native_sims, rat_sims_dxe)
    rho_axe, p_axe = spearmanr(clip_native_sims, rat_sims_axe)
    r_dxe, _ = pearsonr(clip_native_sims, rat_sims_dxe)
    r_axe, _ = pearsonr(clip_native_sims, rat_sims_axe)

    print(f"\n  CLIPネイティブ ↔ RAT D×E: Spearman ρ={rho_dxe:.4f} (p={p_dxe:.2e}), Pearson r={r_dxe:.4f}")
    print(f"  CLIPネイティブ ↔ RAT A×E: Spearman ρ={rho_axe:.4f} (p={p_axe:.2e}), Pearson r={r_axe:.4f}")

    # RAT D×E ↔ RAT A×E の相関
    rho_cross, p_cross = spearmanr(rat_sims_dxe, rat_sims_axe)
    print(f"  RAT D×E ↔ RAT A×E:       Spearman ρ={rho_cross:.4f} (p={p_cross:.2e})")

    # ========================================
    # 2. 成功/失敗分析
    # ========================================
    print("\n" + "=" * 60)
    print("2. 成功 vs 失敗ペアの特徴")
    print("=" * 60)

    ranks_dxe = retrieval_rank(rel_ct, rel_ci)
    ranks_axe = retrieval_rank(rel_ml, rel_ci)

    sf_dxe = analyze_success_failure(ranks_dxe, clip_native_sims, query_captions, "D×E")
    sf_axe = analyze_success_failure(ranks_axe, clip_native_sims, query_captions, "A×E")

    # A×Eで成功してD×Eで失敗（& その逆）
    axe_only = (ranks_axe == 1) & (ranks_dxe > 10)
    dxe_only = (ranks_dxe == 1) & (ranks_axe > 10)
    both_success = (ranks_axe == 1) & (ranks_dxe == 1)

    print(f"\n  --- A×Eのみ成功: {axe_only.sum()}件, D×Eのみ成功: {dxe_only.sum()}件, 両方成功: {both_success.sum()}件 ---")
    if axe_only.sum() > 0:
        print(f"    A×Eのみ成功のCLIPネイティブsim: mean={clip_native_sims[axe_only].mean():.4f}")
    if dxe_only.sum() > 0:
        print(f"    D×Eのみ成功のCLIPネイティブsim: mean={clip_native_sims[dxe_only].mean():.4f}")

    # ========================================
    # 2b. 3パターン分析: CLIPネイティブ vs RAT A×E
    # ========================================
    print("\n" + "=" * 60)
    print("2b. 3パターン分析 (CLIP vs RAT)")
    print("=" * 60)

    # CLIPネイティブで正解ペアの類似度が上位25%か下位25%かで二値化
    clip_high_thresh = np.percentile(clip_native_sims, 75)
    clip_low_thresh = np.percentile(clip_native_sims, 25)
    # RAT A×Eで成功(rank<=5) vs 失敗(rank>50) で二値化
    rat_success = ranks_axe <= 5
    rat_fail = ranks_axe > 50

    # パターン1: 両方高い
    p1 = (clip_native_sims >= clip_high_thresh) & rat_success
    # パターン2: CLIPは高いがRATは低い
    p2 = (clip_native_sims >= clip_high_thresh) & rat_fail
    # パターン3: RATは高いがCLIPは低い
    p3 = (clip_native_sims <= clip_low_thresh) & rat_success

    for label, mask, desc in [
        ("パターン1", p1, "CLIP高 & RAT成功 — RATが拾えている"),
        ("パターン2", p2, "CLIP高 & RAT失敗 — RATの取りこぼし"),
        ("パターン3", p3, "CLIP低 & RAT成功 — RATがCLIPを超えている"),
    ]:
        n = mask.sum()
        print(f"\n  {label}: {desc} ({n}件)")
        if n > 0:
            cap_lens = np.array([len(query_captions[i].split()) for i in range(NUM_QUERIES)])
            print(f"    キャプション長: mean={cap_lens[mask].mean():.1f} words")
            print(f"    CLIPネイティブsim: mean={clip_native_sims[mask].mean():.4f}")
            # サンプル表示
            indices = np.where(mask)[0]
            sample_n = min(30, len(indices))
            # パターン2はCLIP simが高い順、パターン3はRAT rankが低い（良い）順
            if "2" in label:
                sample_idx = indices[np.argsort(-clip_native_sims[indices])[:sample_n]]
            elif "3" in label:
                sample_idx = indices[np.argsort(ranks_axe[indices])[:sample_n]]
            else:
                sample_idx = indices[:sample_n]

            print(f"    サンプル ({sample_n}件):")
            for idx in sample_idx:
                print(f"      CLIP={clip_native_sims[idx]:.3f} RAT_rank={ranks_axe[idx]:>3}: {query_captions[idx][:100]}")

    # ========================================
    # 3. アンカー距離パターンの構造比較
    # ========================================
    print("\n" + "=" * 60)
    print("3. アンカー距離パターン相関（RATの核心）")
    print("=" * 60)

    label_pairs = [
        (rel_ml, rel_ci, "A×E (MiniLM × CLIP-image)"),
        (rel_ct, rel_ci, "D×E (CLIP-text × CLIP-image)"),
        (rel_ml, rel_ct, "A×D (MiniLM × CLIP-text) — テキスト同士参考"),
    ]
    analyze_anchor_pattern_correlation(rel_ml, rel_ci, rel_ct, rel_ci, label_pairs)

    # ========================================
    # 4. CLIPネイティブ類似度バケット別のRAT精度
    # ========================================
    print("\n" + "=" * 60)
    print("4. CLIPネイティブ類似度バケット別のRAT精度")
    print("=" * 60)

    # バケット分割
    percentiles = [0, 25, 50, 75, 100]
    boundaries = np.percentile(clip_native_sims, percentiles)

    print(f"\n  {'バケット':<20} {'N':>5} {'D×E R@1':>10} {'A×E R@1':>10} {'CLIP mean':>12}")
    print(f"  {'-' * 60}")

    bucket_results = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i < len(boundaries) - 2:
            mask = (clip_native_sims >= lo) & (clip_native_sims < hi)
        else:
            mask = (clip_native_sims >= lo) & (clip_native_sims <= hi)

        n = mask.sum()
        if n == 0:
            continue

        dxe_r1 = (ranks_dxe[mask] == 1).mean()
        axe_r1 = (ranks_axe[mask] == 1).mean()
        clip_mean = clip_native_sims[mask].mean()

        label = f"[{lo:.3f}, {hi:.3f})"
        print(f"  {label:<20} {n:>5} {dxe_r1*100:>9.1f}% {axe_r1*100:>9.1f}% {clip_mean:>12.4f}")

        bucket_results.append({
            "range": [float(lo), float(hi)],
            "n": int(n),
            "dxe_recall_at_1": float(dxe_r1),
            "axe_recall_at_1": float(axe_r1),
            "clip_native_mean": float(clip_mean),
        })

    # ========================================
    # 保存
    # ========================================
    elapsed = time.time() - start_time

    output = {
        "config": {"K": K, "num_queries": NUM_QUERIES, "kernel": "poly_deg2"},
        "clip_native_stats": {
            "mean": float(clip_native_sims.mean()),
            "std": float(clip_native_sims.std()),
            "min": float(clip_native_sims.min()),
            "max": float(clip_native_sims.max()),
        },
        "rat_pairwise_stats": {
            "DxE": {"mean": float(rat_sims_dxe.mean()), "std": float(rat_sims_dxe.std())},
            "AxE": {"mean": float(rat_sims_axe.mean()), "std": float(rat_sims_axe.std())},
        },
        "correlations": {
            "clip_native_vs_rat_DxE": {"spearman": float(rho_dxe), "pearson": float(r_dxe)},
            "clip_native_vs_rat_AxE": {"spearman": float(rho_axe), "pearson": float(r_axe)},
            "rat_DxE_vs_rat_AxE": {"spearman": float(rho_cross)},
        },
        "success_failure": {"DxE": sf_dxe, "AxE": sf_axe},
        "cross_analysis": {
            "axe_only_success": int(axe_only.sum()),
            "dxe_only_success": int(dxe_only.sum()),
            "both_success": int(both_success.sum()),
        },
        "bucket_analysis": bucket_results,
        "elapsed_seconds": elapsed,
    }

    # ペアレベルデータ（散布図用）
    output["per_pair_data"] = {
        "clip_native_sim": clip_native_sims.tolist(),
        "rat_sim_DxE": rat_sims_dxe.tolist(),
        "rat_sim_AxE": rat_sims_axe.tolist(),
        "rank_DxE": ranks_dxe.tolist(),
        "rank_AxE": ranks_axe.tolist(),
        "caption_lengths": [len(c.split()) for c in query_captions],
    }

    out_path = config.RESULTS_DIR / "phase4_analysis_correlation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n\n結果保存: {out_path}")
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
