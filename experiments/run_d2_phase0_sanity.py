"""
D2 Phase 0: Pooling検証 + STS Sanity Check

Step 0a: Arctic全サイズのpooling設定が一貫していることを検証
Step 0b: 新規モデル(O,P,Q)のSTSBenchmark dev setでの品質チェック
  - Pass: ρ >= 0.65
  - Warning: 0.50 <= ρ < 0.65
  - Fail: ρ < 0.50
既存モデル(A-N)もベースラインとして計算。
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import config

# Arctic系列 + 比較用にA,C,N
MODELS_TO_CHECK = ["O", "P", "N", "Q"]
BASELINE_MODELS = ["A", "C", "J"]  # 既存のベースライン（小・中・代表）
ALL_CHECK = BASELINE_MODELS + MODELS_TO_CHECK

OUT_DIR = config.RESULTS_DIR / "d2_scale"


def check_pooling_config(model_id: str, label: str) -> dict:
    """モデルのpooling設定を確認する。"""
    print(f"\n--- Pooling検証: {label} ({model_id}) ---")
    model = SentenceTransformer(model_id)

    # SentenceTransformerのpooling layer情報を取得
    pooling_info = {}
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Pooling" in cls_name or "pooling" in name:
            pooling_info["module_name"] = name
            pooling_info["class"] = cls_name
            # sentence_transformersのPoolingモジュールの属性を取得
            if hasattr(module, "pooling_mode_cls_token"):
                pooling_info["cls_token"] = module.pooling_mode_cls_token
            if hasattr(module, "pooling_mode_mean_tokens"):
                pooling_info["mean_tokens"] = module.pooling_mode_mean_tokens
            if hasattr(module, "pooling_mode_max_tokens"):
                pooling_info["max_tokens"] = module.pooling_mode_max_tokens
            if hasattr(module, "pooling_mode_lasttoken"):
                pooling_info["lasttoken"] = module.pooling_mode_lasttoken

    # Normalize layerの確認
    has_normalize = False
    for name, module in model.named_modules():
        if "Normalize" in type(module).__name__:
            has_normalize = True

    pooling_info["has_normalize_layer"] = has_normalize

    print(f"  Pooling: {pooling_info}")
    del model
    return pooling_info


def compute_sts_correlation(model_id: str, label: str, sts_data: list) -> dict:
    """STSBenchmark dev setでSpearman相関を計算する。"""
    print(f"\n--- STS Sanity Check: {label} ({model_id.split('/')[-1]}) ---")

    kwargs = {}
    model_cfg = config.MODEL_CONFIGS.get(model_id, {})
    if model_cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    model = SentenceTransformer(model_id, **kwargs)
    prefix = model_cfg.get("prefix", "")

    sentences1 = [prefix + d["sentence1"] for d in sts_data]
    sentences2 = [prefix + d["sentence2"] for d in sts_data]
    gold_scores = [d["score"] for d in sts_data]

    t0 = time.time()
    emb1 = model.encode(sentences1, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=128)
    emb2 = model.encode(sentences2, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=128)
    elapsed = time.time() - t0

    # コサイン類似度（L2正規化済みなのでdot product）
    cos_sims = np.sum(emb1 * emb2, axis=1)
    rho, p_value = spearmanr(cos_sims, gold_scores)

    # 判定
    if rho >= 0.65:
        status = "PASS"
    elif rho >= 0.50:
        status = "WARNING"
    else:
        status = "FAIL"

    result = {
        "label": label,
        "model": model_id,
        "dim": emb1.shape[1],
        "sts_spearman_rho": round(float(rho), 4),
        "sts_p_value": float(p_value),
        "status": status,
        "n_pairs": len(sts_data),
        "encode_time_s": round(elapsed, 1),
    }

    symbol = {"PASS": "✓", "WARNING": "⚠", "FAIL": "✗"}[status]
    print(f"  dim={emb1.shape[1]}, ρ={rho:.4f}, p={p_value:.2e} "
          f"[{symbol} {status}] ({elapsed:.1f}s)")

    del model
    return result


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("D2 Phase 0: Pooling検証 + STS Sanity Check")
    print("=" * 60)

    # ========================================
    # Step 0a: Pooling検証（Arctic系列）
    # ========================================
    print("\n" + "=" * 60)
    print("Step 0a: Pooling設定検証")
    print("=" * 60)

    arctic_labels = ["O", "P", "N", "Q"]
    pooling_results = {}
    for label in arctic_labels:
        info = config.MATRIX_MODELS[label]
        pooling_results[label] = check_pooling_config(info["name"], label)

    # 一貫性チェック
    print("\n--- Arctic Pooling一貫性サマリー ---")
    pooling_modes = set()
    for label, pr in pooling_results.items():
        mode = "cls" if pr.get("cls_token") else "mean" if pr.get("mean_tokens") else "unknown"
        pooling_modes.add(mode)
        info = config.MATRIX_MODELS[label]
        print(f"  {label} ({info['params']}, {info['dim']}d): {mode} pooling, "
              f"normalize={pr.get('has_normalize_layer', '?')}")

    if len(pooling_modes) == 1:
        print(f"\n  → 全Arctic系列で {pooling_modes.pop()} pooling一貫 ✓")
    else:
        print(f"\n  → 不一貫検出: {pooling_modes} — 要対応 ✗")

    # ========================================
    # Step 0b: STS Sanity Check
    # ========================================
    print("\n" + "=" * 60)
    print("Step 0b: STSBenchmark Dev Set Sanity Check")
    print("=" * 60)

    # STSBenchmark dev set読み込み
    ds = load_dataset(config.DATASET_NAME, split="validation")
    sts_data = [{"sentence1": row["sentence1"], "sentence2": row["sentence2"],
                 "score": row["score"]} for row in ds]
    print(f"STS dev set: {len(sts_data)} pairs")

    sts_results = []
    for label in ALL_CHECK:
        info = config.MATRIX_MODELS[label]
        result = compute_sts_correlation(info["name"], label, sts_data)
        sts_results.append(result)

    # サマリーテーブル
    print("\n" + "=" * 60)
    print("STS Sanity Check サマリー")
    print("=" * 60)
    print(f"{'Label':>5} {'Model':>30} {'Params':>7} {'Dim':>5} {'ρ':>7} {'Status':>8}")
    print("-" * 65)
    for r in sorted(sts_results, key=lambda x: x["sts_spearman_rho"], reverse=True):
        short = r["model"].split("/")[-1][:28]
        info = config.MATRIX_MODELS[r["label"]]
        print(f"{r['label']:>5} {short:>30} {info['params']:>7} {r['dim']:>5} "
              f"{r['sts_spearman_rho']:>7.4f} {r['status']:>8}")

    # Go/No-go判定
    arctic_results = [r for r in sts_results if r["label"] in arctic_labels]
    pass_count = sum(1 for r in arctic_results if r["status"] in ("PASS", "WARNING"))
    print(f"\nArctic系列: {pass_count}/{len(arctic_results)} Pass/Warning")
    if pass_count >= 3:
        print("→ Go: Phase 1に進行可能")
    else:
        print("→ No-go: 設計再検討が必要")

    # 結果保存
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pooling_verification": pooling_results,
        "sts_sanity_check": sts_results,
        "go_nogo": {
            "arctic_pass_count": pass_count,
            "arctic_total": len(arctic_results),
            "decision": "GO" if pass_count >= 3 else "NO-GO",
        },
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    out_path = OUT_DIR / "d2_phase0_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()
