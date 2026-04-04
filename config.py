from pathlib import Path

# ディレクトリ
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# シード
RANDOM_SEED = 42

# データ
DATASET_NAME = "mteb/stsbenchmark-sts"
NUM_ANCHORS = 1000
NUM_QUERIES = 500

# モデル（既存: Phase 0-4 で使用）
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "intfloat/multilingual-e5-large"
MODEL_C = "BAAI/bge-small-en-v1.5"
MODEL_D = "sentence-transformers/clip-ViT-B-32"
MODEL_E = "BAAI/bge-large-en-v1.5"

# Direction 2: モデル多様性マトリクス（12モデル）
# 軸: 目的関数 × サイズ × 言語 × ファミリー
MATRIX_MODELS = {
    # --- 既存モデル ---
    "A": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "family": "MiniLM", "params": "22M", "dim": 384,
        "training": "contrastive_distill", "lang": "en",
        "prefix": "",
    },
    "B": {
        "name": "intfloat/multilingual-e5-large",
        "family": "E5", "params": "560M", "dim": 1024,
        "training": "weak_sup_distill", "lang": "multi",
        "prefix": "passage: ",
    },
    "C": {
        "name": "BAAI/bge-small-en-v1.5",
        "family": "BGE", "params": "33M", "dim": 384,
        "training": "contrastive_retromae", "lang": "en",
        "prefix": "Represent this sentence: ",
    },
    "D": {
        "name": "sentence-transformers/clip-ViT-B-32",
        "family": "CLIP", "params": "63M", "dim": 512,
        "training": "clip_contrastive", "lang": "en",
        "prefix": "",
    },
    "E": {
        "name": "BAAI/bge-large-en-v1.5",
        "family": "BGE", "params": "335M", "dim": 1024,
        "training": "contrastive_retromae", "lang": "en",
        "prefix": "Represent this sentence: ",
    },
    # --- 新規モデル ---
    "F": {
        "name": "intfloat/e5-small-v2",
        "family": "E5", "params": "33M", "dim": 384,
        "training": "weak_sup_distill", "lang": "en",
        "prefix": "passage: ",
    },
    "G": {
        "name": "intfloat/multilingual-e5-small",
        "family": "E5", "params": "118M", "dim": 384,
        "training": "weak_sup_distill", "lang": "multi",
        "prefix": "passage: ",
    },
    "H": {
        "name": "thenlper/gte-small",
        "family": "GTE", "params": "33M", "dim": 384,
        "training": "multistage_contrastive", "lang": "en",
        "prefix": "",
    },
    "I": {
        "name": "thenlper/gte-large",
        "family": "GTE", "params": "335M", "dim": 1024,
        "training": "multistage_contrastive", "lang": "en",
        "prefix": "",
    },
    "J": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "family": "MPNet", "params": "109M", "dim": 768,
        "training": "mse_distill", "lang": "en",
        "prefix": "",
    },
    "K": {
        "name": "BAAI/bge-base-en-v1.5",
        "family": "BGE", "params": "109M", "dim": 768,
        "training": "contrastive_retromae", "lang": "en",
        "prefix": "Represent this sentence: ",
    },
    "L": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "family": "Nomic", "params": "137M", "dim": 768,
        "training": "unsup_contrastive", "lang": "en",
        "prefix": "search_document: ",
        "trust_remote_code": True,
    },
    # --- 2024-2025世代モデル ---
    "M": {
        "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "family": "GTE-Qwen2", "params": "1.5B", "dim": 1536,
        "training": "llm_instruct", "lang": "multi",
        "prefix": "",
    },
    "N": {
        "name": "Snowflake/snowflake-arctic-embed-m",
        "family": "Arctic", "params": "109M", "dim": 768,
        "training": "contrastive_retromae", "lang": "en",
        "prefix": "",
    },
}

# モデルごとのプレフィクス設定（既存スクリプト互換）
MODEL_CONFIGS = {m["name"]: {"prefix": m["prefix"]} for m in MATRIX_MODELS.values()}
# trust_remote_code が必要なモデル
for m in MATRIX_MODELS.values():
    if m.get("trust_remote_code"):
        MODEL_CONFIGS[m["name"]]["trust_remote_code"] = True

# アンカースケーリング実験
ANCHOR_COUNTS = [50, 100, 200, 500, 1000]
