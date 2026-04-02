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

# モデル
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "intfloat/multilingual-e5-large"
MODEL_C = "BAAI/bge-small-en-v1.5"
MODEL_D = "sentence-transformers/clip-ViT-B-32"
MODEL_E = "BAAI/bge-large-en-v1.5"

# モデルごとのプレフィクス設定
MODEL_CONFIGS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"prefix": ""},
    "intfloat/multilingual-e5-large": {"prefix": "passage: "},
    "BAAI/bge-small-en-v1.5": {"prefix": "Represent this sentence: "},
    "sentence-transformers/clip-ViT-B-32": {"prefix": ""},
    "BAAI/bge-large-en-v1.5": {"prefix": "Represent this sentence: "},
}

# アンカースケーリング実験
ANCHOR_COUNTS = [50, 100, 200, 500, 1000]
