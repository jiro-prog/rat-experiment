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

# e5系モデルのプレフィクス
E5_PREFIX = "passage: "
# プレフィクスが必要なモデル（部分一致で判定）
E5_MODEL_PATTERNS = ["e5-"]

# アンカースケーリング実験
ANCHOR_COUNTS = [50, 100, 200, 500, 1000]
