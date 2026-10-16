"""Central place for paths and constants. Change values here, not all over the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42   # makes results reproducible
TEST_SIZE = 0.20    # 20% of data is held back for the final exam
CV_FOLDS = 5        # cross-validation folds used on the training part
