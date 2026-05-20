from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "coin_classifier.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
CLASS_CENTERS_PATH = ARTIFACTS_DIR / "class_centers.json"
OPENSET_CALIBRATION_PATH = ARTIFACTS_DIR / "openset_calibration.json"

# =========================
# Image / training config
# =========================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 18
VALIDATION_SPLIT = 0.2
SEED = 42
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

# =========================
# Open-set / unknown coin logic
# =========================
# How much each signal contributes to the probability of "outra_moeda"
UNKNOWN_WEIGHT_SOFTMAX = 0.60
UNKNOWN_WEIGHT_DISTANCE = 0.40

# Percentiles used during calibration on the validation split.
# Lower confidence than this percentile increases unknown probability.
SOFTMAX_KNOWN_PERCENTILE = 10
# Distance larger than this percentile increases unknown probability.
DISTANCE_KNOWN_PERCENTILE = 90

# Safety limits so the model does not return 100% unknown too easily.
MIN_UNKNOWN_PROB = 0.0
MAX_UNKNOWN_PROB = 0.95

# Output
TOP_K_TO_SHOW = 5
UNKNOWN_CLASS_NAME = "outra_moeda"
