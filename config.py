"""
Configuration constants for the Dynamic Pricing Engine.

Centralizes all paths, hyperparameters, and feature definitions
so nothing is hardcoded across the codebase.
"""

import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Data files ────────────────────────────────────────────────────
TRAIN_FILE = DATA_RAW / "train.tsv"
TEST_FILE = DATA_RAW / "test.tsv"
PROCESSED_TRAIN = DATA_PROCESSED / "train_features.parquet"
PROCESSED_TEST = DATA_PROCESSED / "test_features.parquet"

# ── Model artefacts ───────────────────────────────────────────────
MODEL_FILE = MODELS_DIR / "lgbm_price_model.pkl"

# ── Reproducibility ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_OPTUNA_TRIALS = 50
CV_FOLDS = 5

# ── Pricing sweep parameters ────────────────────────────────────
PRICE_SWEEP_MIN = 0.4
PRICE_SWEEP_MAX = 2.5
PRICE_SWEEP_STEPS = 100

# ── Target columns ──────────────────────────────────────────────
TARGET = "price"
LOG_TARGET = "log_price"

# ── Feature definitions ─────────────────────────────────────────
CAT_FEATURES = [
    "category_main",
    "category_sub",
    "category_leaf",
    "brand_name",
    "item_condition_id",
]

NUM_FEATURES = [
    "shipping",
    "desc_length",
    "name_length",
    "brand_tier",
    "category_price_median",
    "category_price_std",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
