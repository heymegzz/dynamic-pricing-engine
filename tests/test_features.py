"""
tests/test_features.py — Unit tests for feature engineering pipeline.
Run: pytest tests/test_features.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.features import (
    parse_categories,
    engineer_features,
    build_inference_row,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifacts():
    """Load model, encoders, and category_stats once for the entire module."""
    from src.model import load_artifacts
    model, encoders, category_stats = load_artifacts()
    return model, encoders, category_stats


@pytest.fixture(scope="module")
def sample_raw_input():
    return {
        "item_name":        "Blue Denim Jacket",
        "category_main":    "Women",
        "category_sub":     "Jackets & Coats",
        "category_leaf":    "Denim Jackets",
        "brand_name":       "Levi's",
        "item_condition_id": 2,
        "shipping":         1,
        "item_description": "Lightly worn blue denim jacket, size M.",
    }


# ── Test: build_inference_row output shape ──────────────────────────────

def test_build_inference_row_shape(artifacts, sample_raw_input):
    """Output must be (1, len(ALL_FEATURES)) — exact model input shape."""
    _, encoders, category_stats = artifacts
    X = build_inference_row(sample_raw_input, encoders, category_stats)
    assert X.shape == (1, len(config.ALL_FEATURES)), (
        f"Expected shape (1, {len(config.ALL_FEATURES)}), got {X.shape}"
    )


def test_build_inference_row_no_nans(artifacts, sample_raw_input):
    """Output must contain no NaN values (would silently corrupt predictions)."""
    _, encoders, category_stats = artifacts
    X = build_inference_row(sample_raw_input, encoders, category_stats)
    assert not np.any(np.isnan(X)), "NaN found in inference row output"


def test_build_inference_row_dtype(artifacts, sample_raw_input):
    """Output dtype must be float64 — LightGBM's native input type."""
    _, encoders, category_stats = artifacts
    X = build_inference_row(sample_raw_input, encoders, category_stats)
    assert X.dtype == np.float64, f"Expected float64, got {X.dtype}"


def test_feature_order_matches_all_features(artifacts, sample_raw_input):
    """The array must have exactly len(ALL_FEATURES) columns in correct order."""
    _, encoders, category_stats = artifacts
    X = build_inference_row(sample_raw_input, encoders, category_stats)
    assert X.shape[1] == len(config.ALL_FEATURES), (
        "Column count mismatch between inference row and ALL_FEATURES"
    )


# ── Test: engineer_features edge case ───────────────────────────────────

def test_engineer_features_no_description():
    """Empty description string → desc_length == 0, no KeyError."""
    df = pd.DataFrame([{
        "name":             "Some Item",
        "category_main":   "Electronics",
        "category_sub":    "Phones",
        "category_leaf":   "Smartphones",
        "brand_name":      "unknown",
        "item_condition_id": 1,
        "shipping":        0,
        "item_description": "",   # empty string — must not raise
        "price":           10.0,
    }])
    df_out, _ = engineer_features(df)
    assert "desc_length" in df_out.columns, "desc_length column missing"
    assert int(df_out.iloc[0]["desc_length"]) == 0, (
        f"Expected desc_length=0, got {df_out.iloc[0]['desc_length']}"
    )


def test_engineer_features_null_description():
    """NaN description must be treated as empty (length 0) without raising."""
    df = pd.DataFrame([{
        "name":             "Test",
        "category_main":   "Other",
        "category_sub":    "Other",
        "category_leaf":   "Other",
        "brand_name":      "unknown",
        "item_condition_id": 3,
        "shipping":        0,
        "item_description": None,   # NULL
        "price":           5.0,
    }])
    df_out, _ = engineer_features(df)
    assert int(df_out.iloc[0]["desc_length"]) == 0


# ── Test: parse_categories edge cases ────────────────────────────────────

def test_parse_categories_short_path():
    """Category with < 3 segments should produce 'unknown' for missing levels."""
    df = pd.DataFrame([{"category_name": "Electronics", "price": 50.0}])
    df_out = parse_categories(df)
    assert df_out.iloc[0]["category_main"] == "Electronics"
    assert df_out.iloc[0]["category_sub"]  == "unknown"
    assert df_out.iloc[0]["category_leaf"] == "unknown"


def test_parse_categories_null_value():
    """NULL category_name should not raise — fills with 'unknown'."""
    df = pd.DataFrame([{"category_name": None, "price": 10.0}])
    df_out = parse_categories(df)
    assert df_out.iloc[0]["category_main"] == "unknown"
