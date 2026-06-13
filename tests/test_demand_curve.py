"""
tests/test_demand_curve.py — Unit tests for build_demand_curve().
Run: pytest tests/test_demand_curve.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.elasticity import build_demand_curve
from src.optimizer import find_optimal_price


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_mock_model(log_price_return: float):
    """Return a mock LightGBM-like model that always predicts log_price_return."""
    mock = MagicMock()
    mock.predict.return_value = np.array([log_price_return])
    return mock


def _default_features() -> dict:
    """Minimal feature dict with all ALL_FEATURES keys set to 0."""
    return {k: 0.0 for k in config.ALL_FEATURES}


# ── Tests ────────────────────────────────────────────────────────────────

def test_demand_curve_uses_model():
    """
    Changing the mock model's return value MUST change the resulting optimal_price.
    This proves the model actually drives the curve (not a hardcoded formula).
    """
    feats = _default_features()
    cat_median, cat_std = 20.0, 10.0

    # Model A: fair_price = expm1(log(10+1)) ≈ $10
    model_low = _make_mock_model(np.log1p(10.0))
    curve_low = build_demand_curve(feats, model_low, cat_median, cat_std)
    opt_low   = find_optimal_price(curve_low)["optimal_price"]

    # Model B: fair_price = expm1(log(50+1)) ≈ $50
    model_high = _make_mock_model(np.log1p(50.0))
    curve_high = build_demand_curve(feats, model_high, cat_median, cat_std)
    opt_high   = find_optimal_price(curve_high)["optimal_price"]

    assert opt_high != opt_low, (
        "Changing model prediction had no effect on optimal_price — "
        "the model is not being used to drive the demand curve."
    )


def test_demand_curve_structure():
    """
    Returned dict must have lists 'prices', 'demands', 'revenues'.
    All values must be finite and positive; prices must be monotonically increasing.
    """
    model = _make_mock_model(np.log1p(25.0))
    curve = build_demand_curve(_default_features(), model, 25.0, 10.0)

    assert "prices"   in curve
    assert "demands"  in curve
    assert "revenues" in curve

    prices   = np.array(curve["prices"])
    demands  = np.array(curve["demands"])
    revenues = np.array(curve["revenues"])

    assert np.all(np.isfinite(prices)),   "Non-finite value in prices"
    assert np.all(np.isfinite(demands)),  "Non-finite value in demands"
    assert np.all(np.isfinite(revenues)), "Non-finite value in revenues"

    assert np.all(prices   > 0), "Non-positive price in curve"
    assert np.all(demands  > 0), "Non-positive demand in curve"
    assert np.all(revenues > 0), "Non-positive revenue in curve"

    # Prices must be monotonically increasing
    assert np.all(np.diff(prices) > 0), "Prices are not monotonically increasing"


def test_demand_curve_edge_zero_std():
    """
    category_std = 0 must NOT raise ZeroDivisionError.
    The implementation should use max(category_std, 1.0) as a fallback.
    """
    model = _make_mock_model(np.log1p(20.0))
    try:
        curve = build_demand_curve(_default_features(), model, 20.0, category_std=0.0)
        assert len(curve["prices"]) > 0
    except ZeroDivisionError:
        pytest.fail("build_demand_curve raised ZeroDivisionError with category_std=0")


def test_demand_curve_single_category():
    """
    Category with only one training example produces a near-zero std.
    The curve should still be computable and contain at least 2 price points.
    """
    model = _make_mock_model(np.log1p(15.0))
    # std=0.001 mimics a single-item category
    curve = build_demand_curve(_default_features(), model, 15.0, category_std=0.001)
    assert len(curve["prices"]) >= 2, (
        "Expected at least 2 price sweep points for single-category item"
    )
    assert np.all(np.isfinite(curve["demands"])), (
        "Non-finite demand values for single-category item"
    )


def test_demand_curve_elasticity_k_changes_shape():
    """
    A higher elasticity_k must produce steeper demand decay.
    Peak demand relative to trough should be greater for k=2 than k=0.5.
    """
    model = _make_mock_model(np.log1p(25.0))
    feats = _default_features()

    curve_flat  = build_demand_curve(feats, model, 25.0, 10.0, elasticity_k=0.5)
    curve_steep = build_demand_curve(feats, model, 25.0, 10.0, elasticity_k=2.0)

    ratio_flat  = max(curve_flat["demands"])  / (min(curve_flat["demands"])  + 1e-9)
    ratio_steep = max(curve_steep["demands"]) / (min(curve_steep["demands"]) + 1e-9)

    assert ratio_steep > ratio_flat, (
        "Higher elasticity_k should produce a steeper demand curve"
    )
