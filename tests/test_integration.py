"""
tests/test_integration.py — End-to-end integration tests.
Requires trained artifacts in models/ and processed data in data/processed/.
Run: pytest tests/test_integration.py -v
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifacts():
    from src.model import load_artifacts
    return load_artifacts()  # (model, encoders, category_stats)


@pytest.fixture(scope="module")
def sample_raw():
    return {
        "item_name":        "Red Floral Sundress",
        "category_main":    "Women",
        "category_sub":     "Dresses",
        "category_leaf":    "Sun Dresses",
        "brand_name":       "H&M",
        "item_condition_id": 1,
        "shipping":         0,
        "item_description": "Brand new with tags. Size S. Beautiful red floral pattern.",
    }


@pytest.fixture(scope="module")
def api_client():
    """TestClient wrapping the FastAPI app with mocked startup."""
    from fastapi.testclient import TestClient
    from app.api import app, _load_resources

    # Manually trigger startup (TestClient doesn't call lifespan handlers in older versions)
    _load_resources()

    return TestClient(app)


# ── Test: full pipeline ──────────────────────────────────────────────────

def test_full_pipeline(artifacts, sample_raw):
    """
    Load artifacts → build_inference_row → model.predict → build_demand_curve
    → optimal_price is a finite positive float between $1 and $1000.
    """
    from src.features import build_inference_row
    from src.elasticity import build_demand_curve
    from src.optimizer import find_optimal_price

    model, encoders, category_stats = artifacts

    # Step 1: build feature row
    X = build_inference_row(sample_raw, encoders, category_stats)
    assert X.shape == (1, len(config.ALL_FEATURES))
    assert not np.any(np.isnan(X))

    # Step 2: raw model predict
    log_pred = model.predict(X)
    assert np.isfinite(log_pred[0])

    feat_dict  = {k: float(X[0, i]) for i, k in enumerate(config.ALL_FEATURES)}
    cat_median = feat_dict.get("category_price_median", 25.0)
    cat_std    = feat_dict.get("category_price_std",    15.0)

    # Step 3: demand curve
    curve = build_demand_curve(feat_dict, model, cat_median, cat_std)
    assert len(curve["prices"]) > 0

    # Step 4: optimise
    result = find_optimal_price(curve)
    opt_p  = result["optimal_price"]

    assert math.isfinite(opt_p),   "optimal_price is not finite"
    assert opt_p > 1.0,            f"optimal_price too low: {opt_p}"
    assert opt_p < 1000.0,         f"optimal_price unrealistically high: {opt_p}"


def test_api_predict_endpoint(api_client):
    """
    POST to /predict with a valid item.
    Response must be 200 with optimal_price and expected_revenue as positive numbers.
    """
    payload = {
        "item_name":        "Vintage Levi Jeans",
        "category_main":    "Men",
        "category_sub":     "Jeans",
        "category_leaf":    "Straight",
        "brand_name":       "Levi's",
        "item_condition_id": 2,
        "shipping":         1,
        "item_description": "Classic 501 Levi jeans. Size 32x32. Light wash.",
    }

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200, (
        f"Expected 200 OK, got {response.status_code}: {response.text}"
    )

    data = response.json()
    assert "optimal_price"    in data, "Response missing 'optimal_price'"
    assert "expected_revenue" in data, "Response missing 'expected_revenue'"

    assert data["optimal_price"]    > 0, "optimal_price must be positive"
    assert data["expected_revenue"] > 0, "expected_revenue must be positive"
    assert math.isfinite(data["optimal_price"]),    "optimal_price is not finite"
    assert math.isfinite(data["expected_revenue"]), "expected_revenue is not finite"


def test_api_health_endpoint(api_client):
    """Health check must return 200 with model_loaded=True."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("model_loaded") is True
    assert data.get("encoders_loaded") is True


def test_api_invalid_condition_rejected(api_client):
    """item_condition_id outside 1-5 must be rejected with 422."""
    payload = {
        "item_name":        "Test",
        "category_main":    "Other",
        "item_condition_id": 99,  # invalid
        "shipping":         0,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422, (
        "Expected Pydantic validation error (422) for condition=99"
    )
