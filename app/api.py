"""
FastAPI application for the Dynamic Pricing Engine.

All feature construction is delegated to src.features.build_inference_row()
to guarantee consistency with the training pipeline (no training/serving skew).
"""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.model import load_artifacts
from src.features import build_inference_row
from src.elasticity import build_demand_curve
from src.optimizer import find_optimal_price

app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="API for optimizing prices using a trained LightGBM model.",
    version="2.1.0",
)

# Globals populated at startup — never reassigned per-request
model = None
encoders = None
category_stats = None


@app.on_event("startup")
def _load_resources():
    global model, encoders, category_stats
    try:
        model, encoders, category_stats = load_artifacts()
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"[WARN] Error loading artifacts on startup: {e}")


class PricingRequest(BaseModel):
    item_name: str = Field(default="Vintage T-Shirt", description="Product listing title")
    category_main: str = Field(default="Men", description="Top-level Mercari category")
    category_sub: str = Field(default="Tops & Shirts", description="Sub-category")
    category_leaf: str = Field(default="T-Shirts", description="Leaf-level category")
    brand_name: str = Field(default="Nike", description="Brand name (or 'unknown')")
    item_condition_id: int = Field(default=1, ge=1, le=5, description="1=New … 5=Poor")
    shipping: int = Field(default=0, ge=0, le=1, description="0=buyer pays, 1=seller pays")
    item_description: str = Field(default="", description="Full item description text")


def _check_artifacts():
    if model is None or encoders is None or category_stats is None:
        raise HTTPException(status_code=503, detail="Artifacts not yet loaded.")


@app.post("/optimize")
def optimize_price(request: PricingRequest):
    """
    Return the revenue-optimal price for the described item.
    """
    _check_artifacts()
    try:
        # Single canonical call — no manual feature construction here
        X = build_inference_row(request.model_dump(), encoders, category_stats)
        # Extract category stats produced during inference-row construction
        # (they live at fixed positions in ALL_FEATURES)
        feat_dict = {k: float(X[0, i]) for i, k in enumerate(config.ALL_FEATURES)}
        cat_median = feat_dict.get("category_price_median", 25.0)
        cat_std    = feat_dict.get("category_price_std",    15.0)

        curve = build_demand_curve(
            item_features=feat_dict,
            model=model,
            category_median=cat_median,
            category_std=cat_std,
            elasticity_k=1.0,
        )
        result = find_optimal_price(curve)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict_price(request: PricingRequest):
    """
    Return the raw model price prediction (no optimization).
    Also returns optimal_price and expected_revenue from the demand curve.
    """
    _check_artifacts()
    try:
        X = build_inference_row(request.model_dump(), encoders, category_stats)
        feat_dict = {k: float(X[0, i]) for i, k in enumerate(config.ALL_FEATURES)}
        cat_median = feat_dict.get("category_price_median", 25.0)
        cat_std    = feat_dict.get("category_price_std",    15.0)

        predicted_log_price = float(model.predict(X)[0])
        predicted_price = float(np.expm1(predicted_log_price))

        curve  = build_demand_curve(feat_dict, model, cat_median, cat_std, elasticity_k=1.0)
        result = find_optimal_price(curve)

        return {
            "predicted_price":   predicted_price,
            "optimal_price":     result["optimal_price"],
            "expected_revenue":  result["optimal_revenue"],
            "elasticity":        result["elasticity"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status":               "ok",
        "model_loaded":         model is not None,
        "encoders_loaded":      encoders is not None,
        "category_stats_loaded": category_stats is not None,
    }
