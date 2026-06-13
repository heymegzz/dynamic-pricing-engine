"""
FastAPI application for the Dynamic Pricing Engine.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.model import load_artifacts
from src.elasticity import build_demand_curve
from src.optimizer import find_optimal_price
from src.features import parse_categories, engineer_features

app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="API for optimizing prices using a trained LightGBM model.",
    version="2.0.0"
)

# Load resources on startup
model = None
encoders = None
category_stats = None

@app.on_event("startup")
def load_resources():
    global model, encoders, category_stats
    try:
        model, encoders, category_stats = load_artifacts()
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class PricingRequest(BaseModel):
    item_name: str = "Vintage T-Shirt"
    category_main: str = "Men"
    brand_name: str = "Nike"
    item_condition_id: int = 1
    shipping: int = 0
    description_length: int = 50

def preprocess_request(item_dict: dict) -> dict:
    """
    Preprocess raw input into the feature array expected by the model.
    """
    # 1. Map to raw DataFrame format expected by features.py
    raw_row = {
        "name": item_dict.get("item_name", ""),
        "category_name": item_dict.get("category_main", "unknown") + "//",
        "brand_name": item_dict.get("brand_name", "unknown"),
        "item_condition_id": item_dict.get("item_condition_id", 1),
        "shipping": item_dict.get("shipping", 0),
        "item_description": "x" * item_dict.get("description_length", 0),
        "price": 0.0 # Dummy price
    }
    df = pd.DataFrame([raw_row])
    
    # a. Call parse_categories and engineer_features
    from src.features import encode_categoricals
    df = parse_categories(df)
    df, _ = engineer_features(df, category_stats=category_stats)
    
    # b. Apply each encoder using encode_categoricals
    df, _ = encode_categoricals(df, config.CAT_FEATURES, target_col="log_price", fit=False, encoders=encoders)
    
    row = df.iloc[0].to_dict()
    
    # c. Build the final dictionary in ALL_FEATURES column order
    feature_row = {k: float(row.get(k, 0)) for k in config.ALL_FEATURES}
    return feature_row

@app.post("/optimize")
def optimize_price(request: PricingRequest):
    if model is None or encoders is None or category_stats is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded.")
        
    try:
        item_dict = request.dict()
        
        # Preprocess request to get feature row
        feature_row = preprocess_request(item_dict)
        
        # We need category_median and category_std for the demand curve
        # They were computed by engineer_features and stored in the row
        cat_median = feature_row.get("category_price_median", 25.0)
        cat_std = feature_row.get("category_price_std", 15.0)
        
        demand_curve = build_demand_curve(
            item_features=feature_row,
            model=model,
            category_median=cat_median,
            category_std=cat_std,
            elasticity_k=1.0
        )
        
        opt_result = find_optimal_price(demand_curve)
        return opt_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None,
        "category_stats_loaded": category_stats is not None
    }
