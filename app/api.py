"""
FastAPI application for the Dynamic Pricing Engine.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_FILE, PROCESSED_TRAIN, ALL_FEATURES
from src.elasticity import load_model, build_demand_curve
from src.optimizer import find_optimal_price

app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="API for optimizing prices using a trained LightGBM model.",
    version="2.0.0"
)

# Load resources on startup
model = None
train_df = None

@app.on_event("startup")
def load_resources():
    global model, train_df
    try:
        model = load_model(MODEL_FILE)
        train_df = pd.read_parquet(PROCESSED_TRAIN)
        print("Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

class PricingRequest(BaseModel):
    market_base_price: float = 25.0
    market_volatility: float = 15.0
    item_condition_id: int = 1
    shipping: int = 0
    desc_length: int = 50
    name_length: int = 20
    brand_tier: int = 0

@app.post("/optimize")
def optimize_price(request: PricingRequest):
    if model is None or train_df is None:
        raise HTTPException(status_code=500, detail="Models or data not loaded.")
        
    try:
        item_dict = request.dict()
        
        cat_median = item_dict.pop("market_base_price")
        cat_std = item_dict.pop("market_volatility")
        
        feature_row = {}
        for k in ALL_FEATURES:
            val = item_dict.get(k, 0)
            if isinstance(val, str):
                # Fallback for target encoded variables that were dropped
                val = float(train_df[k].mean()) if k in train_df.columns else 0.0
            feature_row[k] = val
        
        demand_curve = build_demand_curve(
            item_features=feature_row,
            model=model,
            category_median=cat_median,
            category_std=cat_std
        )
        
        opt_result = find_optimal_price(demand_curve)
        return opt_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
