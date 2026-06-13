"""
Demand curve construction and elasticity estimation.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

# Add the project root to sys.path so config can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PRICE_SWEEP_MIN, PRICE_SWEEP_MAX, PRICE_SWEEP_STEPS, ALL_FEATURES

def load_model(model_path):
    """
    Load pickle model and return it.
    
    Args:
        model_path (str or Path): Path to the trained model pickle file.
        
    Returns:
        The loaded model object.
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def build_demand_curve(item_features: dict, model, category_median: float, category_std: float, price_multipliers=None, elasticity_k: float = 1.0) -> dict:
    """
    Build a demand curve by sweeping prices and estimating demand based on a model-predicted fair price.
    
    Demand proxy logic:
    The model predicts a 'fair market price' based on the item's features. 
    We approximate demand by assuming that pricing above this fair price decays demand exponentially,
    and pricing below it boosts demand exponentially, scaled by the category's volatility (std) 
    and a tunable elasticity parameter (elasticity_k).
    
    Args:
        item_features (dict): Dictionary with all ALL_FEATURES keys.
        model: Trained pricing model (predicts log_price).
        category_median (float): Median price for the item's category.
        category_std (float): Standard deviation of price for the item's category.
        price_multipliers (np.array, optional): Array of multipliers. Defaults to config constants.
        elasticity_k (float, optional): Tunable elasticity parameter. Defaults to 1.0.
        
    Returns:
        dict: A dictionary containing lists of "prices", "demands", "revenues", and "multipliers".
    """
    if price_multipliers is None:
        price_multipliers = np.linspace(PRICE_SWEEP_MIN, PRICE_SWEEP_MAX, PRICE_SWEEP_STEPS)
        
    base_price = category_median
    
    prices = []
    demands = []
    revenues = []
    multipliers = []
    
    for m in price_multipliers:
        price = base_price * m
        
        # Create a copy to not modify the input dict in-place
        item_features_copy = item_features.copy()
        item_features_copy["price"] = price  # Set current sweep price
        
        feature_row = {k: item_features_copy.get(k, 0) for k in ALL_FEATURES}
        df_features = pd.DataFrame([feature_row])
        
        # The predicted log_price from the model
        pred_log_price = model.predict(df_features)[0]
        
        # Compute fair price estimate from model prediction
        fair_price = np.expm1(pred_log_price)
        
        # Compute demand based on deviation from fair price
        demand = np.exp(-elasticity_k * (price - fair_price) / max(category_std, 1.0))
        
        # Clamp demand to range [0.01, 10.0]
        demand = np.clip(demand, 0.01, 10.0)
        
        revenue = price * demand
        
        prices.append(price)
        demands.append(demand)
        revenues.append(revenue)
        multipliers.append(m)
        
    return {
        "prices": prices,
        "demands": demands,
        "revenues": revenues,
        "multipliers": multipliers
    }

def estimate_elasticity(prices: list, demands: list) -> float:
    """
    Compute point elasticity at the midpoint of the curve using numerical differentiation.
    
    Args:
        prices (list): List of prices.
        demands (list): List of corresponding demands.
        
    Returns:
        float: Estimated point elasticity.
    """
    if len(prices) < 3:
        return 0.0
        
    mid = len(prices) // 2
    
    # Central difference: dQ/dP ≈ (Q[mid+1] - Q[mid-1]) / (P[mid+1] - P[mid-1])
    dP = prices[mid+1] - prices[mid-1]
    dQ = demands[mid+1] - demands[mid-1]
    
    if dP == 0:
        return 0.0
        
    P_mid = prices[mid]
    Q_mid = demands[mid]
    
    if Q_mid == 0:
        return 0.0
        
    elasticity = (dQ / dP) * (P_mid / Q_mid)
    return float(elasticity)

def get_category_stats(train_df: pd.DataFrame, category: str) -> tuple:
    """
    Calculate the median and std of price for a given category.
    Falls back to global stats if the category is not found.
    
    Args:
        train_df (pd.DataFrame): Training data containing the "price" column.
        category (str): The category name.
        
    Returns:
        tuple: (median, std) of prices.
    """
    global_median = float(train_df["price"].median())
    global_std = float(train_df["price"].std())
    
    if "category_main" in train_df.columns:
        cat_df = train_df[train_df["category_main"] == category]
        if not cat_df.empty:
            cat_median = float(cat_df["price"].median())
            cat_std = float(cat_df["price"].std())
            
            if pd.isna(cat_std):
                cat_std = global_std
                
            return cat_median, cat_std
            
    return global_median, global_std
