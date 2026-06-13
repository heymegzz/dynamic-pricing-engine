"""
Price optimization logic.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_FILE, PROCESSED_TRAIN, ALL_FEATURES
from src.elasticity import load_model, build_demand_curve, estimate_elasticity, get_category_stats

def find_optimal_price(demand_curve: dict) -> dict:
    """
    Find the price that maximizes revenue given a demand curve.
    Uses both direct argmax and scipy optimization.
    
    Args:
        demand_curve (dict): Dictionary returned by build_demand_curve.
        
    Returns:
        dict: Result with optimal price, revenue, demand, multiplier, and elasticity.
    """
    prices = demand_curve["prices"]
    demands = demand_curve["demands"]
    revenues = demand_curve["revenues"]
    multipliers = demand_curve["multipliers"]
    
    # Direct method
    best_idx = np.argmax(revenues)
    direct_optimal_price = prices[best_idx]
    
    # SciPy method
    min_price = min(prices)
    max_price = max(prices)
    base_price = prices[0] / multipliers[0]
    
    # Interpolate the discrete demand points to create a continuous demand function
    demand_func = interp1d(prices, demands, kind='linear', bounds_error=False, fill_value=(demands[0], demands[-1]))
    
    def negative_revenue(p):
        return -float(p * demand_func(p))
        
    res = minimize_scalar(negative_revenue, bounds=(min_price, max_price), method='bounded')
    
    optimal_price = float(res.x)
    optimal_demand = float(demand_func(optimal_price))
    optimal_revenue = optimal_price * optimal_demand
    optimal_multiplier = optimal_price / base_price
    
    elasticity = estimate_elasticity(prices, demands)
    
    return {
        "optimal_price": optimal_price,
        "optimal_revenue": optimal_revenue,
        "optimal_demand": optimal_demand,
        "optimal_multiplier": optimal_multiplier,
        "elasticity": elasticity
    }

def run_optimization_demo(n_samples=5):
    """
    Run the price optimization on a few random samples from the training set.
    
    Args:
        n_samples (int): Number of random samples to run.
        
    Returns:
        list: List of result dictionaries for each sample.
    """
    model = load_model(MODEL_FILE)
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    
    sample_df = train_df.sample(n=n_samples, random_state=42)
    results = []
    
    for idx, row in sample_df.iterrows():
        category = row.get("category_main")
        
        # Use get_category_stats with global stats fallback
        cat_median, cat_std = get_category_stats(train_df, str(category))
        
        item_features = {k: row.get(k, 0) for k in ALL_FEATURES}
        
        demand_curve = build_demand_curve(
            item_features=item_features,
            model=model,
            category_median=cat_median,
            category_std=cat_std
        )
        
        opt_result = find_optimal_price(demand_curve)
        
        print(f"Row {idx}: Optimal Price: {opt_result['optimal_price']:.2f}, "
              f"Revenue: {opt_result['optimal_revenue']:.2f}, "
              f"Elasticity: {opt_result['elasticity']:.3f}, "
              f"Multiplier: {opt_result['optimal_multiplier']:.2f}")
              
        opt_result["row_index"] = idx
        results.append(opt_result)
        
    return results

if __name__ == "__main__":
    run_optimization_demo()
