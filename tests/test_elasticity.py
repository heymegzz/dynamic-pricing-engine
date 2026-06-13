import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.elasticity import estimate_elasticity, get_category_stats

def test_estimate_elasticity():
    prices = [10.0, 11.0, 12.0]
    demands = [100.0, 90.0, 80.0]
    # dP = 2.0, dQ = -20.0
    # mid = 1, P_mid = 11.0, Q_mid = 90.0
    # elasticity = (-20 / 2) * (11 / 90) = -10 * 0.1222 = -1.2222
    
    elasticity = estimate_elasticity(prices, demands)
    assert np.isclose(elasticity, -1.2222222222222223)

def test_estimate_elasticity_zero_div():
    prices = [10.0, 10.0, 10.0]
    demands = [100.0, 100.0, 100.0]
    elasticity = estimate_elasticity(prices, demands)
    assert elasticity == 0.0

def test_get_category_stats():
    df = pd.DataFrame({
        "category_main": ["A", "A", "B"],
        "price": [10.0, 20.0, 30.0]
    })
    
    med, std = get_category_stats(df, "A")
    assert med == 15.0
    assert np.isclose(std, np.std([10.0, 20.0], ddof=1))
    
    # Fallback testing
    med_fallback, std_fallback = get_category_stats(df, "C")
    assert med_fallback == 20.0
    assert np.isclose(std_fallback, df["price"].std())
