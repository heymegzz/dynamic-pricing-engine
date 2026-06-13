import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.optimizer import find_optimal_price

def test_find_optimal_price():
    # Construct a simple dummy demand curve
    prices = [10.0, 15.0, 20.0, 25.0, 30.0]
    demands = [100.0, 80.0, 50.0, 20.0, 5.0]
    revenues = [p * d for p, d in zip(prices, demands)]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    demand_curve = {
        "prices": prices,
        "demands": demands,
        "revenues": revenues,
        "multipliers": multipliers
    }
    
    res = find_optimal_price(demand_curve)
    
    # Direct max revenue is at price=15 (15*80=1200)
    # The optimization should find a value close to 15 or slightly off due to interpolation
    assert 13.0 <= res["optimal_price"] <= 17.0
    assert res["optimal_revenue"] >= 1000.0
    assert "elasticity" in res
