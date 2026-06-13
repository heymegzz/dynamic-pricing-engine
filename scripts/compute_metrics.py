"""
Compute honest, verifiable metrics on the test set for inclusion in the README.
Run: venv/bin/python scripts/compute_metrics.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from src.model import load_artifacts
from src.elasticity import build_demand_curve
from src.optimizer import find_optimal_price

print("Loading artifacts and test set...")
model, encoders, category_stats = load_artifacts()
test_df = pd.read_parquet(config.PROCESSED_TEST)

y_true = test_df['price'].values
X_test = test_df[config.ALL_FEATURES]

# ─── Price prediction metrics ─────────────────────────────────────────
y_pred_log = model.predict(X_test)
y_pred_model = np.expm1(y_pred_log)

y_pred_baseline = test_df.groupby('category_main')['price'].transform('mean').values

rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
mae_base  = mean_absolute_error(y_true, y_pred_baseline)
rmse_model = np.sqrt(mean_squared_error(y_true, y_pred_model))
mae_model  = mean_absolute_error(y_true, y_pred_model)

print(f"\n=== Price Prediction Metrics (n={len(y_true):,}) ===")
print(f"Baseline RMSE: ${rmse_base:.2f}   MAE: ${mae_base:.2f}")
print(f"LightGBM RMSE: ${rmse_model:.2f}   MAE: ${mae_model:.2f}")
print(f"RMSE improvement: {(rmse_base - rmse_model) / rmse_base * 100:.1f}%")
print(f"MAE  improvement: {(mae_base  - mae_model)  / mae_base  * 100:.1f}%")

# ─── Optimizer vs naive baseline revenue ─────────────────────────────
# Use a sample of 500 rows for speed
SAMPLE = 500
sample_df = test_df.sample(SAMPLE, random_state=42)
sample_df = sample_df.reset_index(drop=True)

total_naive_revenue = 0.0
total_opt_revenue   = 0.0
opt_prices = []
naive_prices = []

print(f"\nRunning optimizer on {SAMPLE} sampled items (this may take ~30 s)...")
for _, row in sample_df.iterrows():
    feat_dict = row[config.ALL_FEATURES].to_dict()
    cat_median = float(row['category_price_median'])
    cat_std    = float(row['category_price_std'])

    curve = build_demand_curve(feat_dict, model, cat_median, cat_std)
    opt   = find_optimal_price(curve)

    naive_demand = np.interp(cat_median, curve['prices'], curve['demands'])
    naive_rev    = cat_median * naive_demand

    total_naive_revenue += naive_rev
    total_opt_revenue   += opt['optimal_revenue']
    opt_prices.append(opt['optimal_price'])
    naive_prices.append(cat_median)

opt_prices   = np.array(opt_prices)
naive_prices = np.array(naive_prices)
lift         = (total_opt_revenue - total_naive_revenue) / total_naive_revenue * 100
pct_diff     = (opt_prices - naive_prices) / naive_prices * 100

print(f"\n=== Optimizer vs Naive Baseline Revenue (n={SAMPLE}) ===")
print(f"Median optimal price:  ${np.median(opt_prices):.2f}")
print(f"Median baseline price: ${np.median(naive_prices):.2f}")
print(f"Median delta:          ${np.median(opt_prices - naive_prices):.2f}")
print(f"Median %  delta:       {np.median(pct_diff):.1f}%")
print(f"Revenue lift vs naive: {lift:.1f}%")
