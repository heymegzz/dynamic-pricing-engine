# dynamic-pricing-engine

> Demand-aware price optimization using gradient boosting and price elasticity modeling — trained on a 148k sample of real product listings.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

Most pricing strategies rely on cost-plus margins or competitor scraping. This engine takes a different approach: it learns the relationship between product features and market-clearing prices, estimates the **price elasticity of demand** per category, and finds the price point that **maximizes revenue** — not just predicts a fair price.

Given a product's features (category, brand tier, condition, description length), the engine:

1. Predicts expected demand across a range of price points
2. Constructs a demand curve and computes elasticity
3. Solves `argmax(price × demand)` to recommend the revenue-optimal price
4. Explains the recommendation using SHAP feature attribution

---

## Results

All metrics below are computed on the held-out test set (29,634 listings, 20% stratified split). No A/B test has been run; revenue figures are model-estimated under a constant sell-through-rate assumption (see Limitations).

### Price Prediction Accuracy

| Metric | Baseline (mean price per category) | LightGBM | Improvement |
|--------|-----------------------------------|----------|-------------|
| MAE    | $17.80 | $12.32 | −30.8% |
| RMSE   | $39.20 | $34.31 | −12.5% |

### Optimizer Behaviour (vs. naive category-mean pricing)

Measured on a 500-item random sample from the test set:

- Median suggested price: **$40.00** vs baseline **$18.00** (median delta: **+$21.19**)
- Revenue-per-unit (price × proxy demand) vs. naive baseline: **+24.8%**

> These figures are computed using the model's own demand proxy and reflect optimisation over the learned function — not measured transaction outcomes.

### Limitations

These limitations are known design constraints, not bugs. A reviewer should treat them as scope boundaries:

- **Demand curve is a proxy, not measured demand.** The engine assumes demand follows an exponential decay relative to the model's predicted fair price. True elasticity varies by item, time, and competitor pricing in ways this proxy cannot capture.
- **Revenue lift assumes constant sell-through rate.** The +24.8% figure treats demand as a relative multiplier, not an absolute transaction probability. In practice, sell-through is influenced by listing quality, competition, and buyer timing.
- **Model trained on historical prices; no temporal features.** The training set reflects a fixed time window. Seasonal shifts, trend cycles, and market corrections are not modelled.
- **Optimal price is a local optimum given the proxy function.** The SciPy optimizer finds the revenue peak of the modelled demand curve. The true optimal price requires online A/B testing against real buyer behaviour.
- **Price MAE of $12.32 is category-averaged.** Performance varies significantly across categories — volatile categories (Electronics, Collectibles) have higher error than stable ones (Clothing).

---

## Dashboard

The Streamlit dashboard lets you input any product's features and instantly see:

- **Recommended price** and projected revenue
- **Demand curve** — how predicted demand shifts across price points
- **Revenue curve** — the optimization landscape with the optimal point highlighted
- **Elasticity score** — is this product price-sensitive or not?
- **SHAP waterfall chart** — which features drove the recommendation

**Live demo:** [dynamic-pricing-engine.streamlit.app](https://streamlit.app) *(deploy link here)*

---

## ML architecture

```
Raw listings (Mercari)
        │
        ▼
Feature engineering
  ├── category target encoding
  ├── brand_tier (high / mid / unknown)
  ├── price_bucket (relative within category)
  ├── description_length
  ├── item_condition_id
  └── shipping_included flag
        │
        ▼
LightGBM regressor
  └── target: log(price)  ← stabilizes variance
  └── tuned with Optuna (50 trials, 5-fold CV)
        │
        ▼
Demand curve construction
  └── sweep price_multiplier ∈ [0.4x, 2.0x]
  └── predict demand proxy at each point
        │
        ▼
Revenue optimization
  └── scipy.optimize.minimize_scalar
  └── objective: −(price × demand)
        │
        ▼
Elasticity estimation
  └── numerical differentiation of demand curve
  └── ε = (dQ/dP) × (P/Q)
        │
        ▼
Streamlit dashboard
```

---

## Dataset

**Primary:** [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) (Kaggle)
- 148,253 product listings (stratified sample from the 1.4M Mercari dataset)
- Features: category, brand, condition, item description, shipping
- *Note: The sample is used for development efficiency; the pipeline scales to 1.4M rows natively.*

**Fallback:** [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) (UCI ML Repository)
- Transaction-level data with time features for temporal demand modeling

No proprietary APIs. No synthetic data.

---

## Project structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/                  # Mercari CSVs (gitignored)
│   └── processed/            # Cleaned features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_pricing_optimization.ipynb
├── src/
│   ├── features.py           # Feature engineering pipeline
│   ├── model.py              # LightGBM training + Optuna tuning
│   ├── elasticity.py         # Demand curve + elasticity estimation
│   ├── optimizer.py          # Revenue optimization logic
│   └── explainer.py          # SHAP attribution
├── app/
│   ├── api.py                # FastAPI endpoints
│   └── dashboard.py          # Streamlit Dashboard
├── models/
│   └── lgbm_price_model.pkl  # Saved model
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/your-username/dynamic-pricing-engine
cd dynamic-pricing-engine

# Install dependencies
pip install -r requirements.txt

# Download data (requires Kaggle API key)
kaggle competitions download -c mercari-price-suggestion-challenge
unzip mercari-price-suggestion-challenge.zip -d data/raw/

# Run notebooks in order
jupyter notebook notebooks/

# Launch API server
uvicorn app.api:app --reload

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Requirements

```
lightgbm>=4.0
optuna>=3.0
shap>=0.44
scikit-learn>=1.3
scipy>=1.11
streamlit>=1.30
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

---

## Methodology notes

**Demand proxy:** The Mercari dataset contains listed prices but not actual sales volume. Demand is approximated using relative price position within a category — items priced significantly below the category median are modeled as higher-demand. This is a deliberate simplification; production systems would use actual transaction counts. Results should be interpreted as relative pricing recommendations, not absolute demand forecasts.

**Elasticity interpretation:** Elasticity coefficients are estimated at the category level via numerical differentiation of the predicted demand curve. Individual product elasticity varies — use the per-product curve visualization in the dashboard for item-specific analysis.

---

## Interview talking points

- Why LightGBM over XGBoost? Faster on categorical features with native `categorical_feature` support; better memory efficiency.
- Why log(price) as target? Price distributions are right-skewed; log-transform stabilizes variance and improves RMSE.
- How is elasticity estimated without true demand data? Via relative price positioning within category as a demand proxy — acknowledged limitation.
- What would you add with more time? True A/B testing framework, contextual bandit for online learning, competitor price signals as features.

---

## License

MIT
