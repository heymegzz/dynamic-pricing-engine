"""
Streamlit Dashboard for Dynamic Pricing Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_FILE, PROCESSED_TRAIN, ALL_FEATURES
from src.elasticity import load_model, build_demand_curve, get_category_stats
from src.optimizer import find_optimal_price

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="📈", layout="wide")

# Custom CSS for a sleek, modern UI
st.markdown("""
<style>
    .stApp {
        background-color: #0f1115;
        color: #f0f2f6;
    }
    .metric-card {
        background-color: #1e2129;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2ecc71;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #a0aabf;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #f39c12; }
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model(MODEL_FILE)
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    return model, train_df

st.title("Dynamic Pricing Engine")
st.markdown("Optimize your revenue with machine-learning driven price elasticity modeling. Adjust product parameters below to see real-time shifts in demand and market-clearing prices.")

try:
    model, train_df = load_resources()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()

st.sidebar.markdown("### ⚙️ Item Configurations")

with st.sidebar.expander("Product Details", expanded=True):
    category_main = st.text_input("Main Category", value="Women")
    brand_name = st.text_input("Brand Name", value="Nike")
    item_condition_id = st.slider("Condition (1=New, 5=Poor)", 1, 5, 1)

with st.sidebar.expander("Listing Specs", expanded=True):
    shipping = st.selectbox("Shipping (1=Seller pays, 0=Buyer pays)", [0, 1], index=1)
    desc_length = st.number_input("Description Length (chars)", value=50, step=10)
    name_length = st.number_input("Name Length (chars)", value=20, step=5)
    brand_tier = st.number_input("Brand Tier (1=Top, 0=Other)", value=1, min_value=0, max_value=1)

# Main optimization logic
item_features = {
    "category_main": category_main,
    "category_sub": "",
    "category_leaf": "",
    "brand_name": brand_name,
    "item_condition_id": item_condition_id,
    "shipping": shipping,
    "desc_length": desc_length,
    "name_length": name_length,
    "brand_tier": brand_tier,
    "category_price_median": 0.0,
    "category_price_std": 0.0
}

# Fill remaining features with defaults to match ALL_FEATURES length
for f in ALL_FEATURES:
    val = item_features.get(f, 0)
    if isinstance(val, str):
        val = float(train_df[f].mean()) if f in train_df.columns else 0.0
    item_features[f] = val
        
cat_median, cat_std = get_category_stats(train_df, category_main)

demand_curve = build_demand_curve(
    item_features=item_features,
    model=model,
    category_median=cat_median,
    category_std=cat_std
)

opt_result = find_optimal_price(demand_curve)

# Calculate elasticity array for the curve
# Use np.gradient for continuous elasticity
prices_arr = np.array(demand_curve["prices"])
demands_arr = np.array(demand_curve["demands"])
dq_dp = np.gradient(demands_arr, prices_arr)
elasticity_curve = dq_dp * (prices_arr / demands_arr)

st.markdown("---")
st.markdown("### 📊 Pricing Recommendations")

col1, col2, col3, col4 = st.columns(4)

# Logic for deltas
price_diff = opt_result['optimal_price'] - cat_median
price_diff_pct = (price_diff / cat_median) * 100 if cat_median > 0 else 0

elasticity_val = opt_result['elasticity']
if elasticity_val > -1:
    elasticity_status = "Inelastic (Raise Price)"
    e_color = "positive"
else:
    elasticity_status = "Elastic (Drop Price)"
    e_color = "negative"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Optimal Price</div>
        <div class="metric-value">${opt_result['optimal_price']:.2f}</div>
        <div class="metric-delta {'positive' if price_diff >= 0 else 'negative'}">
            {'+' if price_diff >= 0 else ''}{price_diff_pct:.1f}% vs Category Median
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Expected Revenue</div>
        <div class="metric-value" style="color: #3498db;">${opt_result['optimal_revenue']:.2f}</div>
        <div class="metric-delta neutral">Yield Maximized Point</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Demand Likelihood</div>
        <div class="metric-value" style="color: #9b59b6;">{opt_result['optimal_demand']:.2f}x</div>
        <div class="metric-delta neutral">Relative to Median Item</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Point Elasticity</div>
        <div class="metric-value" style="color: {'#2ecc71' if elasticity_val > -1 else '#e74c3c'};">{elasticity_val:.2f}</div>
        <div class="metric-delta {e_color}">{elasticity_status}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Visualizations ──
st.markdown("### 📈 Optimization Curves")

c1, c2 = st.columns(2)

with c1:
    # Revenue Optimization Curve
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=prices_arr, y=demand_curve["revenues"],
        mode='lines',
        name='Revenue',
        line=dict(color='#3498db', width=3),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig_rev.add_trace(go.Scatter(
        x=[opt_result["optimal_price"]], 
        y=[opt_result["optimal_revenue"]],
        mode='markers+text',
        name='Optimal Point',
        marker=dict(color='#e74c3c', size=12, symbol='star'),
        text=[f"Optimal: ${opt_result['optimal_price']:.2f}"],
        textposition="top center",
        textfont=dict(color="#f0f2f6")
    ))

    fig_rev.update_layout(
        title="Revenue vs. Price Strategy",
        xaxis_title="Price ($)",
        yaxis_title="Expected Revenue ($)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_rev, use_container_width=True)

with c2:
    # Demand Curve
    fig_dem = go.Figure()
    fig_dem.add_trace(go.Scatter(
        x=prices_arr, y=demands_arr,
        mode='lines',
        name='Demand',
        line=dict(color='#9b59b6', width=3),
        fill='tozeroy',
        fillcolor='rgba(155, 89, 182, 0.2)'
    ))

    fig_dem.update_layout(
        title="Estimated Demand Curve",
        xaxis_title="Price ($)",
        yaxis_title="Demand Multiplier",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_dem, use_container_width=True)

st.markdown("---")
st.markdown("### 🧠 Market Insights")

c3, c4 = st.columns(2)

with c3:
    # Elasticity Curve
    fig_elas = go.Figure()
    fig_elas.add_trace(go.Scatter(
        x=prices_arr, y=elasticity_curve,
        mode='lines',
        name='Elasticity',
        line=dict(color='#f39c12', width=3)
    ))
    
    # Add horizontal line at -1 (Unit Elasticity)
    fig_elas.add_hline(y=-1, line_dash="dash", line_color="white", annotation_text="Unit Elastic (E = -1)")
    
    # Shade inelastic area (E > -1)
    fig_elas.add_hrect(y0=-1, y1=max(max(elasticity_curve), 0), line_width=0, fillcolor="rgba(46, 204, 113, 0.1)", annotation_text="Inelastic Zone")

    fig_elas.update_layout(
        title="Price Elasticity Sensitivity",
        xaxis_title="Price ($)",
        yaxis_title="Point Elasticity",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_elas, use_container_width=True)

with c4:
    # Market Context Spider / Bar Chart
    # Compare this item to global means
    global_desc_mean = train_df['desc_length'].mean() if 'desc_length' in train_df.columns else 50
    global_name_mean = train_df['name_length'].mean() if 'name_length' in train_df.columns else 20
    
    features = ['Desc Length', 'Name Length', 'Brand Tier']
    item_vals = [desc_length, name_length, brand_tier * 50] # Scale brand tier for visual balance
    global_vals = [global_desc_mean, global_name_mean, 50] # Assuming average brand tier is 0.5 scaled
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=global_vals,
        theta=features,
        fill='toself',
        name='Global Average',
        line_color='rgba(255, 255, 255, 0.5)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=item_vals,
        theta=features,
        fill='toself',
        name='This Item',
        line_color='#2ecc71'
    ))

    fig_radar.update_layout(
        title="Item Context vs. Market Average",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(max(item_vals), max(global_vals)) + 10])
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
