"""
Streamlit Dashboard for Dynamic Pricing Engine.
Refactored to look like a premium SaaS analytics platform (Stripe/Linear style).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_FILE, PROCESSED_TRAIN, ALL_FEATURES
from src.elasticity import load_model, build_demand_curve
from src.optimizer import find_optimal_price

st.set_page_config(page_title="Aura Pricing · Dynamic Pricing Engine", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Premium SaaS Look
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #09090b !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Navbar */
    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        background-color: rgba(9, 9, 11, 0.8);
        border-bottom: 1px solid #27272a;
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        border-radius: 8px;
    }
    .nav-logo {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f4f4f5;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .env-tag {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 0.8rem;
        color: #a1a1aa;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3f3f46;
    }
    .metric-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a1a1aa;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #f4f4f5;
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* specific colors */
    .val-primary { color: #818cf8; }
    .val-success { color: #10b981; }
    .val-warning { color: #f59e0b; }
    .val-danger { color: #f43f5e; }
    
    .text-success { color: #10b981; }
    .text-danger { color: #f43f5e; }
    .text-muted { color: #71717a; }

    /* Panels */
    .panel {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .panel-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #f4f4f5;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Reasoning Bullets */
    .reason-item {
        background-color: rgba(9, 9, 11, 0.5);
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #d4d4d8;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .reason-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-top: 0.3rem;
        flex-shrink: 0;
    }
    .dot-pos { background-color: #10b981; }
    .dot-neg { background-color: #f43f5e; }
    .dot-neu { background-color: #a1a1aa; }

    /* Streamlit overrides for inputs */
    div[data-baseweb="slider"] {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model(MODEL_FILE)
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    return model, train_df

try:
    model, train_df = load_resources()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()

# ── NAVBAR ─────────────────────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div class="saas-navbar">
    <div class="nav-logo">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        Aura Pricing Intelligence
        <div class="env-tag"><span style="width:4px;height:4px;border-radius:50%;background:#10b981;"></span> LIVE</div>
    </div>
    <div class="nav-right">
        <span>{now.strftime('%b %d, %Y')} • {now.strftime('%H:%M')}</span>
        <div style="background:#27272a; padding:0.2rem 0.6rem; border-radius:4px;">Workspace: Production</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT CONFIGURATION (SIDEBAR REPLACEMENT) ──────────────────────────
# We use standard streamlit columns for layout, but organize inputs cleanly
col_main, col_sidebar = st.columns([7, 3])

with col_sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> Simulation Parameters</div>', unsafe_allow_html=True)
    
    cat_median = st.slider("Market Base Price ($)", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    cat_std = st.slider("Market Volatility (Std Dev $)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
    
    st.markdown("<hr style='border-color: #27272a; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    item_condition_id = st.select_slider("Condition", options=[1, 2, 3, 4, 5], value=1, format_func=lambda x: ['New', 'Like New', 'Good', 'Fair', 'Poor'][x-1])
    shipping = st.radio("Shipping Paid By", options=[1, 0], format_func=lambda x: "Seller" if x==1 else "Buyer", horizontal=True)
    brand_tier = st.slider("Brand Tier", 0.0, 1.0, 0.5, step=0.1, help="0 = Unknown, 1 = Premium")
    desc_length = st.number_input("Desc Length (chars)", value=50, step=10)
    name_length = st.number_input("Name Length (chars)", value=20, step=5)
    st.markdown('</div>', unsafe_allow_html=True)

# ── OPTIMIZATION LOGIC ─────────────────────────────────────────────────
item_features = {
    "item_condition_id": item_condition_id,
    "shipping": shipping,
    "desc_length": desc_length,
    "name_length": name_length,
    "brand_tier": brand_tier
}

# Fill defaults for unmodified categories
for f in ALL_FEATURES:
    if f not in item_features:
        val = float(train_df[f].mean()) if f in train_df.columns else 0.0
        item_features[f] = val

demand_curve = build_demand_curve(
    item_features=item_features,
    model=model,
    category_median=cat_median,
    category_std=cat_std
)

opt_result = find_optimal_price(demand_curve)

# Elasticity curve
prices_arr = np.array(demand_curve["prices"])
demands_arr = np.array(demand_curve["demands"])
dq_dp = np.gradient(demands_arr, prices_arr)
elasticity_curve = dq_dp * (prices_arr / demands_arr)

# ── HERO INSIGHTS ──────────────────────────────────────────────────────
opt_p = opt_result['optimal_price']
opt_r = opt_result['optimal_revenue']
opt_d = opt_result['optimal_demand']
elast = opt_result['elasticity']

diff_pct = ((opt_p - cat_median) / cat_median) * 100 if cat_median else 0
diff_color = "text-success" if diff_pct >= 0 else "text-danger"
diff_sign = "+" if diff_pct >= 0 else ""

elast_color = "val-success" if elast > -1 else "val-danger"
elast_label = "Inelastic" if elast > -1 else "Elastic"

# Confidence heuristic
revenues = np.array(demand_curve["revenues"])
peak_rev = revenues.max()
mean_rev = revenues.mean()
confidence = min(0.99, max(0.50, (peak_rev - mean_rev) / (peak_rev + 1e-9) + 0.5))

with col_main:
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Recommended Price</div>
            <div class="metric-value val-primary">${opt_p:.2f}</div>
            <div class="metric-sub {diff_color}">{diff_sign}{diff_pct:.1f}% vs baseline</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Projected Yield</div>
            <div class="metric-value val-success">${opt_r:.2f}</div>
            <div class="metric-sub text-muted">Maximized revenue</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Demand Forecast</div>
            <div class="metric-value val-warning">{opt_d:.2f}×</div>
            <div class="metric-sub text-muted">Conversion multiplier</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Price Elasticity</div>
            <div class="metric-value {elast_color}">{elast:.2f}</div>
            <div class="metric-sub {diff_color}">{elast_label} territory</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Confidence Score</div>
            <div class="metric-value" style="color: #a78bfa;">{confidence*100:.0f}%</div>
            <div class="metric-sub text-muted">Based on curve variance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHARTS AREA ────────────────────────────────────────────────────────
    # Custom plotly layout generator for dark minimal SaaS look
    def get_layout(title, y_title):
        return dict(
            title=dict(text=title, font=dict(size=14, color="#f4f4f5")),
            xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(color="#71717a")),
            yaxis=dict(title=dict(text=y_title, font=dict(size=11, color="#71717a")), showgrid=True, gridcolor="#27272a", gridwidth=1, zeroline=False, showline=False, tickfont=dict(color="#71717a")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=40, b=30),
            hovermode="x unified",
            showlegend=False
        )

    # Split into 2 rows of 2 charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=prices_arr, y=demand_curve["revenues"],
            mode='lines', line=dict(color='#818cf8', width=2),
            fill='tozeroy', fillcolor='rgba(129, 140, 248, 0.15)',
            name='Revenue'
        ))
        fig_rev.add_trace(go.Scatter(
            x=[opt_p], y=[opt_r], mode='markers',
            marker=dict(color='#818cf8', size=8, line=dict(color="#09090b", width=2)),
            name='Optimal'
        ))
        fig_rev.add_vline(x=opt_p, line_dash="dash", line_color="#818cf8", opacity=0.5)
        fig_rev.update_layout(**get_layout("Revenue Curve", "Projected Revenue ($)"))
        st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_dem = go.Figure()
        fig_dem.add_trace(go.Scatter(
            x=prices_arr, y=demands_arr,
            mode='lines', line=dict(color='#10b981', width=2),
            fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)',
            name='Demand'
        ))
        fig_dem.update_layout(**get_layout("Demand Decay", "Conversion Multiplier"))
        st.plotly_chart(fig_dem, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_elas = go.Figure()
        fig_elas.add_trace(go.Scatter(
            x=prices_arr, y=elasticity_curve,
            mode='lines', line=dict(color='#f59e0b', width=2),
            fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)',
            name='Elasticity'
        ))
        fig_elas.add_hline(y=-1, line_dash="dot", line_color="#f43f5e", opacity=0.7, annotation_text="Unit Elastic (-1)", annotation_font_color="#71717a")
        fig_elas.update_layout(**get_layout("Elasticity Sensitivity", "Point Elasticity (ε)"))
        st.plotly_chart(fig_elas, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        # AI Explanation Panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path></svg> AI Reasoning Engine</div>', unsafe_allow_html=True)
        
        # Build reasoning dynamically
        reasoning = []
        if elast > -1:
            reasoning.append((f"Demand is inelastic (ε = {elast:.2f}). Raising price has minimal demand impact.", "dot-pos"))
        else:
            reasoning.append((f"Demand is elastic (ε = {elast:.2f}). Lowering price could boost volume significantly.", "dot-neg"))
            
        if diff_pct > 0:
            reasoning.append((f"Optimal price is {diff_pct:.1f}% above the market baseline of ${cat_median:.2f}.", "dot-pos"))
        else:
            reasoning.append((f"Optimal price is {abs(diff_pct):.1f}% below the market baseline of ${cat_median:.2f}.", "dot-neu"))
            
        reasoning.append((f"Revenue peaks at ${opt_r:.2f} with a demand multiplier of {opt_d:.2f}x.", "dot-pos"))
        
        if shipping == 1:
            reasoning.append(("Seller-paid shipping is active — this typically improves conversion rates by 15-25%.", "dot-pos"))
            
        if brand_tier >= 0.7:
            reasoning.append(("Premium brand tier detected. Higher price tolerance expected from buyers.", "dot-pos"))
            
        html_bullets = ""
        for text, dot_class in reasoning:
            html_bullets += f'<div class="reason-item"><div class="reason-dot {dot_class}"></div><div>{text}</div></div>'
            
        st.markdown(html_bullets, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
