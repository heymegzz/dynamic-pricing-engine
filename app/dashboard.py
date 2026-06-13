"""
Streamlit Dashboard for Dynamic Pricing Engine.
Refactored to look like a premium SaaS analytics platform.
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

st.set_page_config(page_title="Dynamic Pricing AI", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Amazon/Tech Color Combo (Dark Blue + Cyan + Gold)
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0d1117 !important; /* GitHub/Amazon dark blue */
        color: #e6edf3 !important;
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
        padding: 1.2rem 1.5rem;
        background: linear-gradient(90deg, rgba(13, 17, 23, 1) 0%, rgba(33, 38, 45, 1) 100%);
        border-bottom: 1px solid #30363d;
        margin-bottom: 2rem;
        border-radius: 8px;
    }
    .nav-logo {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .env-tag {
        background: rgba(0, 242, 254, 0.1);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00f2fe;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }
    
    /* specific colors */
    .val-primary { color: #00f2fe; } /* Cyan */
    .val-success { color: #2ea043; } /* Green */
    .val-warning { color: #ff9900; } /* Amazon Gold */
    .val-danger { color: #f85149; }  /* Red */
    
    .text-success { color: #2ea043; }
    .text-danger { color: #f85149; }
    .text-muted { color: #8b949e; }

    /* Panels */
    .panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Reasoning Bullets */
    .ai-engine-box {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #ff9900;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .reason-item {
        background-color: transparent;
        padding: 0.4rem 0;
        font-size: 0.9rem;
        color: #e6edf3;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .reason-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-pos { background-color: #2ea043; box-shadow: 0 0 8px rgba(46,160,67,0.5); }
    .dot-neg { background-color: #f85149; box-shadow: 0 0 8px rgba(248,81,73,0.5); }
    .dot-neu { background-color: #00f2fe; box-shadow: 0 0 8px rgba(0,242,254,0.5); }

    /* Streamlit overrides for inputs */
    div[data-baseweb="slider"] {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

from src.model import load_artifacts

@st.cache_resource
def load_resources():
    model, encoders, category_stats = load_artifacts()
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    return model, train_df, encoders, category_stats

try:
    model, train_df, encoders, category_stats = load_resources()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()

# ── NAVBAR ─────────────────────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div class="saas-navbar">
    <div class="nav-logo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00f2fe" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l3-9 5 18 3-9h5"/></svg>
        AI Price Optimizer
        <div class="env-tag"><span style="width:5px;height:5px;border-radius:50%;background:#00f2fe;box-shadow:0 0 5px #00f2fe;"></span> ACTIVE</div>
    </div>
    <div class="nav-right">
        <span>{now.strftime('%b %d, %Y')} • {now.strftime('%H:%M')}</span>
        <div style="background:#21262d; border: 1px solid #30363d; padding:0.3rem 0.8rem; border-radius:6px; color:#c9d1d9;">Engine: LightGBM + SciPy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT CONFIGURATION (SIDEBAR REPLACEMENT) ──────────────────────────
col_sidebar, col_main = st.columns([2.5, 7.5])

with col_sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00f2fe" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> Control Panel</div>', unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;'>Macro Market Baseline</p>", unsafe_allow_html=True)
    cat_median = st.slider("Market Base Price ($)", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    cat_std = st.slider("Market Volatility (Std Dev $)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
    
    st.markdown("<hr style='border-color: #30363d; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;'>Micro Item Specs</p>", unsafe_allow_html=True)
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

for f in ALL_FEATURES:
    if f not in item_features:
        # Use target encoded global means if features are not specified
        val = float(train_df[f].mean()) if f in train_df.columns else 0.0
        item_features[f] = val

demand_curve = build_demand_curve(
    item_features=item_features,
    model=model,
    category_median=cat_median,
    category_std=cat_std,
    elasticity_k=1.0
)

opt_result = find_optimal_price(demand_curve)

prices_arr = np.array(demand_curve["prices"])
demands_arr = np.array(demand_curve["demands"])
dq_dp = np.gradient(demands_arr, prices_arr)
elasticity_curve = dq_dp * (prices_arr / demands_arr)

opt_p = opt_result['optimal_price']
opt_r = opt_result['optimal_revenue']
opt_d = opt_result['optimal_demand']
elast = opt_result['elasticity']

diff_pct = ((opt_p - cat_median) / cat_median) * 100 if cat_median else 0
diff_color = "text-success" if diff_pct >= 0 else "text-danger"
diff_sign = "+" if diff_pct >= 0 else ""

elast_color = "val-success" if elast > -1 else "val-danger"
elast_label = "Inelastic" if elast > -1 else "Elastic"

revenues = np.array(demand_curve["revenues"])
peak_rev = revenues.max()
mean_rev = revenues.mean()
confidence = min(0.99, max(0.50, (peak_rev - mean_rev) / (peak_rev + 1e-9) + 0.5))

with col_main:
    # ── AI REASONING (MOVED TO TOP) ────────────────────────────────────────
    st.markdown('<div class="ai-engine-box">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="margin-bottom:0.5rem;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff9900" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> AI Strategic Reasoning</div>', unsafe_allow_html=True)
    
    reasoning = []
    if elast > -1:
        reasoning.append((f"<b>Demand is inelastic (ε = {elast:.2f})</b>. Raising price yields maximum revenue without significant volume loss.", "dot-pos"))
    else:
        reasoning.append((f"<b>Demand is elastic (ε = {elast:.2f})</b>. A price drop is recommended to recover volume and maximize total yield.", "dot-neg"))
        
    if diff_pct > 0:
        reasoning.append((f"Optimal price is <b>{diff_pct:.1f}% above</b> the market baseline (${cat_median:.2f}).", "dot-pos"))
    else:
        reasoning.append((f"Optimal price is <b>{abs(diff_pct):.1f}% below</b> the market baseline (${cat_median:.2f}).", "dot-neu"))
        
    if shipping == 1:
        reasoning.append(("Seller-paid shipping actively improves conversion rates by ~15%.", "dot-pos"))
        
    if brand_tier >= 0.7:
        reasoning.append(("Premium brand tier detected, allowing for a higher price tolerance from buyers.", "dot-pos"))
        
    html_bullets = ""
    for text, dot_class in reasoning:
        html_bullets += f'<div class="reason-item"><div class="reason-dot {dot_class}"></div><div>{text}</div></div>'
        
    st.markdown(html_bullets, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── HERO INSIGHTS ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Target Price</div>
            <div class="metric-value val-primary">${opt_p:.2f}</div>
            <div class="metric-sub {diff_color}">{diff_sign}{diff_pct:.1f}% vs baseline</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Peak Revenue</div>
            <div class="metric-value val-success">${opt_r:.2f}</div>
            <div class="metric-sub text-muted">Yield maximized</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Est. Demand</div>
            <div class="metric-value val-warning">{opt_d:.2f}×</div>
            <div class="metric-sub text-muted">Conversion rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Elasticity (ε)</div>
            <div class="metric-value {elast_color}">{elast:.2f}</div>
            <div class="metric-sub {diff_color}">{elast_label} market</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Model Confidence</div>
            <div class="metric-value" style="color: #00f2fe;">{confidence*100:.0f}%</div>
            <div class="metric-sub text-muted">High certainty</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHARTS AREA ────────────────────────────────────────────────────────
    def get_layout(title, y_title):
        return dict(
            title=dict(text=title, font=dict(size=15, color="#ffffff")),
            xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(color="#8b949e")),
            yaxis=dict(title=dict(text=y_title, font=dict(size=12, color="#8b949e")), showgrid=True, gridcolor="#30363d", gridwidth=1, zeroline=False, showline=False, tickfont=dict(color="#8b949e")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=30),
            hovermode="x unified",
            showlegend=False
        )

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=prices_arr, y=demand_curve["revenues"],
            mode='lines', line=dict(color='#00f2fe', width=3),
            fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.1)',
            name='Revenue'
        ))
        fig_rev.add_trace(go.Scatter(
            x=[opt_p], y=[opt_r], mode='markers',
            marker=dict(color='#ff9900', size=10, line=dict(color="#161b22", width=2)),
            name='Optimal'
        ))
        fig_rev.add_vline(x=opt_p, line_dash="dash", line_color="#ff9900", opacity=0.8)
        fig_rev.update_layout(**get_layout("Yield Optimization Curve", "Projected Revenue ($)"))
        st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_dem = go.Figure()
        fig_dem.add_trace(go.Scatter(
            x=prices_arr, y=demands_arr,
            mode='lines', line=dict(color='#2ea043', width=3),
            fill='tozeroy', fillcolor='rgba(46, 160, 67, 0.1)',
            name='Demand'
        ))
        fig_dem.update_layout(**get_layout("Demand Decay Simulation", "Conversion Multiplier"))
        st.plotly_chart(fig_dem, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
