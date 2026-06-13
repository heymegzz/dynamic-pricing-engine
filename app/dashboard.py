"""
Streamlit Dashboard for Dynamic Pricing Engine.
Premium SaaS analytics platform aesthetic.
All feature construction delegated to src.features.build_inference_row().
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROCESSED_TRAIN, ALL_FEATURES
from src.features import build_inference_row
from src.elasticity import build_demand_curve
from src.optimizer import find_optimal_price
from src.model import load_artifacts

st.set_page_config(
    page_title="Dynamic Pricing AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────
# Palette is defined once as CSS custom properties.
# Semantic variables map to both dark and light Streamlit themes.
st.markdown("""
<style>
    /* ── Design Tokens ─────────────────────────────────────────── */
    :root {
        --accent:       #00c8ff;
        --accent-glow:  rgba(0, 200, 255, 0.15);
        --accent-border:rgba(0, 200, 255, 0.30);
        --gold:         #f0a500;
        --gold-glow:    rgba(240, 165, 0, 0.15);
        --success:      #2ea043;
        --danger:       #f85149;
        --surface:      rgba(22, 27, 34, 0.95);
        --border:       rgba(48, 54, 61, 0.8);
        --text-primary: #e6edf3;
        --text-muted:   #8b949e;
    }

    /* ── Globals ───────────────────────────────────────────────── */
    .stApp {
        background-color: #0d1117 !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Navbar ────────────────────────────────────────────────── */
    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 1.5rem;
        background: linear-gradient(90deg, #0d1117 0%, #21262d 100%);
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
        border-radius: 8px;
    }
    .nav-logo {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .env-tag {
        background: var(--accent-glow);
        color: var(--accent);
        border: 1px solid var(--accent-border);
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
        color: var(--text-muted);
        font-weight: 500;
    }
    .nav-badge {
        background: #21262d;
        border: 1px solid var(--border);
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        color: #c9d1d9;
    }

    /* ── Metric Cards ──────────────────────────────────────────── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--accent-border);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }

    /* ── Semantic colour helpers ────────────────────────────────── */
    .val-primary { color: var(--accent); }
    .val-success { color: var(--success); }
    .val-warning { color: var(--gold); }
    .val-danger  { color: var(--danger); }
    .text-success { color: var(--success); }
    .text-danger  { color: var(--danger); }
    .text-muted   { color: var(--text-muted); }

    /* ── Panels ─────────────────────────────────────────────────── */
    .panel {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── AI Reasoning Box ───────────────────────────────────────── */
    .ai-engine-box {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--gold);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .reason-item {
        padding: 0.4rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
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
    .dot-pos { background-color: var(--success); box-shadow: 0 0 8px rgba(46,160,67,0.5); }
    .dot-neg { background-color: var(--danger);  box-shadow: 0 0 8px rgba(248,81,73,0.5); }
    .dot-neu { background-color: var(--accent);  box-shadow: 0 0 8px rgba(0,200,255,0.5); }

    /* ── Tooltip ─────────────────────────────────────────────────── */
    .tooltip-label {
        cursor: help;
        border-bottom: 1px dashed var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)


# ── Load Resources ──────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    mdl, enc, cat_stats = load_artifacts()
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    return mdl, enc, cat_stats, train_df

try:
    model, encoders, category_stats, train_df = load_resources()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()


# ── NAVBAR ──────────────────────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div class="saas-navbar">
    <div class="nav-logo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00c8ff"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12h4l3-9 5 18 3-9h5"/>
        </svg>
        AI Price Optimizer
        <div class="env-tag">
            <span style="width:5px;height:5px;border-radius:50%;background:#00c8ff;
                         box-shadow:0 0 5px #00c8ff;"></span> ACTIVE
        </div>
    </div>
    <div class="nav-right">
        <span>{now.strftime('%b %d, %Y')} &bull; {now.strftime('%H:%M')}</span>
        <div class="nav-badge">Engine: LightGBM + SciPy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR / INPUT PANEL ───────────────────────────────────────────────
col_sidebar, col_main = st.columns([2.5, 7.5])

with col_sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00c8ff" stroke-width="2">'
        '<circle cx="12" cy="12" r="3"></circle>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06'
        'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09'
        'A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83'
        'l.06-.06A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09'
        'A1.65 1.65 0 0 0 4.6 3.6a1.65 1.65 0 0 0-.33-1.82L4.21 1.72a2 2 0 0 1 2.83-2.83'
        'l.06.06A1.65 1.65 0 0 0 9 3.28h0A1.65 1.65 0 0 0 10 1.77V1a2 2 0 0 1 4 0v.09'
        'a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83'
        'l-.06.06A1.65 1.65 0 0 0 19.4 7h0A1.65 1.65 0 0 0 20.91 8H21a2 2 0 0 1 0 4'
        'h-.09a1.65 1.65 0 0 0-1.51 1z"></path>'
        '</svg> Control Panel</div>',
        unsafe_allow_html=True,
    )

    # ── Input fields ────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.8rem;color:#8b949e;font-weight:600;text-transform:uppercase;'>"
        "Item Identity</p>",
        unsafe_allow_html=True,
    )
    item_name        = st.text_input("Item Name", value="Vintage Nike T-Shirt")
    category_main    = st.text_input("Category (Main)", value="Men")
    category_sub     = st.text_input("Category (Sub)", value="Tops & Shirts")
    category_leaf    = st.text_input("Category (Leaf)", value="T-Shirts")
    brand_name       = st.text_input("Brand Name", value="Nike")
    item_description = st.text_area("Item Description", value="Good condition, barely worn.", height=80)

    st.markdown("<hr style='border-color:rgba(48,54,61,0.8);margin:1.2rem 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.8rem;color:#8b949e;font-weight:600;text-transform:uppercase;'>"
        "Listing Details</p>",
        unsafe_allow_html=True,
    )
    item_condition_id = st.select_slider(
        "Condition",
        options=[1, 2, 3, 4, 5],
        value=1,
        format_func=lambda x: ["New", "Like New", "Good", "Fair", "Poor"][x - 1],
    )
    shipping = st.radio(
        "Shipping Paid By",
        options=[1, 0],
        format_func=lambda x: "Seller" if x == 1 else "Buyer",
        horizontal=True,
    )

    st.markdown("<hr style='border-color:rgba(48,54,61,0.8);margin:1.2rem 0;'>", unsafe_allow_html=True)

    with st.expander(":material/tune: Advanced Optimizer Settings"):
        st.markdown(
            "<p style='font-size:0.8rem;color:#8b949e;'>Fine-tune the demand decay parameter and price simulation bounds.</p>",
            unsafe_allow_html=True,
        )
        elasticity_k = st.slider("Demand Elasticity (k)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        sweep_range = st.slider("Price Sweep Multipliers", min_value=0.1, max_value=5.0, value=(0.4, 2.5), step=0.1)
        price_multipliers = np.linspace(sweep_range[0], sweep_range[1], 100)

    # ── Input validation ────────────────────────────────────────────────
    errors = []
    if not item_name.strip():
        errors.append("Item Name must not be empty.")
    if item_condition_id not in range(1, 6):
        errors.append("Condition must be between 1 and 5.")
    if shipping not in (0, 1):
        errors.append("Shipping must be 0 (Buyer) or 1 (Seller).")

    for err in errors:
        st.error(err)

    optimize_disabled = len(errors) > 0
    run_btn = st.button(":material/bolt: Optimize Price", disabled=optimize_disabled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── FEATURE CONSTRUCTION — single canonical call ────────────────────────
raw_input = {
    "item_name":        item_name,
    "category_main":    category_main,
    "category_sub":     category_sub,
    "category_leaf":    category_leaf,
    "brand_name":       brand_name,
    "item_condition_id": item_condition_id,
    "shipping":         shipping,
    "item_description": item_description,
}

if not optimize_disabled:
    X = build_inference_row(raw_input, encoders, category_stats)
    feat_dict = {k: float(X[0, i]) for i, k in enumerate(ALL_FEATURES)}

    cat_median = feat_dict.get("category_price_median", 25.0)
    cat_std    = feat_dict.get("category_price_std",    15.0)

    demand_curve = build_demand_curve(
        item_features=feat_dict,
        model=model,
        category_median=cat_median,
        category_std=cat_std,
        elasticity_k=1.0,
    )
    opt_result = find_optimal_price(demand_curve)
else:
    # Render placeholder zeros so the page still renders
    cat_median = 25.0
    cat_std    = 15.0
    demand_curve = {"prices": [0], "demands": [0], "revenues": [0]}
    opt_result   = {"optimal_price": 0, "optimal_revenue": 0,
                    "optimal_demand": 0, "elasticity": 0}

prices_arr  = np.array(demand_curve["prices"])
demands_arr = np.array(demand_curve["demands"])

opt_p = opt_result["optimal_price"]
opt_r = opt_result["optimal_revenue"]
opt_d = opt_result["optimal_demand"]
elast = opt_result["elasticity"]

diff_pct   = ((opt_p - cat_median) / cat_median * 100) if cat_median else 0
diff_color = "text-success" if diff_pct >= 0 else "text-danger"
diff_sign  = "+" if diff_pct >= 0 else ""

elast_color = "val-success" if elast > -1 else "val-danger"
elast_label = "Inelastic" if elast > -1 else "Elastic"

revenues  = np.array(demand_curve["revenues"])
peak_rev  = revenues.max()
mean_rev  = revenues.mean()
# "Optimization Sharpness" — ratio of peak to mean revenue across the sweep.
# Higher = sharper optimum = stronger pricing signal.
# Note: this is NOT statistical confidence / prediction interval.
optim_sharpness = min(0.99, max(0.50, (peak_rev - mean_rev) / (peak_rev + 1e-9) + 0.5))


# ── MAIN COLUMN ─────────────────────────────────────────────────────────
with col_main:

    # ── AI REASONING ───────────────────────────────────────────────────
    st.markdown('<div class="ai-engine-box">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title" style="margin-bottom:0.5rem;">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f0a500" stroke-width="2.5">'
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>'
        '</svg> AI Strategic Reasoning</div>',
        unsafe_allow_html=True,
    )

    reasoning = []
    if elast > -1:
        reasoning.append(
            (f"<b>Demand is inelastic (ε = {elast:.2f})</b>. "
             "Raising price yields maximum revenue without significant volume loss.", "dot-pos")
        )
    else:
        reasoning.append(
            (f"<b>Demand is elastic (ε = {elast:.2f})</b>. "
             "A price reduction is recommended to recover volume and maximise total yield.", "dot-neg")
        )

    if diff_pct > 0:
        reasoning.append(
            (f"Optimal price is <b>{diff_pct:.1f}% above</b> the market baseline (${cat_median:.2f}).", "dot-pos")
        )
    else:
        reasoning.append(
            (f"Optimal price is <b>{abs(diff_pct):.1f}% below</b> the market baseline (${cat_median:.2f}).", "dot-neu")
        )

    if shipping == 1:
        reasoning.append(("Seller-paid shipping improves buyer conversion rates.", "dot-pos"))

    bullets = "".join(
        f'<div class="reason-item"><div class="reason-dot {dc}"></div><div>{txt}</div></div>'
        for txt, dc in reasoning
    )
    st.markdown(bullets, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── HERO METRICS ────────────────────────────────────────────────────
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
            <div class="metric-sub text-muted">Yield maximised</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Est. Demand</div>
            <div class="metric-value val-warning">{opt_d:.2f}&times;</div>
            <div class="metric-sub text-muted">Conversion proxy</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Elasticity (&epsilon;)</div>
            <div class="metric-value {elast_color}">{elast:.2f}</div>
            <div class="metric-sub {diff_color}">{elast_label} market</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">
                <span class="tooltip-label"
                      title="Ratio of peak revenue to mean revenue across the price sweep.
Higher = sharper optimum = stronger pricing recommendation.
Note: this is NOT a statistical confidence interval.">
                    Optimization Sharpness &#9432;
                </span>
            </div>
            <div class="metric-value val-primary">{optim_sharpness*100:.0f}%</div>
            <div class="metric-sub text-muted">Peak / mean revenue ratio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHARTS ──────────────────────────────────────────────────────────
    def _chart_layout(title: str, y_title: str) -> dict:
        return dict(
            title=dict(text=title, font=dict(size=15, color="#e6edf3")),
            xaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(color="#8b949e"),
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=12, color="#8b949e")),
                showgrid=True, gridcolor="rgba(48,54,61,0.6)", gridwidth=1,
                zeroline=False, showline=False,
                tickfont=dict(color="#8b949e"),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=30),
            hovermode="x unified",
            showlegend=False,
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=prices_arr, y=demand_curve["revenues"],
            mode="lines", line=dict(color="#00c8ff", width=3),
            fill="tozeroy", fillcolor="rgba(0, 200, 255, 0.10)",
        ))
        fig_rev.add_trace(go.Scatter(
            x=[opt_p], y=[opt_r], mode="markers",
            marker=dict(color="#f0a500", size=10, line=dict(color="#161b22", width=2)),
        ))
        fig_rev.add_vline(x=opt_p, line_dash="dash", line_color="#f0a500", opacity=0.8)
        fig_rev.update_layout(**_chart_layout("Yield Optimisation Curve", "Revenue ($)"))
        st.plotly_chart(fig_rev, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig_dem = go.Figure()
        fig_dem.add_trace(go.Scatter(
            x=prices_arr, y=demands_arr,
            mode="lines", line=dict(color="#2ea043", width=3),
            fill="tozeroy", fillcolor="rgba(46, 160, 67, 0.10)",
        ))
        fig_dem.update_layout(**_chart_layout("Demand Decay Simulation", "Conversion Multiplier"))
        st.plotly_chart(fig_dem, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── SHAP EXPLAINER ──────────────────────────────────────────────────
    from src.explainer import explain_prediction, plot_waterfall
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    with st.expander(":material/search: Explain this prediction (SHAP)"):
        if not optimize_disabled:
            explain_result = explain_prediction(model, X, ALL_FEATURES)
            fig_shap = plot_waterfall(explain_result, top_n=8, figsize=(8, 4))
            st.pyplot(fig_shap, transparent=True)
            st.markdown("""
                <div style="font-size:0.85rem; color:var(--text-muted); margin-top:1rem;">
                <b>How to read this chart:</b> The dashed line is the baseline price for all items. 
                Red bars push the predicted price higher, while blue bars push it lower. 
                The sum of all effects gives the final raw predicted price before optimization.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Fix input errors to generate an explanation.")
    st.markdown("</div>", unsafe_allow_html=True)
