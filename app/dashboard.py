"""
Streamlit Dashboard for Dynamic Pricing Engine.
"""

import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_FILE, PROCESSED_TRAIN, ALL_FEATURES
from src.elasticity import load_model, build_demand_curve, get_category_stats
from src.optimizer import find_optimal_price

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")

@st.cache_resource
def load_resources():
    model = load_model(MODEL_FILE)
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    return model, train_df

st.title("Dynamic Pricing Engine Optimizer")
st.markdown("Interact with the pricing engine. Adjust the item features below to see the impact on the demand and revenue curves, and find the optimal price point.")

try:
    model, train_df = load_resources()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()

st.sidebar.header("Item Features")

# Sidebar inputs
category_main = st.sidebar.text_input("Main Category", value="Women")
brand_name = st.sidebar.text_input("Brand Name", value="Nike")
item_condition_id = st.sidebar.slider("Condition (1=New, 5=Poor)", 1, 5, 1)
shipping = st.sidebar.selectbox("Shipping (1=Seller pays, 0=Buyer pays)", [0, 1], index=1)
desc_length = st.sidebar.number_input("Description Length", value=50)
name_length = st.sidebar.number_input("Name Length", value=20)
brand_tier = st.sidebar.number_input("Brand Tier", value=1)

if st.sidebar.button("Optimize Price"):
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
        if f not in item_features:
            item_features[f] = 0
            
    cat_median, cat_std = get_category_stats(train_df, category_main)
    
    demand_curve = build_demand_curve(
        item_features=item_features,
        model=model,
        category_median=cat_median,
        category_std=cat_std
    )
    
    opt_result = find_optimal_price(demand_curve)
    
    st.subheader("Optimization Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Price", f"${opt_result['optimal_price']:.2f}")
    col2.metric("Max Revenue", f"${opt_result['optimal_revenue']:.2f}")
    col3.metric("Point Elasticity", f"{opt_result['elasticity']:.2f}")
    
    st.markdown("### Curves")
    
    df_curve = pd.DataFrame({
        "Price": demand_curve["prices"],
        "Demand": demand_curve["demands"],
        "Revenue": demand_curve["revenues"]
    })
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df_curve["Price"], y=df_curve["Revenue"], name="Revenue", mode="lines", line=dict(color="green")),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_curve["Price"], y=df_curve["Demand"], name="Demand", mode="lines", line=dict(color="blue", dash="dash")),
        secondary_y=True,
    )
    
    # Add optimal point marker
    fig.add_trace(
        go.Scatter(
            x=[opt_result["optimal_price"]], 
            y=[opt_result["optimal_revenue"]],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name="Optimal Price Point"
        ),
        secondary_y=False
    )

    # Add figure title
    fig.update_layout(
        title_text="Revenue and Demand vs. Price",
        hovermode="x unified"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Price ($)")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Revenue</b> ($)", secondary_y=False, color="green")
    fig.update_yaxes(title_text="<b>Demand</b> (Quantity Proxy)", secondary_y=True, color="blue")

    st.plotly_chart(fig, use_container_width=True)
