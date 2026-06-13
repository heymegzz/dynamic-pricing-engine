"""
src/explainer.py — SHAP-based prediction explanation for the Dynamic Pricing Engine.

Provides:
    explain_prediction() — produce a structured SHAP summary for one item
    plot_waterfall()     — render a horizontal waterfall chart figure

Usage:
    from src.explainer import explain_prediction, plot_waterfall
    result = explain_prediction(model, X, feature_names)
    fig    = plot_waterfall(result, top_n=10)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in Streamlit & notebooks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap


def explain_prediction(
    model,
    feature_row: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Compute SHAP values for a single prediction and return a structured summary.

    Args:
        model:          Trained LightGBM model (any tree-based model supported by SHAP).
        feature_row:    np.ndarray of shape (1, n_features) — the inference-ready row.
        feature_names:  List of feature name strings matching column order.

    Returns:
        dict with keys:
            "base_value"        — float, the model's expected output (log_price space)
            "predicted_log_price" — float, model.predict(feature_row)[0]
            "shap_values"       — list of (feature_name, shap_value) tuples,
                                  sorted by abs(shap_value) descending
    """
    if feature_row.ndim == 1:
        feature_row = feature_row.reshape(1, -1)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row)   # shape (1, n_features)
    shap_row    = np.array(shap_values[0])             # flatten to (n_features,)

    # Build (name, value) pairs and sort by magnitude descending
    pairs = list(zip(feature_names, shap_row.tolist()))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    predicted_log_price = float(model.predict(feature_row)[0])

    return {
        "base_value":          float(explainer.expected_value),
        "predicted_log_price": predicted_log_price,
        "shap_values":         pairs,
    }


def plot_waterfall(
    explain_result: dict,
    top_n: int = 10,
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """
    Render a horizontal waterfall chart showing the top_n most impactful features.

    Convention:
        Red  bars — push predicted price UP  (positive SHAP value)
        Blue bars — push predicted price DOWN (negative SHAP value)
        Dashed vertical line — model base value (expected log_price across training set)

    Args:
        explain_result: dict returned by explain_prediction().
        top_n:          Number of features to display (sorted by |shap_value|).
        figsize:        Matplotlib figure size.

    Returns:
        matplotlib.figure.Figure — caller is responsible for displaying / saving.
    """
    base_val  = explain_result["base_value"]
    pred_val  = explain_result["predicted_log_price"]
    pairs     = explain_result["shap_values"][:top_n]

    labels    = [p[0] for p in pairs]
    values    = [p[1] for p in pairs]
    colors    = ["#c0392b" if v >= 0 else "#2980b9" for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(labels, values, color=colors, height=0.55,
                   edgecolor="none", zorder=3)

    # Base-value dashed line
    ax.axvline(x=0, color="#8b949e", linewidth=0.8, linestyle="-", zorder=2)
    ax.axvline(x=(pred_val - base_val), color="#f0a500",
               linewidth=1.5, linestyle="--", zorder=4,
               label=f"Net shift (pred {pred_val:.2f})")

    # Value labels on each bar
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        align = "left" if val >= 0 else "right"
        offset = 0.005 if val >= 0 else -0.005
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=align,
                fontsize=8.5, color="#e6edf3")

    # Styling
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (log-price space)", color="#8b949e", fontsize=10)
    ax.set_title(
        f"Feature Attribution  |  base = {base_val:.3f}  →  pred = {pred_val:.3f}",
        color="#e6edf3", fontsize=12, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines[:].set_edgecolor("#30363d")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend
    up_patch   = mpatches.Patch(color="#c0392b", label="↑ Pushes price up")
    down_patch = mpatches.Patch(color="#2980b9", label="↓ Pushes price down")
    ax.legend(
        handles=[up_patch, down_patch],
        loc="lower right", fontsize=8.5,
        facecolor="#21262d", edgecolor="#30363d",
        labelcolor="#e6edf3",
    )

    plt.tight_layout()
    return fig
