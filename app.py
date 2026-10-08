"""
Smart Electricity Bill Predictor
=================================
Streamlit app — run with:  streamlit run app.py

Uses a pure-NumPy polynomial least-squares model (no scikit-learn/scipy DLLs).
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── auto-bootstrap: generate data + train model if artifacts are missing ─────

def _bootstrap():
    if not os.path.exists("data/electricity_data.csv"):
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    if not (os.path.exists("model/weights.npy") and
            os.path.exists("model/mean_std.npy")):
        subprocess.run([sys.executable, "train_model.py"], check=True)

_bootstrap()

# ── load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    weights  = np.load("model/weights.npy")
    mean_std = np.load("model/mean_std.npy")
    mean = mean_std[0]
    std  = mean_std[1]
    return weights, mean, std

weights, MEAN, STD = load_artifacts()

# ── inference helpers ─────────────────────────────────────────────────────────

def poly_features(X_sc):
    """Same polynomial expansion used at training time."""
    n, d = X_sc.shape
    parts = [X_sc, X_sc ** 2]
    for i in range(d):
        for j in range(i + 1, d):
            parts.append((X_sc[:, i] * X_sc[:, j]).reshape(-1, 1))
    return np.hstack(parts)

def predict_bill(consumption, people, daily_hours, prev_consumption):
    x = np.array([[consumption, people, daily_hours, prev_consumption]], dtype=float)
    x_sc = (x - MEAN) / STD
    x_poly = poly_features(x_sc)
    x_b = np.hstack([np.ones((1, 1)), x_poly])
    result = x_b @ weights
    return float(result.ravel()[0])

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Electricity Bill Predictor",
    page_icon="⚡",
    layout="centered",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }

    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #3b82f6;
    }
    .metric-card.green  { border-left-color: #22c55e; }
    .metric-card.orange { border-left-color: #f97316; }
    .metric-card.red    { border-left-color: #ef4444; }

    .section-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .tip-box {
        background: #eff6ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #3b82f6;
        font-size: 0.93rem;
        color: #1e3a5f;
    }

    .disclaimer {
        background: #fefce8;
        border: 1px solid #fde047;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #713f12;
        margin-top: 10px;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## ⚡ Smart Electricity Bill Predictor")
st.markdown(
    "<p style='color:#475569;margin-top:-10px;'>AI-powered monthly bill estimation · India domestic tariff</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔧 Input Parameters")
    st.markdown("---")

    consumption = st.number_input(
        "Monthly Electricity Consumption (kWh)",
        min_value=10.0, max_value=2000.0, value=250.0, step=10.0,
        help="Total units consumed this month as shown on your meter.",
    )
    people = st.slider(
        "Number of People in Household",
        min_value=1, max_value=10, value=3,
    )
    daily_hours = st.slider(
        "Average Daily Electricity Usage (hours)",
        min_value=1.0, max_value=24.0, value=8.0, step=0.5,
    )
    prev_consumption = st.number_input(
        "Previous Month's Consumption (kWh)",
        min_value=10.0, max_value=2000.0, value=230.0, step=10.0,
        help="Units consumed last month for comparison.",
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict My Bill", use_container_width=True, type="primary")

# ── helper functions ──────────────────────────────────────────────────────────

def usage_status(kwh):
    if kwh < 150:
        return "Normal", "green", "✅"
    elif kwh < 400:
        return "Moderate", "orange", "⚠️"
    else:
        return "High", "red", "🔴"

def delta_info(curr, prev):
    diff  = curr - prev
    pct   = (diff / prev * 100) if prev else 0
    arrow = "▲" if diff >= 0 else "▼"
    color = "#ef4444" if diff > 0 else "#22c55e"
    return diff, pct, arrow, color

def make_bar_chart(curr, prev):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    bars = ax.bar(
        ["Previous\nMonth", "Current\nMonth"],
        [prev, curr],
        color=["#93c5fd", "#3b82f6"],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, val in zip(bars, [prev, curr]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.0f} kWh",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#1e3a5f",
        )

    ax.set_ylabel("Consumption (kWh)", fontsize=9, color="#475569")
    ax.set_title("Consumption Comparison", fontsize=10,
                 fontweight="bold", color="#1e3a5f", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#475569", labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

TIPS = {
    "Normal": [
        "💡 Switch remaining incandescent bulbs to LED to save up to 75% on lighting.",
        "🌡️ Set your AC to 24 °C — every degree lower raises consumption by ~6%.",
        "🔌 Unplug chargers and idle electronics to eliminate phantom load.",
    ],
    "Moderate": [
        "❄️ Clean AC filters monthly — dirty filters increase energy use by 10–15%.",
        "🌅 Shift heavy appliance use (washing machine, iron) to off-peak hours.",
        "💡 Replace tube lights with LED panels for 40–50% lighting savings.",
    ],
    "High": [
        "🔥 Audit high-draw appliances (geysers, old ACs, refrigerators) — consider upgrading.",
        "🌞 Explore rooftop solar; even a 1 kW system can offset 100–120 kWh/month.",
        "⏱️ Use smart plugs or timers to auto-cut appliances when not needed.",
    ],
}

# ── main panel ────────────────────────────────────────────────────────────────

if predict_btn:
    predicted_bill = predict_bill(consumption, people, daily_hours, prev_consumption)
    predicted_bill = max(predicted_bill, 50.0)

    status, card_color, icon = usage_status(consumption)
    diff, pct, arrow, delta_col = delta_info(consumption, prev_consumption)

    # ── result card ──
    st.markdown(f"""
    <div class="metric-card {card_color}">
        <div class="section-title">Estimated Monthly Bill</div>
        <div style="font-size:2.6rem;font-weight:800;color:#1e3a5f;">&#8377; {predicted_bill:,.0f}</div>
        <div style="color:#64748b;font-size:0.9rem;margin-top:4px;">
            Based on <strong>{consumption:.0f} kWh</strong> consumption &nbsp;|&nbsp;
            {icon} Usage Status: <strong>{status}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── two-column metrics ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">Current Consumption</div>
            <div style="font-size:1.6rem;font-weight:700;color:#1e3a5f;">
                {consumption:.0f} <span style="font-size:1rem;color:#64748b;">kWh</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">vs Previous Month</div>
            <div style="font-size:1.6rem;font-weight:700;color:{delta_col};">
                {arrow} {abs(diff):.0f}
                <span style="font-size:1rem;color:#64748b;">kWh ({abs(pct):.1f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── chart ──
    st.markdown('<div class="section-title">📊 Consumption Comparison</div>',
                unsafe_allow_html=True)
    st.pyplot(make_bar_chart(consumption, prev_consumption), use_container_width=False)

    # ── tips ──
    st.markdown("")
    st.markdown('<div class="section-title">💚 Energy-Saving Recommendations</div>',
                unsafe_allow_html=True)
    for tip in TIPS[status]:
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

    # ── input summary ──
    st.markdown("")
    with st.expander("📋 Input Summary", expanded=False):
        summary_df = pd.DataFrame({
            "Parameter": [
                "Monthly Consumption",
                "Household Members",
                "Daily Usage Hours",
                "Previous Month Consumption",
            ],
            "Value": [
                f"{consumption:.0f} kWh",
                f"{people} person{'s' if people > 1 else ''}",
                f"{daily_hours:.1f} hrs/day",
                f"{prev_consumption:.0f} kWh",
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        &#9888;&#65039; <strong>Disclaimer:</strong> This is an AI-generated estimate based on a
        regression model trained on synthetic data using approximate Indian domestic tariff slabs.
        Actual electricity bills depend on your state's tariff structure, sanctioned load, fixed
        charges, taxes, and other levies. Please refer to your official electricity bill or DISCOM
        website for accurate charges.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── welcome / placeholder ──
    st.markdown("""
    <div class="metric-card" style="border-left-color:#6366f1;">
        <div class="section-title">How to use</div>
        <ol style="color:#475569;font-size:0.95rem;margin:8px 0 0 16px;line-height:1.9;">
            <li>Fill in your electricity details in the <strong>left sidebar</strong>.</li>
            <li>Click <strong>&#128302; Predict My Bill</strong>.</li>
            <li>View your estimated bill, usage status, and personalised tips.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Poly Regression")
    col2.metric("Training Samples", "2,000")
    col3.metric("Currency", "Indian ₹")

    st.info("⚡ Adjust the sliders on the left and press **Predict My Bill** to get started.",
            icon="💡")

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.78rem;'>"
    "Smart Electricity Bill Predictor &middot; Built with Streamlit &amp; NumPy "
    "&middot; For educational use only"
    "</p>",
    unsafe_allow_html=True,
)
