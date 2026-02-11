import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Tanzania Poverty Risk Dashboard",
    page_icon="🇹🇿",
    layout="wide"
)

# ===================== LOAD MODEL =====================
model = joblib.load("model.pkl")
mappings = joblib.load("mappings.pkl")

# ===================== HEADER =====================
st.markdown("""
<div style="text-align:center;">
    <h1>🇹🇿 Tanzania Poverty Risk Assessment Dashboard</h1>
    <p style="font-size:16px;color:gray;">
        AI-powered household poverty risk prediction system
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===================== INPUT PANEL =====================
with st.form("prediction_form"):
    st.markdown("## 🧾 Household Data Entry")

    c1, c2, c3 = st.columns(3)

    with c1:
        hhsize = st.number_input("👨‍👩‍👧 Household Size", 1, 20, 5)
        education = st.selectbox(
            "🎓 Education of Household Head",
            list(mappings["education_head"].keys())
        )

    with c2:
        income = st.number_input(
            "💰 Monthly Income (TZS)",
            value=250000,
            step=50000,
            format="%d"
        )
        employment = st.selectbox(
            "💼 Employment Status",
            list(mappings["employment_status"].keys())
        )

    with c3:
        house = st.selectbox(
            "🏠 House Ownership",
            list(mappings["own_house"].keys())
        )
        location = st.selectbox(
            "🌍 Area Type",
            list(mappings["urban_rural"].keys())
        )

    submit = st.form_submit_button("🔍 Run Poverty Risk Analysis")

# ===================== PREDICTION =====================
if submit:
    input_df = pd.DataFrame({
        "hhsize": [hhsize],
        "education_head": [mappings["education_head"][education]],
        "employment_status": [mappings["employment_status"][employment]],
        "monthly_income": [income],
        "own_house": [mappings["own_house"][house]],
        "urban_rural": [mappings["urban_rural"][location]]
    })

    raw_pred = model.predict(input_df)[0]
    risk_level = int(np.round(raw_pred))
    risk_level = max(1, min(5, risk_level))

    # ===================== RISK LABEL =====================
    if risk_level <= 2:
        label = "LOW RISK"
        color = "#2ECC71"
    elif risk_level == 3:
        label = "MODERATE RISK"
        color = "#F1C40F"
    else:
        label = "HIGH RISK"
        color = "#E74C3C"

    # ===================== KPI CARDS =====================
    st.markdown("## 📊 Key Indicators")

    k1, k2, k3 = st.columns(3)

    k1.metric("Poverty Risk Level", f"{risk_level} / 5")
    k2.metric("Risk Category", label)
    k3.metric("Model Output", f"{raw_pred:.2f}")

    st.markdown("---")

    # ===================== GAUGE CHART =====================
    st.markdown("## 🚦 Poverty Risk Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_level,
        number={"font": {"size": 48}},
        title={"text": "Household Poverty Risk"},
        gauge={
            "axis": {"range": [1, 5]},
            "bar": {"color": color},
            "steps": [
                {"range": [1, 2], "color": "#A9DFBF"},
                {"range": [2, 3], "color": "#F9E79F"},
                {"range": [3, 5], "color": "#F5B7B1"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ===================== FACTOR IMPACT =====================
    st.markdown("## 📌 Contributing Factors Overview")

    factors = {
        "Household Size": min(hhsize * 10, 100),
        "Income Pressure": 100 - min(income / 10000, 100),
        "Education Level": mappings["education_head"][education] * 20,
        "Employment Status": mappings["employment_status"][employment] * 25,
        "Housing Security": mappings["own_house"][house] * 30,
        "Location Risk": mappings["urban_rural"][location] * 40
    }

    for k, v in factors.items():
        st.markdown(f"**{k}**")
        st.progress(int(v))

    # ===================== SUMMARY CARD =====================
    st.markdown("---")
    st.markdown(f"""
    <div style="
        padding:25px;
        border-radius:15px;
        background-color:{color}22;
        border-left:10px solid {color};
        font-size:18px;
    ">
        <b>Final Assessment:</b><br>
        This household falls under <b>{label}</b>.  
        Immediate policy support is recommended if risk is moderate or high.
    </div>
    """, unsafe_allow_html=True)

    # ===================== INPUT REVIEW =====================
    with st.expander("📂 View Submitted Data"):
        st.dataframe(input_df)
