import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Tanzania Poverty Risk Predictor",
    page_icon="🇹🇿",
    layout="wide"
)

# ================= LOAD MODEL ==================
model = joblib.load("model.pkl")
mappings = joblib.load("mappings.pkl")

# ================= HEADER ======================
st.markdown("""
<div style="text-align:center;">
    <h1>🇹🇿 Tanzania Poverty Risk Assessment System</h1>
    <p style="font-size:16px;color:gray;">
        AI-based household poverty risk prediction dashboard
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= INPUT FORM ==================
with st.form("prediction_form"):
    st.markdown("## 🧾 Household Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        hhsize = st.number_input("👨‍👩‍👧 Household Size", 1, 20, 5)
        education = st.selectbox(
            "🎓 Education Level (Household Head)",
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

# ================= PREDICTION ==================
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

    # ================= RISK CATEGORY =================
    if risk_level <= 2:
        label = "LOW RISK"
        color = "#2ECC71"
    elif risk_level == 3:
        label = "MODERATE RISK"
        color = "#F1C40F"
    else:
        label = "HIGH RISK"
        color = "#E74C3C"

    # ================= KPI METRICS =================
    st.markdown("## 📊 Key Indicators")
    k1, k2, k3 = st.columns(3)

    k1.metric("Poverty Risk Level", f"{risk_level} / 5")
    k2.metric("Risk Category", label)
    k3.metric("Model Output", f"{raw_pred:.2f}")

    st.markdown("---")

    # ================= RISK BAR ====================
    st.markdown("## 🚦 Risk Level Indicator")

    progress_value = int((risk_level / 5) * 100)
    st.progress(progress_value)

    # ================= RESULT CARD =================
    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:12px;
        border-left:8px solid {color};
        background-color:{color}15;
        font-size:18px;
    ">
        <b>Final Assessment:</b><br>
        The household is classified as <b>{label}</b>.
    </div>
    """, unsafe_allow_html=True)

    # ================= FACTOR CONTRIBUTION =================
    st.markdown("## 📌 Contributing Factors Overview")

    st.markdown("Household Size")
    st.progress(min(hhsize * 10, 100))

    st.markdown("Income Pressure")
    st.progress(max(0, 100 - int(income / 10000)))

    st.markdown("Education Level")
    st.progress(mappings["education_head"][education] * 20)

    st.markdown("Employment Status")
    st.progress(mappings["employment_status"][employment] * 25)

    st.markdown("Housing Security")
    st.progress(mappings["own_house"][house] * 30)

    st.markdown("Location Risk")
    st.progress(mappings["urban_rural"][location] * 40)

    # ================= DATA REVIEW =================
    with st.expander("📂 View Submitted Data"):
        st.dataframe(input_df)

