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

# ================= HERO IMAGE ==================
st.image(
    "https://images.unsplash.com/photo-1604088430380-5b5a5c4b1b2a",
    use_container_width=True
)

st.markdown("""
<div style="text-align:center;">
    <h1>🇹🇿 Tanzania Poverty Risk Assessment System</h1>
    <p style="font-size:16px;color:gray;">
        AI-powered household poverty analysis dashboard
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= INFO CARDS ===================
c1, c2, c3 = st.columns(3)

with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.markdown("### 🤖 AI Powered")
    st.write("Machine Learning model trained on socio-economic indicators.")

with c2:
    st.image("https://cdn-icons-png.flaticon.com/128/12357/12357128.png", width=80)
    st.markdown("### 🌍 Tanzania Focused")
    st.write("Designed specifically for Tanzanian household conditions.")

with c3:
    st.image("https://cdn-icons-png.flaticon.com/128/3270/3270865.png", width=80)
    st.markdown("### 📊 Data Driven")
    st.write("Predictions are based on measurable household data.")

st.markdown("---")

# ================= INPUT FORM ===================
with st.form("prediction_form"):
    st.markdown("## 🧾 Household Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        hhsize = st.number_input("👨‍👩‍👧 Household Size", 1, 20, 5)
        education = st.selectbox("🎓 Education Level", list(mappings["education_head"].keys()))

    with col2:
        income = st.number_input("💰 Monthly Income (TZS)", value=250000, step=50000, format="%d")
        employment = st.selectbox("💼 Employment Status", list(mappings["employment_status"].keys()))

    with col3:
        house = st.selectbox("🏠 Owns a House?", list(mappings["own_house"].keys()))
        location = st.selectbox("🌍 Area Type", list(mappings["urban_rural"].keys()))

    submit = st.form_submit_button("🔍 Predict Poverty Risk")

# ================= RESULT =======================
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

    if risk_level <= 2:
        label = "LOW RISK"
        color = "#2ECC71"
    elif risk_level == 3:
        label = "MODERATE RISK"
        color = "#F1C40F"
    else:
        label = "HIGH RISK"
        color = "#E74C3C"

    st.markdown("## 📊 Results Overview")
    k1, k2, k3 = st.columns(3)

    k1.metric("Risk Level", f"{risk_level}/5")
    k2.metric("Risk Category", label)
    k3.metric("Model Output", f"{raw_pred:.2f}")

    st.progress(int((risk_level / 5) * 100))

    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:12px;
        background-color:{color}15;
        border-left:8px solid {color};
        font-size:18px;
    ">
        <b>Final Assessment:</b><br>
        This household is classified as <b>{label}</b>.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 View Input Data"):
        st.dataframe(input_df)




