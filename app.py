import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load resources
model = joblib.load('model.pkl')
mappings = joblib.load('mappings.pkl')

# Page config
st.set_page_config(
    page_title="Tanzania Poverty Risk Predictor",
    page_icon="🇹🇿",
    layout="wide",
)

# Header
st.markdown("""
<div style='text-align:center;'>
    <h1>🇹🇿 Tanzania Poverty Risk Predictor</h1>
    <p style='font-size:16px; color:gray;'>Predict the <b>Poverty Risk Level</b> of a household based on demographic and economic data.</p>
    <p style='font-size:14px; color:gray;'>Risk Levels: <b>1 (Lowest)</b> → <b>5 (Highest)</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input form
with st.form("prediction_form"):
    st.markdown("### 📝 Household Information")
    col1, col2 = st.columns(2)
    
    with col1:
        hhsize = st.number_input("👨‍👩‍👧 Household Size", 1, 20, 5)
        education = st.selectbox("🎓 Head of Household Education", list(mappings['education_head'].keys()))
        employment = st.selectbox("💼 Employment Status", list(mappings['employment_status'].keys()))
        
    with col2:
        income = st.number_input("💰 Monthly Income (TZS)", value=250000, step=50000, format="%d")
        house = st.selectbox("🏠 Owns a House?", list(mappings['own_house'].keys()))
        location = st.selectbox("🌆 Area Type", list(mappings['urban_rural'].keys()))
        
    submit = st.form_submit_button("📊 Generate Prediction")

st.markdown("---")

if submit:
    # Prepare data
    input_data = pd.DataFrame({
        'hhsize': [hhsize],
        'education_head': [mappings['education_head'][education]],
        'employment_status': [mappings['employment_status'][employment]],
        'monthly_income': [income],
        'own_house': [mappings['own_house'][house]],
        'urban_rural': [mappings['urban_rural'][location]]
    })

    # Predict
    raw_prediction = model.predict(input_data)[0]
    final_risk = int(np.round(raw_prediction))
    final_risk = max(1, min(5, final_risk))
    
    # Risk text and color
    if final_risk <= 2:
        risk_text = "Low Risk"
        color = "#2ECC71"  # Green
        bar_value = 30
    elif final_risk == 3:
        risk_text = "Moderate Risk"
        color = "#F1C40F"  # Yellow
        bar_value = 60
    else:
        risk_text = "High Risk"
        color = "#E74C3C"  # Red
        bar_value = 90

    # Display result in a "dashboard card"
    st.markdown("### 📈 Prediction Result")
    
    st.markdown(f"""
    <div style='
        padding:25px;
        border-radius:15px;
        background: linear-gradient(90deg, {color}33, {color}88);
        text-align:center;
        color:black;
        font-size:26px;
        font-weight:bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    '>
        Predicted Poverty Risk: Level {final_risk} <br>
        Category: {risk_text}
    </div>
    """, unsafe_allow_html=True)

    # Gauge-like progress bar
    st.markdown("#### 📊 Risk Gauge")
    st.progress(bar_value)

    # Optional: show raw model output for transparency
    st.info(f"💡 Model raw output: {raw_prediction:.2f}")

    # Optional: show input summary
    st.markdown("#### 📝 Input Summary")
    st.table(input_data)
