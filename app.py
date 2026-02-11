import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and mappings
model = joblib.load('model.pkl')
mappings = joblib.load('mappings.pkl')

# Page configuration
st.set_page_config(
    page_title="Tanzania Poverty Risk Predictor",
    page_icon="🇹🇿",
    layout="wide"
)

# Header
st.markdown("""
<div style='text-align:center'>
    <h1>🇹🇿 Tanzania Poverty Risk Predictor</h1>
    <p style='font-size:16px; color:gray;'>Predict the <b>Poverty Risk Level</b> of a household based on demographic and economic data.</p>
    <p style='font-size:14px; color:gray;'>Risk Levels: <b>1 (Lowest)</b> → <b>5 (Highest)</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input Form in a Card-like Layout
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
    
    # Prediction
    raw_prediction = model.predict(input_data)[0]
    final_risk = int(np.round(raw_prediction))
    final_risk = max(1, min(5, final_risk))
    
    # Display results in a visually appealing box
    st.markdown("### 📈 Prediction Result")
    
    risk_color = "green" if final_risk <= 2 else "orange" if final_risk == 3 else "red"
    risk_text = "Low Risk" if final_risk <= 2 else "Moderate Risk" if final_risk == 3 else "High Risk"
    
    st.markdown(f"""
    <div style='
        padding:20px;
        border-radius:10px;
        background-color:{risk_color};
        color:white;
        text-align:center;
        font-size:24px;
        font-weight:bold;
    '>
        Predicted Poverty Risk: Level {final_risk} <br>
        Category: {risk_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: display probability-like info
    st.info(f"✅ Model raw output: {raw_prediction:.2f}")
