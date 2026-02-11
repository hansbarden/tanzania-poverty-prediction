import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the resources
model = joblib.load('model.pkl')
mappings = joblib.load('mappings.pkl')

st.title("🇹🇿 Tanzania Poverty Risk Predictor")
st.markdown("""
This AI application predicts the **Poverty Risk Level** of a household in Tanzania based on demographic and economic data.
*Risk Levels: 1 (Lowest) to 5 (Highest)*
""")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        hhsize = st.number_input("Household Size", 1, 20, 5)
        education = st.selectbox("Head of Household Education", list(mappings['education_head'].keys()))
        employment = st.selectbox("Employment Status", list(mappings['employment_status'].keys()))
        
    with col2:
        income = st.number_input("Monthly Income (TZS)", value=250000, step=50000)
        house = st.selectbox("Owns a House?", list(mappings['own_house'].keys()))
        location = st.selectbox("Area Type", list(mappings['urban_rural'].keys()))
        
    submit = st.form_submit_button("Generate Prediction")

if submit:
    # Prepare data for prediction
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
    final_risk = max(1, min(5, final_risk)) # Clamp between 1 and 5
    
    # Visual Output
    st.subheader(f"Predicted Poverty Risk: Level {final_risk}")
    if final_risk <= 2:
        st.success("Category: Low Risk")
    elif final_risk == 3:
        st.warning("Category: Moderate Risk")
    else:
        st.error("Category: High Risk")