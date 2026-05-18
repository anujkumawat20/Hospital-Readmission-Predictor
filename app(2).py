import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Readmission Predictor")

st.title("🏥 Hospital Readmission Risk Predictor")

st.markdown("Enter patient details:")

# INPUTS
age = st.slider("Age", 20, 90, 60)
time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
num_lab_procedures = st.slider("Lab Procedures", 1, 100, 40)
num_procedures = st.slider("Procedures", 0, 10, 1)
num_medications = st.slider("Medications", 1, 50, 10)

number_outpatient = st.slider("Outpatient Visits", 0, 20, 0)
number_emergency = st.slider("Emergency Visits", 0, 10, 0)
number_inpatient = st.slider("Inpatient Visits", 0, 10, 0)

number_diagnoses = st.slider("Diagnoses Count", 1, 10, 5)

# ENGINEER FEATURES (same logic as training!)
TOTAL_VISITS = number_outpatient + number_emergency + number_inpatient

SEVERITY_INDEX = (
    time_in_hospital +
    num_lab_procedures +
    num_procedures +
    number_diagnoses
)

CHRONIC_FLAG = 1 if number_diagnoses > 5 else 0
MEDICATION_INTENSITY = num_medications

EMERGENCY_ADMISSION = 1  # assume emergency for now
DISCHARGE_RISK = 0       # default safe

# Create input dataframe
input_data = pd.DataFrame([{
    'AGE_NUM': age,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'TOTAL_VISITS': TOTAL_VISITS,
    'SEVERITY_INDEX': SEVERITY_INDEX,
    'CHRONIC_FLAG': CHRONIC_FLAG,
    'MEDICATION_INTENSITY': MEDICATION_INTENSITY,
    'EMERGENCY_ADMISSION': EMERGENCY_ADMISSION,
    'DISCHARGE_RISK': DISCHARGE_RISK
}])

# Ensure correct feature order
input_data = input_data[features]

# PREDICTION
if st.button("Predict Risk"):
    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.4:
        st.error(f"🔴 High Risk: {prob:.2%}")
    else:
        st.success(f"🟢 Low Risk: {prob:.2%}")

    # SHAP EXPLANATION
    st.subheader("Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0],
            feature_names=input_data.columns
        )
    )
    st.pyplot(fig)