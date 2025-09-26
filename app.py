# app.py

import streamlit as st
import pandas as pd
# Import the prediction function from your glaucoma.py file
from glaucoma import predict_glaucoma

st.title("Glaucoma Diagnosis Predictor")

st.write("Enter patient information to get a glaucoma diagnosis prediction.")

# Create input fields for each feature used in the model
# Make sure the keys here match the column names expected by your predict_glaucoma function
age = st.number_input("Age", min_value=0, max_value=120, value=60)
gender = st.selectbox("Gender", ['Male', 'Female'])
visual_acuity = st.text_input("Visual Acuity Measurements", "LogMAR 0.0")
iop = st.number_input("Intraocular Pressure (IOP)", min_value=0.0, value=15.0)
cdr = st.number_input("Cup-to-Disc Ratio (CDR)", min_value=0.0, max_value=1.0, value=0.5)
family_history = st.selectbox("Family History", ['Yes', 'No'])
medical_history = st.text_input("Medical History", "Unknown") # Or provide specific options
medication_usage = st.text_input("Medication Usage", "Unknown") # Or provide specific options
visual_field_results = st.text_input("Visual Field Test Results", "Sensitivity: 0.0, Specificity: 0.0")
oct_results = st.text_input("Optical Coherence Tomography (OCT) Results", "RNFL Thickness: 0.0 µm, GCC Thickness: 0.0 µm")
pachymetry = st.number_input("Pachymetry", min_value=0.0, value=550.0)
cataract_status = st.selectbox("Cataract Status", ['Present', 'Absent'])
angle_closure_status = st.selectbox("Angle Closure Status", ['Open', 'Closed'])
visual_symptoms = st.text_input("Visual Symptoms", "None") # Or provide specific options

# Create a button to make a prediction
if st.button("Predict Diagnosis"):
    # Create a DataFrame from the input data
    # Column names must exactly match the features used during training
    new_patient_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Visual Acuity Measurements': [visual_acuity],
        'Intraocular Pressure (IOP)': [iop],
        'Cup-to-Disc Ratio (CDR)': [cdr],
        'Family History': [family_history],
        'Medical History': [medical_history],
        'Medication Usage': [medication_usage],
        'Visual Field Test Results': [visual_field_results],
        'Optical Coherence Tomography (OCT) Results': [oct_results],
        'Pachymetry': [pachymetry],
        'Cataract Status': [cataract_status],
        'Angle Closure Status': [angle_closure_status],
        'Visual Symptoms': [visual_symptoms]
    })

    # Make the prediction using the function from glaucoma.py
    predicted_diagnosis = predict_glaucoma(new_patient_data)

    # Display the prediction
    st.subheader("Predicted Diagnosis:")
    st.write(predicted_diagnosis)
