# glaucoma.py

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the saved model
loaded_model = joblib.load('glaucoma_model.pkl')

# Load the saved fitted preprocessor
loaded_preprocessor = joblib.load('fitted_preprocessor.pkl')

print("Trained model and fitted preprocessor loaded successfully.")

def predict_glaucoma(patient_data: pd.DataFrame):
    """
    Makes a glaucoma diagnosis prediction for a new patient.

    Args:
        patient_data: A pandas DataFrame containing the new patient's data.
                      Must have the same columns as the training data's features (X).

    Returns:
        The predicted diagnosis ('Glaucoma' or 'No Glaucoma').
    """
    # Add debugging prints
    print("--- Debugging predict_glaucoma ---")
    print("Input patient_data columns:", patient_data.columns.tolist())
    print("Input patient_data dtypes:\n", patient_data.dtypes)
    # You can also print a sample of the data if needed
    # print("Input patient_data head:\n", patient_data.head())
    print("----------------------------------")


    # Preprocess the new data using the loaded, fitted preprocessor
    try:
        processed_data = loaded_preprocessor.transform(patient_data)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        print("Make sure the new data has the expected columns and data types.")
        return "Error: Could not process patient data."

    # Make prediction
    prediction = loaded_model.predict(processed_data)

    return prediction[0]

if __name__ == '__main__':
    # Example usage of the function
    # Create a sample new patient data (replace with actual new data)
    # This DataFrame should only contain the feature columns, matching the structure of X used for training.
    new_patient_data_example = pd.DataFrame({
        'Age': [75],
        'Gender': ['Male'],
        'Visual Acuity Measurements': ['20/20'],
        'Intraocular Pressure (IOP)': [15.0],
        'Cup-to-Disc Ratio (CDR)': [0.4],
        'Family History': ['No'],
        'Medical History': ['Unknown'], # Use 'Unknown' if not available, matching preprocessing
        'Medication Usage': ['Unknown'], # Use 'Unknown' if not available, matching preprocessing
        'Visual Field Test Results': ['Sensitivity: 0.8, Specificity: 0.95'],
        'Optical Coherence Tomography (OCT) Results': ['RNFL Thickness: 90.0 µm, GCC Thickness: 70.0 µm'],
        'Pachymetry': [560.0],
        'Cataract Status': ['Present'],
        'Angle Closure Status': ['Open'],
        'Visual Symptoms': ['None'] # or use 'Unknown' or similar if not applicable
    })

    predicted_diagnosis = predict_glaucoma(new_patient_data_example)
    print(f"Example prediction for a new patient: {predicted_diagnosis}")
