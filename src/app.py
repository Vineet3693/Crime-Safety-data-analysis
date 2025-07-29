
# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to the model
model_path = 'src/model.pkl'  # Adjust this path based on your structure

# Check if the model file exists and load it
if os.path.exists(model_path):
    model = joblib.load(model_path)  # Load the model
else:
    st.error("Model file not found. Please ensure the model is trained and saved.")
    model = None  # Set model to None if not found

# Streamlit app title
st.title('Crime Data Prediction App')

# User input for prediction
victim_age = st.number_input('Enter Victim Age:', min_value=0, max_value=120)

# Add more input fields as needed for other features

if st.button('Predict'):
    if model is not None:  # Check if the model is loaded
        # Prepare input data for prediction
        input_data = [[victim_age]]  # Add other features as needed
        prediction = model.predict(input_data)
        st.write(f'Predicted Outcome: {prediction}')
    else:
        st.error("Model is not loaded. Cannot make predictions.")
