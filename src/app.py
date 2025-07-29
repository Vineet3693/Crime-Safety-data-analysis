
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')  # Ensure you save the model after training

# Streamlit app title
st.title('Crime Data Prediction App')

# User input for prediction
victim_age = st.number_input('Enter Victim Age:', min_value=0, max_value=120)

# Add more input fields as needed for other features

if st.button('Predict'):
    # Prepare input data for prediction
    input_data = [[victim_age]]  # Add other features as needed
    prediction = model.predict(input_data)
    st.write(f'Predicted Outcome: {prediction}')
