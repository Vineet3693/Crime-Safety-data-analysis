import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Crime Safety Data Visualization')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("/content/crime_safety_dataset.csv")
    return df

crime_df = load_data()

st.write("Crime Data Overview")
st.write(crime_df.head())

# You will add more visualization code here later
