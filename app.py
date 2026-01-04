import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
model_path = "/mnt/data/naive.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Naive Bayes Prediction App")

st.write("Enter the input values to get prediction")

# --------------------------------------------------
# 👉 Replace these sample fields with your dataset features
# Example for 3 features: age, salary, gender
# --------------------------------------------------

age = st.number_input("Age", min_value=1, max_value=100, value=25)
salary = st.number_input("Salary", min_value=1000, max_value=1000000, value=50000)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert categorical values if needed
gender_value = 1 if gender == "Male" else 0

# Prepare input for model
input_data = np.array([[age, salary, gender_value]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.success(f"Prediction: {prediction[0]}")
    st.info(f"Probabilities: {prob}")
